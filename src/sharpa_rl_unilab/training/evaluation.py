"""Fixed-scene deterministic evaluation and training-seed level aggregation."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from unilab.managers.rng import ManagerRng
from unilab.rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CONTRACT_VERSION, HISTORY_SHAPE
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import resolve_grasp_cache_file
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.compat import entity_host_data

from .action_diagnostics import DIAGNOSTIC_METRICS, EpisodeActionDiagnostics
from .checkpoints import file_digest, load_policy
from .logging import write_json
from .policy import inference_distribution

METRICS = (
    "return",
    "survival_seconds",
    "dropped",
    "signed_angle",
    "speed_fixed_window",
    "speed_alive",
)


class BatchedEvaluationRng(ManagerRng):
    """ManagerRng with one Torch stream per evaluation slot."""

    def __init__(self, num_envs: int, *, device: str | torch.device):
        super().__init__(device=device)
        self.num_envs = int(num_envs)
        self._streams = [ManagerRng(device=self.device) for _ in range(self.num_envs)]
        self._reset_slot: int | None = None
        self._force_probability: float | None = None
        self._last_random: torch.Tensor | None = None

    @contextmanager
    def reset_slot(self, slot: int) -> Iterator[None]:
        if self._reset_slot is not None:
            raise RuntimeError("Evaluation RNG reset contexts cannot be nested")
        self._reset_slot = int(slot)
        try:
            yield
        finally:
            self._reset_slot = None

    def manual_seed(self, seed: int) -> BatchedEvaluationRng:
        if self._reset_slot is None:
            for stream in self._streams:
                stream.manual_seed(seed)
        else:
            self._streams[self._reset_slot].manual_seed(seed)
        return self

    def seed_slots(self, seeds: Sequence[int]) -> None:
        if self._reset_slot is not None or len(seeds) != self.num_envs:
            raise ValueError("A step seed is required for every evaluation slot")
        for stream, seed in zip(self._streams, seeds):
            stream.manual_seed(int(seed))
        self._last_random = None

    def bind_persistent_force(self, probability: float) -> None:
        self._force_probability = float(probability)

    @staticmethod
    def _shape(size: int | tuple[int, ...]) -> tuple[int, ...]:
        return (size,) if isinstance(size, int) else tuple(size)

    def _draw(
        self,
        method: str,
        size: int | tuple[int, ...],
        *args,
        size_first: bool = True,
        **kwargs,
    ):
        shape = self._shape(size)
        env_axis: int | None
        if len(shape) == 2 and shape == (2, self.num_envs):
            env_axis = 1
        else:
            env_axis = next(
                (axis for axis, width in enumerate(shape) if width == self.num_envs), None
            )
        if self._reset_slot is not None and env_axis is None:
            call_args = (size, *args) if size_first else (*args, size)
            return getattr(self._streams[self._reset_slot], method)(*call_args, **kwargs)
        if env_axis is None:
            if method != "standard_normal" or self._last_random is None:
                raise ValueError("Batched evaluation draws must contain a num_envs dimension")
            assert self._force_probability is not None
            slots = (self._last_random < self._force_probability).nonzero(as_tuple=False).flatten()
            if len(shape) != 2 or slots.numel() != shape[0]:
                raise ValueError("Persistent-force draw does not match its triggered slots")
            per_slot_shape: tuple[int, ...] = (1, shape[1])
            draws = [
                self._streams[int(slot)].standard_normal(per_slot_shape, **kwargs) for slot in slots
            ]
            self._last_random = None
            return torch.cat(draws, dim=0)
        per_slot_shape = tuple(1 if axis == env_axis else width for axis, width in enumerate(shape))
        draws = []
        for stream in self._streams:
            call_args = (per_slot_shape, *args) if size_first else (*args, per_slot_shape)
            draws.append(getattr(stream, method)(*call_args, **kwargs))
        return torch.cat(draws, dim=env_axis)

    def random(self, size, *, dtype=torch.float32):
        result = self._draw("random", size, dtype=dtype)
        if self._reset_slot is None and self._shape(size) == (self.num_envs,):
            self._last_random = result
        return result

    def uniform(self, low, high, size, *, dtype=torch.float32):
        return self._draw("uniform", size, low, high, size_first=False, dtype=dtype)

    def standard_normal(self, size, *, dtype=torch.float32):
        return self._draw("standard_normal", size, dtype=dtype)

    def normal(self, mean, std, size, *, dtype=torch.float32):
        return self._draw("normal", size, mean, std, size_first=False, dtype=dtype)

    def integers(self, low, high, size, *, dtype=torch.int64):
        return self._draw("integers", size, low, high, size_first=False, dtype=dtype)

    def choice(self, a, size, *, p, replacement=True):
        shape = self._shape(size)
        if self._reset_slot is not None and shape != (self.num_envs,):
            return self._streams[self._reset_slot].choice(a, size, p=p, replacement=replacement)
        if len(shape) != 1 or shape[0] != self.num_envs:
            raise ValueError("Batched choice requires one draw per evaluation slot")
        return torch.cat(
            [stream.choice(a, 1, p=p, replacement=replacement) for stream in self._streams],
            dim=0,
        )


def scene_protocol(cfg):
    return {
        "contract": CONTRACT_VERSION,
        "env": OmegaConf.to_container(cfg.env, resolve=True),
        "reward": OmegaConf.to_container(cfg.reward, resolve=True),
        "scales": list(cfg.evaluation.scales),
        "scale_weights": list(cfg.evaluation.scale_weights),
        "seeds": list(cfg.evaluation.seeds),
        "episodes_per_scale": int(cfg.evaluation.episodes_per_scale),
        "window_seconds": float(cfg.evaluation.window_seconds),
        "split": str(cfg.evaluation.split),
        "rng": "UniLab ManagerRng (Torch); SeedSequence([episode_seed, control_step, 0]) per slot",
        "rotation": "world-frame shortest quaternion difference, projected on the task axis, unclipped",
    }


def validate_evaluation(cfg):
    scales, weights = list(cfg.evaluation.scales), np.asarray(cfg.evaluation.scale_weights)
    if (
        not scales
        or len(set(scales)) != len(scales)
        or len(scales) != len(weights)
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError(
            "Evaluation requires unique scales and fixed nonnegative weights summing to one"
        )
    if int(cfg.evaluation.episodes_per_scale) < 1 or not cfg.evaluation.seeds:
        raise ValueError("Evaluation episode quotas and seed list must be positive")
    if int(OmegaConf.select(cfg.evaluation, "batch_size", default=32)) < 1:
        raise ValueError("Evaluation batch size must be positive")
    if float(cfg.evaluation.window_seconds) != 20.0:
        raise ValueError("Protocol v2 uses a fixed 20-second evaluation window")


def _install_evaluation_rng(env) -> BatchedEvaluationRng:
    rng = BatchedEvaluationRng(env.num_envs, device=env.device)
    if "persistent_force" in env.event_manager.active_terms.get("step", []):
        cfg = env.event_manager.get_term_cfg("persistent_force")
        rng.bind_persistent_force(cfg.params.get("force_probability", 0.25))
    env.env.rng = rng
    env.env._rng_owner = rng
    return rng


def _reset_slot(env, slot: int, seed: int):
    rng = env.env.rng
    if isinstance(rng, BatchedEvaluationRng):
        with rng.reset_slot(slot):
            obs, _ = env.reset([slot], seed=int(seed))
        return {
            key: value if value.shape[0] == 1 else value[slot : slot + 1]
            for key, value in obs.items()
        }
    if slot != 0 or env.num_envs != 1:
        raise ValueError("A scalar ManagerRng supports only single-environment evaluation")
    obs, _ = env.reset(seed=int(seed))
    return obs


def reset_record(env, cfg, scale, seed, *, slot: int = 0):
    obs = _reset_slot(env, slot, seed)
    reset = env.event_manager.get_term_cfg("reset").func
    dr = env.event_manager.get_term_cfg("domain_randomization").func
    cache = resolve_grasp_cache_file(str(cfg.env.events.reset.params.grasp_cache_path), scale)
    record = {
        "cache_sha256": file_digest(cache),
        "cache_row": int(reset.cache_row_indices[slot]),
        "randomization": {
            key: getattr(dr, key)[slot].tolist()
            for key in ("kp", "kd", "mass", "com_offset", "gravity", "friction_scale", "scale")
        },
    }
    return obs, record


def make_manifest(cfg, path):
    validate_evaluation(cfg)
    episodes = []
    for scale_index, scale in enumerate(cfg.evaluation.scales):
        env = SharpaTeacherEnv(cfg, 1, scale=float(scale), auto_reset=False)
        try:
            for seed in cfg.evaluation.seeds:
                for index in range(int(cfg.evaluation.episodes_per_scale)):
                    episode_seed = int(
                        np.random.SeedSequence([int(seed), scale_index, index]).generate_state(1)[0]
                    )
                    _, record = reset_record(env, cfg, float(scale), episode_seed)
                    episodes.append(
                        {
                            "episode_id": f"{scale:g}/{seed}/{index}",
                            "scale": float(scale),
                            "reset_seed": episode_seed,
                            "perturbation_seed": episode_seed,
                            **record,
                        }
                    )
        finally:
            env.close()
    manifest = {"protocol": scene_protocol(cfg), "episodes": episodes}
    write_json(path, manifest)
    return manifest


def load_manifest(cfg, path):
    validate_evaluation(cfg)
    manifest = json.loads(Path(path).read_text())
    if manifest["protocol"] != scene_protocol(cfg):
        raise ValueError("Scene manifest does not match this environment/evaluation protocol")
    expected = int(cfg.evaluation.episodes_per_scale) * len(cfg.evaluation.seeds)
    ids = [episode["episode_id"] for episode in manifest["episodes"]]
    if len(set(ids)) != len(ids) or len(ids) != expected * len(cfg.evaluation.scales):
        raise ValueError("Scene manifest contains duplicate episodes or an incorrect total quota")
    for scale in cfg.evaluation.scales:
        if sum(episode["scale"] == scale for episode in manifest["episodes"]) != expected:
            raise ValueError("Scene manifest violates the fixed per-scale quotas")
    return manifest


def _step_seed(episode_seed: int, step: int) -> int:
    return int(np.random.SeedSequence([int(episode_seed), int(step), 0]).generate_state(1)[0])


def set_step_rng(env, episode_seed: int | Sequence[int], step: int):
    rng = env.env.rng
    if not isinstance(rng, ManagerRng):
        raise TypeError("Evaluation requires the UniLab ManagerRng owner")
    if isinstance(rng, BatchedEvaluationRng):
        seeds = [episode_seed] if isinstance(episode_seed, int) else list(episode_seed)
        rng.seed_slots([_step_seed(seed, step) for seed in seeds])
    else:
        if not isinstance(episode_seed, int) or env.num_envs != 1:
            raise ValueError("A scalar ManagerRng requires one episode seed")
        rng.manual_seed(_step_seed(episode_seed, step))
    if "persistent_force" in env.event_manager.active_terms.get("step", []):
        force_rng = env.event_manager.get_term_cfg("persistent_force").func.rng
        if force_rng is not None and force_rng is not rng:
            raise ValueError("persistent_force must share the environment ManagerRng")


def _copy_slot_obs(obs, selected, slot: int) -> None:
    for key in obs:
        if selected[key].shape[0] != 1:
            raise ValueError("Selected-row reset must return exactly one observation")
        obs[key][slot : slot + 1].copy_(selected[key])


def _inference_batch(actor, obs, device, hist_norm):
    distribution = inference_distribution(actor, obs, device, hist_norm)
    return distribution.deterministic(), distribution.std


def _rollout_episode_batch(
    env,
    cfg,
    actor,
    episodes,
    *,
    device: str,
    hist_norm=None,
    diagnostics_enabled: bool = False,
):
    if len(episodes) != env.num_envs or not episodes:
        raise ValueError("Evaluation batch must contain one episode per environment slot")
    if len({episode["scale"] for episode in episodes}) != 1:
        raise ValueError("One fixed-variant evaluation batch cannot mix scales")
    if env.max_episode_length != 400 or not np.isclose(env.step_dt, 0.05):
        raise ValueError("Evaluation requires the fixed 400-step, 20-second task")

    obs = None
    for slot, episode in enumerate(episodes):
        selected, actual = reset_record(
            env, cfg, float(episode["scale"]), episode["reset_seed"], slot=slot
        )
        if any(actual[key] != episode[key] for key in actual):
            raise ValueError(
                f"Scene {episode['episode_id']} cannot reproduce recorded grasp/DR values"
            )
        if obs is None:
            obs = {
                key: torch.empty(
                    (env.num_envs, *value.shape[1:]), dtype=value.dtype, device=value.device
                )
                for key, value in selected.items()
            }
        _copy_slot_obs(obs, selected, slot)
    assert obs is not None

    active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    angle = torch.zeros(env.num_envs, device=device)
    reward = torch.zeros(env.num_envs, device=device)
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    survival_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=device)
    term = env.action_manager.get_term("hand") if diagnostics_enabled else None
    diagnostics = None
    if term is not None:
        joint_pos = entity_host_data(term._entity, "joint_pos")[:, term.joint_ids]
        diagnostics = [
            EpisodeActionDiagnostics(joint_pos[slot : slot + 1]) for slot in range(env.num_envs)
        ]

    perturbation_seeds = [episode["perturbation_seed"] for episode in episodes]
    for step in range(400):
        if not bool(active.any()):
            break
        inactive = (~active).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for slot in inactive:
            selected = _reset_slot(env, slot, episodes[slot]["reset_seed"])
            _copy_slot_obs(obs, selected, slot)

        set_step_rng(env, perturbation_seeds, step)
        actions, std = _inference_batch(actor, obs, device, hist_norm)
        actions[~active] = 0
        state = env.step(actions)
        if diagnostics is not None:
            assert term is not None
            clipped = term._clipped_action.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            joint_pos = entity_host_data(term._entity, "joint_pos")[:, term.joint_ids]
            target = term.target.detach().cpu().numpy()
            lower = term._target_lower.detach().cpu().numpy()
            upper = term._target_upper.detach().cpu().numpy()
            for slot in active.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                diagnostics[slot].record(
                    clipped[slot : slot + 1],
                    std[slot : slot + 1],
                    joint_pos[slot : slot + 1],
                    target[slot : slot + 1],
                    lower[slot : slot + 1],
                    upper[slot : slot + 1],
                )
        obs = state.obs
        source = env.observation_manager.get_term_cfg("actor", "frame").func
        angle_delta = torch.as_tensor(
            source.object_angvel * source.task_state.rotation_axis,
            device=device,
            dtype=angle.dtype,
        ).sum(dim=-1)
        angle += angle_delta * env.step_dt * active
        reward += state.reward * active
        survival_steps += active
        dropped |= state.terminated & active
        active &= ~(state.terminated | state.truncated)

    results = []
    for slot, episode in enumerate(episodes):
        survival = float(survival_steps[slot]) * env.step_dt
        signed_angle = float(angle[slot])
        results.append(
            {
                "episode_id": episode["episode_id"],
                "scale": float(episode["scale"]),
                "return": float(reward[slot]),
                "survival_seconds": survival,
                "dropped": float(dropped[slot]),
                "signed_angle": signed_angle,
                "speed_fixed_window": signed_angle / 20.0,
                "speed_alive": signed_angle / survival,
                **({"diagnostics": diagnostics[slot].result()} if diagnostics is not None else {}),
            }
        )
    return results


def evaluate_checkpoint(checkpoint, *, output=None, device: str | None = "cpu", evaluation=None):
    print(f"[eval] Loading checkpoint: {checkpoint}", file=sys.stderr, flush=True)
    actor, cfg, snapshot = load_policy(checkpoint, device)
    device = str(next(actor.parameters()).device)
    cfg.training.device = device
    cfg.algo.collector_device = device
    print(
        f"[eval] {snapshot['algorithm']} {snapshot['stage']} | device={device}",
        file=sys.stderr,
        flush=True,
    )
    if evaluation is not None:
        cfg.evaluation = OmegaConf.merge(cfg.evaluation, evaluation)
    output = Path(output or Path(checkpoint).with_suffix(".evaluation.json"))
    manifest_path = Path(cfg.evaluation.manifest or output.parent / "scenes.json")
    manifest_exists = manifest_path.exists()
    print(
        f"[eval] {'Loading' if manifest_exists else 'Generating'} scenes: {manifest_path}",
        file=sys.stderr,
        flush=True,
    )
    manifest = (
        load_manifest(cfg, manifest_path) if manifest_exists else make_manifest(cfg, manifest_path)
    )
    hist_norm = None
    if snapshot["stage"] == "student":
        hist_norm = EmpiricalNormalization(HISTORY_SHAPE, device).eval()
        hist_norm.load_state_dict(snapshot["history_normalizer"], strict=True)
    results = []
    diagnostics_enabled = bool(OmegaConf.select(cfg, "evaluation.diagnostics", default=False))
    total = len(manifest["episodes"])
    batch_size = int(OmegaConf.select(cfg.evaluation, "batch_size", default=32))
    print(
        f"[eval] Starting {total} episodes across {len(cfg.evaluation.scales)} scales"
        f" | batch_size={batch_size}",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()
    for scale in cfg.evaluation.scales:
        scale_episodes = [episode for episode in manifest["episodes"] if episode["scale"] == scale]
        num_envs = min(batch_size, len(scale_episodes))
        env = SharpaTeacherEnv(cfg, num_envs, scale=float(scale), auto_reset=False)
        try:
            _install_evaluation_rng(env)
            for start in range(0, len(scale_episodes), num_envs):
                batch = scale_episodes[start : start + num_envs]
                slots = batch + [scale_episodes[0]] * (num_envs - len(batch))
                batch_results = _rollout_episode_batch(
                    env,
                    cfg,
                    actor,
                    slots,
                    device=device,
                    hist_norm=hist_norm,
                    diagnostics_enabled=diagnostics_enabled,
                )
                results.extend(batch_results[: len(batch)])
                done = len(results)
                elapsed = time.monotonic() - started
                eta = elapsed / done * (total - done)
                print(
                    f"[eval] {done}/{total} ({done / total:.1%}) | scale={scale:g}"
                    f" | elapsed={elapsed:.0f}s | ETA~{eta:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            env.close()
    per_scale = {
        str(scale): {
            metric: float(np.mean([row[metric] for row in results if row["scale"] == scale]))
            for metric in METRICS
        }
        for scale in cfg.evaluation.scales
    }
    summary = {
        metric: sum(
            float(weight) * per_scale[str(scale)][metric]
            for scale, weight in zip(cfg.evaluation.scales, cfg.evaluation.scale_weights)
        )
        for metric in METRICS
    }
    result = {
        "contract": CONTRACT_VERSION,
        "stage": snapshot["stage"],
        "algorithm": snapshot["algorithm"],
        "training_seed": int(cfg.algo.seed),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_digest(checkpoint),
        "manifest_sha256": file_digest(manifest_path),
        "counters": snapshot["counters"],
        "wall_seconds": snapshot["wall_seconds"],
        "teacher_cost": snapshot.get("teacher_cost"),
        "evaluation_seconds": time.monotonic() - started,
        "selection": str(cfg.evaluation.checkpoint_selection),
        "split": str(cfg.evaluation.split),
        "summary": summary,
        "per_scale": per_scale,
        "episodes": results,
    }
    if diagnostics_enabled:
        by_scale = {
            str(scale): {
                metric: float(
                    np.mean(
                        [row["diagnostics"][metric] for row in results if row["scale"] == scale]
                    )
                )
                for metric in DIAGNOSTIC_METRICS
            }
            for scale in cfg.evaluation.scales
        }
        result["diagnostics"] = {
            "per_scale": by_scale,
            "summary": {
                metric: sum(
                    float(weight) * by_scale[str(scale)][metric]
                    for scale, weight in zip(cfg.evaluation.scales, cfg.evaluation.scale_weights)
                )
                for metric in DIAGNOSTIC_METRICS
            },
        }
    write_json(output, result)
    print(f"[eval] Saved: {output.resolve()}", file=sys.stderr, flush=True)
    return result


def aggregate_seeds(paths, output):
    """Each training seed is one independent replicate; episodes are nested."""
    rows = [json.loads(Path(path).read_text()) for path in paths]
    if len(rows) < 3 or len({row["training_seed"] for row in rows}) != len(rows):
        raise ValueError(
            "Aggregate at least three distinct training seeds, one checkpoint per seed"
        )
    for key in ("stage", "algorithm", "manifest_sha256", "selection", "split"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Cannot aggregate evaluations with different {key}")

    def stats(values):
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        replicates = (
            np.random.default_rng(20260914).choice(values, size=(10000, len(values))).mean(axis=1)
        )
        return {"mean": mean, "std": std, "ci95": np.quantile(replicates, [0.025, 0.975]).tolist()}

    result = {
        "algorithm": rows[0]["algorithm"],
        "ci_method": "percentile bootstrap of training-seed aggregates, 10000 replicates, seed 20260914",
        "stage": rows[0]["stage"],
        "training_seeds": [row["training_seed"] for row in rows],
        "summary": {metric: stats([row["summary"][metric] for row in rows]) for metric in METRICS},
        "per_scale": {
            scale: {
                metric: stats([row["per_scale"][scale][metric] for row in rows])
                for metric in METRICS
            }
            for scale in rows[0]["per_scale"]
        },
    }
    write_json(output, result)
    return result


def plot_learning_curves(paths, output):
    """Separate teacher/student figures; each line is one training seed."""
    import matplotlib.pyplot as plt

    rows = [json.loads(Path(path).read_text()) for path in paths]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {
        "supermimic": "tab:red",
        "ppo": "tab:blue",
        "appo": "tab:orange",
        "flashsac": "tab:green",
    }
    labels_seen = set()
    for algorithm, stage, seed in sorted(
        {(row["algorithm"], row["stage"], row["training_seed"]) for row in rows}
    ):
        selected = sorted(
            (
                row
                for row in rows
                if (row["algorithm"], row["stage"], row["training_seed"])
                == (algorithm, stage, seed)
            ),
            key=lambda row: row["counters"]["collected"],
        )
        label = algorithm if (algorithm, stage) not in labels_seen else "_nolegend_"
        labels_seen.add((algorithm, stage))
        for axis, values in zip(
            axes[0 if stage == "teacher" else 1],
            (
                [row["counters"]["collected"] for row in selected],
                [row["wall_seconds"] for row in selected],
            ),
        ):
            axis.plot(
                values,
                [row["summary"]["speed_fixed_window"] for row in selected],
                marker="o",
                alpha=0.65,
                color=colors[algorithm],
                label=label,
            )
    for stage, pair in zip(("Teacher", "Student"), axes):
        for axis, name in zip(
            pair, ("Collected transitions", "Wall time (seconds, initialization included)")
        ):
            axis.set(xlabel=name, ylabel="Signed rotation / fixed 20 s (rad/s)", title=stage)
            axis.grid(alpha=0.3)
        if pair[0].get_legend_handles_labels()[0]:
            pair[0].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
