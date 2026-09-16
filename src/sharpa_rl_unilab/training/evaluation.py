"""Fixed-scene deterministic evaluation and training-seed level aggregation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from uni_rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CONTRACT_VERSION, HISTORY_SHAPE
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import resolve_grasp_cache_file

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
        "rng": "numpy PCG64; SeedSequence([episode_seed, control_step, stream]); observation=0, force=1",
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
    if float(cfg.evaluation.window_seconds) != 20.0:
        raise ValueError("Protocol v2 uses a fixed 20-second evaluation window")


def reset_record(env, cfg, scale, seed):
    obs, _ = env.reset(seed=seed)
    reset = env.event_manager.get_term_cfg("reset").func
    dr = env.event_manager.get_term_cfg("domain_randomization").func
    cache = resolve_grasp_cache_file(str(cfg.env.events.reset.params.grasp_cache_path), scale)
    record = {
        "cache_sha256": file_digest(cache),
        "cache_row": int(reset.cache_row_indices[0]),
        "randomization": {
            key: getattr(dr, key)[0].tolist()
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


def set_step_rng(env, episode_seed, step):
    env.env.rng = np.random.default_rng(np.random.SeedSequence([episode_seed, step, 0]))
    if "persistent_force" in env.event_manager.active_terms.get("step", []):
        env.event_manager.get_term_cfg("persistent_force").func.rng = np.random.default_rng(
            np.random.SeedSequence([episode_seed, step, 1])
        )


def evaluate_checkpoint(checkpoint, *, output=None, device: str | None = "cpu", evaluation=None):
    print(f"[eval] Loading checkpoint: {checkpoint}", file=sys.stderr, flush=True)
    actor, cfg, snapshot = load_policy(checkpoint, device)
    device = str(next(actor.parameters()).device)
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
    print(
        f"[eval] Starting {total} episodes across {len(cfg.evaluation.scales)} scales",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()
    last_logged = started
    for scale in cfg.evaluation.scales:
        env = SharpaTeacherEnv(cfg, 1, scale=float(scale), auto_reset=False)
        try:
            if env.max_episode_length != 400 or not np.isclose(env.step_dt, 0.05):
                raise ValueError("Evaluation requires the fixed 400-step, 20-second task")
            for episode in manifest["episodes"]:
                if episode["scale"] != scale:
                    continue
                obs, actual = reset_record(env, cfg, float(scale), episode["reset_seed"])
                if any(actual[key] != episode[key] for key in actual):
                    raise ValueError(
                        f"Scene {episode['episode_id']} cannot reproduce recorded grasp/DR values"
                    )
                angle = reward = 0.0
                dropped = False
                survival = 0.0
                diagnostics = None
                term = None
                if diagnostics_enabled:
                    term = env.action_manager.get_term("hand")
                    diagnostics = EpisodeActionDiagnostics(
                        term._entity.data.joint_pos[:, term.joint_ids]
                    )
                for step in range(400):
                    set_step_rng(env, episode["perturbation_seed"], step)
                    distribution = inference_distribution(actor, obs, device, hist_norm)
                    actions = distribution.deterministic().cpu().numpy()
                    state = env.step(actions)
                    if diagnostics is not None:
                        assert term is not None
                        # Evaluation disables autoreset, so terminal q/target are real.
                        diagnostics.record(
                            term._clipped_action,
                            distribution.std.cpu().numpy(),
                            term._entity.data.joint_pos[:, term.joint_ids],
                            term.target,
                            term._target_lower,
                            term._target_upper,
                        )
                    obs = state.obs
                    angle += float(env.signed_angle_delta[0])
                    reward += float(state.reward[0])
                    survival = (step + 1) * env.step_dt
                    dropped = bool(state.terminated[0])
                    if dropped or state.truncated[0]:
                        break
                results.append(
                    {
                        "episode_id": episode["episode_id"],
                        "scale": float(scale),
                        "return": reward,
                        "survival_seconds": survival,
                        "dropped": float(dropped),
                        "signed_angle": angle,
                        "speed_fixed_window": angle / 20.0,
                        "speed_alive": angle / survival,
                        **(
                            {"diagnostics": diagnostics.result()} if diagnostics is not None else {}
                        ),
                    }
                )
                done = len(results)
                now = time.monotonic()
                if done == 1 or done == total or now - last_logged >= 10:
                    elapsed = now - started
                    eta = elapsed / done * (total - done)
                    print(
                        f"[eval] {done}/{total} ({done / total:.1%}) | scale={scale:g}"
                        f" | elapsed={elapsed:.0f}s | ETA~{eta:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_logged = now
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
    colors = {"ppo": "tab:blue", "appo": "tab:orange", "flashsac": "tab:green"}
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
