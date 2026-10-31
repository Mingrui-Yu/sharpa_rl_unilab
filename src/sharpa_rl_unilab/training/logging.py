"""UniLab's Rich/TensorBoard logging with explicit training budgets."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from unilab.rl.logging.base import _fmt_time
from unilab.rl.logging.metric_schema import metric_spec
from unilab.rl.logging.offpolicy import OffPolicyLogger
from unilab.training.experiment import get_git_info, write_run_config_snapshot

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CONTRACT_VERSION

# Native log_step arguments (seconds) and their backend metric names.
LEARNER_TIMINGS = {
    "collector_wait_time": "timing/learner_collector_wait_ms",
    "learner_replay_stage_time": "timing/learner_replay_stage_ms",
    "learner_replay_sample_time": "timing/learner_replay_sample_ms",
    "train_time": "timing/learner_train_ms",
    "weight_sync_time": "timing/learner_weight_publish_ms",
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        return str(value)

    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=json_default) + "\n")


def append_metrics(run, metrics):
    """Use the same JSONL format across native and local training runners."""
    with (Path(run) / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics) + "\n")


def create_run(cfg):
    run = Path(
        cfg.training.log_dir or f"logs/{cfg.algo.algo}/seed_{cfg.algo.seed}_{time.time_ns()}"
    )
    run.mkdir(parents=True, exist_ok=False)
    write_run_metadata(run, cfg)
    return run


def training_counters() -> dict[str, int]:
    return dict.fromkeys(
        (
            "collected",
            "received",
            "training_samples",
            "optimizer_updates",
            "actor_updates",
            "critic_updates",
            "temperature_updates",
            "policy_version",
        ),
        0,
    )


def write_run_metadata(run, cfg):
    from unilab.rl.runtime.offpolicy.thread_budget import resolve_torch_thread_runtime

    algorithm = str(cfg.algo.algo)
    teacher = cfg.protocol.stage == "teacher"
    native_flash = teacher and algorithm == "flashsac"
    async_appo = teacher and algorithm == "appo"
    resources = {
        "sampling_architecture": "double_buffer"
        if native_flash
        else "async_queue"
        if async_appo
        else "synchronous",
        "learner_device": str(cfg.training.device),
        "inference_device": str(cfg.algo.collector_device)
        if async_appo
        else str(cfg.training.device),
        "collector_tensor_device": str(
            cfg.algo.get("collector_device", None) or cfg.training.device
        ),
        "observation_boundary": "torch",
        "hot_path_host_transfer": "none; reset-static selectors and explicit diagnostics are cold paths",
        "torch_thread_runtime": resolve_torch_thread_runtime(cfg.training.torch_threads),
        "learner_num_threads": torch.get_num_threads(),
        "learner_num_interop_threads": torch.get_num_interop_threads(),
        "independent_collector": native_flash or async_appo,
    }
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    root = Path(__file__).resolve().parents[3]
    git = get_git_info(root)
    write_run_config_snapshot(
        run,
        full_cfg=cfg,
        run_metadata={"git": git, "stage": str(cfg.protocol.stage)},
        contract_snapshot={"version": CONTRACT_VERSION},
    )
    write_json(
        run / "run.json",
        {
            "contract": CONTRACT_VERSION,
            "revision": git["commit"],
            "dirty": git["dirty"],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "unilab", "rsl-rl-lib", "mujoco")
            },
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "timing": "Includes model/environment initialization, collection, learning and saving; excludes CLI asset verification and separate evaluation.",
            "runtime": resources,
            "budget_tolerance": int(cfg.algo.num_envs) - 1
            if not teacher or cfg.training.max_transitions is not None
            else None,
            "training_samples": "Transition uses in actor and critic updates: joint PPO/APPO minibatch counted once; separate SAC actor and critic uses each counted once.",
            "distributed": False,
            "global_num_envs": int(cfg.algo.num_envs),
        },
    )


class EpisodeStatistics:
    """Track completed episodes across fresh rollout packets, before replay reuse."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.completed_returns: list[float] | list[torch.Tensor] = []
        self.completed_lengths: list[int] | list[torch.Tensor] = []
        self.completed_timeouts: list[bool] | list[torch.Tensor] = []

    @staticmethod
    def _append_window(values: list, value):
        if not value.numel():
            return
        values.append(value.detach())
        if sum(part.numel() for part in values) > 100:
            values[:] = [torch.cat(values)[-100:]]

    def update(self, rewards, terminated, truncated):
        if isinstance(rewards, torch.Tensor):
            device = rewards.device
            returns = (
                self.returns
                if isinstance(self.returns, torch.Tensor)
                else torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            )
            lengths = (
                self.lengths
                if isinstance(self.lengths, torch.Tensor)
                else torch.zeros(self.num_envs, dtype=torch.int64, device=device)
            )
            returns += rewards.sum(dim=0)
            lengths += rewards.shape[0]
            done = terminated.any(dim=0) | truncated.any(dim=0)
            timeout = (truncated & ~terminated).any(dim=0)
            self._append_window(self.completed_returns, returns[done])
            self._append_window(self.completed_lengths, lengths[done])
            self._append_window(self.completed_timeouts, timeout[done])
            returns.masked_fill_(done, 0)
            lengths.masked_fill_(done, 0)
            self.returns, self.lengths = returns, lengths
            return

        for reward, terminal, timeout in zip(rewards, terminated, truncated):
            done = np.asarray(terminal) | np.asarray(timeout)
            self.returns += reward
            self.lengths += 1
            self.completed_returns.extend(self.returns[done].tolist())
            self.completed_lengths.extend(self.lengths[done].tolist())
            # Match the return/length window; real termination wins over timeout.
            self.completed_timeouts.extend((timeout & ~terminal)[done].tolist())
            self.returns[done] = 0
            self.lengths[done] = 0

    def metrics(self):
        if not self.completed_returns:
            return {}

        def completed_mean(values):
            if isinstance(values[0], torch.Tensor):
                return float(torch.cat(values).float().mean())
            return float(np.mean(values))

        return {
            "episode/return": completed_mean(self.completed_returns),
            "episode/length": completed_mean(self.completed_lengths),
            "episode/timeout_rate": completed_mean(self.completed_timeouts),
        }


class TrainingLogger(OffPolicyLogger):
    """Native panels/backends, with iteration or received-transition progress.

    OffPolicyLogger also serves PPO and distillation: unlike OnPolicyLogger,
    it accepts an explicit environment-step axis and measured throughput.
    """

    def __init__(self, run: Path, cfg, target: int):
        backend = str(cfg.training.get("logger", "tensorboard"))
        if backend not in {"tensorboard", "none", "no_print"}:
            raise ValueError("training.logger must be tensorboard, none or no_print")
        self.target = target
        self.by_iterations = (
            cfg.algo.algo in {"ppo", "appo", "supermimic"}
            and cfg.protocol.stage == "teacher"
            and cfg.algo.get("max_iterations") is not None
        )
        self._progress = 0
        self._previous_steps = 0
        self._previous_seconds = 0.0
        super().__init__(
            algo_name=f"{str(cfg.algo.algo).upper()} {cfg.protocol.stage}",
            num_envs=int(cfg.algo.num_envs),
            env_name=str(cfg.training.task_name),
            log_dir=str(run),
            log_backend=backend,
            timing_profile="appo" if cfg.algo.algo == "appo" else "sac_family",
        )

    def log(self, metrics, *, timings=None):
        metrics = dict(metrics)
        steps, seconds = int(metrics["received"]), float(metrics["wall_seconds"])
        self._progress = int(metrics["policy_version"]) if self.by_iterations else steps
        elapsed = max(seconds - self._previous_seconds, 1e-9)
        throughput = (steps - self._previous_steps) / elapsed
        metrics["perf/transitions_per_second"] = throughput
        if timings is not None:
            metrics.update(
                {key: timings.get(arg, 0) * 1000 for arg, key in LEARNER_TIMINGS.items()}
            )
            metrics["perf/iter_ms"] = timings["iteration_time"] * 1000
            metrics["timing/learner_other_ms"] = (
                max(
                    timings["iteration_time"] - sum(timings.get(arg, 0) for arg in LEARNER_TIMINGS),
                    0,
                )
                * 1000
            )
        self._previous_steps, self._previous_seconds = steps, seconds
        append_metrics(self._log_dir, metrics)
        self.log_collector(steps, 0)
        self._mean_ep_length = float(metrics.get("episode/length", 0))
        self.update_timeout_rate(metrics.get("episode/timeout_rate", 0))
        self.update_collector_timing(
            {
                k.removeprefix("timing/collector_"): v
                for k, v in metrics.items()
                if k.startswith("timing/collector_")
            }
        )
        # Averaging cumulative counters would display transitions never received.
        self._terminal_samples.clear()
        self._start_time = time.time() - seconds
        timings = timings or {"iteration_time": elapsed}
        self.log_step(
            iteration=int(metrics.get("policy_version", 0)),
            # Specialized native fields belong in their panels, not the loss table.
            metrics={
                k: v
                for k, v in metrics.items()
                if metric_spec(k) is not None
                and not k.startswith(("reward/", "timing/"))
                and k
                not in {
                    "perf/iter_ms",
                    "perf/collector_active_steps_per_sec",
                    "episode/timeout_rate",
                }
            },
            return_mean_ep100=metrics.get("episode/return"),
            reward_components={k: v for k, v in metrics.items() if k.startswith("reward/")},
            extra_info={
                "env_steps_per_sec": throughput,
            },
            collector_wait_time=timings.get("collector_wait_time", 0),
            learner_replay_stage_time=timings.get("learner_replay_stage_time", 0),
            learner_replay_sample_time=timings.get("learner_replay_sample_time", 0),
            train_time=timings.get("train_time", 0),
            weight_sync_time=timings.get("weight_sync_time", 0),
            iteration_time=timings["iteration_time"],
        )

    def _estimate_eta(self, *, iteration=None):
        if not self._progress:
            return ""
        return _fmt_time(
            self._previous_seconds * max(self.target - self._progress, 0) / self._progress
        )

    def _build_display(self):
        panel = super()._build_display()
        panel.title = (
            f"UniLab Training | {self.algo_name} | {self.env_name} | "
            f"{'Iterations' if self.by_iterations else 'Transitions'} {self._progress:,}/{self.target:,}"
        )
        return panel

    def status(self, message: str):
        self.log_status(message)
        self._refresh(force=True)

    def finish(self, *, title="Training Summary", extra_summary=""):
        # Native finish assumes an iteration budget; keep its resource cleanup
        # and render the transition-budget panel instead of that summary.
        self.status(f"Training complete | Saved {self._last_save}")
        self.close()
