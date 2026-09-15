"""UniLab's Rich/TensorBoard logging with explicit training budgets."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from uni_rl.logging.common import _fmt_time
from uni_rl.logging.offpolicy import OffPolicyLogger
from unilab.training.experiment import get_git_info, write_run_config_snapshot

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import CONTRACT_VERSION

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
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_run_metadata(run, cfg):
    from uni_rl.offpolicy.thread_budget import resolve_torch_thread_runtime

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
                for name in ("torch", "numpy", "unilab", "unilab-rl", "rsl-rl-lib", "mujoco")
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
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self.completed_lengths: deque[int] = deque(maxlen=100)
        self.completed_timeouts: deque[bool] = deque(maxlen=100)

    def update(self, rewards, terminated, truncated):
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
        return {
            "episode/return": float(np.mean(self.completed_returns)),
            "episode/length": float(np.mean(self.completed_lengths)),
            "episode/timeout_rate": float(np.mean(self.completed_timeouts)),
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
            cfg.algo.algo in {"ppo", "appo"}
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
        with (Path(self._log_dir) / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        self.log_collector(steps, 0)
        self.update_ep_length(metrics.get("episode/length", 0))
        self.update_timeout_rate(metrics.get("episode/timeout_rate", 0))
        self.update_collector_timing(
            {
                k.removeprefix("timing/collector_"): v
                for k, v in metrics.items()
                if k.startswith("timing/collector_")
            }
        )
        self.update_staging_pool(
            int(metrics.get("staging_rollouts", 0)), int(metrics.get("staging_pool_capacity", 0))
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
                if not k.startswith(("reward/", "timing/"))
                and k
                not in {
                    "perf/iter_ms",
                    "perf/collector_active_steps_per_sec",
                    "episode/timeout_rate",
                }
            },
            reward=metrics.get("episode/return"),
            reward_components={k: v for k, v in metrics.items() if k.startswith("reward/")},
            extra_info={
                "steps_per_sec": throughput,
                "collector_active_steps_per_sec": metrics.get(
                    "perf/collector_active_steps_per_sec"
                ),
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
