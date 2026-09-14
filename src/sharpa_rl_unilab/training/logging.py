"""UniLab's Rich/TensorBoard logging adapted to transition-budgeted training."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from uni_rl.logging.common import BaseTrainingLogger, _fmt_number, _fmt_time


class EpisodeStatistics:
    """Track completed episodes across fresh rollout packets, before replay reuse."""

    def __init__(self, num_envs: int):
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.completed_returns: deque[float] = deque(maxlen=100)
        self.completed_lengths: deque[int] = deque(maxlen=100)

    def update(self, rewards, terminated, truncated):
        for reward, done in zip(rewards, np.asarray(terminated) | np.asarray(truncated)):
            self.returns += reward
            self.lengths += 1
            self.completed_returns.extend(self.returns[done].tolist())
            self.completed_lengths.extend(self.lengths[done].tolist())
            self.returns[done] = 0
            self.lengths[done] = 0

    def metrics(self):
        if not self.completed_returns:
            return {}
        return {
            "episode/return": float(np.mean(self.completed_returns)),
            "episode/length": float(np.mean(self.completed_lengths)),
        }


class TrainingLogger(BaseTrainingLogger):
    def __init__(self, run: Path, cfg, target: int):
        backend = str(cfg.training.get("logger", "tensorboard"))
        if backend not in {"tensorboard", "none", "no_print"}:
            raise ValueError("training.logger must be tensorboard, none or no_print")
        super().__init__(
            algo_name=f"{str(cfg.algo.algo).upper()} {cfg.protocol.stage}",
            max_iterations=target,
            num_envs=int(cfg.algo.num_envs),
            env_name=str(cfg.training.task_name),
            log_dir=str(run),
            log_backend=backend,
            wandb_project="",
            wandb_entity=None,
            wandb_name="",
            wandb_group=None,
            wandb_job_type=None,
            wandb_tags=None,
            wandb_notes=None,
            tensorboard_subdir=None,
        )
        self._previous_steps = 0
        self._previous_seconds = 0.0

    def log(self, metrics):
        metrics = dict(metrics)
        steps, seconds = int(metrics["received"]), float(metrics["wall_seconds"])
        metrics["perf/transitions_per_second"] = (steps - self._previous_steps) / max(
            seconds - self._previous_seconds, 1e-9
        )
        self._previous_steps, self._previous_seconds = steps, seconds
        self._iteration = steps
        self._latest_metrics = metrics
        self._status = "Training"
        with (Path(self._log_dir) / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=steps)
        self._refresh()

    def status(self, message: str):
        self._status = message
        self._refresh(force=True)

    def _build_display(self) -> Panel:
        metrics = self._latest_metrics
        seconds = float(metrics.get("wall_seconds", 0))
        remaining = seconds * (self.max_iterations - self._iteration) / max(self._iteration, 1)
        progress = Text(
            f"Transitions {self._iteration:,} / {self.max_iterations:,}"
            f" ({100 * self._iteration / self.max_iterations:.1f}%)"
            f"  |  Time {_fmt_time(seconds)}  |  ETA {_fmt_time(remaining)}",
            style="bold cyan",
        )
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
        table.add_column("Losses & Metrics", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Rollouts & Budget", style="green")
        table.add_column("Value", justify="right")
        # Native learners expose aliases; show each quantity once in the panel.
        fields = (
            ("Policy loss", ("loss/policy_loss", "surrogate_loss", "surrogate", "actor_loss")),
            ("Value / critic loss", ("loss/value_loss", "value_loss", "value", "critic_loss")),
            ("Entropy", ("policy/entropy", "entropy")),
            ("Approx. KL", ("ppo/approx_kl", "kl")),
            ("Clip fraction", ("ppo/clip_fraction",)),
            ("Learning rate", ("optim/learning_rate",)),
            ("Gradient norm", ("grad/global_norm",)),
            ("V-trace clip fraction", ("vtrace/rho_clip_fraction",)),
            ("Policy lag (mean)", ("policy_lag_mean",)),
            ("Latent MSE", ("latent_mse",)),
        )
        losses = [
            (label, _fmt_number(float(metrics[key])))
            for label, keys in fields
            for key in [next((key for key in keys if key in metrics), None)]
            if key is not None
        ]
        stats = [
            ("Environments", f"{self.num_envs:,}"),
            ("Collected", f"{int(metrics.get('collected', 0)):,}"),
            ("Received", f"{self._iteration:,}"),
            ("Training samples", f"{int(metrics.get('training_samples', 0)):,}"),
            ("Optimizer updates", f"{int(metrics.get('optimizer_updates', 0)):,}"),
            ("Transitions / second", f"{metrics.get('perf/transitions_per_second', 0):,.0f}"),
        ]
        if "reuse_ratio" in metrics:
            stats.append(("Sample reuse", f"{metrics['reuse_ratio']:.2f}x"))
        if "episode/return" in metrics:
            stats.extend(
                [
                    ("Episode return (last 100)", f"{metrics['episode/return']:.3f}"),
                    ("Episode length (last 100)", f"{metrics['episode/length']:.1f}"),
                ]
            )
        else:
            stats.append(("Episodes", "Waiting for completion"))
        for index in range(max(len(losses), len(stats))):
            table.add_row(
                *(losses[index] if index < len(losses) else ("", "")),
                *(stats[index] if index < len(stats) else ("", "")),
            )
        return Panel(
            Group(
                progress, table, Text(self._status, style="dim"), Text(self._log_dir, style="dim")
            ),
            title=Text(f"UniLab Training | {self.algo_name} | {self.env_name}", style="bold"),
            border_style="bright_blue",
        )

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        self._status = "Training complete"
        if self._last_save:
            self._status += f" | Saved {self._last_save}"
        self.close()
