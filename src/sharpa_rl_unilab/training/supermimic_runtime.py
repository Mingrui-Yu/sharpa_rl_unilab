"""Sharpa SuperMimic teacher and optional single-stage training."""

from __future__ import annotations

import time
from contextlib import ExitStack

import torch

from sharpa_rl_unilab.algos.supermimic.learner import SharpaSuperMimicLearner
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv

from .checkpoints import save_supermimic, save_teacher
from .logging import TrainingLogger, create_run, training_counters
from .policy import make_models
from .supermimic_mdp import SharpaSuperMimicCapture, SharpaSuperMimicMDP


def _save(path, cfg, learner, counters, started):
    save = save_supermimic if bool(cfg.algo.get("single_stage", False)) else save_teacher
    return save(
        path,
        cfg,
        learner.checkpoint_actor,
        learner.critic,
        learner,
        counters,
        time.monotonic() - started,
    )


def train_supermimic(cfg, started):
    cfg.protocol.stage = "direct" if bool(cfg.algo.get("single_stage", False)) else "teacher"
    for name in (
        "buffer_steps",
        "warmup_steps",
        "wm_window",
        "wm_batches",
        "wm_batch",
        "pi_batch",
        "horizon",
        "critic_epochs",
        "critic_minibatches",
    ):
        if int(cfg.algo[name]) < 1:
            raise ValueError(f"algo.{name} must be positive")
    if float(cfg.algo.utd) <= 0:
        raise ValueError("algo.utd must be positive")
    for name in ("pi_opt", "wm_opt"):
        if cfg.algo[name] not in ("adam", "muon"):
            raise ValueError(f"algo.{name} must be 'adam' or 'muon', got {cfg.algo[name]!r}")
    if int(cfg.algo.buffer_steps) < int(cfg.algo.wm_window):
        raise ValueError("algo.buffer_steps must be at least algo.wm_window")
    checkpoint_prefix = "policy" if cfg.protocol.stage == "direct" else "teacher"
    run = create_run(cfg)
    actor, critic, _ = make_models(cfg, cfg.training.device)
    counters = training_counters()
    with ExitStack() as resources:
        env = SharpaTeacherEnv(cfg, int(cfg.algo.num_envs), auto_reset=False)
        resources.callback(env.close)
        capture = SharpaSuperMimicCapture(env)
        mdp = SharpaSuperMimicMDP(env, cfg.training.device)
        learner = SharpaSuperMimicLearner(env, cfg, actor, critic, mdp, capture)
        logger = TrainingLogger(run, cfg, int(cfg.algo.max_iterations))
        resources.callback(logger.close)
        logger.start(status="Warming up Sharpa world model replay...")
        learner.collect(max(int(cfg.algo.warmup_steps), int(cfg.algo.wm_window)))
        counters["collected"] = learner.collected
        counters["received"] = learner.received
        accumulator = 0.0
        for iteration in range(int(cfg.algo.max_iterations)):
            iteration_started = time.monotonic()
            accumulator += 1.0 / float(cfg.algo.utd)
            collect_steps = int(accumulator)
            accumulator -= collect_steps
            metrics = learner.collect(collect_steps)
            counters["collected"] = learner.collected
            counters["received"] = learner.received
            wm_loss = torch.zeros((), device=cfg.training.device)
            wm_error = torch.zeros((), device=cfg.training.device)
            for _ in range(int(cfg.algo.wm_batches)):
                wm_loss, wm_error = learner.wm_update()
            metrics.update({"loss/world_model": wm_loss, "loss/world_model_last": wm_error})
            counters["training_samples"] += (
                int(cfg.algo.wm_batches) * int(cfg.algo.wm_batch) * int(cfg.algo.wm_window)
            )
            counters["optimizer_updates"] += int(cfg.algo.wm_batches)
            if iteration >= int(cfg.algo.wm_only_iters):
                metrics.update(learner.policy_update())
                counters["actor_updates"] += 1
                critic_updates = int(cfg.algo.critic_epochs) * int(cfg.algo.critic_minibatches)
                counters["critic_updates"] += critic_updates
                counters["optimizer_updates"] += 1 + critic_updates
                counters["training_samples"] += (
                    int(cfg.algo.pi_batch) * int(cfg.algo.horizon) * (1 + critic_updates)
                )
            counters["policy_version"] += 1
            metrics.update(counters)
            metrics.update(learner.episodes.metrics())
            metrics["wall_seconds"] = time.monotonic() - started
            metrics = {
                key: float(value) if torch.is_tensor(value) else value
                for key, value in metrics.items()
            }
            elapsed = time.monotonic() - iteration_started
            logger.log(metrics, timings={"iteration_time": elapsed, "train_time": elapsed})
            if (
                int(cfg.algo.save_interval) > 0
                and counters["policy_version"] % int(cfg.algo.save_interval) == 0
            ):
                path = run / f"{checkpoint_prefix}_iteration_{counters['policy_version']}.pt"
                _save(path, cfg, learner, counters, started)
                logger.log_save(str(path))
        final = run / f"{checkpoint_prefix}_final.pt"
        _save(final, cfg, learner, counters, started)
        logger.log_save(str(final))
        logger.finish()
        return final


__all__ = ["train_supermimic"]
