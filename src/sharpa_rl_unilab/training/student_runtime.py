"""Common student rollout: one latent-MSE update, then its own deterministic action."""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.algos.hora.models import make_student
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CONTRACT_VERSION, HISTORY_SHAPE
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv

from .checkpoints import file_digest, frozen_weights, load_policy
from .configuration import configure_threads, resolve_device
from .logging import EpisodeStatistics, TrainingLogger, write_run_metadata
from .policy import deterministic_actions, tensor_obs


class StudentTrainer:
    def __init__(self, teacher, learning_rate, device):
        self.actor = make_student(teacher).eval()
        assert self.actor.shared.adapt_tconv is not None
        self.history_normalizer = EmpiricalNormalization(HISTORY_SHAPE, device)
        self.optimizer = torch.optim.Adam(
            self.actor.shared.adapt_tconv.parameters(), lr=learning_rate
        )
        self.device = device

    def update_and_act(self, obs):
        data = tensor_obs({key: obs[key] for key in ("priv_info", "proprio_hist")}, self.device)
        # The history normalizer's forward updates exactly once for this new batch.
        normalized_history = self.history_normalizer(data["proprio_hist"])
        with torch.no_grad():
            target = self.actor.shared.encode_privileged_info(data["priv_info"])
        latent = self.actor.shared.encode_proprio_history(normalized_history)
        loss = torch.nn.functional.mse_loss(latent, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        # Inference reuses the updated history statistics without accumulating them.
        return deterministic_actions(self.actor, obs, self.device, self.history_normalizer), float(
            loss.detach()
        )


def train_student(checkpoint, *, overrides=(), device=None):
    started = time.monotonic()
    teacher, cfg, source = load_policy(checkpoint, "cpu", stage="teacher", configure_runtime=False)
    allowed = (
        "distillation.",
        "evaluation.",
        "training.device",
        "training.torch_threads.",
        "training.log_dir",
        "training.logger",
        "training.no_play",
        "training.play_render_mode",
        "training.play_env_num",
        "training.play_steps",
        "training.render_spacing",
        "training.cam_",
    )
    if any(not override.split("=", 1)[0].startswith(allowed) for override in overrides):
        raise ValueError(
            "Distillation overrides may set distillation/evaluation fields, training.device, training.torch_threads, "
            "training.log_dir, training.logger and playback settings"
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    cfg.protocol.stage = "student"
    device = resolve_device(device if device is not None else cfg.training.device)
    cfg.training.device = device
    teacher.to(device)
    configure_threads(cfg)
    seed = int(cfg.distillation.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = int(cfg.distillation.num_envs)
    target = math.ceil(int(cfg.distillation.transitions) / n) * n
    if target <= 0:
        raise ValueError("distillation.transitions must be positive")
    cfg.algo.num_envs = n
    # Keep distillation runs separate from teacher runs unless explicitly overridden.
    run_override = next(
        (item.split("=", 1)[1] for item in overrides if item.startswith("training.log_dir=")), None
    )
    run = Path(
        run_override or f"logs/hora_distill/{source['algorithm']}_seed_{seed}_{time.time_ns()}"
    )
    cfg.training.log_dir = str(run)
    run.mkdir(parents=True, exist_ok=False)
    write_run_metadata(run, cfg)
    trainer = StudentTrainer(teacher, float(cfg.distillation.learning_rate), device)
    provenance = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "sha256": file_digest(checkpoint),
        "training_seed": int(source["config"]["algo"]["seed"]),
    }
    teacher_cost = {"counters": source["counters"], "wall_seconds": source["wall_seconds"]}
    counters = {
        "collected": 0,
        "received": 0,
        "training_samples": 0,
        "optimizer_updates": 0,
        "policy_version": 0,
    }

    def save(name):
        path = run / name
        elapsed = time.monotonic() - started
        torch.save(
            {
                "contract": CONTRACT_VERSION,
                "stage": "student",
                "algorithm": source["algorithm"],
                "config": OmegaConf.to_container(cfg, resolve=True),
                "actor": frozen_weights(trainer.actor),
                "history_normalizer": frozen_weights(trainer.history_normalizer),
                "optimizer": trainer.optimizer.state_dict(),
                "teacher": provenance,
                "teacher_cost": teacher_cost,
                "wall_seconds": elapsed,
                "total_wall_seconds": source["wall_seconds"] + elapsed,
                "counters": dict(counters),
            },
            path,
        )
        return path

    with ExitStack() as resources:
        env = SharpaTeacherEnv(cfg, n)
        resources.callback(env.close)
        save_every = int(cfg.distillation.save_every)
        next_save, next_log = save_every, int(cfg.distillation.log_every)
        episodes = EpisodeStatistics(n)
        logger = TrainingLogger(run, cfg, target)
        resources.callback(logger.close)
        logger.start(status="Initializing student rollouts...")
        obs, _ = env.reset(seed=seed)
        metric_sums = defaultdict(float)
        log_steps = 0
        while counters["collected"] < target:
            actions, loss = trainer.update_and_act(obs)
            step_started = time.perf_counter()
            state = env.step(actions)
            metric_sums["timing/collector_env_step_ms"] += (
                time.perf_counter() - step_started
            ) * 1000
            metric_sums["latent_mse"] += loss
            for key, value in state.info.get("log", {}).items():
                if key.startswith("reward/"):
                    metric_sums[key] += float(value)
            log_steps += 1
            episodes.update(state.reward[None], state.terminated[None], state.truncated[None])
            obs = state.obs
            for key in ("collected", "received", "training_samples"):
                counters[key] += n
            counters["optimizer_updates"] += 1
            counters["policy_version"] += 1
            if counters["collected"] >= next_log or counters["collected"] == target:
                metrics = {
                    **counters,
                    **{key: value / log_steps for key, value in metric_sums.items()},
                    "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                    "wall_seconds": time.monotonic() - started,
                    **episodes.metrics(),
                }
                logger.log(metrics)
                metric_sums.clear()
                log_steps = 0
                next_log = counters["collected"] + max(int(cfg.distillation.log_every), 1)
            if save_every > 0 and counters["collected"] >= next_save:
                logger.log_save(str(save(f"student_{counters['collected']}.pt")))
                next_save = (counters["collected"] // save_every + 1) * save_every
        final = save("student_final.pt")
        logger.log_save(str(final))
        logger.finish()
        return final


def main():
    from .playback import play_checkpoint

    parser = argparse.ArgumentParser(
        description="Distill any protocol-v2 PPO/APPO/FlashSAC teacher"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args, overrides = parser.parse_known_args()
    checkpoint = train_student(args.checkpoint, overrides=overrides, device=args.device)
    print(checkpoint)
    play_checkpoint(checkpoint, device=args.device)


if __name__ == "__main__":
    main()
