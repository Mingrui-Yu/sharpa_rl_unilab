"""Common student rollout: one latent-MSE update, then its own deterministic action."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.common.normalization import EmpiricalNormalization

from sharpa_rl_unilab.algos.hora.teacher import frozen_weights, make_student
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import CONTRACT_VERSION, SharpaTeacherEnv

from .evaluation import deterministic_actions, file_digest, write_run_metadata
from .teacher_runtime import load_policy, tensor_obs


class StudentTrainer:
    def __init__(self, teacher, learning_rate, device):
        self.actor = make_student(teacher).eval()
        assert self.actor.shared.adapt_tconv is not None
        self.history_normalizer = EmpiricalNormalization((30, 49), device)
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
    teacher, cfg, source = load_policy(checkpoint, device or "cpu", stage="teacher")
    allowed = ("distillation.", "evaluation.", "hardware.", "training.log_dir")
    if any(not override.split("=", 1)[0].startswith(allowed) for override in overrides):
        raise ValueError(
            "Distillation overrides may set distillation/evaluation/hardware fields and training.log_dir"
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    cfg.protocol.stage = "student"
    device = device or str(cfg.hardware.device)
    teacher.to(device)
    torch.set_num_threads(int(cfg.hardware.torch_threads))
    seed = int(cfg.distillation.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = int(cfg.distillation.num_envs)
    target = math.ceil(int(cfg.distillation.transitions) / n) * n
    if target <= 0:
        raise ValueError("distillation.transitions must be positive")
    cfg.algo.num_envs = n
    # A teacher's run directory must never be reused by its student.
    run_override = next(
        (item.split("=", 1)[1] for item in overrides if item.startswith("training.log_dir=")), None
    )
    run = Path(run_override or Path(checkpoint).parent / f"student_seed_{seed}_{time.time_ns()}")
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

    env = SharpaTeacherEnv(cfg, n)
    save_every = int(cfg.distillation.save_every)
    next_save, next_log = save_every, int(cfg.budget.log_every)
    try:
        obs, _ = env.reset(seed=seed)
        while counters["collected"] < target:
            actions, loss = trainer.update_and_act(obs)
            obs = env.step(actions).obs
            for key in ("collected", "received", "training_samples"):
                counters[key] += n
            counters["optimizer_updates"] += 1
            counters["policy_version"] += 1
            if counters["collected"] >= next_log or counters["collected"] == target:
                metrics = {
                    **counters,
                    "latent_mse": loss,
                    "wall_seconds": time.monotonic() - started,
                }
                with (run / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                next_log = counters["collected"] + max(int(cfg.budget.log_every), 1)
            if save_every > 0 and counters["collected"] >= next_save:
                save(f"student_{counters['collected']}.pt")
                next_save = (counters["collected"] // save_every + 1) * save_every
        return save("student_final.pt")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Distill any protocol-v2 PPO/APPO/FlashSAC teacher"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args, overrides = parser.parse_known_args()
    print(train_student(args.checkpoint, overrides=overrides, device=args.device))


if __name__ == "__main__":
    main()
