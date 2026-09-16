"""Versioned teacher/student snapshots and supported legacy policy loading.

Top-level actor/critic states serve inference. FlashSAC's learner state references
those same tensors so training-state archival does not duplicate model storage.
Training resume is not exposed by the CLI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from sharpa_rl_unilab.algos.hora.flashsac import TeacherFlashLearner
from sharpa_rl_unilab.algos.hora.legacy import migrate_actor_state
from sharpa_rl_unilab.algos.hora.on_policy import TeacherAPPOLearner
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CONTRACT_VERSION

from .configuration import configure_threads, migrate_checkpoint_config, resolve_device
from .policy import make_actor


def frozen_weights(module: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_metadata(learner):
    result = {}
    for name in ("optimizer", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        optimizer = getattr(learner, name, None)
        if optimizer is not None:
            result[name] = [
                {k: v for k, v in group.items() if k != "params"}
                for group in optimizer.param_groups
            ]
    return result


def save_teacher(path, cfg, actor, critic, learner, counters, elapsed):
    snapshot = {
        "contract": CONTRACT_VERSION,
        "stage": "teacher",
        "algorithm": str(cfg.algo.algo),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "actor": frozen_weights(actor),
        "critic": frozen_weights(critic),
        "counters": dict(counters),
        "wall_seconds": elapsed,
        "optimizer_parameters": optimizer_metadata(learner),
    }
    snapshot["parameter_counts"] = {
        "actor": sum(p.numel() for p in actor.parameters()),
        "priv_encoder": sum(p.numel() for p in actor.shared.priv_encoder.parameters()),
        "critic": sum(p.numel() for p in critic.parameters()),
    }
    if isinstance(learner, TeacherFlashLearner):
        snapshot["learner"] = learner.get_state_dict()
        for name in ("actor", "critic"):
            snapshot["learner"][name] = snapshot[name]
    else:
        snapshot["optimizer"] = learner.optimizer.state_dict()
        if isinstance(learner, TeacherAPPOLearner):
            snapshot["target_actor"] = frozen_weights(learner.target_actor)
    torch.save(snapshot, path)
    return snapshot


def load_policy(path, device: str | None = "cpu", *, stage=None, configure_runtime=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT_VERSION:
        raise ValueError(
            f"Checkpoint must use {CONTRACT_VERSION}; old flat/shared-HORA/FlashSAC weights require retraining"
        )
    if stage is not None and checkpoint.get("stage") != stage:
        raise ValueError(f"Expected a {stage} checkpoint")
    if checkpoint.get("stage") not in ("teacher", "student") or checkpoint.get("algorithm") not in (
        "ppo",
        "appo",
        "flashsac",
    ):
        raise ValueError("Unknown checkpoint stage or algorithm")
    cfg = migrate_checkpoint_config(checkpoint["config"])
    device = resolve_device(device if device is not None else cfg.training.device)
    if configure_runtime:
        configure_threads(cfg)
    actor = make_actor(cfg, device, student=checkpoint["stage"] == "student")
    actor.load_state_dict(migrate_actor_state(checkpoint["actor"]), strict=True)
    actor.eval()
    return actor, cfg, checkpoint
