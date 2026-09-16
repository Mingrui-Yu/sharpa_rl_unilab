"""Teacher entry point: validate the budget, set resources and select a runner."""

import time

import numpy as np
import torch

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import validate_observation_config

from .configuration import configure_threads, resolve_device, training_budget


def train_teacher(cfg):
    started = time.monotonic()
    if cfg.algo.checkpoint is not None:
        raise ValueError(
            "algo.checkpoint selects evaluation weights; training resume is not supported"
        )
    validate_observation_config(cfg)
    progress_key, target = training_budget(cfg)
    if cfg.algo.algo == "flashsac":
        from .flashsac_runtime import train_flashsac

        return train_flashsac(cfg)
    configure_threads(cfg)
    torch.manual_seed(int(cfg.algo.seed))
    np.random.seed(int(cfg.algo.seed))
    cfg.training.device = resolve_device(cfg.training.device)
    if cfg.training.get("devices"):
        raise ValueError("The fixed-hardware comparison runner currently uses one learner device")
    if cfg.algo.algo == "appo":
        from .appo_runtime import train_appo

        cfg.algo.collector_device = resolve_device(cfg.algo.collector_device or cfg.training.device)
        if cfg.algo.collector_seed is None:
            cfg.algo.collector_seed = int(cfg.algo.seed) + 1
        return train_appo(cfg, progress_key, target, started)
    if cfg.algo.algo == "ppo":
        from .ppo_runtime import train_ppo

        return train_ppo(cfg, progress_key, target, started)
    raise ValueError(f"Unknown teacher algorithm: {cfg.algo.algo}")
