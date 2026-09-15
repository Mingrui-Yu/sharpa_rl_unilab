"""Shared runtime resources and migration of supported checkpoint configs."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.offpolicy.thread_budget import apply_torch_thread_runtime, resolve_torch_thread_runtime


def resolve_device(device=None):
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_threads(cfg, *, role="learner"):
    runtime = resolve_torch_thread_runtime(cfg.training.torch_threads)
    apply_torch_thread_runtime(runtime, role=role)
    return runtime


def migrate_checkpoint_config(config) -> DictConfig:
    # Resolve old interpolations before removing their source groups. Never
    # merge new task defaults over the physical/model settings in a checkpoint.
    cfg = OmegaConf.create(config)
    assert isinstance(cfg, DictConfig)
    OmegaConf.resolve(cfg)
    if "budget" in cfg:
        old = cfg.pop("budget")
        cfg.training.setdefault("max_transitions", old.get("transitions"))
        cfg.algo.setdefault("checkpoint", old.get("checkpoint"))
        cfg.algo.setdefault("save_interval", 50)
        cfg.distillation.setdefault("log_every", old.get("log_every", 10000))
        if cfg.algo.algo == "appo":
            cfg.algo.setdefault("async_queue_size", old.get("async_queue_size", 4))
    if "hardware" in cfg:
        old = cfg.pop("hardware")
        cfg.training.setdefault("num_envs", old.get("num_envs", cfg.algo.num_envs))
        cfg.training.device = cfg.training.get("device") or old.get("device")
        if cfg.algo.algo == "appo":
            cfg.algo.setdefault("collector_device", old.get("collector_device"))
        threads = old.get("torch_threads")
        if threads is not None:
            # The old scalar set intra-op only, even for FlashSAC. Preserve the
            # caller's inter-op/compile pools instead of imposing new defaults.
            from torch._inductor import config as inductor_config

            cfg.training.torch_threads = {
                "enabled": True,
                "learner_num_threads": int(threads),
                "collector_num_threads": int(threads),
                "learner_num_interop_threads": torch.get_num_interop_threads(),
                "collector_num_interop_threads": torch.get_num_interop_threads(),
                "compile_threads": inductor_config.compile_threads,
                "set_env_vars": False,
            }
        elif cfg.algo.algo != "flashsac":
            cfg.training.torch_threads = {"enabled": False}
        # With a null scalar, FlashSAC already used the structured thread config.
    return cfg
