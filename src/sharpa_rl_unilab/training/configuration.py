"""Shared runtime resources and migration of supported checkpoint configs."""

from __future__ import annotations

import inspect
import math
from typing import Any, cast

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
    # This field never affected observations. Drop it from supported old snapshots.
    cfg.env.observations.actor.terms.frame.params.pop("disable_tactile_ids", None)
    if cfg.algo.algo in {"ppo", "appo"}:
        algorithm = cfg.algo.get("algorithm", {})
        if algorithm.get("kl_mode") == "legacy":
            algorithm["kl_mode"] = "log_epsilon"
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
    # Model defaults are versioned by their saved parameterization, never by
    # the defaults of a newly installed package. Preserve experimental tanh.
    if "model" in cfg and "std_parameterization" not in cfg.model:
        flash = cfg.algo.algo == "flashsac"
        cfg.model.std_parameterization = "legacy_tanh" if flash else "legacy_scalar"
        cfg.model.std_mode = "state_dependent" if flash else "state_independent"
        cfg.model.setdefault("action_mapping", "tanh" if flash else "clip")
        cfg.model.setdefault("initial_std", 1.0)
        cfg.model.log_std_bounds = (
            [-10.0, 2.0] if flash else [-13.815510557964274, 13.815510557964274]
        )
    if cfg.algo.algo == "flashsac":
        params = cfg.algo.algo_params
        old_exploration = {
            "noise_zeta_mu": params.pop("actor_noise_zeta_mu", 2.0),
            "noise_zeta_max": params.pop("actor_noise_zeta_max", 16),
        }
        cfg.algo.setdefault("exploration", old_exploration)
    return cfg


def config_dict(cfg) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def algorithm_options(cfg, cls, *, extra_options=()):
    params = config_dict(cfg.algo.algorithm)
    allowed = set(inspect.signature(cls.__init__).parameters) | set(extra_options)
    if unsupported := params.keys() - allowed:
        raise ValueError(f"Unsupported algorithm options: {sorted(unsupported)}")
    return params


def training_budget(cfg):
    """Exactly one stopping budget; all teacher snapshots use update rounds."""
    iterations = cfg.algo.get("max_iterations")
    transitions = cfg.training.max_transitions
    if cfg.algo.algo == "flashsac" and (iterations is None or transitions is not None):
        raise ValueError(
            "FlashSAC requires algo.max_iterations > 0 and training.max_transitions=null"
        )
    if (iterations is None) == (transitions is None):
        raise ValueError(
            "Select exactly one budget: algo.max_iterations or training.max_transitions; "
            "set the other to null"
        )
    if cfg.algo.algo == "appo":
        if (
            min(
                int(cfg.algo.async_queue_size),
                int(cfg.algo.staging_pool_size),
                int(cfg.algo.steps_per_env),
            )
            < 1
        ):
            raise ValueError(
                "APPO queue capacity, staging capacity and rollout length must be positive"
            )
    if int(cfg.algo.save_interval) < 0:
        raise ValueError("algo.save_interval must be nonnegative")
    n = int(cfg.algo.num_envs)
    target = int(iterations) if iterations is not None else math.ceil(int(transitions) / n) * n
    if target <= 0:
        raise ValueError("Training budget must be positive")
    return ("policy_version" if iterations is not None else "received"), target
