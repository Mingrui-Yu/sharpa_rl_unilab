"""HORA-specific interactive playback session factories.

The generic session cores (``RslRlPlaybackSession``, ``OffPolicyPlaybackSession``)
and the non-HORA factories stay in UniLab's
``unilab.visualization.interactive_playback``; only the HORA branches that depend
on ``sharpa_rl_unilab.algos.hora`` live here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from unilab.training import get_log_root
from unilab.utils.checkpoint import resolve_appo_checkpoint_path, resolve_task_checkpoint_path
from unilab.utils.sim2sim import policy_load_dim_guard, resolve_sim2sim_config
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    select_torch_device,
)

LogFn = Callable[[str], None]


def _normalize_checkpoint_value(value: object) -> str | None:
    """Normalize a raw checkpoint selector value; sentinel values map to ``None``."""
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "-1", "None", "null"} else text


def _resolve_appo_checkpoint_from_cfg(
    cfg: Any,
    *,
    root_dir: str | Path,
) -> tuple[str | None, str | None]:
    selected_checkpoint = _normalize_checkpoint_value(
        OmegaConf.select(cfg, "algo.checkpoint", default=None)
    )
    if selected_checkpoint is not None:
        selected_path, selected_dir = resolve_task_checkpoint_path(
            root_dir,
            task_name=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            algo_log_name=str(cfg.algo.algo_log_name),
            checkpoint=selected_checkpoint,
            log_root=getattr(cfg.training, "log_root", None),
        )
        return (
            str(selected_path) if selected_path is not None else None,
            str(selected_dir) if selected_dir is not None else None,
        )

    base_log_dir = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    checkpoint_path, checkpoint_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _build_hora_appo_actor(
    *,
    env: Any,
    wrapped_env: Any,
    rl_cfg: dict[str, Any],
    device: str,
) -> tuple[Any, int, int]:
    """Build the HORA APPO actor and return it with ``(obs_dim, action_dim)``."""
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from sharpa_rl_unilab.algos.hora.models import build_hora_shared_actor_critic
    from sharpa_rl_unilab.algos.hora.rsl_rl_compat import (
        convert_config_v3_to_v4,
        is_rsl_rl_v4,
        is_rsl_rl_v5,
    )

    from .play_hora_appo import _update_hora_obs_groups

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    rl_cfg_dict = deepcopy(rl_cfg)

    obs_td = wrapped_env.get_observations()
    num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
    obs_dim = int(obs_td["actor"].shape[-1])
    priv_info_dim = int(obs_td["priv_info"].shape[-1])
    if priv_info_dim <= 0:
        raise ValueError("HORA APPO interactive play requires privileged info.")
    _update_hora_obs_groups(rl_cfg_dict, obs_dim=obs_dim, priv_info_dim=priv_info_dim)
    if is_rsl_rl_v5():
        pass
    elif is_rsl_rl_v4():
        rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    critic_cfg = deepcopy(rl_cfg_dict.get("critic") or rl_cfg_dict.get("actor") or {})
    critic_cfg.pop("class_name", None)
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)
    shared_model = build_hora_shared_actor_critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=priv_info_dim,
        actor_cfg=actor_cfg,
        critic_cfg=critic_cfg,
    ).to(device)
    td_example = TensorDict(
        {
            "actor": torch.zeros((num_envs, obs_dim), device=device),
            "priv_info": torch.zeros(
                (num_envs, priv_info_dim),
                device=device,
            ),
        },
        batch_size=num_envs,
    )
    actor = actor_cls(
        td_example,
        rl_cfg_dict["obs_groups"],
        "actor",
        action_dim,
        shared_model=shared_model,
        **actor_cfg,
    )
    return actor.to(device).eval(), obs_dim, action_dim


def create_hora_appo_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    rl_cfg: dict[str, Any],
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a HORA APPO interactive playback session (grouped actor inputs)."""
    from sharpa_rl_unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapped_env = HoraRslRlVecEnvWrapper(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy = None
    actor = None
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path, checkpoint_dir = _resolve_appo_checkpoint_from_cfg(cfg, root_dir=root_dir)
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            log(
                "WARNING: no APPO checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
        else:
            resolve_sim2sim_config(
                checkpoint_dir,
                cfg,
                algo_name="appo",
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            actor, actor_obs_dim, action_dim = _build_hora_appo_actor(
                env=env,
                wrapped_env=wrapped_env,
                rl_cfg=rl_cfg,
                device=device_name,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=actor_obs_dim,
                env_action_dim=action_dim,
                algo_name="appo",
            ):
                actor.load_state_dict(checkpoint["actor"])
            policy = actor
            log(f"Loading HORA APPO checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        RslRlPlaybackSession(
            env=env,
            wrapped_env=wrapped_env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            policy=policy,
            num_envs=playback_cfg.num_envs,
            actor=actor,
        ),
        policy_obs_mode,
        checkpoint_path,
    )


def _default_hora_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    from unilab.base.config_adapter import BackendAdapter, create_env

    from sharpa_rl_unilab.algos.hora.distill import (
        build_student_actor_and_normalizer,
        cfg_with_checkpoint_runtime,
        load_distilled_checkpoint,
        student_policy,
    )
    from sharpa_rl_unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

    from .hora_distill_config import apply_teacher_defaults
    from .stage2 import format_hora_stage2_checkpoint_error, resolve_hora_stage2_checkpoint_path

    def _scene_visual_materializer() -> Any:
        from unilab.base import config_adapter

        return config_adapter.materialize_scene_visual_override

    return {
        "apply_teacher_defaults": apply_teacher_defaults,
        "build_play_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="hora_distill",
            scene_materializer=_scene_visual_materializer(),
        ).build_play_env_cfg_override(),
        "build_student_actor_and_normalizer": build_student_actor_and_normalizer,
        "cfg_with_checkpoint_runtime": cfg_with_checkpoint_runtime,
        "create_env": create_env,
        "format_stage2_play_checkpoint_error": format_hora_stage2_checkpoint_error,
        "get_log_root": get_log_root,
        "load_distilled_checkpoint": load_distilled_checkpoint,
        "resolve_stage2_checkpoint_path": lambda cfg: resolve_hora_stage2_checkpoint_path(
            cfg,
            root_dir=root_dir,
        ),
        "student_policy": student_policy,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
        "checkpoint_reader": torch.load,
    }


def create_hora_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an interactive playback session for HORA stage-2 student checkpoints."""

    resolved_deps = dict(_default_hora_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    load_path, load_path_dir = resolved_deps["resolve_stage2_checkpoint_path"](cfg)
    checkpoint_path = str(load_path) if load_path is not None else None
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if load_path is None or load_path_dir is None or not Path(load_path).exists():
            task_log_root = resolved_deps["get_log_root"](Path(root_dir), cfg) / str(
                cfg.training.task_name
            )
            log(
                resolved_deps["format_stage2_play_checkpoint_error"](
                    cfg,
                    task_log_root=task_log_root,
                    load_path=load_path,
                    load_path_dir=load_path_dir,
                )
            )
            log("WARNING: falling back to zero actions.")
            runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)
        else:
            resolve_sim2sim_config(
                load_path_dir,
                cfg,
                algo_name="hora_distill",
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            log(f"Loading distilled checkpoint: {load_path}")
            checkpoint = resolved_deps["checkpoint_reader"](
                load_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
                    f"(found keys: {set(checkpoint.keys())})."
                )
            # cfg_with_checkpoint_runtime no longer composes teacher
            # defaults; the caller owns that composition (issue #1480).
            runtime_cfg = resolved_deps["cfg_with_checkpoint_runtime"](
                resolved_deps["apply_teacher_defaults"](cfg), checkpoint
            )
    else:
        runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)

    env_cfg_override = resolved_deps["build_play_env_cfg_override"](runtime_cfg)
    create_env = resolved_deps["create_env"]
    try:
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=str(runtime_cfg.training.task_name),
        )
    except TypeError:
        if deps is None:
            raise
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
        )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    torch_device = torch.device(device_name)

    if playback_cfg.action_mode == "policy" and load_path is not None and Path(load_path).exists():
        actor, hist_normalizer = resolved_deps["build_student_actor_and_normalizer"](
            wrapped_env,
            runtime_cfg,
            device=torch_device,
        )
        with policy_load_dim_guard(algo_name="hora_distill"):
            resolved_deps["load_distilled_checkpoint"](
                actor,
                hist_normalizer,
                load_path,
                device=torch_device,
            )
        actor.eval()
        hist_normalizer.eval()
        student_policy = resolved_deps["student_policy"]

        def student_policy_fn(obs: Any) -> Any:
            return student_policy(actor, hist_normalizer, obs, device=torch_device)

        policy = student_policy_fn

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


__all__ = [
    "create_hora_appo_playback_session",
    "create_hora_distill_playback_session",
]
