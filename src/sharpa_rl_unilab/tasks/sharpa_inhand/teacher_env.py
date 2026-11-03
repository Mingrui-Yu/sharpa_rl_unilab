"""Sharpa TorchEnv adapter for the tensor-native UniLab runtime."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.base.torch_env import TorchEnvState
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnv

from sharpa_rl_unilab.assets import ASSETS_ROOT_PATH

from .protocol import CONTRACT_VERSION, OBS_SHAPES, validate_observation_config

EnvState = TorchEnvState


class SharpaTeacherEnv:
    """Keep UniLab's tensor-native observation and autoreset contracts intact.

    UniLab owns physics, managers, histories, terminal observations and
    autoreset. This adapter publishes only the additional signed-angle
    diagnostic; it never introduces a NumPy training boundary.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        *,
        scale: float | None = None,
        variant_scales: Sequence[float] | None = None,
        auto_reset: bool = True,
        tensor_device: str | None = None,
        defer_finite_validation: bool = False,
    ):
        validate_observation_config(cfg)
        overrides = EnvConfigAdapter(
            cfg, root_dir=".", algo_name=str(cfg.algo.algo)
        ).build_task_env_cfg_override()
        # UniLab publishes only policy/critic in TorchEnvState. Sharpa needs all
        # four HORA groups at the terminal boundary, so it owns a tensor-only
        # selected-row autoreset while keeping NumPy out of the hot path.
        overrides["auto_reset"] = False
        if defer_finite_validation:
            overrides["defer_finite_validation"] = True
            # Ignored spectator rows may eventually diverge after their episode
            # finishes. The evaluator clears those verdicts before close.
            overrides["finite_validation_interval"] = 401
        carrier_device = tensor_device or self._requested_tensor_device(cfg)
        if carrier_device is not None:
            overrides["collector_tensor_device"] = carrier_device
        if scale is not None and variant_scales is not None:
            raise ValueError(
                "Specify either one fixed scale or an explicit per-slot scale assignment"
            )
        if scale is not None:
            variant_scales = (float(scale),) * num_envs
        if variant_scales is not None:
            variant_scales = tuple(float(value) for value in variant_scales)
            if len(variant_scales) != num_envs:
                raise ValueError(
                    "Fixed variant scales must contain exactly one scale per environment"
                )
            if not variant_scales or any(
                not np.isfinite(value) or value <= 0.0 for value in variant_scales
            ):
                raise ValueError("Fixed variant scales must be finite and positive")
            variants = []
            seen: set[float] = set()
            for value in variant_scales:
                if value in seen:
                    continue
                seen.add(value)
                name = f"scene_scale_{value:g}"
                variants.append(
                    FixedModelVariantCfg(
                        name=name,
                        source_model_file=str(
                            ASSETS_ROOT_PATH / "robots" / "sharpa_wave" / f"{name}.xml"
                        ),
                    )
                )
            # Explicit names make the row-to-scale assignment deterministic even
            # when a cross-scale evaluation batch is not aligned by scale.
            overrides["fixed_model_variants"] = FixedModelVariantCatalogCfg(
                variants=tuple(variants),
                explicit_variant_names=tuple(f"scene_scale_{value:g}" for value in variant_scales),
            )
        self.env = cast(
            ManagerBasedRlEnv, create_env(cfg, num_envs=num_envs, env_cfg_override=overrides)
        )
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.signed_angle_delta = torch.zeros(num_envs, device=self.env.device)
        self.state: TorchEnvState | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {key: int(torch.Size(shape).numel()) for key, shape in OBS_SHAPES.items()}

    @staticmethod
    def _requested_tensor_device(cfg: DictConfig) -> str | None:
        requested = cfg.algo.get("collector_device", None) or cfg.training.get("device", None)
        if requested is None:
            return None
        device = torch.device(str(requested))
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if device.type not in {"cpu", "cuda"}:
            return None
        return str(device)

    def _observations(self) -> dict[str, torch.Tensor]:
        """Validate all HORA groups without copying them to the host."""
        groups = self.env.observation_manager.compute()
        obs = {
            "obs": self._group(groups, "actor"),
            "critic": self._group(groups, "critic"),
            "priv_info": self._group(groups, "priv_info"),
            "proprio_hist": self._group(groups, "proprio_hist"),
        }
        for key, shape in OBS_SHAPES.items():
            value = obs[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (self.num_envs, *shape)
                or value.dtype != torch.float32
                or value.device != self.env.device
            ):
                raise ValueError(
                    f"{CONTRACT_VERSION}: {key} must be a float32 Tensor of shape "
                    f"{(self.num_envs, *shape)} on {self.env.device}"
                )
        return {key: value.contiguous() for key, value in obs.items()}

    @staticmethod
    def _group(groups: dict[str, torch.Tensor | dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        value = groups[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Observation group {key!r} did not concatenate into a Tensor")
        return value

    @staticmethod
    def _action_tensor(actions: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"Sharpa TorchEnv actions must be a Tensor; got {type(actions).__name__}"
            )
        if actions.dtype != torch.float32:
            actions = actions.to(dtype=torch.float32)
        if actions.device != device or not actions.is_contiguous():
            actions = actions.to(device=device).contiguous()
        return actions

    def reset(
        self, env_indices: torch.Tensor | Sequence[int] | None = None, *, seed: int | None = None
    ):
        ids = (
            None
            if env_indices is None
            else torch.as_tensor(env_indices, dtype=torch.int64, device=self.env.device)
        )
        _, info = self.env.reset(ids, seed=seed)
        obs = self._observations()
        if ids is None or self.state is None:
            self.state = TorchEnvState(
                obs,
                torch.zeros(self.num_envs, dtype=torch.float32, device=self.env.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device),
                info,
            )
            return dict(obs), info

        assert self.state is not None
        for key, value in obs.items():
            self.state.obs[key][ids] = value[ids]
        self.state.terminated[ids] = False
        self.state.truncated[ids] = False
        return {key: value[ids] for key, value in obs.items()}, info

    def init_state(self) -> TorchEnvState:
        self.reset()
        assert self.state is not None
        return self.state

    def step(self, actions: torch.Tensor) -> TorchEnvState:
        state = self.env.step(self._action_tensor(actions, self.env.device))
        obs = self._observations()
        reward = state.reward.detach().clone()
        terminated = state.terminated.detach().clone()
        # Real termination wins if a time limit fires on the same step.
        truncated = state.truncated.detach() & ~terminated
        done = terminated | truncated
        # The all-group terminal snapshot is tensor-native. Publishing it even
        # for an empty done mask avoids a host synchronization in the hot loop.
        final = {key: value.detach().clone() for key, value in obs.items()}
        info = dict(state.info)
        info["timing"] = dict(info.get("timing", {}))
        self.state = TorchEnvState(obs, reward, terminated, truncated, info, final)
        if self.auto_reset:
            reset_started = time.perf_counter()
            ids = done.nonzero(as_tuple=False).flatten()
            self.env.reset(ids)
            reset_obs = self._observations()
            for key in obs:
                selector = done.reshape((self.num_envs, *([1] * (obs[key].dim() - 1))))
                obs[key] = torch.where(selector, reset_obs[key], obs[key])
            info["timing"]["reset_done_ms"] = (time.perf_counter() - reset_started) * 1000
        self.state = TorchEnvState(obs, reward, terminated, truncated, info, final)
        return self.state


def transition_next(state: TorchEnvState) -> dict[str, torch.Tensor]:
    """Return synchronized old-episode next inputs for rollout/replay targets."""
    result = dict(state.obs)
    done = state.terminated | state.truncated
    final_observation = getattr(state, "final_observation", None)
    if final_observation is not None:
        for key in result:
            selector = done.reshape((done.shape[0], *([1] * (result[key].dim() - 1))))
            result[key] = torch.where(selector, final_observation[key], result[key])
    return result
