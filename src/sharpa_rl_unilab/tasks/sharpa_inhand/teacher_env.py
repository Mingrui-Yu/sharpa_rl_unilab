"""Sharpa observation/autoreset adapter over the unchanged UniLab environment."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from omegaconf import DictConfig
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.base.np_env import NpEnvState
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnv

from sharpa_rl_unilab.assets import ASSETS_ROOT_PATH

OBS_SHAPES = {"obs": (147,), "critic": (174,), "priv_info": (9,), "proprio_hist": (30, 49)}
CONTRACT_VERSION = "sharpa-hora-v2"


class SharpaTeacherEnv:
    """Capture all observation groups before resetting any completed rows.

    UniLab owns physics, managers and histories. This task adapter owns only the
    additional policy inputs and their autoreset boundary; it does not put them
    in info or alter the installed dependency.
    """

    def __init__(
        self, cfg: DictConfig, num_envs: int, *, scale: float | None = None, auto_reset: bool = True
    ):
        overrides = BackendAdapter(
            cfg, root_dir=".", algo_name=str(cfg.algo.algo)
        ).build_task_env_cfg_override()
        overrides["auto_reset"] = False
        if scale is not None:
            name = f"scene_scale_{scale:g}"
            overrides["fixed_model_variants"] = FixedModelVariantCatalogCfg(
                variants=(
                    FixedModelVariantCfg(
                        name=name,
                        source_model_file=str(
                            ASSETS_ROOT_PATH / "robots" / "sharpa_wave" / f"{name}.xml"
                        ),
                    ),
                )
            )
        self.env = cast(
            ManagerBasedRlEnv, create_env(cfg, num_envs=num_envs, env_cfg_override=overrides)
        )
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.signed_angle_delta = np.zeros(num_envs)
        self.state: NpEnvState | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {key: int(np.prod(shape)) for key, shape in OBS_SHAPES.items()}

    def _observations(self) -> dict[str, np.ndarray]:
        groups = self.env.observation_manager.compute()
        obs = {
            "obs": groups["actor"],
            "critic": groups["critic"],
            "priv_info": groups["priv_info"],
            "proprio_hist": groups["proprio_hist"],
        }
        for key, shape in OBS_SHAPES.items():
            value = obs[key]
            if not isinstance(value, np.ndarray) or value.shape != (self.num_envs, *shape):
                raise ValueError(
                    f"{CONTRACT_VERSION}: {key} must have shape {(self.num_envs, *shape)}"
                )
        return {key: cast(np.ndarray, value).copy() for key, value in obs.items()}

    def reset(self, env_indices: np.ndarray | None = None, *, seed: int | None = None):
        _, info = self.env.reset(env_indices, seed=seed)
        obs = self._observations()
        if self.state is None or env_indices is None:
            self.state = NpEnvState(
                obs,
                np.zeros(self.num_envs),
                np.zeros(self.num_envs, bool),
                np.zeros(self.num_envs, bool),
                info,
            )
        else:
            for key, value in obs.items():
                self.state.obs[key][env_indices] = value[env_indices]
            self.state.terminated[env_indices] = False
            self.state.truncated[env_indices] = False
        return (obs if env_indices is None else {k: v[env_indices] for k, v in obs.items()}), info

    def init_state(self) -> NpEnvState:
        self.reset()
        assert self.state is not None
        return self.state

    def step(self, actions: np.ndarray) -> NpEnvState:
        state = self.env.step(actions)
        obs = self._observations()
        source = self.env.observation_manager.get_term_cfg("actor", "frame").func
        self.signed_angle_delta = (
            np.sum(source.object_angvel * source.task_state.rotation_axis, axis=-1)
            * self.env.step_dt
        )
        reward = state.reward.copy()
        terminated = state.terminated.copy()
        # Real termination wins if the time limit fires on the same step.
        truncated = state.truncated.copy() & ~terminated
        done = terminated | truncated
        info = dict(state.info)
        final = None
        if np.any(done):
            final = {key: value.copy() for key, value in obs.items()}
        if np.any(done) and self.auto_reset:
            ids = np.flatnonzero(done).astype(np.int32)
            self.env.reset(ids)
            reset_obs = self._observations()
            for key in obs:
                obs[key][ids] = reset_obs[key][ids]
        self.state = NpEnvState(obs, reward, terminated, truncated, info, final)
        return self.state

    def close(self) -> None:
        self.env.close()


def transition_next(state: NpEnvState) -> dict[str, np.ndarray]:
    """Return synchronized old-episode next inputs for rollout/replay targets."""
    result = {key: value.copy() for key, value in state.obs.items()}
    done = state.terminated | state.truncated
    if np.any(done):
        if state.final_observation is None:
            raise ValueError("A completed transition requires its final observation")
        for key in result:
            result[key][done] = state.final_observation[key][done]
    return result
