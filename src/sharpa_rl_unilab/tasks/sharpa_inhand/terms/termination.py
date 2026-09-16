"""Sharpa object-drop termination term."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from unilab.managers import ManagerTermBase

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.observation import SharpaRotationObservation
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_name,
    resolve_env_ids,
    validate_term_params,
)

if TYPE_CHECKING:
    from unilab.managers import ManagerTermBaseCfg

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


class SharpaDropTermination(ManagerTermBase):
    """Terminate when the object leaves its sampled reset-height band."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: SharpaEnv):
        super().__init__(env)
        term = type(self).__name__
        validate_term_params(term, cfg, {"observation_group", "observation_term"})
        group = require_name(term, "observation_group", cfg.params.get("observation_group"))
        name = require_name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, SharpaRotationObservation):
            raise TypeError(f"{term} requires SharpaRotationObservation")
        self.observation = observation
        self.dropped = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.dropped[resolve_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("SharpaEnv", self._env).common_step_counter)

    def __call__(self, env: SharpaEnv, **params: Any) -> np.ndarray:
        del params
        state = self.observation.snapshot(env)
        self.dropped[:] = (
            state.object_pos[:, 2] > self.observation.task_state.reset_height_upper
        ) | (state.object_pos[:, 2] < self.observation.task_state.reset_height_lower)
        self._last_counter = int(env.common_step_counter)
        return self.dropped
