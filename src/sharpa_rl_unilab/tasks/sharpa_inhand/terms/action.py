"""Incremental hand position action term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    ACTUATOR_NAMES,
    HAND_JOINT_NAMES,
    NUM_HAND_JOINTS,
    SOURCE_DEFAULT_HAND_JOINT_POS,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import (
    require_bool,
    require_names,
    require_pair,
    require_real,
    resolve_env_ids,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SharpaIncrementalPositionActionCfg(ActionTermCfg):
    actuator_names: tuple[str, ...] | list[str]
    action_scale: float
    raw_action_clip: tuple[float, float] | list[float]
    dof_limits_scale: float = 0.9
    zero_action: bool = False

    def build(self, env: ManagerBasedRlEnv) -> SharpaIncrementalPositionAction:
        return SharpaIncrementalPositionAction(self, env)


class SharpaIncrementalPositionAction(ActionTerm):
    """Integrate bounded policy deltas into hand position targets."""

    cfg: SharpaIncrementalPositionActionCfg
    _entity: Entity

    def __init__(self, cfg: SharpaIncrementalPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(f"{term} uses raw_action_clip, not actuator-name clip")
        names = require_names(term, "actuator_names", cfg.actuator_names)
        joint_ids, joint_names = self._entity.find_joints_by_actuator_names(list(HAND_JOINT_NAMES))
        actuator_ids, matched_actuators = self._entity.find_actuators(
            list(names), preserve_order=True
        )
        if tuple(joint_names) != HAND_JOINT_NAMES or tuple(matched_actuators) != ACTUATOR_NAMES:
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete or out of order")
        if len(joint_ids) != NUM_HAND_JOINTS:
            raise ValueError(f"{term} requires {NUM_HAND_JOINTS} actuators, got {len(joint_ids)}")

        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[self._actuator_ids, 0], copy=True)
        self._ctrl_upper = np.array(ranges[self._actuator_ids, 1], copy=True)
        if np.any(self._ctrl_lower >= self._ctrl_upper):
            raise ValueError(f"{term} actuator control ranges must have lower < upper")
        self._action_scale = require_real(term, "action_scale", cfg.action_scale, minimum=0.0)
        self._raw_clip = require_pair(term, "raw_action_clip", cfg.raw_action_clip)
        self._dof_limits_scale = require_real(
            term, "dof_limits_scale", cfg.dof_limits_scale, minimum=0.0
        )
        self._zero_action = require_bool(term, "zero_action", cfg.zero_action)
        if self._dof_limits_scale == 0.0:
            raise ValueError(f"{term} dof_limits_scale must be positive")

        dtype = get_global_dtype()
        self._raw_action = np.zeros((env.num_envs, NUM_HAND_JOINTS), dtype=dtype)
        self._clipped_action = np.zeros_like(self._raw_action)
        self._target = np.broadcast_to(
            SOURCE_DEFAULT_HAND_JOINT_POS.astype(dtype), (env.num_envs, NUM_HAND_JOINTS)
        ).copy()
        self._target_lower = self._ctrl_lower * self._dof_limits_scale
        self._target_upper = self._ctrl_upper * self._dof_limits_scale

    @property
    def action_dim(self) -> int:
        return NUM_HAND_JOINTS

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def ctrl_lower(self) -> np.ndarray:
        return self._ctrl_lower

    @property
    def ctrl_upper(self) -> np.ndarray:
        return self._ctrl_upper

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        if self._zero_action:
            actions = np.zeros_like(actions)
        np.clip(actions, self._raw_clip[0], self._raw_clip[1], out=self._clipped_action)
        self._target += self._action_scale * self._clipped_action
        np.clip(self._target, self._target_lower, self._target_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = resolve_env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        self._target[ids] = self._entity.data.joint_pos[ids][:, self._joint_ids]
