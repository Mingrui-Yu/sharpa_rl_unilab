"""Integrated or direct hand position action term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.compat import entity_host_data, numpy_dtype
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
    from unilab.managers.types import ManagerBasedRlEnv


CONTROL_MODE_INTEGRATED: Literal["integrated", "direct"] = "integrated"
CONTROL_MODE_DIRECT: Literal["integrated", "direct"] = "direct"
_LEGACY_CONTROL_MODES = {
    "incremental": CONTROL_MODE_INTEGRATED,
    "default_pose": CONTROL_MODE_DIRECT,
}


def normalize_control_mode(term: str, value: object) -> Literal["integrated", "direct"]:
    """Normalize canonical target modes and deprecated pre-release aliases."""
    if value == CONTROL_MODE_INTEGRATED or value == CONTROL_MODE_DIRECT:
        return cast(Literal["integrated", "direct"], value)
    if isinstance(value, str) and value in _LEGACY_CONTROL_MODES:
        return cast(Literal["integrated", "direct"], _LEGACY_CONTROL_MODES[value])
    raise ValueError(f"{term} control_mode must be 'integrated' or 'direct'; got {value!r}")


@dataclass(kw_only=True)
class SharpaIncrementalPositionActionCfg(ActionTermCfg):
    actuator_names: tuple[str, ...] | list[str]
    action_scale: float
    raw_action_clip: tuple[float, float] | list[float]
    dof_limits_scale: float = 0.9
    zero_action: bool = False
    control_mode: Literal["integrated", "direct"] = CONTROL_MODE_INTEGRATED

    def build(self, env: ManagerBasedRlEnv) -> SharpaIncrementalPositionAction:
        return SharpaIncrementalPositionAction(self, env)


class SharpaIncrementalPositionAction(ActionTerm):
    """Produce integrated or direct default-pose hand position targets."""

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
        ranges = np.asarray(
            entity_host_data(self._entity, "actuator_ctrl_range"),
            dtype=numpy_dtype(get_global_dtype()),
        )
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
        self._control_mode = normalize_control_mode(term, cfg.control_mode)
        cfg.control_mode = self._control_mode

        dtype = numpy_dtype(get_global_dtype())
        self._raw_action = torch.zeros(
            (env.num_envs, NUM_HAND_JOINTS), dtype=torch.float32, device=env.device
        )
        self._clipped_action = torch.zeros_like(self._raw_action)
        self._target = torch.from_numpy(
            np.broadcast_to(
                SOURCE_DEFAULT_HAND_JOINT_POS.astype(dtype),
                (env.num_envs, NUM_HAND_JOINTS),
            ).copy()
        ).to(device=env.device, dtype=torch.float32)
        self._target_lower = torch.from_numpy(self._ctrl_lower * self._dof_limits_scale).to(
            device=env.device, dtype=torch.float32
        )
        self._target_upper = torch.from_numpy(self._ctrl_upper * self._dof_limits_scale).to(
            device=env.device, dtype=torch.float32
        )
        self._default_pose = torch.from_numpy(
            np.broadcast_to(
                SOURCE_DEFAULT_HAND_JOINT_POS.astype(dtype),
                (env.num_envs, NUM_HAND_JOINTS),
            ).copy()
        ).to(device=env.device, dtype=torch.float32)

    @property
    def action_dim(self) -> int:
        return NUM_HAND_JOINTS

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def target(self) -> torch.Tensor:
        return self._target

    @property
    def control_mode(self) -> str:
        return self._control_mode

    @property
    def default_pose(self) -> torch.Tensor:
        return self._default_pose

    @property
    def ctrl_lower(self) -> np.ndarray:
        return self._ctrl_lower

    @property
    def ctrl_upper(self) -> np.ndarray:
        return self._ctrl_upper

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids

    def process_actions(self, actions: torch.Tensor) -> None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"expected torch.Tensor actions, got {type(actions).__name__}")
        if actions.dtype != torch.float32 or actions.device != self._raw_action.device:
            raise TypeError(
                f"expected contiguous float32 actions on {self._raw_action.device}, "
                f"got {actions.dtype} on {actions.device}"
            )
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        if self._zero_action:
            actions = torch.zeros_like(actions)
        torch.clamp(actions, self._raw_clip[0], self._raw_clip[1], out=self._clipped_action)
        if self._control_mode == CONTROL_MODE_INTEGRATED:
            self._target.add_(self._clipped_action, alpha=self._action_scale)
        else:
            torch.add(
                self._default_pose,
                self._clipped_action,
                alpha=self._action_scale,
                out=self._target,
            )
        torch.clamp(self._target, min=self._target_lower, max=self._target_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = resolve_env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        if self._control_mode == CONTROL_MODE_INTEGRATED:
            self._target[ids] = torch.as_tensor(
                entity_host_data(self._entity, "joint_pos")[ids][:, self._joint_ids],
                dtype=torch.float32,
                device=self._target.device,
            )
        else:
            self._target[ids] = self._default_pose[ids]

    def reset_to_positions(
        self, positions: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Synchronize the action target after a reset term stages joints.

        Integrated mode resumes from the newly staged joint positions. Direct mode always
        re-anchors its next target to the default pose.
        """
        ids = (
            torch.arange(self.num_envs, dtype=torch.int64, device=self._target.device)
            if env_ids is None
            else env_ids.to(device=self._target.device, dtype=torch.int64)
        )
        self._raw_action[ids] = 0.0
        self._clipped_action[ids] = 0.0
        if self._control_mode == CONTROL_MODE_INTEGRATED:
            self._target[ids] = positions.to(device=self._target.device, dtype=torch.float32)
        else:
            self._target[ids] = self._default_pose[ids]
