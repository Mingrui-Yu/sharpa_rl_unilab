"""Reward functions over the observation term's post-physics state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from unilab.dtype_config import get_global_dtype

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import SOURCE_DEFAULT_HAND_JOINT_POS
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.observation import SharpaRotationObservation
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.termination import SharpaDropTermination
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.validation import require_name

if TYPE_CHECKING:
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.types import SharpaEnv


def _rotation_state(
    env: SharpaEnv, state_term_name: str
) -> tuple[SharpaDropTermination, SharpaRotationObservation]:
    state = env.termination_manager.get_term_cfg(
        require_name("Sharpa reward", "state_term_name", state_term_name)
    ).func
    if not isinstance(state, SharpaDropTermination):
        raise TypeError("Sharpa reward state term must be SharpaDropTermination")
    if state.last_counter != int(env.common_step_counter):
        raise RuntimeError("Sharpa reward state was not computed for the current control step")
    return state, state.observation


def rotate_reward(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.clip(
            np.sum(state.object_angvel * state.task_state.rotation_axis, axis=1),
            -0.5,
            0.5,
        ),
        dtype=get_global_dtype(),
    )


def object_linear_velocity_l1(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.abs(state.object_linvel), axis=1), dtype=get_global_dtype())


def hand_pose_deviation_l2(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    default = np.broadcast_to(
        SOURCE_DEFAULT_HAND_JOINT_POS.astype(get_global_dtype()),
        state.dof_pos.shape,
    )
    return np.asarray(np.sum(np.square(state.dof_pos - default), axis=1), dtype=get_global_dtype())


def estimated_torque_l2(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.square(state.torques), axis=1), dtype=get_global_dtype())


def estimated_work_l2(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.square(np.sum(state.torques * state.dof_vel, axis=1)), dtype=get_global_dtype()
    )


def object_position_reward(env: SharpaEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    delta = state.object_pos - state.task_state.object_pos_anchor
    return np.asarray(1.0 / (np.linalg.norm(delta, axis=1) + 0.001), dtype=get_global_dtype())

