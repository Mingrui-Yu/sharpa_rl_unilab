"""Reward functions over the observation term's post-physics tensor state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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


def rotate_reward(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    return (state.object_angvel * state.task_state.rotation_axis).sum(dim=1).clamp(-0.5, 0.5)


def object_linear_velocity_l1(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    return state.object_linvel.abs().sum(dim=1)


def hand_pose_deviation_l2(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    return (state.dof_pos - state._default_dof_pos).square().sum(dim=1)


def estimated_torque_l2(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    return state.torques.square().sum(dim=1)


def estimated_work_l2(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    return (state.torques * state.dof_vel).sum(dim=1).square()


def object_position_reward(env: SharpaEnv, state_term_name: str) -> torch.Tensor:
    _, state = _rotation_state(env, state_term_name)
    delta = state.object_pos - state.task_state.object_pos_anchor
    return 1.0 / (delta.norm(dim=1) + 0.001)
