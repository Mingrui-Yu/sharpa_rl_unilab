"""Physical, tactile and position-controller state used by the world model."""

from __future__ import annotations

import torch

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTION_DIM, FRAME_DIM

from .geometry import mat_to_quat, matmul, quat_to_mat, so3_exp, so3_log

TACTILE_DIM = 2 * (FRAME_DIM - 2 * ACTION_DIM)
PHYSICAL_DIM = 13 + 2 * ACTION_DIM
STATE_DIM = PHYSICAL_DIM + TACTILE_DIM + ACTION_DIM
DELTA_DIM = 12 + 2 * ACTION_DIM + TACTILE_DIM
State = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def unflatten_state(value: torch.Tensor) -> State:
    tactile_end = PHYSICAL_DIM + TACTILE_DIM
    return (
        value[..., 0:3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13 : 13 + ACTION_DIM],
        value[..., 13 + ACTION_DIM : PHYSICAL_DIM],
        value[..., PHYSICAL_DIM:tactile_end],
        value[..., tactile_end:],
    )


def flatten_state(state: State) -> torch.Tensor:
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel, tactile, target = state
    return torch.cat(
        (position, mat_to_quat(rotation), lin_vel, ang_vel, joint_pos, joint_vel, tactile, target),
        -1,
    )


def state_features(state: State) -> torch.Tensor:
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel, tactile, target = state
    return torch.cat(
        (
            position,
            rotation[..., :, :2].flatten(-2),
            lin_vel,
            ang_vel,
            joint_pos,
            joint_vel,
            tactile,
            target,
        ),
        -1,
    )


def state_difference(state: State, next_state: State) -> torch.Tensor:
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel, tactile, _ = state
    n_position, n_rotation, n_lin_vel, n_ang_vel, n_joint_pos, n_joint_vel, n_tactile, _ = (
        next_state
    )
    return torch.cat(
        (
            n_position - position,
            so3_log(matmul(rotation.transpose(-1, -2), n_rotation)),
            n_lin_vel - lin_vel,
            n_ang_vel - ang_vel,
            n_joint_pos - joint_pos,
            n_joint_vel - joint_vel,
            n_tactile - tactile,
        ),
        -1,
    )


def integrate_state(state: State, delta: torch.Tensor, action: torch.Tensor, lower, upper, scale):
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel, tactile, target = state
    splits = (3, 3, 3, 3, ACTION_DIM, ACTION_DIM, TACTILE_DIM)
    d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel, d_tactile = delta.split(
        splits, -1
    )
    next_target = (target + scale * action.clamp(-1, 1)).clamp(lower, upper)
    return (
        position + d_position,
        matmul(rotation, so3_exp(d_rotation)),
        lin_vel + d_lin_vel,
        ang_vel + d_ang_vel,
        joint_pos + d_joint_pos,
        joint_vel + d_joint_vel,
        (tactile + d_tactile).clamp(0, 4),
        next_target,
    )


__all__ = [
    "DELTA_DIM",
    "PHYSICAL_DIM",
    "STATE_DIM",
    "State",
    "TACTILE_DIM",
    "flatten_state",
    "integrate_state",
    "state_difference",
    "state_features",
    "unflatten_state",
]
