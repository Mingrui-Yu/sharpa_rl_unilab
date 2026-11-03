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


# Weight of the *new* velocity in the semi-implicit position drift. For ``n`` semi-implicit
# substeps with a linearly changing velocity, the position advance is a weighted sum of the old
# and new velocity with weight sum_k k/n / n = (n + 1) / (2 n); BlinkMimic uses n = 4 -> 0.625.
DRIFT_W = 0.625


def semi_implicit_drift(
    state: State,
    lin_vel1: torch.Tensor,
    ang_vel1: torch.Tensor,
    joint_vel1: torch.Tensor,
    dt: float,
    drift_w: float = DRIFT_W,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Position/rotation/joint-position advance implied by the old and new velocities.

    The structured world model predicts only the residual correction on top of this drift
    instead of the full increment. Position advances in the world frame, the rotation vector
    in the body frame and joint positions as plain differences, matching :func:`state_difference`.
    """
    _, _, lin_vel, ang_vel, _, joint_vel, _, _ = state
    a = dt * (1.0 - drift_w)
    b = dt * drift_w
    return a * lin_vel + b * lin_vel1, a * ang_vel + b * ang_vel1, a * joint_vel + b * joint_vel1


def structured_state_difference(
    state: State, next_state: State, dt: float, drift_w: float = DRIFT_W
) -> torch.Tensor:
    """Regression targets of the structured world model.

    Velocity increments and the tactile delta stay as in :func:`state_difference`; the position,
    rotation and joint-position components become the corrections on top of the semi-implicit
    drift implied by the velocities of ``next_state``.
    """
    n_lin_vel, n_ang_vel, n_joint_vel = next_state[2], next_state[3], next_state[5]
    d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel, d_tactile = (
        state_difference(state, next_state).split(
            (3, 3, 3, 3, ACTION_DIM, ACTION_DIM, TACTILE_DIM), -1
        )
    )
    gp, gth, gq = semi_implicit_drift(state, n_lin_vel, n_ang_vel, n_joint_vel, dt, drift_w)
    return torch.cat(
        (
            d_position - gp,
            d_rotation - gth,
            d_lin_vel,
            d_ang_vel,
            d_joint_pos - gq,
            d_joint_vel,
            d_tactile,
        ),
        -1,
    )


def structured_integrate_state(
    state: State,
    delta: torch.Tensor,
    action: torch.Tensor,
    lower,
    upper,
    scale,
    dt: float,
    drift_w: float = DRIFT_W,
) -> State:
    """Semi-implicit next state from the denormalized structured output ``delta``.

    Velocities integrate explicitly; positions follow the velocity-implied drift plus the
    network's residual correction. The action target is integrated analytically as before.
    """
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel, tactile, target = state
    d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel, d_tactile = delta.split(
        (3, 3, 3, 3, ACTION_DIM, ACTION_DIM, TACTILE_DIM), -1
    )
    lin_vel1 = lin_vel + d_lin_vel
    ang_vel1 = ang_vel + d_ang_vel
    joint_vel1 = joint_vel + d_joint_vel
    gp, gth, gq = semi_implicit_drift(state, lin_vel1, ang_vel1, joint_vel1, dt, drift_w)
    next_target = (target + scale * action.clamp(-1, 1)).clamp(lower, upper)
    return (
        position + d_position + gp,
        matmul(rotation, so3_exp(d_rotation + gth)),
        lin_vel1,
        ang_vel1,
        joint_pos + d_joint_pos + gq,
        joint_vel1,
        (tactile + d_tactile).clamp(0, 4),
        next_target,
    )


__all__ = [
    "DELTA_DIM",
    "DRIFT_W",
    "PHYSICAL_DIM",
    "STATE_DIM",
    "State",
    "TACTILE_DIM",
    "flatten_state",
    "integrate_state",
    "semi_implicit_drift",
    "state_difference",
    "state_features",
    "structured_integrate_state",
    "structured_state_difference",
    "unflatten_state",
]
