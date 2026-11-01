"""Task-local NumPy reset geometry and Torch runtime geometry."""

from __future__ import annotations

import numpy as np
import torch
from unilab.base.rotation import np_quat_conjugate, np_quat_mul, np_quat_to_axis_angle


def np_normalize_axis(axis: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
    """Return a unit-length copy of a rotation axis vector."""
    array = np.asarray(axis)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"axis must be non-zero, got {axis!r}")
    return array / norm


def np_quat_angular_velocity_from_pair(
    quat: np.ndarray, prev_quat: np.ndarray, dt: float
) -> np.ndarray:
    """Angular velocity from two consecutive w-first quaternions."""
    relative = np_quat_mul(quat, np_quat_conjugate(prev_quat))
    return np_quat_to_axis_angle(relative) / dt


def torch_quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def torch_quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    quat = quat * torch.where(quat[..., :1] < 0.0, -1.0, 1.0)
    xyz = quat[..., 1:]
    w = quat[..., :1]
    norms = xyz.norm(dim=-1, keepdim=True)
    half_angle = torch.atan2(norms, w)
    angle = 2.0 * half_angle
    small = angle.abs() < 1.0e-6
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    scale = torch.where(
        small,
        0.5 - angle.square() / 48.0,
        torch.sin(half_angle) / safe_angle,
    )
    return xyz / scale


def torch_quat_angular_velocity_from_pair(
    quat: torch.Tensor, prev_quat: torch.Tensor, dt: float
) -> torch.Tensor:
    previous_inverse = prev_quat.clone()
    previous_inverse[..., 1:] *= -1.0
    return torch_quat_to_axis_angle(torch_quat_mul(quat, previous_inverse)) / dt


def torch_quat_error_magnitude(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    second_inverse = second.clone()
    second_inverse[..., 1:] *= -1.0
    relative = torch_quat_mul(first, second_inverse)
    return torch_quat_to_axis_angle(relative).norm(dim=-1)
