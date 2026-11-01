"""Batched differentiable rotations; quaternions use wxyz."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(
                (qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1
            ),
            torch.stack(
                ((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1
            ),
            torch.stack(
                ((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1
            ),
            torch.stack(
                ((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1
            ),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(
        candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))
    ).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def matmul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.unsqueeze(-1) * second.unsqueeze(-3)).sum(-2)


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    theta2 = vector.square().sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    matrix = skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(matrix.shape)
    return eye + a[..., None] * matrix + b[..., None] * matmul(matrix, matrix)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    vector = q[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(q[..., :1], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return vector * scale


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    return quat_to_rotvec(mat_to_quat(rotation))


__all__ = ["matmul", "quat_to_mat", "so3_exp", "so3_log"]
