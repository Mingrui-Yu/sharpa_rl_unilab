"""Two NumPy geometry helpers retained by the Sharpa task.

UniLab 1.5 moved rotation helpers into ``unilab.base.rotation`` and removed the
old general geometry module. Quaternion-axis conversion is task-local here so
the task does not depend on an incidental utility-package layout.
"""

from __future__ import annotations

import numpy as np
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
