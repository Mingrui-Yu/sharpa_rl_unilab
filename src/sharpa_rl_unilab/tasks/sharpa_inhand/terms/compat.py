"""Small adapters at UniLab 1.5's Torch Manager boundary.

Sharpa's entity reads, reset writes, and cached task state remain NumPy arrays
because the MuJoCo HOST_BRIDGE public carrier is CPU-resident. UniLab 1.5,
however, requires Manager observations, rewards, terminations, selectors, and
actions to be Torch tensors. These helpers keep that conversion explicit and
fail closed instead of spreading implicit Tensor/ndarray behavior through task
terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


_NUMPY_DTYPE = {
    torch.bool: np.bool_,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def numpy_dtype(dtype: torch.dtype = torch.float32) -> np.dtype[Any]:
    """Map UniLab's global Torch dtype to the task's NumPy host dtype."""
    try:
        return _NUMPY_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported Sharpa host dtype: {dtype}") from exc


def manager_tensor(
    env: ManagerBasedRlEnv,
    value: np.ndarray | tuple[Any, ...] | list[Any],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize a NumPy Manager term result on the environment carrier."""
    array = np.ascontiguousarray(value, dtype=numpy_dtype(dtype))
    return torch.from_numpy(array).to(device=env.device, dtype=dtype)


def numpy_ids(env_ids: torch.Tensor | np.ndarray | slice | None, num_envs: int) -> np.ndarray:
    """Convert a UniLab 1.5 reset selector to the task's NumPy host boundary."""
    if env_ids is None:
        return np.arange(num_envs, dtype=np.intp)
    if isinstance(env_ids, slice):
        return np.arange(num_envs, dtype=np.intp)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.detach().cpu().numpy().astype(np.intp, copy=False)
    return np.asarray(env_ids, dtype=np.intp)


def rng_numpy(
    value: torch.Tensor | np.ndarray, dtype: np.typing.DTypeLike = np.float32
) -> np.ndarray:
    """Return a NumPy or Torch ManagerRng draw as a contiguous NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def rng_random(
    rng: Any, size: int | tuple[int, ...], *, dtype: torch.dtype = torch.float32
) -> np.ndarray:
    """Draw random values from ManagerRng or a NumPy generator."""
    try:
        value = rng.random(size, dtype=dtype)  # type: ignore[union-attr]
    except TypeError:
        value = rng.random(size, dtype=numpy_dtype(dtype))  # type: ignore[union-attr]
    return rng_numpy(value, dtype=numpy_dtype(dtype))


def rng_uniform(
    rng: Any,
    low: float,
    high: float,
    size: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Draw uniformly from ManagerRng or a NumPy generator used by unit fakes."""
    try:
        value = rng.uniform(low, high, size, dtype=dtype)  # type: ignore[union-attr]
    except TypeError:
        value = rng.uniform(low, high, size)  # type: ignore[union-attr]
    return rng_numpy(value, dtype=numpy_dtype(dtype))


def host_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return task-cached NumPy state without an implicit Tensor conversion."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def entity_host_data(entity: object, field: str) -> np.ndarray:
    """Read a NumPy Entity table across UniLab's 1.2 and 1.5 facades."""
    data = getattr(entity, "data")
    value = getattr(data, field, None)
    if value is None:
        # UniLab 1.5 renamed the explicit host adapters with an underscore.
        value = getattr(data, f"_{field}", None)
    if value is None:
        legacy_name = {
            "joint_pos": "_host_joint_pos",
            "root_link_pos_w": "_host_root_link_pos_w",
            "root_link_quat_w": "_host_root_link_quat_w",
        }.get(field)
        if legacy_name is not None:
            value = getattr(data, legacy_name, None)
    if value is None:
        raise AttributeError(f"Entity data has no Sharpa host table {field!r}")
    return host_numpy(cast("torch.Tensor | np.ndarray", value))


def entity_default_root_state(entity: object, device: torch.device) -> np.ndarray:
    """Read the 13-D default root state as a NumPy host table."""
    value = getattr(getattr(entity, "data"), "default_root_state")
    if callable(value):
        value = value(device)
    return host_numpy(cast("torch.Tensor | np.ndarray", value))


def torch_ids(env_ids: np.ndarray | slice | None, device: torch.device) -> torch.Tensor:
    """Return a Sharpa reset selector in UniLab 1.5's Torch form."""
    return torch.as_tensor(np.asarray(env_ids), dtype=torch.int64, device=device)
