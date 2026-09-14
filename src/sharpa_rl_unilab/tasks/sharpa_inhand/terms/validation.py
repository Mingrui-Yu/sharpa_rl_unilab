"""Fail-closed parameter validation for Sharpa Manager-Based terms."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from unilab.managers import ManagerTermBaseCfg
    from unilab.managers._types import ManagerBasedRlEnv


def require_name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def require_names(term: str, field: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {field} must be a sequence of names")
    result = tuple(require_name(term, field, item) for item in value)
    if not result:
        raise ValueError(f"{term} {field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{term} {field} contains duplicate names")
    return result


def require_real(
    term: str,
    field: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{term} {field} must be at least {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {field} must be at most {maximum}, got {result}")
    return result


def require_int(term: str, field: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{term} {field} must be positive, got {result}")
    return result


def require_bool(term: str, field: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{term} {field} must be boolean")
    return bool(value)


def require_pair(term: str, field: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {field} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {field} must contain two values")
    lower = require_real(term, f"{field}[0]", value[0])
    upper = require_real(term, f"{field}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {field} lower bound exceeds upper bound")
    return lower, upper


def resolve_env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return np.asarray(env_ids, dtype=np.int32)


def validate_term_params(term: str, cfg: ManagerTermBaseCfg, allowed: set[str]) -> None:
    unexpected = set(cfg.params) - allowed
    if unexpected:
        raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
