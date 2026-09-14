"""Grasp-cache path resolution and fixed-variant sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sharpa_rl_unilab.assets import cache_root, resolve_asset


def _scale_tag(value: float) -> str:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"object scale must be finite and positive, got {value}")
    return f"{value:g}"


def resolve_grasp_cache_file(prefix: str, scale: float) -> Path:
    """Resolve the package/writable-cache path for one fixed object scale."""
    prefix_path = Path(prefix)
    if prefix_path.suffix == ".npy":
        path = prefix_path.with_name(f"{prefix_path.stem}_{_scale_tag(scale)}.npy")
    else:
        path = Path(f"{prefix}_{_scale_tag(scale)}.npy")
    if path.is_absolute():
        return path
    try:
        return Path(resolve_asset(str(path)))
    except FileNotFoundError:
        return cache_root() / path


def sample_scale_grasp_caches(
    caches: tuple[np.ndarray, ...], variant_ids: np.ndarray
) -> np.ndarray:
    """Sample one cached grasp for each environment from its fixed variant."""
    if not caches:
        raise ValueError("at least one grasp cache is required")
    for index, cache in enumerate(caches):
        cache = np.asarray(cache)
        if cache.ndim != 2 or cache.shape[1] != 29 or cache.shape[0] == 0:
            raise ValueError(f"grasp cache {index} must have shape (N, 29), got {cache.shape}")
        if not np.isfinite(cache).all():
            raise ValueError(f"grasp cache {index} contains NaN or Inf")
    sampled = np.zeros((len(variant_ids), 29), dtype=np.float64)
    for variant_id, cache in enumerate(caches):
        ids = np.flatnonzero(variant_ids == variant_id)
        if ids.size == 0:
            continue
        choices = np.random.randint(0, np.asarray(cache).shape[0], size=ids.size)
        sampled[ids] = np.asarray(cache)[choices]
    return sampled
