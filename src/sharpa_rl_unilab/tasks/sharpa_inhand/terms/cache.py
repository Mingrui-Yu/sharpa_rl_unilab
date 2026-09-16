"""Grasp-cache path resolution and fixed-variant sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sharpa_rl_unilab.assets import generated_root, resolve_asset


def grasp_cache_output_file(prefix: str, scale: float) -> Path:
    """Choose a writable user-data path, never a bundled asset fallback."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"object scale must be finite and positive, got {scale}")
    prefix_path = Path(prefix)
    if prefix_path.suffix == ".npy":
        path = prefix_path.with_name(f"{prefix_path.stem}_{scale:g}.npy")
    else:
        path = Path(f"{prefix}_{scale:g}.npy")
    return path if path.is_absolute() else generated_root() / path


def resolve_grasp_cache_file(prefix: str, scale: float) -> Path:
    """Resolve an existing user cache, managed copy or bundled cache."""
    output = grasp_cache_output_file(prefix, scale)
    if output.is_file():
        return output
    return resolve_asset(Path(prefix).parent / output.name)


def validate_grasp_caches(caches: tuple[np.ndarray, ...]) -> None:
    """Scan immutable cache contents once, when loading them."""
    if not caches:
        raise ValueError("at least one grasp cache is required")
    for index, cache in enumerate(caches):
        if cache.ndim != 2 or cache.shape[1] != 29 or cache.shape[0] == 0:
            raise ValueError(f"grasp cache {index} must have shape (N, 29), got {cache.shape}")
        if not np.isfinite(cache).all():
            raise ValueError(f"grasp cache {index} contains NaN or Inf")


def sample_scale_grasp_caches(
    caches: tuple[np.ndarray, ...],
    variant_ids: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    selected_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Sample prevalidated caches without rescanning their contents on each reset."""
    variant_ids = np.asarray(variant_ids)
    if (
        variant_ids.ndim != 1
        or not np.issubdtype(variant_ids.dtype, np.integer)
        or np.any(variant_ids < 0)
        or np.any(variant_ids >= len(caches))
    ):
        raise ValueError("variant_ids must contain valid integer cache indices")
    sampled = np.empty((len(variant_ids), 29), dtype=np.float64)
    for variant_id, cache in enumerate(caches):
        ids = np.flatnonzero(variant_ids == variant_id)
        if ids.size == 0:
            continue
        choices = (
            rng.integers(0, cache.shape[0], size=ids.size)
            if rng is not None
            else np.random.randint(0, cache.shape[0], size=ids.size)
        )
        if selected_rows is not None:
            selected_rows[ids] = choices
        sampled[ids] = cache[choices]
    return sampled
