from __future__ import annotations

from pathlib import Path

import numpy as np

from sharpa_rl_unilab.tasks.sharpa_inhand.config import (
    SharpaInhandRotationCfg,
    SharpaInhandRotationGraspCfg,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.manager_terms import (
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
)


def test_fixed_variant_catalog_and_xml_sizes_are_consistent() -> None:
    cfg = SharpaInhandRotationCfg()
    expected = ("0.8", "0.9", "1", "1.1", "1.2", "1.3", "1.4", "1.5")
    names = tuple(
        variant.name.removeprefix("scene_scale_")
        for variant in cfg.fixed_model_variants.variants
    )
    assert names == expected

    for variant in cfg.fixed_model_variants.variants:
        source = Path(variant.source_model_file)
        assert source.is_file()
        scale = float(variant.name.removeprefix("scene_scale_"))
        text = source.read_text()
        assert f'size="{0.02 * scale:.12g} {0.016 * scale:.12g}"' in text
        assert 'simple="false"' in text
        assert 'include file="right_sharpa_wave_manager.xml"' in text

    grasp_cfg = SharpaInhandRotationGraspCfg()
    assert len(grasp_cfg.fixed_model_variants.variants) == 1
    assert grasp_cfg.fixed_model_variants.variants[0].name == "scene_scale_0.8"


def test_grasp_cache_resolution_and_sampling_use_variant_buckets() -> None:
    path = resolve_grasp_cache_file("caches/sharpa_grasp_linspace", 1.2)
    assert path.name == "sharpa_grasp_linspace_1.2.npy"
    assert path.is_file()

    caches = (
        np.arange(29, dtype=np.float64).reshape(1, 29),
        np.arange(100, 129, dtype=np.float64).reshape(1, 29),
    )
    sampled = sample_scale_grasp_caches(caches, np.asarray((1, 0, 1), dtype=np.int32))
    assert sampled.shape == (3, 29)
    np.testing.assert_allclose(sampled[1], caches[0][0])
    np.testing.assert_allclose(sampled[[0, 2]], np.broadcast_to(caches[1], (2, 29)))


def test_task_source_no_longer_uses_legacy_direct_runtime() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "sharpa_rl_unilab"
    forbidden = ("adapt_legacy_factory", "unilab.dr", "DomainRandomizationProvider", "ResetPlan")
    for path in (root / "tasks").rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{token} found in {path}"
