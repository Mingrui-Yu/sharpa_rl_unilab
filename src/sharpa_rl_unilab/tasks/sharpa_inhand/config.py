"""Registered Manager-Based configuration owners for the Sharpa task family."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from unilab.base.base import FixedModelVariantCatalogCfg
from unilab.base.variants import FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg

from sharpa_rl_unilab.assets import ASSETS_ROOT_PATH

_ROBOT_ROOT = ASSETS_ROOT_PATH / "robots" / "sharpa_wave"
_SCALE_NAMES = ("0.8", "0.9", "1", "1.1", "1.2", "1.3", "1.4", "1.5")

def _variants() -> FixedModelVariantCatalogCfg:
    return FixedModelVariantCatalogCfg(
        variants=tuple(
            FixedModelVariantCfg(
                name=f"scene_scale_{name}",
                source_model_file=str(_ROBOT_ROOT / f"scene_scale_{name}.xml"),
            )
            for name in _SCALE_NAMES
        )
    )


@dataclass
class SharpaInhandRotationCfg(ManagerBasedRlEnvCfg):
    """Common typed owner for Hydra Manager-Based term declarations."""

    fixed_model_variants: FixedModelVariantCatalogCfg | None = field(
        default_factory=_variants
    )
    sim_dt: float = 1.0 / 240.0
    ctrl_dt: float = 12.0 / 240.0
    max_episode_seconds: float = 20.0  # pyright: ignore[reportIncompatibleVariableOverride]


SharpaWaveRotationCfg = SharpaInhandRotationCfg

__all__ = ["SharpaInhandRotationCfg", "SharpaWaveRotationCfg"]


def _grasp_variants() -> FixedModelVariantCatalogCfg:
    """Resolve the one-scale grasp-generation variant from the helper environment."""
    scale = os.environ.get("SHARPA_GRASP_SCALE", "0.8")
    source = os.environ.get(
        "SHARPA_GRASP_VARIANT_FILE",
        str(_ROBOT_ROOT / f"scene_scale_{scale}.xml"),
    )
    return FixedModelVariantCatalogCfg(
        variants=(FixedModelVariantCfg(name=f"scene_scale_{scale}", source_model_file=source),)
    )


@dataclass
class SharpaInhandRotationGraspCfg(SharpaInhandRotationCfg):
    """Single-scale owner used to regenerate one grasp cache at a time."""

    fixed_model_variants: FixedModelVariantCatalogCfg | None = field(
        default_factory=_grasp_variants
    )


SharpaInhandGraspEnvCfg = SharpaInhandRotationGraspCfg

__all__.extend(["SharpaInhandGraspEnvCfg", "SharpaInhandRotationGraspCfg"])
