"""Compatibility names and checkpoint migration for older Actor implementations."""

from __future__ import annotations

import torch
from torch import nn

from .distribution import DirectStd


class LegacyScalarStd(DirectStd):
    """Compatibility name for direct std parameters."""


class LegacyTanhStd(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (-10 + 6 * (1 + self.head(features).float().tanh())).exp()


def migrate_actor_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename old active parameters and remove only known unused FlashSAC state."""
    if "shared.distribution.std_param" not in state:
        return state
    state = dict(state)
    old_std = state.pop("shared.distribution.std_param")
    if "std_head.weight" in state:
        state["std_module.head.weight"] = state.pop("std_head.weight")
        state["std_module.head.bias"] = state.pop("std_head.bias")
        state.pop("zeta_cdf", None)
    else:
        state["std_module.std_param"] = old_std
    return state
