"""Stateless diagonal Gaussian policy math in normalized action coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def tanh_log_det(raw: torch.Tensor) -> torch.Tensor:
    """Log absolute Jacobian, summed over joints, including saturated samples."""
    return (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(-1)


@dataclass(frozen=True)
class PolicySample:
    raw: torch.Tensor
    action: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor | None


@dataclass(frozen=True)
class PolicyDistribution:
    mean: torch.Tensor
    std: torch.Tensor
    action_mapping: str

    def __post_init__(self):
        if self.action_mapping not in {"clip", "tanh"}:
            raise ValueError("model.action_mapping must be clip or tanh")
        # AMP is useful for the MLP, but density/variance math needs FP32.
        # Preserve FP64 for numerical reference checks.
        dtype = torch.float64 if self.mean.dtype == torch.float64 else torch.float32
        object.__setattr__(self, "mean", self.mean.to(dtype))
        object.__setattr__(self, "std", self.std.to(dtype).expand_as(self.mean))

    def map_action(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.tanh() if self.action_mapping == "tanh" else raw.clamp(-1, 1)

    def deterministic(self) -> torch.Tensor:
        return self.map_action(self.mean)

    def sample(self, noise: torch.Tensor | None = None, *, log_prob=True) -> PolicySample:
        if noise is None:
            noise = torch.randn_like(self.mean)
        elif noise.shape != self.mean.shape or noise.device != self.mean.device:
            raise ValueError("Policy noise must match the mean shape and device")
        raw = self.mean + self.std * noise.to(self.mean.dtype)
        return PolicySample(
            raw,
            self.map_action(raw),
            self.mean,
            self.std,
            self.log_prob(raw) if log_prob else None,
        )

    def log_prob(self, raw: torch.Tensor) -> torch.Tensor:
        # Do not detach here: SAC needs the path through the sampled action.
        # PPO/APPO supply detached rollout samples instead.
        raw = raw.to(self.mean.dtype)
        normalized = (raw - self.mean) / self.std
        result = (-0.5 * normalized.square() - self.std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        return result - tanh_log_det(raw) if self.action_mapping == "tanh" else result

    def entropy(self, noise: torch.Tensor | None = None) -> torch.Tensor:
        entropy = (self.std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        if self.action_mapping == "tanh":
            entropy = entropy + tanh_log_det(self.sample(noise, log_prob=False).raw)
        return entropy

    def kl_from(self, old_mean: torch.Tensor, old_std: torch.Tensor) -> torch.Tensor:
        """KL(old || self); latent Gaussian KL for clip, exact action KL for tanh."""
        old_mean, old_std = old_mean.to(self.mean.dtype), old_std.to(self.std.dtype)
        return (
            self.std.log()
            - old_std.log()
            + 0.5
            * ((old_std / self.std).square() + ((old_mean - self.mean) / self.std).square() - 1)
        ).sum(-1)


class LogStd(nn.Module):
    """Direct log-std parameters; an optional clamp is separate numerical policy."""

    def __init__(self, feature_dim: int, action_dim: int, model: dict):
        super().__init__()
        initial = float(model.get("initial_std", 1.0))
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError("model.initial_std must be finite and positive")
        initial_fp32 = torch.tensor(math.log(initial), dtype=torch.float32).exp()
        if not torch.isfinite(initial_fp32) or initial_fp32 <= 0:
            raise ValueError("model.initial_std must be representable after FP32 exp(log_std)")
        bounds = model.get("log_std_bounds")
        self.bounds = None
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError("model.log_std_bounds must contain two finite increasing values")
            lower, upper = map(float, bounds)
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError("model.log_std_bounds must contain two finite increasing values")
            endpoints = torch.tensor([lower, upper], dtype=torch.float32).exp()
            if not torch.isfinite(endpoints).all() or (endpoints <= 0).any():
                raise ValueError("model.log_std_bounds must yield finite positive FP32 std")
            if not lower <= math.log(initial) <= upper:
                raise ValueError("model.initial_std is outside model.log_std_bounds")
            self.bounds = (lower, upper)
        mode = model.get("std_mode", "state_independent")
        if mode == "state_independent":
            self.log_std = nn.Parameter(torch.full((action_dim,), math.log(initial)))
            self.head = None
        elif mode == "state_dependent":
            self.register_parameter("log_std", None)
            self.head = nn.Linear(feature_dim, action_dim)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, math.log(initial))
        else:
            raise ValueError("model.std_mode must be state_independent or state_dependent")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        log_std = self.head(features) if self.head is not None else self.log_std
        assert log_std is not None
        log_std = log_std.float()
        if self.bounds is not None:
            log_std = log_std.clamp(*self.bounds)
        return log_std.exp()
