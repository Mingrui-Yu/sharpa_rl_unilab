"""Per-environment held Gaussian noise, owned by the runner's inference path."""

from __future__ import annotations

import math

import torch


class HeldGaussianNoise:
    def __init__(self, exponent: float = 2.0, max_steps: int = 16):
        if not math.isfinite(exponent):
            raise ValueError("noise_zeta_mu must be finite")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
            raise ValueError("noise_zeta_max must be a positive integer")
        ns = torch.arange(1, max_steps + 1, dtype=torch.float32)
        pmf = ns.pow(-exponent)
        if not torch.isfinite(pmf).all() or not torch.isfinite(pmf.sum()):
            raise ValueError("noise_zeta_mu/max produce nonfinite duration probabilities")
        self.cdf = torch.cumsum(pmf / pmf.sum(), dim=0)
        self.noise = torch.zeros(0)
        self.count = torch.zeros(0, dtype=torch.int32)
        self.target = torch.zeros(0, dtype=torch.int32)

    @torch.no_grad()
    def sample(self, mean: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
        if (
            self.noise.shape != mean.shape
            or self.noise.device != mean.device
            or self.noise.dtype != mean.dtype
        ):
            self.noise = torch.zeros_like(mean)
            self.count = torch.zeros(mean.shape[0], device=mean.device, dtype=torch.int32)
            self.target = torch.zeros_like(self.count)
            self.cdf = self.cdf.to(mean.device)
        done = (
            torch.zeros_like(self.count, dtype=torch.bool)
            if dones is None
            else dones.reshape(-1) > 0.5
        )
        reinit = done | (self.count <= 0) | (self.count >= self.target)
        # Preserve the upstream draw order and full-batch draws for exact migration.
        if torch.any(reinit):
            noise = torch.randn_like(mean)
            draws = torch.rand(mean.shape[0], device=mean.device)
            target = torch.searchsorted(self.cdf, draws).to(torch.int32) + 1
            self.noise = torch.where(reinit.unsqueeze(-1), noise, self.noise)
            self.target = torch.where(reinit, target, self.target)
            self.count = torch.where(reinit, torch.zeros_like(self.count), self.count)
        self.count = self.count + 1
        return self.noise
