from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


class _MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int] | tuple[int, ...], activation: str
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(_build_activation(activation))
            current_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal adaptation encoder used by HORA stage-2 distillation."""

    def __init__(self, frame_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(frame_dim, frame_dim, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(frame_dim * 3, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                fan_out = module.kernel_size[0] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=(2.0 / fan_out) ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        x = self.channel_transform(proprio_hist)
        x = x.permute(0, 2, 1)
        x = self.temporal_aggregation(x)
        return self.low_dim_proj(x.flatten(1))


@dataclass
class HoraCoreOutput:
    policy_obs: torch.Tensor
    trunk_latent: torch.Tensor
    privileged_latent: torch.Tensor
    privileged_target: torch.Tensor
