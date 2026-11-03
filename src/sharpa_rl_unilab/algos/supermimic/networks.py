"""Normalized multi-step dynamics model adapted from Uni-SuperMimic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from sharpa_rl_unilab.algos.hora.distribution import LogStd, PolicyDistribution
from sharpa_rl_unilab.algos.hora.models import _MLP
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTION_DIM, ACTOR_DIM

from .state import (
    DELTA_DIM,
    DRIFT_W,
    State,
    integrate_state,
    state_difference,
    state_features,
    structured_integrate_state,
    structured_state_difference,
)


class SuperMimicActor(nn.Module):
    def __init__(self, model: dict):
        super().__init__()
        if (
            model["actor_hidden_dims"] != [512, 256, 128]
            or model["activation"] != "elu"
            or not model["actor_normalization"]
        ):
            raise ValueError(
                "Protocol v2 fixes the SuperMimic actor architecture and normalization"
            )
        self.obs_normalizer = EmpiricalNormalization(ACTOR_DIM)
        self.trunk = _MLP(ACTOR_DIM, model["actor_hidden_dims"], model["activation"])
        self.mu_head = nn.Linear(self.trunk.output_dim, ACTION_DIM)
        self.action_mapping = str(model.get("action_mapping", "tanh"))
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        self.std_module = LogStd(self.trunk.output_dim, ACTION_DIM, model)

    def policy(self, obs: TensorDict | torch.Tensor) -> PolicyDistribution:
        if isinstance(obs, torch.Tensor):
            obs = TensorDict({"actor": obs}, batch_size=obs.shape[:-1])
        actor = obs["policy"] if "policy" in obs else obs["actor"]
        features = self.trunk(self.obs_normalizer(actor))
        mean = self.mu_head(features)
        return PolicyDistribution(mean, self.std_module(features), self.action_mapping)


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for width in hidden:
        layers.extend((nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU()))
        dim = width
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.reshape(-1, value.shape[-1]).float()
        if self.count >= self.max_count:
            return
        count = value.shape[0]
        mean = value.mean(0)
        var = value.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean
        next_mean = self.mean + delta * count / total
        next_var = (
            self.var * self.count + var * count + delta.square() * self.count * count / total
        ) / total
        self.mean.copy_(next_mean)
        self.var.copy_(next_var)
        self.count.copy_(total)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std()

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


class WorldModel(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden: Sequence[int],
        amp: bool = True,
        structured: bool = False,
        dt: float = 0.02,
        drift_w: float = DRIFT_W,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.amp = amp
        # ``structured`` = semi-implicit integration (positions follow the predicted velocities,
        # the net predicts velocity increments plus small position corrections)
        self.structured = structured
        self.dt = dt
        self.drift_w = drift_w
        input_dim = state_features(_zero_state(1)).shape[-1] + context_dim + 22
        self.in_norm = RunningNorm(input_dim)
        self.out_norm = RunningNorm(DELTA_DIM, eps=1e-4)
        # scale of the plain frame-local increments (state_difference), used to normalize the
        # training loss; equal to ``out_norm`` unless ``structured`` changes what the net outputs
        self.inc_norm = RunningNorm(DELTA_DIM, eps=1e-4)
        self.net = mlp(input_dim, hidden, DELTA_DIM)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        floor = [5e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3
        floor += [3e-3] * 22 + [5e-2] * 22 + [5e-3] * 10
        self.error_floor: torch.Tensor
        self.register_buffer("error_floor", torch.tensor(floor))

    def inputs(self, state: State, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat((state_features(state), action, context), -1)

    def forward(
        self, state: State, action: torch.Tensor, context: torch.Tensor, lower, upper, scale
    ):
        features = self.in_norm(self.inputs(state, action, context))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and features.is_cuda):
            prediction = self.net(features).float()
        delta = prediction * self.out_norm.std() + self.out_norm.mean
        if self.structured:
            return structured_integrate_state(
                state, delta, action, lower, upper, scale, self.dt, self.drift_w
            )
        return integrate_state(state, delta, action, lower, upper, scale)

    def targets(self, state: State, next_state: State) -> torch.Tensor:
        if self.structured:
            return structured_state_difference(state, next_state, self.dt, self.drift_w)
        return state_difference(state, next_state)

    @staticmethod
    def increments(state: State, next_state: State) -> torch.Tensor:
        """Full frame-local state increments (unstructured targets; the loss-normalization scale)."""
        return state_difference(state, next_state)


def _zero_state(batch: int) -> State:
    rotation = torch.eye(3).expand(batch, 3, 3)
    return (
        torch.zeros(batch, 3),
        rotation,
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, 22),
        torch.zeros(batch, 22),
        torch.zeros(batch, 10),
        torch.zeros(batch, 22),
    )


__all__ = ["RunningNorm", "WorldModel", "mlp"]
