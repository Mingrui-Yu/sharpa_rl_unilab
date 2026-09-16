from __future__ import annotations

import copy
from typing import Any

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import (
    ACTION_DIM,
    ACTOR_DIM,
    FRAME_DIM,
    HISTORY_SHAPE,
    PRIV_DIM,
)

from .distribution import DirectStd, LogStd, PolicyDistribution
from .legacy import LegacyScalarStd, LegacyTanhStd


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


def pack_actor(obs: dict[str, torch.Tensor] | TensorDict) -> torch.Tensor:
    """Declared transport layout: raw base history followed by current privilege."""
    return torch.cat((obs["obs"], obs["priv_info"]), dim=-1)


def split_actor(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if packed.shape[-1] != ACTOR_DIM + PRIV_DIM:
        raise ValueError("HORA transport requires [actor_obs147, priv_info9]")
    return packed[..., :ACTOR_DIM], packed[..., ACTOR_DIM:]


class TeacherCore(nn.Module):
    def __init__(self, model: dict[str, Any], *, student: bool = False):
        super().__init__()
        if (
            model["actor_hidden_dims"] != [512, 256, 128]
            or model["priv_mlp_hidden_dims"] != [256, 128, 9]
            or model["activation"] != "elu"
            or not model["actor_normalization"]
        ):
            raise ValueError(
                "Protocol v2 fixes the common HORA Actor architecture and normalization"
            )
        self.obs_normalizer = EmpiricalNormalization(ACTOR_DIM)
        self.priv_encoder = _MLP(PRIV_DIM, model["priv_mlp_hidden_dims"], model["activation"])
        self.trunk = _MLP(ACTOR_DIM + PRIV_DIM, model["actor_hidden_dims"], model["activation"])
        self.mu_head = nn.Linear(self.trunk.output_dim, ACTION_DIM)
        self.adapt_tconv = ProprioAdaptTConv(FRAME_DIM, PRIV_DIM) if student else None
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_privileged_info(self, priv: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.priv_encoder(priv))

    def encode_proprio_history(self, hist: torch.Tensor) -> torch.Tensor:
        if self.adapt_tconv is None or hist.shape[-2:] != HISTORY_SHAPE:
            raise ValueError("Student requires the adaptation encoder and [N,30,49] history")
        return torch.tanh(self.adapt_tconv(hist))

    def policy_mean(self, obs: TensorDict, *, prefer_student: bool):
        if "policy" in obs:
            actor, priv = split_actor(obs["policy"])
        else:
            actor, priv = obs["actor"], obs.get("priv_info")
        normalized = self.obs_normalizer(actor)
        if prefer_student:
            latent = self.encode_proprio_history(obs["proprio_hist"])
        else:
            if priv is None:
                raise ValueError("Teacher inference requires explicit priv_info")
            latent = self.encode_privileged_info(priv)
        trunk = self.trunk(torch.cat((normalized, latent), dim=-1))
        return self.mu_head(trunk), trunk


class HoraActor(nn.Module):
    """Common teacher/student network. Policy results never depend on a past call."""

    def __init__(self, model: dict[str, Any], *, student: bool = False):
        super().__init__()
        self.shared = TeacherCore(model, student=student)
        self.prefer_student = student
        self.action_mapping = str(model.get("action_mapping", "tanh"))
        if self.action_mapping not in {"clip", "tanh"}:
            raise ValueError("model.action_mapping must be clip or tanh")
        parameterization = model.get("std_parameterization", "log")
        width = self.shared.trunk.output_dim
        self.std_module: nn.Module
        if parameterization == "log":
            self.std_module = LogStd(width, ACTION_DIM, model)
        elif parameterization == "direct":
            self.std_module = DirectStd(ACTION_DIM, model.get("initial_std", 1.0))
        elif parameterization == "legacy_scalar":
            self.std_module = LegacyScalarStd(ACTION_DIM)
        elif parameterization == "legacy_tanh":
            self.std_module = LegacyTanhStd(width, ACTION_DIM)
        else:
            raise ValueError(f"Unsupported std parameterization: {parameterization}")

    def policy(self, obs: TensorDict | torch.Tensor) -> PolicyDistribution:
        if isinstance(obs, torch.Tensor):
            obs = TensorDict({"policy": obs}, batch_size=obs.shape[:-1])
        mean, features = self.shared.policy_mean(obs, prefer_student=self.prefer_student)
        return PolicyDistribution(mean, self.std_module(features), self.action_mapping)


def make_student(teacher: HoraActor) -> HoraActor:
    student = copy.deepcopy(teacher)
    student.prefer_student = True
    student.shared.adapt_tconv = ProprioAdaptTConv(FRAME_DIM, PRIV_DIM).to(
        next(teacher.parameters()).device
    )
    student.requires_grad_(False)
    student.shared.adapt_tconv.requires_grad_(True)
    student.shared.obs_normalizer.eval()
    return student
