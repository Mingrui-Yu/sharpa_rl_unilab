"""Common HORA Actor and independent V/Q adapters for protocol v2."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn
from uni_rl.algos.appo.learner import APPOLearner
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.flash_sac.network import FlashSACDoubleCritic
from uni_rl.algos.flash_sac.update import build_lr_lambda

from .distribution import LogStd, PolicyDistribution
from .legacy import LegacyScalarStd, LegacyTanhStd
from .models import _MLP, HoraCoreOutput, ProprioAdaptTConv

ACTOR_DIM = 147
PRIV_DIM = 9
CRITIC_DIM = 174


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
        self.obs_dim, self.priv_info_dim, self.priv_info_embed_dim = ACTOR_DIM, PRIV_DIM, PRIV_DIM
        self.proprio_hist_len, self.proprio_frame_dim = 30, 49
        self.obs_normalizer = EmpiricalNormalization(ACTOR_DIM)
        self.priv_encoder = _MLP(PRIV_DIM, model["priv_mlp_hidden_dims"], model["activation"])
        self.trunk = _MLP(ACTOR_DIM + PRIV_DIM, model["actor_hidden_dims"], model["activation"])
        self.mu_head = nn.Linear(self.trunk.output_dim, 22)
        self.adapt_tconv = ProprioAdaptTConv(49, PRIV_DIM) if student else None
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_privileged_info(self, priv: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.priv_encoder(priv))

    def encode_proprio_history(self, hist: torch.Tensor) -> torch.Tensor:
        if self.adapt_tconv is None or hist.shape[-2:] != (30, 49):
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
            with torch.no_grad():
                target = (
                    self.encode_privileged_info(priv)
                    if priv is not None
                    else torch.zeros_like(latent)
                )
        else:
            if priv is None:
                raise ValueError("Teacher inference requires explicit priv_info")
            latent = target = self.encode_privileged_info(priv)
        trunk = self.trunk(torch.cat((normalized, latent), dim=-1))
        return self.mu_head(trunk), HoraCoreOutput(normalized, trunk, latent, target)


class HoraActor(nn.Module):
    """Common teacher/student network. Policy results never depend on a past call."""

    def __init__(self, model: dict[str, Any], *, student: bool = False):
        super().__init__()
        self.shared = TeacherCore(model, student=student)
        self.prefer_student = student
        self.action_mapping = str(model.get("action_mapping", "clip"))
        if self.action_mapping not in {"clip", "tanh"}:
            raise ValueError("model.action_mapping must be clip or tanh")
        parameterization = model.get("std_parameterization", "log")
        width = self.shared.trunk.output_dim
        self.std_module: nn.Module
        if parameterization == "log":
            self.std_module = LogStd(width, 22, model)
        elif parameterization == "legacy_scalar":
            self.std_module = LegacyScalarStd(22)
        elif parameterization == "legacy_tanh":
            self.std_module = LegacyTanhStd(width, 22)
        else:
            raise ValueError(f"Unsupported std parameterization: {parameterization}")

    def policy(self, obs: TensorDict | torch.Tensor) -> PolicyDistribution:
        if isinstance(obs, torch.Tensor):
            obs = TensorDict({"policy": obs}, batch_size=obs.shape[:-1])
        mean, core = self.shared.policy_mean(obs, prefer_student=self.prefer_student)
        return PolicyDistribution(mean, self.std_module(core.trunk_latent), self.action_mapping)


class TeacherActor(HoraActor):
    """Thin RSL-RL/APPO adapter; only the upstream interface needs a call cache."""

    is_recurrent = False

    def __init__(self, model: dict[str, Any], *, student: bool = False):
        super().__init__(model, student=student)
        self._distribution: PolicyDistribution | None = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        self._distribution = self.policy(obs)
        # Upstream storage/log_prob use the latent sample, never the mapped action.
        if stochastic_output:
            return self._distribution.sample(log_prob=False).raw
        return self._distribution.deterministic()

    @property
    def distribution(self) -> PolicyDistribution:
        if self._distribution is None:
            raise RuntimeError("Call the adapter before reading its distribution")
        return self._distribution

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy()

    @property
    def output_distribution_params(self):
        return self.output_mean, self.output_std

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, torch.Tensor],
        new_params: tuple[torch.Tensor, torch.Tensor],
    ):
        return PolicyDistribution(*new_params, self.action_mapping).kl_from(*old_params)

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    def update_normalization(self, obs):
        # Only the fresh-data receiver updates statistics, never replay/epochs.
        pass


class CleanValue(nn.Module):
    is_recurrent = False

    def __init__(self, model: dict[str, Any]):
        super().__init__()
        if model["critic_hidden_dims"] != [512, 256, 128] or not model["critic_normalization"]:
            raise ValueError("Protocol v2 fixes the PPO/APPO independent clean V network")
        self.obs_normalizer = EmpiricalNormalization(CRITIC_DIM)
        trunk = _MLP(CRITIC_DIM, model["critic_hidden_dims"], model["activation"])
        self.mlp = nn.Sequential(trunk, nn.Linear(trunk.output_dim, 1))

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
        return self.mlp(self.obs_normalizer(obs["critic"] if "critic" in obs else obs["policy"]))

    def update_normalization(self, obs):
        pass

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass


class TeacherAPPOLearner(APPOLearner):
    """Reuse upstream V-trace and PPO losses, replacing only distribution math."""

    def _minibatch_policy_value(self, obs_mini, critic_obs_mini):
        actor = cast(TeacherActor, self.actor)
        critic = cast(CleanValue, self.critic)
        self._loss_distribution = actor.policy(obs_mini)
        value = critic.mlp(critic.obs_normalizer(critic_obs_mini)).squeeze(-1)
        return self._loss_distribution.mean, self._loss_distribution.std, value

    def _gaussian_log_prob(self, actions, mean, std):  # pyright: ignore[reportIncompatibleMethodOverride]
        return PolicyDistribution(
            mean, std, cast(TeacherActor, self.actor).action_mapping
        ).log_prob(actions)

    def _gaussian_entropy(self, std):  # pyright: ignore[reportIncompatibleMethodOverride]
        # The upstream hook accepts only std; the preceding policy/value call
        # supplies the mean needed for transformed entropy without another MLP.
        return self._loss_distribution.entropy()

    def _minibatch_loss_tensors(
        self,
        obs_mini,
        critic_obs_mini,
        actions_mini,
        target_values_mini,
        advantages_mini,
        behavior_logp_mini,
        old_values_mini,
        target_logp_mini,
        old_mu_mini,
        old_sigma_mini,
    ):
        result = super()._minibatch_loss_tensors(
            obs_mini,
            critic_obs_mini,
            actions_mini,
            target_values_mini,
            advantages_mini,
            behavior_logp_mini,
            old_values_mini,
            target_logp_mini,
            old_mu_mini,
            old_sigma_mini,
        )
        kl = self._loss_distribution.kl_from(old_mu_mini, old_sigma_mini).mean()
        # Upstream adds 1e-5 inside log(sigma / old_sigma); use exact KL instead.
        return (*result[:4], kl, *result[5:])


class TeacherFlashActor(HoraActor):
    """Thin FlashSAC return adapter. Exploration is supplied by the runner."""

    def __init__(self, model, *, student=False):
        super().__init__(model, student=student)
        if self.action_mapping != "tanh":
            raise ValueError("FlashSAC requires model.action_mapping=tanh")
        self.exploration_sampler: Callable | None = None

    def get_mean_and_std(self, observations, training=False):
        dist = self.policy(observations)
        return dist.mean, dist.std

    def forward(self, observations, training=False):
        sample = self.policy(observations).sample()
        return sample.action, {"log_prob": sample.log_prob, "mean": sample.mean, "std": sample.std}

    @torch.no_grad()
    def explore(self, obs, dones=None, deterministic=False):
        if deterministic:
            return self.policy(obs).deterministic()
        if self.exploration_sampler is None:
            raise RuntimeError("FlashSAC exploration must be bound by its runner")
        return self.exploration_sampler(obs, dones)

    def normalize_parameters(self):
        # Unit-normalizing HORA weights would change the public MLP contract.
        pass


class CleanQ(nn.Module):
    """Native Q network with observation-only, fresh-sample normalization."""

    def __init__(self, network: FlashSACDoubleCritic, normalizer: EmpiricalNormalization):
        super().__init__()
        self.network = network
        self.obs_normalizer = normalizer

    @property
    def predictor(self):
        return self.network.predictor

    def normalize_parameters(self):
        self.network.normalize_parameters()

    def forward(self, observations, actions, training):
        return self.network(self.obs_normalizer(observations), actions, training=training)


class TeacherFlashLearner(FlashSACLearner):
    def __init__(self, model, **kwargs):
        super().__init__(
            obs_dim=ACTOR_DIM + PRIV_DIM,
            action_dim=22,
            critic_obs_dim=CRITIC_DIM,
            obs_normalization=False,
            **kwargs,
        )
        if model["critic_normalization"]:
            # Share only statistics. Native optimizers retain the same network
            # parameters, and Polyak updates never touch normalization buffers.
            normalizer = EmpiricalNormalization(CRITIC_DIM).to(self.device)
            self.critic = cast(Any, CleanQ(self.critic, normalizer))
            self.target_critic = cast(Any, CleanQ(self.target_critic, normalizer))
        self.actor = TeacherFlashActor(model).to(self.device)
        peak = kwargs.get("learning_rate_peak", 3e-4)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=peak, fused=self.device.type == "cuda"
        )
        schedule = build_lr_lambda(
            init_lr=kwargs.get("learning_rate_init", peak),
            peak_lr=peak,
            end_lr=kwargs.get("learning_rate_end", 1.5e-4),
            warmup_steps=kwargs.get("learning_rate_warmup_steps", 0),
            decay_steps=kwargs.get("learning_rate_decay_steps", 500000),
        )
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, schedule)
        self.saturation_metrics: dict[str, float] = {}

    def _critic_loss_tensors(
        self,
        next_q_values,
        next_q_log_probs_full,
        support,
        rewards,
        dones,
        truncated,
        actor_entropy,
        pred_log_probs,
        gamma,
    ):
        with torch.no_grad():
            bootstrap = (1.0 - dones + truncated).clamp(0, 1).view(-1, 1)
            target = rewards.view(-1, 1) + bootstrap * gamma * (
                support.view(1, -1) - actor_entropy.view(-1, 1)
            )
            self.saturation_metrics = {
                "q/target_out_of_support": ((target < support[0]) | (target > support[-1]))
                .float()
                .mean()
                .item(),
                "q/boundary_mass": pred_log_probs.exp()[..., [0, -1]].sum(-1).mean().item(),
            }
        return super()._critic_loss_tensors(
            next_q_values,
            next_q_log_probs_full,
            support,
            rewards,
            dones,
            truncated,
            actor_entropy,
            pred_log_probs,
            gamma,
        )


@torch.no_grad()
def observe_new_samples(actor, critic, packed: torch.Tensor, clean: torch.Tensor) -> None:
    base, _ = split_actor(packed)
    actor.shared.obs_normalizer.update(base.reshape(-1, ACTOR_DIM))
    if isinstance(critic, (CleanValue, CleanQ)):
        critic.obs_normalizer.update(clean.reshape(-1, CRITIC_DIM))


def frozen_weights(module: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def make_student(teacher: TeacherActor | TeacherFlashActor) -> TeacherActor | TeacherFlashActor:
    student = copy.deepcopy(teacher)
    student.prefer_student = True
    student.shared.adapt_tconv = ProprioAdaptTConv(49, PRIV_DIM).to(
        next(teacher.parameters()).device
    )
    student.requires_grad_(False)
    student.shared.adapt_tconv.requires_grad_(True)
    student.shared.obs_normalizer.eval()
    return student
