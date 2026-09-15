"""Common HORA Actor and independent V/Q adapters for protocol v2."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F
from uni_rl.algos.appo.learner import APPOLearner, _distribution_std
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.flash_sac.network import FlashSACActor, FlashSACDoubleCritic
from uni_rl.algos.flash_sac.update import build_lr_lambda

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
        self.distribution = GaussianDistribution(22, init_std=1.0, std_type="scalar")
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

    def policy_mean_from_tensors(self, packed: torch.Tensor) -> torch.Tensor:
        actor, priv = split_actor(packed)
        return self.mu_head(
            self.trunk(
                torch.cat((self.obs_normalizer(actor), self.encode_privileged_info(priv)), dim=-1)
            )
        )


class TeacherActor(nn.Module):
    is_recurrent = False

    def __init__(self, model: dict[str, Any], *, student: bool = False):
        super().__init__()
        self.shared = TeacherCore(model, student=student)
        self.prefer_student = student

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state
        mean, _ = self.shared.policy_mean(obs, prefer_student=self.prefer_student)
        self.shared.distribution.update(mean)
        if stochastic_output:
            return self.shared.distribution.sample()
        return self.shared.distribution.deterministic_output(mean)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def distribution(self) -> GaussianDistribution:
        return self.shared.distribution

    @property
    def output_mean(self) -> torch.Tensor:
        return self.shared.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.shared.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.shared.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return cast(tuple[torch.Tensor, ...], self.shared.distribution.params)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.shared.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.shared.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        # Only the fresh-data receiver updates statistics. PPO epochs and APPO
        # process_batch may call this method again on reused data.
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
    def _minibatch_policy_value(self, obs_mini, critic_obs_mini):
        actor = cast(TeacherActor, self.actor)
        critic = cast(CleanValue, self.critic)
        mean = actor.shared.policy_mean_from_tensors(obs_mini)
        std = _distribution_std(self.actor.distribution, mean)
        value = critic.mlp(critic.obs_normalizer(critic_obs_mini)).squeeze(-1)
        return mean, std, value


class TeacherFlashActor(FlashSACActor):
    """HORA MLP with FlashSAC's bounded std, tanh density and held exploration."""

    def __init__(self, model, *, noise_zeta_mu=2.0, noise_zeta_max=16, student=False):
        nn.Module.__init__(self)
        self.shared = TeacherCore(model, student=student)
        # SAC learns state-dependent std; PPO's Gaussian parameters are unused.
        self.shared.distribution.requires_grad_(False)
        self.std_head = nn.Linear(self.shared.trunk.output_dim, 22)
        nn.init.zeros_(self.std_head.bias)
        self.noise_zeta_mu, self.noise_zeta_max = noise_zeta_mu, noise_zeta_max
        ns = torch.arange(1, noise_zeta_max + 1, dtype=torch.float32)
        pmf = ns.pow(-noise_zeta_mu)
        self.register_buffer("zeta_cdf", torch.cumsum(pmf / pmf.sum(), dim=0))
        self.register_buffer("_noise", torch.zeros(0), persistent=False)
        self.register_buffer("_repeat_count", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_repeat_target", torch.zeros(0, dtype=torch.int32), persistent=False)

    def get_mean_and_std(self, observations, training=False):
        obs = TensorDict({"policy": observations}, batch_size=observations.shape[0])
        mean, core = self.shared.policy_mean(obs, prefer_student=False)
        log_std = -10.0 + 6.0 * (1.0 + torch.tanh(self.std_head(core.trunk_latent)))
        return mean, log_std.exp()

    def forward(self, observations, training=False):
        mean, std = self.get_mean_and_std(observations, training)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        log_det = 2.0 * (
            torch.log(torch.tensor(2.0, device=raw.device)) - raw - F.softplus(-2.0 * raw)
        )
        return raw.tanh(), {
            "log_prob": (dist.log_prob(raw) - log_det).sum(-1),
            "mean": mean,
            "std": std,
        }

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
        self.actor = TeacherFlashActor(
            model,
            noise_zeta_mu=kwargs.get("actor_noise_zeta_mu", 2.0),
            noise_zeta_max=kwargs.get("actor_noise_zeta_max", 16),
        ).to(self.device)
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
    if isinstance(student, TeacherActor):
        student.prefer_student = True
    student.shared.adapt_tconv = ProprioAdaptTConv(49, PRIV_DIM).to(
        next(teacher.parameters()).device
    )
    student.requires_grad_(False)
    student.shared.adapt_tconv.requires_grad_(True)
    student.shared.obs_normalizer.eval()
    return student
