"""FlashSAC actor, Q and learner adapters; native losses remain upstream."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.flash_sac.network import FlashSACDoubleCritic
from uni_rl.algos.flash_sac.update import build_lr_lambda

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import (
    ACTION_DIM,
    ACTOR_DIM,
    CRITIC_DIM,
    PRIV_DIM,
)

from .models import HoraActor


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
            action_dim=ACTION_DIM,
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
        # Upstream currently constructs its own actor. Keep replacement and optimizer
        # setup here until it supports an actor factory; preserve initialization RNG.
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
