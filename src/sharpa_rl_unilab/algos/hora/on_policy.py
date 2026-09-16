"""PPO/APPO adapters for the common HORA policy and independent value network."""

from __future__ import annotations

from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn
from uni_rl.algos.appo.learner import APPOLearner

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import CRITIC_DIM

from .distribution import PolicyDistribution, validate_kl_mode
from .kl_schedule import adaptive_learning_rate
from .models import _MLP, HoraActor


class TeacherActor(HoraActor):
    """Thin RSL-RL/APPO adapter; only the upstream interface needs a call cache."""

    is_recurrent = False

    def __init__(self, model: dict[str, Any], *, student: bool = False, kl_mode: str = "exact"):
        self.kl_mode = validate_kl_mode(kl_mode)
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
        return PolicyDistribution(*new_params, self.action_mapping).kl_from(
            *old_params, mode=self.kl_mode
        )

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
    """Reuse upstream losses with HORA distribution math and KL scheduling."""

    learning_rate: float

    def __init__(self, *args, kl_mode: str = "exact", **kwargs):
        self.kl_mode = validate_kl_mode(kl_mode)
        super().__init__(*args, **kwargs)
        self._skip_kl_schedule = True  # The initial target is a copy of the actor.

    def update_target_network(self):
        super().update_target_network()
        if self.tau == 1.0:
            self._skip_kl_schedule = True

    def _update_adaptive_learning_rate(self, kl_mean: float):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if self._skip_kl_schedule:
            self._skip_kl_schedule = False
            return
        self.learning_rate = adaptive_learning_rate(
            self.learning_rate,
            kl_mean,
            self.desired_kl,
            self.adaptive_kl_factor,
            self.adaptive_lr_factor,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

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
        if self.kl_mode == "log_epsilon":
            # Preserve upstream's full formula, including log(sigma / old_sigma + 1e-5).
            return result
        kl = self._loss_distribution.kl_from(old_mu_mini, old_sigma_mini).mean()
        return (*result[:4], kl, *result[5:])


class TeacherRolloutStorage(RolloutStorage):
    """Expose the active batch to the optimizer's KL scheduling hook."""

    current_batch: RolloutStorage.Batch | None = None

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        try:
            for batch in super().mini_batch_generator(num_mini_batches, num_epochs):
                self.current_batch = batch
                yield batch
        finally:
            self.current_batch = None


class TeacherPPO(PPO):
    """Reuse upstream PPO updates, scheduling LR immediately before each step."""

    def update(self):
        # Each update consumes a fresh rollout from a new reference policy.
        self._skip_kl_schedule = True
        schedule = self.schedule
        if schedule != "adaptive" or self.desired_kl is None:
            return super().update()
        assert isinstance(self.storage, TeacherRolloutStorage)
        # Upstream has no scheduler hook. Disable its inline scheduler and use
        # the existing forward result in a scoped optimizer hook, with no extra RNG.
        self.schedule = "fixed"
        try:
            with self.optimizer.register_step_pre_hook(self._schedule_before_step):
                return super().update()
        finally:
            self.schedule = schedule

    @torch.no_grad()
    def _schedule_before_step(self, optimizer, args, kwargs):
        if self._skip_kl_schedule:
            self._skip_kl_schedule = False
            return
        assert isinstance(self.storage, TeacherRolloutStorage)
        batch = self.storage.current_batch
        assert batch is not None and batch.old_distribution_params is not None
        size = batch.old_distribution_params[0].shape[0]
        current = tuple(p[:size] for p in self.actor.output_distribution_params)
        kl = self.actor.get_kl_divergence(batch.old_distribution_params, current).mean()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
            kl /= self.gpu_world_size
        self.learning_rate = adaptive_learning_rate(
            self.learning_rate, kl.item(), self.desired_kl, 2.0, 1.5
        )
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rate
