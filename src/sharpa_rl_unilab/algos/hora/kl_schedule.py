"""KL feedback for HORA's on-policy learners."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


def adaptive_learning_rate(lr: float, kl: float, target: float, kl_factor: float, lr_factor: float):
    """Keep negative/NaN KL out of the low-KL branch; retain upstream LR limits."""
    if kl > target * kl_factor:
        return max(1e-5, lr / lr_factor)
    if 0 <= kl < target / kl_factor:
        return min(1e-2, lr * lr_factor)
    return lr


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
