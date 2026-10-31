"""Collect, fit dynamics, and optimize HORA through differentiable imagination."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_HISTORY

from .networks import WorldModel
from .replay import ReplayBuffer
from .state import state_difference, unflatten_state


def _requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SharpaSuperMimicLearner:
    def __init__(self, env, cfg, actor, critic, mdp, capture):
        self.env = env
        self.cfg = cfg
        self.actor = actor
        self.critic = critic
        self.mdp = mdp
        self.capture = capture
        self.device = torch.device(cfg.training.device)
        self.wm = WorldModel(
            55,
            tuple(int(value) for value in cfg.algo.wm_hidden),
            bool(cfg.algo.use_amp),
        ).to(self.device)
        fused = self.device.type == "cuda"
        self.opt_wm = torch.optim.Adam(
            self.wm.parameters(), lr=float(cfg.algo.learning_rate), fused=fused
        )
        self.opt_pi = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(cfg.algo.learning_rate),
            betas=(0.9, 0.95),
            fused=fused,
        )
        self.opt_v = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg.algo.learning_rate), fused=fused
        )
        self.optimizer = self.opt_pi
        self.actor_avg = None
        if float(cfg.algo.pi_ema) > 0:
            self.actor_avg = AveragedModel(
                self.actor,
                multi_avg_fn=get_ema_multi_avg_fn(float(cfg.algo.pi_ema)),
                use_buffers=True,
            )
        self.buffer = ReplayBuffer(int(cfg.algo.buffer_steps), env.num_envs, self.device)
        self._wm_forward_fn = self._compile(self._wm_forward)
        self._imagination_step_fn = self._compile(self._imagination_step)
        from sharpa_rl_unilab.training.logging import EpisodeStatistics

        self.episodes = EpisodeStatistics(env.num_envs)
        self.obs, _ = env.reset(seed=int(cfg.algo.seed))
        tensor_observation = isinstance(next(iter(self.obs.values())), torch.Tensor)
        self.episode_steps = (
            torch.zeros(env.num_envs, dtype=torch.int32, device=self.device)
            if tensor_observation
            else np.zeros(env.num_envs, dtype=np.int32)
        )
        self.received = 0
        self.collected = 0

    @property
    def checkpoint_actor(self):
        return self.actor_avg.module if self.actor_avg is not None else self.actor

    def _tensors(self, capture):
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in capture.items()
        }

    @torch.no_grad()
    def collect(self, steps: int):
        from sharpa_rl_unilab.training.policy import inference_distribution, observe_new_samples

        rewards = []
        reward_gaps = []
        for _ in range(steps):
            current = self._tensors(self.capture())
            distribution = inference_distribution(self.actor, self.obs, self.device)
            action = distribution.deterministic()
            action = (action + float(self.cfg.algo.action_noise) * torch.randn_like(action)).clamp(
                -1, 1
            )
            state = self.env.step(action)
            following = self._tensors(self.capture())
            done = state.terminated | state.truncated
            transition = {
                "s": current["s"],
                "s1": following["s"],
                "a": action,
                "xi": current["xi"],
                "r": torch.as_tensor(state.reward, dtype=torch.float32, device=self.device),
                "ep_step": torch.as_tensor(self.episode_steps, device=self.device),
                "done": torch.as_tensor(done, device=self.device),
            }
            self.buffer.add(transition)
            initial = unflatten_state(current["s"])
            next_state = unflatten_state(following["s"])
            self.wm.in_norm.update(self.wm.inputs(initial, action, current["xi"]))
            self.wm.out_norm.update(self.wm.targets(initial, next_state))
            analytic_reward = self.mdp.reward(next_state, current["xi"])
            reward_gaps.append(float((analytic_reward - transition["r"]).abs().mean()))
            from sharpa_rl_unilab.algos.hora.models import pack_actor

            tensor_obs = {
                key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for key, value in self.obs.items()
            }
            observe_new_samples(
                self.actor, self.critic, pack_actor(tensor_obs), tensor_obs["critic"]
            )
            self.episodes.update(state.reward[None], state.terminated[None], state.truncated[None])
            self.episode_steps += 1
            self.episode_steps[done] = 0
            if isinstance(done, torch.Tensor):
                if done.any():
                    self.env.reset(done.nonzero(as_tuple=False).flatten())
            elif np.any(done):
                self.env.reset(np.flatnonzero(done).astype(np.int32))
            self.obs = self.env.state.obs
            self.received += self.env.num_envs
            self.collected += self.env.num_envs
            rewards.append(
                float(
                    state.reward.mean()
                    if isinstance(state.reward, torch.Tensor)
                    else np.mean(state.reward)
                )
            )
        return {
            "collected": self.collected,
            "received": self.received,
            "sim/reward": float(np.mean(rewards)) if rewards else 0.0,
            "sim/reward_parity": float(np.mean(reward_gaps)) if reward_gaps else 0.0,
        }

    def _optimize(self, loss, optimizer, module):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        if torch.isfinite(norm):
            optimizer.step()
        return norm.detach()

    def _compile(self, function):
        if not bool(self.cfg.algo.compile) or self.device.type != "cuda":
            return function
        return torch.compile(function)

    def _wm_forward(self, window):
        state = unflatten_state(window["s"][:, 0])
        losses = []
        for index in range(int(self.cfg.algo.wm_window)):
            state = self.wm(
                state,
                window["a"][:, index],
                window["xi"][:, index],
                self.mdp.lower,
                self.mdp.upper,
                self.mdp.action_scale,
            )
            expected = unflatten_state(window["s1"][:, index])
            std = torch.maximum(self.wm.out_norm.std(), self.wm.error_floor)
            losses.append((state_difference(state, expected) / std).square().mean(-1))
        error = torch.stack(losses, 1)
        mask = window["mask"]
        loss = (error * mask).sum() / error.shape[0]
        last = (error[:, -1] * mask[:, -1]).sum() / mask[:, -1].sum().clamp(min=1)
        return loss, last

    def wm_update(self):
        window = self.buffer.sample_windows(
            int(self.cfg.algo.wm_batch), int(self.cfg.algo.wm_window)
        )
        loss, last = self._wm_forward_fn(window)
        self._optimize(loss, self.opt_wm, self.wm)
        return loss.detach() / int(self.cfg.algo.wm_window), last.detach()

    def _imagination_step(self, state, packed, context, noisy, clean, priv):
        action = self.actor.policy(
            TensorDict(
                {"actor": packed[..., :147], "priv_info": packed[..., 147:]},
                batch_size=packed.shape[0],
            )
        ).deterministic()
        state = self.wm(
            state,
            action,
            context,
            self.mdp.lower,
            self.mdp.upper,
            self.mdp.action_scale,
        )
        reward = self.mdp.reward(state, context)
        done = self.mdp.soft_termination(state, context)
        frame = self.mdp.frame(state, noisy=True)
        clean_frame = self.mdp.frame(state, noisy=False)
        current_priv = self.mdp.priv(state, context)
        noisy = torch.cat((noisy[:, 1:], frame[:, None]), 1)
        clean = torch.cat((clean[:, 1:], clean_frame[:, None]), 1)
        priv = torch.cat((priv[:, 1:], current_priv[:, None]), 1)
        packed = torch.cat((noisy.flatten(-2), current_priv), -1)
        features = self._critic_features(clean, priv)
        return state, packed, noisy, clean, priv, features, reward, done

    def _initial_history(self, history):
        states = unflatten_state(history["s"])
        noisy = self.mdp.frame(states, noisy=True)
        clean = self.mdp.frame(states, noisy=False)
        priv = self.mdp.priv(states, history["xi"])
        return noisy, clean, priv

    @staticmethod
    def _critic_features(clean, priv):
        return torch.cat((clean, priv), -1).flatten(-2)

    def policy_update(self):
        start, history = self.buffer.sample_starts(int(self.cfg.algo.pi_batch), ACTOR_HISTORY)
        state = unflatten_state(start["s"])
        context = start["xi"]
        noisy, clean, priv = self._initial_history(history)
        packed = torch.cat((noisy.flatten(-2), priv[:, -1]), -1)
        features = [self._critic_features(clean, priv)]
        rewards, dones = [], []
        objective = torch.zeros(start["s"].shape[0], device=self.device)
        survival = torch.ones_like(objective)
        gamma = float(self.cfg.algo.gamma)
        _requires_grad(self.wm, False)
        _requires_grad(self.critic, False)
        try:
            for index in range(int(self.cfg.algo.horizon)):
                state, packed, noisy, clean, priv, feature, reward, done = (
                    self._imagination_step_fn(state, packed, context, noisy, clean, priv)
                )
                objective = objective + gamma**index * survival * reward
                survival = survival * (1 - done)
                rewards.append(reward.detach())
                dones.append(done.detach())
                features.append(feature)
            objective = objective + gamma ** int(self.cfg.algo.horizon) * survival * self.critic(
                TensorDict({"critic": features[-1]}, batch_size=features[-1].shape[0])
            ).squeeze(-1)
            actor_loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            actor_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            if torch.isfinite(grad_norm):
                self.opt_pi.step()
            if self.actor_avg is not None:
                self.actor_avg.update_parameters(self.actor)
        finally:
            _requires_grad(self.wm, True)
            _requires_grad(self.critic, True)

        inputs = torch.stack(features).detach()
        self.critic.obs_normalizer.update(inputs.flatten(0, 1))
        with torch.no_grad():
            values = self.critic(
                TensorDict(
                    {"critic": inputs.flatten(0, 1)},
                    batch_size=inputs.shape[0] * inputs.shape[1],
                )
            ).view(*inputs.shape[:2])
            target = values[-1]
            targets = []
            lam = float(self.cfg.algo.lam)
            for index in reversed(range(int(self.cfg.algo.horizon))):
                target = rewards[index] + gamma * (1 - dones[index]) * (
                    (1 - lam) * values[index + 1] + lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])
        flat_inputs = inputs[:-1].flatten(0, 1)
        flat_targets = targets.flatten()
        count = flat_inputs.shape[0]
        minibatches = int(self.cfg.algo.critic_minibatches)
        minibatch = count // minibatches
        if minibatch == 0:
            raise ValueError("critic_minibatches exceeds imagined transition count")
        value_loss = torch.zeros((), device=self.device)
        for _ in range(int(self.cfg.algo.critic_epochs)):
            permutation = torch.randperm(count, device=self.device)
            for index in range(minibatches):
                ids = permutation[index * minibatch : (index + 1) * minibatch]
                values = self.critic(
                    TensorDict({"critic": flat_inputs[ids]}, batch_size=ids.shape[0])
                ).squeeze(-1)
                value_loss = (values - flat_targets[ids]).square().mean()
                self._optimize(value_loss, self.opt_v, self.critic)
        return {
            "loss/actor": actor_loss.detach(),
            "loss/critic": value_loss.detach(),
            "loss/actor_grad_norm": grad_norm,
            "imagination/reward": torch.stack(rewards).mean().detach(),
            "imagination/survival": survival.mean().detach(),
        }


__all__ = ["SharpaSuperMimicLearner"]
