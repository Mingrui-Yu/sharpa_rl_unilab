"""Differentiable Sharpa observations, rewards and drop termination for imagination."""

from __future__ import annotations

import numpy as np
import torch

from sharpa_rl_unilab.algos.supermimic.state import State
from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import ACTOR_HISTORY, PRIV_DIM
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import SOURCE_DEFAULT_HAND_JOINT_POS

CONTEXT_DIM = 55
CONTEXT_SPLITS = (22, 22, 1, 1, 3, 1, 3, 1, 1)


class SharpaSuperMimicCapture:
    def __init__(self, env):
        self.env = env
        self.observation = env.observation_manager.get_term_cfg("actor", "frame").func
        self.action = env.action_manager.get_term("hand")
        self.reset = env.event_manager.get_term_cfg("reset").func
        self.randomization = env.event_manager.get_term_cfg("domain_randomization").func

    def __call__(self) -> dict[str, np.ndarray | torch.Tensor]:
        source = self.observation.snapshot(self.env.env)
        state_parts = (
            source.object_pos,
            source.object_quat,
            source.object_linvel,
            source.object_angvel,
            source.dof_pos,
            source.dof_vel,
            source._clean_tactile,
            source._tactile_output,
            self.action.target,
        )
        dr = self.randomization
        task = self.reset
        context_parts = (
            dr.kp,
            dr.kd,
            dr.friction_scale,
            dr.mass[:, None],
            dr.com_offset,
            dr.scale,
            task.object_pos_anchor,
            task.reset_height_lower[:, None],
            task.reset_height_upper[:, None],
        )
        tensor_parts = [
            value for value in (*state_parts, *context_parts) if isinstance(value, torch.Tensor)
        ]
        if tensor_parts:
            device = tensor_parts[0].device

            def concatenate(parts):
                return torch.cat(
                    [torch.as_tensor(value, dtype=torch.float32, device=device) for value in parts],
                    dim=-1,
                )

            return {"s": concatenate(state_parts), "xi": concatenate(context_parts)}
        state = np.concatenate(state_parts, axis=-1).astype(np.float32)
        context = np.concatenate(context_parts, axis=-1).astype(np.float32)
        return {"s": state.copy(), "xi": context.copy()}


class SharpaSuperMimicMDP:
    def __init__(self, env, device):
        self.device = torch.device(device)
        action = env.action_manager.get_term("hand")

        def tensor(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=torch.float32)
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

        self.default = tensor(SOURCE_DEFAULT_HAND_JOINT_POS)
        self.mid = tensor((action.ctrl_lower + action.ctrl_upper) / 2)
        self.range = tensor(action.ctrl_upper - action.ctrl_lower)
        self.lower = tensor(action._target_lower)
        self.upper = tensor(action._target_upper)
        self.action_scale = float(action._action_scale)
        self.dt = float(env.step_dt)
        self.rotation_axis = tensor(env.event_manager.get_term_cfg("reset").func._rotation_axis)
        reward_cfg = getattr(env.cfg, "rewards", None)
        if reward_cfg is None:
            reward_cfg = env.cfg.reward
        self.reward_weights = {
            name: float(reward_cfg[name].weight)
            for name in ("rotate", "obj_linvel", "pose_diff", "torque", "work", "object_pos")
        }

    @staticmethod
    def context_parts(context: torch.Tensor):
        return context.split(CONTEXT_SPLITS, -1)

    def priv(self, state: State, context: torch.Tensor) -> torch.Tensor:
        position = state[0]
        _, _, friction, mass, com, scale, anchor, _, _ = self.context_parts(context)
        return torch.cat((position - anchor, friction, mass, com, scale), -1)

    def frame(self, state: State, *, noisy: bool) -> torch.Tensor:
        joint_pos, tactile = state[4], state[6]
        normalized = 2 * (joint_pos - self.mid) / (self.range + 1e-8)
        if noisy:
            normalized = normalized + (2 * torch.rand_like(normalized) - 1) * 0.02
        contact = tactile[..., 5:] if noisy else tactile[..., :5]
        return torch.cat((normalized, state[7], contact), -1).clamp(-5, 5)

    def actor(self, states: State, contexts: torch.Tensor) -> torch.Tensor:
        frames = self.frame(states, noisy=True)
        history = frames.flatten(-2)
        current_state: State = (
            states[0][:, -1],
            states[1][:, -1],
            states[2][:, -1],
            states[3][:, -1],
            states[4][:, -1],
            states[5][:, -1],
            states[6][:, -1],
            states[7][:, -1],
        )
        current_priv = self.priv(current_state, contexts[:, -1])
        return torch.cat((history, current_priv), -1)

    def critic(self, states: State, contexts: torch.Tensor) -> torch.Tensor:
        frames = self.frame(states, noisy=False)
        priv = self.priv(states, contexts)
        return torch.cat((frames, priv), -1).flatten(-2)

    def current_actor(
        self, state: State, context: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.frame(state, noisy=True)
        history = torch.cat((history[..., 1:, :], frame.unsqueeze(-2)), -2)
        return history, torch.cat((history.flatten(-2), self.priv(state, context)), -1)

    def reward(self, state: State, context: torch.Tensor) -> torch.Tensor:
        position, _, lin_vel, ang_vel, joint_pos, joint_vel, _, target = state
        kp, kd, _, _, _, _, anchor, _, _ = self.context_parts(context)
        torques = kp * (target - joint_pos) - kd * joint_vel
        weights = self.reward_weights
        reward = (
            weights["rotate"] * (ang_vel * self.rotation_axis).sum(-1).clamp(-0.5, 0.5)
            + weights["obj_linvel"] * lin_vel.abs().sum(-1)
            + weights["pose_diff"] * (joint_pos - self.default).square().sum(-1)
            + weights["torque"] * torques.square().sum(-1)
            + weights["work"] * (torques * joint_vel).sum(-1).square()
            + weights["object_pos"] / ((position - anchor).norm(dim=-1) + 0.001)
        )
        return reward * self.dt

    def soft_termination(self, state: State, context: torch.Tensor) -> torch.Tensor:
        height = state[0][..., 2]
        *_, lower, upper = self.context_parts(context)
        lower, upper = lower.squeeze(-1), upper.squeeze(-1)
        alive = (1 - torch.sigmoid((height - upper) / 0.002)) * (
            1 - torch.sigmoid((lower - height) / 0.002)
        )
        return 1 - alive


__all__ = [
    "ACTOR_HISTORY",
    "CONTEXT_DIM",
    "PRIV_DIM",
    "SharpaSuperMimicCapture",
    "SharpaSuperMimicMDP",
]
