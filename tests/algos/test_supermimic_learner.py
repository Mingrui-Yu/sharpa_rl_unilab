from types import SimpleNamespace

import numpy as np
import torch
from unilab.base.np_env import NpEnvState

from sharpa_rl_unilab.algos.supermimic.learner import SharpaSuperMimicLearner
from sharpa_rl_unilab.algos.supermimic.state import STATE_DIM, integrate_state, unflatten_state
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.policy import make_models
from sharpa_rl_unilab.training.supermimic_mdp import SharpaSuperMimicMDP


class FakeEnv:
    def __init__(self, cfg, num_envs=2):
        self.cfg = cfg
        self.num_envs = num_envs
        self.step_dt = 0.05
        self.action = SimpleNamespace(
            ctrl_lower=np.full(22, -1.0, dtype=np.float32),
            ctrl_upper=np.full(22, 1.0, dtype=np.float32),
            _target_lower=np.full(22, -0.9, dtype=np.float32),
            _target_upper=np.full(22, 0.9, dtype=np.float32),
            _action_scale=0.05,
        )
        self.action_manager = SimpleNamespace(get_term=lambda name: self.action)
        reset = SimpleNamespace(_rotation_axis=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        self.event_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=reset))
        self.physics = np.zeros((num_envs, STATE_DIM), dtype=np.float32)
        self.physics[:, 3] = 1.0
        self.context = np.zeros((num_envs, 55), dtype=np.float32)
        self.context[:, :44] = 1.0
        self.context[:, -2:] = 1.0
        self.state = None

    def mdp(self, device):
        return SharpaSuperMimicMDP(self, device)

    def capture(self):
        return {"s": self.physics.copy(), "xi": self.context.copy()}

    def _obs(self, mdp):
        state = unflatten_state(torch.as_tensor(self.physics))
        context = torch.as_tensor(self.context)
        history = tuple(value[:, None].expand(-1, 3, *value.shape[1:]) for value in state)
        contexts = context[:, None].expand(-1, 3, -1)
        frame = mdp.frame(history, noisy=False).numpy()
        actor = mdp.actor(history, contexts).numpy()
        critic = mdp.critic(history, contexts).numpy()
        return {
            "obs": actor[:, :147],
            "priv_info": actor[:, 147:],
            "critic": critic,
            "proprio_hist": np.repeat(frame[:, :1], 30, axis=1),
        }

    def reset(self, env_indices=None, *, seed=None):
        if env_indices is not None:
            self.physics[env_indices] = 0
            self.physics[env_indices, 3] = 1
        obs = self._obs(self.mdp("cpu"))
        if self.state is None or env_indices is None:
            self.state = NpEnvState(
                obs,
                np.zeros(self.num_envs, dtype=np.float32),
                np.zeros(self.num_envs, dtype=bool),
                np.zeros(self.num_envs, dtype=bool),
                {},
            )
        else:
            for key, value in obs.items():
                self.state.obs[key][env_indices] = value[env_indices]
        return obs, {}

    def step(self, actions):
        mdp = self.mdp("cpu")
        state = unflatten_state(torch.as_tensor(self.physics))
        action = torch.as_tensor(actions, dtype=torch.float32)
        following = integrate_state(
            state,
            torch.zeros(2, 66),
            action,
            mdp.lower,
            mdp.upper,
            mdp.action_scale,
        )
        following[4][:, :22] += 0.001 * action
        following[3][:, 2] = 0.1
        from sharpa_rl_unilab.algos.supermimic.state import flatten_state

        self.physics = flatten_state(following).numpy()
        reward = mdp.reward(following, torch.as_tensor(self.context)).numpy()
        self.state = NpEnvState(
            self._obs(mdp),
            reward,
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {},
        )
        return self.state


def test_complete_learner_updates_on_cpu():
    cfg = compose_config(
        "supermimic",
        "mujoco",
        [
            "training.device=cpu",
            "training.num_envs=2",
            "algo.wm_hidden=[8]",
            "algo.wm_batch=4",
            "algo.wm_window=2",
            "algo.wm_batches=1",
            "algo.pi_batch=4",
            "algo.horizon=2",
            "algo.critic_epochs=1",
            "algo.critic_minibatches=2",
            "algo.use_amp=false",
            "algo.pi_ema=0",
            "algo.action_noise=0",
        ],
    )
    env = FakeEnv(cfg)
    actor, critic, _ = make_models(cfg, "cpu")
    learner = SharpaSuperMimicLearner(env, cfg, actor, critic, env.mdp("cpu"), env.capture)

    metrics = learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    policy = learner.policy_update()

    assert metrics["received"] == 6
    assert learner.buffer.size == 3
    assert torch.isfinite(wm_loss) and torch.isfinite(wm_error)
    for value in policy.values():
        assert value.ndim == 0 and torch.isfinite(value)
