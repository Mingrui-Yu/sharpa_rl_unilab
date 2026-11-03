from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.base.torch_env import TorchEnvState

from sharpa_rl_unilab.algos.supermimic.learner import SharpaSuperMimicLearner
from sharpa_rl_unilab.algos.supermimic.optim import MuonAdam
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
        self.physics = torch.zeros(num_envs, STATE_DIM)
        self.physics[:, 3] = 1.0
        self.context = torch.zeros(num_envs, 55)
        self.context[:, :44] = 1.0
        self.context[:, -2:] = 1.0
        self.state = None

    def mdp(self, device):
        return SharpaSuperMimicMDP(self, device)

    def capture(self):
        return {"s": self.physics.clone(), "xi": self.context.clone()}

    def _obs(self, mdp):
        state = unflatten_state(self.physics)
        history = tuple(value[:, None].expand(-1, 3, *value.shape[1:]) for value in state)
        contexts = self.context[:, None].expand(-1, 3, -1)
        frame = mdp.frame(history, noisy=False)
        actor = mdp.actor(history, contexts)
        critic = mdp.critic(history, contexts)
        return {
            "obs": actor[:, :147],
            "priv_info": actor[:, 147:],
            "critic": critic,
            "proprio_hist": frame[:, :1].repeat(1, 30, 1),
        }

    def reset(self, env_indices=None, *, seed=None):
        if env_indices is not None:
            self.physics[env_indices] = 0
            self.physics[env_indices, 3] = 1
        obs = self._obs(self.mdp("cpu"))
        if self.state is None or env_indices is None:
            self.state = TorchEnvState(
                obs,
                torch.zeros(self.num_envs),
                torch.zeros(self.num_envs, dtype=torch.bool),
                torch.zeros(self.num_envs, dtype=torch.bool),
                {},
            )
        else:
            for key, value in obs.items():
                self.state.obs[key][env_indices] = value[env_indices]
        return obs, {}

    def step(self, actions):
        mdp = self.mdp("cpu")
        state = unflatten_state(self.physics)
        following = integrate_state(
            state,
            torch.zeros(2, 66),
            actions,
            mdp.lower,
            mdp.upper,
            mdp.action_scale,
        )
        following[4][:, :22] += 0.001 * actions
        following[3][:, 2] = 0.1
        from sharpa_rl_unilab.algos.supermimic.state import flatten_state

        self.physics = flatten_state(following)
        reward = mdp.reward(following, self.context)
        self.state = TorchEnvState(
            self._obs(mdp),
            reward,
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )
        return self.state


@pytest.mark.parametrize("single_stage", [False, True])
def test_complete_learner_updates_on_cpu(single_stage):
    cfg = compose_config(
        "supermimic",
        "mujoco",
        [
            "training.device=cpu",
            f"algo.single_stage={str(single_stage).lower()}",
            "training.num_envs=2",
            "algo.wm_hidden=[8,8]",
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


def test_supermimic_defaults_use_mixed_muon_adam_and_adam_remains_available():
    overrides = [
        "training.device=cpu",
        "training.num_envs=2",
        "algo.wm_hidden=[8,8]",
    ]
    cfg = compose_config("supermimic", "mujoco", overrides)
    env = FakeEnv(cfg)
    actor, critic, _ = make_models(cfg, "cpu")
    learner = SharpaSuperMimicLearner(env, cfg, actor, critic, env.mdp("cpu"), env.capture)

    assert isinstance(learner.opt_pi, MuonAdam)
    assert isinstance(learner.opt_wm, MuonAdam)
    assert not isinstance(learner.opt_v, MuonAdam)
    policy_muon = {
        id(parameter) for group in learner.opt_pi.muon.param_groups for parameter in group["params"]
    }
    world_model_muon = {
        id(parameter) for parameter in learner.opt_wm.muon.param_groups[0]["params"]
    }
    trunk = learner.actor.shared.trunk
    priv_encoder = learner.actor.shared.priv_encoder
    assert id(trunk.net[2].weight) in policy_muon
    assert id(trunk.net[4].weight) in policy_muon
    assert id(trunk.net[0].weight) not in policy_muon
    assert id(priv_encoder.net[2].weight) in policy_muon
    assert id(priv_encoder.net[4].weight) not in policy_muon
    assert id(priv_encoder.net[0].weight) not in policy_muon
    assert id(learner.actor.shared.mu_head.weight) not in policy_muon
    assert id(learner.wm.net[3].weight) in world_model_muon
    assert id(learner.wm.net[0].weight) not in world_model_muon
    assert id(learner.wm.net[-1].weight) not in world_model_muon

    cfg = compose_config(
        "supermimic", "mujoco", [*overrides, "algo.pi_opt=adam", "algo.wm_opt=adam"]
    )
    env = FakeEnv(cfg)
    actor, critic, _ = make_models(cfg, "cpu")
    learner = SharpaSuperMimicLearner(env, cfg, actor, critic, env.mdp("cpu"), env.capture)

    assert isinstance(learner.opt_pi, torch.optim.Adam)
    assert isinstance(learner.opt_wm, torch.optim.Adam)
