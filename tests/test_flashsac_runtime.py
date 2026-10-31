from types import SimpleNamespace

import torch

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import TorchEnvState, transition_next
from sharpa_rl_unilab.training.flashsac_runtime import FlashTeacherEnv


def test_runner_noise_is_not_touched_by_learning_or_deterministic_inference():
    from sharpa_rl_unilab.cli import compose_config
    from sharpa_rl_unilab.training.exploration import HeldGaussianNoise
    from sharpa_rl_unilab.training.flashsac_runtime import TeacherDoubleBufferRunner
    from sharpa_rl_unilab.training.policy import make_models

    cfg = compose_config("flashsac", "mujoco", [])
    actor, _, learner = make_models(cfg, "cpu")
    runner = TeacherDoubleBufferRunner.__new__(TeacherDoubleBufferRunner)
    runner.learner = learner
    runner.held_noise = HeldGaussianNoise()
    actor.exploration_sampler = runner._sample_exploration
    obs = torch.randn(8, 156)
    calls = []
    handle = actor.shared.trunk.register_forward_hook(lambda *_: calls.append(True))
    actor.explore(obs, dones=torch.zeros(8))
    assert len(calls) == 1  # Distribution and exploration share one network forward.
    handle.remove()
    before = [
        value.clone()
        for value in (runner.held_noise.noise, runner.held_noise.count, runner.held_noise.target)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Only collection can advance held noise")

    runner.held_noise.sample = forbidden
    rng = torch.random.get_rng_state()
    actor.explore(obs, deterministic=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    batch = dict(
        obs=obs,
        next_obs=obs + 1,
        critic=torch.randn(8, 174),
        next_critic=torch.randn(8, 174),
        actions=torch.zeros(8, 22),
        rewards=torch.ones(8),
        dones=torch.zeros(8),
        truncated=torch.zeros(8),
    )
    learner.update_critic(batch)
    learner.update_actor(batch)
    for expected, actual in zip(
        before, (runner.held_noise.noise, runner.held_noise.count, runner.held_noise.target)
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transport_preserves_terminal_privilege_and_clean_critic():
    def observations(offset):
        return {
            key: torch.full((2, width), float(offset + i), dtype=torch.float32)
            for i, (key, width) in enumerate((("obs", 147), ("priv_info", 9), ("critic", 174)))
        }

    state = TorchEnvState(
        observations(0),
        torch.ones(2),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        {},
        observations(10),
    )
    env = FlashTeacherEnv.__new__(FlashTeacherEnv)
    env.teacher = SimpleNamespace(step=lambda _: state)
    packed = env.step(torch.zeros((2, 22)))
    assert packed.obs["obs"].shape == (2, 156)
    nxt = transition_next(packed)
    torch.testing.assert_close(nxt["obs"][:, :147], state.final_observation["obs"])
    torch.testing.assert_close(nxt["obs"][:, 147:], state.final_observation["priv_info"])
    torch.testing.assert_close(nxt["critic"], state.final_observation["critic"])
    torch.testing.assert_close(packed.obs["obs"][:, 147:], state.obs["priv_info"])
