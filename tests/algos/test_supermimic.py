from types import SimpleNamespace

import torch

from sharpa_rl_unilab.algos.supermimic.networks import WorldModel
from sharpa_rl_unilab.algos.supermimic.replay import ReplayBuffer
from sharpa_rl_unilab.algos.supermimic.state import (
    DELTA_DIM,
    STATE_DIM,
    flatten_state,
    integrate_state,
    state_difference,
    structured_integrate_state,
    structured_state_difference,
    unflatten_state,
)
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.checkpoints import load_policy, save_supermimic
from sharpa_rl_unilab.training.policy import inference_distribution, make_models


def _state(batch=2):
    value = torch.randn(batch, STATE_DIM)
    value[:, 3:7] = torch.nn.functional.normalize(value[:, 3:7], dim=-1)
    value[:, 3] = value[:, 3].abs()
    value[:, 57:67] = value[:, 57:67].clamp(0, 4)
    value[:, -22:] = value[:, -22:].clamp(-0.5, 0.5)
    return unflatten_state(value)


def test_world_model_zero_initialization_preserves_physical_state():
    model = WorldModel(55, (16,), amp=False)
    state = _state()
    action = torch.randn(2, 22).clamp(-1, 1)
    context = torch.randn(2, 55)
    following = model(state, action, context, -0.75, 0.75, 0.05)

    assert model.inputs(state, action, context).shape == (2, 168)
    assert model.targets(state, following).shape == (2, DELTA_DIM)
    torch.testing.assert_close(flatten_state(following)[..., :57], flatten_state(state)[..., :57])
    expected_target = (state[-1] + 0.05 * action).clamp(-0.75, 0.75)
    torch.testing.assert_close(following[-1], expected_target)
    torch.testing.assert_close(state_difference(state, following), torch.zeros(2, DELTA_DIM))


def test_structured_state_difference_round_trips():
    state = _state()
    action = torch.randn(2, 22).clamp(-1, 1)
    next_state = integrate_state(state, torch.randn(2, DELTA_DIM), action, -0.75, 0.75, 0.05)

    targets = structured_state_difference(state, next_state, 0.05)
    assert targets.shape == (2, DELTA_DIM)

    recovered = structured_integrate_state(state, targets, action, -0.75, 0.75, 0.05, 0.05)
    torch.testing.assert_close(flatten_state(recovered), flatten_state(next_state))


def test_structured_world_model_zero_init_drifts_with_velocity():
    model = WorldModel(55, (16,), amp=False, structured=True, dt=0.05)
    state = _state()
    following = model(state, torch.zeros(2, 22), torch.randn(2, 55), -0.75, 0.75, 0.05)

    position, _, lin_vel, ang_vel, joint_pos, joint_vel, tactile, _ = state
    f_position, _, f_lin_vel, f_ang_vel, f_joint_pos, f_joint_vel, f_tactile, _ = following
    torch.testing.assert_close(f_lin_vel, lin_vel)
    torch.testing.assert_close(f_ang_vel, ang_vel)
    torch.testing.assert_close(f_joint_vel, joint_vel)
    torch.testing.assert_close(f_position, position + 0.05 * lin_vel)
    torch.testing.assert_close(f_joint_pos, joint_pos + 0.05 * joint_vel)
    torch.testing.assert_close(f_tactile, tactile)


def test_replay_masks_done_and_pads_history_at_episode_start(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(size, dtype=torch.long, **kwargs),
    )
    buffer = ReplayBuffer(4, 1, "cpu")
    for step in range(4):
        value = torch.full((1, STATE_DIM), float(step))
        value[:, 3] = 1
        buffer.add(
            {
                "s": value,
                "s1": value + 1,
                "a": torch.zeros(1, 22),
                "xi": torch.zeros(1, 55),
                "r": torch.zeros(1),
                "ep_step": torch.tensor([step % 2], dtype=torch.int32),
                "done": torch.tensor([step == 1]),
            }
        )

    window = buffer.sample_windows(8, 4)
    assert window["mask"].shape == (8, 4)
    assert torch.all(window["mask"][:, 2:] == 0)
    start, history = buffer.sample_starts(8, 3)
    assert start["s"].shape == (8, STATE_DIM)
    assert history["s"].shape == (8, 3, STATE_DIM)


def test_supermimic_default_remains_hora_teacher():
    cfg = compose_config("supermimic", "mujoco", ["training.device=cpu"])
    actor, _, _ = make_models(cfg, "cpu")

    assert not cfg.algo.single_stage
    assert cfg.protocol.stage == "teacher"
    assert actor.shared.priv_encoder is not None
    assert actor.shared.trunk.net[0].in_features == 156


def test_supermimic_checkpoint_restores_direct_policy(tmp_path):
    cfg = compose_config("supermimic", "mujoco", ["training.device=cpu", "algo.single_stage=true"])
    actor, critic, _ = make_models(cfg, "cpu")
    wm = WorldModel(55, (8,), amp=False)
    learner = SimpleNamespace(
        wm=wm,
        opt_wm=torch.optim.Adam(wm.parameters()),
        opt_pi=torch.optim.Adam(actor.parameters()),
        opt_v=torch.optim.Adam(critic.parameters()),
        optimizer=torch.optim.Adam(actor.parameters()),
    )
    path = tmp_path / "policy.pt"
    save_supermimic(path, cfg, actor, critic, learner, {"received": 0}, 0)

    restored, restored_cfg, snapshot = load_policy(path, configure_runtime=False)
    assert restored_cfg.algo.algo == "supermimic"
    assert snapshot["stage"] == "direct"
    assert snapshot["world_model"]["net.0.weight"].shape == wm.net[0].weight.shape
    assert not any(key.startswith("shared.priv_encoder") for key in snapshot["actor"])
    obs = torch.randn(3, 147)
    torch.testing.assert_close(
        restored.policy(obs).deterministic(), actor.policy(obs).deterministic(), rtol=0, atol=0
    )


def test_supermimic_actor_ignores_privileged_observations():
    cfg = compose_config("supermimic", "mujoco", ["training.device=cpu", "algo.single_stage=true"])
    actor, _, _ = make_models(cfg, "cpu")
    obs = {"obs": torch.randn(3, 147), "priv_info": torch.randn(3, 9)}
    first = inference_distribution(actor, obs, "cpu").deterministic()
    obs["priv_info"] = torch.randn(3, 9)
    second = inference_distribution(actor, obs, "cpu").deterministic()

    torch.testing.assert_close(first, second, rtol=0, atol=0)
