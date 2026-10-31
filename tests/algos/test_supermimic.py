from types import SimpleNamespace

import torch

from sharpa_rl_unilab.algos.supermimic.networks import WorldModel
from sharpa_rl_unilab.algos.supermimic.replay import ReplayBuffer
from sharpa_rl_unilab.algos.supermimic.state import (
    DELTA_DIM,
    STATE_DIM,
    flatten_state,
    state_difference,
    unflatten_state,
)
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.checkpoints import load_policy, save_teacher
from sharpa_rl_unilab.training.policy import make_models


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


def test_supermimic_checkpoint_uses_shared_hora_contract(tmp_path):
    cfg = compose_config("supermimic", "mujoco", ["training.device=cpu"])
    actor, critic, _ = make_models(cfg, "cpu")
    wm = WorldModel(55, (8,), amp=False)
    learner = SimpleNamespace(
        wm=wm,
        opt_wm=torch.optim.Adam(wm.parameters()),
        opt_pi=torch.optim.Adam(actor.parameters()),
        opt_v=torch.optim.Adam(critic.parameters()),
        optimizer=torch.optim.Adam(actor.parameters()),
    )
    path = tmp_path / "teacher.pt"
    save_teacher(path, cfg, actor, critic, learner, {"received": 0}, 0)

    restored, restored_cfg, snapshot = load_policy(path, configure_runtime=False)
    assert restored_cfg.algo.algo == "supermimic"
    assert snapshot["world_model"]["net.0.weight"].shape == wm.net[0].weight.shape
    obs = torch.randn(3, 156)
    torch.testing.assert_close(
        restored.policy(obs).deterministic(), actor.policy(obs).deterministic(), rtol=0, atol=0
    )
