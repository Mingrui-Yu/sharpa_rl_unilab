import torch

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.checkpoints import load_policy, save_teacher
from sharpa_rl_unilab.training.policy import make_models


def test_flash_snapshot_shares_model_storage_and_restores_learner(tmp_path):
    cfg = compose_config("flashsac", "mujoco", [])
    actor, critic, learner = make_models(cfg, "cpu")
    path = tmp_path / "teacher.pt"
    save_teacher(path, cfg, actor, critic, learner, {"received": 0}, 0)
    state = torch.load(path, weights_only=False)
    for name in ("actor", "critic"):
        assert state["learner"][name] is state[name]
    _, _, restored = make_models(cfg, "cpu")
    restored.load_state_dict(state["learner"])
    torch.testing.assert_close(restored.actor.state_dict(), actor.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.critic.state_dict(), critic.state_dict(), rtol=0, atol=0)
    policy, _, _ = load_policy(path, configure_runtime=False)
    obs = torch.randn(3, 156)
    torch.testing.assert_close(
        policy.policy(obs).deterministic(), actor.policy(obs).deterministic(), rtol=0, atol=0
    )
