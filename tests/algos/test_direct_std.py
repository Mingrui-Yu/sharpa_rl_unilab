import copy

import pytest
import torch
from omegaconf import OmegaConf

from sharpa_rl_unilab.algos.hora.distribution import DirectStd
from sharpa_rl_unilab.algos.hora.on_policy import TeacherActor, TeacherAPPOLearner
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.checkpoints import load_policy
from sharpa_rl_unilab.training.policy import make_models
from sharpa_rl_unilab.training.teacher_runtime import train_teacher


def test_direct_std_preserves_original_forward_and_gradient():
    module = DirectStd(5)
    raw = torch.tensor([-1.0, 0.2, 1.0, 3.0, 1e7])
    with torch.no_grad():
        module.std_param.copy_(raw)
    std = module(torch.randn(4, 128))
    torch.testing.assert_close(std, torch.tensor([1e-6, 0.2, 1.0, 3.0, 1e6]))
    # Clamp affects the distribution, not the underlying optimizer parameter.
    torch.testing.assert_close(module.std_param, raw, atol=0, rtol=0)
    std.log().sum().backward()
    torch.testing.assert_close(module.std_param.grad, torch.tensor([0.0, 5.0, 1.0, 1 / 3, 0.0]))


@pytest.mark.parametrize("parameterization", ["direct", "legacy_scalar"])
@pytest.mark.parametrize("initial", [1e-6, 0.7, 1.0, 1e6])
def test_direct_initialization_alias_and_appo_synchronization(parameterization, initial):
    cfg = compose_config(
        "appo",
        "mujoco",
        [f"model.std_parameterization={parameterization}", f"model.initial_std={initial}"],
    )
    actor, critic, _ = make_models(cfg, "cpu")
    assert cfg.model.std_parameterization == parameterization
    assert set(actor.std_module.state_dict()) == {"std_param"}
    obs = torch.randn(4, 156)
    torch.testing.assert_close(actor.policy(obs).std, torch.full((4, 22), initial))
    learner = TeacherAPPOLearner(actor=actor, critic=critic, device="cpu")
    # The collector constructs directly from saved config, then installs weights.
    collector = TeacherActor(OmegaConf.to_container(cfg.model, resolve=True))
    with torch.no_grad():
        actor.std_module.std_param.copy_(torch.linspace(0.2, 0.8, 22))
    collector.load_state_dict(actor.state_dict())
    learner.update_target_network()
    for replica in (collector, learner.target_actor):
        torch.testing.assert_close(replica.policy(obs).std, actor.policy(obs).std, atol=0, rtol=0)


@pytest.mark.parametrize("parameterization", ["direct", "legacy_scalar"])
@pytest.mark.parametrize(
    "override,match",
    [
        ("model.std_mode=state_dependent", "state_independent"),
        ("model.std_mode=invalid", "state_independent"),
        ("model.log_std_bounds=[-10,2]", "log_std_bounds=null"),
        ("model.initial_std=0", "initial_std"),
        ("model.initial_std=-1", "initial_std"),
        ("model.initial_std=1e-7", "initial_std"),
        ("model.initial_std=1e7", "initial_std"),
        ("model.initial_std=.nan", "initial_std"),
        ("model.initial_std=.inf", "initial_std"),
        ("model.initial_std=nan", "initial_std"),
        ("model.initial_std=inf", "initial_std"),
        ("model.initial_std=null", "initial_std"),
    ],
)
def test_invalid_direct_training_config(parameterization, override, match):
    cfg = compose_config(
        "ppo", "mujoco", [f"model.std_parameterization={parameterization}", override]
    )
    with pytest.raises(ValueError, match=match):
        make_models(cfg, "cpu")


@pytest.mark.parametrize("parameterization", ["direct", "legacy_scalar"])
def test_flashsac_direct_training_is_rejected(parameterization):
    cfg = compose_config("flashsac", "mujoco", [f"model.std_parameterization={parameterization}"])
    with pytest.raises(ValueError, match="only for PPO/APPO"):
        make_models(cfg, "cpu")


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "appo"])
@pytest.mark.parametrize("parameterization", ["direct", "legacy_scalar"])
def test_direct_training_checkpoint_roundtrip(tmp_path, algo, parameterization):
    cfg = compose_config(
        algo,
        "mujoco",
        [
            f"model.std_parameterization={parameterization}",
            "model.initial_std=0.7",
            "training.device=cpu",
            "training.num_envs=8",
            "training.torch_threads.learner_num_threads=2",
            "training.torch_threads.collector_num_threads=2",
            "training.no_play=true",
            "training.logger=no_print",
            f"training.log_dir={tmp_path / 'run'}",
            "algo.max_iterations=2",
            "algo.save_interval=1",
            "algo.algorithm.num_learning_epochs=1",
            "algo.algorithm.num_mini_batches=1",
            "algo.num_steps_per_env=2" if algo == "ppo" else "algo.steps_per_env=2",
        ],
    )
    path = train_teacher(cfg)
    actor, restored_cfg, snapshot = load_policy(path, configure_runtime=False)
    assert restored_cfg.model.std_parameterization == parameterization
    assert snapshot["counters"]["policy_version"] == 2
    assert snapshot["counters"]["optimizer_updates"] == 2
    torch.testing.assert_close(actor.state_dict(), snapshot["actor"], atol=0, rtol=0)
    assert torch.isfinite(actor.std_module.std_param).all()
    assert not torch.equal(actor.std_module.std_param, torch.full((22,), 0.7))
    obs, noise = torch.randn(4, 156), torch.randn(4, 22)
    sample = actor.policy(obs).sample(noise)
    assert torch.isfinite(sample.log_prob).all()
    if algo == "appo":
        target = copy.deepcopy(actor)
        target.load_state_dict(snapshot["target_actor"], strict=True)
        assert torch.isfinite(target.policy(obs).sample(noise).log_prob).all()
    # A checkpoint can be loaded under either spelling without weight conversion.
    snapshot["config"]["model"]["std_parameterization"] = (
        "legacy_scalar" if parameterization == "direct" else "direct"
    )
    alias_path = tmp_path / "alias.pt"
    torch.save(snapshot, alias_path)
    alias, _, _ = load_policy(alias_path, configure_runtime=False)
    other = alias.policy(obs).sample(noise)
    torch.testing.assert_close(other.raw, sample.raw, atol=0, rtol=0)
    torch.testing.assert_close(other.log_prob, sample.log_prob, atol=0, rtol=0)
