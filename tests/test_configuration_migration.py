import copy

import pytest
import torch
from omegaconf import OmegaConf

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.configuration import migrate_checkpoint_config
from sharpa_rl_unilab.training.teacher_runtime import (
    load_policy,
    make_models,
    save_teacher,
    training_budget,
)


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
def test_training_budget_validation_and_rounding(algo):
    cfg = compose_config(algo, "mujoco", ["training.num_envs=8", "algo.max_iterations=2"])
    assert training_budget(cfg) == ("policy_version", 2)
    for iterations, transitions in [(2, 17), (None, None), (0, None), (-1, None)]:
        cfg.algo.max_iterations, cfg.training.max_transitions = iterations, transitions
        with pytest.raises(ValueError):
            training_budget(cfg)
    cfg.algo.max_iterations, cfg.training.max_transitions = None, 17
    if algo == "flashsac":
        with pytest.raises(ValueError, match="FlashSAC requires"):
            training_budget(cfg)
    else:
        assert training_budget(cfg) == ("received", 24)
        cfg.training.max_transitions = 0
        with pytest.raises(ValueError, match="positive"):
            training_budget(cfg)
    cfg.algo.max_iterations, cfg.training.max_transitions = 2, None
    cfg.algo.save_interval = -1
    with pytest.raises(ValueError, match="save_interval"):
        training_budget(cfg)


def legacy_config(algo):
    cfg = OmegaConf.to_container(
        compose_config(algo, "mujoco", ["training.num_envs=8"]), resolve=True
    )
    cfg["hardware"] = dict(
        num_envs=cfg["training"].pop("num_envs"),
        device="cpu",
        collector_device="cpu",
        torch_threads=2 if algo == "ppo" else None,
    )
    cfg["training"]["device"] = None
    cfg["budget"] = dict(transitions=17, log_every=19)
    cfg["training"].pop("max_transitions")
    cfg["algo"]["max_iterations"] = None
    cfg["algo"]["num_envs"] = "${hardware.num_envs}"
    cfg["distillation"].pop("log_every")
    cfg["algo"].pop("collector_device", None)
    cfg["model"]["critic_normalization"] = False
    cfg["env"]["events"]["persistent_force"] = None
    return cfg


@pytest.mark.parametrize("algo", ["ppo", "appo"])
@pytest.mark.parametrize(
    "saved_mode,expected",
    [
        ("legacy", "log_epsilon"),
        ("log_epsilon", "log_epsilon"),
        ("exact", "exact"),
        (None, "exact"),
    ],
)
def test_checkpoint_kl_mode_migration_preserves_policy(tmp_path, algo, saved_mode, expected):
    cfg = compose_config(algo, "mujoco", [])
    actor, critic, _ = make_models(cfg, "cpu")
    if saved_mode is None:
        del cfg.algo.algorithm.kl_mode
    else:
        cfg.algo.algorithm.kl_mode = saved_mode
    path = tmp_path / "teacher.pt"
    from types import SimpleNamespace

    save_teacher(
        path,
        cfg,
        actor,
        critic,
        SimpleNamespace(optimizer=torch.optim.Adam(actor.parameters())),
        {"received": 0},
        0,
    )
    restored, migrated, _ = load_policy(path, configure_runtime=False)
    assert restored.kl_mode == expected
    assert migrated.algo.algorithm.get("kl_mode", "exact") == expected
    torch.testing.assert_close(restored.state_dict(), actor.state_dict(), atol=0, rtol=0)
    assert cfg.algo.algorithm.get("kl_mode") == saved_mode


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
def test_migration_preserves_checkpoint_semantics(algo):
    old = legacy_config(algo)
    before = copy.deepcopy(old)
    cfg = migrate_checkpoint_config(old)
    assert old == before
    assert "hardware" not in cfg and "budget" not in cfg
    assert cfg.training.num_envs == cfg.algo.num_envs == 8
    assert cfg.training.device == "cpu" and cfg.training.max_transitions == 17
    assert cfg.distillation.log_every == 19
    assert cfg.model == old["model"] and cfg.env == old["env"]
    if algo == "ppo":
        assert cfg.training.torch_threads.learner_num_threads == 2
    elif algo == "appo":
        assert cfg.algo.collector_device == "cpu" and not cfg.training.torch_threads.enabled
    else:
        assert cfg.training.torch_threads == old["training"]["torch_threads"]
    assert migrate_checkpoint_config(cfg) == cfg


def test_legacy_flash_checkpoint_loads_without_enabling_q_normalization(tmp_path):
    cfg = OmegaConf.create(legacy_config("flashsac"))
    actor, critic, learner = make_models(cfg, "cpu")
    path = tmp_path / "legacy.pt"
    save_teacher(path, cfg, actor, critic, learner, {"received": 0}, 0)
    restored, migrated, _ = load_policy(path)
    assert not migrated.model.critic_normalization
    assert migrated.training.max_transitions == 17
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    _, old_q, old_learner = make_models(migrated, "cpu")
    old_learner.load_state_dict(learner.get_state_dict())
    obs, actions = torch.randn(3, 174), torch.randn(3, 22)
    torch.testing.assert_close(
        old_q(obs, actions, training=False)[0], critic(obs, actions, training=False)[0]
    )


def test_old_noise_config_keeps_custom_duration_distribution():
    cfg = OmegaConf.to_container(compose_config("flashsac", "mujoco", []), resolve=True)
    del cfg["algo"]["exploration"]
    cfg["algo"]["algo_params"].update(actor_noise_zeta_mu=0.0, actor_noise_zeta_max=3)
    migrated = migrate_checkpoint_config(cfg)
    assert migrated.algo.exploration.noise_zeta_mu == 0.0
    assert migrated.algo.exploration.noise_zeta_max == 3
    assert "actor_noise_zeta_mu" not in migrated.algo.algo_params
    assert migrate_checkpoint_config(migrated) == migrated


@pytest.mark.parametrize(
    "override", ["model.std_parameterization=legacy_scalar", "~model.std_parameterization"]
)
def test_legacy_parameterization_cannot_be_selected_for_new_training(override):
    cfg = compose_config("ppo", "mujoco", [override])
    with pytest.raises(ValueError, match="checkpoint-only"):
        make_models(cfg, "cpu")
