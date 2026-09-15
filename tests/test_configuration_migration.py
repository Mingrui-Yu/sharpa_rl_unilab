import copy
import json

import pytest
import torch
from omegaconf import OmegaConf

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.configuration import configure_threads, migrate_checkpoint_config
from sharpa_rl_unilab.training.teacher_runtime import load_policy, make_models, save_teacher


def legacy_config(algo):
    cfg = OmegaConf.to_container(compose_config(algo, "mujoco", []), resolve=True)
    training = cfg["training"]
    cfg["hardware"] = {
        "num_envs": training.pop("num_envs"),
        "device": "cpu",
        "collector_device": "cpu",
        "torch_threads": None if algo in ("appo", "flashsac") else 2,
    }
    cfg["budget"] = {
        "transitions": training.pop("max_transitions"),
        "checkpoint": cfg["algo"].pop("checkpoint"),
        "save_every": None if algo == "appo" else 1000000,
        "evaluate_every": 100,
        "log_every": 17,
        "async_queue_size": cfg["algo"].pop("async_queue_size", 3),
    }
    training["device"] = None
    if algo != "flashsac":
        training.pop("torch_threads")
    else:
        training["torch_threads"]["learner_num_threads"] = "auto"
        cfg["model"]["critic_normalization"] = False
    cfg["algo"].pop("collector_device", None)
    cfg["distillation"].pop("log_every")
    cfg["algo"]["num_envs"] = "${hardware.num_envs}"
    cfg["env"]["events"]["persistent_force"] = None
    return cfg


@pytest.mark.parametrize("algo", ["appo", "ppo", "flashsac"])
def test_migration_preserves_saved_semantics_and_does_not_mutate_source(algo):
    old = legacy_config(algo)
    before = copy.deepcopy(old)
    cfg = migrate_checkpoint_config(old)
    assert old == before
    assert "budget" not in cfg and "hardware" not in cfg
    assert cfg.algo.num_envs == cfg.training.num_envs == 2048
    assert cfg.training.device == "cpu"
    assert cfg.distillation.log_every == 17
    assert cfg.env.events.persistent_force is None
    assert cfg.model.critic_normalization == (algo != "flashsac")
    if algo == "appo":
        assert cfg.algo.collector_device == "cpu"
        assert cfg.algo.async_queue_size == 4
        assert cfg.training.torch_threads.enabled is False
    elif algo == "ppo":
        assert cfg.training.torch_threads.learner_num_threads == 2
        assert cfg.training.torch_threads.collector_num_threads == 2
        assert (
            cfg.training.torch_threads.learner_num_interop_threads
            == torch.get_num_interop_threads()
        )
        assert cfg.training.torch_threads.set_env_vars is False
    else:
        assert cfg.training.torch_threads.learner_num_threads == "auto"
    assert migrate_checkpoint_config(OmegaConf.to_container(cfg, resolve=True)) == cfg


def test_legacy_device_override_and_disabled_threads():
    old = legacy_config("appo")
    old["training"]["device"] = "cuda:1"
    old["hardware"]["collector_device"] = None
    cfg = migrate_checkpoint_config(old)
    assert cfg.training.device == "cuda:1"
    assert cfg.algo.collector_device is None
    before = torch.get_num_threads(), torch.get_num_interop_threads()
    configure_threads(cfg)
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == before


def test_legacy_flash_checkpoint_keeps_unnormalized_q_and_loads_policy(tmp_path):
    old = legacy_config("flashsac")
    cfg = OmegaConf.create(old)
    actor, critic, learner = make_models(cfg, "cpu")
    assert not hasattr(critic, "obs_normalizer")
    path = tmp_path / "legacy.pt"
    save_teacher(path, cfg, actor, critic, learner, {"received": 0}, 0)
    restored, migrated, _ = load_policy(path)
    assert "budget" not in migrated and "hardware" not in migrated
    assert migrated.model.critic_normalization is False
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    _, old_q, old_learner = make_models(migrated, "cpu")
    old_learner.load_state_dict(learner.get_state_dict())
    obs, actions = torch.randn(3, 174), torch.randn(3, 22)
    torch.testing.assert_close(
        old_q(obs, actions, training=False)[0], critic(obs, actions, training=False)[0]
    )


@pytest.mark.slow
def test_student_preserves_legacy_log_and_save_triggers(tmp_path):
    from sharpa_rl_unilab.training.student_runtime import train_student

    cfg = OmegaConf.create(legacy_config("flashsac"))
    actor, critic, learner = make_models(cfg, "cpu")
    path = tmp_path / "teacher.pt"
    save_teacher(path, cfg, actor, critic, learner, {"received": 0}, 0)
    final = train_student(
        path,
        overrides=[
            "training.device=cpu",
            "training.torch_threads.learner_num_threads=2",
            "training.logger=no_print",
            "distillation.num_envs=8",
            "distillation.transitions=48",
            "distillation.save_every=16",
            f"training.log_dir={tmp_path / 'student'}",
        ],
    )
    rows = [json.loads(line) for line in (final.parent / "metrics.jsonl").read_text().splitlines()]
    assert [row["collected"] for row in rows] == [24, 48]
    assert {p.name for p in final.parent.glob("student_*.pt")} == {
        "student_16.pt",
        "student_32.pt",
        "student_48.pt",
        "student_final.pt",
    }
    _, saved_cfg, snapshot = load_policy(final)
    assert saved_cfg.distillation.log_every == 17
    assert "budget" not in snapshot["config"] and "hardware" not in snapshot["config"]
    assert snapshot["counters"]["collected"] == 48
