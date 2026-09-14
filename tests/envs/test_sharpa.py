from __future__ import annotations

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks import sharpa_inhand


def _materialize(algo: str):
    registry.ensure_registries()
    cfg = compose_config(algo, "mujoco", [])
    overrides = BackendAdapter(cfg, root_dir=".", algo_name=algo).build_task_env_cfg_override()
    return cfg, registry.materialize_env_config(str(cfg.training.task_name)), overrides


def test_manager_config_materializes_typed_terms() -> None:
    expected_groups = {
        algo: {"actor", "critic", "priv_info", "proprio_hist"}
        for algo in ("ppo", "appo", "flashsac")
    }
    for algo, groups in expected_groups.items():
        _, env_cfg, overrides = _materialize(algo)
        assert isinstance(env_cfg, sharpa_inhand.config.SharpaInhandRotationCfg)
        registry.apply_cfg_overrides(env_cfg, overrides)
        assert set(env_cfg.observations) == groups
        assert set(env_cfg.actions) == {"hand"}
        assert set(env_cfg.events) == {
            "reset_scene_to_default",
            "reset",
            "domain_randomization",
            "persistent_force",
        }
        assert set(env_cfg.terminations) == {"dropped", "time_out"}
        assert len(env_cfg.rewards) == 6
        assert env_cfg.scene is not None and env_cfg.scene.entities


def test_fixed_variant_plan_is_immutable_and_complete() -> None:
    _, env_cfg, overrides = _materialize("appo")
    registry.apply_cfg_overrides(env_cfg, overrides)
    assert env_cfg.fixed_model_variants is not None
    assert len(env_cfg.fixed_model_variants.variants) == 8
    assert env_cfg.scene is not None
    assert env_cfg.scene.entities
