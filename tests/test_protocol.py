import pytest
from omegaconf import OmegaConf

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand import teacher_env
from sharpa_rl_unilab.training.configuration import migrate_checkpoint_config


@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("env.observations.actor.terms.frame.params.enable_tactile", False),
        ("env.events.domain_randomization.params.include_friction_scale", False),
        ("env.events.domain_randomization.params.include_gravity_direction", True),
        ("env.observations.actor.history_length", 4),
        ("env.observations.critic.flatten_history_dim", False),
        ("env.observations.proprio_hist.history_length", 20),
        ("env.observations.priv_info.history_length", 2),
        ("env.observations.priv_info.terms.current.history_length", 2),
        ("env.observations.actor.concatenate_terms", False),
    ],
)
def test_invalid_protocol_rejected_before_simulation(monkeypatch, algo, field, value):
    cfg = compose_config(algo, "mujoco", [])
    OmegaConf.update(cfg, field, value, force_add=True)
    monkeypatch.setattr(teacher_env, "create_env", lambda *a, **kw: pytest.fail("Built simulation"))
    with pytest.raises(ValueError, match="sharpa-hora-v2 requires") as error:
        teacher_env.SharpaTeacherEnv(cfg, 2)
    assert field in str(error.value) and "got" in str(error.value)
    with pytest.raises(ValueError, match="sharpa-hora-v2 requires"):
        compose_config(algo, "mujoco", [f"++{field}={str(value).lower()}"])


def test_legacy_unused_tactile_option_is_removed_without_changing_observations():
    cfg = compose_config("ppo", "mujoco", [])
    saved = OmegaConf.to_container(cfg, resolve=True)
    saved["env"]["observations"]["actor"]["terms"]["frame"]["params"]["disable_tactile_ids"] = [
        0,
        2,
    ]
    migrated = migrate_checkpoint_config(saved)
    assert migrated.env == cfg.env
