"""Contract tests for the HORA-SAC off-policy actor adapter registration."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture()
def adapter():
    # Importing the sac module runs the module-level registration exactly once.
    from uni_rl.offpolicy.actor_adapter import get_offpolicy_actor_adapter

    import sharpa_rl_unilab.algos.hora.sac  # noqa: F401

    resolved = get_offpolicy_actor_adapter("hora_sac")
    assert resolved is not None
    return resolved


def test_hora_sac_adapter_is_registered(adapter) -> None:
    assert adapter.algo_type == "hora_sac"
    assert adapter.build_actor is not None
    assert adapter.sample_actions is not None
    assert adapter.resolve_priv_info is not None
    assert adapter.actor_context_from_obs is not None


def test_hora_sac_adapter_build_actor_matches_actor_factory_contract(adapter) -> None:
    from sharpa_rl_unilab.algos.hora.sac_models import HoraSACActor

    with pytest.raises(ValueError, match="priv_info_dim"):
        adapter.build_actor(
            obs_dim=5,
            action_dim=2,
            actor_hidden_dim=16,
            use_layer_norm=False,
            device="cpu",
            priv_info_dim=None,
        )

    actor = adapter.build_actor(
        obs_dim=5,
        action_dim=2,
        actor_hidden_dim=16,
        use_layer_norm=False,
        device="cpu",
        priv_info_dim=3,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
    )

    assert isinstance(actor, HoraSACActor)
    assert actor.obs_dim == 5
    assert actor.priv_info_dim == 3
    assert actor.action_dim == 2


def test_hora_sac_adapter_build_actor_via_uni_rl_actor_factory(adapter) -> None:
    from uni_rl.algos.common.actor_factory import build_actor

    from sharpa_rl_unilab.algos.hora.sac_models import HoraSACActor

    actor = build_actor(
        "hora_sac",
        obs_dim=5,
        action_dim=2,
        actor_hidden_dim=16,
        use_layer_norm=False,
        device="cpu",
        priv_info_dim=3,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
    )

    assert isinstance(actor, HoraSACActor)


def test_hora_sac_adapter_sample_actions_requires_priv_info(adapter) -> None:
    actor = adapter.build_actor(
        obs_dim=5,
        action_dim=2,
        actor_hidden_dim=16,
        use_layer_norm=False,
        device="cpu",
        priv_info_dim=3,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
    )
    obs = torch.zeros(6, 5)
    dones = torch.zeros(6)

    with pytest.raises(ValueError, match="priv_info_torch"):
        adapter.sample_actions(actor, obs, dones, None)

    actions = adapter.sample_actions(actor, obs, dones, torch.zeros(6, 3))

    assert actions.shape == (6, 2)


def test_hora_sac_adapter_sample_actions_via_uni_rl_worker(adapter) -> None:
    from uni_rl.offpolicy.worker import sample_offpolicy_actions

    actor = adapter.build_actor(
        obs_dim=5,
        action_dim=2,
        actor_hidden_dim=16,
        use_layer_norm=False,
        device="cpu",
        priv_info_dim=3,
    )
    actions = sample_offpolicy_actions(
        actor,
        "hora_sac",
        torch.zeros(4, 5),
        torch.zeros(4),
        torch.zeros(4, 3),
    )

    assert actions.shape == (4, 2)


def test_hora_sac_adapter_resolve_priv_info_from_critic_tail(adapter) -> None:
    obs_np = np.zeros((4, 5), dtype=np.float32)
    priv_np = np.arange(12, dtype=np.float32).reshape(4, 3)
    critic_np = np.concatenate([obs_np, priv_np], axis=1)

    resolved = adapter.resolve_priv_info(obs_np, critic_np, None)

    np.testing.assert_array_equal(resolved, priv_np)
    assert resolved.dtype == np.float32


def test_hora_sac_adapter_resolve_priv_info_prefers_critic_info(adapter) -> None:
    obs_np = np.zeros((4, 5), dtype=np.float32)
    critic_np = np.zeros((4, 8), dtype=np.float32)
    explicit = np.ones((4, 2), dtype=np.float32)

    resolved = adapter.resolve_priv_info(obs_np, critic_np, {"critic_info": explicit})

    np.testing.assert_array_equal(resolved, explicit)


def test_hora_sac_adapter_resolve_priv_info_raises_when_missing(adapter) -> None:
    obs_np = np.zeros((4, 5), dtype=np.float32)
    critic_np = np.zeros((4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="privileged info"):
        adapter.resolve_priv_info(obs_np, critic_np, None)


def test_hora_sac_adapter_resolve_priv_info_via_uni_rl_worker(adapter) -> None:
    from uni_rl.offpolicy.worker import resolve_offpolicy_actor_priv_info

    obs_np = np.zeros((4, 5), dtype=np.float32)
    priv_np = np.arange(12, dtype=np.float32).reshape(4, 3)
    critic_np = np.concatenate([obs_np, priv_np], axis=1)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="hora_sac",
        obs_np=obs_np,
        critic_np=critic_np,
        info=None,
    )

    np.testing.assert_array_equal(resolved, priv_np)


def test_hora_sac_adapter_actor_context_slices_obs_tail(adapter) -> None:
    obs_device = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    context = adapter.actor_context_from_obs(obs_device, 4)

    torch.testing.assert_close(context, obs_device[:, 4:])


def test_resolve_hora_sac_runtime_carries_actor_adapter_modules() -> None:
    from uni_rl.offpolicy.runtime import resolve_actor_adapter_modules

    from sharpa_rl_unilab.algos.hora.sac import resolve_hora_sac_runtime

    assert resolve_hora_sac_runtime({"runtime_impl": "hora_ppo"}) is None

    runtime = resolve_hora_sac_runtime(
        {"runtime_impl": "hora_sac", "actor": {"priv_info_embed_dim": 4}}
    )

    assert runtime is not None
    assert runtime.algo_type == "hora_sac"
    assert "sharpa_rl_unilab.algos.hora.sac" in runtime.actor_adapter_modules

    modules = resolve_actor_adapter_modules({}, runtime)
    assert "sharpa_rl_unilab.algos.hora.sac" in modules

    model_kwargs = runtime.build_model_kwargs(obs_dim=5, critic_obs_dim=8)
    assert model_kwargs["priv_info_dim"] == 3
    assert model_kwargs["priv_info_embed_dim"] == 4
