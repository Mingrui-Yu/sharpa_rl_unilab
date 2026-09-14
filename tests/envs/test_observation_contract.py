"""Marked frames exercise the real term and manager history/reset lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest
from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

from sharpa_rl_unilab.tasks.sharpa_inhand.terms.action import SharpaIncrementalPositionAction
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.constants import (
    HAND_JOINT_NAMES,
    TACTILE_SENSOR_NAMES,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.observation import (
    SharpaCriticObservation,
    SharpaPrivilegedObservation,
    SharpaProprioObservation,
    SharpaRotationObservation,
)
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.randomization import SharpaDomainRandomization
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.reset import SharpaHandObjectReset


def make_observations(*, order=("actor", "critic", "priv_info", "proprio_hist"), **params):
    n = 2
    raw = np.zeros((n, 15))
    reads = []

    def read():
        reads.append(True)
        return raw

    sensor = SimpleNamespace(names=TACTILE_SENSOR_NAMES, dimensions=(3,) * 5, read=read)
    data = SimpleNamespace(
        joint_pos=np.zeros((n, 22)),
        root_link_pos_w=np.zeros((n, 3)),
        root_link_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        actuator_ctrl_range=np.tile([-1.0, 1.0], (22, 1)),
    )
    entity = SimpleNamespace(
        data=data,
        find_joints_by_actuator_names=lambda _: (np.arange(22), HAND_JOINT_NAMES),
    )

    class Scene(dict):
        def bind_sensor_data(self, _):
            return sensor

    reset = SharpaHandObjectReset.__new__(SharpaHandObjectReset)
    reset.object_pos_anchor = np.zeros((n, 3))
    dr = SharpaDomainRandomization.__new__(SharpaDomainRandomization)
    dr.kp = dr.kd = np.ones((n, 22))
    dr._include_friction = True
    dr._include_gravity = False
    dr.friction_scale = np.ones((n, 1))
    dr.mass = np.full(n, 2.0)
    dr.com_offset = np.tile([3.0, 4.0, 5.0], (n, 1))
    dr.scale = np.full((n, 1), 6.0)
    action = SharpaIncrementalPositionAction.__new__(SharpaIncrementalPositionAction)
    action._target = np.zeros((n, 22))
    events = {"reset": reset, "dr": dr}
    env = SimpleNamespace(
        num_envs=n,
        common_step_counter=0,
        step_dt=0.05,
        rng=np.random.default_rng(42),
        scene=Scene(robot=entity),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=events[name])),
        action_manager=SimpleNamespace(get_term=lambda _: action),
    )
    source_params = dict(
        entity_name="robot",
        action_name="hand",
        event_state_name="reset",
        domain_randomization_name="dr",
        joint_noise=0.0,
        contact_smoothing=1.0,
        contact_latency=0.0,
        clip_obs=0.0,
        tactile_force_clip_max=0.0,
    )
    source_params.update(params)
    view_params = dict(observation_group="actor", observation_term="frame")
    groups = {
        "actor": ObservationGroupCfg(
            terms={
                "frame": ObservationTermCfg(func=SharpaRotationObservation, params=source_params)
            },
            history_length=3,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=SharpaCriticObservation, params=view_params)},
            history_length=3,
            flatten_history_dim=True,
        ),
        "priv_info": ObservationGroupCfg(
            terms={
                "frame": ObservationTermCfg(func=SharpaPrivilegedObservation, params=view_params)
            },
        ),
        "proprio_hist": ObservationGroupCfg(
            terms={"frame": ObservationTermCfg(func=SharpaProprioObservation, params=view_params)},
            history_length=30,
            flatten_history_dim=False,
        ),
    }
    manager = ObservationManager({name: groups[name] for name in order}, env)
    env.observation_manager = manager
    manager.reset()
    return env, manager, data, raw, action, reads


def mark_step(env, data, raw, action, value):
    env.common_step_counter += 1
    data.joint_pos[:] = value
    data.root_link_pos_w[:] = value + 1
    raw[:] = 0.0
    raw[:, ::3] = value + 2
    action.target[:] = value + 3


def test_marked_fields_history_and_explicit_privilege():
    env, manager, data, raw, action, _ = make_observations()
    initial = manager.compute(update_history=True)
    assert initial["proprio_hist"].shape == (2, 30, 49)
    for value in range(1, 5):
        mark_step(env, data, raw, action, value)
        obs = manager.compute(update_history=True)
    actor = obs["actor"].reshape(2, 3, 49)
    critic = obs["critic"].reshape(2, 3, 58)
    for index, value in enumerate((2, 3, 4)):
        np.testing.assert_allclose(actor[:, index, :22], value)
        np.testing.assert_allclose(actor[:, index, 22:44], value + 3)
        np.testing.assert_allclose(actor[:, index, 44:], value + 2)
        np.testing.assert_allclose(critic[:, index, :49], actor[:, index])
        expected = np.tile([value + 1] * 3 + [1, 2, 3, 4, 5, 6], (2, 1))
        np.testing.assert_allclose(critic[:, index, 49:], expected)
    np.testing.assert_array_equal(obs["priv_info"], critic[:, -1, 49:])
    np.testing.assert_array_equal(obs["proprio_hist"][:, -3:], actor)
    assert obs["priv_info"].shape == (2, 9)


@pytest.mark.parametrize(
    "order",
    [
        ("actor", "critic", "priv_info", "proprio_hist"),
        ("critic", "proprio_hist", "priv_info", "actor"),
    ],
)
def test_clean_critic_ignores_actor_corruption_and_call_order(order):
    clean = make_observations()
    noisy = make_observations(
        order=order,
        joint_noise=0.5,
        contact_smoothing=0.0,
        contact_latency=1.0,
        binary_contact=True,
        contact_sensor_noise=1.0,
    )
    for bundle in (clean, noisy):
        env, manager, data, raw, action, reads = bundle
        manager.compute(update_history=True)
        for value in (1.0, 2.0):
            mark_step(env, data, raw, action, value)
            before = len(reads)
            manager.compute(update_history=True)
            assert len(reads) == before + 1
    expected, actual = clean[1].compute(), noisy[1].compute()
    np.testing.assert_array_equal(actual["critic"], expected["critic"])
    np.testing.assert_array_equal(actual["priv_info"], expected["priv_info"])
    assert not np.array_equal(actual["actor"], expected["actor"])
    np.testing.assert_array_equal(actual["proprio_hist"][:, -3:], actual["actor"].reshape(2, 3, 49))
    np.testing.assert_array_equal(actual["critic"].reshape(2, 3, 58)[:, -1, 44:49], 4.0)
    source = noisy[1].get_term_cfg("actor", "frame").func
    before = len(noisy[-1])
    frame = source.frame(noisy[0], noisy=True).copy()
    np.testing.assert_array_equal(source.frame(noisy[0], noisy=True), frame)
    assert len(noisy[-1]) == before


def test_partial_reset_backfills_only_new_episode_after_state_commit():
    env, manager, data, raw, action, _ = make_observations(joint_noise=0.02)
    manager.compute(update_history=True)
    for value in (1.0, 2.0, 3.0):
        mark_step(env, data, raw, action, value)
        terminal = manager.compute(update_history=True)
    terminal = {key: value.copy() for key, value in terminal.items()}
    ids = np.array([0], dtype=np.int32)
    manager.reset(ids)
    # Writes occur after term.reset(), as in the manager environment transaction.
    data.joint_pos[0] = 10.0
    data.root_link_pos_w[0] = 11.0
    raw[0, ::3] = 12.0
    action.target[0] = 13.0
    reset = manager.compute(update_history=True, env_ids=ids)
    actor = reset["actor"].reshape(1, 3, 49)
    hist = reset["proprio_hist"]
    np.testing.assert_array_equal(hist, np.repeat(hist[:, -1:], 30, axis=1))
    np.testing.assert_array_equal(actor, hist[:, -3:])
    critic = reset["critic"].reshape(1, 3, 58)
    np.testing.assert_array_equal(critic, np.repeat(critic[:, -1:], 3, axis=1))
    np.testing.assert_allclose(critic[:, :, :22], 10.0)
    np.testing.assert_array_equal(critic[:, :, 44:49], 12.0)
    np.testing.assert_array_equal(reset["priv_info"][0, :3], 11.0)
    # A non-updating query must preserve the unaffected environment's history.
    current = manager.compute()
    for key in terminal:
        np.testing.assert_array_equal(current[key][1], terminal[key][1])
    np.testing.assert_array_equal(terminal["priv_info"][0, :3], 4.0)
    mark_step(env, data, raw, action, 4.0)
    obs = manager.compute(update_history=True)
    np.testing.assert_array_equal(obs["proprio_hist"][:, -3:], obs["actor"].reshape(2, 3, 49))


def test_clean_tactile_uses_vector_norm_and_deterministic_clip():
    env, manager, _, raw, _, _ = make_observations(
        tactile_force_clip_max=4.0,
        contact_smoothing=0.0,
        binary_contact=True,
    )
    raw[:] = np.tile([3.0, 4.0, 0.0], 5)
    obs = manager.compute(update_history=True)
    np.testing.assert_array_equal(obs["critic"].reshape(2, 3, 58)[:, :, 44:49], 4.0)
    np.testing.assert_array_equal(obs["actor"].reshape(2, 3, 49)[:, :, 44:], 0.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "algo,profile", [("ppo", None), ("appo", None), ("appo", "hora"), ("flashsac", None)]
)
def test_mujoco_observations_survive_timeout_autoreset(algo, profile, monkeypatch):
    from pathlib import Path

    from unilab.base.config_adapter import BackendAdapter, create_env

    from sharpa_rl_unilab.cli import compose_config

    cfg = compose_config(algo, "mujoco", [], profile=profile)
    root = Path(__file__).resolve().parents[2]
    overrides = BackendAdapter(cfg, root_dir=root, algo_name=algo).build_task_env_cfg_override()
    overrides["max_episode_seconds"] = 0.1
    env = create_env(cfg, num_envs=8, env_cfg_override=overrides)
    terminal = {}
    original_reset = env.reset

    def capture_reset(*args, **kwargs):
        terminal.update({key: values.copy() for key, values in env.state.obs.items()})
        return original_reset(*args, **kwargs)

    try:
        env.reset(seed=7)
        monkeypatch.setattr(env, "reset", capture_reset)
        for _ in range(2):
            state = env.step(np.zeros((8, 22)))
        assert np.any(state.truncated)
        assert terminal
        done = state.terminated | state.truncated
        for key, values in state.obs.items():
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(state.final_observation[key][done], terminal[key][done])
            frames = values[done].reshape(-1, 3, values.shape[1] // 3)
            np.testing.assert_array_equal(frames, np.repeat(frames[:, -1:], 3, axis=1))
        source = env.termination_manager.get_term_cfg("dropped").func.observation
        np.testing.assert_allclose(source.dof_vel[done], 0.0)
        np.testing.assert_allclose(source.object_linvel[done], 0.0)
        if "critic" in state.obs:
            clean = state.obs["critic"].reshape(8, 3, 58)[:, -1]
            np.testing.assert_array_equal(clean[:, :49], source.frame(env, noisy=False))
            np.testing.assert_array_equal(clean[:, 49:], source.priv_info)
    finally:
        env.close()
