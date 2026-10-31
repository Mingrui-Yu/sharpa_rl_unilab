"""Marked frames exercise the real term and manager history/reset lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers.rng import ManagerRng

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


def test_gain_randomization_accepts_manager_and_numpy_rng():
    randomization = SharpaDomainRandomization.__new__(SharpaDomainRandomization)
    for rng in (ManagerRng(device="cpu").manual_seed(1), np.random.default_rng(1)):
        values = randomization._sample_split_around_one(rng, (0.5, 1.5), (2, 22))
        assert values.shape == (2, 22)
        assert np.isfinite(values).all()


def make_observations(*, order=("actor", "critic", "priv_info", "proprio_hist"), **params):
    n = 2
    raw = torch.zeros((n, 15))
    reads = []
    sensor = SimpleNamespace(names=TACTILE_SENSOR_NAMES, dimensions=(3,) * 5)
    data = SimpleNamespace(
        joint_pos=torch.zeros((n, 22)),
        root_link_pos_w=torch.zeros((n, 3)),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        actuator_ctrl_range=np.tile([-1.0, 1.0], (22, 1)),
    )
    entity = SimpleNamespace(
        data=data,
        root_body_name="object",
        find_joints_by_actuator_names=lambda _: (np.arange(22), HAND_JOINT_NAMES),
    )

    class ReadPlan:
        def joint_tensor_view(self, _):
            return SimpleNamespace(
                joint_pos=data.joint_pos, joint_vel=torch.zeros_like(data.joint_pos)
            )

        def body_tensor_view(self, _, names):
            assert tuple(names) == ("object",)
            return SimpleNamespace(
                pos_w=data.root_link_pos_w[:, None],
                quat_w=data.root_link_quat_w[:, None],
            )

        def sensor_tensor_views(self, _, names):
            reads.append(True)
            return SimpleNamespace(
                names=tuple(names),
                values={
                    name: raw[:, index * 3 : (index + 1) * 3] for index, name in enumerate(names)
                },
            )

    class Scene(dict):
        read_plan = ReadPlan()

        def bind_sensor_data(self, _):
            return sensor

    reset = SharpaHandObjectReset.__new__(SharpaHandObjectReset)
    reset.object_pos_anchor = torch.zeros((n, 3))
    dr = SharpaDomainRandomization.__new__(SharpaDomainRandomization)
    dr.kp = dr.kd = torch.ones((n, 22))
    dr._include_friction = True
    dr._include_gravity = False
    dr.friction_scale = torch.ones((n, 1))
    dr.mass = torch.full((n,), 2.0)
    dr.com_offset = torch.tensor([3.0, 4.0, 5.0]).repeat(n, 1)
    dr.scale = torch.full((n, 1), 6.0)
    action = SharpaIncrementalPositionAction.__new__(SharpaIncrementalPositionAction)
    action._target = torch.zeros((n, 22))
    events = {"reset": reset, "dr": dr}
    env = SimpleNamespace(
        num_envs=n,
        device=torch.device("cpu"),
        common_step_counter=0,
        step_dt=0.05,
        rng=ManagerRng(device="cpu").manual_seed(42),
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
    torch.testing.assert_close(obs["priv_info"], critic[:, -1, 49:])
    torch.testing.assert_close(obs["proprio_hist"][:, -3:], actor)
    assert obs["priv_info"].shape == (2, 9)


def test_clean_critic_ignores_actor_corruption_and_call_order():
    order = ("critic", "proprio_hist", "priv_info", "actor")
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
    torch.testing.assert_close(actual["critic"], expected["critic"])
    torch.testing.assert_close(actual["priv_info"], expected["priv_info"])
    assert not np.array_equal(actual["actor"], expected["actor"])
    torch.testing.assert_close(actual["proprio_hist"][:, -3:], actual["actor"].reshape(2, 3, 49))
    np.testing.assert_allclose(actual["critic"].reshape(2, 3, 58)[:, -1, 44:49], 4.0)
    source = noisy[1].get_term_cfg("actor", "frame").func
    before = len(noisy[-1])
    frame = source.frame(noisy[0], noisy=True).clone()
    torch.testing.assert_close(source.frame(noisy[0], noisy=True), frame)
    assert len(noisy[-1]) == before


def test_partial_reset_backfills_only_new_episode_after_state_commit():
    env, manager, data, raw, action, _ = make_observations(joint_noise=0.02)
    manager.compute(update_history=True)
    for value in (1.0, 2.0, 3.0):
        mark_step(env, data, raw, action, value)
        terminal = manager.compute(update_history=True)
    terminal = {key: value.cpu().numpy().copy() for key, value in terminal.items()}
    ids = torch.tensor([0], dtype=torch.int64)
    manager.reset(ids)
    # Writes occur after term.reset(), as in the manager environment transaction.
    data.joint_pos[0] = 10.0
    data.root_link_pos_w[0] = 11.0
    raw[0, ::3] = 12.0
    action.target[0] = 13.0
    reset = {
        key: value[ids].cpu().numpy().copy()
        for key, value in manager.compute(update_history=True, env_ids=ids).items()
    }
    actor = reset["actor"].reshape(1, 3, 49)
    hist = reset["proprio_hist"]
    torch.testing.assert_close(hist, np.repeat(hist[:, -1:], 30, axis=1))
    torch.testing.assert_close(actor, hist[:, -3:])
    critic = reset["critic"].reshape(1, 3, 58)
    torch.testing.assert_close(critic, np.repeat(critic[:, -1:], 3, axis=1))
    np.testing.assert_allclose(critic[:, :, :22], 10.0)
    np.testing.assert_allclose(critic[:, :, 44:49], 12.0)
    np.testing.assert_allclose(reset["priv_info"][0, :3], 11.0)
    # A non-updating query must preserve the unaffected environment's history.
    current = manager.compute()
    for key in terminal:
        torch.testing.assert_close(current[key][1], torch.as_tensor(terminal[key][1]))
    np.testing.assert_allclose(terminal["priv_info"][0, :3], 4.0)
    mark_step(env, data, raw, action, 4.0)
    obs = manager.compute(update_history=True)
    torch.testing.assert_close(obs["proprio_hist"][:, -3:], obs["actor"].reshape(2, 3, 49))


def test_clean_tactile_uses_vector_norm_and_deterministic_clip():
    env, manager, _, raw, _, _ = make_observations(
        tactile_force_clip_max=4.0,
        contact_smoothing=0.0,
        binary_contact=True,
    )
    raw[:] = torch.tensor([3.0, 4.0, 0.0]).repeat(5)
    obs = manager.compute(update_history=True)
    np.testing.assert_allclose(obs["critic"].reshape(2, 3, 58)[:, :, 44:49], 4.0)
    np.testing.assert_allclose(obs["actor"].reshape(2, 3, 49)[:, :, 44:], 0.0)


def test_observation_hot_path_has_no_host_conversion(monkeypatch):
    env, manager, data, raw, action, _ = make_observations(joint_noise=0.02)
    manager.compute(update_history=True)
    mark_step(env, data, raw, action, 1.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("observation hot path requested a host conversion")

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms import compat

    monkeypatch.setattr(compat, "host_numpy", forbidden)
    monkeypatch.setattr(compat, "manager_tensor", forbidden)
    monkeypatch.setattr(compat, "rng_numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    obs = manager.compute(update_history=True)
    assert all(isinstance(value, torch.Tensor) for value in obs.values())

    from sharpa_rl_unilab.tasks.sharpa_inhand.terms import rewards
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.termination import SharpaDropTermination

    source = manager.get_term_cfg("actor", "frame").func
    source.task_state.reset_height_lower = torch.full((env.num_envs,), -1.0)
    source.task_state.reset_height_upper = torch.full((env.num_envs,), 100.0)
    source.task_state.rotation_axis = torch.tensor([0.0, 0.0, 1.0]).repeat(env.num_envs, 1)
    termination = SharpaDropTermination.__new__(SharpaDropTermination)
    termination.observation = source
    termination.dropped = torch.zeros(env.num_envs, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(
        get_term_cfg=lambda _: SimpleNamespace(func=termination)
    )
    assert not termination(env).any()
    reward_values = (
        rewards.rotate_reward(env, "dropped"),
        rewards.object_linear_velocity_l1(env, "dropped"),
        rewards.hand_pose_deviation_l2(env, "dropped"),
        rewards.estimated_torque_l2(env, "dropped"),
        rewards.estimated_work_l2(env, "dropped"),
        rewards.object_position_reward(env, "dropped"),
    )
    assert all(isinstance(value, torch.Tensor) for value in reward_values)


def test_torch_quat_velocity_matches_legacy_numpy_snapshot():
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.geometry import (
        np_quat_angular_velocity_from_pair,
        torch_quat_angular_velocity_from_pair,
    )

    rng = np.random.default_rng(19)
    previous = rng.normal(size=(16, 4)).astype(np.float32)
    current = previous + rng.normal(scale=0.02, size=(16, 4)).astype(np.float32)
    previous /= np.linalg.norm(previous, axis=1, keepdims=True)
    current /= np.linalg.norm(current, axis=1, keepdims=True)
    expected = np_quat_angular_velocity_from_pair(current, previous, 0.05)
    actual = torch_quat_angular_velocity_from_pair(
        torch.from_numpy(current), torch.from_numpy(previous), 0.05
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected).to(actual.dtype), rtol=1e-5, atol=1e-6
    )


@pytest.mark.slow
def test_mujoco_autoreset_and_simultaneous_termination(monkeypatch):
    from sharpa_rl_unilab.cli import compose_config
    from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv, transition_next
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.termination import SharpaDropTermination

    cfg = compose_config("ppo", "mujoco", ["+env.max_episode_seconds=0.1"])
    env = SharpaTeacherEnv(cfg, num_envs=8)
    terminal = {}
    original_reset = env.env.reset

    def capture_reset(*args, **kwargs):
        captured = env.state.final_observation if env.state is not None else None
        terminal.update(captured if captured is not None else env._observations())
        return original_reset(*args, **kwargs)

    try:
        env.reset(seed=7)
        monkeypatch.setattr(env.env, "reset", capture_reset)
        for _ in range(2):
            state = env.step(torch.zeros((8, 22)))
        assert torch.any(state.truncated)
        assert terminal
        done = state.terminated | state.truncated
        for key, values in state.obs.items():
            assert torch.isfinite(values).all()
            torch.testing.assert_close(state.final_observation[key][done], terminal[key][done])
            if key != "priv_info":
                length = 30 if key == "proprio_hist" else 3
                frames = values[done].reshape(int(done.sum()), length, -1)
                torch.testing.assert_close(frames, frames[:, -1:].repeat(1, length, 1))
        assert not {"priv_info", "proprio_hist"}.intersection(state.info)
        torch.testing.assert_close(
            state.obs["obs"], state.obs["proprio_hist"][:, -3:].reshape(8, 147)
        )
        source = env.termination_manager.get_term_cfg("dropped").func.observation
        done_host = done.detach().cpu()
        np.testing.assert_allclose(source.dof_vel[done_host].cpu(), 0.0)
        np.testing.assert_allclose(source.object_linvel[done_host].cpu(), 0.0)
        if "critic" in state.obs:
            clean = state.obs["critic"].reshape(8, 3, 58)[:, -1]
            frame = torch.as_tensor(
                source.frame(env, noisy=False), dtype=clean.dtype, device=clean.device
            )
            privilege = torch.as_tensor(source.priv_info, dtype=clean.dtype, device=clean.device)
            torch.testing.assert_close(clean[:, :49], frame)
            torch.testing.assert_close(clean[:, 49:], privilege)
        # The following episode ends in a drop and timeout on the same step.
        env.step(torch.zeros((8, 22)))
        original_drop = SharpaDropTermination.__call__

        def force_drop(self, env, **kwargs):
            original_drop(self, env, **kwargs)
            self.dropped.fill_(True)
            return self.dropped

        monkeypatch.setattr(SharpaDropTermination, "__call__", force_drop)
        state = env.step(torch.zeros((8, 22)))
        assert state.terminated.all() and not state.truncated.any()
        for key, values in transition_next(state).items():
            torch.testing.assert_close(values, state.final_observation[key])
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU residency is a CUDA contract")
def test_ppo_collection_hot_path_remains_gpu_resident():
    from sharpa_rl_unilab.cli import compose_config
    from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
    from sharpa_rl_unilab.training.policy import make_models
    from sharpa_rl_unilab.training.rollouts import collect

    cfg = compose_config("ppo", "mujoco", ["training.num_envs=8"])
    actor, _, _ = make_models(cfg, cfg.training.device)
    env = SharpaTeacherEnv(cfg, 8)
    try:
        obs, _ = env.reset(seed=17)
        assert env.env.device.type == "cuda"
        assert all(value.is_cuda for value in obs.values())
        raw, last_obs = collect(env, obs, actor, 2, cfg.training.device, metrics={})
        assert all(value.is_cuda for value in last_obs.values())
        assert raw
        for key, value in raw.items():
            assert isinstance(value, torch.Tensor), key
            assert value.is_cuda, key
            assert value.dtype == (
                torch.bool if key in {"terminated", "truncated"} else torch.float32
            ), key
    finally:
        env.close()
