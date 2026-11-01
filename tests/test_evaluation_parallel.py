from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.managers.rng import ManagerRng

import sharpa_rl_unilab.training.evaluation as evaluation
from sharpa_rl_unilab.training.evaluation import (
    BatchedEvaluationRng,
    _install_evaluation_rng,
    _rollout_episode_batch,
    reset_record,
    set_step_rng,
)


class FakeEnv:
    def __init__(self, num_envs, done_after, scale=1.0):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.max_episode_length = 400
        self.step_dt = 0.05
        self.done_after = done_after
        self.scale = scale
        owner = ManagerRng(device="cpu")
        self.env = SimpleNamespace(rng=owner, _rng_owner=owner, seed=self.seed)
        self.obs = torch.zeros(num_envs, 1)
        self.steps = torch.zeros(num_envs, dtype=torch.int64)
        self.reset_term = SimpleNamespace(cache_row_indices=np.zeros(num_envs, dtype=np.int64))
        self.dr = SimpleNamespace(
            kp=np.zeros((num_envs, 2)),
            kd=np.zeros((num_envs, 2)),
            mass=np.zeros((num_envs, 1)),
            com_offset=np.zeros((num_envs, 3)),
            gravity=np.zeros((num_envs, 3)),
            friction_scale=np.zeros((num_envs, 1)),
            scale=np.full((num_envs, 1), scale),
        )
        terms = {
            "reset": SimpleNamespace(func=self.reset_term),
            "domain_randomization": SimpleNamespace(func=self.dr),
        }
        self.event_manager = SimpleNamespace(active_terms={}, get_term_cfg=lambda name: terms[name])
        self.frame = SimpleNamespace(
            object_angvel=np.zeros((num_envs, 3)),
            task_state=SimpleNamespace(rotation_axis=np.array([1.0, 0.0, 0.0])),
        )
        self.observation_manager = SimpleNamespace(
            get_term_cfg=lambda group, name: SimpleNamespace(func=self.frame)
        )

    def seed(self, seed):
        self.env._rng_owner.manual_seed(seed)

    def reset(self, env_indices=None, *, seed=None):
        ids = list(range(self.num_envs)) if env_indices is None else list(env_indices)
        if seed is not None:
            self.seed(seed)
        count = len(ids)
        self.reset_term.cache_row_indices[ids] = self.env.rng.integers(0, 1000, count).numpy()
        values = self.env.rng.random((count, 2)).numpy()
        self.dr.kp[ids] = values
        self.dr.kd[ids] = values + 1
        self.dr.mass[ids] = values[:, :1]
        self.dr.com_offset[ids, 0] = values[:, 0]
        self.dr.gravity[ids, 2] = -9.81
        self.dr.friction_scale[ids] = 1.0
        self.obs[ids] = torch.from_numpy(values[:, :1])
        self.steps[ids] = 0
        return {key: value[ids].clone() for key, value in (("obs", self.obs),)}, {}

    def step(self, actions):
        random = self.env.rng.random((self.num_envs, 1))
        self.obs = self.obs + actions + random
        self.steps += 1
        terminated = torch.tensor(
            [self.steps[slot] >= self.done_after[slot] for slot in range(self.num_envs)]
        )
        truncated = self.steps >= 6
        self.frame.object_angvel[:] = 0
        self.frame.object_angvel[:, 0] = random[:, 0].numpy()
        return SimpleNamespace(
            obs={"obs": self.obs.clone()},
            reward=random[:, 0].clone(),
            terminated=terminated,
            truncated=truncated & ~terminated,
        )


def fake_cfg():
    return SimpleNamespace(
        env=SimpleNamespace(
            events=SimpleNamespace(
                reset=SimpleNamespace(params=SimpleNamespace(grasp_cache_path="grasps"))
            )
        )
    )


def fake_episodes(env, cfg, seeds):
    episodes = []
    for slot, seed in enumerate(seeds):
        _, record = reset_record(env, cfg, 1.0, seed, slot=slot)
        episodes.append(
            {
                "episode_id": f"1/{seed}",
                "scale": 1.0,
                "reset_seed": seed,
                "perturbation_seed": seed + 10000,
                **record,
            }
        )
    return episodes


def zero_policy(actor, obs, device, hist_norm):
    return SimpleNamespace(
        deterministic=lambda: torch.zeros_like(obs["obs"]),
        std=torch.ones_like(obs["obs"]),
    )


def test_batched_rng_matches_independent_manager_streams():
    rng = BatchedEvaluationRng(2, device="cpu")
    rng.seed_slots([17, 29])
    actual = rng.random((2, 4))
    expected = torch.cat(
        [ManagerRng.seeded(seed, device="cpu").random((1, 4)) for seed in (17, 29)]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    rng.seed_slots([17, 29])
    with rng.reset_slot(1):
        rng.manual_seed(99)
        slot_one = rng.random((1, 4))
    torch.testing.assert_close(
        slot_one, ManagerRng.seeded(99, device="cpu").random((1, 4)), rtol=0, atol=0
    )

    with rng.reset_slot(1):
        rng.manual_seed(567)
        selected_gravity = rng.uniform(-1.0, 1.0, (2, 1))
    torch.testing.assert_close(
        selected_gravity,
        ManagerRng.seeded(567, device="cpu").uniform(-1.0, 1.0, (2, 1)),
        rtol=0,
        atol=0,
    )

    with rng.reset_slot(0, full_width=True):
        rng.manual_seed(1234)
        full_width = rng.random((2, 2))
    torch.testing.assert_close(
        full_width[:, :1],
        ManagerRng.seeded(1234, device="cpu").random((2, 1)),
        rtol=0,
        atol=0,
    )


def test_batched_rng_maps_sparse_persistent_force_draws_to_slots():
    seeds = [17, 29, 43]
    rng = BatchedEvaluationRng(3, device="cpu")
    rng.bind_persistent_force(0.5)
    rng.seed_slots(seeds)
    trigger = rng.random(3)
    slots = (trigger < 0.5).nonzero(as_tuple=False).flatten()
    actual = rng.standard_normal((slots.numel(), 3))
    expected = []
    for slot, seed in enumerate(seeds):
        stream = ManagerRng.seeded(seed, device="cpu")
        if float(stream.random(1)[0]) < 0.5:
            assert slot in slots
            expected.append(stream.standard_normal((1, 3)))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)


def test_parallel_rollout_matches_serial_with_done_and_manifest_boundaries(tmp_path, monkeypatch):
    cache = tmp_path / "grasps_1.npy"
    cache.write_bytes(b"fixed cache")
    monkeypatch.setattr(evaluation, "resolve_grasp_cache_file", lambda prefix, scale: cache)
    monkeypatch.setattr(evaluation, "inference_distribution", zero_policy)
    cfg = fake_cfg()
    seeds = [101, 202, 303]

    batched = FakeEnv(3, done_after=[2, 4, 3])
    owner = _install_evaluation_rng(batched)
    assert batched.env.rng is owner and batched.env._rng_owner is owner
    episodes = fake_episodes(batched, cfg, seeds)
    parallel = _rollout_episode_batch(batched, cfg, None, episodes, device="cpu")

    serial = []
    for slot, episode in enumerate(episodes):
        env = FakeEnv(1, done_after=[[2, 4, 3][slot]])
        _install_evaluation_rng(env)
        serial.extend(_rollout_episode_batch(env, cfg, None, [episode], device="cpu"))
    assert parallel == serial
    assert [row["survival_seconds"] for row in parallel] == [0.1, 0.2, 0.15000000000000002]
    assert [row["dropped"] for row in parallel] == [1.0, 1.0, 1.0]

    invalid = [dict(episode) for episode in episodes]
    invalid[1]["cache_row"] += 1
    env = FakeEnv(3, done_after=[2, 4, 3])
    _install_evaluation_rng(env)
    with pytest.raises(ValueError, match="cannot reproduce"):
        _rollout_episode_batch(env, cfg, None, invalid, device="cpu")


def test_batch_rejects_mixed_scales_and_foreign_persistent_force_rng(tmp_path, monkeypatch):
    cache = tmp_path / "grasps_1.npy"
    cache.write_bytes(b"fixed cache")
    monkeypatch.setattr(evaluation, "resolve_grasp_cache_file", lambda prefix, scale: cache)
    env = FakeEnv(2, done_after=[2, 2])
    _install_evaluation_rng(env)
    episodes = fake_episodes(env, fake_cfg(), [1, 2])
    episodes[1]["scale"] = 2.0
    with pytest.raises(ValueError, match="cannot mix scales"):
        _rollout_episode_batch(env, fake_cfg(), None, episodes, device="cpu")

    force = SimpleNamespace(rng=object())
    env.event_manager.active_terms = {"step": ["persistent_force"]}
    env.event_manager.get_term_cfg = lambda name: (
        SimpleNamespace(func=force)
        if name == "persistent_force"
        else (_ for _ in ()).throw(AssertionError(name))
    )
    with pytest.raises(ValueError, match="must share"):
        set_step_rng(env, [1, 2], 0)
    assert env.env.rng is env.env._rng_owner
