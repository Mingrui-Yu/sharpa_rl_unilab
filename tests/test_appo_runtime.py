import multiprocessing as mp
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.teacher import (
    TeacherAPPOLearner,
    frozen_weights,
    observe_new_samples,
)
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training import teacher_runtime as runtime


def test_latest_policy_snapshot_replaces_backlog_without_aliasing():
    cfg = compose_config("appo", "mujoco", [])
    actor, _, _ = runtime.make_models(cfg, "cpu")
    rollouts = runtime.AsyncRollouts.__new__(runtime.AsyncRollouts)
    rollouts.weights = mp.get_context("spawn").Queue(maxsize=1)
    try:
        for version in range(1, 4):
            with torch.no_grad():
                actor.shared.mu_head.bias.fill_(version)
                actor.shared.obs_normalizer._mean.fill_(version * 10)
            rollouts.publish(version, actor)
        expected = frozen_weights(actor)
        with torch.no_grad():
            actor.shared.mu_head.bias.zero_()
            actor.shared.obs_normalizer._mean.zero_()
        version, state = rollouts.weights.get(timeout=5)
        assert version == 3
        for key, value in expected.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        with pytest.raises(queue.Empty):
            rollouts.weights.get_nowait()
    finally:
        rollouts.weights.close()
        rollouts.weights.join_thread()


def test_collector_switches_complete_policy_only_between_rollouts(monkeypatch):
    cfg = compose_config(
        "appo",
        "mujoco",
        [
            "training.num_envs=8",
            "training.device=cuda:0",
            "algo.collector_device=cpu",
            "algo.collector_seed=2",
            "training.torch_threads.collector_num_threads=2",
        ],
    )
    actor, critic, _ = runtime.make_models(cfg, "cpu")
    with torch.no_grad():
        actor.shared.mu_head.bias.fill_(3)
    snapshots = {0: frozen_weights(actor)}
    for version in (1, 2):
        with torch.no_grad():
            actor.shared.mu_head.bias.fill_(3 + version)
        observe_new_samples(actor, critic, torch.full((8, 156), float(version)), torch.ones(8, 174))
        snapshots[version] = frozen_weights(actor)
    weights = queue.Queue(maxsize=2)
    ctx = mp.get_context("spawn")
    stop, count = ctx.Event(), ctx.Value("q", 0)
    packets, seeds, closed = [], [], []

    class Env:
        def __init__(self, config, num_envs):
            self.num_envs = num_envs
            self.steps = 0
            self.obs = {
                key: np.zeros((num_envs, dim), np.float32)
                for key, dim in (("obs", 147), ("priv_info", 9), ("critic", 174))
            }

        def reset(self, *, seed):
            seeds.append(seed)
            return self.obs, {}

        def step(self, actions):
            self.steps += 1
            if self.steps == 1:
                weights.put((1, snapshots[1]))
                weights.put((2, snapshots[2]))
            self.obs = {key: value + 0.1 for key, value in self.obs.items()}
            return SimpleNamespace(
                obs=self.obs,
                reward=np.ones(8),
                terminated=np.zeros(8, bool),
                truncated=np.zeros(8, bool),
                info={},
            )

        def close(self):
            closed.append(True)

    class Output:
        def put(self, packet, timeout=None):
            assert isinstance(packet, tuple), packet
            packets.append(packet)
            if len(packets) == 2:
                stop.set()

    monkeypatch.setattr(runtime, "SharpaTeacherEnv", Env)
    runtime._collector(runtime.config_dict(cfg), snapshots[0], Output(), weights, stop, count)
    assert seeds == [2] and closed == [True]
    assert [packet[2] for packet in packets] == [0, 2]
    assert [packet[3] for packet in packets] == [64, 128]
    assert count.value == 128
    verifier, _, _ = runtime.make_models(cfg, "cpu")
    verifier.eval()
    pool = None
    for raw, last_obs, version, _, _ in packets:
        verifier.load_state_dict(snapshots[version])
        before = frozen_weights(verifier)
        packed = torch.tensor(np.concatenate((raw["obs"], raw["priv_info"]), axis=-1)).flatten(0, 1)
        verifier(TensorDict({"policy": packed}, batch_size=64))
        logp = verifier.get_output_log_prob(torch.tensor(raw["actions"]).flatten(0, 1))
        torch.testing.assert_close(logp.view(8, 8), torch.tensor(raw["actions_log_prob"]))
        assert np.abs(raw["actions"]).max() > 1  # Retain unbounded Gaussian samples.
        for key, value in before.items():
            torch.testing.assert_close(verifier.state_dict()[key], value, rtol=0, atol=0)
        pool, _ = runtime.stage_rollout(pool, raw, last_obs, version, actor, critic, "cpu")
    assert pool is not None
    batch = pool.batch()
    original_logp = batch["actions_log_prob"].clone()
    learner = TeacherAPPOLearner(actor=actor, critic=critic, device="cpu")
    observe_new_samples(actor, critic, torch.full((8, 156), 50.0), torch.zeros(8, 174))
    learner.sync_target_actor_buffers()
    normalizers = [
        actor.shared.obs_normalizer,
        critic.obs_normalizer,
        learner.target_actor.shared.obs_normalizer,
    ]
    before = [frozen_weights(norm) for norm in normalizers]
    learner.process_batch(batch)
    torch.testing.assert_close(batch["actions_log_prob"], original_logp, rtol=0, atol=0)
    learner.update(batch)
    for norm, state in zip(normalizers, before):
        for key, value in state.items():
            torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)
