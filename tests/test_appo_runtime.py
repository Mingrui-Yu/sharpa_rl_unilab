import json
import multiprocessing as mp
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from sharpa_rl_unilab.algos.hora.teacher import (
    TeacherAPPOLearner,
    frozen_weights,
    observe_new_samples,
)
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training import teacher_runtime as runtime
from sharpa_rl_unilab.training.evaluation import deterministic_actions
from sharpa_rl_unilab.training.logging import TrainingLogger


@pytest.fixture(autouse=True)
def small_torch_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(previous)


def test_hora_parameters_and_other_algorithm_defaults():
    cfg = compose_config("appo", "mujoco", [])
    baseline = OmegaConf.load(
        Path(__file__).parents[1] / "docs/migrations/issue-2-baseline/appo_hora.yaml"
    )
    assert cfg.algo.algorithm == baseline.algo.algorithm
    for key in ("num_envs", "steps_per_env", "seed"):
        assert cfg.algo[key] == baseline.algo[key]
    assert cfg.algo.staging_pool_size == 8
    assert cfg.algo.async_queue_size == 4
    assert cfg.training.max_transitions is None
    assert cfg.algo.max_iterations == 501
    assert cfg.algo.save_interval == 50
    assert cfg.training.device == "cuda:0"
    assert cfg.algo.collector_device is None
    assert cfg.protocol.appo_baseline == "appo-hora-790ae32"
    for algo in ("ppo", "flashsac"):
        other = compose_config(algo, "mujoco", [])
        assert other.training.num_envs == 2048
        assert other.training.device == "cuda:0"
        assert "collector_device" not in other.algo
        assert other.training.torch_threads == cfg.training.torch_threads
        assert other.training.max_transitions is None
        assert other.algo.save_interval == 50
        assert other.distillation == cfg.distillation


def test_exclusive_budgets_and_optional_threads():
    cfg = compose_config("appo", "mujoco", [])
    assert runtime.training_budget(cfg) == ("policy_version", 501)
    runtime.configure_threads(cfg)
    assert torch.get_num_threads() == 4
    assert torch.get_num_interop_threads() == 1
    cfg.training.torch_threads.enabled = False
    torch.set_num_threads(2)
    runtime.configure_threads(cfg)
    assert torch.get_num_threads() == 2
    cfg.training.max_transitions = 17
    with pytest.raises(ValueError, match="exactly one"):
        runtime.training_budget(cfg)
    cfg.algo.max_iterations = None
    cfg.training.num_envs = 8
    assert runtime.training_budget(cfg) == ("received", 24)
    cfg.algo.save_interval = -1
    with pytest.raises(ValueError, match="save_interval"):
        runtime.training_budget(cfg)


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_priority_and_explicit_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert runtime.resolve_device() == expected
    assert runtime.resolve_device("cpu") == "cpu"


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


def test_cumulative_statistics_are_independent_of_batch_partition():
    cfg = compose_config("appo", "mujoco", [])
    full_actor, full_critic, _ = runtime.make_models(cfg, "cpu")
    split_actor, split_critic, _ = runtime.make_models(cfg, "cpu")
    packed, clean = torch.randn(57, 156), torch.randn(57, 174) * 3 + 4
    observe_new_samples(full_actor, full_critic, packed, clean)
    for actor_chunk, critic_chunk in zip(packed.split(7), clean.split(7)):
        observe_new_samples(split_actor, split_critic, actor_chunk, critic_chunk)
    for full, split in (
        (full_actor.shared.obs_normalizer, split_actor.shared.obs_normalizer),
        (full_critic.obs_normalizer, split_critic.obs_normalizer),
    ):
        assert full.count == split.count == 57
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, split.state_dict()[key], rtol=1e-5, atol=1e-6)


def test_collector_switches_complete_policy_only_between_rollouts(monkeypatch):
    cfg = compose_config(
        "appo",
        "mujoco",
        [
            "training.num_envs=8",
            "training.device=cuda:0",
            "algo.collector_device=cpu",
            "algo.collector_seed=2",
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
    assert torch.get_num_threads() == 4  # Collector applies its own role budget.
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
    learner.process_batch(batch)
    torch.testing.assert_close(batch["actions_log_prob"], original_logp, rtol=0, atol=0)


@pytest.mark.parametrize("algo", ["ppo", "appo"])
def test_iteration_progress_keeps_real_sampling_axis(tmp_path, algo):
    cfg = compose_config(algo, "mujoco", ["training.logger=no_print"])
    logger = TrainingLogger(tmp_path, cfg, 10)
    try:
        logger.log({"policy_version": 2, "received": 1234, "collected": 1456, "wall_seconds": 10})
        assert logger._estimate_eta() == "40s"
        assert "Iterations 2/10" in logger._build_display().title
        row = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert row["received"] == 1234 and row["collected"] == 1456
        assert row["perf/transitions_per_second"] == 123.4
    finally:
        logger.close()


def test_comparison_smoke_explicitly_selects_sampling_budgets(tmp_path, monkeypatch):
    from sharpa_rl_unilab.training import compare

    commands = []
    monkeypatch.setattr(
        sys, "argv", ["sharpa-compare", "--output", str(tmp_path / "comparison"), "--smoke"]
    )
    monkeypatch.setattr(compare, "make_manifest", lambda *_: None)
    monkeypatch.setattr(compare, "evaluate_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(compare, "aggregate_seeds", lambda *_: {})
    monkeypatch.setattr(compare, "write_json", lambda *_: None)
    monkeypatch.setattr(compare, "plot_learning_curves", lambda *_: None)
    monkeypatch.setattr(
        compare.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    compare.main()
    teacher_commands = [command for command in commands if command[2] == "sharpa_rl_unilab.cli"]
    assert len(teacher_commands) == 9
    for command in teacher_commands:
        algo = command[4]
        cfg = compose_config(algo, "mujoco", command[5:])
        assert runtime.training_budget(cfg) == ("received", 128)
        assert cfg.training.num_envs == 8
        assert cfg.algo.max_iterations is None
        assert cfg.algo.save_interval == 0


@pytest.mark.slow
def test_iteration_training_freezes_statistics_and_roundtrips_checkpoint(tmp_path, monkeypatch):
    cfg = compose_config(
        "appo",
        "mujoco",
        [
            "training.num_envs=8",
            "training.device=cpu",
            "training.torch_threads.learner_num_threads=2",
            "algo.max_iterations=10",
            "algo.save_interval=3",
            "training.logger=no_print",
            f"training.log_dir={tmp_path / 'run'}",
        ],
    )
    original_process = TeacherAPPOLearner.process_batch
    original_update = TeacherAPPOLearner.update
    original_close = runtime.AsyncRollouts.close
    closed = []
    counts = []

    def normalizers(learner):
        return [
            learner.actor.shared.obs_normalizer,
            learner.critic.obs_normalizer,
            learner.target_actor.shared.obs_normalizer,
        ]

    def process(learner, batch):
        assert batch["observations"].shape[0] == 8
        states = [frozen_weights(norm) for norm in normalizers(learner)]
        behavior_logp = batch["actions_log_prob"].clone()
        original_process(learner, batch)
        for norm, state in zip(normalizers(learner), states):
            for key, value in state.items():
                torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(batch["actions_log_prob"], behavior_logp, rtol=0, atol=0)
        counts.append(int(learner.actor.shared.obs_normalizer.count))

    def update(learner, batch):
        states = [frozen_weights(norm) for norm in normalizers(learner)]

        # Check every optimizer step, not just the final count after all epochs.
        def check(*_args, **_kwargs):
            for norm, state in zip(normalizers(learner), states):
                for key, value in state.items():
                    torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)

        handle = learner.optimizer.register_step_pre_hook(check)
        try:
            metrics = original_update(learner, batch)
            check()
            return metrics
        finally:
            handle.remove()

    def close(rollouts):
        original_close(rollouts)
        assert not rollouts.process.is_alive()
        assert rollouts.process.exitcode == 0
        closed.append(rollouts.count.value)

    monkeypatch.setattr(TeacherAPPOLearner, "process_batch", process)
    monkeypatch.setattr(TeacherAPPOLearner, "update", update)
    monkeypatch.setattr(runtime.AsyncRollouts, "close", close)
    path = runtime.train_teacher(cfg)
    actor, effective, snapshot = runtime.load_policy(path)
    assert effective.training.device == effective.algo.collector_device == "cpu"
    assert effective.algo.collector_seed == 2
    counters = snapshot["counters"]
    assert counters["policy_version"] == 10
    assert counters["optimizer_updates"] == 200
    assert counters["collected"] == closed[-1] >= counters["received"]
    assert counters["collected"] % 64 == counters["received"] % 64 == 0
    assert (
        actor.shared.obs_normalizer.count
        == snapshot["critic"]["obs_normalizer.count"]
        == counts[-1]
        == counters["received"]
    )
    assert {p.name for p in path.parent.glob("teacher_iteration_*.pt")} == {
        "teacher_iteration_3.pt",
        "teacher_iteration_6.pt",
        "teacher_iteration_9.pt",
    }
    rows = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
    assert [row["policy_version"] for row in rows] == list(range(1, 11))
    assert [row["received"] for row in rows] == counts
    assert rows[-1]["staging_rollouts"] == rows[-1]["staging_pool_capacity"] == 8
    assert rows[-1]["training_samples"] == sum(row["staging_rollouts"] * 64 * 5 for row in rows)
    assert rows[-1]["collected"] == counters["collected"]
    assert rows[-1]["runtime/collector_num_threads"] == 4
    assert rows[-1]["runtime/collector_num_interop_threads"] == 1
    metadata = json.loads((path.parent / "run.json").read_text())["runtime"]
    assert metadata["learner_num_threads"] == 2
    assert metadata["learner_num_interop_threads"] == 1
    assert metadata["sampling_architecture"] == "async_queue"
    assert not list(path.parent.glob("evaluation_*.json"))
    adam = snapshot["optimizer_parameters"]["optimizer"][0]
    assert adam["betas"] == (0.9, 0.999) and adam["eps"] == 1e-8 and adam["weight_decay"] == 0
    obs = {"obs": np.zeros((8, 147), np.float32), "priv_info": np.ones((8, 9), np.float32)}
    expected = deterministic_actions(actor, obs, "cpu")
    before = frozen_weights(actor)
    # Evaluation actions and attempted normalization updates cannot change RMS.
    actor.shared.obs_normalizer.update(torch.full((8, 147), 100.0))
    np.testing.assert_array_equal(deterministic_actions(actor, obs, "cpu"), expected)
    for key, value in before.items():
        torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
    restored, _, _ = runtime.load_policy(path)
    np.testing.assert_array_equal(deterministic_actions(restored, obs, "cpu"), expected)
