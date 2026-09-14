import json

import numpy as np
import pytest
import torch

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv, transition_next
from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint, load_manifest, make_manifest
from sharpa_rl_unilab.training.student_runtime import train_student
from sharpa_rl_unilab.training.teacher_runtime import load_policy, train_teacher


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
@pytest.mark.parametrize("budget, expected", [(17, 24), (25, 32)])
def test_train_save_load_evaluate_distill(algo, budget, expected, tmp_path, monkeypatch):
    if algo == "ppo":
        from rsl_rl.algorithms import PPO

        original_update = PPO.update

        def check_behavior_density(learner):
            # Stored density must match the behavior parameters even though the
            # learner now has updated normalization statistics.
            mean, std = learner.storage.distribution_params
            logp = torch.distributions.Normal(mean, std).log_prob(learner.storage.actions).sum(-1)
            torch.testing.assert_close(logp, learner.storage.actions_log_prob.squeeze(-1))
            return original_update(learner)

        monkeypatch.setattr(PPO, "update", check_behavior_density)
    overrides = [
        "hardware.num_envs=8",
        "hardware.device=cpu",
        "hardware.torch_threads=2",
        f"budget.transitions={budget}",
        "budget.save_every=16",
        "budget.evaluate_every=16",
        "evaluation.scales=[1.0]",
        "evaluation.scale_weights=[1.0]",
        "evaluation.seeds=[10001]",
        "evaluation.episodes_per_scale=1",
        f"training.log_dir={tmp_path / 'teacher'}",
    ]
    if algo == "flashsac":
        overrides += ["algo.batch_size=16", "algo.updates_per_step=2"]
    else:
        overrides += ["algo.algorithm.num_learning_epochs=1", "algo.algorithm.num_mini_batches=1"]
        overrides += ["algo.steps_per_env=2" if algo == "appo" else "algo.num_steps_per_env=2"]
    if algo == "appo":
        overrides += ["algo.max_iterations=null", "budget.save_every=null", "algo.save_interval=1"]
    cfg = compose_config(algo, "mujoco", overrides)
    path = train_teacher(cfg)
    metrics = [
        json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["received"] == expected
    assert metrics[-1]["perf/transitions_per_second"] > 0
    for name in ("rotate", "obj_linvel", "pose_diff", "torque", "work", "object_pos"):
        assert np.isfinite(metrics[-1][f"reward/{name}"])
    assert metrics[-1]["timing/collector_env_step_ms"] > 0
    assert metrics[-1]["timing/collector_mlp_infer_ms"] > 0
    assert metrics[-1]["perf/collector_active_steps_per_sec"] > 0
    if algo == "appo":
        assert metrics[-1]["timing/learner_train_ms"] > 0
        assert metrics[-1]["perf/iter_ms"] >= metrics[-1]["timing/learner_train_ms"]
    assert list(path.parent.glob("events.out.tfevents.*"))
    metadata = json.loads((path.parent / "run_config.json").read_text())
    assert metadata["contract_snapshot"]["version"] == "sharpa-hora-v2"
    actor, _, checkpoint = load_policy(path, stage="teacher")
    assert checkpoint["counters"]["collected"] == checkpoint["counters"]["received"] == expected
    assert actor.shared.obs_normalizer.count.item() == expected
    periodic = list(
        path.parent.glob("teacher_iteration_*.pt" if algo == "appo" else "teacher_[0-9]*.pt")
    )
    assert periodic
    for saved in periodic:
        policy, _, snapshot = load_policy(saved)
        assert policy.shared.obs_normalizer.count.item() == snapshot["counters"]["received"]
    result = evaluate_checkpoint(path)
    repeated = evaluate_checkpoint(path)
    assert result["episodes"] == repeated["episodes"]
    assert len(result["episodes"]) == 1
    assert result["summary"]["speed_fixed_window"] == pytest.approx(
        result["summary"]["signed_angle"] / 20
    )
    assert result["summary"]["survival_seconds"] <= 20
    student = train_student(
        path,
        device="cpu",
        overrides=[
            "distillation.transitions=16",
            "distillation.num_envs=8",
            "distillation.save_every=0",
            f"training.log_dir={tmp_path / 'student'}",
        ],
    )
    policy, _, snapshot = load_policy(student, stage="student")
    student_metrics = json.loads((student.parent / "metrics.jsonl").read_text().splitlines()[-1])
    assert student_metrics["received"] == 16
    assert "latent_mse" in student_metrics
    assert list(student.parent.glob("events.out.tfevents.*"))
    assert snapshot["history_normalizer"]["count"].item() == 16
    assert policy.shared.obs_normalizer.count.item() == expected
    measured = evaluate_checkpoint(student)
    assert measured["manifest_sha256"] == result["manifest_sha256"]
    assert measured["teacher_cost"]["counters"]["collected"] == expected
    assert snapshot["teacher"]["sha256"] == result["checkpoint_sha256"]


@pytest.mark.slow
def test_simultaneous_drop_and_timeout_and_partial_reset(monkeypatch):
    from sharpa_rl_unilab.tasks.sharpa_inhand.terms.termination import SharpaDropTermination

    cfg = compose_config("ppo", "mujoco", ["+env.max_episode_seconds=0.05"])
    env = SharpaTeacherEnv(cfg, 8)
    try:
        obs, _ = env.reset(seed=11)
        clean = {key: value.copy() for key, value in obs.items()}
        env.reset(np.array([0], dtype=np.int32))
        for key in clean:
            np.testing.assert_array_equal(env.state.obs[key][1:], clean[key][1:])
        original = SharpaDropTermination.__call__

        def force_drop(self, env, **kwargs):
            original(self, env, **kwargs)
            self.dropped.fill(True)
            return self.dropped

        monkeypatch.setattr(SharpaDropTermination, "__call__", force_drop)
        state = env.step(np.zeros((8, 22)))
        assert state.terminated.all() and not state.truncated.any()
        assert state.info["timing"]["reset_done_ms"] > 0
        for key, values in transition_next(state).items():
            np.testing.assert_array_equal(values, state.final_observation[key])
    finally:
        env.close()


@pytest.mark.slow
def test_manifest_records_actual_scenes_and_enforces_quota(tmp_path):
    cfg = compose_config(
        "ppo",
        "mujoco",
        [
            "evaluation.scales=[0.8,1.5]",
            "evaluation.scale_weights=[0.5,0.5]",
            "evaluation.seeds=[5]",
            "evaluation.episodes_per_scale=1",
        ],
    )
    path = tmp_path / "scenes.json"
    first = make_manifest(cfg, path)
    other = compose_config(
        "appo",
        "mujoco",
        [
            "evaluation.scales=[0.8,1.5]",
            "evaluation.scale_weights=[0.5,0.5]",
            "evaluation.seeds=[5]",
            "evaluation.episodes_per_scale=1",
        ],
    )
    second = make_manifest(other, tmp_path / "other.json")
    assert first == second
    assert len(first["episodes"]) == 2
    assert all(
        episode["cache_row"] >= 0 and len(episode["randomization"]["kp"]) == 22
        for episode in first["episodes"]
    )
    first["episodes"].pop()
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="quota"):
        load_manifest(cfg, path)


@pytest.mark.slow
def test_reset_failure_closes_environment(tmp_path, monkeypatch):
    closed = []
    original = SharpaTeacherEnv.close

    def close(env):
        closed.append(True)
        original(env)

    def fail_reset(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(SharpaTeacherEnv, "close", close)
    monkeypatch.setattr(SharpaTeacherEnv, "reset", fail_reset)
    cfg = compose_config(
        "ppo",
        "mujoco",
        ["hardware.num_envs=8", "hardware.device=cpu", f"training.log_dir={tmp_path / 'run'}"],
    )
    with pytest.raises(RuntimeError, match="reset failed"):
        train_teacher(cfg)
    assert closed == [True]


@pytest.mark.slow
@pytest.mark.parametrize("fail_update", [False, True])
def test_appo_drains_multiple_packets_and_closes(tmp_path, monkeypatch, fail_update):
    from types import SimpleNamespace

    from sharpa_rl_unilab.training import teacher_runtime as runtime

    closed = []

    class ReadyRollouts:
        def __init__(self, cfg, actor):
            self.env = SharpaTeacherEnv(cfg, 8)
            self.obs, _ = self.env.reset(seed=11)
            self.actor = actor
            self.count = SimpleNamespace(value=0)
            self.closed = False

        def receive_ready(self):
            packets = []
            for index in range(3):
                collector_metrics = {}
                raw, self.obs = runtime.collect(
                    self.env, self.obs, self.actor, 1, "cpu", metrics=collector_metrics
                )
                # Distinct reports verify aggregation across every fresh packet.
                collector_metrics["reward/rotate"] = float(index + 1)
                self.count.value += 8
                packets.append((raw, self.obs, 0, self.count.value, collector_metrics))
            return packets

        def publish(self, version, actor):
            assert version == 1

        def close(self):
            if self.closed:
                return
            self.closed = True
            self.env.close()
            closed.append(True)

    monkeypatch.setattr(runtime, "AsyncRollouts", ReadyRollouts)
    if fail_update:

        def fail(*args):
            raise RuntimeError("learner failed")

        monkeypatch.setattr(runtime.TeacherAPPOLearner, "update", fail)
    cfg = compose_config(
        "appo",
        "mujoco",
        [
            "hardware.num_envs=8",
            "hardware.device=cpu",
            "budget.transitions=24",
            "algo.max_iterations=null",
            "budget.save_every=null",
            "algo.save_interval=0",
            "budget.evaluate_every=0",
            "algo.algorithm.num_learning_epochs=1",
            "algo.algorithm.num_mini_batches=1",
            "training.logger=no_print",
            f"training.log_dir={tmp_path / 'run'}",
        ],
    )
    if fail_update:
        with pytest.raises(RuntimeError, match="learner failed"):
            train_teacher(cfg)
    else:
        actor, _, checkpoint = load_policy(train_teacher(cfg))
        assert actor.shared.obs_normalizer.count.item() == 24
        counters = checkpoint["counters"]
        assert counters["received"] == counters["collected"] == counters["training_samples"] == 24
        assert counters["optimizer_updates"] == 1
        metrics = json.loads((tmp_path / "run/metrics.jsonl").read_text().splitlines()[-1])
        assert metrics["reward/rotate"] == 2
        assert metrics["rollouts_read"] == 3
        assert metrics["staging_rollouts"] == 3
        assert metrics["staging_pool_capacity"] == 8
    assert closed == [True]
