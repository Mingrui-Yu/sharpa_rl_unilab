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
def test_train_save_load_evaluate_distill(algo, tmp_path, monkeypatch):
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
        "budget.transitions=25",
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
    cfg = compose_config(algo, "mujoco", overrides)
    path = train_teacher(cfg)
    metrics = [
        json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["received"] == 32
    assert metrics[-1]["perf/transitions_per_second"] > 0
    assert list(path.parent.glob("events.out.tfevents.*"))
    actor, _, checkpoint = load_policy(path, stage="teacher")
    assert checkpoint["counters"]["collected"] == checkpoint["counters"]["received"] == 32
    assert actor.shared.obs_normalizer.count.item() == 32
    periodic = list(path.parent.glob("teacher_[0-9]*.pt"))
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
    assert policy.shared.obs_normalizer.count.item() == 32
    measured = evaluate_checkpoint(student)
    assert measured["manifest_sha256"] == result["manifest_sha256"]
    assert measured["teacher_cost"]["counters"]["collected"] == 32
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
