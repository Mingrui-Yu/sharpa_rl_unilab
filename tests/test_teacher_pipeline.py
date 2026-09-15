import json
import subprocess
import sys

import pytest
import torch

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training import teacher_runtime as runtime
from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint
from sharpa_rl_unilab.training.student_runtime import train_student


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "appo", "flashsac"])
def test_train_checkpoint_evaluate_distill(tmp_path, monkeypatch, algo):
    if algo == "flashsac" and not torch.cuda.is_available():
        pytest.skip("The FlashSAC integration smoke uses CUDA replay")
    run = tmp_path / "teacher"
    overrides = [
        "training.num_envs=8",
        f"training.device={'cuda:0' if algo == 'flashsac' else 'cpu'}",
        "training.torch_threads.learner_num_threads=2",
        "training.torch_threads.collector_num_threads=2",
        "training.no_play=true",
        "training.logger=no_print",
        f"training.log_dir={run}",
        "algo.max_iterations=2",
        "algo.save_interval=1",
        "evaluation.scales=[1.0]",
        "evaluation.scale_weights=[1.0]",
        "evaluation.seeds=[10001]",
        "evaluation.episodes_per_scale=1",
    ]
    if algo == "flashsac":
        # Warmup exceeds one sampling packet; keep the native process/CLI boundary.
        overrides += ["algo.batch_size=16", "algo.replay_buffer_n=16", "algo.learning_starts=5"]
        result = subprocess.run(
            [sys.executable, "-m", "sharpa_rl_unilab.cli", "--algo", algo, *overrides],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        path = run / "teacher_final.pt"
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert rows[0]["received"] >= 40
    else:
        overrides += ["algo.algorithm.num_learning_epochs=1", "algo.algorithm.num_mini_batches=1"]
        if algo == "ppo":
            # A non-divisible budget exercises the shortened final rollout.
            overrides += [
                "algo.num_steps_per_env=2",
                "algo.max_iterations=null",
                "training.max_transitions=17",
            ]
        else:
            overrides += ["algo.steps_per_env=2"]
        closed = []
        original_close = runtime.AsyncRollouts.close

        def close(rollouts):
            original_close(rollouts)
            assert not rollouts.process.is_alive() and rollouts.process.exitcode == 0
            closed.append(True)

        monkeypatch.setattr(runtime.AsyncRollouts, "close", close)
        path = runtime.train_teacher(compose_config(algo, "mujoco", overrides))
        assert bool(closed) == (algo == "appo")
    actor, _, snapshot = runtime.load_policy(path)
    counters = snapshot["counters"]
    assert counters["policy_version"] == 2
    assert counters["collected"] >= counters["received"] > 0
    if algo == "ppo":
        assert counters["received"] == 24
    assert actor.shared.obs_normalizer.count.item() == counters["received"]
    assert snapshot["critic"]["obs_normalizer.count"].item() == counters["received"]
    assert {p.name for p in run.glob("teacher_iteration_*.pt")} == {
        "teacher_iteration_1.pt",
        "teacher_iteration_2.pt",
    }
    torch.testing.assert_close(actor.state_dict(), snapshot["actor"], rtol=0, atol=0)
    teacher_result = evaluate_checkpoint(path, device="cpu")
    assert len(teacher_result["episodes"]) == 1
    if algo == "ppo":
        assert evaluate_checkpoint(path, device="cpu")["episodes"] == teacher_result["episodes"]
    student_path = train_student(
        path,
        device="cpu",
        overrides=[
            "distillation.transitions=16",
            "distillation.num_envs=8",
            "distillation.save_every=0",
            f"training.log_dir={tmp_path / 'student'}",
        ],
    )
    student, _, state = runtime.load_policy(student_path, stage="student")
    assert student.shared.obs_normalizer.count.item() == counters["received"]
    assert state["history_normalizer"]["count"].item() == 16
    measured = evaluate_checkpoint(student_path, device="cpu")
    assert measured["manifest_sha256"] == teacher_result["manifest_sha256"]
    assert state["teacher"]["sha256"] == teacher_result["checkpoint_sha256"]
