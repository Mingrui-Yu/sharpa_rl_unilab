import json
import subprocess
import sys

import pytest
import torch

from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint
from sharpa_rl_unilab.training.student_runtime import train_student


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SuperMimic smoke uses CUDA training")
def test_train_evaluate_and_distill_supermimic(tmp_path):
    run = tmp_path / "teacher"
    overrides = [
        "training.num_envs=8",
        "training.device=cuda:0",
        "training.no_play=true",
        "training.logger=no_print",
        f"training.log_dir={run}",
        "algo.max_iterations=2",
        "algo.save_interval=1",
        "algo.warmup_steps=4",
        "algo.wm_window=2",
        "algo.wm_batches=1",
        "algo.wm_batch=4",
        "algo.wm_hidden=[64]",
        "algo.pi_batch=4",
        "algo.horizon=2",
        "algo.critic_epochs=1",
        "algo.critic_minibatches=2",
        "algo.wm_only_iters=0",
        "evaluation.scales=[1.0]",
        "evaluation.scale_weights=[1.0]",
        "evaluation.seeds=[10001]",
        "evaluation.episodes_per_scale=1",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "sharpa_rl_unilab.cli", "--algo", "supermimic", *overrides],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    path = run / "teacher_final.pt"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert max(row["sim/reward_parity"] for row in rows) < 1e-6
    assert rows[-1]["policy_version"] == 2

    teacher = evaluate_checkpoint(path, device="cpu")
    assert len(teacher["episodes"]) == 1
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
    student = evaluate_checkpoint(student_path, device="cpu")
    assert student["manifest_sha256"] == teacher["manifest_sha256"]
