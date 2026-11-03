import json
import subprocess
import sys
from importlib.util import find_spec

import pytest
import torch

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint
from sharpa_rl_unilab.training.student_runtime import train_student
from sharpa_rl_unilab.training.supermimic_mdp import SharpaSuperMimicCapture


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJWarp capture requires CUDA")
def test_mjwarp_capture_is_gpu_resident(monkeypatch):
    cfg = compose_config("supermimic", "mjwarp", ["training.num_envs=2", "training.device=cuda:0"])
    env = SharpaTeacherEnv(cfg, 2, auto_reset=False)
    try:
        env.reset(seed=7)
        capture = SharpaSuperMimicCapture(env)
        capture.refresh()
        capture()
        env.step(torch.zeros(2, 22, device=env.device))
        result = capture()
        source = capture.observation.snapshot(env.env)
        expected = torch.cat(
            [
                torch.as_tensor(value, device=env.device)
                for value in (
                    source.object_pos,
                    source.object_quat,
                    source.object_linvel,
                    source.object_angvel,
                    source.dof_pos,
                    source.dof_vel,
                    source._clean_tactile,
                    source._tactile_output,
                )
            ]
            + [capture.action.target],
            dim=-1,
        )
        torch.testing.assert_close(result["s"], expected, rtol=1e-5, atol=1e-6)

        def forbidden(*args, **kwargs):
            raise AssertionError("resident capture requested a host-to-device conversion")

        monkeypatch.setattr(torch, "as_tensor", forbidden)
        resident = capture()
        assert resident["s"].device.type == "cuda"
        assert resident["xi"].device.type == "cuda"
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SuperMimic smoke uses CUDA training")
def test_train_and_evaluate_supermimic_direct(tmp_path):
    run = tmp_path / "direct"
    overrides = [
        "training.num_envs=8",
        "training.device=cuda:0",
        "training.no_play=true",
        "training.logger=no_print",
        f"training.log_dir={run}",
        "algo.single_stage=true",
        "algo.max_iterations=2",
        "algo.save_interval=1",
        "algo.warmup_steps=4",
        "algo.wm_window=2",
        "algo.wm_batches=1",
        "algo.wm_batch=4",
        "algo.wm_hidden=[64,64]",
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
    path = run / "policy_final.pt"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert max(row["sim/reward_parity"] for row in rows) < 1e-6
    assert rows[-1]["policy_version"] == 2
    assert (run / "policy_iteration_2.pt").is_file()
    assert not (run / "teacher_final.pt").exists()

    direct = evaluate_checkpoint(path, device="cpu")
    assert direct["stage"] == "direct"
    assert len(direct["episodes"]) == 1


@pytest.mark.slow
@pytest.mark.skipif(
    find_spec("motrixsim") is None, reason="SuperMimic Motrix smoke requires MotrixSim"
)
def test_motrix_reset_preserves_hand_joint_state():
    cfg = compose_config(
        "supermimic",
        "motrix",
        ["training.num_envs=2", "training.device=cpu", "algo.single_stage=true"],
    )
    env = SharpaTeacherEnv(cfg, 2, auto_reset=False)
    try:
        env.reset(seed=12345)
        env.observation_manager.compute()
        hand = env.termination_manager.get_term_cfg("dropped").func.observation.dof_pos
        target = env.action_manager.get_term("hand").target.cpu()
        torch.testing.assert_close(hand, target)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(
    find_spec("motrixsim") is None, reason="SuperMimic Motrix smoke requires MotrixSim"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SuperMimic smoke uses CUDA training")
def test_train_and_evaluate_supermimic_direct_motrix(tmp_path):
    run = tmp_path / "direct_motrix"
    overrides = [
        "training.num_envs=8",
        "training.device=cuda:0",
        "training.no_play=true",
        "training.logger=no_print",
        f"training.log_dir={run}",
        "algo.single_stage=true",
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
        "evaluation.seeds=[10001]",
        "evaluation.scales=[0.8,1.2]",
        "evaluation.scale_weights=[0.5,0.5]",
        "evaluation.episodes_per_scale=1",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sharpa_rl_unilab.cli",
            "--algo",
            "supermimic",
            "--sim",
            "motrix",
            *overrides,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    path = run / "policy_final.pt"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["policy_version"] == 2
    assert (run / "policy_iteration_2.pt").is_file()

    direct = evaluate_checkpoint(path, device="cpu")
    assert direct["stage"] == "direct"
    assert [episode["scale"] for episode in direct["episodes"]] == [0.8, 1.2]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SuperMimic smoke uses CUDA training")
def test_default_train_evaluate_and_distill_supermimic(tmp_path):
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
        "algo.wm_hidden=[64,64]",
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
    assert (run / "teacher_iteration_2.pt").is_file()
    assert not (run / "policy_final.pt").exists()

    teacher = evaluate_checkpoint(path, device="cpu")
    assert teacher["stage"] == "teacher"
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
