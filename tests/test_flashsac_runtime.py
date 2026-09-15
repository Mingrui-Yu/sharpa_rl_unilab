import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from uni_rl.offpolicy.thread_budget import resolve_torch_thread_runtime
from unilab.base.np_env import NpEnvState

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import transition_next
from sharpa_rl_unilab.training.flashsac_runtime import FlashTeacherEnv, TeacherDoubleBufferRunner
from sharpa_rl_unilab.training.teacher_runtime import load_policy, training_budget


def test_flashsac_defaults_and_native_loop():
    cfg = compose_config("flashsac", "mujoco", [])
    assert cfg.algo.num_envs == cfg.training.num_envs == 2048
    assert cfg.algo.max_iterations == 3000
    assert cfg.training.max_transitions is None
    assert cfg.algo.use_amp is False
    assert cfg.training.torch_threads.learner_num_threads == 4
    assert training_budget(cfg) == ("policy_version", 3000)
    assert TeacherDoubleBufferRunner.learn is DoubleBufferOffPolicyRunner.learn
    runtime = resolve_torch_thread_runtime(cfg.training.torch_threads, cpu_count=64)
    assert runtime["learner"] == {"num_threads": 4, "num_interop_threads": 1}
    assert runtime["collector"] == {"num_threads": 4, "num_interop_threads": 1}
    assert runtime["compile_threads"] == 2
    with pytest.raises(ValueError, match="exactly one budget"):
        training_budget(compose_config("flashsac", "mujoco", ["training.max_transitions=17"]))
    cfg = compose_config(
        "flashsac",
        "mujoco",
        [
            "training.num_envs=8",
            "algo.max_iterations=null",
            "training.max_transitions=17",
        ],
    )
    assert training_budget(cfg) == ("received", 24)


def test_transport_preserves_terminal_privilege_and_clean_critic():
    def observations(offset):
        return {
            key: np.full((2, width), offset + i, dtype=np.float32)
            for i, (key, width) in enumerate((("obs", 147), ("priv_info", 9), ("critic", 174)))
        }

    state = NpEnvState(
        observations(0),
        np.ones(2),
        np.array([True, False]),
        np.array([False, True]),
        {},
        observations(10),
    )
    env = FlashTeacherEnv.__new__(FlashTeacherEnv)
    env.teacher = SimpleNamespace(step=lambda _: state)
    packed = env.step(np.zeros((2, 22)))
    assert packed.obs["obs"].shape == (2, 156)
    nxt = transition_next(packed)
    np.testing.assert_array_equal(nxt["obs"][:, :147], state.final_observation["obs"])
    np.testing.assert_array_equal(nxt["obs"][:, 147:], state.final_observation["priv_info"])
    np.testing.assert_array_equal(nxt["critic"], state.final_observation["critic"])
    np.testing.assert_array_equal(packed.obs["obs"][:, 147:], state.obs["priv_info"])


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native device replay requires CUDA or MPS"
)
@pytest.mark.parametrize(
    "learning_starts, save_interval, horizon", [(1, 1, 2), (5, 1, 2), (1, 0, 3)]
)
def test_native_warmup_checkpoints_evaluation_and_distillation(
    tmp_path, learning_starts, save_interval, horizon
):
    from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint
    from sharpa_rl_unilab.training.student_runtime import train_student

    thread_overrides = (
        []
        if save_interval
        else [
            "training.torch_threads.learner_num_threads=3",
            "training.torch_threads.collector_num_threads=2",
        ]
    )
    run = tmp_path / "teacher"
    script = tmp_path / "train_checked.py"
    script.write_text("""
import json
import os
from pathlib import Path
import torch
from sharpa_rl_unilab.training import flashsac_runtime as runtime
from sharpa_rl_unilab.cli import train_main

class ThreadCheckedEnv(runtime.FlashTeacherEnv):
    def __init__(self, num_envs, env_cfg_override, *, config):
        if num_envs == 8:
            Path(config["training"]["log_dir"], "collector_threads.json").write_text(json.dumps({
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            }))
        super().__init__(num_envs, env_cfg_override, config=config)

runtime.FlashTeacherEnv = ThreadCheckedEnv
if __name__ == "__main__":
    train_main()
""")
    # A fresh process also tests spawn, independent thread budgets and CLI dispatch.
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--algo",
            "flashsac",
            "training.num_envs=8",
            "algo.max_iterations=3",
            "algo.batch_size=16",
            "algo.replay_buffer_n=16",
            f"algo.learning_starts={learning_starts}",
            f"algo.save_interval={save_interval}",
            f"training.env_steps_per_sync={horizon}",
            *thread_overrides,
            "training.no_play=true",
            "training.logger=no_print",
            f"training.log_dir={run}",
            "evaluation.scales=[1.0]",
            "evaluation.scale_weights=[1.0]",
            "evaluation.seeds=[10001]",
            "evaluation.episodes_per_scale=1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["last_checkpoint"] == str(run / "teacher_final.pt")
    assert summary["completed_iterations"] == 3
    assert summary["runtime_manifest"]["inference_owner"] == "learner"
    assert summary["runtime_manifest"]["collector_actor"] is False
    assert summary["runtime_manifest"]["use_amp"] is False
    cfg = compose_config("flashsac", "mujoco", thread_overrides)
    threads = resolve_torch_thread_runtime(cfg.training.torch_threads)
    collector = json.loads((run / "collector_threads.json").read_text())
    assert collector["num_threads"] == threads["collector"]["num_threads"]
    assert collector["num_interop_threads"] == threads["collector"]["num_interop_threads"]
    assert collector["OMP_NUM_THREADS"] == str(threads["collector"]["num_threads"])
    assert summary["runtime_manifest"]["learner_torch_threads"] == threads["learner"]["num_threads"]
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["policy_version"] for row in metrics] == [1, 2, 3]
    assert metrics[0]["received"] >= max(16, learning_starts * 8)
    counters = summary["counters"]
    assert counters["critic_updates"] == 42
    assert counters["actor_updates"] == counters["temperature_updates"] == 21
    assert counters["collected"] >= counters["received"]
    assert counters["collected"] == summary["total_env_steps"]
    for path in run.glob("teacher_*.pt"):
        actor, _, checkpoint = load_policy(path)
        assert actor.shared.obs_normalizer.count.item() == checkpoint["counters"]["received"]
        assert (
            checkpoint["critic"]["obs_normalizer.count"].item()
            == checkpoint["counters"]["received"]
        )
        assert checkpoint["learner"]["update_count"] == checkpoint["counters"]["policy_version"]
        assert (
            checkpoint["learner"]["critic_scheduler"]["last_epoch"]
            == checkpoint["counters"]["critic_updates"]
        )
        assert (
            checkpoint["learner"]["actor_scheduler"]["last_epoch"]
            == checkpoint["counters"]["actor_updates"]
        )
    expected_saves = {f"teacher_iteration_{i}.pt" for i in range(1, 4)} if save_interval else set()
    assert {p.name for p in run.glob("teacher_iteration_*.pt")} == expected_saves
    assert not list(run.glob("evaluation_*.json"))
    final = run / "teacher_final.pt"
    evaluation = evaluate_checkpoint(final, device="cpu")
    assert len(evaluation["episodes"]) == 1
    student_path = train_student(
        final,
        device="cpu",
        overrides=[
            "distillation.transitions=16",
            "distillation.num_envs=8",
            "distillation.save_every=0",
            f"training.log_dir={tmp_path / 'student'}",
        ],
    )
    student, _, _ = load_policy(student_path, stage="student")
    assert student.shared.obs_normalizer.count.item() == counters["received"]
