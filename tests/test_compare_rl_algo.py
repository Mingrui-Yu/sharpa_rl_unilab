import subprocess
import sys
from pathlib import Path

import pytest

from sharpa_rl_unilab.tools import compare_rl_algo


@pytest.mark.parametrize(
    "distill,evaluate", [(False, False), (True, False), (False, True), (True, True)]
)
def test_stages_seeds_and_shared_evaluation(monkeypatch, tmp_path, distill, evaluate):
    monkeypatch.chdir(tmp_path)
    argv = ["sharpa-compare", "--num-seeds", "2"]
    if distill:
        argv.append("--distill")
    if evaluate:
        argv.append("--eval")
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    teacher_runs = []
    manifests = []

    def run(command, *, check):
        assert check
        assert command[0] == sys.executable
        if command[1] == "-c":
            stage = "eval"
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            assert checkpoint.is_file()
            manifests.append(next(arg for arg in command if arg.startswith("evaluation.manifest=")))
        elif command[2] == "sharpa_rl_unilab.cli":
            stage = "teacher"
            run_dir = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("training.log_dir="))
            )
            seed = int(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("algo.seed="))
            )
            algorithm = command[command.index("--algo") + 1]
            assert run_dir.name == algorithm
            assert run_dir.parent.name.startswith(f"seed_{seed}_")
            run_dir.mkdir(parents=True)
            (run_dir / "teacher_final.pt").touch()
            teacher_runs.append((algorithm, seed, run_dir))
        else:
            stage = "student"
            assert len(teacher_runs) == 6
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            assert checkpoint.is_file()
            run_dir = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("training.log_dir="))
            )
            assert run_dir == checkpoint.parent / "student"
            run_dir.mkdir()
            (run_dir / "student_final.pt").touch()
        calls.append((stage, command))

    monkeypatch.setattr(compare_rl_algo.subprocess, "run", run)
    compare_rl_algo.main()

    stages = [stage for stage, _ in calls]
    assert stages == ["teacher"] * 6 + ["student"] * (6 if distill else 0) + ["eval"] * (
        (12 if distill else 6) if evaluate else 0
    )
    assert {(algo, seed) for algo, seed, _ in teacher_runs} == {
        (algo, seed) for algo in ("ppo", "appo", "flashsac") for seed in (1, 2)
    }
    assert len({run.parent.name.split("_", 2)[2] for _, _, run in teacher_runs}) == 1
    if evaluate:
        assert len(set(manifests)) == 1
        assert Path(manifests[0].split("=", 1)[1]).is_absolute()
        evaluated = [
            Path(cmd[cmd.index("--checkpoint") + 1]) for stage, cmd in calls if stage == "eval"
        ]
        expected = {run / "teacher_final.pt" for _, _, run in teacher_runs}
        if distill:
            expected |= {run / "student/student_final.pt" for _, _, run in teacher_runs}
        assert len(evaluated) == len(expected)
        assert set(evaluated) == expected
    else:
        assert not list(tmp_path.rglob("scenes.json"))


def test_defaults_preserve_hydra_parameters(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sharpa-compare"])
    calls = []
    monkeypatch.setattr(compare_rl_algo.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    compare_rl_algo.main()
    assert len(calls) == 3
    for algorithm, command in zip(("ppo", "appo", "flashsac"), calls):
        cfg = compare_rl_algo.compose_config(algorithm, "mujoco", [])
        assert command[3:6] == ["--algo", algorithm, f"algo.seed={cfg.algo.seed}"]
        assert len(command) == 7
        assert command[6].startswith("training.log_dir=logs/compare/")


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_seed_count_does_not_launch(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["sharpa-compare", "--num-seeds", value])
    monkeypatch.setattr(
        compare_rl_algo.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected launch")
    )
    with pytest.raises(SystemExit) as error:
        compare_rl_algo.main()
    assert error.value.code == 2


def test_teacher_failure_stops_before_distillation_and_evaluation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sharpa-compare", "--distill", "--eval", "--num-seeds", "2"])
    calls = []

    def fail(command, *, check):
        calls.append(command)
        if len(calls) == 4:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(compare_rl_algo.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        compare_rl_algo.main()
    assert len(calls) == 4
    assert all(command[2] == "sharpa_rl_unilab.cli" for command in calls)
