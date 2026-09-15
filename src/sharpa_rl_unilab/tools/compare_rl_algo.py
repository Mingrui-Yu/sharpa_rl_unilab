"""Train PPO, APPO and FlashSAC with Hydra defaults, optionally distill and evaluate."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sharpa_rl_unilab.cli import compose_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distill", action="store_true", help="Distill after all teachers finish")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate all trained models on shared scenes"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds starting at each algorithm's Hydra seed (default: 1)",
    )
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    configs = [compose_config(algorithm, "mujoco", []) for algorithm in ("ppo", "appo", "flashsac")]
    runs = [
        (
            cfg.algo.algo,
            int(cfg.algo.seed) + offset,
            Path("logs/compare")
            / f"seed_{int(cfg.algo.seed) + offset}_{timestamp}"
            / cfg.algo.algo,
        )
        for offset in range(args.num_seeds)
        for cfg in configs
    ]
    for algorithm, seed, run in runs:
        print(f"Training {algorithm} teacher, seed {seed}; logs: {run}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sharpa_rl_unilab.cli",
                "--algo",
                algorithm,
                f"algo.seed={seed}",
                f"training.log_dir={run}",
            ],
            check=True,
        )

    if args.distill:
        for algorithm, seed, run in runs:
            print(f"Training {algorithm} student, seed {seed}; logs: {run / 'student'}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharpa_rl_unilab.training.student_runtime",
                    "--checkpoint",
                    str(run / "teacher_final.pt"),
                    f"training.log_dir={run / 'student'}",
                ],
                check=True,
            )

    if args.eval:
        # The first evaluation creates the manifest; all later evaluations reuse it.
        manifest = (runs[0][2].parent / "scenes.json").resolve()
        for stage in ("teacher", "student") if args.distill else ("teacher",):
            for algorithm, seed, run in runs:
                checkpoint = (
                    run / "teacher_final.pt"
                    if stage == "teacher"
                    else run / "student/student_final.pt"
                )
                print(
                    f"Evaluating {algorithm} {stage}, seed {seed}; scenes: {manifest}", flush=True
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        # Invoke the sharpa-eval entry point with this interpreter.
                        "from sharpa_rl_unilab.cli import eval_main; eval_main()",
                        "--algo",
                        algorithm,
                        "--checkpoint",
                        str(checkpoint),
                        f"evaluation.manifest={manifest}",
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
