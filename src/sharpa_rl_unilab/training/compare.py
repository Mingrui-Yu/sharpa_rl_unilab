"""Run the same teacher/student protocol across algorithms and training seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sharpa_rl_unilab.cli import compose_config

from .evaluation import (
    aggregate_seeds,
    evaluate_checkpoint,
    make_manifest,
    plot_learning_curves,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["ppo", "appo", "flashsac"],
        default=["ppo", "appo", "flashsac"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate the pipeline at 128 teacher / 32 student transitions; no performance ranking",
    )
    parser.add_argument("--nodr", action="store_true")
    args, overrides = parser.parse_known_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        parser.error("Use at least three distinct training seeds")
    if any(item.lstrip("+").startswith("algo.max_iterations=") for item in overrides):
        parser.error("sharpa-compare uses sampling budgets; set budget.transitions instead")
    args.output.mkdir(parents=True, exist_ok=False)
    common = [f"hardware.device={args.device}", *overrides]
    common += [f"evaluation.training_seeds={args.seeds}", f"distillation.seeds={args.seeds}"]
    # Comparisons explicitly use equal sampling budgets across algorithms.
    # APPO's standalone HORA default instead stops after 305 learner rounds.
    sample_budget = "128" if args.smoke else "10000000"
    if not any(item.startswith("budget.transitions=") for item in common):
        common.append(f"budget.transitions={sample_budget}")
    if args.smoke:
        common += [
            "hardware.num_envs=8",
            "hardware.torch_threads=2",
            "budget.evaluate_every=0",
            "distillation.transitions=32",
            "distillation.num_envs=8",
            "distillation.save_every=0",
            "evaluation.seeds=[10001]",
            "evaluation.episodes_per_scale=1",
        ]
    manifest = args.output.resolve() / "scenes.json"
    cfg = compose_config(args.algorithms[0], "mujoco", common, nodr=args.nodr)
    make_manifest(cfg, manifest)
    evaluations = []
    summaries = []
    for algorithm in args.algorithms:
        teacher_results, student_results = [], []
        for seed in args.seeds:
            run = args.output.resolve() / algorithm / f"seed_{seed}"
            specific = [
                f"algo.seed={seed}",
                f"training.log_dir={run}",
                f"evaluation.manifest={manifest}",
            ]
            if args.smoke and algorithm == "flashsac":
                specific += ["algo.batch_size=32"]
            if algorithm == "appo":
                specific += ["algo.max_iterations=null", "budget.save_every=null"]
                if args.smoke:
                    specific += ["algo.save_interval=0"]
            elif args.smoke:
                specific += ["budget.save_every=0"]
            cfg = compose_config(algorithm, "mujoco", [*common, *specific], nodr=args.nodr)
            # A fresh process per stage keeps CUDA/module initialization from
            # being charged only to the first algorithm or training seed.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharpa_rl_unilab.cli",
                    "--algo",
                    algorithm,
                    *(["--nodr"] if args.nodr else []),
                    *common,
                    *specific,
                ],
                check=True,
            )
            teacher = run / "teacher_final.pt"
            teacher_result = run / "evaluation_final.json"
            evaluate_checkpoint(teacher, output=teacher_result, device=args.device)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharpa_rl_unilab.training.student_runtime",
                    "--checkpoint",
                    str(teacher),
                    "--device",
                    args.device,
                    f"training.log_dir={run / 'student'}",
                ],
                check=True,
            )
            student = run / "student" / "student_final.pt"
            student_result = run / "evaluation_student.json"
            evaluate_checkpoint(student, output=student_result, device=args.device)
            teacher_results.append(teacher_result)
            student_results.append(student_result)
        for stage, paths in (("teacher", teacher_results), ("student", student_results)):
            summary = aggregate_seeds(paths, args.output / f"{algorithm}_{stage}.json")
            summary["kind"] = "pipeline-smoke" if args.smoke else "comparison"
            summaries.append(summary)
            evaluations.extend(paths)
    write_json(
        args.output / "summary.json",
        {"kind": "pipeline-smoke" if args.smoke else "comparison", "groups": summaries},
    )
    # Full runs include intermediate teacher snapshots; final student points show
    # its distillation cost separately from the teacher cost stored in each JSON.
    curves = list(args.output.glob("*/seed_*/evaluation_*.json"))
    plot_learning_curves(curves or evaluations, args.output / "learning_curves.png")


if __name__ == "__main__":
    main()
