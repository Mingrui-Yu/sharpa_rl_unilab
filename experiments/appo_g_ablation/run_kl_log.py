"""New user-requested follow-up: one full N run, separate from the completed matrix."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

import torch
from adapter import OUT, ROOT, write
from freeze import digest, hashes
from run_additional import execute

NAME = "N_kl_log_epsilon"


def main():
    manifest_path = OUT / "kl_log_manifest.json"
    assert not manifest_path.exists()
    baseline = json.loads((OUT / "manifest.json").read_text())
    assert hashes(ROOT / "src") == baseline["sources"]["current"]["files"]
    for dependency in baseline["dependencies"].values():
        from pathlib import Path

        current = hashes(Path(dependency["path"]))
        assert all(current.get(k) == v for k, v in dependency["files"].items())
    manifest = {
        "status": "running",
        "authorization": "Latest explicit user selection: N setup, restore only original log(std/old_std + 1e-5) term",
        "name": NAME,
        "mode": "original_log",
        "epsilon": 1e-5,
        "definition": "Only replace log(std)-log(old_std) by log(std/old_std + 1e-5); retain new normalized-square operation order",
        "limitation": "Log term intervention includes its original ratio/log rounding; not a full restoration of old normalized-square order or a zero-KL-only scheduler change",
        "training_seed": 1,
        "collector_seed": 2,
        "iterations": 501,
        "full_runs": 1,
        "reference_groups": ["N", "R", "C"],
        "baseline_manifest_sha256": digest(OUT / "manifest.json"),
        "scenes_sha256": digest(OUT / "scenes.json"),
        "code_hashes": hashes(ROOT / "experiments/appo_g_ablation"),
        "main_checkpoint": "teacher_final.pt",
    }
    write(manifest_path, manifest)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4",
    )
    execute(
        "smoke_N_kl_log_epsilon",
        "train_kl_bias.py",
        ["--mode", "original_log", "--iterations", "2", "--name", "smoke_N_kl_log_epsilon"],
        env,
    )
    control = torch.load(
        OUT / "smoke_N_bias_control/teacher_final.pt", weights_only=False, map_location="cpu"
    )
    zero = torch.load(
        OUT / "smoke_N_bias_zero/teacher_final.pt", weights_only=False, map_location="cpu"
    )
    equality = {
        key: all(torch.equal(control[key][k], zero[key][k]) for k in control[key])
        for key in ["actor", "critic", "target_actor"]
    }
    assert all(equality.values())
    from sharpa_rl_unilab.training.teacher_runtime import load_policy

    load_policy(OUT / "smoke_N_kl_log_epsilon/teacher_final.pt", "cpu", configure_runtime=False)
    update = json.loads((OUT / "smoke_N_kl_log_epsilon/updates.jsonl").read_text().splitlines()[0])
    first = update["minibatches"][0]
    assert first["exact_kl"] == 0 and first["branch"] == "up"
    assert first["lr_after"] == first["lr_before"] * 1.1
    write(
        OUT / "audit/kl_log_smoke.json",
        {"zero_hook_matches_N": equality, "positive_first_minibatch": first, "save_load": True},
    )
    execute(NAME, "train_kl_bias.py", ["--mode", "original_log", "--name", NAME + "_seed1"], env)
    manifest["status"] = "evaluating"
    write(manifest_path, manifest)
    evaluation_env = dict(env, CUDA_VISIBLE_DEVICES="")

    def evaluate(iteration, index):
        run = OUT / f"{NAME}_seed1"
        checkpoint = run / (
            "teacher_final.pt" if iteration == 501 else f"teacher_iteration_{iteration}.pt"
        )
        output = OUT / f"evaluation/{NAME}_{iteration}.json"
        command = [
            "taskset",
            "-c",
            f"{128 + index * 4}-{131 + index * 4}",
            sys.executable,
            str(ROOT / "experiments/appo_g_ablation/evaluate.py"),
            str(checkpoint),
            str(output),
        ]
        if iteration != 501:
            command.append("--subset")
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(
                command,
                env=evaluation_env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(NAME, iteration, "evaluated", time.ctime(), flush=True)
        return command

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [
            pool.submit(evaluate, iteration, index)
            for index, iteration in enumerate([501, 100, 200, 300, 400, 500])
        ]
        commands = [task.result() for task in tasks]
    write(OUT / "evaluation/kl_log_complete.json", {"commands": commands})
    manifest["status"] = "evaluated"
    write(manifest_path, manifest)


if __name__ == "__main__":
    main()
