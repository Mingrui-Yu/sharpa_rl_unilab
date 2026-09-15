"""CPU-only offline evaluations, after the entire training matrix has stopped."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time

from adapter import OUT, ROOT, write


def evaluate_group(group, index):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4",
    )
    run = OUT / f"{group}_seed1"
    dest = OUT / "evaluation"
    dest.mkdir(exist_ok=True)
    commands = []
    for iteration in [501, 100, 200, 300, 400, 500]:
        checkpoint = run / (
            "teacher_final.pt" if iteration == 501 else f"teacher_iteration_{iteration}.pt"
        )
        output = dest / f"{group}_{iteration}.json"
        command = [
            "taskset",
            "-c",
            f"{128 + 4 * index}-{131 + 4 * index}",
            sys.executable,
            str(ROOT / "experiments/appo_g_ablation/evaluate.py"),
            str(checkpoint),
            str(output),
        ]
        if iteration != 501:
            command.append("--subset")
        commands.append(command)
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(
                command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print(f"{group} {iteration} evaluated {time.ctime()}", flush=True)
    write(
        dest / f"{group}_commands.json",
        {
            "commands": commands,
            "device": "cpu",
            "affinity": list(range(128 + 4 * index, 132 + 4 * index)),
        },
    )


def main():
    last = OUT / "N_exit.json"
    while not last.exists():
        time.sleep(10)
    if json.loads(last.read_text())["returncode"] != 0:
        raise RuntimeError("Training failed")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(evaluate_group, g, i) for i, g in enumerate("RABDCN")]
        for f in jobs:
            f.result()
    write(OUT / "evaluation/complete.json", {"wall_seconds": time.time() - start})


if __name__ == "__main__":
    main()
