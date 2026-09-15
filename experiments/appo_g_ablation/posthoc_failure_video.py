"""Post hoc diagnostic: earliest N height exit, paired with R; never a selection metric."""

import json
import os
import subprocess
import sys
import time

from adapter import OUT, ROOT, write


def main():
    while not (OUT / "additional_complete.json").exists():
        time.sleep(10)
    evaluation = json.loads((OUT / "evaluation/N_501.json").read_text())
    failure = min(
        (row for row in evaluation["episodes"] if row["dropped"]),
        key=lambda row: (row["survival_seconds"], row["episode_id"]),
    )
    dest = OUT / "videos"
    commands = []
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES="0",
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for group in ["R", "N"]:
        cmd = [
            "taskset",
            "-c",
            "152-155",
            sys.executable,
            str(ROOT / "experiments/appo_g_ablation/video.py"),
            str(OUT / f"{group}_seed1/teacher_final.pt"),
            "--manifest",
            str(OUT / "scenes.json"),
            "--episode",
            failure["episode_id"],
            "--output",
            str(dest / f"posthoc_failure_{group}.mp4"),
        ]
        commands.append(cmd)
        with (dest / f"posthoc_failure_{group}.log").open("w") as log:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(
        dest / "posthoc_failure.json",
        {
            "selection": "Post hoc: earliest height exit of N; diagnostic only",
            "episode_id": failure["episode_id"],
            "N_survival_seconds": failure["survival_seconds"],
            "commands": commands,
        },
    )


if __name__ == "__main__":
    main()
