"""Record the predeclared scene, serially, after core training finishes."""

import json
import os
import subprocess
import sys
import time

from adapter import OUT, ROOT, write


def main():
    while not (OUT / "N_exit.json").exists():
        time.sleep(10)
    assert json.loads((OUT / "N_exit.json").read_text())["returncode"] == 0
    dest = OUT / "videos"
    dest.mkdir(exist_ok=True)
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
    commands = []
    for g in "RABDCN":
        command = [
            "taskset",
            "-c",
            "152-155",
            sys.executable,
            str(ROOT / "experiments/appo_g_ablation/video.py"),
            str(OUT / f"{g}_seed1/teacher_final.pt"),
            "--manifest",
            str(OUT / "scenes.json"),
            "--episode",
            "1/10001/0",
            "--output",
            str(dest / f"{g}.mp4"),
        ]
        commands.append(command)
        with (dest / f"{g}.log").open("w") as log:
            subprocess.run(
                command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print(g, "video complete", flush=True)
    write(dest / "complete.json", {"commands": commands, "scene": "1/10001/0"})


if __name__ == "__main__":
    main()
