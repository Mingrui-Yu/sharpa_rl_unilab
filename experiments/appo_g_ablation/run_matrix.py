"""Sequential core matrix; refuses to overwrite runs and preserves every command."""

import os
import subprocess
import sys
import time

from adapter import OUT, ROOT, write


def main():
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4",
    )
    for g in "RABDCN":
        run = OUT / f"{g}_seed1"
        if run.exists():
            raise RuntimeError(f"Refusing to overwrite {run}")
        command = [
            "taskset",
            "-c",
            "0-127",
            sys.executable,
            str(ROOT / "experiments/appo_g_ablation/train.py"),
            g,
        ]
        write(
            OUT / f"{g}_command.json",
            {
                "argv": command,
                "env": {
                    k: env[k]
                    for k in [
                        "CUDA_VISIBLE_DEVICES",
                        "PYTHONPATH",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    ]
                },
            },
        )
        start = time.time()
        print(f"{g} started {time.ctime()}", flush=True)
        with (OUT / f"{g}.log").open("w") as log:
            r = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(
            OUT / f"{g}_exit.json",
            {"returncode": r.returncode, "wall_seconds": time.time() - start},
        )
        if r.returncode:
            raise RuntimeError(f"{g} failed; inspect {OUT}/{g}.log")
        print(f"{g} complete {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
