"""Four evidence-triggered full runs, with short validation before frozen execution."""

import json
import os
import subprocess
import sys
import time

from adapter import OUT, ROOT, write
from freeze import hashes

JOBS = [
    ("C_fixed_lr", "train_fixed_lr.py", ["C", "--reference", "C"]),
    ("D_fixed_lr", "train_fixed_lr.py", ["D", "--reference", "C"]),
    ("A_learner_density", "train_learner_density.py", []),
    ("A_target_density", "train_target_density.py", []),
]


def execute(name, script, args, env):
    command = [
        "taskset",
        "-c",
        "0-127",
        sys.executable,
        str(ROOT / "experiments/appo_g_ablation" / script),
        *args,
    ]
    write(
        OUT / f"{name}_command.json",
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
    print(name, "started", time.ctime(), flush=True)
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write(
        OUT / f"{name}_exit.json",
        {"returncode": result.returncode, "wall_seconds": time.time() - start},
    )
    if result.returncode:
        raise RuntimeError(f"{name} failed")
    print(name, "completed", time.ctime(), flush=True)


def main():
    while (
        not (OUT / "evaluation/complete.json").exists()
        or not (OUT / "videos/complete.json").exists()
    ):
        time.sleep(5)
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert len(JOBS) == 4 and not manifest.get("additional_runs")
    manifest["additional_runs"] = [
        {"name": name, "script": script, "args": args} for name, script, args in JOBS
    ]
    manifest["additional_triggers"] = {
        "fixed_LR": "B-A and D-C both lower speed and increase exits; all 501 packet/version traces agree. D-C has the larger speed effect (-0.0327998 rad/s), so C/D replay the predeclared C reference frozen before evaluation.",
        "numeric": "A-R=-0.0182752 rad/s, +2 exits with identical std parameterization/KL, initialization and packet traces. Fixed-data probes identify learner density backward roundoff and target Normal.log_prob forward/V-trace roundoff. Each restoration changes one path independently from A.",
        "async_repeats": "Not triggered: R/A and D/N have identical packet/version traces; D/N also have identical parameters, minibatch diagnostics and final metrics.",
    }
    manifest["additional_code_hashes"] = hashes(ROOT / "experiments/appo_g_ablation")
    write(OUT / "manifest.json", manifest)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4",
    )
    for name, script, args in JOBS:
        execute(
            "smoke_" + name, script, [*args, "--iterations", "2", "--name", "smoke_" + name], env
        )
    # Verify the mediation hooks actually produce identical models before full training.
    import torch

    a = torch.load(
        OUT / "smoke_C_fixed_lr/teacher_final.pt", weights_only=False, map_location="cpu"
    )
    b = torch.load(
        OUT / "smoke_D_fixed_lr/teacher_final.pt", weights_only=False, map_location="cpu"
    )
    for key in ["actor", "critic", "target_actor"]:
        assert all(torch.equal(a[key][k], b[key][k]) for k in a[key])
    from sharpa_rl_unilab.training.teacher_runtime import load_policy

    for name, _, _ in JOBS:
        load_policy(OUT / f"smoke_{name}/teacher_final.pt", "cpu", configure_runtime=False)
    write(
        OUT / "audit/additional_smoke.json",
        {"load_all": True, "fixed_LR_C_D_models_identical": True},
    )
    for name, script, args in JOBS:
        assert not (OUT / f"{name}_seed1").exists()
        execute(name, script, args, env)
    write(OUT / "additional_complete.json", {"runs": [job[0] for job in JOBS]})


if __name__ == "__main__":
    main()
