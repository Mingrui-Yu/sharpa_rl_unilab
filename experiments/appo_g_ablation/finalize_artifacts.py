"""Compact tracked evidence and cross-scale tables after every evaluation completes."""

import csv
import json
from pathlib import Path

from adapter import OUT, ROOT, write
from freeze import digest, hashes


def main():
    core = json.loads((OUT / "analysis.json").read_text())
    extra = json.loads((OUT / "additional_analysis.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    names = list("RABCDN") + [run["name"] for run in manifest["additional_runs"]]
    assert len(names) == 10
    rows = []
    for name in names:
        result = json.loads((OUT / f"evaluation/{name}_501.json").read_text())
        for scale, summary in result["per_scale"].items():
            episodes = [row for row in result["episodes"] if row["scale"] == float(scale)]
            rows.append(
                {
                    "group": name,
                    "scale": float(scale),
                    "episodes": len(episodes),
                    "height_exits": sum(int(row["dropped"]) for row in episodes),
                    **summary,
                    **result["diagnostics"]["per_scale"][scale],
                }
            )
    with (OUT / "per_scale_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def compact(analysis):
        return {
            **analysis,
            "paired_scenes": {
                key: {k: v for k, v in value.items() if k != "scenes"}
                for key, value in analysis["paired_scenes"].items()
            },
        }

    tracked = ROOT / "docs/validation/appo-g-ablation-20260916.json"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "training_seed": 1,
        "collector_seed": 2,
        "scope": "Six core plus four conditional runs, 501 iterations each; one training seed only.",
        "artifact_root": str(OUT),
        "manifest_sha256": digest(OUT / "manifest.json"),
        "core": compact(core),
        "additional": compact(extra),
        "audit": {
            name: json.loads((OUT / "audit" / name).read_text())
            for name in [
                "N_weight_and_trace_comparison.json",
                "historical_R_weight_comparison.json",
                "historical_current_D_weight_comparison.json",
                "fixed_data_report.json",
                "density_residual.json",
                "mediation_all_snapshots.json",
                "posthoc_video_trajectory_check.json",
                "numeric_components_cpu.json",
                "numeric_components_cuda.json",
                "final_verification.json",
            ]
        },
    }
    write(tracked, evidence)
    checks = {}
    for name, definition in manifest["dependencies"].items():
        current = hashes(Path(definition["path"]))
        changed = [key for key, old in definition["files"].items() if current.get(key) != old]
        checks[name] = {"changed_files": changed, "unchanged": not changed}
    current_experiment = hashes(ROOT / "experiments/appo_g_ablation")
    for label, entrypoint_names in [
        ("experiment_code_hashes", ["adapter.py", "train.py", "run_matrix.py"]),
        (
            "additional_code_hashes",
            [
                "adapter.py",
                "train_fixed_lr.py",
                "train_learner_density.py",
                "train_target_density.py",
                "run_additional.py",
            ],
        ),
    ]:
        assert all(current_experiment[name] == manifest[label][name] for name in entrypoint_names)
    checks["training_entrypoints"] = {"unchanged": True}
    current_source = hashes(ROOT / "src")
    checks["production_source"] = {
        "unchanged": current_source == manifest["sources"]["current"]["files"]
    }
    checks["history"] = {
        name: digest(definition["checkpoint"]) == definition["sha256"]
        for name, definition in manifest["history"].items()
    }
    assert all(value.get("unchanged", True) for value in checks.values())
    assert all(checks["history"].values())
    write(OUT / "audit/final_source_integrity.json", checks)
    evidence["audit"]["final_source_integrity.json"] = checks
    artifacts = {}
    for name in names:
        for path in sorted((OUT / f"{name}_seed1").glob("*.pt")):
            artifacts[str(path.relative_to(OUT))] = digest(path)
        for path in sorted((OUT / "evaluation").glob(f"{name}_*.json")):
            artifacts[str(path.relative_to(OUT))] = digest(path)
    artifacts["scenes.json"] = digest(OUT / "scenes.json")
    for path in sorted((OUT / "evaluation").glob("*.trajectories.npz")):
        artifacts[str(path.relative_to(OUT))] = digest(path)
    write(OUT / "artifact_hashes.json", artifacts)
    manifest["final_experiment_code_hashes"] = hashes(ROOT / "experiments/appo_g_ablation")
    manifest["status"] = "complete"
    write(OUT / "manifest.json", manifest)
    evidence["manifest_sha256"] = digest(OUT / "manifest.json")
    write(tracked, evidence)
    print("Verified source integrity and wrote", tracked)


if __name__ == "__main__":
    main()
