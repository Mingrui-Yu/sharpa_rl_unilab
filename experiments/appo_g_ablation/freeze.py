"""Freeze inputs before running the single-seed APPO experiment."""

import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/appo_g_ablation"
OLD = ROOT.parent / "sharpa_rl_unilab-action-clipping"
NEW = ROOT.parent / "sharpa_rl_unilab-issue-2"
HISTORY = {
    "original": OLD / "logs/action_clipping_phase2/appo_G_seed1/teacher_final.pt",
    "current": NEW / "logs/actor_distribution/appo_G_log_std_seed1/teacher_final.pt",
}


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def hashes(root):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "__pycache__" not in str(p)
        and p.suffix in {".py", ".so", ".xml", ".npy", ".STL", ".yaml"}
    }


def main():
    manifest = {
        "order": list("RABDCN"),
        "iterations": 501,
        "train_seed": 1,
        "collector_seed": 2,
        "gpu": 0,
        "affinity": list(range(128)),
        "torch_threads": 4,
        "R_provenance": "Reconstruction: surviving dirty worktree frozen now; no contemporaneous full source hash was archived. Original actor/loss/collector with current unified runtime.",
        "history": {},
        "sources": {},
        "dependencies": {},
        "groups": {
            "R": ["frozen original", "S", "clamp", "old"],
            "A": ["current", "S", "clamp", "old"],
            "B": ["current", "S", "clamp", "exact"],
            "C": ["current", "L", "wide", "old"],
            "D": ["current", "L", "wide", "exact"],
            "N": ["current", "L", "none", "exact"],
        },
    }
    for label, path in HISTORY.items():
        c = torch.load(path, map_location="cpu", weights_only=False)
        write(OUT / "audit" / f"{label}_config.json", c["config"])
        manifest["history"][label] = {
            "checkpoint": str(path),
            "sha256": digest(path),
            "counters": c["counters"],
        }
    for label, root in [("original", OLD), ("current", ROOT)]:
        dest = OUT / "frozen" / label
        shutil.copytree(
            root / "src",
            dest / "src",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (dest / "dirty.diff").write_bytes(
            subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        )
        manifest["sources"][label] = {
            "revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "files": hashes(dest / "src"),
        }
    for name in ["uni_rl", "rsl_rl", "unilab", "torch", "mujoco", "unisim", "mjbatch"]:
        module = importlib.import_module(name)
        root = Path(module.__file__).parent
        manifest["dependencies"][name] = {"path": str(root), "files": hashes(root)}
    scenes = NEW / "logs/actor_distribution/scenes.json"
    shutil.copy2(scenes, OUT / "scenes.json")
    assert digest(scenes) == "e592d62b6f716a30d3033938debed34b0985e771cfdae72682caabba09b3510a"
    manifest["scenes_sha256"] = digest(scenes)
    data = json.loads(scenes.read_text())
    subset = [e["episode_id"] for e in data["episodes"] if e["episode_id"].endswith("/0")]
    assert len(subset) == 24
    manifest["intermediate_24_ids"] = subset
    manifest["video_scene"] = "1/10001/0"
    manifest["device_info"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"], text=True
    )
    manifest["numeric_runtime"] = {
        "torch": torch.__version__,
        "default_dtype": str(torch.get_default_dtype()),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "autocast": False,
        "compile": False,
    }
    write(OUT / "manifest.json", manifest)

    def flatten(x, prefix=""):
        if not isinstance(x, dict):
            return {prefix: x}
        return {
            k: v
            for a, b in x.items()
            for k, v in flatten(b, prefix + "." + a if prefix else a).items()
        }

    a, b = [
        flatten(json.loads((OUT / "audit" / f"{label}_config.json").read_text()))
        for label in ["original", "current"]
    ]
    write(
        OUT / "audit/config_diff.json",
        {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)},
    )
    print("Frozen:", OUT)


if __name__ == "__main__":
    main()
