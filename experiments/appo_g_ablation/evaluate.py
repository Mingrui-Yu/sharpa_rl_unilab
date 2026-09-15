"""Unchanged production evaluator plus passive trajectories and frozen subset selection."""

import argparse
import json
from pathlib import Path

import numpy as np
from adapter import OUT

from sharpa_rl_unilab.training import evaluation as ev


def evaluate(checkpoint, output, subset=False):
    traces = []
    original_diagnostics = ev.EpisodeActionDiagnostics

    class TraceDiagnostics(original_diagnostics):
        def __init__(self, initial_q):
            super().__init__(initial_q)
            self.trace = {
                "q": [np.asarray(initial_q).copy()],
                "action": [],
                "std": [],
                "target": [],
            }

        def record(self, action, std, q_next, target, lower, upper):
            super().record(action, std, q_next, target, lower, upper)
            for k, v in [("q", q_next), ("action", action), ("std", std), ("target", target)]:
                self.trace[k].append(np.asarray(v).copy())

        def result(self):
            traces.append({k: np.concatenate(v, axis=0) for k, v in self.trace.items()})
            return super().result()

    original_manifest = ev.load_manifest

    def load_manifest(cfg, path):
        full = original_manifest(cfg, path)
        if subset:
            ids = json.loads((OUT / "manifest.json").read_text())["intermediate_24_ids"]
            return dict(full, episodes=[e for e in full["episodes"] if e["episode_id"] in ids])
        return full

    ev.EpisodeActionDiagnostics = TraceDiagnostics
    ev.load_manifest = load_manifest
    try:
        result = ev.evaluate_checkpoint(
            checkpoint,
            output=output,
            device="cpu",
            evaluation={"diagnostics": True, "manifest": str(OUT / "scenes.json")},
        )
        assert len(traces) == len(result["episodes"]) == (24 if subset else 240)
        arrays = {
            e["episode_id"] + "__" + k: v
            for e, t in zip(result["episodes"], traces)
            for k, v in t.items()
        }
        np.savez_compressed(output.with_suffix(".trajectories.npz"), **arrays)
        if subset:
            result["subset_ids"] = [e["episode_id"] for e in result["episodes"]]
            ev.write_json(output, result)
        return result
    finally:
        ev.EpisodeActionDiagnostics = original_diagnostics
        ev.load_manifest = original_manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--subset", action="store_true")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(a.checkpoint, a.output, a.subset)
