"""Mediation and independent numerical restorations, with matched-scene diagnostics."""

import csv
import json

import numpy as np
import torch
from adapter import OUT, write
from analyze import compare, lines, train_summary, weighted


def main():
    names = [r["name"] for r in json.loads((OUT / "manifest.json").read_text())["additional_runs"]]
    assert len(names) <= 4
    results = {
        name: json.loads((OUT / f"evaluation/{name}_501.json").read_text())
        for name in ["R", "A", "C", "D", *names]
    }
    pairs = [
        ("C", "C_fixed_lr"),
        ("C_fixed_lr", "D_fixed_lr"),
        ("D", "D_fixed_lr"),
        ("A", "A_learner_density"),
        ("R", "A_learner_density"),
        ("A", "A_target_density"),
        ("R", "A_target_density"),
    ]
    rows = []
    for name in names:
        result = results[name]
        training, _, _ = train_summary(name)
        rows.append(
            {
                "group": name,
                **result["summary"],
                **result["diagnostics"]["summary"],
                "height_exits": sum(int(e["dropped"]) for e in result["episodes"]),
                **training,
                "branch_interpretation": "native counterfactual only"
                if "fixed_lr" in name
                else "applied adaptive schedule",
            }
        )
    with (OUT / "additional_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoints = {
        name: torch.load(
            OUT / f"{name}_seed1/teacher_final.pt", weights_only=False, map_location="cpu"
        )
        for name in ["C", *names[:2]]
    }
    equal = {}
    for a, b in pairs[:2]:
        equal[f"{b}-{a}"] = {
            key: all(
                torch.equal(checkpoints[a][key][k], checkpoints[b][key][k])
                for k in checkpoints[a][key]
            )
            for key in ["actor", "critic", "target_actor"]
        }
    sequences = {
        name: [
            m["lr_after"]
            for u in lines(OUT / f"{name}_seed1/updates.jsonl")
            for m in u["minibatches"]
        ]
        for name in ["C", *names[:2]]
    }
    equal["LR_sequences"] = all(sequences[name] == sequences["C"] for name in names[:2])
    traces = {
        name: np.load(OUT / f"evaluation/{name}_501.trajectories.npz") for name in ["C", *names[:2]]
    }
    equal["evaluation_trajectory_arrays"] = all(
        np.array_equal(traces["C"][k], traces[name][k])
        for name in names[:2]
        for k in traces["C"].files
    )
    core = json.loads((OUT / "analysis.json").read_text())
    by_group = {r["group"]: r for r in core["summaries"] + rows}
    keys = [
        "speed_fixed_window",
        "height_exits",
        "survival_seconds",
        "q_second_difference_rms",
        "std",
        "saturation",
        "action_delta_rms",
        "target_at_limit",
    ]
    effects = {
        f"{b}-{a}": {key: by_group[b][key] - by_group[a][key] for key in keys} for a, b in pairs
    }
    paired = {f"{b}-{a}": compare(a, b, results) for a, b in pairs}
    packet_reference = lines(OUT / "R_seed1/updates.jsonl")
    packet_matches = {
        name: all(
            a["packets"] == b["packets"]
            for a, b in zip(packet_reference, lines(OUT / f"{name}_seed1/updates.jsonl"))
        )
        for name in names
    }
    updates_by_group = {name: lines(OUT / f"{name}_seed1/updates.jsonl") for name in results}
    assert all(len(updates) == 501 for updates in updates_by_group.values())
    divergences = {}
    for a, b in pairs:

        def first(predicate):
            return next(
                (
                    left["iteration"]
                    for left, right in zip(updates_by_group[a], updates_by_group[b])
                    if predicate(left, right)
                ),
                None,
            )

        divergences[f"{b}-{a}"] = {
            "first_std_difference": first(lambda left, right: left["std"] != right["std"]),
            "first_std_reference_tolerance_exceeded": first(
                lambda left, right: not np.allclose(left["std"], right["std"], atol=1e-6, rtol=1e-5)
            ),
            "first_lr_difference": first(
                lambda left, right: (
                    [m["lr_after"] for m in left["minibatches"]]
                    != [m["lr_after"] for m in right["minibatches"]]
                )
            ),
            "first_packet_version_difference": first(
                lambda left, right: left["packets"] != right["packets"]
            ),
            "first_staging_difference": first(
                lambda left, right: left["staging_versions"] != right["staging_versions"]
            ),
        }
    write(
        OUT / "additional_analysis.json",
        {
            "summaries": rows,
            "effects": effects,
            "paired_scenes": paired,
            "mediation_equalities": equal,
            "packet_trace_matches_R": packet_matches,
            "divergences": divergences,
            "failures": {
                name: [e["episode_id"] for e in results[name]["episodes"] if e["dropped"]]
                for name in names
            },
        },
    )
    import matplotlib.pyplot as plt

    scene_ids = json.loads((OUT / "manifest.json").read_text())["intermediate_24_ids"]
    curve_rows = []
    for label, groups in [
        ("mediation", ["C", "D", "C_fixed_lr", "D_fixed_lr"]),
        ("numerical", ["R", "A", "A_learner_density", "A_target_density"]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for group in groups:
            updates = lines(OUT / f"{group}_seed1/updates.jsonl")
            axes[0, 0].plot(
                [row["iteration"] for row in updates],
                [np.mean([mb["lr_after"] for mb in row["minibatches"]]) for row in updates],
                label=group,
                alpha=0.8,
            )
            axes[0, 1].plot(
                [row["iteration"] for row in updates],
                [np.mean(row["std"]) for row in updates],
                label=group,
            )
            selected = []
            for iteration in [100, 200, 300, 400, 500, 501]:
                result = json.loads((OUT / f"evaluation/{group}_{iteration}.json").read_text())
                episodes = [row for row in result["episodes"] if row["episode_id"] in scene_ids]
                row = {
                    "group": group,
                    "iteration": iteration,
                    "speed": weighted(episodes, "speed_fixed_window"),
                    "exits": sum(row["dropped"] for row in episodes),
                }
                selected.append(row)
                curve_rows.append(row)
            axes[1, 0].plot(
                [row["iteration"] for row in selected],
                [row["speed"] for row in selected],
                marker=".",
                label=group,
            )
            axes[1, 1].plot(
                [row["iteration"] for row in selected],
                [row["exits"] for row in selected],
                marker=".",
                label=group,
            )
        for ax, title in zip(
            axes.flat,
            [
                "Applied mean minibatch LR",
                "Mean std",
                "Speed [rad/s], frozen 24 scenes",
                "Height exits / 24",
            ],
        ):
            ax.set(xlabel="iteration", title=title)
            ax.grid(alpha=0.2)
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"additional_{label}_curves.png", dpi=150)
        plt.close(fig)
    write(OUT / "additional_intermediate_curves.json", curve_rows)
    print(
        json.dumps(
            {
                "summaries": rows,
                "effects": effects,
                "mediation_equalities": equal,
                "packet_trace_matches_R": packet_matches,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
