"""Paired scene evidence, factorial effects, and iteration/transition curves."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adapter import OUT, write

GROUPS = list("RABCDN")
PAIRS = [("R", "A"), ("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("D", "N"), ("R", "N")]


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def weighted(rows, key):
    return float(
        np.mean(
            [
                np.mean([r[key] for r in rows if r["scale"] == s])
                for s in sorted({r["scale"] for r in rows})
            ]
        )
    )


def train_summary(group):
    run = OUT / f"{group}_seed1"
    metrics = [r["metrics"] for r in lines(run / "runtime_metrics.jsonl")]
    updates = lines(run / "updates.jsonl")
    mb = [r for u in updates for r in u["minibatches"]]
    std = np.asarray([u["std"] for u in updates])
    final = metrics[-1]
    return (
        {
            "received": final["received"],
            "collected": final["collected"],
            "training_samples": final["training_samples"],
            "optimizer_updates": final["optimizer_updates"],
            "iterations": final["policy_version"],
            "training_seconds": final["wall_seconds"],
            "reuse_ratio": final["reuse_ratio"],
            "final_lr": mb[-1]["lr_after"],
            "lr_up": sum(r["branch"] == "up" for r in mb),
            "lr_down": sum(r["branch"] == "down" for r in mb),
            "lr_same": sum(r["branch"] == "same" for r in mb),
            "lr_actual_increases": sum(r["lr_after"] > r["lr_before"] for r in mb),
            "lr_actual_decreases": sum(r["lr_after"] < r["lr_before"] for r in mb),
            "lr_actual_unchanged": sum(r["lr_after"] == r["lr_before"] for r in mb),
            "shadow_branch_disagreements": sum(r["branch"] != r["shadow_branch"] for r in mb),
            "first_minibatch_disagreements": sum(
                u["minibatches"][0]["branch"] != u["minibatches"][0]["shadow_branch"]
                for u in updates
            ),
            "std_min_all": min(r["std_min"] for r in mb),
            "std_max_all": max(r["std_max"] for r in mb),
            "boundary_hits": sum(u["boundary_hits"] for u in updates),
            "grad_clip_fraction": float(np.mean([r["grad_norm"] > 1 for r in mb])),
            "mean_grad_norm": float(np.mean([r["grad_norm"] for r in mb])),
            "packets_total": sum(len(u["packets"]) for u in updates),
            "packet_merges": sum(len(u["packets"]) > 1 for u in updates),
            "mean_lag": float(np.mean([r["policy_lag_mean"] for r in metrics])),
            "final_std_median": float(np.median(std[-1])),
            "save_seconds": sum(r["seconds"] for r in lines(run / "saves.jsonl")),
        },
        metrics,
        updates,
    )


def compare(left, right, results):
    a, b = results[left], results[right]
    ea = {e["episode_id"]: e for e in a["episodes"]}
    eb = {e["episode_id"]: e for e in b["episodes"]}
    assert ea.keys() == eb.keys()
    ta = np.load(OUT / f"evaluation/{left}_501.trajectories.npz")
    tb = np.load(OUT / f"evaluation/{right}_501.trajectories.npz")
    contingency = {"both_survive": 0, "left_only_exit": 0, "right_only_exit": 0, "both_exit": 0}
    rows = []
    survivors = []
    for key, x in ea.items():
        y = eb[key]
        failed_x = bool(x["dropped"])
        failed_y = bool(y["dropped"])
        which = (
            "both_exit"
            if failed_x and failed_y
            else "left_only_exit"
            if failed_x
            else "right_only_exit"
            if failed_y
            else "both_survive"
        )
        contingency[which] += 1
        qa, qb = ta[key + "__q"], tb[key + "__q"]
        end = min(len(qa), len(qb))

        def rms(q):
            return float(np.sqrt(np.mean(np.diff(q.astype("float64"), n=2, axis=0) ** 2)))

        # Passive trace must reproduce production evaluation's per-episode diagnostic.
        assert np.isclose(
            rms(qa), x["diagnostics"]["q_second_difference_rms"], rtol=1e-7, atol=1e-10
        )
        row = {
            "episode_id": key,
            "scale": x["scale"],
            "common_seconds": (end - 1) * 0.05,
            "left_common_q_rms": rms(qa[:end]),
            "right_common_q_rms": rms(qb[:end]),
        }
        row["difference"] = row["right_common_q_rms"] - row["left_common_q_rms"]
        rows.append(row)
        if which == "both_survive":
            survivors.append(row)
    return {
        "contingency": contingency,
        "common_window": {
            k: weighted(rows, k) for k in ["left_common_q_rms", "right_common_q_rms", "difference"]
        },
        "both_full_survivor_subset": {
            "count": len(survivors),
            **{
                k: weighted(survivors, k)
                for k in ["left_common_q_rms", "right_common_q_rms", "difference"]
            },
        },
        "scenes": rows,
    }


def main():
    results = {g: json.loads((OUT / f"evaluation/{g}_501.json").read_text()) for g in GROUPS}
    summaries = {}
    metrics = {}
    updates = {}
    rows = []
    for g, result in results.items():
        summary, metrics[g], updates[g] = train_summary(g)
        summaries[g] = summary
        rows.append(
            {
                "group": g,
                **result["summary"],
                **result["diagnostics"]["summary"],
                "height_exits": sum(int(r["dropped"]) for r in result["episodes"]),
                **summary,
            }
        )
    with (OUT / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_group = {r["group"]: r for r in rows}
    columns = [
        "speed_fixed_window",
        "height_exits",
        "survival_seconds",
        "q_second_difference_rms",
        "std",
        "saturation",
        "action_delta_rms",
        "target_at_limit",
    ]
    effects = {f"{b}-{a}": {k: by_group[b][k] - by_group[a][k] for k in columns} for a, b in PAIRS}
    effects["interaction_(D-C)-(B-A)"] = {k: effects["D-C"][k] - effects["B-A"][k] for k in columns}
    comparisons = {f"{b}-{a}": compare(a, b, results) for a, b in PAIRS}
    divergences = {}
    for a, b in PAIRS:
        x, y = updates[a], updates[b]

        def first(predicate):
            return next(
                (i + 1 for i, (left, r) in enumerate(zip(x, y)) if predicate(left, r)), None
            )

        old_key = "old_kl" if a in "RAC" else "exact_kl"
        new_key = "old_kl" if b in "RAC" else "exact_kl"
        divergences[f"{b}-{a}"] = {
            "first_std_reference_tolerance_exceeded": first(
                lambda left, right: not np.allclose(left["std"], right["std"], atol=1e-6, rtol=1e-5)
            ),
            "first_selected_kl_difference": first(
                lambda left, right: (
                    [row[old_key] for row in left["minibatches"]]
                    != [row[new_key] for row in right["minibatches"]]
                )
            ),
            "first_selected_kl_reference_tolerance_exceeded": first(
                lambda left, right: (
                    not np.allclose(
                        [row[old_key] for row in left["minibatches"]],
                        [row[new_key] for row in right["minibatches"]],
                        atol=1e-6,
                        rtol=1e-5,
                    )
                )
            ),
            "first_packet_version_difference": first(
                lambda left, r: left["packets"] != r["packets"]
            ),
            "first_staging_difference": first(
                lambda left, r: left["staging_versions"] != r["staging_versions"]
            ),
            "first_std_difference": first(lambda left, r: left["std"] != r["std"]),
            "first_lr_difference": first(
                lambda left, r: (
                    [v["lr_after"] for v in left["minibatches"]]
                    != [v["lr_after"] for v in r["minibatches"]]
                )
            ),
        }
    write(
        OUT / "analysis.json",
        {
            "summaries": rows,
            "effects": effects,
            "paired_scenes": comparisons,
            "divergences": divergences,
            "failures": {
                g: [r["episode_id"] for r in d["episodes"] if r["dropped"]]
                for g, d in results.items()
            },
        },
    )
    for axis in ["iteration", "received"]:
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        for g in GROUPS:
            x = (
                np.arange(1, len(updates[g]) + 1)
                if axis == "iteration"
                else np.array([m["received"] for m in metrics[g]])
            )
            values = [np.mean([v["lr_after"] for v in u["minibatches"]]) for u in updates[g]]
            axes[0, 0].plot(x, values, label=g, alpha=0.8)
            axes[0, 1].plot(x, [np.mean(u["std"]) for u in updates[g]], label=g)
            axes[1, 0].plot(x, [m["ppo/approx_kl"] for m in metrics[g]], label=g, alpha=0.7)
            axes[1, 1].plot(x, [m["policy_lag_mean"] for m in metrics[g]], label=g, alpha=0.7)
            axes[2, 0].plot(x, [m["grad/global_norm"] for m in metrics[g]], label=g, alpha=0.7)
            axes[2, 1].plot(x, [m["reuse_ratio"] for m in metrics[g]], label=g)
        for ax, title in zip(
            axes.flat,
            [
                "Mean minibatch LR",
                "Mean std",
                "Scheduler KL",
                "Mean policy lag",
                "Global gradient norm",
                "Cumulative sample reuse",
            ],
        ):
            ax.set(title=title, xlabel=axis)
            ax.grid(alpha=0.2)
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend(ncol=3)
        fig.tight_layout()
        fig.savefig(OUT / f"training_{axis}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    intermediate = []
    for g in GROUPS:
        for iteration in [100, 200, 300, 400, 500, 501]:
            d = json.loads((OUT / f"evaluation/{g}_{iteration}.json").read_text())
            if iteration == 501:
                ids = json.loads((OUT / "manifest.json").read_text())["intermediate_24_ids"]
                subset = [r for r in d["episodes"] if r["episode_id"] in ids]
            else:
                subset = d["episodes"]
            intermediate.append(
                {
                    "group": g,
                    "iteration": iteration,
                    "received": d["counters"]["received"],
                    "speed": weighted(subset, "speed_fixed_window"),
                    "exits": sum(r["dropped"] for r in subset),
                    "q_rms": weighted(
                        [
                            dict(r, q_rms=r["diagnostics"]["q_second_difference_rms"])
                            for r in subset
                        ],
                        "q_rms",
                    ),
                }
            )
        r = [r for r in intermediate if r["group"] == g]
        for ax, key in zip(axes, ["speed", "exits", "q_rms"]):
            ax.plot([v["iteration"] for v in r], [v[key] for v in r], marker=".", label=g)
            ax.set(
                xlabel="iteration",
                title={
                    "speed": "Speed [rad/s]",
                    "exits": "Height exits / 24",
                    "q_rms": "Joint second-diff RMS [rad]",
                }[key],
            )
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "evaluation_iterations.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for group in GROUPS:
        selected = [row for row in intermediate if row["group"] == group]
        for ax, key in zip(axes, ["speed", "exits", "q_rms"]):
            ax.plot(
                [row["received"] for row in selected],
                [row[key] for row in selected],
                marker=".",
                label=group,
            )
            ax.set(
                xlabel="received transitions",
                title={
                    "speed": "Speed [rad/s]",
                    "exits": "Height exits / 24",
                    "q_rms": "Joint second-diff RMS [rad]",
                }[key],
            )
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "evaluation_received.png", dpi=150)
    plt.close(fig)
    write(OUT / "intermediate_curves.json", intermediate)
    report = [
        "# APPO G ablation: seed 1",
        "",
        "| Group | speed rad/s | exits / 240 | q second-diff RMS rad | std | received | samples used | minutes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['group']} | {r['speed_fixed_window']:.6f} | {r['height_exits']} | {r['q_second_difference_rms']:.8f} | {r['std']:.6f} | {r['received']} | {r['training_samples']} | {r['training_seconds'] / 60:.2f} |"
        )
    report += [
        "",
        "## Factorial effects (right minus left)",
        "",
        "| Comparison | speed | exits | q RMS |",
        "|---|---:|---:|---:|",
    ]
    for k, r in effects.items():
        report.append(
            f"| {k} | {r['speed_fixed_window']:+.6f} | {r['height_exits']:+d} | {r['q_second_difference_rms']:+.8f} |"
        )
    report += [
        "",
        "Full raw evidence: `analysis.json`, `comparison.csv`, `audit/fixed_data_report.json`, and per-group JSONL.",
        "Interpretation and follow-up decisions are recorded in the journal.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
