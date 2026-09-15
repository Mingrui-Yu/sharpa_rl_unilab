"""Verify the N logarithmic-term follow-up and compare matched evaluation scenes."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from adapter import OUT, ROOT, branch, write
from analyze import compare, lines, train_summary, weighted
from freeze import digest, hashes

NAME = "N_kl_log_epsilon"


def first_difference(left, right, predicate):
    return next((a["iteration"] for a, b in zip(left, right) if predicate(a, b)), None)


def main():
    run = OUT / f"{NAME}_seed1"
    manifest = json.loads((OUT / "kl_log_manifest.json").read_text())
    assert manifest["status"] in ["evaluated", "complete"]
    assert json.loads((OUT / f"{NAME}_exit.json").read_text())["returncode"] == 0
    assert digest(OUT / "manifest.json") == manifest["baseline_manifest_sha256"]
    baseline = json.loads((OUT / "manifest.json").read_text())
    integrity = {
        "production_source": hashes(ROOT / "src") == baseline["sources"]["current"]["files"]
    }
    for key, dependency in baseline["dependencies"].items():
        current = hashes(Path(dependency["path"]))
        integrity[key] = all(current.get(k) == v for k, v in dependency["files"].items())
    current = hashes(ROOT / "experiments/appo_g_ablation")
    integrity["training_entrypoints"] = all(
        current[k] == manifest["code_hashes"][k]
        for k in ["adapter.py", "train_kl_bias.py", "run_kl_log.py"]
    )
    assert all(integrity.values())
    init = torch.load(run / "initial.pt", weights_only=False, map_location="cpu")
    reference = torch.load(OUT / "N_seed1/initial.pt", weights_only=False, map_location="cpu")
    for key in ["actor", "critic"]:
        assert init[key].keys() == reference[key].keys()
        assert all(torch.equal(init[key][k], reference[key][k]) for k in init[key])
    for file in ["initial.pt", "collector_rng.pt"]:
        a, b = [
            torch.load(p / file, weights_only=False, map_location="cpu")["rng"]
            for p in [run, OUT / "N_seed1"]
        ]
        assert torch.equal(a["cpu"], b["cpu"])
        assert len(a["cuda"]) == len(b["cuda"]) and all(
            torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])
        )
        assert all(np.array_equal(x, y) for x, y in zip(a["numpy"], b["numpy"]))
    checkpoints = {}
    for iteration in [100, 200, 300, 400, 500, 501]:
        path = run / (
            "teacher_final.pt" if iteration == 501 else f"teacher_iteration_{iteration}.pt"
        )
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        counters = checkpoint["counters"]
        assert (
            counters["policy_version"] == iteration
            and counters["optimizer_updates"] == iteration * 20
        )
        checkpoints[iteration] = counters
    final = checkpoint
    ref_final = torch.load(OUT / "N_seed1/teacher_final.pt", weights_only=False, map_location="cpu")
    cfgs = [json.loads(json.dumps(c["config"])) for c in [final, ref_final]]
    for cfg in cfgs:
        cfg.pop("experiment", None)
        cfg["training"].pop("log_dir", None)
    assert cfgs[0] == cfgs[1], (
        "Unexpected config difference beyond KL intervention/output directory"
    )
    groups = ["N", "R", "C", NAME]
    updates = {g: lines(OUT / f"{g}_seed1/updates.jsonl") for g in groups}
    selected = lines(run / "selected_kl.jsonl")
    assert len(updates[NAME]) == len(selected) == 501
    slots, writes, received, training_samples = [None] * 8, 0, 0, 0
    schedule_records = []
    for update, logged in zip(updates[NAME], selected):
        assert update["iteration"] == logged["iteration"]
        assert len(update["minibatches"]) == len(logged["selected_kl"]) == 20
        for packet in update["packets"]:
            received += packet["size"]
            assert packet["end"] == received
            slots[writes % 8] = packet
            writes += 1
        counts = Counter()
        for packet in slots:
            if packet:
                counts[packet["behavior_version"]] += packet["size"]
        assert dict(counts) == dict(update["staging_versions"])
        training_samples += sum(counts.values()) * 5
        for index, (mb, kl) in enumerate(zip(update["minibatches"], logged["selected_kl"])):
            actual = branch(kl)
            assert actual == mb["branch"]
            rate = mb["lr_before"]
            expected = (
                min(1e-2, rate * 1.1)
                if actual == "up"
                else max(1e-5, rate / 1.1)
                if actual == "down"
                else rate
            )
            assert expected == mb["lr_after"]
            schedule_records.append(
                {
                    "iteration": update["iteration"],
                    "minibatch": index,
                    "selected_kl": kl,
                    "exact_kl": mb["exact_kl"],
                    "old_kl": mb["old_kl"],
                    "selected_branch": actual,
                    "exact_branch": branch(mb["exact_kl"]),
                    "old_branch": branch(mb["old_kl"]),
                }
            )
    assert received == checkpoints[501]["received"]
    assert training_samples == checkpoints[501]["training_samples"]
    divergences = {}
    equality = {}
    for group in groups[:-1]:
        a, b = updates[group], updates[NAME]
        divergences[group] = {
            "first_std_difference": first_difference(a, b, lambda x, y: x["std"] != y["std"]),
            "first_LR_difference": first_difference(
                a,
                b,
                lambda x, y: (
                    [m["lr_after"] for m in x["minibatches"]]
                    != [m["lr_after"] for m in y["minibatches"]]
                ),
            ),
            "first_packet_difference": first_difference(
                a, b, lambda x, y: x["packets"] != y["packets"]
            ),
            "first_staging_difference": first_difference(
                a, b, lambda x, y: x["staging_versions"] != y["staging_versions"]
            ),
        }
        ref = torch.load(
            OUT / f"{group}_seed1/teacher_final.pt", weights_only=False, map_location="cpu"
        )
        equality[group] = {
            key: final[key].keys() == ref[key].keys()
            and all(torch.equal(final[key][k], ref[key][k]) for k in final[key])
            for key in ["actor", "critic", "target_actor"]
        }
    results = {g: json.loads((OUT / f"evaluation/{g}_501.json").read_text()) for g in groups}
    with (
        np.load(OUT / "evaluation/C_501.trajectories.npz") as reference_trace,
        np.load(OUT / f"evaluation/{NAME}_501.trajectories.npz") as current_trace,
    ):
        trajectory_equal_C = reference_trace.files == current_trace.files and all(
            np.array_equal(reference_trace[key], current_trace[key])
            for key in reference_trace.files
        )
    rows = []
    for g in groups:
        result = results[g]
        training, _, _ = train_summary(g)
        rows.append(
            {
                "group": g,
                **result["summary"],
                **result["diagnostics"]["summary"],
                "height_exits": sum(int(e["dropped"]) for e in result["episodes"]),
                **training,
            }
        )
    by_group = {r["group"]: r for r in rows}
    metrics = [
        "speed_fixed_window",
        "height_exits",
        "q_second_difference_rms",
        "survival_seconds",
        "std",
        "saturation",
        "action_delta_rms",
        "target_at_limit",
    ]
    effects = {g: {k: by_group[NAME][k] - by_group[g][k] for k in metrics} for g in groups[:-1]}
    paired = {g: compare(g, NAME, results) for g in groups[:-1]}
    statistics = {
        "selected_vs_exact_branch_disagreements": sum(
            r["selected_branch"] != r["exact_branch"] for r in schedule_records
        ),
        "first_minibatch_disagreements": sum(
            r["minibatch"] == 0 and r["selected_branch"] != r["exact_branch"]
            for r in schedule_records
        ),
        "selected_vs_full_old_branch_disagreements": sum(
            r["selected_branch"] != r["old_branch"] for r in schedule_records
        ),
        "same_policy_bias_first_minibatch_min": min(
            r["selected_kl"] for r in schedule_records if r["minibatch"] == 0
        ),
        "same_policy_bias_first_minibatch_max": max(
            r["selected_kl"] for r in schedule_records if r["minibatch"] == 0
        ),
    }
    assert all(r["exact_kl"] == 0 for r in schedule_records if r["minibatch"] == 0)
    evidence = {
        "name": NAME,
        "summaries": rows,
        "effects_new_minus_reference": effects,
        "paired_scenes": paired,
        "divergences": divergences,
        "final_weight_equality": equality,
        "C_evaluation_trajectory_equal": trajectory_equal_C,
        "C_all_checkpoint_weight_equality": json.loads(
            (OUT / "audit/kl_log_C_weight_comparison.json").read_text()
        ),
        "schedule_statistics": statistics,
        "source_integrity": integrity,
        "initialization_rng_and_config_verified": True,
        "checkpoints": checkpoints,
        "failures": [e["episode_id"] for e in results[NAME]["episodes"] if e["dropped"]],
    }
    write(OUT / "kl_log_analysis.json", evidence)
    write(OUT / "audit/kl_log_schedule.json", schedule_records)
    ids = baseline["intermediate_24_ids"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    curve_rows = []
    for group in groups:
        us = updates[group]
        x = [u["iteration"] for u in us]
        axes[0, 0].plot(
            x,
            [np.mean([m["lr_after"] for m in u["minibatches"]]) for u in us],
            label=group,
            alpha=0.8,
        )
        axes[0, 1].plot(x, [np.mean(u["std"]) for u in us], label=group)
        axes[0, 2].plot(
            x,
            [np.mean([m["exact_kl"] for m in u["minibatches"]]) for u in us],
            label=group,
            alpha=0.8,
        )
        values = []
        for iteration in [100, 200, 300, 400, 500, 501]:
            d = json.loads((OUT / f"evaluation/{group}_{iteration}.json").read_text())
            episodes = [e for e in d["episodes"] if e["episode_id"] in ids]
            row = {
                "group": group,
                "iteration": iteration,
                "speed": weighted(episodes, "speed_fixed_window"),
                "exits": sum(e["dropped"] for e in episodes),
                "q_rms": weighted(
                    [dict(e, q_rms=e["diagnostics"]["q_second_difference_rms"]) for e in episodes],
                    "q_rms",
                ),
            }
            values.append(row)
            curve_rows.append(row)
        for ax, key in zip(axes[1], ["speed", "exits", "q_rms"]):
            ax.plot(
                [r["iteration"] for r in values], [r[key] for r in values], marker=".", label=group
            )
    for ax, title in zip(
        axes.flat,
        [
            "Applied mean LR",
            "Mean std",
            "Unbiased exact KL",
            "Speed [rad/s], frozen 24 scenes",
            "Height exits /24",
            "q second-diff RMS [rad], frozen 24 scenes",
        ],
    ):
        ax.set(title=title, xlabel="iteration")
        ax.grid(alpha=0.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kl_log_curves.png", dpi=150)
    plt.close(fig)
    write(OUT / "kl_log_intermediate_curves.json", curve_rows)
    print(
        json.dumps(
            {
                "summaries": rows,
                "effects": effects,
                "divergences": divergences,
                "weights": equality,
                "schedule": statistics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
