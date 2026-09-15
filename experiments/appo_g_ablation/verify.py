"""Fail closed on missing checkpoints, changed initialization, RNG, or sample accounting."""

import json
from collections import Counter

import numpy as np
import torch
from adapter import OUT, write


def main():
    audit = json.loads((OUT / "audit/fixed_data_report.json").read_text())
    residual = json.loads((OUT / "audit/density_residual.json").read_text())
    reference_failures = {}
    for state_name, state in audit["states"].items():
        assert state["process_rng_equal"] and state["chain_rule"]["allclose"]
        failures = {k: r for k, r in state["process_R_A"].items() if not r["allclose"]}
        if failures:
            assert state_name == "trained" and set(failures) <= {"returns", "target_log_probs"}
            assert all(r["max_abs"] == 0 for r in residual["only_target_density_restored"].values())
            reference_failures[state_name] = failures
        assert all(r["allclose"] for r in state["loss_R_A"].values())
        assert max(r["max_abs"] for r in state["grad_R_A"].values()) < 1e-6
        assert max(r["max_abs"] for r in state["adam_R_A"].values()) < 1e-6
        assert state["fixed_LR_A_B_max_update_diff"] == 0
        assert state["fixed_LR_C_D_max_update_diff"] == 0
        assert state["D_N_max_update_diff"] == 0
    for row in audit["sampling"]:
        assert row["native_rng_equal"] and row["entropy_rng_equal"]
        assert row["native_sample"]["max_abs"] == row["entropy"]["max_abs"] == 0
    records = {}
    canonical = torch.load(
        OUT / "audit/canonical_initial.pt", weights_only=False, map_location="cpu"
    )
    baseline_rng = None
    manifest = json.loads((OUT / "manifest.json").read_text())
    names = list("RABCDN") + [run["name"] for run in manifest.get("additional_runs", [])]
    assert len(names) <= 10
    for g in names:
        run = OUT / f"{g}_seed1"
        if not (OUT / f"{g}_exit.json").exists():
            continue
        assert json.loads((OUT / f"{g}_exit.json").read_text())["returncode"] == 0
        init = torch.load(run / "initial.pt", weights_only=False, map_location="cpu")
        assert all(torch.equal(init["actor"][k], v) for k, v in canonical["actor"].items())
        assert all(torch.equal(init["critic"][k], v) for k, v in canonical["critic"].items())
        collector = torch.load(run / "collector_rng.pt", weights_only=False, map_location="cpu")
        rng = {"learner": init["rng"], "collector": collector["rng"]}
        if baseline_rng is None:
            baseline_rng = rng
        for role in rng:
            x, y = rng[role], baseline_rng[role]
            assert torch.equal(x["cpu"], y["cpu"])
            assert all(torch.equal(a, b) for a, b in zip(x["cuda"], y["cuda"]))
            assert all(np.array_equal(a, b) for a, b in zip(x["numpy"], y["numpy"]))
        snapshots = {}
        for iteration in [100, 200, 300, 400, 500, 501]:
            p = run / (
                "teacher_final.pt" if iteration == 501 else f"teacher_iteration_{iteration}.pt"
            )
            c = torch.load(p, weights_only=False, map_location="cpu")["counters"]
            assert c["policy_version"] == iteration
            assert c["optimizer_updates"] == 20 * iteration
            assert c["collected"] >= c["received"]
            snapshots[iteration] = c
        updates = [json.loads(s) for s in (run / "updates.jsonl").read_text().splitlines()]
        assert len(updates) == 501
        slots = [None] * 8
        writes = 0
        staging_records = []
        training_samples = 0
        for update in updates:
            for packet in update["packets"]:
                assert packet["size"] == 16384
                slots[writes % 8] = packet
                writes += 1
            active = [p for p in slots if p is not None]
            counts = Counter()
            for packet in active:
                counts[packet["behavior_version"]] += packet["size"]
            assert dict(update["staging_versions"]) == dict(counts)
            training_samples += sum(p["size"] for p in active) * 5
            staging_records.append(
                {"iteration": update["iteration"], "slots_in_batch_order": list(slots)}
            )
        assert training_samples == snapshots[501]["training_samples"]
        write(run / "staging_reconstructed.json", staging_records)
        packets = [p for u in updates for p in u["packets"]]
        received = 0
        for p in packets:
            received += p["size"]
            assert received == p["end"]
        assert received == snapshots[501]["received"]
        assert all(len(u["minibatches"]) == 20 for u in updates)
        records[g] = {
            "initialization_and_rng_identical": True,
            "snapshots": snapshots,
            "packets_contiguous": True,
        }
    write(
        OUT / "audit/final_verification.json",
        {
            "stage0_passed_with_explained_roundoff": True,
            "reference_tolerance_failures": reference_failures,
            "groups": records,
        },
    )
    print(f"Verified stage 0 and {len(records)} complete groups")


if __name__ == "__main__":
    main()
