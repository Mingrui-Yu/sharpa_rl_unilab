"""Off-policy throughput comparisons use incremental, same-budget metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sharpa_rl_unilab.training.offpolicy_throughput import (
    compare_offpolicy_runs,
    steady_throughput,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def make_run(
    path: Path,
    *,
    received: tuple[int, ...],
    walls: tuple[float, ...],
    manifest: dict,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "received": count,
            "wall_seconds": wall,
            "optimizer_updates": (index + 1) * 28,
        }
        for index, (count, wall) in enumerate(zip(received, walls, strict=True))
    ]
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write_json(path / "summary.json", {"runtime_manifest": manifest})
    return path


def tensor_native_manifest() -> dict:
    return {
        "collector_tensor_native": True,
        "inference_transport": {
            "mode": "cuda",
            "staging_policy": "cuda_no_host_boundary",
        },
        "replay_h2d_submitter": "device_ingress",
    }


def legacy_manifest() -> dict:
    return {
        "collector_tensor_native": False,
        "replay_h2d_submitter": "gpu_resident_ingress",
    }


def test_steady_throughput_uses_incremental_rows(tmp_path):
    run = make_run(
        tmp_path / "run",
        received=(4096, 8192, 12288),
        walls=(2.0, 2.5, 3.0),
        manifest=tensor_native_manifest(),
    )
    result = steady_throughput(run)
    assert result.values == (8192.0, 8192.0)
    assert result.mean == 8192.0


def test_offpolicy_comparison_checks_budget_and_gpu_topology(tmp_path):
    native = make_run(
        tmp_path / "native",
        received=(4096, 8192, 12288),
        walls=(2.0, 2.25, 2.5),
        manifest=tensor_native_manifest(),
    )
    legacy = make_run(
        tmp_path / "legacy",
        received=(4096, 8192, 12288),
        walls=(2.0, 2.75, 3.5),
        manifest=legacy_manifest(),
    )
    comparison = compare_offpolicy_runs(native, legacy, minimum_speedup=1.5)
    assert comparison.iterations == 3
    assert comparison.received == 12288
    assert comparison.optimizer_updates == 84
    assert comparison.tensor_native_transitions_per_second == pytest.approx(16384.0)
    assert comparison.legacy_transitions_per_second == pytest.approx(5461.333333333333)
    assert comparison.speedup == pytest.approx(3.0)
    assert comparison.speedup >= 1.5


def test_offpolicy_comparison_rejects_a_cuda_native_legacy_candidate(tmp_path):
    native = make_run(
        tmp_path / "native",
        received=(4096, 8192),
        walls=(1.0, 1.5),
        manifest=tensor_native_manifest(),
    )
    same_native = make_run(
        tmp_path / "legacy",
        received=(4096, 8192),
        walls=(1.0, 2.0),
        manifest=tensor_native_manifest(),
    )
    with pytest.raises(ValueError, match="legacy comparator"):
        compare_offpolicy_runs(native, same_native)


def test_offpolicy_comparison_rejects_mismatched_transition_budgets(tmp_path):
    native = make_run(
        tmp_path / "native",
        received=(4096, 8192),
        walls=(1.0, 1.5),
        manifest=tensor_native_manifest(),
    )
    legacy = make_run(
        tmp_path / "legacy",
        received=(4096, 8448),
        walls=(1.0, 2.0),
        manifest=legacy_manifest(),
    )
    with pytest.raises(ValueError, match="received budgets at iteration 1"):
        compare_offpolicy_runs(native, legacy)


@pytest.mark.slow
def test_recorded_offpolicy_throughput_comparison():
    """Validate explicit benchmark artifacts without making CI timing flaky."""
    native = os.environ.get("SHARPA_OFFPOLICY_BENCHMARK_TENSOR_NATIVE")
    legacy = os.environ.get("SHARPA_OFFPOLICY_BENCHMARK_LEGACY")
    if not native or not legacy:
        pytest.skip("set SHARPA_OFFPOLICY_BENCHMARK_TENSOR_NATIVE and ..._LEGACY to run")
    minimum = float(os.environ.get("SHARPA_OFFPOLICY_BENCHMARK_MIN_SPEEDUP", "1.25"))
    comparison = compare_offpolicy_runs(
        Path(native),
        Path(legacy),
        warmup_iterations=1,
        minimum_speedup=minimum,
    )
    print(
        "off-policy throughput: "
        f"tensor-native={comparison.tensor_native_transitions_per_second:.3f} transitions/s, "
        f"legacy={comparison.legacy_transitions_per_second:.3f}, "
        f"speedup={comparison.speedup:.3f}x"
    )
    assert comparison.received > 0
    assert comparison.optimizer_updates > 0
    assert comparison.speedup >= minimum
