"""Validated throughput comparison for tensor-native off-policy training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SteadyThroughput:
    """Per-iteration throughput after an explicit warmup row."""

    values: tuple[float, ...]
    mean: float


@dataclass(frozen=True)
class OffPolicyThroughputComparison:
    """Metrics and topology evidence for two same-budget off-policy runs."""

    iterations: int
    received: int
    optimizer_updates: int
    tensor_native_transitions_per_second: float
    legacy_transitions_per_second: float
    speedup: float


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_metrics(run: Path) -> list[dict[str, Any]]:
    lines = (run / "metrics.jsonl").read_text().splitlines()
    rows = []
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{run / 'metrics.jsonl'} contains a non-object row")
        rows.append(value)
    required = {"received", "wall_seconds", "optimizer_updates"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"metrics row {index} is missing {sorted(missing)}")
    if len(rows) < 2:
        raise ValueError(
            "A steady-state comparison needs a warmup row and at least one measured row"
        )
    return rows


def steady_throughput(run: Path, *, warmup_iterations: int = 1) -> SteadyThroughput:
    """Return incremental received-transition throughput after warmup.

    FlashSAC metrics are cumulative. Deriving each interval from consecutive
    ``received`` and ``wall_seconds`` rows avoids the common mistake of dividing
    the cumulative total by cumulative wall time.
    """
    rows = _read_metrics(run)
    if int(warmup_iterations) < 1:
        raise ValueError("warmup_iterations must be at least one for cumulative metrics")
    warmup = int(warmup_iterations)
    if warmup >= len(rows):
        raise ValueError("warmup_iterations must leave at least one measured row")
    values: list[float] = []
    for previous, current in zip(rows[warmup - 1 :], rows[warmup:], strict=False):
        transitions = int(current["received"]) - int(previous["received"])
        seconds = float(current["wall_seconds"]) - float(previous["wall_seconds"])
        if transitions <= 0 or seconds <= 0:
            raise ValueError("throughput rows must have positive transition and wall-time deltas")
        values.append(transitions / seconds)
    return SteadyThroughput(tuple(values), sum(values) / len(values))


def _require_tensor_native_manifest(run: Path) -> None:
    summary = _read_json(run / "summary.json")
    manifest = summary.get("runtime_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{run / 'summary.json'} has no runtime_manifest")
    transport = manifest.get("inference_transport")
    if (
        manifest.get("collector_tensor_native") is not True
        or not isinstance(transport, dict)
        or transport.get("mode") != "cuda"
        or transport.get("staging_policy") != "cuda_no_host_boundary"
        or manifest.get("replay_h2d_submitter") != "device_ingress"
    ):
        raise ValueError(f"{run} is not a CUDA tensor-native off-policy run")


def _reject_tensor_native_legacy_manifest(run: Path) -> None:
    summary = _read_json(run / "summary.json")
    manifest = summary.get("runtime_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{run / 'summary.json'} has no runtime_manifest")
    transport = manifest.get("inference_transport")
    if manifest.get("collector_tensor_native") is True and isinstance(transport, dict):
        if transport.get("mode") == "cuda":
            raise ValueError(f"{run} is tensor-native and cannot be the legacy comparator")


def compare_offpolicy_runs(
    tensor_native_run: Path,
    legacy_run: Path,
    *,
    warmup_iterations: int = 1,
    minimum_speedup: float = 1.0,
) -> OffPolicyThroughputComparison:
    """Compare same-budget FlashSAC runs and fail closed on topology mismatch."""
    _require_tensor_native_manifest(tensor_native_run)
    _reject_tensor_native_legacy_manifest(legacy_run)

    native_rows = _read_metrics(tensor_native_run)
    legacy_rows = _read_metrics(legacy_run)
    if len(native_rows) != len(legacy_rows):
        raise ValueError("runs have different iteration counts")
    for index, (native_row, legacy_row) in enumerate(zip(native_rows, legacy_rows, strict=True)):
        for field in ("received", "optimizer_updates"):
            if native_row[field] != legacy_row[field]:
                raise ValueError(f"runs have different {field} budgets at iteration {index}")

    native = steady_throughput(tensor_native_run, warmup_iterations=warmup_iterations)
    legacy = steady_throughput(legacy_run, warmup_iterations=warmup_iterations)
    speedup = native.mean / legacy.mean
    if speedup < float(minimum_speedup):
        raise ValueError(
            f"tensor-native throughput {native.mean:.3f} transitions/s is below "
            f"the required {float(minimum_speedup):.3f}x of {legacy.mean:.3f}"
        )
    return OffPolicyThroughputComparison(
        iterations=len(native_rows),
        received=int(native_rows[-1]["received"]),
        optimizer_updates=int(native_rows[-1]["optimizer_updates"]),
        tensor_native_transitions_per_second=native.mean,
        legacy_transitions_per_second=legacy.mean,
        speedup=speedup,
    )
