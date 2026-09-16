"""Small deterministic evaluation metrics; no policy RNG or training feedback."""

from __future__ import annotations

import numpy as np

DIAGNOSTIC_METRICS = (
    "std",
    "saturation",
    "action_delta_rms",
    "q_second_difference_rms",
    "target_at_limit",
)


class EpisodeActionDiagnostics:
    """Record post-step physical state before reset, including the terminal step."""

    def __init__(self, initial_q):
        self.q = np.asarray(initial_q, dtype=np.float64).copy()
        self.previous_q: np.ndarray | None = None
        self.previous_action: np.ndarray | None = None
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _add(self, name, values):
        values = np.asarray(values, dtype=np.float64)
        self.sums[name] = self.sums.get(name, 0.0) + float(values.sum())
        self.counts[name] = self.counts.get(name, 0) + values.size

    def record(self, action, std, q_next, target, lower, upper):
        action = np.asarray(action, dtype=np.float64)
        q_next = np.asarray(q_next, dtype=np.float64)
        self._add("std", std)
        self._add("saturation", np.abs(action) > 0.99)
        self._add(
            "target_at_limit", (np.abs(target - lower) < 1e-6) | (np.abs(target - upper) < 1e-6)
        )
        if self.previous_action is not None:
            self._add("action_delta_rms", (action - self.previous_action) ** 2)
        if self.previous_q is not None:
            self._add("q_second_difference_rms", (q_next - 2 * self.q + self.previous_q) ** 2)
        self.previous_action = action.copy()
        self.previous_q, self.q = self.q, q_next.copy()

    def result(self):
        result = {}
        for name in DIAGNOSTIC_METRICS:
            count = self.counts.get(name, 0)
            value = self.sums.get(name, 0.0) / count if count else 0.0
            result[name] = float(np.sqrt(value) if name.endswith("_rms") else value)
        return result
