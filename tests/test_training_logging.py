import json

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.training.logging import EpisodeStatistics, TrainingLogger


def test_episode_statistics_span_packets_and_reset_only_completed_rows():
    episodes = EpisodeStatistics(2)
    episodes.update(np.array([[1, 10]]), np.array([[False, False]]), np.array([[False, False]]))
    assert episodes.metrics() == {}
    # The first episode spans packets; termination and timeout coincide once.
    episodes.update(
        np.array([[2, 20], [3, 30]]),
        np.array([[True, False], [False, False]]),
        np.array([[True, False], [False, True]]),
    )
    assert episodes.metrics() == {
        "episode/return": 31.5,
        "episode/length": 2.5,
        "episode/timeout_rate": 0.5,
    }
    np.testing.assert_array_equal(episodes.returns, [3, 0])
    np.testing.assert_array_equal(episodes.lengths, [1, 0])
    episodes.update(np.array([[4, 40]]), np.array([[True, False]]), np.array([[False, False]]))
    assert episodes.metrics()["episode/return"] == pytest.approx((3 + 60 + 7) / 3)
    assert episodes.metrics()["episode/timeout_rate"] == pytest.approx(1 / 3)


def test_logs_preserve_sampling_axis_and_flush_tensorboard(tmp_path):
    cfg = compose_config("appo", "mujoco", ["training.logger=tensorboard"])
    logger = TrainingLogger(tmp_path, cfg, 10)
    try:
        for version, samples, seconds in [(1, 100, 10), (2, 200, 12)]:
            logger.log(
                {
                    "policy_version": version,
                    "received": samples,
                    "wall_seconds": seconds,
                    "episode/return": 2.5,
                }
            )
    finally:
        logger.close()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["received"] for row in rows] == [100, 200]
    assert rows[-1]["perf/transitions_per_second"] == 50
    events = EventAccumulator(str(tmp_path)).Reload().Scalars("episode/return")
    assert [event.step for event in events] == [100, 200]
    assert [event.value for event in events] == [2.5, 2.5]
