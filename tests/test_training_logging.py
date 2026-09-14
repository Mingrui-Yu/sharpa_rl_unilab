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
    assert episodes.metrics() == {"episode/return": 31.5, "episode/length": 2.5}
    np.testing.assert_array_equal(episodes.returns, [3, 0])
    np.testing.assert_array_equal(episodes.lengths, [1, 0])
    episodes.update(np.array([[4, 40]]), np.array([[True, False]]), np.array([[False, False]]))
    assert episodes.metrics()["episode/return"] == pytest.approx((3 + 60 + 7) / 3)


def test_rich_logging_preserves_metrics_and_flushes_tensorboard(tmp_path, capsys):
    cfg = compose_config("appo", "mujoco", [])
    logger = TrainingLogger(tmp_path, cfg, 200)
    try:
        logger.start(status="Waiting for first rollout...")
        for steps, seconds in ((100, 10), (200, 12)):
            logger.log(
                {
                    "received": steps,
                    "collected": steps + 10,
                    "training_samples": steps * 10,
                    "wall_seconds": seconds,
                    "loss/policy_loss": -0.1,
                    "surrogate_loss": -0.1,
                    "episode/return": 2.5,
                    "episode/length": 20,
                }
            )
        assert logger._terminal_snapshot.metrics["received"] == 200
        assert logger._terminal_snapshot.scalars["steps_per_sec"] == 50
        assert logger._estimate_eta() == "0s"
        logger.log_save("teacher_final.pt")
        logger.finish()
    finally:
        logger.close()
    output = capsys.readouterr().out
    assert "UniLab Training" in output
    assert "Loss/Policy Loss" in output
    assert "Transitions 200/200" in output
    assert "iter 200" not in output
    assert '"surrogate_loss"' not in output
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["surrogate_loss"] == -0.1
    assert rows[-1]["perf/transitions_per_second"] == 50
    events = EventAccumulator(str(tmp_path)).Reload()
    rewards = events.Scalars("episode/return")
    assert [event.step for event in rewards] == [100, 200]
    assert [event.value for event in rewards] == [2.5, 2.5]


@pytest.mark.parametrize("backend", ["none", "no_print"])
def test_logging_backends_close_on_failure(tmp_path, capsys, backend):
    from contextlib import closing

    cfg = compose_config("ppo", "mujoco", [f"training.logger={backend}"])
    with (
        pytest.raises(RuntimeError, match="training failed"),
        closing(TrainingLogger(tmp_path, cfg, 8)) as logger,
    ):
        logger.start()
        logger.log({"received": 8, "wall_seconds": 1, "policy_version": 1})
        raise RuntimeError("training failed")
    output = capsys.readouterr().out
    assert ("UniLab Training" in output) == (backend == "none")
    assert not list(tmp_path.glob("events.out.tfevents.*"))
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["received"] == 8
