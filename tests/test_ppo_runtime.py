import json

import pytest
from omegaconf import OmegaConf
from rsl_rl.algorithms import PPO

from sharpa_rl_unilab.cli import CONF_ROOT, compose_config
from sharpa_rl_unilab.training.teacher_runtime import load_policy, train_teacher, training_budget


def test_ppo_default_scale_and_model():
    cfg = compose_config("ppo", "mujoco", [])
    assert cfg.training.num_envs == cfg.algo.num_envs == 2048
    assert cfg.training.max_transitions is None
    assert training_budget(cfg) == ("policy_version", 501)
    assert cfg.algo.num_steps_per_env == 8
    rollout = cfg.algo.num_envs * cfg.algo.num_steps_per_env
    assert rollout == 16384
    assert rollout // cfg.algo.algorithm.num_mini_batches == 4096
    assert rollout * cfg.algo.max_iterations == 8208384
    assert (
        cfg.algo.max_iterations
        * cfg.algo.algorithm.num_learning_epochs
        * cfg.algo.algorithm.num_mini_batches
    ) == 10020
    assert cfg.algo.save_interval == 50
    assert cfg.model == OmegaConf.load(CONF_ROOT / "common/sharpa_inhand.yaml").model
    overridden = compose_config(
        "ppo", "mujoco", ["training.num_envs=8", "algo.num_steps_per_env=3"]
    )
    assert overridden.algo.num_envs == 8
    assert training_budget(overridden) == ("policy_version", 501)


@pytest.mark.parametrize("iterations, transitions", [(501, 17), (None, None)])
def test_ppo_requires_exactly_one_budget(iterations, transitions):
    cfg = compose_config("ppo", "mujoco", [])
    cfg.algo.max_iterations = iterations
    cfg.training.max_transitions = transitions
    with pytest.raises(ValueError, match="exactly one"):
        training_budget(cfg)


@pytest.mark.parametrize("iterations, transitions", [(0, None), (-1, None), (None, 0)])
def test_ppo_requires_positive_budget(iterations, transitions):
    cfg = compose_config("ppo", "mujoco", [])
    cfg.algo.max_iterations = iterations
    cfg.training.max_transitions = transitions
    with pytest.raises(ValueError, match="positive"):
        training_budget(cfg)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_envs, horizon, iterations, transitions, rollout_lengths, save_interval",
    [
        (8, 8, 2, None, [8, 8], 1),
        (4, 3, 3, None, [3, 3, 3], 1),
        (8, 2, None, 17, [2, 1], 1),
        (4, 2, 2, None, [2, 2], 0),
    ],
)
def test_ppo_stops_after_updates_and_saves(
    tmp_path,
    monkeypatch,
    num_envs,
    horizon,
    iterations,
    transitions,
    rollout_lengths,
    save_interval,
):
    cfg = compose_config(
        "ppo",
        "mujoco",
        [
            f"training.num_envs={num_envs}",
            "training.device=cpu",
            "training.torch_threads.learner_num_threads=2",
            f"algo.num_steps_per_env={horizon}",
            f"algo.save_interval={save_interval}",
            "training.logger=no_print",
            f"training.log_dir={tmp_path / 'run'}",
        ],
    )
    cfg.algo.max_iterations = iterations
    cfg.training.max_transitions = transitions
    original_update = PPO.update
    lengths, optimizer_steps = [], []

    def update(learner):
        lengths.append(learner.storage.num_transitions_per_env)
        handle = learner.optimizer.register_step_post_hook(lambda *_: optimizer_steps.append(True))
        try:
            return original_update(learner)
        finally:
            handle.remove()

    monkeypatch.setattr(PPO, "update", update)
    path = train_teacher(cfg)
    assert path.name == "teacher_final.pt"
    actor, _, snapshot = load_policy(path)
    assert lengths == rollout_lengths
    rounds, samples = len(rollout_lengths), sum(rollout_lengths) * num_envs
    counters = snapshot["counters"]
    assert counters["policy_version"] == rounds
    assert counters["received"] == counters["collected"] == samples
    assert counters["optimizer_updates"] == len(optimizer_steps) == rounds * 20
    assert counters["training_samples"] == samples * 5
    assert actor.shared.obs_normalizer.count.item() == samples
    assert snapshot["critic"]["obs_normalizer.count"].item() == samples
    rows = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
    assert [row["policy_version"] for row in rows] == list(range(1, rounds + 1))
    assert rows[-1]["received"] == samples
    assert {p.name for p in path.parent.glob("teacher_iteration_*.pt")} == {
        f"teacher_iteration_{step}.pt" for step in range(1, rounds + 1) if save_interval
    }
