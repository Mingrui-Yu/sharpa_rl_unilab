import json
from types import SimpleNamespace

import pytest
from unilab.managers.rng import ManagerRng

from sharpa_rl_unilab.training.evaluation import METRICS, aggregate_seeds, set_step_rng


def test_step_rng_preserves_manager_rng_owner():
    manager = ManagerRng(device="cpu")
    env = SimpleNamespace(
        env=SimpleNamespace(rng=manager), event_manager=SimpleNamespace(active_terms={})
    )
    set_step_rng(env, 123, 4)
    assert env.env.rng is manager
    expected = manager.random(2)
    set_step_rng(env, 123, 4)
    assert (manager.random(2) == expected).all()


def test_seed_aggregation_uses_independent_training_seeds(tmp_path):
    paths = []
    for seed in (1, 2, 3):
        path = tmp_path / f"{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "stage": "teacher",
                    "algorithm": "ppo",
                    "manifest_sha256": "same",
                    "selection": "final",
                    "split": "validation",
                    "training_seed": seed,
                    "summary": dict.fromkeys(METRICS, seed),
                    "per_scale": {"1.0": dict.fromkeys(METRICS, seed)},
                }
            )
        )
        paths.append(path)
    result = aggregate_seeds(paths, tmp_path / "summary.json")
    assert result["summary"]["return"]["mean"] == 2
    assert result["summary"]["return"]["std"] == 1
    with pytest.raises(ValueError, match="distinct training seeds"):
        aggregate_seeds([paths[0]] * 3, tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="at least three"):
        aggregate_seeds(paths[:2], tmp_path / "invalid.json")
