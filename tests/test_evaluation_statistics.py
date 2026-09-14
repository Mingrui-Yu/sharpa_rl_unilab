import json

import pytest

from sharpa_rl_unilab.training.evaluation import METRICS, aggregate_seeds, git_output


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


def test_installed_package_does_not_require_a_git_checkout(tmp_path):
    assert git_output(tmp_path, "rev-parse", "HEAD") is None
