from __future__ import annotations

import json
import socket
import subprocess
import sys

import numpy as np
import pytest

from sharpa_rl_unilab import assets
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.cache import (
    resolve_grasp_cache_file,
    sample_scale_grasp_caches,
)


@pytest.mark.parametrize(
    "algo,sim,task",
    [
        ("ppo", "mujoco", "sharpa_inhand"),
        ("ppo", "mujoco", "sharpa_inhand_grasp"),
        ("appo", "mujoco", "sharpa_inhand"),
        ("flashsac", "mujoco", "sharpa_inhand"),
    ],
)
def test_owner_composition_and_identity(algo, sim, task):
    key = "training.num_envs" if task == "sharpa_inhand" else "algo.num_envs"
    cfg = compose_config(algo, sim, [f"{key}=8"], task=task)
    assert cfg.algo.num_envs == 8
    expected_task_name = {
        "sharpa_inhand": "SharpaInhandRotation",
        "sharpa_inhand_grasp": "SharpaInhandRotationGrasp",
    }[task]
    assert cfg.training.task_name == expected_task_name
    assert cfg.training.sim_backend == sim
    with pytest.raises(ValueError, match="identity"):
        compose_config(algo, sim, ["training={sim_backend:unknown}"], task=task)


def test_reserved_overrides_are_rejected():
    with pytest.raises(ValueError, match="CLI flags"):
        compose_config("ppo", "mujoco", ["task=sharpa_inhand/motrix"])
    with pytest.raises(ValueError, match="CLI flags"):
        compose_config("ppo", "mujoco", ["training.play_only=true"])


def test_registration_outside_checkout(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unilab.base.registry import ensure_registries, list_registered_envs
ensure_registries()
assert {"SharpaInhandRotation", "SharpaInhandRotationGrasp"} <= set(list_registered_envs())
""",
        ],
        check=True,
        cwd=tmp_path,
        timeout=30,
    )


@pytest.mark.slow
def test_bundled_assets_work_offline_and_repair_cache(monkeypatch, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setenv("SHARPA_RL_UNILAB_ASSET_CACHE", str(tmp_path / "assets"))
    monkeypatch.setattr(
        socket.socket, "connect", lambda *_: pytest.fail("Unexpected network access")
    )
    root = assets.ensure_assets()
    model = mujoco.MjModel.from_xml_path(str(root / "robots/sharpa_wave/scene.xml"))
    assert model.nu == 22
    manifest = json.loads((assets.ASSETS_ROOT_PATH / "manifest.json").read_text())
    relative = next(path for path in manifest["sha256"] if path.endswith(".STL"))
    mesh = root / relative
    original = mesh.read_bytes()
    mesh.write_bytes(b"x" * len(original))
    assert assets.ensure_assets() == root
    assert mesh.read_bytes() == original


def test_grasp_cache_resolution_and_sampling_use_variant_buckets() -> None:
    path = resolve_grasp_cache_file("caches/sharpa_grasp_linspace", 1.2)
    assert path.name == "sharpa_grasp_linspace_1.2.npy"
    assert path.is_file()

    caches = (
        np.arange(29, dtype=np.float64).reshape(1, 29),
        np.arange(100, 129, dtype=np.float64).reshape(1, 29),
    )
    sampled = sample_scale_grasp_caches(caches, np.asarray((1, 0, 1), dtype=np.int32))
    assert sampled.shape == (3, 29)
    np.testing.assert_allclose(sampled[1], caches[0][0])
    np.testing.assert_allclose(sampled[[0, 2]], np.broadcast_to(caches[1], (2, 29)))
