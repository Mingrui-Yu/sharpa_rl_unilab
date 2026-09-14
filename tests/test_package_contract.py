from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest

from sharpa_rl_unilab import assets
from sharpa_rl_unilab.cli import compose_config


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
    cfg = compose_config(algo, sim, [], task=task)
    expected_task_name = {
        "sharpa_inhand": "SharpaInhandRotation",
        "sharpa_inhand_grasp": "SharpaInhandRotationGrasp",
    }[task]
    assert cfg.training.task_name == expected_task_name
    assert cfg.training.sim_backend == sim
    with pytest.raises(ValueError, match="identity"):
        compose_config(algo, sim, ["training={sim_backend:unknown}"], task=task)


def test_common_task_and_nodr_are_identical_for_all_algorithms():
    from omegaconf import OmegaConf

    for nodr in (False, True):
        configs = [
            compose_config(algo, "mujoco", [], nodr=nodr) for algo in ("ppo", "appo", "flashsac")
        ]
        for key in ("env", "reward", "budget", "hardware", "distillation", "evaluation"):
            values = [OmegaConf.to_container(cfg[key], resolve=True) for cfg in configs]
            assert values[0] == values[1] == values[2]
        if nodr:
            cfg = configs[0]
            assert cfg.env.events.persistent_force is None
            assert not cfg.env.events.domain_randomization.params.randomize_pd_gains
            assert cfg.env.observations.actor.terms.frame.params.contact_smoothing == 1
            assert cfg.env.observations.actor.terms.frame.params.joint_noise == 0


def test_reserved_overrides_are_rejected():
    with pytest.raises(ValueError, match="CLI flags"):
        compose_config("ppo", "mujoco", ["task=sharpa_inhand/motrix"])
    with pytest.raises(ValueError, match="CLI flags"):
        compose_config("ppo", "mujoco", ["training.play_only=true"])


def test_installed_registration_in_spawn(tmp_path):
    script = tmp_path / "spawn_check.py"
    script.write_text("""
import multiprocessing
def worker(queue):
    from unilab.base.registry import ensure_registries, list_registered_envs
    ensure_registries()
    envs = list_registered_envs()
    queue.put({
        "SharpaInhandRotation" in envs,
        "SharpaInhandRotationGrasp" in envs,
    })
if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    assert queue.get(timeout=30) == {True}
    process.join(timeout=30)
    assert process.exitcode == 0
""")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path, timeout=60)


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


def test_grasp_caches_are_bundled_and_resolvable():
    cache = assets.resolve_asset("caches/sharpa_grasp_linspace_1.npy")
    assert cache.is_file()
    import numpy as np

    arr = np.load(cache)
    assert arr.ndim == 2 and arr.shape[1] >= 29
