import os
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from sharpa_rl_unilab import cli
from sharpa_rl_unilab.training import rendering


def test_failed_gpu_probe_falls_back_without_changing_parent_until_success(monkeypatch):
    monkeypatch.setattr(rendering.sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    attempted = []

    def probe(*args, **kwargs):
        assert "MUJOCO_GL" not in os.environ
        backend = kwargs["env"]["MUJOCO_GL"]
        attempted.append(backend)
        if backend == "egl":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rendering.subprocess, "run", probe)
    assert rendering.prepare_recording(OmegaConf.create({})) == "osmesa"
    assert attempted == ["egl", "osmesa"]
    assert os.environ["MUJOCO_GL"] == "osmesa"


def test_explicit_backend_failure_is_reported_without_silent_fallback(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    attempted = []

    def probe(*args, **kwargs):
        attempted.append(kwargs["env"]["MUJOCO_GL"])
        return SimpleNamespace(returncode=1, stderr="encoder unavailable", stdout="")

    monkeypatch.setattr(rendering.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        rendering.prepare_recording(OmegaConf.create({}))
    assert attempted == ["egl"]
    assert os.environ["MUJOCO_GL"] == "egl"


@pytest.mark.parametrize(
    "settings",
    [{"no_play": True}, {"play_render_mode": "none"}, {"play_render_mode": "interactive"}],
)
def test_non_recording_runs_do_not_probe(monkeypatch, settings):
    monkeypatch.setattr(
        rendering.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected render probe")
    )
    assert rendering.prepare_recording(OmegaConf.create(settings)) is None


def test_cli_precheck_failure_precedes_asset_setup(monkeypatch):
    from sharpa_rl_unilab import assets

    def unavailable(training):
        raise RuntimeError("No renderer")

    monkeypatch.setattr(rendering, "prepare_recording", unavailable)
    monkeypatch.setattr(assets, "ensure_assets", lambda: pytest.fail("Assets loaded before check"))
    with pytest.raises(RuntimeError, match="No renderer"):
        cli._main(play=False, argv=[])
    # Configuration inspection must remain possible on hosts without graphics.
    cli._main(play=False, argv=["--cfg"])
