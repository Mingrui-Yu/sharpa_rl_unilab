"""Check off-screen rendering before the CLI imports the simulation runtime."""

from __future__ import annotations

import os
import subprocess
import sys

from omegaconf import DictConfig

# Use a fresh process: MuJoCo selects its GL context on import, and broken
# graphics drivers can abort the interpreter rather than raise an exception.
_PROBE = """
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size="0.1 0.1 0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
with mujoco.Renderer(model, height=32, width=32) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    frame = renderer.render()
with tempfile.TemporaryDirectory(prefix="sharpa-render-check-") as directory:
    video = Path(directory) / "probe.mp4"
    imageio.mimsave(str(video), [frame], fps=20)
    decoded = imageio.mimread(str(video))
    assert len(decoded) == 1 and decoded[0].shape == (32, 32, 3)
"""


def prepare_recording(training: DictConfig) -> str | None:
    """Select and verify GL plus MP4 encoding, or fail before training starts.

    Explicit MUJOCO_GL settings are respected. No MuJoCo/OpenGL modules are
    imported in this process; the selected backend is inherited by workers.
    """
    if training.get("no_play", False) or training.get("play_render_mode", "auto") not in (
        "auto",
        "record",
    ):
        return None

    explicit = os.environ.get("MUJOCO_GL", "").strip().lower()
    if explicit:
        candidates = [explicit]
    elif sys.platform.startswith("linux"):
        candidates = ["egl", "osmesa"]
        if os.environ.get("DISPLAY"):
            candidates.append("glfw")
    else:
        candidates = ["glfw"]

    failures = []
    for backend in candidates:
        env = os.environ.copy()
        env["MUJOCO_GL"] = backend
        try:
            result = subprocess.run(
                [sys.executable, "-c", _PROBE],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{backend}: {exc}")
            continue
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            print(f"[render] Recording precheck passed: MUJOCO_GL={backend}, MP4 encoding ready.")
            return backend
        detail = (result.stderr or result.stdout).strip()
        failures.append(f"{backend}: exit {result.returncode}\n{detail[-2000:]}")

    raise RuntimeError(
        "Video recording is enabled, but the rendering/MP4 precheck failed. "
        "Training has not started.\n"
        + "\n".join(failures)
        + "\nInstall a working graphics runtime once (on Ubuntu/Debian: libegl1 with "
        "the NVIDIA graphics driver, or libosmesa6 for software rendering). "
        "Check the encoder error above if rendering succeeded. "
        "If MUJOCO_GL was set explicitly, correct or unset it. "
        "Then rerun the same sharpa-train command; no MUJOCO_GL setting is needed. "
        "To intentionally train without video, use training.no_play=true."
    )
