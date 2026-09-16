"""Bundled Sharpa robot assets; no network or Hugging Face access at runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from filelock import FileLock

ASSETS_ROOT_PATH = Path(__file__).resolve().parent

CACHE_ENV_VAR = "SHARPA_RL_UNILAB_ASSET_CACHE"


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    digest = hashlib.sha256((ASSETS_ROOT_PATH / "manifest.json").read_bytes()).hexdigest()[:16]
    return root / "sharpa-rl-unilab" / digest


def generated_root() -> Path:
    """User data has a stable location independent of the bundled manifest."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve() / "generated"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "sharpa-rl-unilab" / "generated"


def ensure_assets() -> Path:
    """Copy bundled assets to a writable cache for XML materialization tools.

    All meshes, XML and grasp caches ship in Git and the wheel. The local cache
    allows scene tools to create temporary XML even with read-only site-packages.
    Every listed file is checked for existence and sha256, so a partial or
    corrupted cache is repaired.
    """
    manifest = json.loads((ASSETS_ROOT_PATH / "manifest.json").read_text())
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".assets.lock")):
        for relative in manifest["sha256"]:
            source, target = ASSETS_ROOT_PATH / relative, root / relative
            if (
                not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"][relative]
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
    return root


def resolve_asset(path: str | Path) -> Path:
    """Resolve user data first, then the managed asset cache and package.

    New grasp caches collected at runtime use the stable generated directory
    and take precedence; bundled copies are the fallback. A missing
    file raises ``FileNotFoundError`` instead of reaching the network.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
    else:
        for base in (generated_root(), cache_root(), ASSETS_ROOT_PATH):
            resolved = base / candidate
            if resolved.is_file():
                return resolved
    raise FileNotFoundError(
        f"Asset file not found in the writable cache ({cache_root()}) or the "
        f"bundled package assets ({ASSETS_ROOT_PATH}): {path}"
    )


def resolve_scene(path: str | Path | None = None) -> str:
    """Materialize the bundled scene; preserve explicitly supplied custom scenes."""
    bundled = ASSETS_ROOT_PATH / "robots/sharpa_wave/scene.xml"
    if path is not None and Path(path).resolve() != bundled:
        return str(path)
    return str(ensure_assets() / "robots/sharpa_wave/scene.xml")


def main() -> None:
    print(ensure_assets())
