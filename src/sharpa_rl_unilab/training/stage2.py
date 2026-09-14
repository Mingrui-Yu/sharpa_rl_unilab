"""HORA stage-2 (distillation) checkpoint resolution helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from unilab.training import get_log_root
from unilab.utils.checkpoint import get_latest_run


def resolve_hora_stage2_checkpoint_path(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    """Resolve the HORA stage-2 distillation checkpoint selected by ``cfg.algo``."""
    task_log_root = get_log_root(root_dir, cfg) / str(OmegaConf.select(cfg, "training.task_name"))
    load_run = str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)

    run_dir: Path | None
    if load_run == "-1":
        run_dir = get_latest_run(task_log_root)
    else:
        candidate = Path(load_run)
        if not candidate.exists():
            candidate = task_log_root / load_run
        if candidate.is_file():
            return candidate, candidate.parent
        run_dir = candidate if candidate.is_dir() else None

    if run_dir is None:
        return None, None

    if selected_checkpoint not in (None, "", -1, "-1"):
        checkpoint_name = (
            f"hora_stage2_{selected_checkpoint}.pt"
            if str(selected_checkpoint).isdigit()
            else str(selected_checkpoint)
        )
        checkpoint_path = run_dir / checkpoint_name
        return (checkpoint_path, run_dir) if checkpoint_path.exists() else (None, run_dir)

    last_path = run_dir / "hora_stage2_last.pt"
    if last_path.exists():
        return last_path, run_dir

    numbered = [
        path for path in run_dir.glob("hora_stage2_*.pt") if path.stem.split("_")[-1].isdigit()
    ]
    if not numbered:
        return None, run_dir
    return max(numbered, key=lambda path: int(path.stem.split("_")[-1])), run_dir


def format_hora_stage2_checkpoint_error(
    cfg: DictConfig,
    *,
    task_log_root: Path,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    """Build the user-facing diagnostic for an unresolvable stage-2 checkpoint."""
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )
    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested stage-2 checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        else:
            reason = "Requested run or stage-2 checkpoint could not be resolved."
    return (
        "Could not resolve a stage-2 HORA checkpoint for play mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={cfg.algo.load_run!r}{checkpoint_hint}. "
        "Use algo.load_run=<run-dir-or-checkpoint-path> and optionally "
        "algo.checkpoint=<iteration-or-filename>."
    )


__all__ = [
    "format_hora_stage2_checkpoint_error",
    "resolve_hora_stage2_checkpoint_path",
]
