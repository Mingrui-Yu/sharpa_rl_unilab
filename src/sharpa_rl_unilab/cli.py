"""Package-owned owner selection; algorithms consume the composed configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_ROOT = Path(__file__).resolve().parent / "conf"

TASK_NAMES = {
    "sharpa_inhand": "SharpaInhandRotation",
    "sharpa_inhand_grasp": "SharpaInhandRotationGrasp",
}


def compose_config(
    algo: str,
    sim: str,
    overrides: list[str],
    *,
    task: str = "sharpa_inhand",
) -> DictConfig:
    owner = f"{task}/{sim}"
    if not (CONF_ROOT / algo / "task" / f"{owner}.yaml").is_file():
        raise ValueError(f"No owner for algo={algo}, task={task}, sim={sim}")
    reserved = {"task", "training.task_name", "training.sim_backend", "training.play_only"}
    for override in overrides:
        if override.lstrip("+~").split("=", 1)[0] in reserved:
            raise ValueError("Use CLI flags to select task, backend and train/eval mode")
    with initialize_config_dir(config_dir=str(CONF_ROOT / algo), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={owner}", *overrides])
    if cfg.training.task_name != TASK_NAMES[task] or cfg.training.sim_backend != sim:
        raise ValueError("Overrides must preserve the selected task owner identity")
    if task == "sharpa_inhand":
        from sharpa_rl_unilab.tasks.sharpa_inhand.protocol import validate_observation_config

        validate_observation_config(cfg)
    return cfg


def _main(*, play: bool, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sharpa Wave in-hand manipulation")
    parser.add_argument("--algo", choices=["ppo", "appo", "flashsac"], default="ppo")
    parser.add_argument("--sim", choices=["mujoco"], default="mujoco")
    parser.add_argument("--task", choices=sorted(TASK_NAMES), default="sharpa_inhand")
    parser.add_argument(
        "--checkpoint", help="Versioned teacher/student checkpoint for quantitative evaluation"
    )
    parser.add_argument("--output", help="Evaluation JSON output path")
    parser.add_argument(
        "--cfg", action="store_true", help="Print composed config without loading assets"
    )
    parser.add_argument("--export", action="store_true", help="Export checkpoint during eval")
    args, overrides = parser.parse_known_args(argv)
    cfg = compose_config(args.algo, args.sim, overrides, task=args.task)
    cfg.training.play_only = play
    if args.cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    if not play:
        from sharpa_rl_unilab.training.rendering import prepare_recording

        prepare_recording(cfg.training)

    from sharpa_rl_unilab.assets import ensure_assets

    ensure_assets()

    if args.task == "sharpa_inhand":
        if args.export:
            raise ValueError(
                "Protocol v2 uses explicit teacher/student inputs; --export is only available for grasp playback"
            )
        if play:
            from sharpa_rl_unilab.training.configuration import resolve_device
            from sharpa_rl_unilab.training.evaluation import evaluate_checkpoint

            checkpoint = args.checkpoint or cfg.algo.checkpoint
            if checkpoint is None:
                raise ValueError("sharpa-eval requires --checkpoint PATH")
            # The checkpoint owns model, physical task and normalization settings.
            # Only explicit evaluation overrides replace its saved evaluator config.
            if any(
                not item.startswith(("evaluation.", "training.device=", "algo.checkpoint="))
                for item in overrides
            ):
                raise ValueError(
                    "Evaluation restores the checkpoint task; only evaluation.*, training.device and algo.checkpoint overrides are accepted"
                )
            evaluation = OmegaConf.from_dotlist(
                [item for item in overrides if item.startswith("evaluation.")]
            ).get("evaluation")
            result = evaluate_checkpoint(
                checkpoint,
                output=args.output,
                device=resolve_device(cfg.training.device)
                if any(item.startswith("training.device=") for item in overrides)
                else None,
                evaluation=evaluation,
            )
            print(result["summary"])
        else:
            from sharpa_rl_unilab.training.playback import play_checkpoint
            from sharpa_rl_unilab.training.teacher_runtime import train_teacher

            if args.checkpoint:
                raise ValueError(
                    "--checkpoint is an evaluation argument; training resume is not supported"
                )
            checkpoint = train_teacher(cfg)
            play_checkpoint(checkpoint, device=cfg.training.device)
    else:
        from unilab.scripts import train_rsl_rl

        # The common PPO launcher re-enters its own script under torchrun.
        # Forward our config root and owner to every worker explicitly.
        original_argv = sys.argv
        original_export = train_rsl_rl.EXPORT_POLICY
        try:
            train_rsl_rl.EXPORT_POLICY = args.export
            sys.argv = [
                original_argv[0],
                f"--config-path={CONF_ROOT / 'ppo'}",
                f"task={args.task}/{args.sim}",
                *overrides,
                f"training.play_only={str(play).lower()}",
            ]
            train_rsl_rl.main(cfg)
        finally:
            sys.argv = original_argv
            train_rsl_rl.EXPORT_POLICY = original_export


def train_main() -> None:
    _main(play=False)


def eval_main() -> None:
    _main(play=True)


if __name__ == "__main__":
    train_main()
