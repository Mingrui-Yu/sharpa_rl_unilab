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


def _owner_name(task: str, sim: str, profile: str | None) -> str:
    if profile == "hora":
        if sim != "mujoco":
            raise ValueError("The hora profile only provides a MuJoCo owner")
        return f"{task}/mujoco_hora"
    return f"{task}/{sim}"


def compose_config(
    algo: str,
    sim: str,
    overrides: list[str],
    *,
    task: str = "sharpa_inhand",
    profile: str | None = None,
) -> DictConfig:
    owner = _owner_name(task, sim, profile)
    if profile is None and not (CONF_ROOT / algo / "task" / f"{owner}.yaml").is_file():
        # SAC ships only the HORA teacher owner; fall back to the hora variant.
        hora_owner = _owner_name(task, sim, "hora")
        if (CONF_ROOT / algo / "task" / f"{hora_owner}.yaml").is_file():
            owner = hora_owner
    if not (CONF_ROOT / algo / "task" / f"{owner}.yaml").is_file():
        raise ValueError(f"No owner for algo={algo}, task={task}, sim={sim}, profile={profile}")
    reserved = {"task", "training.task_name", "training.sim_backend", "training.play_only"}
    for override in overrides:
        if override.lstrip("+~").split("=", 1)[0] in reserved:
            raise ValueError("Use CLI flags to select task, backend and train/eval mode")
    with initialize_config_dir(config_dir=str(CONF_ROOT / algo), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={owner}", *overrides])
    if cfg.training.task_name != TASK_NAMES[task] or cfg.training.sim_backend != sim:
        raise ValueError("Overrides must preserve the selected task owner identity")
    return cfg


def _main(*, play: bool, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sharpa Wave in-hand manipulation")
    parser.add_argument("--algo", choices=["ppo", "appo", "sac"], default="ppo")
    parser.add_argument("--sim", choices=["mujoco", "motrix"], default="mujoco")
    parser.add_argument("--task", choices=sorted(TASK_NAMES), default="sharpa_inhand")
    parser.add_argument("--profile", choices=["hora"], default=None)
    parser.add_argument(
        "--cfg", action="store_true", help="Print composed config without loading assets"
    )
    parser.add_argument("--export", action="store_true", help="Export checkpoint during eval")
    args, overrides = parser.parse_known_args(argv)
    cfg = compose_config(args.algo, args.sim, overrides, task=args.task, profile=args.profile)
    cfg.training.play_only = play
    if args.cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    from sharpa_rl_unilab.assets import ensure_assets

    ensure_assets()

    if args.algo == "appo":
        from unilab.scripts import train_appo

        train_appo.main(cfg)
    elif args.algo == "sac":
        from unilab.scripts import train_offpolicy

        train_offpolicy.main(cfg)
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
                f"task={_owner_name(args.task, args.sim, args.profile)}",
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
