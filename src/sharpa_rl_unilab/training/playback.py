"""Replay a saved teacher or student through UniLab's video playback interface."""

from pathlib import Path

import torch
from uni_rl.algos.common.normalization import EmpiricalNormalization
from unilab.training import should_run_playback
from unilab.visualization.playback import camera_cfg_from_training

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv

from .evaluation import deterministic_actions
from .teacher_runtime import load_policy, resolve_device


def play_checkpoint(checkpoint, *, device=None):
    """Save play_video.mp4 beside the checkpoint, honoring its playback settings."""
    actor, cfg, snapshot = load_policy(checkpoint, "cpu")
    training = cfg.training
    mode = training.get("play_render_mode", "auto")
    if not should_run_playback(play_only=False, no_play=training.no_play, play_render_mode=mode):
        return None
    num_envs = int(training.get("play_env_num", 16))
    steps = int(training.play_steps)
    if num_envs < 1 or steps < 1:
        raise ValueError("Playback requires positive training.play_env_num and training.play_steps")
    device = resolve_device(device or cfg.training.device)
    actor.to(device)
    history_normalizer = None
    if snapshot["stage"] == "student":
        history_normalizer = EmpiricalNormalization((30, 49), device).eval()
        history_normalizer.load_state_dict(snapshot["history_normalizer"], strict=True)
    output = Path(checkpoint).parent / "play_video.mp4"
    env = SharpaTeacherEnv(cfg, num_envs)
    try:
        print(f"Recording playback to {output} ...")
        with torch.inference_mode():
            video = env.run_playback_mode(
                play_render_mode=mode,
                play_steps=steps,
                output_video=output,
                render_spacing=float(training.render_spacing),
                initialize=lambda: env.reset(seed=int(cfg.algo.seed))[0],
                step=lambda obs: (
                    env.step(deterministic_actions(actor, obs, device, history_normalizer)).obs
                ),
                camera_kwargs=camera_cfg_from_training(training),
            )
        if video is not None:
            print(f"Saved video to {video}")
        return video
    finally:
        env.close()
