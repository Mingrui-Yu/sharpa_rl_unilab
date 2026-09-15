"""Same manifest scene, camera and real-time 20 fps for every checkpoint."""

import argparse
import json
import os
from pathlib import Path

# Rendering backend must be selected before importing MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from unilab.visualization.playback import camera_cfg_from_training

from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.training.evaluation import deterministic_actions, reset_record, set_step_rng
from sharpa_rl_unilab.training.teacher_runtime import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--manifest", default="logs/action_clipping/scenes.json")
    p.add_argument("--episode", default="1/10001/0")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    actor, cfg, _ = load_policy(args.checkpoint, "cpu")
    torch.set_num_threads(1)
    ep = next(
        e
        for e in json.loads(Path(args.manifest).read_text())["episodes"]
        if e["episode_id"] == args.episode
    )
    cfg.training.action_diagnostics = False
    cfg.training.cam_distance = 0.45
    cfg.training.cam_lookat = [0.0, 0.0, 0.55]
    env = SharpaTeacherEnv(cfg, 1, scale=ep["scale"], auto_reset=False)
    step_index = 0
    done = False
    dropped = False
    trajectory = []

    def initialize():
        obs, record = reset_record(env, cfg, ep["scale"], ep["reset_seed"])
        assert all(record[k] == ep[k] for k in record)
        return obs

    def step(obs):
        nonlocal step_index, done, dropped
        if done:
            return obs
        set_step_rng(env, ep["perturbation_seed"], step_index)
        action = deterministic_actions(actor, obs, "cpu")
        state = env.step(action)
        term = env.action_manager.get_term("hand")
        trajectory.append(
            dict(
                action=action[0].copy(),
                q=term._entity.data.joint_pos[0, term.joint_ids].copy(),
                target=term.target[0].copy(),
                angle=env.signed_angle_delta[0],
            )
        )
        done = bool(state.terminated[0] or state.truncated[0])
        dropped = bool(state.terminated[0])
        step_index += 1
        return state.obs

    font = ImageFont.truetype("DejaVuSans.ttf", 23)

    def annotate(index, frame):
        canvas = Image.fromarray(frame)
        draw = ImageDraw.Draw(canvas)
        label = f"{Path(args.output).stem} | scene {args.episode} | {(index + 1) / 20:.2f} s"
        if dropped and index >= step_index - 1:
            label += f" | HEIGHT EXIT at {step_index / 20:.2f} s (frozen)"
        draw.rectangle((0, 0, canvas.width, 40), fill=(25, 25, 25))
        draw.text((12, 6), label, font=font, fill=(255, 255, 255))
        return np.asarray(canvas)

    try:
        with torch.inference_mode():
            env.run_playback_mode(
                play_render_mode="record",
                play_steps=400,
                output_video=args.output,
                render_spacing=0.5,
                initialize=initialize,
                step=step,
                camera_kwargs=camera_cfg_from_training(cfg.training),
            )
        # UniLab's current backend does not forward on_frame. Annotate the
        # finished recording without changing simulation state or timing.
        output = Path(args.output)
        annotated = output.with_name(output.stem + ".annotated.mp4")
        with (
            imageio.get_reader(str(output)) as reader,
            imageio.get_writer(str(annotated), fps=20, codec="libx264", quality=8) as writer,
        ):
            for index, frame in enumerate(reader):
                writer.append_data(annotate(index, frame))
        annotated.replace(output)
        np.savez_compressed(
            Path(args.output).with_suffix(".trajectory.npz"),
            **{k: np.array([r[k] for r in trajectory]) for k in trajectory[0]},
        )
        Path(args.output).with_suffix(".scene.json").write_text(
            json.dumps(
                dict(
                    episode=ep,
                    alive_steps=step_index,
                    freeze_after_termination=True,
                    dropped=dropped,
                    termination_definition="Object exits the initial +/- 0.04 m height band",
                    camera=dict(distance=0.45, lookat=[0.0, 0.0, 0.55], elevation=-25, azimuth=45),
                    fps=20,
                ),
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
