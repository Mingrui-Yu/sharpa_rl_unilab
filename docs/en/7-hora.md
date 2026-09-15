# HORA teacher and student

PPO, APPO and FlashSAC now train the same HORA Actor by default. Each rotation
algorithm has one `mujoco.yaml`; no HORA profile is needed. A teacher consumes
147 measurable observation values plus a separate current 9-D privilege vector.
The independent critic receives 174 clean values. FlashSAC adapts the common
Actor while retaining its native distributional double Q and temperature losses.

```bash
uv run sharpa-train --algo appo
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

`sharpa-train` and `sharpa-distill` replay their final checkpoint after training
and save `play_video.mp4` in that checkpoint's directory. Playback uses 16
environments and 400 control steps by default. Set `training.no_play=true` or
`training.play_render_mode=none` to skip it; `training.play_env_num` and
`training.play_steps` control its size and duration. MuJoCo's default `auto`
mode records video. Playback is independent of quantitative evaluation, so
disabling quantitative evaluation still allows video recording.
Headless recording requires a working EGL or OSMesa runtime. On Ubuntu/Debian,
install `libegl1` with a working NVIDIA graphics driver, or `libosmesa6` for
software rendering. `sharpa-train` automatically selects the backend and
checks a rendered frame plus MP4 encoding/decoding before training starts.
It fails early if recording is enabled but unavailable; no environment-variable
prefix is required. Explicit `MUJOCO_GL` settings are respected, and `--cfg`
or disabled recording skip the check. The separate `sharpa-distill` entry
still uses UniLab's playback-time check and skips export if rendering is unavailable.

Distillation accepts all three teacher algorithms. It freezes the inherited
policy and base normalization, trains only the 30-frame history encoder by
latent MSE, and acts with the updated student's deterministic action after each
update. Deployment needs only the 147-D base observation and [N,30,49] history.

Checkpoints contain the full run configuration and normalization statistics.
Old shared HORA, flat PPO and native FlashSAC checkpoints require retraining.
Evaluation uses fixed scenes and complete 20-second windows; video replay is
not a quantitative evaluation.

See the [protocol and migration record](../migrations/issue-2.md).
