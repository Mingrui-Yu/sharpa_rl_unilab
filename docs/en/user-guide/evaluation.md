# Evaluation and playback

`sharpa-eval` is the evaluation entrypoint. It composes the same task owner as
training, sets `training.play_only=true`, resolves a checkpoint, and optionally
records video or exports a deployment model.

## Locate a checkpoint

Run from the same repository working directory. The canonical default is:

```text
logs/<algo_log_name>/SharpaInhandRotation/<timestamp>/model_<iteration>.pt
```

For example:

```text
logs/hora_appo/SharpaInhandRotation/2026-09-14_01-02-03/model_305.pt
logs/flash_sac/SharpaInhandRotation/2026-09-14_01-02-03/model_5371.pt
```

## Evaluate the latest checkpoint

Leave `training.log_root` unset and leave `algo.load_run=-1`. The loader chooses
the latest run under the canonical `./logs/<algo_log_name>/<task>/` tree.

HORA APPO:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record
```

FlashSAC:

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

PPO:

```bash
uv run sharpa-eval --algo ppo --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

`play_render_mode=record` loads the policy and exercises the checkpoint path on
a headless service while writing an offscreen video.

## Select an exact run or checkpoint

`algo.load_run` accepts:

- `-1`: latest run under the task log root.
- A run directory path.
- A checkpoint file path.

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run \
  training.play_render_mode=record
```

To select one exact file, pass the checkpoint itself:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run/model_305.pt \
  training.play_render_mode=record
```

For PPO run-level selection, `algo.checkpoint=<iteration>` may also be used.
FlashSAC always selects the newest `model_*.pt` inside the selected run unless a
checkpoint file is passed directly.

## Record a video

Use `training.play_render_mode=record` on a headless host. The output is written
as `play_video.mp4` beside the selected checkpoint.

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record \
  training.play_env_num=16 training.play_steps=200
```

Useful playback controls:

| Override | Meaning |
| --- | --- |
| `training.play_render_mode=none` | Skip playback entirely; no checkpoint is loaded by this entrypoint |
| `training.play_render_mode=record` | Offscreen MP4 recording |
| `training.play_render_mode=interactive` | Native interactive viewer; requires a display |
| `training.play_env_num=N` | Number of playback environments |
| `training.play_steps=N` | Number of policy steps |
| `training.render_spacing`, camera fields | Environment grid and camera placement |

`auto` selects a backend-supported interactive/rendering path. For deterministic
headless jobs, use explicit `record`; `none` skips the playback path entirely.

## Export a deployment model

FlashSAC exports and verifies `policy.onnx` beside the selected checkpoint by
default. Disable that step with:

```text
training.export_onnx=false
```

Example:

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

Expected artifacts:

```text
<path-to-selected-run>/policy.onnx
<path-to-selected-run>/play_video.mp4   # only in record mode
```

The PPO entrypoint accepts `--export` for its launcher-owned export path.
APPO/FlashSAC export behavior is controlled by the training fields above.

## Troubleshooting

- **Checkpoint not found:** run from the repository root used for training, or
  set `algo.load_run` to an exact run/checkpoint path. This is also required
  when the run was created with `training.log_dir`.
- **No video on a headless host:** explicitly use
  `training.play_render_mode=record`; do not rely on `auto`. `none` skips
  playback rather than performing a renderer-free checkpoint load.
- **FlashSAC ONNX export fails:** ensure the `export` extra is installed with
  `uv sync --extra mujoco --extra export`, or disable export for policy-only
  evaluation.
