# Evaluation guide

## Select the matching command

Evaluation must use the same algorithm and profile as training.

| Trained checkpoint | Evaluation command |
| --- | --- |
| PPO | `sharpa-eval --algo ppo` |
| APPO | `sharpa-eval --algo appo` |
| HORA APPO | `sharpa-eval --algo appo --profile hora` |
| FlashSAC | `sharpa-eval --algo flashsac` |

Do not add `--profile hora` to a baseline APPO checkpoint. The profile selects
the HORA model and observation contract; it does not convert a checkpoint.

## Evaluate the newest checkpoint

Run from the repository root and leave `algo.load_run=-1`:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 training.play_render_mode=record
```

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

```bash
uv run sharpa-eval --algo ppo --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

`record` is the headless mode: it loads the policy, runs playback, and writes an
MP4 without opening an interactive window.

## Select an exact checkpoint

`algo.load_run` accepts `-1`, a run directory, or a checkpoint file:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/run/model_305.pt \
  training.play_render_mode=record
```

For PPO run-level selection, use `algo.checkpoint=<iteration>`. FlashSAC chooses
the newest checkpoint in a selected run unless a checkpoint file is passed.

## Playback controls

| Control | Purpose |
| --- | --- |
| `training.play_render_mode=record` | Record offscreen video |
| `training.play_render_mode=interactive` | Open the native viewer |
| `training.play_render_mode=none` | Skip playback entirely |
| `training.play_env_num=N` | Set playback environment count |
| `training.play_steps=N` | Set policy steps |
| `training.cam_distance`, `training.cam_elevation` | Position the camera |

Use `interactive` only on a host with a display. Use `record` for SSH and other
headless sessions.

## Export a deployment model

FlashSAC writes and verifies `policy.onnx` beside the selected checkpoint by
default. Disable this with:

```text
training.export_onnx=false
```

Expected artifacts:

```text
<path-to-selected-run>/policy.onnx
<path-to-selected-run>/play_video.mp4
```

## Troubleshooting

- **Checkpoint not found:** run from the repository root used for training, or
  pass an exact run/checkpoint path with `algo.load_run`.
- **No video:** use `training.play_render_mode=record`; `none` skips playback.
- **ONNX export fails:** install with `uv sync --extra mujoco --extra export`.
- **Contract mismatch:** use the algorithm and profile recorded in the run's
  `run_config.json`; do not change the profile between training and evaluation.
