# HORA teacher and student

HORA is the two-stage teacher/student workflow. The Manager-Based teacher owner
is HORA APPO. FlashSAC is a separate teacher option and is not the HORA stage-1
runtime.

## Train the HORA APPO teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  training.no_play=true
```

The owner:

- uses `algo.algo_log_name=hora_appo`;
- resolves `sharpa_rl_unilab.training.play_hora_appo:resolve_hora_appo_runtime`;
- gives the actor a noisy 147-dimensional history;
- gives the critic a clean privileged 174-dimensional history.

See [training workflows](training.md) for output layout, TensorBoard, and
checkpoint selection.

## Evaluate the teacher

```bash
uv run sharpa-eval --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record
```

Use `training.play_render_mode=record` for a headless checkpoint/actor load check.

## Distill a student

Run the dedicated entrypoint after the teacher checkpoint exists:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-appo/run/model_305.pt
```

Checkpoint resolution and teacher metadata are handled by
`src/sharpa_rl_unilab/training/hora_distill_config.py`. Student output uses the
`hora_distill` log family. Use the composed configuration flags in
`src/sharpa_rl_unilab/conf/hora_distill/` to select the teacher family and
student architecture.

For current CLI capabilities, inspect the composed distillation config before
launching a long run:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```
