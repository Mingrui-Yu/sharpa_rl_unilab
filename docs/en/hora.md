# HORA guide

## When to use HORA

Use HORA when you want a teacher/student pipeline: train an asymmetric teacher
with privileged information, then distill a student that runs from the actor
observation history.

The HORA teacher in this package is HORA APPO. FlashSAC is another teacher
option, but it is not the HORA stage-one runtime.

## Train the HORA APPO teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

The teacher uses:

- a 147-dimensional noisy actor history;
- a 174-dimensional clean privileged critic history;
- the HORA APPO actor/critic models.

## Evaluate the teacher

```bash
uv run sharpa-eval --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.load_run=-1 training.play_render_mode=record
```

Keep `--profile hora` on both commands for this teacher.

## Distill a student

After the teacher checkpoint exists:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-teacher/model_305.pt
```

Inspect the composed distillation configuration before a long run:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```

Student artifacts use the `hora_distill` log family.
