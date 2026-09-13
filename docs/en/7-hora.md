# HORA

HORA is the Sharpa teacher/student flow. The current Manager-Based teacher owner
is HORA APPO; FlashSAC is available as a separate asymmetric-critic teacher.

## Teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora training.no_play=true
```

The owner sets `algo.algo_log_name=hora_appo`, resolves
`sharpa_rl_unilab.training.play_hora_appo:resolve_hora_appo_runtime`, and exposes
a noisy 147-D actor history plus a clean privileged 174-D critic history.

## Student distillation

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-appo/run
```

Checkpoint resolution is owned by `training/hora_distill_config.py`; student logs
use the `hora_distill` family.
