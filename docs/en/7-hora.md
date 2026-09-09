# HORA

The committed HORA path is the Sharpa in-hand teacher/student flow. Teacher
owners live under the PPO, APPO and SAC task trees through the `hora` profile
for `sharpa_inhand`; student distillation uses the `sharpa-distill`
entrypoint and `src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml`.

## Teacher

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
uv run sharpa-train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --task sharpa_inhand --sim mujoco training.no_play=true
```

The HORA PPO owner sets `algo.algo_log_name=hora_ppo` and resolves the runtime
through `sharpa_rl_unilab.algos.hora.rsl_rl:resolve_hora_ppo_runtime`. The APPO
variant sets `algo.algo_log_name=hora_appo`; the SAC teacher sets
`algo.algo_log_name=hora_sac` and resolves
`sharpa_rl_unilab.algos.hora.sac:resolve_hora_sac_runtime`.

## Student Distillation

Student distillation is implemented by
`src/sharpa_rl_unilab/training/train_hora_distill.py` and configured by
`src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml`:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=-1
```

Teacher checkpoint resolution is implemented in
`src/sharpa_rl_unilab/training/hora_distill_config.py`. The student log family
is `hora_distill`.
