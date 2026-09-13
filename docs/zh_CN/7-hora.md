# HORA

HORA 是 Sharpa teacher/student 流程。当前 Manager-Based teacher owner 为 HORA
APPO；FlashSAC 是独立的异步 critic teacher 入口。

## Teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora training.no_play=true
```

该 owner 设置 `algo.algo_log_name=hora_appo`，解析
`sharpa_rl_unilab.training.play_hora_appo:resolve_hora_appo_runtime`，并向
actor 提供 147 维带噪历史观测，向 critic 提供 174 维干净特权历史观测。

## Student 蒸馏

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-appo/run
```

检查点解析由 `training/hora_distill_config.py` 负责；student 日志族为
`hora_distill`。
