# HORA teacher 与 student

HORA 是两阶段 teacher/student 工作流。Manager-Based teacher owner 是 HORA
APPO。FlashSAC 是独立的 teacher 选项，不是 HORA 第一阶段 runtime。

## 训练 HORA APPO teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  training.no_play=true
```

该 owner：

- 使用 `algo.algo_log_name=hora_appo`；
- 解析 `sharpa_rl_unilab.training.play_hora_appo:resolve_hora_appo_runtime`；
- 给 actor 147 维带噪历史观测；
- 给 critic 174 维干净特权历史观测。

输出布局、TensorBoard 与 checkpoint 选择见[训练工作流](training.md)。

## 评估 teacher

```bash
uv run sharpa-eval --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record
```

headless checkpoint/actor 加载检查使用
`training.play_render_mode=record`。

## 蒸馏 student

teacher checkpoint 存在后，运行专用入口：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-appo/run/model_305.pt
```

checkpoint 解析与 teacher 元数据由
`src/sharpa_rl_unilab/training/hora_distill_config.py` 处理。student 输出使用
`hora_distill` 日志族。使用
`src/sharpa_rl_unilab/conf/hora_distill/` 中的合成配置标志选择 teacher 族与
student 结构。

启动长时间运行前，先检查当前 CLI 能力的合成蒸馏配置：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```
