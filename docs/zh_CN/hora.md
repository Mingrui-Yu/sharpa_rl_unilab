# HORA 指南

## 何时使用 HORA

需要一个 teacher/student 流程时使用 HORA：先训练使用特权信息的非对称
teacher，再蒸馏一个只使用 actor 观测历史的 student。

本包中的 HORA teacher 是 HORA APPO。FlashSAC 是另一个 teacher 选项，但不是
HORA 第一阶段 runtime。

## 训练 HORA APPO teacher

```bash
uv run sharpa-train --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

该 teacher 使用：

- 147 维带噪 actor 历史；
- 174 维干净特权 critic 历史；
- HORA APPO actor/critic 模型。

## 评估 teacher

```bash
uv run sharpa-eval --algo appo --task sharpa_inhand \
  --sim mujoco --profile hora \
  algo.load_run=-1 training.play_render_mode=record
```

该 teacher 的两个命令都保留 `--profile hora`。

## 蒸馏 student

teacher checkpoint 存在后：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco \
  algo.load_run=/absolute/path/to/hora-teacher/model_305.pt
```

长时间运行前先检查合成蒸馏配置：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```

Student 产物使用 `hora_distill` 日志族。
