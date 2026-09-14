# UniLab Sharpa RL 任务仓库

本仓库提供 Sharpa Wave 手内操作任务与 HORA 算法，是独立的 UniLab task
package。任务已使用 UniLab 当前 **Manager-Based API**：本仓库拥有 action、
observation、reset、event、reward、termination 与 recorder terms；UniLab 保持
外部依赖，负责 manager 生命周期与仿真后端。

[English README](README.md) · [文档索引](docs/README.md)

## 支持的工作流

| 工作流 | 命令族 |
| --- | --- |
| PPO baseline | `sharpa-train --algo ppo --sim mujoco` |
| APPO baseline | `sharpa-train --algo appo --sim mujoco` |
| HORA APPO teacher | `sharpa-train --algo appo --sim mujoco --profile hora` |
| FlashSAC teacher | `sharpa-train --algo flashsac --sim mujoco` |
| 抓取缓存生成 | `sharpa-train --algo ppo --task sharpa_inhand_grasp` |
| HORA student 蒸馏 | `sharpa-distill` |
| 评估 / 回放 | `sharpa-eval` |

## 安装

开发环境的 `unilab` 依赖来自兄弟目录 checkout，因此两个仓库需要并排放置：

```bash
mkdir -p ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab

uv sync --extra mujoco --extra export
uv run sharpa-assets
```

安装校验：

```bash
uv run ruff check src tests
uv run pytest -q
uv run pyright
```

## 快速运行指南

先打印配置，不分配仿真环境：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

执行最小完整训练：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-smoke

uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-smoke
```

成功后会写出 `model_1.pt` 与 `run_summary.json`。

完整训练：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true

uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

默认输出目录：

```text
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
```

在同一仓库根目录运行以下命令即可评估最新 checkpoint：

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record

uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

评估指定 checkpoint：

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/run/model_305.pt \
  training.play_render_mode=record
```

无显示器录制视频使用 `training.play_render_mode=record`；FlashSAC 默认会在
checkpoint 旁导出并验证 `policy.onnx`。

## 文档

完整命令、输出目录、resume、视频、ONNX 与排障说明见中文文档：

- [文档索引](docs/README.md)
- [入门](docs/zh_CN/getting-started.md)
- [训练流程](docs/zh_CN/user-guide/training.md)
- [评估与回放](docs/zh_CN/user-guide/evaluation.md)
- [HORA](docs/zh_CN/user-guide/hora.md)
- [任务参考](docs/zh_CN/reference/task.md)
- [架构](docs/zh_CN/developer/architecture.md)
- [验证记录](docs/zh_CN/developer/validation.md)

## 正确性要点

- 物体尺寸随机化使用 immutable fixed model variants
  （`0.8` 到 `1.5`），不做运行时 geom-size DR。
- fixed variant 中所有 body geom 均唯一命名，满足 `mjbatch.VariantPack`。
- 自由物体保留 `simple="false"`，避免 reset-time mass/CoM DR 触发
  `mj_setConst` sameframe 崩溃。
- 触觉延迟/平滑、特权信息、执行器增益、物体物理参数、重力与衰减外力均为
  显式 manager terms，并只通过 Entity facade 访问状态。
