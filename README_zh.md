# UniLab Sharpa RL

[English](README.md) | [文档](docs/README.md)

## 任务概述

这是一个独立的 UniLab Sharpa Wave 手内操作任务包，提供 MuJoCo 可用的
手/物体任务、内置机器人资产、grasp cache，以及 PPO、APPO、HORA APPO、
FlashSAC 和 HORA student 蒸馏训练入口。

## 亮点与展示

- 训练在 22 自由度 Sharpa Wave 手内旋转自由圆柱的策略。
- 使用触觉历史、特权 critic 信息与物体尺度随机化。
- 导出 FlashSAC actor ONNX，并录制评估视频。
- 训练 HORA teacher，并蒸馏只使用 actor 观测的 student。

| 真机 | APPO | FlashSAC |
| --- | --- | --- |
| ![真机部署回放](docs/media/sharpa-real-eval.gif) | ![APPO 评估回放](docs/media/sharpa-appo-eval.gif) | ![FlashSAC 评估回放](docs/media/sharpa-flashsac-eval.gif) |

## 安装

将 UniLab 与本任务包作为兄弟目录克隆：

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco --extra export
uv run sharpa-assets
```

推荐使用 NVIDIA CUDA 训练。评估可通过
`training.play_render_mode=record` 在无显示环境运行。

## 快速开始

### 训练 APPO

```bash
uv run sharpa-train --algo appo --sim mujoco \
  algo.seed=1 training.no_play=true
```

### 训练 FlashSAC

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

### 评估 checkpoint

APPO：

```bash
uv run sharpa-eval --algo appo --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

FlashSAC：

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

视频会写在所选 checkpoint 旁。FlashSAC 还会写出并验证 `policy.onnx`。

## 文档

- [入门](docs/zh_CN/getting-started.md)
- [训练指南](docs/zh_CN/training.md)
- [评估指南](docs/zh_CN/evaluation.md)
- [HORA 指南](docs/zh_CN/hora.md)
- [任务与环境指南](docs/zh_CN/task.md)
- [参考结果](docs/zh_CN/results.md)

## 许可

代码使用 [Apache License 2.0](LICENSE)。资产来源与第三方说明见
[NOTICE.md](NOTICE.md)。
