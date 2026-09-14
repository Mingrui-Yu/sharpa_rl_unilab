# 评估指南

## 选择匹配命令

评估必须使用与训练相同的算法和 profile。

| 训练的 checkpoint | 评估命令 |
| --- | --- |
| PPO | `sharpa-eval --algo ppo` |
| APPO | `sharpa-eval --algo appo` |
| HORA APPO | `sharpa-eval --algo appo --profile hora` |
| FlashSAC | `sharpa-eval --algo flashsac` |

不要给 baseline APPO checkpoint 添加 `--profile hora`。该 profile 选择 HORA
模型和观测契约，不会转换 checkpoint。

## 评估最新 checkpoint

从仓库根目录运行，并保持 `algo.load_run=-1`：

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

`record` 是 headless 模式：它会加载策略、运行 playback，并在不打开交互窗
口的情况下写出 MP4。

## 选择精确 checkpoint

`algo.load_run` 接受 `-1`、run 目录或 checkpoint 文件：

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/run/model_305.pt \
  training.play_render_mode=record
```

PPO 按 run 选择时使用 `algo.checkpoint=<iteration>`。如果没有直接传入
checkpoint 文件，FlashSAC 会选择所选 run 中的最新 checkpoint。

## Playback 控制

| 控制 | 用途 |
| --- | --- |
| `training.play_render_mode=record` | 录制离屏视频 |
| `training.play_render_mode=interactive` | 打开原生 viewer |
| `training.play_render_mode=none` | 完全跳过 playback |
| `training.play_env_num=N` | 设置 playback 环境数 |
| `training.play_steps=N` | 设置策略步数 |
| `training.cam_distance`、`training.cam_elevation` | 设置相机位置 |

只有在有显示服务的主机上使用 `interactive`。SSH 和其他 headless 环境使用
`record`。

## 导出部署模型

FlashSAC 默认在所选 checkpoint 旁写出并验证 `policy.onnx`。使用以下配置
禁用：

```text
training.export_onnx=false
```

预期产物：

```text
<path-to-selected-run>/policy.onnx
<path-to-selected-run>/play_video.mp4
```

## 排障

- **找不到 checkpoint：** 在训练时使用的仓库根目录运行，或用
  `algo.load_run` 传入精确 run/checkpoint 路径。
- **没有视频：** 使用 `training.play_render_mode=record`；`none` 会跳过
  playback。
- **ONNX 导出失败：** 使用 `uv sync --extra mujoco --extra export` 安装。
- **契约不匹配：** 使用 run 的 `run_config.json` 记录的算法和 profile；训
  练与评估之间不要改变 profile。
