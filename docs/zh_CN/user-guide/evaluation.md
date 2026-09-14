# 评估与回放

`sharpa-eval` 是评估入口。它会合成与训练相同的 task owner，设置
`training.play_only=true`，解析 checkpoint，并按需录制视频或导出部署模型。

## 定位 checkpoint

在同一个仓库工作目录中运行。规范默认路径为：

```text
logs/<algo_log_name>/SharpaInhandRotation/<timestamp>/model_<iteration>.pt
```

示例：

```text
logs/hora_appo/SharpaInhandRotation/2026-09-14_01-02-03/model_305.pt
logs/flash_sac/SharpaInhandRotation/2026-09-14_01-02-03/model_5371.pt
```

## 评估最新 checkpoint

保持 `training.log_root` 未设置，并保持 `algo.load_run=-1`。loader 会选择
规范 `./logs/<algo_log_name>/<task>/` 树下的最新 run。

HORA APPO：

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record
```

FlashSAC：

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

PPO：

```bash
uv run sharpa-eval --algo ppo --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

`play_render_mode=record` 会在 headless 服务上加载策略并验证 checkpoint
路径，同时写离屏视频。

## 选择精确 run 或 checkpoint

`algo.load_run` 接受：

- `-1`：task log root 下的最新 run。
- run 目录路径。
- checkpoint 文件路径。

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run \
  training.play_render_mode=record
```

要选择精确文件，请直接传入 checkpoint 本身：

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run/model_305.pt \
  training.play_render_mode=record
```

PPO 按 run 选择时也可以使用 `algo.checkpoint=<iteration>`。如果没有直接传入
checkpoint 文件，FlashSAC 总是选择所选 run 中最新的 `model_*.pt`。

## 录制视频

在 headless 主机上使用 `training.play_render_mode=record`。输出会写为所选
checkpoint 旁边的 `play_video.mp4`。

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record \
  training.play_env_num=16 training.play_steps=200
```

有用的 playback 控制：

| Override | 含义 |
| --- | --- |
| `training.play_render_mode=none` | 完全跳过 playback；该 entrypoint 不会加载 checkpoint |
| `training.play_render_mode=record` | 离屏 MP4 录制 |
| `training.play_render_mode=interactive` | 原生交互 viewer；需要显示服务 |
| `training.play_env_num=N` | playback 环境数量 |
| `training.play_steps=N` | 策略步数 |
| `training.render_spacing` 与相机字段 | 环境网格与相机位置 |

`auto` 会选择后端支持的交互/渲染路径。确定性 headless 任务请显式使用
`record`；`none` 会完全跳过 playback 路径。

## 导出部署模型

FlashSAC 默认在所选 checkpoint 旁导出并验证 `policy.onnx`。可用以下配置
禁用该步骤：

```text
training.export_onnx=false
```

示例：

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

预期产物：

```text
<path-to-selected-run>/policy.onnx
<path-to-selected-run>/play_video.mp4   # only in record mode
```

PPO entrypoint 的 launcher 专属导出路径接受 `--export`。APPO/FlashSAC 的
导出行为由上述 training 字段控制。

## 排障

- **找不到 checkpoint：** 在训练时使用的仓库根目录运行，或将
  `algo.load_run` 设置为精确 run/checkpoint 路径。run 使用
  `training.log_dir` 创建时也必须这样做。
- **headless 主机没有视频：** 显式使用 `training.play_render_mode=record`；
  不要依赖 `auto`。`none` 是跳过 playback，不是无渲染 checkpoint 加载。
- **FlashSAC ONNX 导出失败：** 确认 `export` extra 已通过
  `uv sync --extra mujoco --extra export` 安装，或在仅评估
  策略时禁用导出。
