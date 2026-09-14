# 训练工作流

## 选择 owner

| CLI 选择 | Owner 文件 | 用途 |
| --- | --- | --- |
| `--algo ppo --sim mujoco` | `conf/ppo/task/sharpa_inhand/mujoco.yaml` | On-policy PPO baseline |
| `--algo appo --sim mujoco` | `conf/appo/task/sharpa_inhand/mujoco.yaml` | Flat asymmetric APPO baseline |
| `--algo appo --sim mujoco --profile hora` | `conf/appo/task/sharpa_inhand/mujoco_hora.yaml` | 分离 actor/critic 观测的 HORA teacher |
| `--algo flashsac --sim mujoco` | `conf/flashsac/task/sharpa_inhand/mujoco.yaml` | FlashSAC asymmetric-critic teacher |
| `--algo ppo --task sharpa_inhand_grasp` | `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml` | 重新生成单尺度 grasp cache |

MuJoCo 是 fixed-variant 任务的生产后端。

## 预览生效配置

Hydra 会合成算法根配置、task owner 和命令行 override。在分配仿真环境之前
先打印结果：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

`--cfg` 显示的任何 override 都可以追加到训练命令。

## 启动完整训练

无人值守训练使用 `training.no_play=true`。留在仓库根目录，并保持
`training.log_root` 未设置；每个 launcher 会写入规范的
`./logs/<algo_log_name>/<task>/<timestamp>/` 目录树。

PPO：

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  training.no_play=true
```

HORA APPO：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  training.no_play=true
```

APPO baseline：

```bash
uv run sharpa-train --algo appo --sim mujoco \
  training.no_play=true
```

FlashSAC：

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  training.no_play=true
```

可复现实验请显式设置 seed：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

## 常用训练 override

| Override | 含义 |
| --- | --- |
| `training.no_play=true` | 训练后不进入 playback |
| 未设置 `training.log_root` | 规范 `./logs/<algo_log_name>/<task>/<run>/` 树（推荐） |
| `training.log_dir=/absolute/run` | 使用一个精确 run 目录，而不生成时间戳 |
| `algo.seed=1` | 设置训练 seed |
| `training.device=cuda:0` | 在算法支持处选择 learner 设备 |
| `algo.num_envs=N` | 设置向量化环境数量 |
| `algo.max_iterations=N` | 设置训练日程长度 |
| `algo.save_interval=N` | checkpoint 间隔 |
| `training.play_render_mode=none` | 训练后禁用 playback |
| `training.logger=tensorboard` | 使用配置的 TensorBoard logger |

算法专属控制：

| 算法 | 常用控制 |
| --- | --- |
| PPO / APPO | `num_steps_per_env` 或 `steps_per_env`、学习率与 PPO 参数 |
| FlashSAC | `batch_size`、`replay_buffer_n`、`updates_per_step`、`learning_starts` |

精确默认值位于 `src/sharpa_rl_unilab/conf/<algorithm>/`。

## 检查输出

一个完成的 run 目录包含：

- `run_config.json`：完全解析后的配置快照。
- `events.out.tfevents.*`：TensorBoard 标量数据。
- `model_<iteration>.pt`：checkpoint。
- `run_summary.json`：完成状态、环境步数、最终/最佳 return、episode 长度、
  checkpoint 路径和 wall time（由当前 APPO 与 FlashSAC launcher 写入）。
- FlashSAC 评估还可能写入 `policy.onnx` 和 `play_video.mp4`。

按 owner 区分的默认根目录：

```text
logs/rsl_rl_ppo/SharpaInhandRotation/<timestamp>/
logs/appo/SharpaInhandRotation/<timestamp>/
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
```

使用 TensorBoard 监控根目录：

```bash
uv run tensorboard --logdir ./logs --port 6006
```

## 选择或恢复 checkpoint

对 PPO 和 APPO，可将配置日志树中的 run 目录或 checkpoint 文件赋给
`algo.load_run`。`-1` 表示“最新 run”。

```bash
# PPO example: continue the latest run under ./logs/rsl_rl_ppo
uv run sharpa-train --algo ppo --sim mujoco \
  algo.load_run=-1 training.no_play=true

# APPO example: continue an exact run directory
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run \
  training.no_play=true
```

PPO 还可以用 `algo.checkpoint=<iteration>` 选择迭代。当前 FlashSAC launcher
不使用 `algo.load_run` 恢复训练；请保留完整的 FlashSAC run 与 checkpoint，
或启动新的 seeded run。

`training.log_dir` 是为新训练指定精确 run 目录的可移植方式。在当前 UniLab
entrypoints 中，`training.log_root` 的语义随 launcher 不同；文档化的默认流程
应避免使用它。如果 run 是用 `training.log_dir` 创建的，评估时请将该 run 或
checkpoint 路径直接传给 `algo.load_run`。

## 重新生成 grasp cache

抓取生成是单尺度工具流程，不属于常规 teacher 训练。helper 会设置 fixed
variant 并调用 PPO grasp owner：

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

环境变量：

- `SHARPA_GRASP_TARGET`：保存的成功抓取数量。
- `SHARPA_GRASP_NUM_ENVS`：向量化环境数量。
- `SHARPA_GRASP_CACHE_PATH`：输出前缀。

仓库已内置生产 cache；只有在修改手/物体模型或抓取生成策略后才需要重新生成。
