# 训练指南

## 选择训练配方

| 目标 | 命令族 |
| --- | --- |
| 训练标准 on-policy baseline | `--algo ppo` |
| 训练 flat asymmetric APPO baseline | `--algo appo` |
| 训练 HORA APPO teacher | `--algo appo --profile hora` |
| 训练 FlashSAC teacher | `--algo flashsac` |
| 重新生成抓取状态 | `--algo ppo --task sharpa_inhand_grasp` |

本指南中的所有生产运行都使用 MuJoCo。

## 训练 teacher

HORA APPO teacher：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

FlashSAC teacher：

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

PPO baseline：

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  algo.seed=1 training.no_play=true
```

保持 `training.log_root` 未设置，并从仓库根目录运行。训练会写入规范目录
树：

```text
logs/<algorithm-log-name>/SharpaInhandRotation/<timestamp>/
```

示例：

```text
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
logs/rsl_rl_ppo/SharpaInhandRotation/<timestamp>/
```

## 检查 run

完成的 run 包含：

- `model_<iteration>.pt`：策略 checkpoint。
- `run_config.json`：精确解析后的配置。
- `run_summary.json`：最终 reward、episode 长度、步数与 checkpoint 路径。
- `events.out.tfevents.*`：TensorBoard 曲线。

观察所有进行中的 run：

```bash
uv run tensorboard --logdir ./logs --port 6006
```

打开：

```text
http://localhost:6006
```

## 常用控制

| 控制 | 用途 |
| --- | --- |
| `algo.seed=1` | 让运行可复现 |
| `training.no_play=true` | 训练后退出，不进入 playback |
| `training.play_render_mode=none` | 禁用训练后 playback |
| `training.play_render_mode=record` | 训练后录制视频 |
| `training.device=cuda:0` | 在支持处选择 learner 设备 |
| `algo.max_iterations=N` | 设置训练日程长度 |
| `algo.save_interval=N` | 设置 checkpoint 间隔 |
| `training.log_dir=/absolute/run` | 使用一个精确输出目录 |

算法专属示例：

```text
PPO/APPO: algo.num_envs, algo.steps_per_env or algo.num_steps_per_env
FlashSAC: algo.batch_size, algo.replay_buffer_n, algo.updates_per_step
```

## 继续 PPO 或 APPO

使用相同仓库根目录并选择之前的 run：

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  algo.seed=1 algo.load_run=-1 training.no_play=true
```

PPO 可以用 `algo.checkpoint=<iteration>` 选择 checkpoint 迭代。当前
FlashSAC launcher 不会从 `algo.load_run` 恢复训练；请保留完成的 FlashSAC
checkpoint，或启动新的 seeded run。

如果 run 使用 `training.log_dir` 创建，评估时请将该 run 或 checkpoint 路径
直接传给 `algo.load_run`。

## 重新生成抓取状态

仓库内置生产 grasp cache。只有修改手、物体或抓取生成策略后才需要重新生
成：

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

有用的环境变量：

```text
SHARPA_GRASP_TARGET=<number of saved grasps>
SHARPA_GRASP_NUM_ENVS=<parallel environments>
SHARPA_GRASP_CACHE_PATH=<output prefix>
```
