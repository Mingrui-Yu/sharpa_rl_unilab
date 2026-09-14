# 入门

## 前置条件

- 训练推荐使用带 NVIDIA GPU 的 Linux 系统。
- [uv](https://docs.astral.sh/uv/) 0.12 或更新版本。
- 可访问 UniLab 与本仓库的 Git 权限。
- MuJoCo 原生渲染包由 `mujoco` extra 自动安装；不需要系统级 MuJoCo。

本包将 UniLab 视为外部依赖。`pyproject.toml` 中的开发元数据从兄弟目录
`../UniLab` checkout 解析该依赖，因此两个仓库需要并排克隆：

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
```

安装锁定环境：

```bash
uv sync --extra mujoco --extra export
```

准备并校验内置的机器人与抓取资产：

```bash
uv run sharpa-assets
```

这些资产属于 Python 包。只有在需要时才会创建可写修复缓存；
`SHARPA_RL_UNILAB_ASSET_CACHE` 可以将该缓存指向其他目录。

## 校验安装

运行包检查，并检查一个合成后的 Manager-Based 配置：

```bash
uv run ruff check src tests
uv run pytest -q
uv run pyright
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

配置打印命令会在加载仿真资产前退出。这是查看 Hydra override 并确认 owner
是否存在的最快方式。

## 运行一次 one-iteration smoke train

使用较小的环境数和一个显式临时 run 目录。以下命令会覆盖完整 collector、
learner、checkpoint 与 summary 路径，但不会启动基准时长的训练。

HORA APPO：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-smoke
```

FlashSAC：

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-smoke
```

一次成功的 smoke run 会写入 `model_1.pt` 和 `run_summary.json` 到其
`training.log_dir`。

## 下一步

- 启动基准训练：[训练工作流](user-guide/training.md)
- 加载 checkpoint：[评估与回放](user-guide/evaluation.md)
- 理解 terms 与 fixed object variants：[任务参考](reference/task.md)
