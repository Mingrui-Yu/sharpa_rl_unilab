# 入门

## 你需要准备什么

- 训练推荐使用带 NVIDIA GPU 的 Linux 系统。
- [uv](https://docs.astral.sh/uv/) 0.12 或更新版本。
- 可访问 UniLab 与本仓库的 Git 权限。
- 不需要显示服务；评估可以录制离屏视频。

## 与 UniLab 并排安装

本包将 UniLab 视为外部任务 runtime。开发元数据从兄弟 checkout 解析
UniLab：

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco --extra export
```

准备内置的机器人、场景与 grasp-cache 资产：

```bash
uv run sharpa-assets
```

常规训练和评估不需要下载 Hugging Face 资产。

## 检查安装

在不启动物理仿真的情况下打印两个完整 Hydra 配置：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

然后运行包测试：

```bash
uv run pytest -q
```

## 运行两分钟完整性检查

最小完整训练使用 4 个环境和 1 次 iteration。它会验证资产加载、仿真、采
集、学习、checkpoint 与 run summary。

HORA APPO：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-check
```

FlashSAC：

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-check
```

每次成功运行都会写出一个 checkpoint 和 `run_summary.json` 到配置的
`training.log_dir`。

## 下一步

- 启动完整 teacher 训练：[训练指南](training.md)
- 加载 checkpoint 并录制视频：[评估指南](evaluation.md)
- 了解任务、观测与随机化：[任务指南](task.md)
