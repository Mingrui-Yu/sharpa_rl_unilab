# UniLab Sharpa 手内操作与 HORA

Sharpa Wave 灵巧手内旋转任务，提供 PPO、APPO、SAC teacher 以及 HORA student
蒸馏。本仓库是从 UniLab 与 unilab_rl 拆出的任务、配置、机器人资源、抓取缓存
和 HORA 实现的唯一维护位置。

[English](README.md) · [拆分 roadmap：UniLab #1547](https://github.com/unilabsim/UniLab/issues/1547)

## 安装

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco
```

Motrix 使用 `uv sync --extra motrix`；需要两个后端时同时选择两个 extra。
开发期间 UniLab 与 unilab-rl 通过 [pyproject.toml](pyproject.toml) 中的本地
editable path 来源解析（`../UniLab`、`../unilab_rl`）。

约 40 MB 的机器人 XML、网格和抓取缓存随 Git 仓库与 Python 包提供。
加载机器人资源无需访问 Hugging Face，也无需运行时联网。
`uv run sharpa-assets` 准备可写本地缓存；
可通过 `SHARPA_RL_UNILAB_ASSET_CACHE` 指定缓存目录。

## 训练与回放

| 算法 | 已提供的 owner 配置 |
| --- | --- |
| PPO | MuJoCo、Motrix、MuJoCo HORA |
| APPO | MuJoCo、Motrix、MuJoCo HORA |
| SAC（HORA teacher） | MuJoCo |
| HORA 蒸馏 | MuJoCo |

该表列出配置范围，不代表训练效果基准。

```bash
uv run sharpa-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run sharpa-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run sharpa-train --algo appo --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --sim mujoco training.no_play=true
```

任务默认为 `sharpa_inhand`；抓取缓存采集使用 `--task sharpa_inhand_grasp`，
HORA teacher 变体使用 `--profile hora`。调参时追加 Hydra override；
`--cfg` 打印组合后的配置；通过 `--sim` 选择后端。

评估时将 `algo.load_run` 指向包含 checkpoint 的 run 目录；
`algo.checkpoint=-1` 选择其最新 checkpoint。

```bash
uv run sharpa-eval --algo ppo --sim mujoco training.log_root=./logs/eval-ppo algo.load_run=/absolute/path/to/ppo/run algo.checkpoint=-1
uv run sharpa-eval --algo appo --sim mujoco --profile hora algo.load_run=/absolute/path/to/hora-appo/run algo.checkpoint=-1
```

Student 蒸馏通过专用入口运行：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=/absolute/path/to/teacher/run
```

teacher/student 流程见 [HORA](docs/zh_CN/7-hora.md)；抓取缓存采集与逐 scale
override 见[灵巧手内操作](docs/zh_CN/1-dexterous_inhand.md)。

## 工具与文档

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
uv run python -m sharpa_rl_unilab.tools.benchmark_sharpa_init_dr_construct
```

采集脚本对每个 scale 依次驱动 `sharpa-train` 的 grasp owner；benchmark 测量
init-DR 构造成本与 variant 数量的关系。

- [任务 owner 与命令](docs/zh_CN/1-dexterous_inhand.md)
- [HORA teacher/student](docs/zh_CN/7-hora.md)
- [架构与拆分记录](docs/ARCHITECTURE.md)
- [已执行的迁移检查](docs/VALIDATION.md)
- [来源清单](MIGRATION_MANIFEST.json) 与 [许可说明](NOTICE.md)
