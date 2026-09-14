# UniLab Sharpa 手内操作与 HORA

Sharpa Wave 手内旋转任务已迁移到 UniLab 当前 **Manager-Based API**。任务只
拥有 action / observation / reset / event / termination / reward terms；
UniLab 保持外部依赖，负责 manager 生命周期与仿真后端。此前任务自有的
legacy/direct `NpEnv` 实现和 compatibility factory 已移除。

PPO、APPO、FlashSAC 默认使用同一 HORA teacher setup，均支持统一 student
蒸馏；Critic 独立建模，保留算法所需的 V/Q 差异。

## 安装与校验

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco
uv run pytest
uv run pyright
```

开发环境通过兄弟目录 `../UniLab` 作为 `unilab` source；包依赖仍是外部的
`unilab>=1.2.0,<1.3`，`unilab-rl>=1.2.0,<1.3` 来自发布包。

## 训练、评估与蒸馏

```bash
uv run sharpa-train --algo appo
uv run sharpa-train --algo flashsac
uv run sharpa-train --algo ppo
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
uv run sharpa-compare --output logs/comparison --seeds 1 2 3
```

`--cfg` 打印合并配置。三个算法共用任务、观测预处理与
固定场景定量评估；`--nodr` 使用同一覆盖，`+preset=throughput` 标记独立吞吐实验。
绘图需 `uv sync --extra mujoco --extra evaluation`；`sharpa-compare --smoke`
执行短预算流程验证。旧 checkpoint 必须重训。当前 v2 入口支持单 learner
设备和从头训练，详见[协议、迁移和验证范围](docs/migrations/issue-2.md)。

teacher 和 student 默认显示 UniLab Rich 终端面板，并在运行目录写入 TensorBoard
事件与完整的 `metrics.jsonl`。运行 `uv run tensorboard --logdir logs` 查看曲线；
横轴为实际收到的新 transition 数。`training.logger=none` 只关闭 TensorBoard，
`training.logger=no_print` 仅保留 JSONL。APPO 和原生 FlashSAC 每轮记录指标，按更新轮数显示进度和 ETA；
其余训练使用 `budget.log_every` 控制记录间隔。

APPO 默认使用 2048 个环境，训练 305 轮，每 51 轮保存一次；Learner 自动选择
CUDA、MPS 或 CPU，Collector 默认跟随 Learner，可用 `hardware.collector_device=cpu`
覆盖。历史池为 8 批，待接收队列为 4 批。采样预算实验须显式设置
`algo.max_iterations=null budget.transitions=N`，保存间隔仍使用 `algo.save_interval`。
PPO 默认使用 2048 个环境，训练 301 轮，每轮每环境采样 8 步，共 4,931,584 条
transition；进度和 ETA 按轮数显示，checkpoint 仍按 `budget.save_every` 保存，默认
间隔为 1,000,000 条 transition。改用采样预算时须显式设置
`algo.max_iterations=null budget.transitions=N`。FlashSAC 默认使用 2048 个环境、
10000 轮更新，复用 UniLab DoubleBuffer 流水线，关闭 AMP，按进程角色自动配置线程。
checkpoint 仍按 `budget.save_every` 的实际采样阈值保存。显式设置
`algo.max_iterations=null budget.transitions=N` 才使用同步采样兼容路径，
详见 [FlashSAC 实现与限制](docs/migrations/issue-2-flashsac-native.md)。

`sharpa-compare` 为各算法显式选择统一采样预算。

## 正确性说明

- 物体尺寸随机化使用 UniLab fixed model variants
  （`scene_scale_0.8.xml` … `scene_scale_1.5.xml`），不再运行时改 geom size。
- fixed variant 中所有 body geom 均唯一命名，满足 `mjbatch.VariantPack`。
- 所有 variant 的自由物体保留 `simple="false"`，避免 reset 时 mass/CoM DR 触发
  `mj_setConst` sameframe 崩溃。
- 触觉平滑/延迟、特权信息、位置目标、执行器增益、物体质量/质心/摩擦/重力与
  衰减外力均为显式 manager terms，仅通过 Entity facade 访问状态。

详见 [架构](docs/ARCHITECTURE.md)、[验证](docs/VALIDATION.md) 与
[完整训练流程](docs/zh_CN/8-training_pipeline.md)。

原生 TensorBoard 日志保留含 `/` 的指标名，其余指标位于 `train/` 下；完整原始字段
保存在 JSONL。组件复用与验证范围见[训练代码简化结果](docs/migrations/issue-2-simplify.md)。
