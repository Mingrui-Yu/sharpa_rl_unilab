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

`uv run pytest` / `make test` 默认运行快速回归测试。MuJoCo、内置资源编译以及
每种算法各一条训练 → 保存 → 评估 → 蒸馏冒烟测试，通过
`uv run pytest -m slow` / `make test-slow` 显式运行。
FlashSAC 冒烟测试需要 CUDA，不可用时跳过。
`make test-all` 运行静态检查及两组测试。

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

`--cfg` 打印合并配置。三个算法通过 `conf/common/sharpa_inhand.yaml` 共用任务、
观测预处理、模型和运行设置。teacher 默认 2048 个环境，Learner 使用 `cuda:0`，
Learner 与独立 Collector 均为 4 个 Torch intra-op 线程、1 个 inter-op 线程。
APPO 采样推理默认跟随 Learner，可用 `algo.collector_device=cpu` 覆盖。
`training.device=null` 自动选择 CUDA、MPS 或 CPU。

APPO/PPO 默认训练 501 轮，FlashSAC 默认 3000 轮。三个 teacher 均每轮记录，
每 50 轮保存；`algo.save_interval=0` 关闭中间保存，正常结束仍保存 `teacher_final.pt`。
相同轮数不代表相同采样量或计算量。PPO/APPO 按采样量停止时，显式设置
`algo.max_iterations=null training.max_transitions=N`，保存仍按轮触发。
训练不再调度定量评估；独立评估入口和 `sharpa-compare` 训练后的显式评估保留。

FlashSAC 仅使用 UniLab DoubleBuffer 流水线并关闭 AMP，要求 `algo.max_iterations>0`、
`training.max_transitions=null`。teacher 训练需要 CUDA/MPS，评估与蒸馏可使用 CPU。
clean174 Q 输入使用只由新采样更新的经验统计，
当前 Q 与目标 Q 共用统计量，保留原生 Q 结构。详见
[FlashSAC 实现与限制](docs/migrations/issue-2-flashsac-native.md)。

运行资源与各项随机化使用显式参数调节，例如：

```bash
uv run sharpa-train --algo flashsac training.num_envs=1024 training.torch_threads.learner_num_threads=8
uv run sharpa-train --algo appo training.device=cuda:0 algo.collector_device=cpu
```

teacher 和 student 默认显示 UniLab Rich 面板，并写入 TensorBoard 和 `metrics.jsonl`。
`uv run tensorboard --logdir logs` 查看曲线，横轴为实际收到的新 transition 数。
`training.logger=none` 关闭 TensorBoard，`training.logger=no_print` 仅保留 JSONL。
student 仍按 `distillation.log_every`（默认 10000 条采样）记录，停止和保存规则不变。

`sharpa-compare` 使用各算法配置的预算，在相同场景评估最终模型，记录实际采样量与耗时，
不保证等采样量或等计算成本。`--smoke` 执行 2 轮 teacher 更新和 32 条 student 采样。
绘图需 `uv sync --extra mujoco --extra evaluation`。现有契约内的 v2 checkpoint
加载时迁移配置路径，保留历史模型与环境语义；更早的不兼容格式仍需重训。
当前入口支持单 Learner 和从头训练，详见[迁移说明](docs/migrations/issue-2.md)。

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
