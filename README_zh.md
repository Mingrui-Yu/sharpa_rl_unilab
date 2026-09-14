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

`--cfg` 打印合并配置。三个算法共用任务、观测预处理、全局 transition 预算与
固定场景定量评估；`--nodr` 使用同一覆盖，`+preset=throughput` 标记独立吞吐实验。
绘图需 `uv sync --extra mujoco --extra evaluation`；`sharpa-compare --smoke`
执行短预算流程验证。旧 checkpoint 必须重训。当前 v2 入口支持单 learner
设备和从头训练，详见[协议、迁移和验证范围](docs/migrations/issue-2.md)。

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
