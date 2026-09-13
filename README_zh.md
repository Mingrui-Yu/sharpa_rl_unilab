# UniLab Sharpa 手内操作与 HORA

Sharpa Wave 手内旋转任务已迁移到 UniLab 当前 **Manager-Based API**。任务只
拥有 action / observation / reset / event / termination / reward terms；
UniLab 保持外部依赖，负责 manager 生命周期与仿真后端。此前任务自有的
legacy/direct `NpEnv` 实现和 compatibility factory 已移除。

支持 PPO、HORA APPO、FlashSAC teacher，以及基于 APPO teacher 的 HORA student
蒸馏。

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

## 训练与回放

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo flashsac --sim mujoco training.no_play=true
uv run sharpa-train --algo ppo --sim mujoco training.no_play=true
```

`--cfg` 可打印合成后的 Manager-Based 配置。评估时将 `algo.load_run` 指向运行
目录，并用 `algo.checkpoint=-1` 选择最新 checkpoint。

## 正确性说明

- 物体尺寸随机化使用 UniLab fixed model variants
  （`scene_scale_0.8.xml` … `scene_scale_1.5.xml`），不再运行时改 geom size。
- fixed variant 中所有 body geom 均唯一命名，满足 `mjbatch.VariantPack`。
- 所有 variant 的自由物体保留 `simple="false"`，避免 reset 时 mass/CoM DR 触发
  `mj_setConst` sameframe 崩溃。
- 触觉平滑/延迟、特权信息、位置目标、执行器增益、物体质量/质心/摩擦/重力与
  衰减外力均为显式 manager terms，仅通过 Entity facade 访问状态。

详见 [架构](docs/ARCHITECTURE.md)、[验证](docs/VALIDATION.md) 与
[HORA](docs/zh_CN/7-hora.md)。
