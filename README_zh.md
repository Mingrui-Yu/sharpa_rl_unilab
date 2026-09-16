# UniLab Sharpa 手内操作与 HORA

Sharpa Wave 手内旋转任务已迁移到 UniLab 当前 **Manager-Based API**。任务只
拥有 action / observation / reset / event / termination / reward terms；
UniLab 保持外部依赖，负责 manager 生命周期与仿真后端。此前任务自有的
legacy/direct `NpEnv` 实现和 compatibility factory 已移除。

支持 PPO、APPO、FlashSAC teacher，三种算法默认共用 HORA 策略与 student
蒸馏流程；Critic 保留各算法所需的 V/Q 结构。

## 安装与校验

源码开发需要先准备兼容 UniLab 1.2 的兄弟目录 `../UniLab`，再执行 `uv sync`。

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco
uv run pytest
uv run pyright
```

开发环境通过兄弟目录 `../UniLab` 作为 `unilab` source；包依赖仍是外部的
`unilab>=1.2.0,<1.3`，`unilab-rl>=1.2.0,<1.3` 来自发布包。若要使用发布版依赖，可在独立虚拟环境中
运行 `pip install ".[mujoco]"`；pip 不采用 `tool.uv.sources` 中的本地目录映射。

`uv run pytest` 默认运行快速回归测试；仿真与训练冒烟测试使用
`uv run pytest -m slow`，其中 FlashSAC 冒烟测试需要 CUDA。

## 快速上手

安装完成后，在仓库根目录一键依次训练 PPO、APPO、FlashSAC，让 Sharpa Wave 手学习手内旋转圆柱体：

```bash
uv run sharpa-compare
```

另开终端启动 TensorBoard，浏览器打开 <http://localhost:6006> 查看训练曲线：

```bash
uv run tensorboard --logdir logs/compare
```

每种算法训练结束后，回放视频保存至
`logs/compare/seed_<seed>_<时间戳>/<算法>/play_video.mp4`。

## 训练、评估与蒸馏

```bash
uv run sharpa-train --algo appo
uv run sharpa-train --algo flashsac
uv run sharpa-train --algo ppo
```

`--cfg` 可打印合成后的 Manager-Based 配置。可在命令末尾追加 Hydra 参数覆盖。

训练结束后保存 `teacher_final.pt`，并在同目录录制 `play_video.mp4`。
无显示器的 Linux 机器使用 EGL 或 OSMesa 离屏渲染；Ubuntu/Debian 需要安装
`libegl1` 并使用可用的 NVIDIA 图形驱动，或安装 `libosmesa6` 使用软件渲染。
`uv sync` 不会安装这些系统依赖；无需录制时可设置 `training.no_play=true`。

评估与蒸馏直接指定 checkpoint 文件：

```bash
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

`uv run sharpa-compare` 按默认配置依次训练三种算法；追加
`--distill --eval --num-seeds 3` 可运行多 seed 蒸馏与评估。
各算法使用各自的训练预算，不保证等采样量或等计算成本。

任务背景见[手内旋转](docs/zh_CN/task.md)，操作步骤、运行限制和 checkpoint 兼容范围
见[训练指南](docs/zh_CN/training.md)。

## 正确性说明

- 物体尺寸随机化使用 UniLab fixed model variants
  （`scene_scale_0.8.xml` … `scene_scale_1.5.xml`），不再运行时改 geom size。
- fixed variant 中所有 body geom 均唯一命名，满足 `mjbatch.VariantPack`。
- 所有 variant 的自由物体保留 `simple="false"`，避免 reset 时 mass/CoM DR 触发
  `mj_setConst` sameframe 崩溃。
- 触觉平滑/延迟、特权信息、位置目标、执行器增益、物体质量/质心/摩擦/重力与
  衰减外力均为显式 manager terms，仅通过 Entity facade 访问状态。

详见 [架构](docs/zh_CN/architecture.md) 与 [验证](docs/VALIDATION.md)。
