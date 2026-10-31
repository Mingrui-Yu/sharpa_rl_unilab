# UniLab Sharpa 手内操作与 HORA

Sharpa Wave 手内旋转任务使用 UniLab 当前 **Manager-Based API**。任务只
拥有 action / observation / reset / event / termination / reward terms；
UniLab 保持外部依赖，负责 manager 生命周期与仿真后端。

SuperMimic、PPO、APPO、FlashSAC 默认使用 HORA teacher/student 流程。
SuperMimic 另提供显式可选的单阶段训练：actor 只依赖可测观测历史，critic 使用特权信息
辅助训练，通过真实回放学习可微世界模型并直接优化最终策略，无需蒸馏。

## 安装与校验

UniLab 1.5.4 及其 CUDA Torch wheel 会从已配置的包索引解析。

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync
uv run pytest
uv run pyright
```

若使用 pip，可在独立虚拟环境中运行 `pip install .`。默认安装 MuJoCo 和
MJWarp；MJWarp 需要 CUDA，四种 teacher 算法均支持 `--sim mjwarp`。训练使用
UniLab 集成的 tensor-native RL runtime：PPO/APPO 采样与 FlashSAC
推理/replay 的观测、动作、奖励和结束标志均保持 Torch tensor 边界，CUDA
训练热路径驻留 GPU。

`uv run pytest` 默认运行快速回归测试；仿真与训练冒烟测试使用
`uv run pytest -m slow`，其中 FlashSAC 冒烟测试需要 CUDA。

## 快速上手

安装完成后，在仓库根目录一键依次训练 SuperMimic、PPO、APPO、FlashSAC，让 Sharpa Wave 手学习手内旋转圆柱体：

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
uv run sharpa-train --algo supermimic
# 可选单阶段 SuperMimic，不走 teacher/student 蒸馏：
uv run sharpa-train --algo supermimic algo.single_stage=true
uv run sharpa-train --algo appo
uv run sharpa-train --algo flashsac
uv run sharpa-train --algo ppo
```

`--cfg` 可打印合成后的 Manager-Based 配置。可在命令末尾追加 Hydra 参数覆盖。

默认训练结束后保存 `teacher_final.pt`，可选单阶段 SuperMimic 保存可直接评估的
`policy_final.pt`，并在同目录录制 `play_video.mp4`。无显示器的 Linux 机器使用
EGL 或 OSMesa 离屏渲染；Ubuntu/Debian 需要安装 `libegl1` 并使用可用的 NVIDIA
图形驱动，或安装 `libosmesa6` 使用软件渲染。`uv sync` 不会安装这些系统依赖；
无需录制时可设置 `training.no_play=true`。

评估与 HORA 蒸馏直接指定 checkpoint 文件：

```bash
uv run sharpa-eval --checkpoint /absolute/path/to/policy_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

`uv run sharpa-compare` 按默认配置依次训练四种算法；追加
`--distill --eval --num-seeds 3` 可运行多 seed 蒸馏与评估。各算法使用各自的训练预算，
不保证等采样量或等计算成本。

任务背景见[手内旋转](docs/zh_CN/task.md)，操作步骤、运行限制和 checkpoint 兼容范围
见[训练指南](docs/zh_CN/training.md)。

## 任务契约

- 物体尺寸随机化使用 fixed model variants
  （`scene_scale_0.8.xml` … `scene_scale_1.5.xml`），使不同 scale 样本保持稳定的
  scene topology。
- fixed variant 中所有 body geom 均唯一命名，保证
  `mjbatch.VariantPack` 选择可靠。
- 所有 variant 的自由物体保留 `simple="false"`，使 reset 时的质量与质心采样保持一致。
- 触觉平滑/延迟、特权信息、位置目标、执行器增益、物体质量/质心/摩擦/重力与
  衰减外力均为显式 manager terms，仅通过 Entity facade 访问状态。

详见 [架构](docs/zh_CN/architecture.md) 与 [验证](docs/VALIDATION.md)。
