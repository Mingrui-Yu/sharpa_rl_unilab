# Sharpa HORA：从抓取到 student

```text
稳定抓取状态缓存 → HORA teacher 学习旋转 → student 学习历史信息适应 → 固定场景评估
```

按 [README](../../README_zh.md) 安装环境后，在仓库根目录、同一终端依次执行。
PPO、APPO、FlashSAC 默认使用同一 HORA Actor，并共用下面的蒸馏与评估流程。

## 1. Grasp generation：准备初始抓取

采样手部姿态，在仿真中筛选稳定抓取，保存关节位置和物体位姿，供训练时 reset 使用。
包内已附带缓存，快速复现可跳过生成，仅设置：

```bash
export SHARPA_GRASP_CACHE_PATH=caches/sharpa_grasp_linspace
```

这个相对前缀由资产解析器定位到包内缓存。若要重新生成，改用独立的绝对输出前缀，
并覆盖全部八个尺度：

```bash
export SHARPA_GRASP_CACHE_PATH="$PWD/generated_grasps/sharpa_grasp_linspace"
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh \
  0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
```

产物为 `<前缀>_<scale>.npy`，其中尺度 1 使用 `_1.npy`。teacher 显式读取此前缀，
student 和评估从 checkpoint 继承。抓取生成沿用独立的 PPO 工具入口，作为旋转训练的准备步骤。

## 2. Teacher：用特权信息学习旋转

teacher Actor 读取 147 维可测基础历史与独立的当前 9 维 `priv_info`，在域随机化下
学习手内旋转。特权信息通过观测组传入；Critic 独立读取 174 维 clean 历史。

下面以 APPO 为例；将 `--algo appo` 换为 `ppo` 或 `flashsac` 即可训练另外两种算法。
无需指定 `--profile hora`。为本次训练选择尚不存在的输出目录，重复运行时更换目录名：

```bash
export SHARPA_TEACHER_RUN="$PWD/logs/appo_teacher_1"
uv run sharpa-train --algo appo \
  training.log_dir="$SHARPA_TEACHER_RUN" \
  env.events.reset.params.grasp_cache_path="$SHARPA_GRASP_CACHE_PATH"
```

最终模型为 `$SHARPA_TEACHER_RUN/teacher_final.pt`，保存完整运行配置与归一化统计。
默认使用单 learner、2048 个环境、500 万个新 transition；可通过 `hardware.num_envs`
与 `budget.transitions` 调整规模。需要关闭域随机化时，在 teacher 命令中添加 `--nodr`，
student 和评估会继承该配置。

## 3. Student：从历史观测估计特权表示

加载 teacher，冻结其特权编码器、策略主干、动作头与基础归一化统计，仅通过 latent MSE
训练 30 帧历史编码器。每个向量步先更新一次编码器，再执行更新后的 student 确定性动作。
训练时用仿真特权信息监督；推理只需要 147 维基础观测与形状为 `[N,30,49]` 的历史。

同样选择尚不存在的输出目录，并直接指定上一步的 checkpoint 文件：

```bash
export SHARPA_STUDENT_RUN="$PWD/logs/appo_student_1"
uv run sharpa-distill --checkpoint "$SHARPA_TEACHER_RUN/teacher_final.pt" \
  training.log_dir="$SHARPA_STUDENT_RUN"
```

最终模型为 `$SHARPA_STUDENT_RUN/student_final.pt`。默认蒸馏预算为 1 亿个 transition、
2048 个环境、学习率 `3e-4`，可通过 `distillation.transitions`、`distillation.num_envs`
与 `distillation.learning_rate` 调整。

蒸馏继承 teacher 的完整环境配置，包括抓取缓存前缀，不接受 `env.*` 覆盖。
本分支 PPO、APPO、FlashSAC 生成的 v2 teacher checkpoint 均支持此流程；旧版 HORA、
flat PPO 和原生 FlashSAC checkpoint 需要重新训练。checkpoint 格式与迁移限制详见
[协议与迁移记录](../migrations/issue-2.md)。

## 4. 评估 teacher 与 student

```bash
uv run sharpa-eval --checkpoint "$SHARPA_TEACHER_RUN/teacher_final.pt"
uv run sharpa-eval --checkpoint "$SHARPA_STUDENT_RUN/student_final.pt"
```

评估从 checkpoint 恢复任务与归一化统计，在八个尺度的固定场景中使用 20 秒窗口，
提前掉落不补跑。结果分别写入 checkpoint 同目录的 `teacher_final.evaluation.json`
与 `student_final.evaluation.json`。评估仅接受 `evaluation.*` 与 `hardware.device` 配置覆盖。

需要统一运行三种算法、多训练种子的 teacher、student 与评估时，使用
`uv run sharpa-compare --output logs/comparison --seeds 1 2 3`，输出目录须尚不存在；
绘图依赖通过 `uv sync --extra mujoco --extra evaluation` 安装。验证范围见
[VALIDATION.md](../VALIDATION.md)。
