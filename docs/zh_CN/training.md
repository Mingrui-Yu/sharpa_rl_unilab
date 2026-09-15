# 训练、蒸馏与评估

先按 [README](../../README_zh.md#安装与校验) 安装环境。
以下命令在仓库根目录、同一终端执行；任务背景见[手内旋转](task.md)。

## 1. 准备抓取缓存

默认使用包内缓存，设置路径前缀即可：

```bash
export SHARPA_GRASP_CACHE_PATH=caches/sharpa_grasp_linspace
```

如需重新生成，改用独立的绝对路径，并生成全部八个尺度：

```bash
export SHARPA_GRASP_CACHE_PATH="$PWD/generated_grasps/sharpa_grasp_linspace"
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
```

产物为 `<前缀>_<尺度>.npy`，尺度 1 对应 `_1.npy`。
生成脚本逐尺度运行独立的 PPO 抓取任务。

## 2. 训练 teacher

以 APPO 为例；将 `--algo appo` 改为 `ppo` 或 `flashsac` 可切换算法。
每次运行选择尚不存在的输出目录：

```bash
export SHARPA_TEACHER_RUN="$PWD/logs/appo_teacher_1"
uv run sharpa-train --algo appo \
  training.log_dir="$SHARPA_TEACHER_RUN" \
  env.events.reset.params.grasp_cache_path="$SHARPA_GRASP_CACHE_PATH"
```

最终模型为 `$SHARPA_TEACHER_RUN/teacher_final.pt`，包含运行配置和归一化统计。
运行目录同时保存配置快照、`metrics.jsonl` 和 TensorBoard 日志。

FlashSAC teacher 训练需要 CUDA/MPS，仅支持按更新轮数停止。
PPO/APPO 也支持采样预算：同时设置 `algo.max_iterations=null training.max_transitions=N`。
采样数向上取整到一个向量步，最多多采集“环境数减 1”条 transition；保存仍按轮触发。

## 3. 蒸馏 student

加载 teacher，训练历史编码器来拟合 teacher 的特权表示。
Teacher 的策略和基础归一化统计保持冻结。

```bash
export SHARPA_STUDENT_RUN="$PWD/logs/appo_student_1"
uv run sharpa-distill --checkpoint "$SHARPA_TEACHER_RUN/teacher_final.pt" \
  training.log_dir="$SHARPA_STUDENT_RUN"
```

最终模型为 `$SHARPA_STUDENT_RUN/student_final.pt`。Student 继承 teacher 的任务和
抓取缓存配置，可调整 `distillation.*`、`evaluation.*`、设备、线程、日志及录像设置，
不接受 `env.*` 或模型配置覆盖。评估和蒸馏均可通过 `training.device=cpu` 使用 CPU。

## 4. 评估与算法比较

评估时直接指定 checkpoint：

```bash
uv run sharpa-eval --checkpoint "$SHARPA_TEACHER_RUN/teacher_final.pt"
uv run sharpa-eval --checkpoint "$SHARPA_STUDENT_RUN/student_final.pt"
```

默认在八种尺度上各使用 3 个评估 seed、每个 seed 10 个 episode。
每个 episode 最长 20 秒，提前掉落不补跑；各尺度等权汇总。
结果记录回报、存活时间、掉落率、有符号旋转角，以及按固定窗口和实际存活时间计算的转速。
结果保存为 checkpoint 旁的 `<模型名>.evaluation.json`，场景清单为 `scenes.json`。

评估恢复 checkpoint 中的任务，仅允许覆盖 `evaluation.*` 和 `training.device`
（也可用 `algo.checkpoint` 指定文件）。复用场景清单时会校验任务配置、抓取和随机化值。

统一训练、蒸馏并评估三种算法：

```bash
uv run sharpa-compare --distill --eval --num-seeds 3
```

默认仅训练每种算法的一个 seed；`--num-seeds N` 从各算法配置的 seed 起连续取 N 个。
所有 teacher 完成后才开始 student，全部训练完成后再评估。
输出位于 `logs/compare/seed_<seed>_<时间戳>/<算法>/`，student 位于其 `student/` 子目录。
所有评估共用第一个 seed 目录下的场景清单；各算法使用自己的预算，不保证等采样量或耗时。
该命令不自动汇总多 seed 或绘图。比较多 seed 时，以训练 seed 为独立重复，避免把 episode 当作独立训练结果。

## 5. 常用设置与兼容性

训练命令末尾可追加 Hydra 参数覆盖；用 `uv run sharpa-train --algo appo --cfg`
查看完整合并配置。公共参数见[公共配置](../../src/sharpa_rl_unilab/conf/common/sharpa_inhand.yaml)，
算法参数见[配置目录](../../src/sharpa_rl_unilab/conf)。

| 设置 | 参数与默认值 |
| --- | --- |
| Teacher 环境数 | `training.num_envs=2048` |
| 设备 | `training.device=cuda:0`；`null` 自动选择 CUDA、MPS、CPU；APPO 的 `algo.collector_device=null` 跟随 learner |
| 每进程线程 | `training.torch_threads.*`：learner/collector 各 4 个 intra-op、1 个 inter-op 线程 |
| Teacher 轮数 | `algo.max_iterations`：PPO/APPO 为 501，FlashSAC 为 2000 |
| Teacher 保存间隔 | `algo.save_interval`：PPO/APPO 为 50，FlashSAC 为 500；0 关闭中间保存，正常结束仍保存 final |
| FlashSAC 精度 | `algo.use_amp=true`；设为 `false` 关闭混合精度 |
| Student 预算 | `distillation.transitions=100000000`、`distillation.num_envs=4096` |
| Student 学习率 | `distillation.learning_rate=0.0003` |
| Student 保存与日志 | `distillation.save_every=10000000`、`distillation.log_every=10000`，单位为 transition |
| 日志 | Teacher 每轮记录；`training.logger=none` 关闭 TensorBoard，`no_print` 仅保留 JSONL |
| 录像 | 默认训练后保存 `play_video.mp4`；`training.no_play=true` 关闭录像 |

用 `uv run tensorboard --logdir logs` 查看曲线，横轴为实际接收的新 transition 数。
中间 teacher 文件名为 `teacher_iteration_N.pt`。采样数与数据复用次数分别记录，不能用轮数推算实际采样量。

录像用于查看动作，定量评估需单独执行。无显示器的 Linux 机器需要 EGL 或 OSMesa，
安装说明见 [README](../../README_zh.md#训练评估与蒸馏)。

当前支持本项目 v2 teacher/student checkpoint 的评估，以及 v2 teacher 的蒸馏。
加载旧 v2 配置时自动迁移字段，保留原有任务、模型和归一化设置；
旧同步 FlashSAC 生成的受支持 v2 checkpoint 也可加载。
旧版共享 HORA、flat PPO 和原生 FlashSAC 的不兼容格式需要重训。
当前入口仅支持单 learner 和从头训练，不支持训练断点恢复或 v2 的 JIT/ONNX 导出。

实现约定见[架构说明](../ARCHITECTURE.md)，验证范围见[验证记录](../VALIDATION.md)。
