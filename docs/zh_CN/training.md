# 训练、蒸馏与评估

先按 [README](../../README_zh.md#安装与校验) 安装环境。
以下命令在仓库根目录执行；任务背景见[手内旋转](task.md)。

## 1. 准备抓取缓存

默认使用包内缓存，无需配置。如需重新生成全部八个尺度：

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
```

生成脚本逐尺度运行独立的 PPO 抓取任务，默认输出为 `caches/sharpa_grasp_linspace_<尺度>.npy`。
如需自定义位置，用环境变量 `SHARPA_GRASP_CACHE_PATH` 指定输出前缀，
训练 teacher 时再通过 `env.events.reset.params.grasp_cache_path` 指定相同前缀。

## 2. 训练 teacher

以 APPO 为例；将 `--algo appo` 改为 `ppo` 或 `flashsac` 可切换算法。

```bash
uv run sharpa-train --algo appo
```

默认输出到 `logs/<算法>/seed_<seed>_<时间戳>/`，最终模型为 `teacher_final.pt`，
包含运行配置和归一化统计。同目录保存配置快照、`metrics.jsonl` 和 TensorBoard 日志。

FlashSAC teacher 训练需要 CUDA/MPS，仅支持按更新轮数停止。
PPO/APPO 也支持采样预算：同时设置 `algo.max_iterations=null training.max_transitions=N`。
采样数向上取整到一个向量步，最多多采集“环境数减 1”条 transition；保存仍按轮触发。

### 策略分布

PPO、APPO、FlashSAC 共用 HORA Actor 与高斯分布实现。新训练默认
`model.std_mode=state_independent`：每个关节一个可学习的 log-std，
`model.initial_std=1.0` 指定实际初始 std。设为 `state_dependent` 时，
从 Actor 主干特征输出 log-std；输出头权重初始化为零，偏置为 `log(initial_std)`，
因此两种模式在初始时都具有指定的实际 std。

三种算法均默认 `model.action_mapping=tanh`；PPO/APPO 也支持 `clip`，FlashSAC 必须使用 `tanh`。
例如：

```bash
uv run sharpa-train --algo appo model.action_mapping=tanh \
  model.std_mode=state_independent model.initial_std=1.0 \
  algo.algorithm.entropy_coef=0.01
```

`model.std_parameterization=log` 使用 `std=exp(log_std)`。
`model.log_std_bounds=null` 表示不约束 log-std；可选用有限递增区间，
例如 `model.log_std_bounds=[-10,2]`，在 exp 前裁剪。初始 std 必须处于该范围内；
裁剪会改变区间外的梯度。网络保留各算法原有 AMP 设置，采样、密度、熵和 KL
使用 FP32 计算。更完整的 AMP 数值审计仍为后续专项 TODO。

tanh 的确定性动作为 `tanh(mean)`，熵奖励使用当前策略新采样估计变换后的动作熵。
clip 的训练密度、熵和 KL 指潜在高斯分布。PPO/APPO 均保存映射前的原始样本、
行为 log-prob 和 mean/std。自适应学习率保留原有规则，使用 `KL(old || current)`；
APPO 在该调度中使用目标策略作为旧分布。公共解析 KL 移除了上游的加性数值偏移：
相同策略的 KL 为零，不再触发学习率增大；阈值和调整倍数保持原值。

FlashSAC 采集仍按环境保持高斯噪声，由 `algo.exploration.noise_zeta_mu=2.0`
和 `algo.exploration.noise_zeta_max=16` 控制；最大持续步数为 1 时每步刷新。
runner 在 Learner 设备上持有噪声状态，Actor/Critic 更新和确定性评估均不推进它，
原生 IPC、推理及更新调度保持不变。

新默认配置将 FlashSAC 从原有的状态相关、有界 std 改为初始值为 1 的状态无关 log-std，
因此是新的训练设置，不能视为旧训练轨迹的复现。已有 protocol-v2 teacher/student
checkpoint 通过明确的兼容路径保留原来的 scalar/tanh std 参数化和动作映射。
评估与蒸馏不允许覆盖 checkpoint 的模型配置。

## 3. 蒸馏 student

加载 teacher，训练历史编码器来拟合 teacher 的特权表示。
Teacher 的策略和基础归一化统计保持冻结。

```bash
uv run sharpa-distill --checkpoint /path/to/teacher_final.pt
```

将 `/path/to/teacher_final.pt` 替换为实际文件路径。
默认输出到 `logs/hora_distill/<算法>_seed_<seed>_<时间戳>/`，最终模型为 `student_final.pt`。
Student 继承 teacher 的任务和抓取缓存配置，可调整 `distillation.*`、`evaluation.*`、设备、线程、日志及录像设置，
不接受 `env.*` 或模型配置覆盖。评估和蒸馏均可通过 `training.device=cpu` 使用 CPU。

## 4. 评估与算法比较

评估时直接指定 checkpoint：

```bash
uv run sharpa-eval --checkpoint /path/to/teacher_final.pt
```

替换为实际文件路径；评估 student 时指定 `student_final.pt` 即可。

默认在八种尺度上各使用 3 个评估 seed、每个 seed 10 个 episode。
每个 episode 最长 20 秒，提前掉落不补跑；各尺度等权汇总。
结果记录回报、存活时间、掉落率、有符号旋转角，以及按固定窗口和实际存活时间计算的转速。
结果保存为 checkpoint 旁的 `<模型名>.evaluation.json`，场景清单为 `scenes.json`。

评估恢复 checkpoint 中的任务，仅允许覆盖 `evaluation.*` 和 `training.device`
（也可用 `algo.checkpoint` 指定文件）。复用场景清单时会校验任务配置、抓取和随机化值。

统一训练三种算法：

```bash
uv run sharpa-compare
```

默认仅训练每种算法的一个 teacher。需要蒸馏时追加 `--distill`，需要评估时追加 `--eval`；
`--num-seeds N` 从各算法配置的 seed 起连续取 N 个。
启用对应阶段后，所有 teacher 完成才开始 student，全部训练完成后再评估。
输出位于 `logs/compare/seed_<seed>_<时间戳>/<算法>/`，student 位于其 `student/` 子目录。
所有评估共用第一个 seed 目录下的场景清单；各算法使用自己的预算，不保证等采样量或耗时。
该命令不自动汇总多 seed 或绘图。比较多 seed 时，以训练 seed 为独立重复，避免把 episode 当作独立训练结果。

可选评估诊断通过 `evaluation.diagnostics=true` 开启，报告实际访问状态的 std、
动作饱和率（`abs(action)>0.99`）、相邻动作变化 RMS、关节位置二阶差分 RMS 和目标限位率。
每场独立计算，包含真实终止步，再按尺度权重汇总，避免跨重置边界计算运动差分。
诊断不采样策略随机数，不改变评估轨迹。

## 5. 常用设置与兼容性

训练命令末尾可追加 Hydra 参数覆盖；用 `uv run sharpa-train --algo appo --cfg`
查看完整合并配置。公共参数见[公共配置](../../src/sharpa_rl_unilab/conf/common/sharpa_inhand.yaml)，
算法参数见[配置目录](../../src/sharpa_rl_unilab/conf)。

例如，减少 teacher 环境数：

```bash
uv run sharpa-train --algo appo training.num_envs=1024
```

| 设置 | 参数与默认值 |
| --- | --- |
| Teacher 环境数 | `training.num_envs=2048` |
| 输出目录 | 默认自动创建；用 `training.log_dir=/path/to/new_run` 指定尚不存在的目录 |
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

实现约定见[架构说明](architecture.md)，验证范围见[验证记录](../VALIDATION.md)。
