# Issue 2：统一 HORA teacher/student 实施记录

依据：2026-09-14 的 `dist/journals/01-analysis-refactor.md` 与 issue 2。
按用户补充约束，所有实现位于 Sharpa；UniLab、unilab-rl 及其已安装源码均未修改。

## 输入与模型

三个旋转入口现在都默认训练 HORA teacher，每个算法只有 `mujoco.yaml`。
`conf/common/task.yaml` 定义物理任务、奖励与观测，`common/protocol.yaml`
定义公共模型、硬件、预算和评估，`--nodr` 应用同一份公共覆盖。

| 输入 | 布局 |
| --- | --- |
| obs | 带噪基础帧49 × 3，147维 |
| critic | [clean基础49, 特权9] × 3，174维 |
| priv_info | 当前原始特权向量9维 |
| proprio_hist | 同一带噪帧源的 [N,30,49] 历史，从旧到新 |

一次控制步只读取一次传感器。clean 触觉只取模和确定性限幅，不经过
Actor 的平滑、延迟、噪声、丢失或二值化。reset 后用新首帧重复填充历史，
局部 reset 不污染其他环境。clean174 保留原字段，不代表完整马尔可夫状态。

`SharpaTeacherEnv` 使用未修改的 UniLab managers，将四组输入直接放入
`NpEnvState.obs`。底层先返回未 reset 的 transition；Sharpa 保存完整
`final_observation` 后才局部 reset。策略输入不写入 `info`。
纯超时使用旧 episode 的下一观测 bootstrap；掉落与超时同时发生时按真实终止处理。

Actor 特权编码器为 `9→256→128→9`，每层 ELU，随后 tanh；与147维基础历史
拼接进入 `512→256→128` ELU 主干。PPO/APPO 使用独立同构 `V(clean174)`。
FlashSAC 保留原生分布式双Q、target Q、温度、奖励归一化与 [-5,5] support；
它是公共 HORA Actor 下的适配版本，Actor 不执行原生单位权重归一化。
日志给出 target support 超界的 bin 比例及边界概率质量，checkpoint 保存参数量。

PPO/APPO 保存高斯采样动作及其采样时 log-prob；环境执行 [-1,1] 限幅。
PPO 同时保存行为均值/标准差，GAE 使用采集策略对应的旧 V/统计。
FlashSAC 保留有界 log-std、重参数化 tanh 及 Jacobian 修正、持久探索噪声。
其 native Actor 更新批次同时计算当前与下一输入，但策略梯度只使用当前部分。

## 数据与归一化

APPO 使用独立 spawn collector、有限队列和三批 staging；读取当时可用的
rollout，由原生 RolloutStagingPool 在环境轴提供批次视图，避免跨轨迹连接。权重和统计一起带版本同步，
实际行为 log-prob 随数据保存。最后不足一个 rollout 时缩短采样；不同长度
的旧 staging 不拼接到最后短批次。

FlashSAC replay 分别保存当前与下一步原始 obs、priv_info、critic，以及
动作、原始奖励、terminated/truncated。读取时重新应用当前统计并计算 latent。
Q 更新不修改 Actor；Actor 更新保留 Q 对动作的梯度而不更新 Q 参数。

Actor 基础历史的经验统计只累计新接收的 o_t，每条 transition 一次，
不额外累计最后一个 o_next，也不累计 epochs、staging 或 replay 复用。
PPO/APPO 的 clean V 使用独立统计；FlashSAC Q 沿用不做经验输入归一化的基线。
评估冻结全部统计。旧的 critic 尾部推断、info 输入、共享价值主干与
APPO 专用 teacher/student 运行器已经删除。

## 保留的优化基线

[冻结 YAML 与版本](issue-2-baseline/) 保留四个旧入口。
[effective-optimizers.json](issue-2-baseline/effective-optimizers.json) 通过旧 Sharpa
790ae32、冻结 YAML 和已安装依赖重新构建初始 learner，记录实际 Adam 参数、
KL 设置及 scheduler 曲线；这不是历史 run 的 optimizer state。

- PPO/APPO：gamma=.99、lr=.001、5 epochs、4 minibatches，保留各自 KL 调度。
- APPO 选择旧普通 APPO 基线：desired KL=.02、staging=3。旧 HORA 的 .04/8
  只作为迁移来源保留，不混入新默认值。
- FlashSAC：gamma=.97、每轮14次 Q/7次 Actor/7次温度更新，
  实际 LR 从3e-4线性衰减到1.5e-4。原未生效 alpha_lr 不启用。
- 本地实现继续调用 RSL-RL PPO、uni_rl APPO/V-trace 与 FlashSAC 原生损失；
  不复制损失公式。FlashSAC 保留 AMP；编译/CUDA graph 路径不在本适配器内启用。

## 预算、评估和蒸馏

默认固定硬件实验为单 learner、全局4096环境、5M新 transition。
各方法保留自己的 rollout/update 频率。最终预算最多向上取整一个向量步，
容差为 N-1；APPO 每个实际向量步更新独立全局计数，接收端检查重复/缺包。
日志分别记录 collected、received、training_samples、各类 optimizer updates、
复用比、策略版本延迟和墙钟时间。联合 PPO/APPO minibatch 计一次数据使用，
SAC 分开的 Actor/Q 使用各计一次。

保存与评估由实际 collected 步数触发；冻结 checkpoint 同时绑定已接收数、
更新计数、策略版本和时间。默认同步评估期间 APPO collector 可继续采样；
评估图使用 checkpoint 计数，不使用评估完成时计数。最终额外保存并评估
`teacher_final.pt`。墙钟包含模型/环境初始化、训练、保存、训练内同步评估，
不包含 CLI 的资产校验；checkpoint 时间不包含该快照后才进行的评估。

评估固定20秒窗口，默认8尺度等权、每尺度3个评估 seed × 10 episode。
清单记录 episode ID、尺度、缓存文件 hash/行号、reset seed 和实际 PD、
质量、摩擦、CoM、重力参数。重新评估必须逐项复现这些值，否则报错。
观测与外力分别使用按 episode/控制步确定的独立 PCG64 流；提前掉落不补跑。

报告原始 return、存活时间、掉落率、有符号旋转角、角度/固定20秒，
以及角度/实际存活时间。角增量取 world-frame 最短四元数差分在任务轴的投影，
在 reset 前累计，未经旋转奖励裁剪。先按预定尺度权重聚合每个训练 seed，
再跨至少3个训练 seed 报均值、样本标准差和 seed 级 bootstrap 95%区间
（10000次，固定seed）；不能把 episode 当作独立训练重复。
训练内评估用 validation，默认选择 final；测试清单应使用不同评估种子，
仅在最终测试时使用，不根据测试成绩选 checkpoint。

所有 teacher 均使用相同 student 流程：冻结 teacher 编码器、基础统计、
主干和动作头，仅训练原有30帧 `ProprioAdaptTConv`；历史统计每新批次更新一次。
每个向量步先做一次 latent MSE 更新，再执行更新后 student 的确定性动作。
默认100M蒸馏 transition、4096环境、lr=3e-4，seed集合[1,2,3]。
checkpoint 保存完整环境/模型/蒸馏配置、历史统计、teacher hash和两阶段成本；
student 推理只需要 obs 与 proprio_hist。

## 使用

```bash
uv run sharpa-train --algo ppo
uv run sharpa-train --algo appo --nodr
uv run sharpa-train --algo flashsac +preset=throughput
uv run sharpa-eval --checkpoint /path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /path/to/student_final.pt
uv run sharpa-compare --output logs/comparison --seeds 1 2 3
uv run sharpa-compare --output logs/smoke --smoke
```

绘图需要 `uv sync --extra mujoco --extra evaluation`。
`--cfg` 打印实际合并配置；`+preset=throughput` 是独立吞吐实验标记，
默认使用4096个环境，可显式调整硬件。这里的环境数始终是全局数；当前入口明确拒绝多 learner
设备配置，避免把未实现的多 rank 统计当作已支持。旧 checkpoint 必须重训，
不做静默结构迁移；训练断点恢复和旧视频/JIT/ONNX 导出入口不在 v2 支持范围。
抓取缓存生成继续保留独立的旧 flat 观测与原 UniLab PPO 工具入口。

验证结果见 [VALIDATION.md](../VALIDATION.md)。短预算 pipeline smoke 只证明
流程与契约可执行，不能用于算法性能或样本效率排名。

本次组件简化、原生接口限制及验证结果见 [训练代码简化结果](issue-2-simplify.md)。
