# 架构说明

本文面向代码修改；任务背景见[手内旋转](task.md)，操作步骤见[训练指南](training.md)。

## 职责边界

Sharpa 是 UniLab 的任务包。UniLab 管理仿真、场景和 manager 生命周期；
RSL-RL、uni_rl 提供 PPO、APPO 和 FlashSAC 算法组件。
Sharpa 定义任务、HORA 模型、观测适配及训练、蒸馏和评估流程。

任务通过 `ManagerBasedRlEnvCfg` 注册，由 UniLab 工厂创建环境。
YAML 声明场景实体和 manager terms；Python term 通过 Entity API 访问仿真状态。
抓取生成使用独立配置和原生 PPO 入口，旋转训练使用公共 HORA 配置。

## 代码导航

以下路径相对于 [`src/sharpa_rl_unilab/`](../../src/sharpa_rl_unilab)：

| 修改内容 | 位置 |
| --- | --- |
| CLI 参数与配置合并 | `cli.py` |
| 公共物理任务 | `conf/common/sharpa_task.yaml` |
| HORA 观测、模型和运行参数 | `conf/common/sharpa_inhand.yaml` |
| 算法参数、抓取配置 | `conf/{ppo,appo,flashsac}/` |
| 任务注册、固定尺度与时间步 | `tasks/sharpa_inhand/{config,rotation_registry,grasp_registry}.py` |
| 动作、观测、奖励、重置和随机化 | `tasks/sharpa_inhand/terms/` |
| 固定观测协议与构建期校验 | `tasks/sharpa_inhand/protocol.py` |
| 四组观测与终止快照 | `tasks/sharpa_inhand/teacher_env.py` |
| 公共 HORA 模型与分布 | `algos/hora/{models,distribution,legacy}.py` |
| 算法适配与 KL 调度 | `algos/hora/{on_policy,flashsac,kl_schedule}.py` |
| Teacher 分发与 PPO/APPO 训练 | `training/{teacher_runtime,ppo_runtime,appo_runtime}.py` |
| 公共采样与 APPO 进程生命周期 | `training/{rollouts,appo_collector}.py` |
| 模型构建、输入转换与推理 | `training/policy.py` |
| 公共 checkpoint 格式与加载 | `training/checkpoints.py` |
| 原生 FlashSAC 运行器适配 | `training/flashsac_runtime.py` |
| Student 蒸馏 | `training/student_runtime.py` |
| 设备、线程与 checkpoint 配置迁移 | `training/configuration.py` |
| 日志、运行元数据与配置快照 | `training/logging.py` |
| 固定场景评估、多算法编排 | `training/evaluation.py`、`tools/compare_rl_algo.py` |
| 视频回放与渲染检查 | `training/{playback,rendering}.py` |
| 模型资产、缓存与抓取生成脚本 | `assets/`、`tools/sharpa_collect_grasps.sh` |

## 观测与数据流

`SharpaTeacherEnv` 将 manager 观测转换为四组显式输入。表中 N 为环境数：

| 输入 | 形状 | 内容与使用方 |
| --- | --- | --- |
| `obs` | `[N,147]` | 带噪基础帧 49 × 3；teacher/student Actor 共用 |
| `critic` | `[N,174]` | 每帧 clean 基础 49 + 特权 9，堆叠 3 帧；仅 Critic 使用 |
| `priv_info` | `[N,9]` | 当前原始特权信息；teacher 编码器使用 |
| `proprio_hist` | `[N,30,49]` | 与 `obs` 同源的带噪历史，从旧到新；student 编码器使用 |

Teacher 将特权信息编码后与基础观测拼接，输出动作。
Student 用历史编码器替代特权编码器；其余策略部分继承 teacher。
PPO/APPO 使用独立的 V 网络，FlashSAC 使用分布式双 Q 和目标 Q。
v2 维度固定；不兼容的触觉、特权信息和历史设置在构建阶段拒绝，错误会指出字段及协议要求。

一次控制步只读取一次传感器。Clean 触觉保留取模和确定性限幅，
不经过 Actor 的平滑、延迟和噪声；clean 观测不代表完整的马尔可夫状态。

## 三种算法的接入

| 算法 | 复用组件 | Sharpa 负责的适配 |
| --- | --- | --- |
| PPO | RSL-RL `PPO`、`RolloutStorage` | 四组输入、终止 bootstrap、短 rollout 和预算调度 |
| APPO | uni_rl APPO/V-trace、`RolloutStagingPool` | 独立 collector、有界队列、策略版本、历史复用及预算调度 |
| FlashSAC | `DoubleBufferOffPolicyRunner`、原生 replay 与 learner 损失 | 观测传输、新数据统计、公共 checkpoint 和计数 |

PPO/APPO 保留本地采样调度，以处理终止观测和采样预算的末尾短批次。
FlashSAC 继承原生 `learn()`，由其管理 collector、推理、replay、预取和优化循环；
warmup 不计入更新轮数。设备和预算限制见[训练指南](training.md#2-训练-teacher)。

FlashSAC 传输组为 `obs=[actor147,current_priv9]` 和 `critic174`，reset 与终止快照布局一致。
Replay 保留原始观测和动作，抽样后再应用当前统计与特权编码。
采样推理与学习共用 Actor，`CleanQ` 在原生 Q 前应用独立的 Critic 输入归一化。

`HoraActor.policy()` 返回无调用缓存的 `PolicyDistribution`，采样结果同时携带原始样本和
执行动作。`TeacherActor`、`TeacherFlashActor` 是共用此实现的上游接口薄适配层。
APPO 只覆盖策略/价值前向及分布钩子，保留上游损失实现。
`training/exploration.py` 为 runner 在 Learner 设备上维护采集噪声，
不改变原生调度，也不通过 IPC 传输噪声。

训练日志复用 `OffPolicyLogger`，Sharpa 补充 JSONL、阶段和实际采样计数。
`sharpa-compare` 只编排各阶段命令，评估算法及统计函数位于 `training/evaluation.py`。

## 必须保持的约定

- **终止与重置**：底层关闭 autoreset；先保存四组终止观测，再局部 reset。
  新 episode 用首帧填充历史，不影响其他环境。纯 timeout 使用旧 episode 的下一观测
  bootstrap；掉落与 timeout 同时发生时按真实终止处理。
- **归一化**：每条新接收 transition 的当前观测只累计一次。
  下一观测、重复 epoch、staging 和 replay 重采样不累计；评估冻结全部统计。
  Actor 与 Critic 统计独立；FlashSAC 当前 Q 与目标 Q 共用统计，目标参数软更新不混合统计。
- **行为策略**：PPO/APPO 保存原始高斯动作及采样时的 log-prob，环境执行 clip/tanh 映射后的动作。
  APPO 的版本号、权重和统计整体同步，collector 只在 rollout 边界安装新版本。
  各 rollout 保留独立末帧，避免跨轨迹拼接。
- **梯度与蒸馏**：Q 更新不修改 Actor；Actor 更新保留 Q 对动作的梯度，不更新 Q 参数。
  Student 仅训练历史编码器，基础统计冻结、历史统计按新批次更新；每步先做 latent MSE
  更新，再用更新后的 student 执行动作。
- **实际计数**：区分 collected、received 和 training_samples。
  FlashSAC 结束时可能已有提交但尚未累计统计的尾部数据，collected 与 received 可以不同。
- **固定尺度**：物体尺度使用独立 MJCF，不能在运行中修改 geom 大小。
  每个 variant 的 geom 名唯一，自由物体保留 `simple="false"`，避免 reset 修改质量或重心时
  触发 MuJoCo `sameframe` 错误。Task term 使用 Entity API，不访问 backend 内部对象。

Checkpoint 使用范围见[兼容性说明](training.md#5-常用设置与兼容性)，
历史测试与实验结果见[验证记录](../VALIDATION.md)。

## 资产与生成的抓取缓存

`ensure_assets()` 只修复 manifest 管理的内置资产副本。Recorder 默认输出到可写
缓存的 `generated/` 子目录，读取时优先选择这些用户数据；仍支持绝对路径前缀。
缓存内容在加载时完整校验一次，reset 采样时只检查 variant 索引。

FlashSAC checkpoint 的顶层 actor/critic 用于推理，归档的 learner 状态引用同一份
张量存储，避免重复保存模型。现有 v2 checkpoint 继续支持加载，CLI 仍不提供训练恢复。
