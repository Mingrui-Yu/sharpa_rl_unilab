# Issue 2：训练代码简化结果

依据主工作目录 `dist/journals/02-plan-simplify.md`，2026-09-14 完成。
实现位于 `sharpa_rl_unilab-issue-2` / `issue-2-observation-contract`。
原有未提交日志修改先保存为 `81460f7`，再进行本次简化。

## 基线与兼容性

[基线快照](issue-2-simplify-baseline.json) 保存三种算法完整合并配置、实际初始
optimizer 参数、PPO/APPO adaptive schedule，以及 FlashSAC 原生 scheduler 状态与曲线采样。
依赖为 UniLab/unilab-rl 1.2.0、RSL-RL 5.5.0、Torch 2.8.0+cu128、
NumPy 2.2.6、MuJoCo 3.11.0。重构前后重新构造的完整配置与 optimizer 参数完全一致。
FlashSAC scheduler 补录使用未改动的模型/learner 和相同配置，来源在 JSON 中标明。

| 组件 | 输入与输出 | 必须保持的统计时机 |
| --- | --- | --- |
| 环境包装 | 四组观测、动作 → reset 后观测及完整终止快照 | 同一控制步先保存终止观测，再局部 reset |
| PPO storage | 原始观测、行为动作/密度、旧 V → 原生 PPO minibatch | 旧 V/returns 先计算，当前新样本统计在 update 前累计一次 |
| APPO staging | 原始 rollout、独立末帧、行为版本 → `[T,K*N,...]` 原生 pool 视图 | 每个新包入库时累计一次；复用和 learner 不累计 |
| FlashSAC replay | 当前/下一原始 obs、priv_info、clean 状态 → 原生损失输入 | 新接收时更新观测/奖励统计，抽样时重新编码特权信息 |
| 日志 | 原始 metrics/counters → JSONL、TensorBoard、Rich 面板 | 以 received 为记录间隔与横轴；吞吐取两次记录的真实增量 |
| student | 当前历史及 teacher latent → 一次 MSE 更新和新策略动作 | 每个向量步一次历史统计/优化，基础统计冻结 |

## 已复用及删除

- 日志直接继承 `uni_rl.logging.offpolicy.OffPolicyLogger`，复用终端表格、
  指标别名去重、TensorBoard 写入和关闭。不再本地实现 Rich 表格、数值格式或 scalar 写入循环。
  本地只补充完整 JSONL、transition 标题/ETA、阶段信息及计数映射。
- PPO 沿用原生 `PPO`/`RolloutStorage`；相同 horizon 复用 storage，最后短 rollout
  才按实际长度重建，行为 log-prob 和旧 V 的生成时机保持不变。
- APPO 用原生 `RolloutStagingPool` 代替 deque 和每次 update 的逐字段 `torch.cat`。
  NumPy rollout 只转入 pool 一次，删除接收路径中未使用的重复 device 转换。
  每条轨迹保留独立末帧；不同长度的最后短包替换旧 pool。
  pool 的物理槽顺序可能改变 minibatch 排列，不要求 APPO 位级复现异步基线。
- 公共记录复用 `ExperimentTracker` 所用的 `get_git_info` 和
  `write_run_config_snapshot`，删除本地 Git subprocess 包装。
  新增标准 `run_config.json`，保留已有 `config.yaml`、`run.json` 与加载契约。
- teacher/student 共用标准库 `ExitStack` 管理 env、collector 和 logger，
  删除嵌套 finally，并覆盖以前未保护的 teacher reset 失败路径。

## 保留的本地适配及原因

逐个核查 PPO → APPO → FlashSAC 的原生入口后，保留本地预算调度，没有复制或改名原生 `learn()`。

| 接口 | 当前限制与保留理由 |
| --- | --- |
| UniLab EnvFactory / create_env | 已通过原生注册和配置适配构造环境；Manager 环境对外映射仍只给 obs/critic。`SharpaTeacherEnv` 必须保留四组观测及全部终止快照，不能直接打开底层 autoreset 后丢掉 priv_info/history。 |
| `RslRlPPORuntime` / `RslRlVecEnvWrapper` / RSL runner | runtime 可选择 wrapper/model，但默认 flat wrapper 会拼接非 critic 的不同秩输入；actor 模式又只保留 obs。原生 learn 固定每次 rollout 长度，RSL PPO 超时补偿使用当前 transition 的 V，不读取 wrapper 提供的终止 bootstrap obs。仅配置 class_name 无法保留当前纯超时目标和尾批次。 |
| APPO runtime / runner | 原生 learn 按固定 iteration/rollout 和固定 IPC slot 运行，缺少全局真实采集预算与最后短包的同等调度。原生 process_batch 会在复用数据/末帧上调用 normalization，HORA 仍需将这些调用设为无更新，并在新包入库时更新。保留有限 Queue、包序检查、计数与版本调度。 |
| `SharedWeightSync` | 原生共享缓冲统一转换为 float32；HORA normalizer 的计数为 int64。保留能原样传递 dtype 的 state_dict Queue，同一版本携带权重和统计，避免大预算下损失计数精度。 |
| FlashSAC runner / replay | learner/Actor 注入可以完成组装，但原生 replay transfer 只接受 CUDA/MPS，并不提供当前 CPU 验证路径；固定采样周期也不能直接替代精确尾批次。当前约 40 行原始字段 replay 保持 CPU/CUDA 同一实现，继续复用原生 Q、温度、optimizer、scheduler、损失和 learner checkpoint。没有新增第二套设备分支。 |
| `OnPolicyLogger` | backend 横轴和进度使用 iteration，吞吐按固定 num_envs*num_steps 推算。统一选择支持显式 total_steps 与实测吞吐的 OffPolicyLogger，也用于 PPO/student；替换预算标题和 ETA，绝不把 transitions 当 iteration。关闭原生终端的跨记录平滑，避免累计 counters 显示平均值。 |
| `ExperimentTracker` | 其完整生命周期还管理 W&B 和另一套计时 summary；当前入口已有本地 backend 和包含初始化/评估的时间协议。直接复用其无副作用的记录函数，保留 HORA 契约、依赖版本、评估清单与 teacher/student 来源字段。 |
| student / evaluator | 原生 distillation 学习 teacher 动作，不能替代 latent MSE。保留仅历史编码器更新及更新后动作顺序；固定场景、八尺度配额和跨种子统计也是实验协议，继续本地维护。 |

未修改 UniLab、uni_rl、RSL-RL 或依赖安装目录中的源码；未通过全局 monkey patch 改写依赖。
未改超参数、CLI 默认路径、输入形状、损失公式或 checkpoint 版本。

## 验证

- 基线：41 tests passed。简化后：50 tests passed，包含全部物理集成测试。
- 三算法分别验证 17→24 和 25→32 transitions 的预算取整、尾批次、保存/加载、
  重复固定场景评估、蒸馏和冻结统计；PPO 检查行为密度，APPO 检查同时接收三个包、
  pool 环绕覆盖、旧数据复用、短包替换和异常清理。
- 观测测试覆盖历史时序、局部 reset、clean/带噪分离、完整终止观测及掉落和超时同时发生。
- 日志测试覆盖 Rich、TensorBoard 的真实 step、完整 JSONL、none/no_print、累计计数不平滑，
  以及异常关闭；teacher reset 失败与 learner 失败均检查资源清理。
- `ruff check src tests`、Pyright（0 errors / 0 warnings）、`git diff --check` 通过。
- A800 三算法各执行一次重构前/后的 256 transitions 启动对照。所有运行 collected =
  received = normalization count = 256；训练使用次数分别为 PPO 256、APPO 672、FlashSAC 768。
  PPO 和 FlashSAC 最终 Actor 权重逐张量完全一致，实际 optimizer 参数一致。

| 算法 | 首个记录前时间：前→后 (s) | 后续窗口吞吐：前→后 (transitions/s) |
| --- | --- | --- |
| PPO | 3.340 → 3.359 | 430.0 → 428.0 |
| APPO | 4.600 → 4.742 | 718.4 → 627.2 |
| FlashSAC | 1.157 → 1.317 | 308.1 → 256.0 |

首个记录包括初始化及前 32 transitions；后续窗口排除该部分。
后续窗口不足 1 秒，不能据此认定稳定吞吐或速度改善；本次短对照中 APPO/FlashSAC
更慢，未宣称性能提升，也不用于算法效果排名。完整计数、配置覆盖、scheduler 状态和
原始 metrics 保存在[验证记录](issue-2-simplify-validation.json)。
尚未验证长时间稳定吞吐、恢复训练、多卡或编译；没有宣称这些能力已支持。


### 较长的有界吞吐对照

另以相同配置运行各 4096 transitions；使用基线提交的四个运行文件执行前测，
再逐字节恢复简化后的文件执行后测。跳过前 512 transitions 的初始化/预热窗口，
单独统计余下接收窗口。每次运行最终采集数、接收数、归一化计数均为 4096。

| 算法 | 首个记录时间：前→后 (s) | 预热后吞吐：前→后 (transitions/s) | 预热后测量时长：前→后 (s) |
| --- | --- | --- | --- |
| PPO | 3.300 → 3.546 | 433.6 → 429.7 | 8.27 → 8.34 |
| APPO | 4.733 → 4.784 | 439.5 → 445.0 | 8.15 → 8.05 |
| FLASHSAC | 1.295 → 1.358 | 280.5 → 248.9 | 12.78 → 14.40 |

这仍是单次、小环境数的本地对照，数据包含异步调度和仿真轨迹差异；
不代表默认 4096 环境配置的生产吞吐，也不用于算法效果排名。
原始边界记录、计数和完整配置覆盖保存在同一验证 JSON 的 `longer_comparison` 字段。
