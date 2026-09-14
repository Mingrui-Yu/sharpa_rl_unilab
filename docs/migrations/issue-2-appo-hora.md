# APPO HORA 参数与归一化约定

APPO teacher 的默认采样、优化和更新预算对齐 `main@790ae32` 的 HORA 配置。
基线标识为 `appo-hora-790ae32`。保留当前分支的显式 9 维特权输入、独立 clean
Critic、观测/reset 修正、行为 log-prob 和 checkpoint 格式，不代表完整复现 main。

- 2048 个环境，每批 8 步；历史池 8 批，填满后每轮训练 131072 个样本。
- 每轮 5 epochs × 4 minibatches，即 20 次 optimizer step。
- Adam 初始学习率 0.001，desired KL 0.04；保留原生 KL 调度阈值、上下限和执行时机。
- 默认完成 305 轮更新，每 51 轮保存 `teacher_iteration_N.pt`，结束保存 `teacher_final.pt`。
- 不默认限制新 transition 数。采样预算实验必须同时设置
  `algo.max_iterations=null budget.transitions=N`；两种预算同时非空会报错。
- APPO 保存始终使用 `algo.save_interval`（0 表示关闭中间保存），不接受
  `budget.save_every`。定量评估默认关闭；训练后录像沿用现有配置。
- 默认自动选择 CUDA、MPS、CPU，Collector 跟随 Learner，可显式覆盖；
  Torch 线程数为 null 时不覆盖进程设置。Collector seed 默认为 Learner seed + 1。
  checkpoint 和运行配置记录生效设备和 seed。

待接收队列容量 `budget.async_queue_size=4` 与历史池
`algo.staging_pool_size=8` 独立。main 的 4 槽传输 ring 会覆盖旧数据；本分支保留
有界队列背压：最多 4 批待接收，加上 Collector 正在采集或等待提交的 1 批。
每轮最多读取 4 批已就绪数据，队列数据不丢弃。结束时停止 Collector 并排空队列，
然后读取最终 collected；停止期间完成但未训练的数据不会计入 received。

Actor 的带噪观测和 Critic 的 clean 观测分别使用现有累计均值/方差统计。
Learner 在接收阶段仅累计每批新数据的 o_t，不累计 o_next，不因历史淘汰撤销统计。
统计在 V-trace 和全部优化 epochs 中保持固定，target actor buffers 在优化前同步。
原生 `process_batch` 的 normalization 回调由适配器保持为空，避免重复累计历史或末帧。

策略版本由 `(version, model_state_dict)` 整体传递，state_dict 已包含全部 RMS buffers。
发布时复制 tensor，队列满时移除旧快照并重试提交最新完整快照；Collector 仅在 rollout
边界安装最新版本。行为密度始终使用动作采样时保存的原始 log-prob，动作保留原始
高斯样本。checkpoint 的 `actor` 与 `counters.policy_version` 保存同一策略版本；
加载、评估和录像均冻结统计。

APPO 每轮记录进度和 ETA；TensorBoard 横轴仍为真实 received transitions。
同时记录 collected、received、training_samples、optimizer_updates、历史池占用和
策略版本滞后。`sharpa-compare` 显式选择各算法的采样预算；student 优化参数和预算不变。

回归验证覆盖配置对比、互斥预算、设备与线程选择、真实多进程策略队列替换、rollout
边界版本切换、原始行为密度、累计统计分批等价、每次 optimizer step 的统计冻结、
10 轮 CPU/MuJoCo 短训练、周期保存、8 批历史池、最终计数、资源释放和 checkpoint
动作一致性。短训练仅验证运行行为，不用于比较训练效果。
