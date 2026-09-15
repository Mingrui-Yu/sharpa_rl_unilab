# Teacher 配置与运行行为对齐

本次对应 journal 07 的“后续修改计划”，适用于旋转任务 teacher；抓取缓存生成的
配置加载关系不变。公共配置合并为 `conf/common/sharpa_inhand.yaml`。

## 新配置

三个 teacher 均默认使用2048环境、`training.device=cuda:0`、每进程4个 Torch
intra-op 线程和1个 inter-op 线程。APPO 采样设备默认跟随 Learner，可通过
`algo.collector_device` 单独指定。线程配置统一使用 `training.torch_threads.*`；
未设置独立 Collector 的算法不会因此新增进程。相同线程配置不代表总 CPU 用量相同。

APPO/PPO 保留当前501轮，FlashSAC 保留3000轮。三者 `algo.save_interval` 均为50，
0关闭中间保存，正常结束仍保存 `teacher_final.pt`；中间文件为
`teacher_iteration_<已完成轮数>.pt`。每轮记录指标，采样计数保留为真实值。
FlashSAC 的14次 Q更新属于一轮，warmup 不增加更新轮数。

`training.max_transitions` 与 `algo.max_iterations` 恰好启用一个。
`sharpa-compare` 继续使用采样量停止模式，对 FlashSAC 选择同步兼容 runner。
训练不再调度定期或 final 定量评估；独立评估和比较入口的显式评估保留。

## 配置路径迁移

| 旧路径 | 新路径或处理 |
| --- | --- |
| `budget.transitions` | `training.max_transitions`，teacher 默认 null |
| `budget.save_every` | 删除；teacher 使用 `algo.save_interval` 按轮保存 |
| `budget.log_every` | teacher 每轮记录；student 使用 `distillation.log_every`，默认10000 |
| `budget.evaluate_every` | 删除参数与调度 |
| `budget.checkpoint` | `algo.checkpoint`，评估 CLI 的 `--checkpoint` 优先 |
| `budget.async_queue_size` | APPO 专属 `algo.async_queue_size=4`，历史池仍为8 |
| `hardware.num_envs` | `training.num_envs`，`algo.num_envs` 引用它 |
| `hardware.device` | `training.device`，null 自动选择 CUDA、MPS、CPU |
| `hardware.collector_device` | APPO 专属 `algo.collector_device` |
| `hardware.torch_threads` | `training.torch_threads` 结构 |

删除旧公共 YAML、preset 目录和 nodr 文件及快捷入口。运行资源与各项随机化均可
显式覆盖，默认物理任务、噪声、外力与奖励不变。student 环境数仍由
`distillation.num_envs=4096` 控制，停止、保存、日志触发规则保持不变。

## Critic 输入归一化与 checkpoint

新 teacher 均使用 `model.critic_normalization=true`。FlashSAC 的 `CleanQ` 在
原生 Q 之前归一化 clean174，不改变动作、原生 Q 结构、内部归一化或奖励归一化。
当前 Q 和目标 Q 共用 Critic 统计，Actor 的统计独立。Replay 保存原始输入；
只有新接收/提交 transition 的当前观测累计一次统计。下一观测、Replay 重采样和
梯度更新不累计统计；终止观测与 timeout bootstrap 的处理不变。

统计量包含在模型状态中，当前 Q 和目标 Q 的参数软更新不混合统计量。
原生与同步兼容 FlashSAC 均支持保存、加载后验证相同 Q 输出。

兼容范围限于当前加载器已支持的 v2 checkpoint。加载时迁移配置路径，保留
历史环境参数及有效设备/线程配置，不将新默认值强加到旧权重。
旧 `budget.log_every` 迁移到蒸馏配置，包括随后用于蒸馏的 teacher checkpoint。
旧 FlashSAC 的 false 归一化配置仍按未归一化 Q 加载；启用归一化的模型状态
必须包含统计量。此项不新增训练恢复入口，更早的不兼容格式仍需重训。

## 验证范围

配置测试检查公共默认值、插值与覆盖、独立抓取入口；运行测试检查两种停止模式、
按轮保存、每轮日志、final、独立评估与 student 流程。多进程短训练检查 APPO
和原生 FlashSAC 的线程配置；Q 测试检查所有前向路径、统计计数、参数更新隔离、
timeout 语义及 checkpoint 往返。短训练用于检查实现正确性，不用于比较学习效果。

2026-09-15 本地验证：97项测试分批通过（74项非 slow、20项 CPU/MuJoCo slow、
3项 A800 原生 FlashSAC slow）。GPU 测试覆盖普通/延长 warmup、关闭中间保存、
改变采样步数和线程覆盖；最后修正的 final 摘要路径另行重跑通过。
Ruff lint、修改文件的格式检查、Mypy、Pyright 和 diff 空白检查通过。
与修改前合成配置逐项比对，任务、奖励、评估参数及抓取配置保持不变。
