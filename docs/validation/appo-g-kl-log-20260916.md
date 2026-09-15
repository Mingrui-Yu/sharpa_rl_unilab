## 11. 用户指定后续实验：N 设置下仅恢复旧 KL 对数项（2026-09-16，已完成）

**结论：在 N 设置下，仅恢复旧 KL 对数项就重现了完整旧 KL 参考组 C 的结果。它恢复了旋转速度，但未恢复原 G 的零退出表现。**

- **N→N-logε：** 固定窗口速度从 0.597009 提升到 0.629809 rad/s（+0.032800，约 +5.49%）；高度退出从 8/240 变为 7/240，平均存活从 19.4292 变为 19.4583 秒；q 二阶差分 RMS 从 0.00671349 增至 0.00716147 rad（约 +6.67%）。速度与运动幅度都增加，不能将其概括为全面改善。失败场景也发生替换：5 个原 N 失败场景被修复，同时出现 4 个 N 成功而新组失败的场景，另有 3 个共同失败。
- **旧对数项足以产生本次 KL 效应：** 新组和 C 的 100/200/300/400/500/501 六份 actor、critic、target actor 权重、全程 std/LR/packet/staging 记录，以及最终 240 场景评估轨迹逐位一致。C 使用完整旧 KL，但其宽 std 边界全程不激活。因而在此次 seed 下，完整旧 KL 其余平方项的运算顺序没有产生额外可观测训练差异。
- **机制仍然是学习率调度：** 相同数据和固定 LR 时，干预不改变 loss、梯度或 Adam 更新；完整训练与 N 从第 1 轮 LR/std 开始分化，packet/staging 轨迹始终一致。在新组自身数据上，带偏置对数项与精确 KL 的影子分支有 611 次分歧，其中 501 次是每轮首 minibatch 的 KL=0→正数、触发 LR×1.1，另外 110 次发生在其他 minibatch。新组与完整旧 KL 的影子分支分歧为 0。
- **未解决的部分：** 新组速度已接近原 G 的 R（0.628088 rad/s），但 R 是 0/240 退出，新组仍为 7/240。恢复这一项不足以消除 N 与原 G 的全部差异。此次干预同时恢复了旧对数项的数学正偏置和运算舍入顺序，未将二者再拆分；也未单独检验“只对 KL=0 改调度分支”。

这些结果支持旧对数项通过 LR 调度影响该 seed 的性能，不支持直接把有偏 KL 作为生产修复。若继续改进，宜保持 KL 的数学定义正确，并另做“零 KL 调度语义”单因素实验；本次没有自动追加该实验。240 个评估场景不提供训练 seed 泛化证据。


### 11.1 单因素定义与执行

按用户最后明确选择，保留 N 的 log-std、无 std 边界、公共 actor/collector/learner、tanh、熵、V-trace、优化器及全部环境设置，仅把精确 KL 的 `log(std) - log(old_std)` 替换为 `log(std / old_std + 1e-5)`。其余平方项继续使用新公式的运算顺序：

```python
KL = (log(std / old_std + 1e-5)
      + 0.5 * ((old_std / std)**2 + ((old_mean - mean) / std)**2 - 1)).sum(-1).mean()
```

KL 仅用于自适应学习率及记录，不加入优化 loss。忽略舍入时，此干预相对精确 KL 增加 `sum(log(1 + 1e-5 * old_std / std))`，不是固定常数，也不是只修改 KL=0 分支。它同时恢复了原对数项的比值/取对数运算顺序；没有恢复完整旧 KL 的其余运算顺序。

原生 N 是本 worktree 已完成的同 seed 基线。新组从相同初始化重新训练 501 iterations，seed=1、collector seed=2、2048 环境、GPU 0、CPU 0–127、各进程 4 个 Torch 线程；FP32、compile=false，其余配置经完整对照一致。新旧依赖和生产源码未修改。最终使用同一 240 场景、20 秒固定窗口评估；五个中间检查点使用冻结的 24 场景，未选最佳 checkpoint。

这是一项用户在原六＋四组实验完成后新指定的独立后续实验。此前固定常数方案已随用户澄清停止：只完成 53 轮、未评估，保留为 `N_kl_bias_seed1/` 并标记 `aborted_by_user_steering`，不计入结果。用户指定的 N-logε 从头训练，没有从该中止运行续训。

### 11.2 最终结果

| 组 | 速度 rad/s | 高度退出 /240 | 存活 s | q 二阶差分 RMS rad | 平均 std | 饱和率 % | 目标限位 % |
|---|---|---|---|---|---|---|---|
| N（精确 KL） | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 6.4318 |
| R（原 G 重建） | 0.628088 | 0 | 20.0000 | 0.00722120 | 0.459803 | 0.0241 | 3.2487 |
| C（完整旧 KL＋宽边界） | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 4.4267 |
| N-logε（仅恢复旧对数项） | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 4.4267 |

| N-logε 减参考组 | 速度差 rad/s | 退出差 | q RMS 差 rad |
|---|---|---|---|
| N | +0.032800 | -1 | +0.00044798 |
| R | +0.001721 | +7 | -0.00005973 |
| C | +0.000000 | +0 | +0.00000000 |

C 的宽边界在原实验全程未触发；它作为完整旧 KL 的补充参考，不能省略这个配置差异。N-logε 与原生 N 才是本次严格的单因素对照。q RMS 为关节位置二阶差分，不直接代表高频振动或控制品质，需结合速度和存活解读。

### 11.3 机制与一致性核验

新组 received=8,208,384、collected=8,224,768、optimizer updates=10,020、training samples=326,041,600，纯训练 7.62 分钟。初始化 actor/critic、learner/collector RNG、除实验元数据和输出路径外的完整配置、六份 checkpoint 及 staging/样本计数均核验通过。

在新组自身数据上，带偏置与精确 KL 的影子调度分歧共 611 次，其中首 minibatch 占 501 次；与完整旧 KL 的影子调度分歧为 0 次。首 minibatch 精确 KL 全为 0，干预 KL 范围为 0.000220297617489–0.000220297661144。首轮实际 LR 从 0.001 升为 0.0011；原生 N 首 minibatch 保持 0.001。

影子调度比较在同一组自身策略上计算，不能当作另一条闭环训练轨迹。固定数据审计覆盖初始与训练后状态：干预前后除 KL 外的 loss 输出、梯度及固定 LR 的 Adam 更新逐位相同；零偏置 hook 的两轮短跑与原生 N 权重一致。

| 参考组 | std 首次差异轮 | LR 首次差异轮 | packet 首次差异轮 | staging 首次差异轮 |
|---|---|---|---|---|
| N | 1 | 1 | None | None |
| R | 1 | 4 | None | None |
| C | None | None | None | None |

`None` 表示全程未分化。最终 state dict 相等性、原始调度 KL 和每 minibatch LR 记录见分析 JSON；R 的 std 参数化及 state dict 名称不同，不能把其字典不相等单独作为机制证据。

| N-logε 减参考组 | 共同成功 | 仅参考组退出 | 仅新组退出 | 共同退出 | 共同窗口 q RMS 差 | 共同成功子集 q RMS 差 |
|---|---|---|---|---|---|---|
| N | 228 | 5 | 4 | 3 | +0.00042446 | +0.00043644 |
| R | 233 | 0 | 7 | 0 | -0.00013174 | -0.00015136 |
| C | 233 | 0 | 0 | 7 | +0.00000000 | +0.00000000 |

共同窗口包含终止步，截止双方较早退出，不做退出后零填充；共同成功子集有选择偏差。新组失败场景：0.8/10002/8, 0.9/10001/8, 1.4/10001/8, 1.5/10001/3, 1.5/10003/0, 1.5/10003/1, 1.5/10003/8。

### 11.4 产物

- [训练与中间评估曲线](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/kl_log_curves.png)。
- [完整分析](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/kl_log_analysis.json)、[执行清单](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/kl_log_manifest.json)、[逐场景最终评估](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/evaluation/N_kl_log_epsilon_501.json)。
- [每 minibatch 调度对照](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/audit/kl_log_schedule.json)、[固定数据审计](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/audit/kl_bias_fixed_data.json)。

结论只针对训练 seed=1。未运行额外训练 seed、未改动生产调度器，也未据此推荐恢复有偏 KL。
