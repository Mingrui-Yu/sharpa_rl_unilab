# APPO exact KL：取消首次调度跳过的单因素实验

日期：2026-09-16。生产代码 revision：`43317b1602b976b485b40191ef1d15ae3924d1e8`；仅通过实验入口临时改变调度，生产默认配置未修改。

**结果：相对当前默认基线，速度提高 3.31%，高度退出从 8/240 降至 5/240；没有复现 N-logε 的完整结果。** 首次升学习率对该 seed 的速度和存活有改善作用，但旧 KL 还会通过后续 minibatch 的阈值穿越改变调度。

## 实验定义与核验

使用刚完成的当前默认 APPO 为基线，保留 exact KL，取消参考策略同步后的首次调度跳过。因此首个 KL=0 也按低于下阈值处理，执行 `LR=min(1e-2, LR*1.1)`。其余阈值、倍率、loss、熵、V-trace、优化器和环境设置不变。

seed=1、collector seed=2、2048 环境、501 iterations、GPU 0（A800）、CPU 0–127、每进程 4 个 Torch 线程。log-std、初值 1、无 std 边界、tanh、保存间隔 50、TensorBoard 均保持当前默认值。最终评估使用原 240 场景、确定性动作和 20 秒固定窗口；五个中间快照使用冻结的 24 场景。

2 轮控制短训练的 loss、KL、学习率、采样计数和 lag 与基线前两轮一致。控制组与实验组初始 actor/critic/target actor 权重及 CPU/CUDA RNG 状态相同；完整实验初始化也与控制组相同。最终 resolved config 与基线的差异仅为输出目录和实验元数据；生产源码哈希与基线相同。

实验增加了逐 minibatch 的被动 KL/LR 记录，每轮批量落盘，不改变 loss 或消耗训练 RNG；异步运行时开销仍可能影响采样时序，需结合实际计数和 lag 解读。

## 最终 240 场景结果

| 组别 | 速度 rad/s | 高度退出 /240 | 存活 s | 关节二阶差分 RMS rad | 平均 std | 饱和率 % | 目标限位 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前默认基线（exact＋首次跳过） | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 6.4318 |
| 实验组（exact＋首次零 KL 升 LR） | 0.616798 | 5 | 19.6521 | 0.00683429 | 0.469027 | 0.0737 | 4.8605 |
| N-logε（旧对数项） | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 4.4267 |
| R（原 G 重建） | 0.628088 | 0 | 20.0000 | 0.00722120 | 0.459803 | 0.0241 | 3.2487 |

实验组减参考组：

| 参考 | 速度差 rad/s | 速度变化 % | 退出数差 | RMS 差 rad |
|---|---:|---:|---:|---:|
| 当前默认基线（exact＋首次跳过） | +0.019789 | +3.31 | -3 | +0.00012080 |
| N-logε（旧对数项） | -0.013011 | -2.07 | -2 | -0.00032719 |
| R（原 G 重建） | -0.011290 | -1.80 | +5 | -0.00038692 |

这不是所有指标同时改善：平均评估回报从 47.348912 降至 46.045352，关节二阶差分 RMS 增加 1.80%。退出场景也有替换：修复基线的 6 个失败场景，同时新增 3 个失败场景，另有 2 个共同失败。

## 调度机制

全程首 minibatch 的 KL=0 共 501 次，实际首步学习率增加 501 次。后续 minibatch 的零 KL 共 0 次、负 KL 共 0 次。共记录 10,020 次调度；学习率增加/降低/不变分别为 2494/2499/5027。

与 N-logε 的学习率首次分歧发生在第 **6 轮、第 17 个 minibatch**。之前的逐 minibatch exact KL 与 LR 全部一致：True；分歧点 exact KL 也逐位一致：True。

| 分歧点指标 | exact＋首步升 LR | N-logε |
|---|---:|---:|
| exact KL | 0.047932140529 | 0.047932140529 |
| 实际用于调度的 KL | 0.047932140529 | 0.048153452575 |
| 调度前 LR | 0.006115909045 | 0.006115909045 |
| 调度后 LR | 0.006115909045 | 0.005559917313 |

上阈值为 `0.04*1.2=0.048`。此处 exact KL 尚未越界，旧对数项的正偏置使 N-logε 越界并降速。这是在恢复首次升 LR 后，仍然存在的后续阈值效应；不能把旧 KL 的全部作用归结为每轮首次多升一次 LR。

六个对应 checkpoint 与 N-logε 的 actor/critic/target actor 权重全部逐位一致：False；最终评估轨迹全部逐位一致：False。

## 采样量与逐场景结果

| 组别 | received | collected | optimizer updates | training samples | 训练分钟 |
|---|---:|---:|---:|---:|---:|
| 当前默认基线（exact＋首次跳过） | 8,208,384 | 8,224,768 | 10,020 | 326,041,600 | 7.64 |
| 实验组（exact＋首次零 KL 升 LR） | 8,208,384 | 8,224,768 | 10,020 | 326,041,600 | 7.92 |
| N-logε（旧对数项） | 8,208,384 | 8,224,768 | 10,020 | 326,041,600 | 7.62 |
| R（原 G 重建） | 8,208,384 | 8,224,768 | 10,020 | 326,041,600 | 7.51 |

相对默认基线的采样计数/lag 首次分歧轮（null 表示 501 轮都未出现）：

```json
{
  "received": null,
  "training_samples": null,
  "policy_lag_mean": null,
  "policy_lag_max": null,
  "rollouts_read": null
}
```

| 参考 | 共同成功 | 仅参考退出 | 仅实验组退出 | 共同退出 | 共同窗口 RMS 差 rad |
|---|---:|---:|---:|---:|---:|
| 当前默认基线（exact＋首次跳过） | 229 | 6 | 3 | 2 | +0.00013249 |
| N-logε（旧对数项） | 231 | 4 | 2 | 3 | -0.00031437 |
| R（原 G 重建） | 235 | 0 | 5 | 0 | -0.00040249 |

实验组失败场景：0.9/10001/8, 1.4/10003/2, 1.4/10003/5, 1.5/10001/3, 1.5/10003/0。

共同窗口截止双方较早退出，包含终止步，不补零；共同成功子集存在选择偏差。RMS 是关节位置二阶差分，不应脱离速度和存活单独评价。

## 结论范围与产物

本次检验的是首次升学习率这一具体干预；性能变化见结果表，不能由调度分歧次数计算各机制的性能贡献。只训练 seed=1，240 场景不提供跨训练 seed 的泛化证据。没有据此修改生产默认配置，也没有追加其他训练变量或挑选最佳 checkpoint。

- [最终 checkpoint](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/train/teacher_final.pt)
- [完整评估](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/evaluation/current_501.json)
- [训练与中间评估曲线](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/comparison_curves.png)
- [完整分析](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/comparison.json) / [CSV](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/comparison.csv)
- [逐 minibatch 调度记录](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/train/schedule.jsonl)
- [执行清单](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/manifest.json)
- [实验训练入口](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/experiments/appo_g_ablation/train_zero_kl.py)
- [完整执行驱动](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/run.py) / [分析脚本](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_exact_zero_increase_20260916/compare.py)
