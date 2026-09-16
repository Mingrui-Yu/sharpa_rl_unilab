# 当前 APPO 默认配置训练与历史结果对比

日期：2026-09-16。训练代码：`30889393b2a7c726b917c0825f489c944e397643`。

**结果：当前默认配置与历史 N 的最终评估结果一致。**

本次从头训练 seed=1、collector seed=2、2048 环境、501 iterations；最终 checkpoint 在原有 240 场景上做确定性评估，每场最多 20 秒。训练使用 GPU 0（A800）、CPU 0–127，每进程 4 个 Torch 线程。评估在训练结束后离线使用 CPU。

## 配置与比较边界

当前默认：`tanh`、`state_independent`、`std_parameterization=log`、初始 std=1、无 std 边界；`kl_mode=exact`。target 完整同步后的首次调度保持原学习率，后续 `0 <= KL < lower_threshold` 时升学习率。保持默认保存间隔 50 和 TensorBoard 日志，仅覆盖输出目录与 `training.no_play=true`。

历史消融使用保存间隔 100、`no_print` 日志器和实验入口的 RNG 重置；本次直接使用生产 CLI。环境、奖励和评估协议一致，源码哈希核验见 audit.json。此次是当前默认配置的整体复测，不是将所有其他运行条件冻结的 KL 单因素实验。

## 最终 240 场景结果

| 组别 | 固定窗口速度 rad/s | 高度退出 /240 | 平均存活 s | 关节二阶差分 RMS rad | 平均 std | 动作饱和率 % | 目标限位比例 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| R（原 G 重建） | 0.628088 | 0 | 20.0000 | 0.00722120 | 0.459803 | 0.0241 | 3.2487 |
| N（旧 exact 调度） | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 6.4318 |
| N-logε（旧对数项） | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 4.4267 |
| 当前 APPO 默认 | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 6.4318 |

当前减去历史组：

| 参考 | 速度差 rad/s | 速度变化 % | 退出数差 | RMS 差 rad |
|---|---:|---:|---:|---:|
| R（原 G 重建） | -0.031079 | -4.95 | +8 | -0.00050771 |
| N（旧 exact 调度） | +0.000000 | +0.00 | +0 | +0.00000000 |
| N-logε（旧对数项） | -0.032800 | -5.21 | +1 | -0.00044798 |

RMS 为关节位置二阶差分，不能独立解释为控制质量改善；需同时看速度和存活。本次只做一个训练 seed，240 个场景不代表 240 次训练重复。

## 训练与一致性核验

当前 received=8,208,384、collected=8,224,768、optimizer updates=10,020、training samples=326,041,600。训练记录耗时 7.64 分钟。

100/200/300/400/500/501 六份 checkpoint 的 actor、critic、target actor 权重与 N 全部逐位一致：**True**。最终 240 场景的关节、动作、std 和目标位置轨迹全部逐位一致：**True**。

与 N 首次出现差异的训练 iteration（null 表示全部 501 轮未出现）：

```json
{
  "kl": null,
  "optim/learning_rate": null,
  "surrogate_loss": null,
  "value_loss": null,
  "entropy": null,
  "grad/global_norm": null,
  "received": null,
  "training_samples": null,
  "policy_lag_mean": null,
  "policy_lag_max": null,
  "rollouts_read": null
}
```

历史 N 的首 minibatch KL=0 共 501 次；后续 minibatch 的零 KL 共 0 次，负 KL 共 0 次，最小后续 KL=0.000153645917。因此在其记录的输入上，新调度的首次跳过与旧调度的零值不调整一致，后续零值增大分支未被触发。

本次结果支持：调度语义已明确，但该 seed 下没有改变观测到的训练或评估结果，也没有恢复 N-logε 的速度或 R 的零退出表现。

## 逐场景比较

| 参考 | 共同成功 | 仅参考退出 | 仅当前退出 | 共同退出 | 共同存活窗口 RMS 差 rad |
|---|---:|---:|---:|---:|---:|
| R（原 G 重建） | 232 | 0 | 8 | 0 | -0.00052788 |
| N（旧 exact 调度） | 232 | 0 | 0 | 8 | +0.00000000 |
| N-logε（旧对数项） | 228 | 4 | 5 | 3 | -0.00042446 |

共同窗口截止双方较早退出，包含终止步；不对退出后补零。共同成功子集有选择偏差，详细数据在 comparison.json。

当前失败场景：1.4/10001/8, 1.4/10002/3, 1.4/10002/5, 1.4/10002/6, 1.4/10003/5, 1.5/10003/0, 1.5/10003/8, 1.5/10003/9。

## 产物

- [最终 checkpoint](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/train/teacher_final.pt)
- [完整 240 场景评估](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/evaluation/current_501.json)
- [训练与固定 24 场景中间评估曲线](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/comparison_curves.png)
- [结构化比较](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/comparison.json) / [CSV](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/comparison.csv)
- [运行清单与配置差异](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/manifest.json) / [源码核验](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/audit.json)
- [复现驱动](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/run.py) / [分析脚本](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_default_exact_20260916/compare.py)

中间评估只用于观察训练过程，最终结果始终使用 iteration 501，没有挑选最优 checkpoint。
