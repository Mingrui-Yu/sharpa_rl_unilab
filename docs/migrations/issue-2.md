# Issue 2：观测契约实施记录

依据：2026-09-14 的 `dist/journals/01-analysis-refactor.md` 和
[issue 2](https://github.com/unilabsim/sharpa_rl_unilab/issues/2)。

本提交是第一批任务侧修改，**不代表 issue 2 已完成，也不是统一 HORA
teacher 的可训练版本**。三个算法的默认入口、模型和运行器尚未合并。

## 已实现

- `SharpaRotationObservation` 每个控制步只采集一次状态和触觉，缓存带噪
  Actor 帧、clean 帧和当前特权向量。重复调用不再重采关节噪声或推进滤波。
- `SharpaCriticObservation` 引用 Actor 的帧源，读取真实关节位置、当前
  接触力及特权信息。触觉仅经过向量取模和确定性限幅，绕过 Actor 的
  平滑、延迟、二值化和丢失路径。当前 HORA APPO/FlashSAC 配置均已接入，
  critic 仍为 `[基础49, 特权9] × 3 = 174` 维。
- 新增 `SharpaPrivilegedObservation` 与 `SharpaProprioObservation` 观测项，
  分别提供显式当前特权信息和同一份带噪基础帧。后者使用 ObservationManager
  的 30 帧非展平历史，产生 `[N, 30, 49]`；与 Actor 的三帧历史逐值一致。
- reset 只标记需要刷新的行，待物理状态写回及动作目标 reset 完成后采集
  新 episode 首帧。ObservationManager 对 reset 行重复填充历史；其他行不
  重新采样噪声或修改历史。reset 后的速度估计清零。

观测项只通过返回值传递数据，不向 `info` 写入策略输入。特权/长历史观测项
目前由契约测试组合，尚未加入公开任务入口：当前 UniLab 的环境映射只导出
`obs/critic`，额外组不会自动进入 `NpEnvState.obs` 或终止观测。

## 迁移基线

[`issue-2-baseline/`](issue-2-baseline/) 保存了修改前 PPO、APPO、HORA APPO、
FlashSAC 的完整合并默认配置及源码/依赖版本。它们是默认 YAML 快照，不是
历史 run 的 CLI 覆盖或实际 optimizer state。历史成绩保留在
[`VALIDATION.md`](../VALIDATION.md)，已移除未经对齐的优劣/样本效率结论。

本提交未修改任何算法优化参数。后续入口合并前仍须捕获实际 optimizer 与
scheduler 参数；特别保留 PPO/APPO gamma=0.99、FlashSAC gamma=0.97，
FlashSAC 实际 peak learning rate=3e-4 的来源，以及普通 APPO/HORA APPO
不同的 KL 基线。不能将 YAML 中未生效字段当作已启用参数。

clean 触觉和 reset 首帧的语义已经改变；现有模型仍是旧结构。此阶段不能
声称旧 checkpoint 已迁移，也不能将新旧观测协议下的成绩合并比较。

## 验证

- 带字段/时间标记的测试验证 147/174/9 维布局、30 帧历史、局部 reset、
  Actor/Critic 调用顺序，以及每步只读取一次触觉。
- 固定物理读数，改变 Actor 噪声、延迟、平滑及二值化/丢失设置，clean
  critic 保持一致；接触力在当前帧响应，向量取模与限幅单独验证。
- PPO、APPO、HORA APPO、FlashSAC 各用 8 个 MuJoCo 环境验证真实超时
  自动 reset：终止观测等于 reset 前快照，新 episode 历史重复填充。
- 未执行 RL 训练、模型保存加载、蒸馏或定量 evaluator；这些需要下一阶段
  的模型与运行器契约。本提交不验证 learner 的超时 bootstrap。
- 本地结果：`63 passed`；全量 Ruff lint、mypy、pyright 通过。修改文件的
  Ruff 格式检查通过；全量格式检查仍报告 12 个未修改文件的既有格式问题。
- 四份基线与当前配置逐项对比，`algo`、`training`、`reward` 保持一致。

测试使用原虚拟环境解释器及显式 `PYTHONPATH` 指向本 worktree 的 `src`
和 UniLab 的 `src`，避免原虚拟环境内已失效的 editable 路径。

## 下一批必须联动的修改

1. 在 UniLab 源码中扩展显式观测组映射及形状契约，使 `priv_info` 和
   `proprio_hist` 随 `NpEnvState.obs`、reset 和 `final_observation` 传输；
   不使用 `info` 补传终止帧。随后在此仓库接入任务配置及 HORA 适配器，
   同时删除观测拆分、APPO 维度探测和 learner 中的 critic 尾部推断。
2. 拆开 Actor/Critic，适配 PPO/APPO/FlashSAC；rollout/replay 保存原始
   当前与下一步特权向量。纯超时用旧 episode 同步快照 bootstrap，真实
   掉落及与超时同时发生的掉落不 bootstrap。
3. 新样本归一化计数、异步权重/统计同步、checkpoint 契约校验、统一
   student 蒸馏及 teacher run 配置重建。
4. 合并公共配置，移除独立 HORA profile，修复 nodr 的事件覆盖；选定并
   记录 APPO 实际超参数基线。
5. 全局环境步预算、固定场景 evaluator、FlashSAC 分布饱和诊断，然后
   验证 train → save → load → eval → distill，再执行多种子比较。
