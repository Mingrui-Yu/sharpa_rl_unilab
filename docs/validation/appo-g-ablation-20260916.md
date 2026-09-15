## 10. 实验结果与最终结论（2026-09-16，已完成）

本轮在一个训练 seed 下复现了历史差异：R 的最终权重逐位复现历史原 G，N 逐位复现历史当前 G。R→N 的固定窗口速度从 0.628088 降至 0.597009 rad/s（−4.95%），高度退出从 0/240 增至 8/240，关节二阶差分 RMS 从 0.00722120 降至 0.00671349 rad（−7.03%）。运动差分下降不能抵消速度与存活的损失。

**已排除（限于本轮运行与已审计输入）：**

- **移除宽 std 边界不是这次差异的来源。** D/N 全程未触边，最终 actor/critic/target actor、minibatch 记录、std 与 packet/staging 记录逐位一致，评估相同。这个结论不涵盖未来触边或出现异常 std 的训练。
- **已记录的异步 packet 合并、行为版本滞后和样本复用差异不能解释核心对照。** 六组轨迹一致，均无 packet 合并，received=8,208,384、训练样本使用量=326,041,600。统一轻量诊断和保存开销后仍逐位复现两端历史权重。没有触发额外异步重复组；未据此宣称 APPO 对任意运行条件都完全确定。
- CPU/CUDA 审计中，旧/公共原生采样的输出及后继 RNG 状态、tanh 熵语义相同。实际训练使用 FP32、无 autocast，没有支持 AMP/dtype 改变或采样 RNG 消耗改变的证据。环境、奖励、资产、初态与评估口径也已核对。

**有支持证据：**

- **KL→学习率调度是 C/D 性能差异的中介。** 同策略时旧 KL 约为 2.203e−4，触发 LR×1.1；精确 KL=0，保持 LR。每组所有 501 轮的首 minibatch 都出现这项影子分支分歧。原生 D−C 为速度 −0.032800 rad/s、退出 +1；固定数据上固定同一 LR 后更新一致，完整闭环中 C/D 回放事前指定的 C LR 序列后也得到相同最终权重。最终 240 场景评估同为 0.629809 rad/s、7/240 退出、q RMS=0.00716147 rad，与原生 C 相同；六个保存检查点及全程 std 轨迹也均逐位一致。这验证的是该 C 序列下 KL 的 LR 中介作用，不是固定 LR 的生产建议，也没有单独隔离零 KL 首 minibatch 与其他阈值分支各自的贡献。
- **std 参数化与 KL 调度有交互。** 旧 KL 下 C−A 为速度 +0.019996 rad/s、退出 +5；精确 KL 下 D−B 为速度 −0.006764 rad/s、退出 +2。速度交互为 −0.026760 rad/s。链式法则审计通过，实际 std 更新及全局裁剪系数可随参数化改变；不能将最终均值 std 或梯度裁剪单独认定为完整机制，也不能给各因素分配独立贡献百分比。
- **数值运算路径足以使相同 seed 的闭环训练分化。** R/A 首轮有微小梯度/std 舍入差异，第 3 轮 LR 分化，packet 轨迹始终相同。离线恢复旧 learner 密度顺序可在 CPU/CUDA 消除共享 targets 下的梯度残差；恢复旧 target 密度顺序可消除 CPU 回放的 target log-prob/V-trace 差异。完整单因素训练中，仅恢复 learner 密度顺序的 A-LP 为 0.639850 rad/s、1/240 退出，相比 A 速度 +0.030037、退出 −1；其第 1 轮 std 与 R 相同，但第 2 轮 std、第 4 轮 LR 仍分化。仅恢复 target 密度顺序的 A-TP 为 0.625653 rad/s、7/240 退出，相比 A 速度 +0.015841、退出 +5。相对 R，A-LP 速度 +0.011762、退出 +1，A-TP 速度 −0.002434、退出 +7。两种独立干预都支持数值路径会影响最终策略，但都未恢复 R 的完整轨迹或零退出表现；target 干预只提高速度而恶化存活，不能统称为“恢复性能”。

**仍不能区分：**

- R/A 差异不能被赋予一个可加的“框架损失”百分比；没有进行 learner+target 等多个数值路径的联合恢复，也没有用完备组合实验分解它们之间的交互。四组追加额度已经用满，不再按结果继续搜索。
- 参数化究竟主要经实际 std 步长、Adam 状态、全局梯度裁剪还是随后 KL/LR 分化影响性能，本轮未继续做中介隔离。
- 全部训练 seed=1；240 场景及历史复现都不是独立训练重复。无法估计跨训练 seed 的稳定性，不能将当前成绩差当作普适算法优劣。R 虽逐位复现历史权重，仍保留“原 G 重建版”的源码来源限定。

**最小实现建议：** 保留精确解析 KL 和公共分布框架；不要因这个 seed 的结果恢复有偏 KL 或直接上线固定 LR 回放。下一步如开展实现改动，优先将“相同策略 KL=0 时调度应否升 LR”明确为调度器的独立语义，并用单因素对照验证；保持 KL 指标本身数学正确。保留 std 参数化、边界、实际 LR 与数值路径的显式记录，并保留此次固定数据与保存加载回归检查。生产实现本轮未修改；AMP/FP16/BF16 专项继续留作 TODO。


### 10.1 执行条件与可追溯性

工作目录：`/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation`；分支 `experiment/appo-g-ablation`，生产源码基准 `678e3dc`。产物位于 `/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation`。完成六组核心及四组按证据触发的追加训练，均为训练 seed=1、collector seed=2、2048 环境、8 steps/rollout、501 iterations、5 epochs × 4 minibatches、tanh、entropy_coef=0.01、初始实际 std=1。未改变奖励、网络、PD、观测或外部依赖安装；实验 adapter 未放松生产 `make_models` 的 legacy 限制。

使用 A800 GPU 0、CPU 0–127、learner/collector 各 4 个 Torch 线程，compile=false、FP32、无 autocast、matmul TF32=false。全部完整训练串行；评估在训练结束后使用 CPU，视频使用空闲 GPU。16 核初次 R 尝试仅作为吞吐校准归档，未评估；改用 128 核后六组全部重新从头训练，选择依据只有吞吐量。核心顺序 R→A→B→D→C→N；追加顺序 C-LR→D-LR→A-LP→A-TP。

R 仍标为“原 G 重建版”：保留 worktree 缺少历史运行时完整源码哈希。冻结其 actor/loss/collector 并统一运行器后，R 最终 actor、critic、target actor 与历史原 G **逐位一致**；D/N 与历史新 G 也逐位一致。模型/资产/缓存/场景及核心依赖源码哈希已归档，生产源码与历史 checkpoint 的最终完整性检查通过。NumPy/TensorDict/OmegaConf 的源码哈希为核心运行结束后补录，未将其伪称为事前快照；运行中未安装或编辑依赖。

最终评估使用原 manifest `e592d62b6f716a30d3033938debed34b0985e771cfdae72682caabba09b3510a`，8 尺度 × 3 评估 seed × 10 初态，确定性 `tanh(mean)`，固定 20 秒；100/200/300/400/500 快照使用事前冻结的 24 场景，仅作分化定位。所有主结果取 501 最终 checkpoint。

### 10.2 最终指标

A-LP 仅恢复旧 learner Gaussian log-prob 运算顺序；A-TP 仅恢复旧 target actor `Normal.log_prob` 运算顺序。二者都以 A 为起点，分别独立干预，保留当前 collector、直接 std、旧 KL 和其他实现。C-LR / D-LR 均回放事前指定并在最终评估前冻结的 C 组 10,020 项 LR 序列，仍记录各自 KL。

| 组 | 速度 rad/s | 高度退出 /240 | 平均存活 s | q 二阶差分 RMS rad | 平均 std | 饱和率 % | 动作差 RMS | 目标限位 % |
|---|---|---|---|---|---|---|---|---|
| R | 0.628088 | 0 | 20.0000 | 0.00722120 | 0.459803 | 0.0241 | 0.240156 | 3.2487 |
| A | 0.609812 | 2 | 19.8592 | 0.00660325 | 0.462698 | 0.0327 | 0.223369 | 6.3335 |
| B | 0.603773 | 6 | 19.6646 | 0.00666984 | 0.461187 | 0.0190 | 0.208663 | 5.3826 |
| C | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 0.233780 | 4.4267 |
| D | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 0.219893 | 6.4318 |
| N | 0.597009 | 8 | 19.4292 | 0.00671349 | 0.475110 | 0.1135 | 0.219893 | 6.4318 |
| C-LR | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 0.233780 | 4.4267 |
| D-LR | 0.629809 | 7 | 19.4583 | 0.00716147 | 0.463016 | 0.0891 | 0.233780 | 4.4267 |
| A-LP | 0.639850 | 1 | 19.9173 | 0.00708014 | 0.458071 | 0.0614 | 0.234226 | 3.1736 |
| A-TP | 0.625653 | 7 | 19.5365 | 0.00737850 | 0.467758 | 0.1014 | 0.241330 | 5.4978 |

q 二阶差分单位为 rad，未直接称作加速度或高频振动；诊断包含终止步，使用实际执行动作，排除 reset 跨界。固定窗口速度为终止前累计角位移 /20，不用存活期间速度替代。

| 组 | collected | received | optimizer updates | training samples | 训练分钟 |
|---|---|---|---|---|---|
| R | 8224768 | 8208384 | 10020 | 326041600 | 7.51 |
| A | 8224768 | 8208384 | 10020 | 326041600 | 7.48 |
| B | 8224768 | 8208384 | 10020 | 326041600 | 7.87 |
| C | 8224768 | 8208384 | 10020 | 326041600 | 7.57 |
| D | 8224768 | 8208384 | 10020 | 326041600 | 7.56 |
| N | 8224768 | 8208384 | 10020 | 326041600 | 7.62 |
| C-LR | 8224768 | 8208384 | 10020 | 326041600 | 7.68 |
| D-LR | 8224768 | 8208384 | 10020 | 326041600 | 7.65 |
| A-LP | 8224768 | 8208384 | 10020 | 326041600 | 7.62 |
| A-TP | 8224768 | 8208384 | 10020 | 326041600 | 7.62 |

### 10.3 核心成对效应与交互

| 右−左 | 速度差 rad/s | 退出差 | q RMS 差 rad |
|---|---|---|---|
| A-R | -0.018275 | +2 | -0.00061795 |
| B-A | -0.006039 | +4 | +0.00006659 |
| D-C | -0.032800 | +1 | -0.00044798 |
| C-A | +0.019996 | +5 | +0.00055822 |
| D-B | -0.006764 | +2 | +0.00004365 |
| N-D | +0.000000 | +0 | +0.00000000 |
| N-R | -0.031079 | +8 | -0.00050771 |
| interaction_(D-C)-(B-A) | -0.026760 | -3 | -0.00051457 |

参数化的速度效应在旧/精确 KL 下方向相反，不能把单条回退路径分解成独立贡献百分比。两种 KL 下改为 log-std 都增加退出，但单一训练 seed 不支持跨 seed 泛化结论。

| 比较 | std 首次逐位差异轮 | std 首次超参考容差轮 | KL 首次差异轮 | LR 首次差异轮 | packet/行为版本差异轮 |
|---|---|---|---|---|---|
| A-R | 1 | 3 | 1 | 3 | None |
| B-A | 1 | 1 | 1 | 1 | None |
| D-C | 1 | 1 | 1 | 1 | None |
| C-A | 1 | 1 | 1 | 3 | None |
| D-B | 1 | 1 | 1 | 3 | None |
| N-D | None | None | None | None | None |
| N-R | 1 | 1 | 1 | 1 | None |

`None` 表示全程未发生；std 参考容差沿用 atol=1e-6、rtol=1e-5。所有组接收顺序、packet 行为版本和 staging 轨迹均有原始记录与重建核验。

### 10.4 固定数据机制与训练诊断

相同策略时旧 KL=0.000220298767、精确 KL=0；旧公式在首 minibatch 将 LR×1.1，精确 KL 保持。每组 501 轮首 minibatch 均出现该分支分歧，另有接近调度阈值的分歧。固定同一 LR 后，A/B 与 C/D 的离线 loss、梯度、Adam 更新一致。

| 核心组 | LR up/down/hold 分支数 | 全部影子分支分歧 | 实际 std 最小/最大 | 边界命中 | 梯度裁剪 % |
|---|---|---|---|---|---|
| R | 2824/2831/4365 | 601 | 0.249770/1.007447 | 0 | 89.10 |
| A | 2880/2892/4248 | 606 | 0.275083/1.007448 | 0 | 91.82 |
| B | 1981/1990/6049 | 618 | 0.268398/1.006934 | 0 | 90.29 |
| C | 2567/2581/4872 | 611 | 0.290436/1.007155 | 0 | 90.73 |
| D | 2596/2608/4816 | 603 | 0.301330/1.006829 | 0 | 90.89 |
| N | 2596/2608/4816 | 603 | 0.301330/1.006829 | 0 | 90.89 |

直接 std / log-std 的链式法则检查通过。代表性训练后状态的梯度范数约 22.23157 /22.23045，均触发全局裁剪；参数化会改变 std 实际更新与裁剪系数，但没有证据把最终性能差异全部归给裁剪或最终 std 均值。

采样原语在 CPU/CUDA 的测试输入上产生相同 raw action 与后继 RNG 状态；tanh 熵值及语义一致。R/A 在共享 targets/advantages 的单 minibatch 对照中 loss 和一次 Adam 参数更新一致，梯度最大误差约 5.59e-9（初始）/1.49e-8（训练后）。独立恢复旧 learner 密度运算顺序后，这些梯度残差在 CPU/CUDA 都归零；仅改变 std 的广播/clamp 布局未消除它们。

训练后状态的 target log-prob / V-trace return 有少量元素未通过原始参考容差，GPU 最大绝对差约 3.81e-6 /2.19e-5；保留原始误差，没有放宽容差掩盖。CPU 回放只恢复旧 target 密度顺序即可使 log-prob、returns、advantages 一致。使用各自 V-trace 的完整单步中，新建 Adam 最大更新差约 2.10e-5，独立加载历史 Adam moments 后为 5.96e-8；所以“共享 targets 的 Adam 更新一致”不能外推为完整训练路径一致。

### 10.5 四组追加实验

触发依据：B−A 与 D−C 均降低速度并增加退出，其中 D−C 的速度效应更大，故选择 C/D 的预指定 C 序列做 LR 中介对照。R/A 在相同参数化、KL、初始化及 packet 轨迹下仍有差异，固定数据审计已定位两处密度数值路径，故分别单因素恢复。R/A 与 D/N 未出现异步轨迹差异，D/N 更是权重与训练记录一致，未触发原生异步重复组。追加额度已用满四组，未继续搜索。

| 追加比较 | 速度差 rad/s | 退出差 | q RMS 差 rad |
|---|---|---|---|
| C_fixed_lr-C | +0.000000 | +0 | +0.00000000 |
| D_fixed_lr-C_fixed_lr | +0.000000 | +0 | +0.00000000 |
| D_fixed_lr-D | +0.032800 | -1 | +0.00044798 |
| A_learner_density-A | +0.030037 | -1 | +0.00047689 |
| A_learner_density-R | +0.011762 | +1 | -0.00014106 |
| A_target_density-A | +0.015841 | +5 | +0.00077525 |
| A_target_density-R | -0.002434 | +7 | +0.00015730 |

固定 LR 组日志里的 `branch` 是原生调度的反事实分支，`lr_after` 才是实际回放值；没有把反事实分支当作实际 LR 干预。完整 LR/权重/评估轨迹相等性见 `additional_analysis.json` 的 `mediation_equalities`。

| 追加比较 | std 首次逐位差异轮 | std 首次超参考容差轮 | LR 首次差异轮 | packet/行为版本差异轮 |
|---|---|---|---|---|
| C_fixed_lr-C | None | None | None | None |
| D_fixed_lr-C_fixed_lr | None | None | None | None |
| D_fixed_lr-D | 1 | 1 | 1 | None |
| A_learner_density-A | 1 | 3 | 3 | None |
| A_learner_density-R | 2 | 3 | 4 | None |
| A_target_density-A | 2 | 4 | 6 | None |
| A_target_density-R | 1 | 3 | 3 | None |

### 10.6 生存窗口与逐场景结果

| 右−左 | 共同成功 | 仅左退出 | 仅右退出 | 共同退出 | 共同窗口 q RMS 差 | 共同成功子集 q RMS 差 |
|---|---|---|---|---|---|---|
| A-R | 238 | 0 | 2 | 0 | -0.00064536 | -0.00063163 |
| B-A | 233 | 1 | 5 | 1 | +0.00007041 | +0.00007439 |
| D-C | 228 | 4 | 5 | 3 | -0.00042446 | -0.00043644 |
| C-A | 232 | 1 | 6 | 1 | +0.00048164 | +0.00048044 |
| D-B | 228 | 4 | 6 | 2 | +0.00001904 | -0.00001915 |
| N-D | 232 | 0 | 0 | 8 | +0.00000000 | +0.00000000 |
| N-R | 232 | 0 | 8 | 0 | -0.00052788 | -0.00058919 |
| C_fixed_lr-C | 233 | 0 | 0 | 7 | +0.00000000 | +0.00000000 |
| D_fixed_lr-C_fixed_lr | 233 | 0 | 0 | 7 | +0.00000000 | +0.00000000 |
| D_fixed_lr-D | 228 | 5 | 4 | 3 | +0.00042446 | +0.00043644 |
| A_learner_density-A | 237 | 2 | 1 | 0 | +0.00042873 | +0.00045349 |
| A_learner_density-R | 239 | 0 | 1 | 0 | -0.00015037 | -0.00018098 |
| A_target_density-A | 231 | 2 | 7 | 0 | +0.00070952 | +0.00068380 |
| A_target_density-R | 233 | 0 | 7 | 0 | +0.00012583 | +0.00005959 |

共同窗口截止双方较早退出，包含终止步；未用退出后零填充压低 RMS。N−R 的 RMS 下降在共同窗口和双方成功子集中仍存在，说明并非完全由早退窗口造成；但速度与存活同时变差，不能据此称控制改善。共同成功子集有选择偏差；例如 D−B 的 q RMS 差在全体与共同成功子集中方向不同，只能作为补充解释。

失败场景 ID（追加组详见 JSON；C-LR/D-LR 若等同 C，则保留相同列表）：

| 组 | 高度退出场景 |
|---|---|
| R | 无 |
| A | 1.4/10002/6, 1.5/10003/0 |
| B | 0.8/10002/1, 0.8/10003/5, 1/10001/2, 1.4/10002/8, 1.5/10003/0, 1.5/10003/9 |
| C | 0.8/10002/8, 0.9/10001/8, 1.4/10001/8, 1.5/10001/3, 1.5/10003/0, 1.5/10003/1, 1.5/10003/8 |
| D | 1.4/10001/8, 1.4/10002/3, 1.4/10002/5, 1.4/10002/6, 1.4/10003/5, 1.5/10003/0, 1.5/10003/8, 1.5/10003/9 |
| N | 1.4/10001/8, 1.4/10002/3, 1.4/10002/5, 1.4/10002/6, 1.4/10003/5, 1.5/10003/0, 1.5/10003/8, 1.5/10003/9 |
| C_fixed_lr | 0.8/10002/8, 0.9/10001/8, 1.4/10001/8, 1.5/10001/3, 1.5/10003/0, 1.5/10003/1, 1.5/10003/8 |
| D_fixed_lr | 0.8/10002/8, 0.9/10001/8, 1.4/10001/8, 1.5/10001/3, 1.5/10003/0, 1.5/10003/1, 1.5/10003/8 |
| A_learner_density | 0.9/10001/8 |
| A_target_density | 0.8/10001/1, 0.9/10001/8, 0.9/10003/5, 1.4/10002/8, 1.4/10003/1, 1.5/10001/3, 1.5/10003/9 |

逐场景及逐尺度完整结果保存在 `evaluation/*_501.json`、`per_scale_comparison.csv` 与配对分析 JSON。240 场景的差异描述的是这些策略的评估分布，不是 240 次训练重复。

N 的 8 次退出均在尺度 1.4/1.5（分别 5/3 次），存活时间 0.15–6.90 秒；R 在这 8 个场景均存活完整 20 秒。事后视频选择最早退出的 `1.4/10002/6`，N 在 0.15 秒终止后冻结画面，并显示退出标记；这种高度退出不能仅凭视频定性为某一种失稳机制。

### 10.7 图表、视频与复现入口

- [训练曲线：iteration](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/training_iteration.png) / [received transitions](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/training_received.png)。
- [固定 24 场景中间评估：iteration](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/evaluation_iterations.png) / [received transitions](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/evaluation_received.png)。
- [LR 中介曲线](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/additional_mediation_curves.png) / [数值路径对照曲线](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/additional_numerical_curves.png)。
- [预定场景六组对比视频](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/videos/comparison.mp4)，`1/10001/0`，上排 R/A/B、下排 C/D/N。
- 事后失败诊断视频：[R](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/videos/posthoc_failure_R.mp4) / [N](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/videos/posthoc_failure_N.mp4)，按 N 最早退出选择，选择依据另存 `videos/posthoc_failure.json`；不用于挑选模型。
- [完整数值记录](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/comparison.csv)、[追加记录](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/additional_comparison.csv)、[源码与运行清单](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/logs/appo_g_ablation/manifest.json)、[实验入口说明](/home/openpai/mingrui/research/UniLab-Dev/sharpa_rl_unilab-appo-g-ablation/experiments/appo_g_ablation/README.md)。

验证：相关既有测试 36 项通过；十组完整训练、各六份 checkpoint、初始化/RNG、连续 packet、staging 重建与样本使用量核验通过；实验脚本 Ruff 检查通过。AMP/FP16/BF16 专项仍为后续 TODO，本轮未启动。
