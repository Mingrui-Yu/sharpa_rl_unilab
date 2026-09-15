"""Write the reviewed conclusions and mechanically generated evidence into the journal."""

import json
import re

from adapter import OUT, ROOT


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def main():
    core = json.loads((OUT / "analysis.json").read_text())
    extra = json.loads((OUT / "additional_analysis.json").read_text())
    verification = json.loads((OUT / "audit/final_verification.json").read_text())
    assert len(verification["groups"]) == 10
    assert json.loads((OUT / "manifest.json").read_text())["status"] == "complete"
    conclusions = (OUT / "conclusions.md").read_text()
    assert "FINAL_" not in conclusions
    assert (OUT / "videos/posthoc_failure.json").exists()
    assert (OUT / "audit/final_source_integrity.json").exists()
    rows = core["summaries"] + extra["summaries"]
    labels = {
        "C_fixed_lr": "C-LR",
        "D_fixed_lr": "D-LR",
        "A_learner_density": "A-LP",
        "A_target_density": "A-TP",
    }

    def name(group):
        return labels.get(group, group)

    lines = [
        "## 10. 实验结果与最终结论（2026-09-16，已完成）",
        "",
        conclusions,
        "",
        "### 10.1 执行条件与可追溯性",
        "",
        f"工作目录：`{ROOT}`；分支 `experiment/appo-g-ablation`，生产源码基准 `678e3dc`。产物位于 `{OUT}`。完成六组核心及四组按证据触发的追加训练，均为训练 seed=1、collector seed=2、2048 环境、8 steps/rollout、501 iterations、5 epochs × 4 minibatches、tanh、entropy_coef=0.01、初始实际 std=1。未改变奖励、网络、PD、观测或外部依赖安装；实验 adapter 未放松生产 `make_models` 的 legacy 限制。",
        "",
        "使用 A800 GPU 0、CPU 0–127、learner/collector 各 4 个 Torch 线程，compile=false、FP32、无 autocast、matmul TF32=false。全部完整训练串行；评估在训练结束后使用 CPU，视频使用空闲 GPU。16 核初次 R 尝试仅作为吞吐校准归档，未评估；改用 128 核后六组全部重新从头训练，选择依据只有吞吐量。核心顺序 R→A→B→D→C→N；追加顺序 C-LR→D-LR→A-LP→A-TP。",
        "",
        "R 仍标为“原 G 重建版”：保留 worktree 缺少历史运行时完整源码哈希。冻结其 actor/loss/collector 并统一运行器后，R 最终 actor、critic、target actor 与历史原 G **逐位一致**；D/N 与历史新 G 也逐位一致。模型/资产/缓存/场景及核心依赖源码哈希已归档，生产源码与历史 checkpoint 的最终完整性检查通过。NumPy/TensorDict/OmegaConf 的源码哈希为核心运行结束后补录，未将其伪称为事前快照；运行中未安装或编辑依赖。",
        "",
        "最终评估使用原 manifest `e592d62b6f716a30d3033938debed34b0985e771cfdae72682caabba09b3510a`，8 尺度 × 3 评估 seed × 10 初态，确定性 `tanh(mean)`，固定 20 秒；100/200/300/400/500 快照使用事前冻结的 24 场景，仅作分化定位。所有主结果取 501 最终 checkpoint。",
        "",
        "### 10.2 最终指标",
        "",
        "A-LP 仅恢复旧 learner Gaussian log-prob 运算顺序；A-TP 仅恢复旧 target actor `Normal.log_prob` 运算顺序。二者都以 A 为起点，分别独立干预，保留当前 collector、直接 std、旧 KL 和其他实现。C-LR / D-LR 均回放事前指定并在最终评估前冻结的 C 组 10,020 项 LR 序列，仍记录各自 KL。",
        "",
        table(
            [
                "组",
                "速度 rad/s",
                "高度退出 /240",
                "平均存活 s",
                "q 二阶差分 RMS rad",
                "平均 std",
                "饱和率 %",
                "动作差 RMS",
                "目标限位 %",
            ],
            [
                [
                    name(r["group"]),
                    f"{r['speed_fixed_window']:.6f}",
                    r["height_exits"],
                    f"{r['survival_seconds']:.4f}",
                    f"{r['q_second_difference_rms']:.8f}",
                    f"{r['std']:.6f}",
                    f"{r['saturation'] * 100:.4f}",
                    f"{r['action_delta_rms']:.6f}",
                    f"{r['target_at_limit'] * 100:.4f}",
                ]
                for r in rows
            ],
        ),
        "",
        "q 二阶差分单位为 rad，未直接称作加速度或高频振动；诊断包含终止步，使用实际执行动作，排除 reset 跨界。固定窗口速度为终止前累计角位移 /20，不用存活期间速度替代。",
        "",
        table(
            ["组", "collected", "received", "optimizer updates", "training samples", "训练分钟"],
            [
                [
                    name(r["group"]),
                    r["collected"],
                    r["received"],
                    r["optimizer_updates"],
                    r["training_samples"],
                    f"{r['training_seconds'] / 60:.2f}",
                ]
                for r in rows
            ],
        ),
        "",
        "### 10.3 核心成对效应与交互",
        "",
        table(
            ["右−左", "速度差 rad/s", "退出差", "q RMS 差 rad"],
            [
                [
                    k,
                    f"{r['speed_fixed_window']:+.6f}",
                    f"{r['height_exits']:+d}",
                    f"{r['q_second_difference_rms']:+.8f}",
                ]
                for k, r in core["effects"].items()
            ],
        ),
        "",
        "参数化的速度效应在旧/精确 KL 下方向相反，不能把单条回退路径分解成独立贡献百分比。两种 KL 下改为 log-std 都增加退出，但单一训练 seed 不支持跨 seed 泛化结论。",
        "",
        table(
            [
                "比较",
                "std 首次逐位差异轮",
                "std 首次超参考容差轮",
                "KL 首次差异轮",
                "LR 首次差异轮",
                "packet/行为版本差异轮",
            ],
            [
                [
                    k,
                    r["first_std_difference"],
                    r["first_std_reference_tolerance_exceeded"],
                    r["first_selected_kl_difference"],
                    r["first_lr_difference"],
                    r["first_packet_version_difference"],
                ]
                for k, r in core["divergences"].items()
            ],
        ),
        "",
        "`None` 表示全程未发生；std 参考容差沿用 atol=1e-6、rtol=1e-5。所有组接收顺序、packet 行为版本和 staging 轨迹均有原始记录与重建核验。",
        "",
        "### 10.4 固定数据机制与训练诊断",
        "",
        "相同策略时旧 KL=0.000220298767、精确 KL=0；旧公式在首 minibatch 将 LR×1.1，精确 KL 保持。每组 501 轮首 minibatch 均出现该分支分歧，另有接近调度阈值的分歧。固定同一 LR 后，A/B 与 C/D 的离线 loss、梯度、Adam 更新一致。",
        "",
        table(
            [
                "核心组",
                "LR up/down/hold 分支数",
                "全部影子分支分歧",
                "实际 std 最小/最大",
                "边界命中",
                "梯度裁剪 %",
            ],
            [
                [
                    r["group"],
                    f"{r['lr_up']}/{r['lr_down']}/{r['lr_same']}",
                    r["shadow_branch_disagreements"],
                    f"{r['std_min_all']:.6f}/{r['std_max_all']:.6f}",
                    r["boundary_hits"],
                    f"{r['grad_clip_fraction'] * 100:.2f}",
                ]
                for r in core["summaries"]
            ],
        ),
        "",
        "直接 std / log-std 的链式法则检查通过。代表性训练后状态的梯度范数约 22.23157 /22.23045，均触发全局裁剪；参数化会改变 std 实际更新与裁剪系数，但没有证据把最终性能差异全部归给裁剪或最终 std 均值。",
        "",
        "采样原语在 CPU/CUDA 的测试输入上产生相同 raw action 与后继 RNG 状态；tanh 熵值及语义一致。R/A 在共享 targets/advantages 的单 minibatch 对照中 loss 和一次 Adam 参数更新一致，梯度最大误差约 5.59e-9（初始）/1.49e-8（训练后）。独立恢复旧 learner 密度运算顺序后，这些梯度残差在 CPU/CUDA 都归零；仅改变 std 的广播/clamp 布局未消除它们。",
        "",
        "训练后状态的 target log-prob / V-trace return 有少量元素未通过原始参考容差，GPU 最大绝对差约 3.81e-6 /2.19e-5；保留原始误差，没有放宽容差掩盖。CPU 回放只恢复旧 target 密度顺序即可使 log-prob、returns、advantages 一致。使用各自 V-trace 的完整单步中，新建 Adam 最大更新差约 2.10e-5，独立加载历史 Adam moments 后为 5.96e-8；所以“共享 targets 的 Adam 更新一致”不能外推为完整训练路径一致。",
        "",
        "### 10.5 四组追加实验",
        "",
        "触发依据：B−A 与 D−C 均降低速度并增加退出，其中 D−C 的速度效应更大，故选择 C/D 的预指定 C 序列做 LR 中介对照。R/A 在相同参数化、KL、初始化及 packet 轨迹下仍有差异，固定数据审计已定位两处密度数值路径，故分别单因素恢复。R/A 与 D/N 未出现异步轨迹差异，D/N 更是权重与训练记录一致，未触发原生异步重复组。追加额度已用满四组，未继续搜索。",
        "",
        table(
            ["追加比较", "速度差 rad/s", "退出差", "q RMS 差 rad"],
            [
                [
                    k,
                    f"{r['speed_fixed_window']:+.6f}",
                    f"{r['height_exits']:+d}",
                    f"{r['q_second_difference_rms']:+.8f}",
                ]
                for k, r in extra["effects"].items()
            ],
        ),
        "",
        "固定 LR 组日志里的 `branch` 是原生调度的反事实分支，`lr_after` 才是实际回放值；没有把反事实分支当作实际 LR 干预。完整 LR/权重/评估轨迹相等性见 `additional_analysis.json` 的 `mediation_equalities`。",
        "",
        table(
            [
                "追加比较",
                "std 首次逐位差异轮",
                "std 首次超参考容差轮",
                "LR 首次差异轮",
                "packet/行为版本差异轮",
            ],
            [
                [
                    k,
                    r["first_std_difference"],
                    r["first_std_reference_tolerance_exceeded"],
                    r["first_lr_difference"],
                    r["first_packet_version_difference"],
                ]
                for k, r in extra["divergences"].items()
            ],
        ),
        "",
        "### 10.6 生存窗口与逐场景结果",
        "",
        table(
            [
                "右−左",
                "共同成功",
                "仅左退出",
                "仅右退出",
                "共同退出",
                "共同窗口 q RMS 差",
                "共同成功子集 q RMS 差",
            ],
            [
                [
                    k,
                    v["contingency"]["both_survive"],
                    v["contingency"]["left_only_exit"],
                    v["contingency"]["right_only_exit"],
                    v["contingency"]["both_exit"],
                    f"{v['common_window']['difference']:+.8f}",
                    f"{v['both_full_survivor_subset']['difference']:+.8f}",
                ]
                for k, v in {**core["paired_scenes"], **extra["paired_scenes"]}.items()
            ],
        ),
        "",
        "共同窗口截止双方较早退出，包含终止步；未用退出后零填充压低 RMS。N−R 的 RMS 下降在共同窗口和双方成功子集中仍存在，说明并非完全由早退窗口造成；但速度与存活同时变差，不能据此称控制改善。共同成功子集有选择偏差；例如 D−B 的 q RMS 差在全体与共同成功子集中方向不同，只能作为补充解释。",
        "",
        "失败场景 ID（追加组详见 JSON；C-LR/D-LR 若等同 C，则保留相同列表）：",
        "",
        table(
            ["组", "高度退出场景"],
            [
                [g, ", ".join(ids) or "无"]
                for g, ids in {**core["failures"], **extra["failures"]}.items()
            ],
        ),
        "",
        "逐场景及逐尺度完整结果保存在 `evaluation/*_501.json`、`per_scale_comparison.csv` 与配对分析 JSON。240 场景的差异描述的是这些策略的评估分布，不是 240 次训练重复。",
        "",
        "N 的 8 次退出均在尺度 1.4/1.5（分别 5/3 次），存活时间 0.15–6.90 秒；R 在这 8 个场景均存活完整 20 秒。事后视频选择最早退出的 `1.4/10002/6`，N 在 0.15 秒终止后冻结画面，并显示退出标记；这种高度退出不能仅凭视频定性为某一种失稳机制。",
        "",
        "### 10.7 图表、视频与复现入口",
        "",
        f"- [训练曲线：iteration]({OUT}/training_iteration.png) / [received transitions]({OUT}/training_received.png)。",
        f"- [固定 24 场景中间评估：iteration]({OUT}/evaluation_iterations.png) / [received transitions]({OUT}/evaluation_received.png)。",
        f"- [LR 中介曲线]({OUT}/additional_mediation_curves.png) / [数值路径对照曲线]({OUT}/additional_numerical_curves.png)。",
        f"- [预定场景六组对比视频]({OUT}/videos/comparison.mp4)，`1/10001/0`，上排 R/A/B、下排 C/D/N。",
        f"- 事后失败诊断视频：[R]({OUT}/videos/posthoc_failure_R.mp4) / [N]({OUT}/videos/posthoc_failure_N.mp4)，按 N 最早退出选择，选择依据另存 `videos/posthoc_failure.json`；不用于挑选模型。",
        f"- [完整数值记录]({OUT}/comparison.csv)、[追加记录]({OUT}/additional_comparison.csv)、[源码与运行清单]({OUT}/manifest.json)、[实验入口说明]({ROOT}/experiments/appo_g_ablation/README.md)。",
        "",
        "验证：相关既有测试 36 项通过；十组完整训练、各六份 checkpoint、初始化/RNG、连续 packet、staging 重建与样本使用量核验通过；实验脚本 Ruff 检查通过。AMP/FP16/BF16 专项仍为后续 TODO，本轮未启动。",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    tracked = ROOT / "docs/validation/appo-g-ablation-20260916.md"
    tracked.write_text(report)
    journal = ROOT.parent / "sharpa_rl_unilab/dist/journals/10-plan-appo-g-ablation.md"
    original = journal.read_text()
    (OUT / "audit/journal_before_final.md").write_text(original)
    prefix = original.split("\n## 10. 执行记录", 1)[0].split("\n## 10. 实验结果", 1)[0].rstrip()
    prefix = re.sub(
        r"日期：2026-09-16。状态：[^\n]*",
        "日期：2026-09-16。状态：已完成六组核心及四组追加实验；第 1–9 节保留原计划，第 10 节为实测结果与最终结论。",
        prefix,
        count=1,
    )
    journal.write_text(prefix + "\n\n" + report)
    print("Wrote", journal, "and", tracked)


if __name__ == "__main__":
    main()
