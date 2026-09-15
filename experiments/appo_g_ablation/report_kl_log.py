"""Append the reviewed logarithmic-term follow-up to the original journal."""

import json
import re

from adapter import OUT, ROOT, write
from freeze import digest, hashes
from write_report import table


def main():
    evidence = json.loads((OUT / "kl_log_analysis.json").read_text())
    manifest_path = OUT / "kl_log_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert all(evidence["source_integrity"].values())
    conclusions = (OUT / "kl_log_conclusions.md").read_text()
    assert "TODO" not in conclusions and "FINAL_" not in conclusions
    rows = evidence["summaries"]
    labels = {
        "N": "N（精确 KL）",
        "R": "R（原 G 重建）",
        "C": "C（完整旧 KL＋宽边界）",
        evidence["name"]: "N-logε（仅恢复旧对数项）",
    }
    new = rows[-1]
    stats = evidence["schedule_statistics"]
    lines = [
        "## 11. 用户指定后续实验：N 设置下仅恢复旧 KL 对数项（2026-09-16，已完成）",
        "",
        conclusions,
        "",
        "### 11.1 单因素定义与执行",
        "",
        "按用户最后明确选择，保留 N 的 log-std、无 std 边界、公共 actor/collector/learner、tanh、熵、V-trace、优化器及全部环境设置，仅把精确 KL 的 `log(std) - log(old_std)` 替换为 `log(std / old_std + 1e-5)`。其余平方项继续使用新公式的运算顺序：",
        "",
        "```python",
        "KL = (log(std / old_std + 1e-5)",
        "      + 0.5 * ((old_std / std)**2 + ((old_mean - mean) / std)**2 - 1)).sum(-1).mean()",
        "```",
        "",
        "KL 仅用于自适应学习率及记录，不加入优化 loss。忽略舍入时，此干预相对精确 KL 增加 `sum(log(1 + 1e-5 * old_std / std))`，不是固定常数，也不是只修改 KL=0 分支。它同时恢复了原对数项的比值/取对数运算顺序；没有恢复完整旧 KL 的其余运算顺序。",
        "",
        "原生 N 是本 worktree 已完成的同 seed 基线。新组从相同初始化重新训练 501 iterations，seed=1、collector seed=2、2048 环境、GPU 0、CPU 0–127、各进程 4 个 Torch 线程；FP32、compile=false，其余配置经完整对照一致。新旧依赖和生产源码未修改。最终使用同一 240 场景、20 秒固定窗口评估；五个中间检查点使用冻结的 24 场景，未选最佳 checkpoint。",
        "",
        "这是一项用户在原六＋四组实验完成后新指定的独立后续实验。此前固定常数方案已随用户澄清停止：只完成 53 轮、未评估，保留为 `N_kl_bias_seed1/` 并标记 `aborted_by_user_steering`，不计入结果。用户指定的 N-logε 从头训练，没有从该中止运行续训。",
        "",
        "### 11.2 最终结果",
        "",
        table(
            [
                "组",
                "速度 rad/s",
                "高度退出 /240",
                "存活 s",
                "q 二阶差分 RMS rad",
                "平均 std",
                "饱和率 %",
                "目标限位 %",
            ],
            [
                [
                    labels[r["group"]],
                    f"{r['speed_fixed_window']:.6f}",
                    r["height_exits"],
                    f"{r['survival_seconds']:.4f}",
                    f"{r['q_second_difference_rms']:.8f}",
                    f"{r['std']:.6f}",
                    f"{100 * r['saturation']:.4f}",
                    f"{100 * r['target_at_limit']:.4f}",
                ]
                for r in rows
            ],
        ),
        "",
        table(
            ["N-logε 减参考组", "速度差 rad/s", "退出差", "q RMS 差 rad"],
            [
                [
                    g,
                    f"{r['speed_fixed_window']:+.6f}",
                    f"{r['height_exits']:+d}",
                    f"{r['q_second_difference_rms']:+.8f}",
                ]
                for g, r in evidence["effects_new_minus_reference"].items()
            ],
        ),
        "",
        "C 的宽边界在原实验全程未触发；它作为完整旧 KL 的补充参考，不能省略这个配置差异。N-logε 与原生 N 才是本次严格的单因素对照。q RMS 为关节位置二阶差分，不直接代表高频振动或控制品质，需结合速度和存活解读。",
        "",
        "### 11.3 机制与一致性核验",
        "",
        f"新组 received={new['received']:,}、collected={new['collected']:,}、optimizer updates={new['optimizer_updates']:,}、training samples={new['training_samples']:,}，纯训练 {new['training_seconds'] / 60:.2f} 分钟。初始化 actor/critic、learner/collector RNG、除实验元数据和输出路径外的完整配置、六份 checkpoint 及 staging/样本计数均核验通过。",
        "",
        f"在新组自身数据上，带偏置与精确 KL 的影子调度分歧共 {stats['selected_vs_exact_branch_disagreements']} 次，其中首 minibatch 占 {stats['first_minibatch_disagreements']} 次；与完整旧 KL 的影子调度分歧为 {stats['selected_vs_full_old_branch_disagreements']} 次。首 minibatch 精确 KL 全为 0，干预 KL 范围为 {stats['same_policy_bias_first_minibatch_min']:.12g}–{stats['same_policy_bias_first_minibatch_max']:.12g}。首轮实际 LR 从 0.001 升为 0.0011；原生 N 首 minibatch 保持 0.001。",
        "",
        "影子调度比较在同一组自身策略上计算，不能当作另一条闭环训练轨迹。固定数据审计覆盖初始与训练后状态：干预前后除 KL 外的 loss 输出、梯度及固定 LR 的 Adam 更新逐位相同；零偏置 hook 的两轮短跑与原生 N 权重一致。",
        "",
        table(
            [
                "参考组",
                "std 首次差异轮",
                "LR 首次差异轮",
                "packet 首次差异轮",
                "staging 首次差异轮",
            ],
            [
                [
                    g,
                    d["first_std_difference"],
                    d["first_LR_difference"],
                    d["first_packet_difference"],
                    d["first_staging_difference"],
                ]
                for g, d in evidence["divergences"].items()
            ],
        ),
        "",
        "`None` 表示全程未分化。最终 state dict 相等性、原始调度 KL 和每 minibatch LR 记录见分析 JSON；R 的 std 参数化及 state dict 名称不同，不能把其字典不相等单独作为机制证据。",
        "",
        table(
            [
                "N-logε 减参考组",
                "共同成功",
                "仅参考组退出",
                "仅新组退出",
                "共同退出",
                "共同窗口 q RMS 差",
                "共同成功子集 q RMS 差",
            ],
            [
                [
                    g,
                    d["contingency"]["both_survive"],
                    d["contingency"]["left_only_exit"],
                    d["contingency"]["right_only_exit"],
                    d["contingency"]["both_exit"],
                    f"{d['common_window']['difference']:+.8f}",
                    f"{d['both_full_survivor_subset']['difference']:+.8f}",
                ]
                for g, d in evidence["paired_scenes"].items()
            ],
        ),
        "",
        "共同窗口包含终止步，截止双方较早退出，不做退出后零填充；共同成功子集有选择偏差。新组失败场景："
        + (", ".join(evidence["failures"]) or "无")
        + "。",
        "",
        "### 11.4 产物",
        "",
        f"- [训练与中间评估曲线]({OUT}/kl_log_curves.png)。",
        f"- [完整分析]({OUT}/kl_log_analysis.json)、[执行清单]({OUT}/kl_log_manifest.json)、[逐场景最终评估]({OUT}/evaluation/N_kl_log_epsilon_501.json)。",
        f"- [每 minibatch 调度对照]({OUT}/audit/kl_log_schedule.json)、[固定数据审计]({OUT}/audit/kl_bias_fixed_data.json)。",
        "",
        "结论只针对训练 seed=1。未运行额外训练 seed、未改动生产调度器，也未据此推荐恢复有偏 KL。",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "kl_log_report.md").write_text(report)
    tracked = ROOT / "docs/validation/appo-g-kl-log-20260916.md"
    tracked.write_text(report)
    journal = ROOT.parent / "sharpa_rl_unilab/dist/journals/10-plan-appo-g-ablation.md"
    original = journal.read_text()
    backup = OUT / "audit/journal_before_kl_log.md"
    if not backup.exists():
        backup.write_text(original)
    prefix = original.split("\n## 11. 用户指定后续实验", 1)[0].rstrip()
    prefix = re.sub(
        r"日期：2026-09-16。状态：[^\n]*",
        "日期：2026-09-16。状态：已完成六组核心、四组追加及一组用户指定后续对照；第 1–9 节保留原计划，第 10–11 节为实测结果与结论。",
        prefix,
        count=1,
    )
    journal.write_text(prefix + "\n\n" + report)
    artifacts = {}
    for pattern in [
        "N_kl_log_epsilon_seed1/*.pt",
        "N_kl_log_epsilon_seed1/*.jsonl",
        "evaluation/N_kl_log_epsilon_*.json",
        "evaluation/N_kl_log_epsilon_*.npz",
    ]:
        artifacts.update({str(p.relative_to(OUT)): digest(p) for p in OUT.glob(pattern)})
    write(OUT / "kl_log_artifact_hashes.json", artifacts)
    manifest["status"] = "complete"
    manifest["final_analysis_code_hashes"] = hashes(ROOT / "experiments/appo_g_ablation")
    write(manifest_path, manifest)
    compact = {
        **evidence,
        "paired_scenes": {
            g: {k: v for k, v in d.items() if k != "scenes"}
            for g, d in evidence["paired_scenes"].items()
        },
        "manifest_sha256": digest(manifest_path),
    }
    write(ROOT / "docs/validation/appo-g-kl-log-20260916.json", compact)
    print("Wrote", journal, "and", tracked)


if __name__ == "__main__":
    main()
