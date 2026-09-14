# 验证记录

## Manager-Based migration

日期：2026-09-14

- UniLab source：`b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`。
- `uv run ruff check src tests`：通过。
- `uv run pytest -q`：54 项通过。
- `uv run pyright`：0 个错误。
- `uv build`：source distribution 与 wheel 构建成功。
- APPO one-iteration smoke（`4` env）：collector 与 learner 完成，并写入
  `model_1.pt`。
- FlashSAC one-iteration smoke（`4` env）：collector 与 learner 完成，并写入
  `model_1.pt`。
- Manager 观测契约：APPO flat observation 174；FlashSAC actor 147、特权
  critic 174。
- Fixed variants：8 个 XML source 可编译，geom 名称唯一，并保留
  `simple="false"`。

## 完整训练基准

日期：2026-09-14

两个历史 run 均在指定训练主机上使用 MuJoCo、seed 1 与
`training.no_play=true`。

| 指标 | APPO | FlashSAC |
| --- | ---: | ---: |
| 状态 / iterations | completed / 301 | completed / 5,371 |
| Environment steps | 39,452,672 | 11,000,832 |
| Training wall time | 1 h 13 m 13 s | 41 m 23 s |
| Final mean episode return | 48.8223 | 34.8279 |
| Best mean episode return | 49.2989 | 38.9166 |
| Final mean episode length | 390.35 | 351.57 |
| Final timeout rate | 0.9636 | 0.8333 |
| Final collector throughput | 9,755.5 env steps/s | 4,483.6 env steps/s |
| Final learner throughput | — | 62,770.9 samples/s |
| Final policy / actor loss | -0.00131 | -1.21809 |
| Final value / critic loss | 0.03056 | 1.74715 |

APPO 获得更高的最终/最佳 return，物体保持时间也更长。FlashSAC 完成了完整
off-policy 日程，learner 样本效率更高，但在这个固定基准上的最终 return 低于
APPO。

## 文档工作流验证

日期：2026-09-14

文档命令被视为可执行接口。重构本指南后，以下检查均在仓库根目录执行。

配置合成：

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
uv run sharpa-train --algo ppo --sim mujoco --cfg
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```

语言一致性：

```bash
uv run pytest tests/test_documentation_parity.py -q
```

精确目录 smoke training：

- HORA APPO：4 env、2 steps/env、1 iteration；完成并写入 `model_1.pt`、
  `run_summary.json`，随后生成 `play_video.mp4`。
- FlashSAC：4 env、batch 8、replay 16、1 update/step、1 iteration；完成并写入
  `model_1.pt`、`run_summary.json`、`policy.onnx` 与 `play_video.mp4`。
  ONNX/PyTorch 最大输出差异为 `9.13e-07`。
- PPO：4 env、2 steps/env、1 iteration；完成并写入 `model_0.pt`、
  `run_summary.json`，随后生成 `play_video.mp4`。

规范默认树训练与最新 run 评估：

- 未设置 `training.log_root`，训练 HORA APPO、FlashSAC 与 PPO 的
  one-iteration run；输出位于
  `logs/{hora_appo,flash_sac,rsl_rl_ppo}/SharpaInhandRotation/`。
- 使用 `algo.load_run=-1` 与 `training.play_render_mode=record` 评估三者。
- 每次评估都加载了预期的最新 checkpoint 并生成 `play_video.mp4`；FlashSAC
  还导出并验证了 `policy.onnx`。
