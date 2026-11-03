# SuperMimic mixed Muon/Adam optimizer experiment

- Date: 2026-11-03 (Asia/Shanghai)
- Code: structured-WM commit `43686d4` plus the uncommitted mixed Muon/Adam optimizer diff archived as `muon_optimizer.patch` with the run.
- Hardware/host: two NVIDIA RTX 6000D GPUs on `d6k`.
- Runtime: Torch 2.11.0+cu128, UniLab 1.5.6, MuJoCo 3.11.0, Python 3.11.15.
- Training: SuperMimic direct, MuJoCo backend, 3,000 iterations, 2,048 envs, seeds 1/2, one RTX 6000D per seed.
- Experimental setting: `wm_structured=true`, `wm_drift_w=0.625`, `pi_opt=muon`, `wm_opt=muon`, existing learning rate 4e-4, `training.no_play=true`.
- Baseline: `logs/backend_compare/mujoco_seed_*` (`wm_structured` absent/false-equivalent, Adam actor/world model), MuJoCo train/eval.
- Evaluation: final checkpoint, deterministic actions, same fixed 240-episode manifest (SHA256 `f1ef05ee11db`), 20-second window.
- Secondary reference: existing structured-WM + Adam seeds in `logs/wm_structured_mujoco`; it separates world-model change from optimizer change.

## Aggregate evaluation

| Method | Seeds | Return | Fixed-window speed | Signed angle | Survival | Drop rate |
|---|---:|---:|---:|---:|---:|---:|
| structured WM + Muon/Adam | 2 | 30.105 ± 6.765 | 0.708 ± 0.117 | 14.162 ± 2.336 | 17.969 ± 1.930 | 0.177 ± 0.180 |
| unstructured WM + Adam (requested baseline) | 3 | 32.531 ± 2.926 | 0.731 ± 0.012 | 14.615 ± 0.243 | 19.108 ± 0.618 | 0.065 ± 0.039 |
| structured WM + Adam (control) | 2 | 39.171 ± 3.140 | 0.704 ± 0.045 | 14.074 ± 0.902 | 19.507 ± 0.085 | 0.033 ± 0.012 |

## Per-seed results

| Seed | Return (Muon / baseline) | Speed (Muon / baseline) | Survival (Muon / baseline) | Drop (Muon / baseline) |
|---:|---:|---:|---:|---:|
| 1 | 34.888 / 29.224 | 0.791 / 0.722 | 19.333 / 18.427 | 0.050 / 0.108 |
| 2 | 25.321 / 33.586 | 0.626 / 0.726 | 16.604 / 19.264 | 0.304 / 0.054 |

## Paired deltas versus the requested baseline (Muon minus unstructured Adam)

| Seed | Return Δ | Speed Δ | Survival Δ | Drop-rate Δ |
|---:|---:|---:|---:|---:|
| 1 | +5.665 | +0.069 | +0.906 | -5.83 pp |
| 2 | -8.264 | -0.101 | -2.660 | +25.00 pp |
| mean | -1.300 | -0.016 | -0.877 | +9.58 pp |

The paired comparison uses the two baseline seeds that match this run. For context, the three-seed baseline aggregate is shown above.

## Training cost and final training diagnostics

| Method | Seed | Wall (s) | Final WM loss | Final WM last-step error |
|---|---:|---:|---:|---:|
| Muon | 1 | 624.3 | 0.1007 | 0.1158 |
| Muon | 2 | 624.4 | 0.1879 | 0.2235 |
| baseline Adam | 1 | 558.2 | 0.1017 | 0.1226 |
| baseline Adam | 2 | 529.9 | 0.1197 | 0.1497 |
| baseline Adam | 3 | 531.5 | 0.1268 | 0.1520 |
| structured Adam control | 1 | 567.9 | 0.0876 | 0.1085 |
| structured Adam control | 2 | 570.2 | 0.1275 | 0.1677 |

## Interpretation

- The structured-WM + Muon run averages **30.105 return**, below the requested unstructured/Adam baseline mean **32.531**. That is an aggregate delta of **-2.426 return**; on only the two matched training seeds, the paired mean delta is **-1.300 return**.
- Results are seed-split: seed 1 improves over its baseline (+5.664 return), while seed 2 is substantially worse (-8.265 return).
- On matched seeds, mean survival is lower (-0.877 s), drop rate is higher (+9.58 pp), and fixed-window speed is marginally lower (-0.016 rad/s).
- Compared with the existing structured-WM + Adam control, Muon is worse on both seeds (mean return 30.105 vs 39.171; paired delta **-9.066**). This experiment therefore does not support making Muon the default for this structured-WM Sharpa task at the reused 4e-4 learning rate.
- This is a two-seed descriptive engineering comparison, not a significance test. The learning rate was intentionally reused; Muon may need a task-specific LR sweep.
- The historical unstructured/Adam baseline was run with Torch 2.14.1/CUDA and UniLab 1.5.4, while the Muon run used Torch 2.11.0+cu128 and UniLab 1.5.6. The fixed task manifest and evaluation protocol match, but dependency versions are an additional confound when comparing across those two logged campaigns.

## Reproduction

Each seed used one GPU (`CUDA_VISIBLE_DEVICES=0` for seed 1 and `1` for seed 2):

```bash
uv run sharpa-train --algo supermimic --sim mujoco \
  algo.single_stage=true algo.seed=<SEED> \
  algo.wm_structured=true algo.wm_drift_w=0.625 \
  algo.pi_opt=muon algo.wm_opt=muon \
  training.no_play=true \
  training.log_dir=logs/wm_structured_muon_mujoco/seed_<SEED>
```

Evaluate the final checkpoint against the copied fixed manifest:

```bash
CUDA_VISIBLE_DEVICES=<GPU> uv run sharpa-eval --algo supermimic \
  --checkpoint logs/wm_structured_muon_mujoco/seed_<SEED>/policy_final.pt \
  --output logs/wm_structured_muon_mujoco/eval/seed_<SEED>.json \
  evaluation.manifest=logs/wm_structured_muon_mujoco/eval/scenes.json \
  training.device=cuda:0
```

## Artifacts

- Training (retained on `d6k`): `/home/user/ws/sharpa_rl_unilab/logs/wm_structured_muon_mujoco/seed_{1,2}/policy_final.pt` and `metrics.jsonl`.
- Evaluation: `/home/user/ws/sharpa_rl_unilab/logs/wm_structured_muon_mujoco/eval/seed_{1,2}.json` and the copied fixed manifest.
- Machine-readable copy committed here: [2026-11-03-supermimic-muon-optimizer.json](2026-11-03-supermimic-muon-optimizer.json). The original remains at `/home/user/ws/sharpa_rl_unilab/logs/wm_structured_muon_mujoco/summary.json`.
- Applied optimizer diff snapshot: `/home/user/ws/sharpa_rl_unilab/logs/wm_structured_muon_mujoco/muon_optimizer.patch`.
