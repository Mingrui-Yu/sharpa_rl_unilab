# Validation record

## 2026-09-14 Issue 2: common HORA protocol v2

Implementation and checks are confined to Sharpa. UniLab remains at
`b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`, with a clean working tree.
The [migration record](migrations/issue-2.md) describes supported interfaces and
the selected optimizer baselines.

- **39 tests passed**, including real MuJoCo timeout/autoreset, simultaneous
  drop/timeout, partial reset, independent gradients, raw replay privilege,
  behavior-policy density, normalization counts, and all three algorithms'
  train → periodic save/eval → load → distill → student eval flows.
- Retired tests for shared Actor–Critic, critic-tail privilege extraction,
  APPO-only distillation and HORA profiles were replaced with v2 contract tests.
- Ruff lint, mypy and pyright passed. Changed Python files pass format checks;
  the full format check still reports nine untouched files with existing issues.
- The lockfile check passed; source distribution and wheel built successfully.
  The wheel includes common configuration and excludes retired HORA modules.
- Real GPU pipeline smoke: **PPO/APPO/FlashSAC × training seeds [1,2,3]**, one
  NVIDIA A800 80 GB learner GPU (`cuda:0`), eight global MuJoCo environments,
  two Torch CPU threads. Each teacher collected exactly **128 transitions**;
  each student collected **32**, with four latent-MSE updates. FlashSAC used
  batch 32 for this smoke; other effective optimization settings were retained.
- Every teacher's base normalization count was 128. Every student retained that
  frozen count and had history normalization count 32. All 18 checkpoint loads
  and deterministic evaluations succeeded, using one shared manifest with eight
  scales and one episode per scale: **144 evaluation episodes**, each ending at
  drop or the full 20-second limit. No dropped episode was replaced.
- Teacher and student stages launch in separate processes. Snapshots include
  collected/received counts, optimization work and initialization-inclusive
  timing. Initialization, evaluation, and training timings in this tiny smoke
  are diagnostic only and do not establish throughput or learning efficiency.

| Model | Actor parameters | Independent critic parameters |
| --- | ---: | ---: |
| PPO / APPO | 284,085 | 253,953 (V) |
| FlashSAC adaptation | 286,923 | 8,715,226 (double Q) |

The common privilege encoder has 36,617 parameters. Target Q parameters are
additional copies; FlashSAC also retains its native learned temperature.

Committed artifacts: [run/seed summaries](migrations/issue-2-smoke/summary.json),
[all episode measurements](migrations/issue-2-smoke/episodes.csv),
[fixed scenes](migrations/issue-2-smoke/scenes.json), and
[step/time plots](migrations/issue-2-smoke/learning_curves.png).
Full checkpoints/configs remain locally under `dist/issue-2-validated/`.
The summaries include per-scale and training-seed bootstrap statistics.
These are **pipeline validation results**, not a ranking: no 5M teacher / 100M
student performance experiment was run, and old checkpoints require retraining.

Reproduce the short GPU validation with:

```bash
uv sync --extra mujoco --extra evaluation
uv run sharpa-compare --output dist/issue-2-validated --device cuda:0 --smoke
```

For this development worktree, checks used the existing virtualenv interpreter
and absolute `PYTHONPATH` entries for this worktree's `src` and UniLab's `src`.
Neither dependency sources nor the original worktree's user edits were changed.

## Historical protocols

The runs below use the pre-issue-2 observation and training protocols. Their
Actor inputs, sample budgets and evaluation procedures differ. They are
historical execution records, not an algorithm ranking or evidence of relative
sample efficiency. See [the issue 2 migration record](migrations/issue-2.md).

## 2026-09-14 Manager-Based migration

- UniLab source: `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c` (`origin/main`).
- Package contract: `54 passed` with `uv run pytest -q`.
- Static typing: `0 errors` with `uv run pyright`.
- APPO one-iteration smoke (`4` envs): collector and learner completed, checkpoint
  `model_1.pt` written.
- FlashSAC one-iteration smoke (`4` envs): collector and learner completed,
  checkpoint `model_1.pt` written.
- Manager observation contracts: APPO flattened actor/critic obs `174`; FlashSAC
  actor obs `147`, privileged critic obs `174`.
- Fixed variants: eight XML sources compile with unique geom names and retain
  `simple="false"`.

## 2026-09-14 full training on pc823

Both runs used the synced checkout at
`/home/pc823/ws/unilabsim/sharpa_rl_unilab`, UniLab `b4e6b58f`, MuJoCo, seed
`1`, and `training.no_play=true`.

| Metric | APPO | FlashSAC |
| --- | ---: | ---: |
| Status / iterations | completed / 301 | completed / 5,371 |
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
| Checkpoint | `logs/full_runs/appo/model_301.pt` | `logs/full_runs/flashsac/model_5371.pt` |

Both runs completed their respective schedules. These measurements do not
establish a fair APPO–FlashSAC comparison.
