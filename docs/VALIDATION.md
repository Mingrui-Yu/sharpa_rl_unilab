# Validation record

Entries below describe the code and settings at their recorded dates. Current
usage and implementation contracts are in the [training guide](en/training.md)
and [architecture](en/architecture.md). Historical commands may no longer be supported.

## 2026-09-16 Common Actor and policy distribution

PPO/APPO and FlashSAC now share `HoraActor`, `PolicyDistribution` and
`PolicySample`, using thin interfaces to retain the installed RSL-RL/uni_rl
training loops. New training uses direct log-std, with no default clamp and an
optional saved `model.log_std_bounds`. Historical scalar and bounded-tanh std
parameters are isolated in the checkpoint compatibility path.

- `.venv/bin/pytest -q`: 88 passed; slow cases are run separately.
- `.venv/bin/pytest -q -m slow`: 13 passed. Ten train/save/load/evaluate/distill
  cases cover both std modes, PPO/APPO clip and tanh, and FlashSAC tanh. The two
  remaining environment/asset cases check reset/termination and bundled assets.
  An additional real-environment case verifies that forced zero actions report
  zero saturation and action differences despite saturated policy requests.
  Turning diagnostics on preserves the evaluated episode outcomes in all ten
  policy combinations.
- Distribution tests cover raw-sample probability ratios, pathwise gradients,
  current-policy tanh entropy against independent quadrature/finite differences,
  finite FP32 density math from FP16/BF16 inputs, KL and adaptive learning rate,
  state-dependent std updates, adapter cache isolation, optional clamp agreement
  across APPO current/target/sampling paths, and legacy teacher/student loading.
- Held-noise tests match the installed upstream implementation's noise, duration,
  counters and actions exactly with fixed seeds and per-environment resets,
  including maximum duration 1. Learning and deterministic inference cannot
  advance the runner's noise state. Collection uses one actor forward.
- CUDA APPO compiled minibatch forward/backward smoke passed for clip and tanh
  with state-dependent std; the normal training default remains uncompiled.
- Ruff lint, changed-file formatting, Mypy, Pyright and `git diff --check` passed.
  Full-repository formatting still reports the same eight untouched task files
  recorded in the previous validation entry. `uv build` produced the sdist and
  wheel successfully. Installed dependency code was not edited.

A held-noise inference comparison used identical weights and the original
FlashSAC std parameterization on an A800, batch size 2048, 12 alternating rounds
of 200 calls each. Median wall time including one network forward and device
synchronization was 0.82646 ms before and 0.79469 ms after migration (ratio
0.9616). This isolates actor/exploration overhead; it does not measure simulator
or IPC throughput. Native IPC and scheduling code is inherited unchanged.

The original G checkpoint was loaded with the new compatibility path and
reevaluated on its archived 240-scene manifest. Speed was 0.62808764 rad/s,
height exits 0/240 and joint-position second-difference RMS 0.007221203495 rad.
Speed, std, saturation and limit frequency match the archive; the RMS difference
is about 1.7e-12 rad. The recorded `1/10001/0` scene has bitwise-equal action,
joint-position, target and angle arrays compared with its archived video trace.

The new G-configuration run used seed 1, 2048 environments, APPO tanh,
state-independent log-std initialized at 1, no log-std clamp, and entropy
coefficient 0.01. It completed exactly 8,208,384 collected/received transitions,
501 policy versions, 10,020 optimizer updates and 326,041,600 training samples,
matching the original G budgets and update counts. Training took 2043.34 seconds.
Both checkpoints were evaluated on the same 240-scene manifest.

| Metric | Original G | New log-std G |
| --- | ---: | ---: |
| Joint-position second-difference RMS (rad) | 0.00722120 | 0.00671349 |
| Fixed-window rotation speed (rad/s) | 0.628088 | 0.597009 |
| Height exits | 0/240 | 8/240 |
| Mean std | 0.459803 | 0.475110 |
| Action saturation | 0.0241% | 0.1135% |
| Adjacent action difference RMS | 0.240156 | 0.219893 |
| Target at joint limit | 3.2487% | 6.4318% |

Motion RMS is 7.0% lower and speed is 4.9% lower; survival worsened. The early
exits must be considered when interpreting the motion metrics. This single-seed
result does not establish cross-seed stability or preserve the original G's
survival performance. No reward tuning or A–H matrix rerun was used to alter this
comparison.

Both videos use the preselected scene `1/10001/0`, the same camera, 400 frames
and 20 fps. A side-by-side recording is saved as
`logs/actor_distribution/G_comparison.mp4`; individual videos, trajectories,
checkpoints and evaluation JSON are in the same local run directory.
[Recorded metrics and dependency versions](validation/actor-distribution-20260916.json)
include checkpoint/manifest hashes and the isolated noise benchmark. The run
used Torch 2.8.0+cu128, RSL-RL 5.5.0 and uni_rl/UniLab 1.2.0.

The new run can be reproduced with the same settings (individual training
trajectories need not be bitwise identical):

```bash
uv run sharpa-train --algo appo algo.seed=1 \
  model.action_mapping=tanh model.std_mode=state_independent model.initial_std=1.0 \
  algo.algorithm.entropy_coef=0.01 training.num_envs=2048 \
  algo.max_iterations=null training.max_transitions=8208384 algo.save_interval=0 \
  training.no_play=true training.logger=no_print \
  training.log_dir=logs/actor_distribution/appo_G_log_std_seed1 training.device=cuda:0
```

Use a new output directory when repeating the command. Evaluation uses
`evaluation.diagnostics=true` and the archived manifest copied to
`logs/actor_distribution/scenes.json`.

AMP TODO: the common distribution promotes critical arithmetic to FP32, while
MLPs retain native AMP. A broader audit of autocast boundaries, small std and
large raw samples across FP16/BF16 devices remains separate from this migration.
The shared KL also removes upstream additive numerical offsets; identical
policies now have zero KL instead of triggering an artificial learning-rate
increase. Existing scheduling thresholds and adjustment factors remain intact.

## 2026-09-15 Optional comparison stages and seed count

`tools/compare_rl_algo.py` replaces `tools/compare.py` and `tools/compare_teacher.py`.
`sharpa-compare` uses Hydra defaults with optional `--distill`, `--eval` and
`--num-seeds N` (default 1). All teachers finish before any student starts;
evaluation follows training and invokes the `sharpa-eval` entry point with a shared
scene manifest. Per-checkpoint JSON results remain available; the new launcher
does not aggregate or plot results. Earlier comparison commands below are historical.

- 17 focused tests passed, with one slow asset test deselected. Nine launcher tests
  mock subprocesses to cover stage combinations, multiple seeds, default arguments,
  shared manifests, invalid seed counts and stopping on teacher failure.
- Ruff lint and format checks for changed Python files, Mypy and Pyright for the
  new launcher, and diff whitespace checks passed.
- Reinstalled the editable package without dependencies to refresh `sharpa-compare`.
  This change did not run full teacher/student training or physical evaluations.

## 2026-09-15 Focused regression suite

The suite now has 40 cases instead of 102. It drops default-parameter snapshots,
source-token bans, upstream-only checks, terminal UI details and repeated
training matrices. Retained regressions cover observation/reset boundaries,
gradient isolation, normalization, APPO policy consistency, FlashSAC transport,
checkpoint compatibility and evaluation statistics.

- Default `pytest` / `make test`: 35 fast tests passed (about 14 seconds).
- Explicit `pytest -m slow` / `make test-slow`: 5 tests passed (about 38 seconds),
  including one real train/checkpoint/evaluate/distill case per algorithm,
  MuJoCo autoreset/termination and bundled asset compilation/cache repair.
  FlashSAC ran on CUDA; the smoke skips when CUDA is unavailable.
- Ruff lint, test formatting, Mypy, Pyright and diff whitespace checks passed.
  The full format check still reports eight untouched source files.
- Checks used the existing virtualenv and absolute `PYTHONPATH` entries for
  this worktree's `src` and the sibling UniLab checkout. `make test-all` combines
  static checks with both test suites.

## 2026-09-15 Training layout and native FlashSAC

`sharpa-compare` now lives in `tools/compare.py` and uses each algorithm's
configured budget. FlashSAC training uses only the native DoubleBuffer runner;
sampling-budget overrides are rejected before training. PPO/APPO sampling limits
and student transition budgets remain supported. Device resolution belongs to
`training/configuration.py`; run metadata and JSON output belong to
`training/logging.py`. The old `train_hora_distill` forwarding module is removed.

- 102 tests passed in separate non-slow (81) and MuJoCo/CUDA (21) runs, including
  three native FlashSAC warmup/checkpoint/evaluation/distillation cases.
- After strengthening the former sampling-budget checkpoint fixtures, all 18
  configuration-migration and teacher-model tests passed again. CPU student
  training still accepts those checkpoints.
- Ruff lint, Mypy, Pyright (including `tools/compare.py`), changed-file format
  checks and diff whitespace checks passed. The full format check reports the
  same nine untouched files with existing formatting issues.
- Lockfile validation, source distribution and wheel builds passed. The wheel
  includes the new comparison module/entrypoint and excludes both retired paths.
  Reinstalling the editable package refreshes the `sharpa-compare` command.
- The installed `sharpa-compare --smoke` completed PPO/APPO/FlashSAC × seeds
  1/2/3 on CUDA, with one shared scale-1.0 scene. All nine teachers completed two
  update rounds; all nine students collected 32 transitions. The 18 evaluations,
  six aggregate groups and plot were produced successfully. FlashSAC used the
  native DoubleBuffer runner with its `torch_copy_stream` replay transfer.
  These short runs validate execution, not learning performance.

## 2026-09-14 Issue 2: common HORA protocol v2

Implementation and checks are confined to Sharpa. UniLab remains at
`b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`, with a clean working tree.
The [architecture](en/architecture.md) describes current interfaces; the original
optimizer baselines remain in the historical snapshot linked below.

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

Historical summaries, episode measurements, fixed scenes and plots remain in
the [smoke snapshot at d904331](https://github.com/unilabsim/sharpa_rl_unilab/tree/d904331cf9c9ec291634f4c8e4d9d8b30f04f074/docs/validation/issue-2-smoke).
Locally, inspect the summary with `git show d904331:docs/validation/issue-2-smoke/summary.json`.
Full checkpoints/configs were local artifacts under `dist/issue-2-validated/`.
The summaries include per-scale and training-seed bootstrap statistics.
These are **pipeline validation results**, not a ranking: no 5M teacher / 100M
student performance experiment was run, and old checkpoints require retraining.

The later native-runner smoke at commit `61a5fd7` used the command below with
two teacher update rounds. This historical command is no longer supported;
current commands are in the training guide. The 128-transition results above
describe the earlier sampling-budget implementation.

```bash
uv sync --extra mujoco --extra evaluation
uv run sharpa-compare --output dist/issue-2-validated --device cuda:0 --smoke
```

For this development worktree, checks used the existing virtualenv interpreter
and absolute `PYTHONPATH` entries for this worktree's `src` and UniLab's `src`.
Neither dependency sources nor the original worktree's user edits were changed.

## Issue 2 implementation history

The [snapshot at 61a5fd7](https://github.com/unilabsim/sharpa_rl_unilab/tree/61a5fd73b5675dd6db750157354aa1dd65df3c7a/docs/migrations)
retains the implementation notes, original YAML/optimizer baselines and raw
before/after measurements. Locally, inspect it with
`git show 61a5fd7:docs/migrations/issue-2-simplify.md`.

- The component simplification reused native logging and rollout storage;
  50 tests passed at that revision. Later APPO and native FlashSAC changes have
  separate validation entries above.
- Single-run comparisons over 256 and 4096 transitions checked execution and
  normalization counts. The longer run measured post-warmup throughput over only
  8–14 seconds; it does not establish production throughput or learning quality.
- The smoke snapshot at `d904331` retains the three-seed pipeline evidence.
  Full training checkpoints were local artifacts, not committed files.
  These runs do not rank the algorithms.

## Historical protocols

The runs below use the pre-issue-2 observation and training protocols. Their
Actor inputs, sample budgets and evaluation procedures differ. They are
historical execution records, not an algorithm ranking or evidence of relative
sample efficiency. See [implementation history](#issue-2-implementation-history).

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
