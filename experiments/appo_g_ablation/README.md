# APPO G ablation

This is an experiment-only entrypoint for the journal
`../sharpa_rl_unilab/dist/journals/10-plan-appo-g-ablation.md`.
Production model construction, training losses, and installed dependencies are unchanged.

The artifacts live in this worktree's `logs/appo_g_ablation/` (ignored by Git).
Do not rerun `freeze.py` over an existing experiment directory: preserve that directory
first. The frozen original source and historical checkpoints are prerequisites for R.
R is explicitly a reconstruction, because the historical run did not archive a complete
contemporaneous source hash. Its actor/loss/collection code is frozen from the surviving
original worktree; all groups use the same current runner.

The fixed core order is R, A, B, D, C, N. All groups use seed 1, collector seed 2,
2048 environments, 501 iterations, GPU 0, CPUs 0–127, and four Torch threads per process.
The initial 16-CPU attempt is a throughput calibration only; it was interrupted before
any evaluation, archived under `calibration16/`, and excluded from the matrix.

Use the existing sibling virtualenv without changing its installations:

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/freeze.py
# Short R/A/C runs, real-rollout capture, and audit precede the matrix.
APPO_CAPTURE_BATCH=1 taskset -c 0-127 ../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/train.py A --iterations 2 --name smoke_A
taskset -c 0-127 ../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/audit.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/run_matrix.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/run_evaluations.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/audit_residual.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/verify.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/analyze.py
```

The matrix driver refuses to overwrite run directories. Every command and exit code is
archived. The evaluation driver starts only after the last training process has exited;
evaluations run on CPU, using the existing evaluator and original 240-scene manifest.
The 24 intermediate scenes are frozen in `manifest.json` before training.

`adapter.py` replaces constructors only in the experiment process. A/B instantiate the
existing `LegacyScalarStd` directly, without changing the production restriction or
fabricating a checkpoint. R uses frozen original methods. A/B/C/D/N retain the upstream
loss. KL changes only the scheduling metric. Both shadow KLs and std bounds are recorded
without sampling, and minibatch diagnostics are buffered until the update ends.

Artifacts include per-minibatch KL/LR/gradient norm, received packet order and behavior
versions, staging versions, 22 std values, save timing, counters, initial tensor hashes,
and learner/collector RNG states. `runtime_metrics.jsonl` is separate from the native
logger's `metrics.jsonl`. The passive evaluation trajectories permit common-survival
window and full-survivor-subset comparisons without changing environment actions.

`audit/fixed_data_report.json` retains the default tolerance results, including expected
FP32 failures in trained-state target log-probability/V-trace values. The additional
`density_residual.json` isolates their original `Normal.log_prob` operation order;
these failures are not erased by silently relaxing the reference tolerance.

After the six core evaluations, four conditional runs were selected and recorded in
`manifest.json`: C/D replaying the frozen C learning-rate sequence, A restoring only
learner Gaussian density order, and A restoring only target density order. This uses
the full four-run allowance. The replay reference was specified before evaluation;
its values and hash are in `audit/lr_reference_C.json`. The fixed-LR logs' `branch`
is the native counterfactual; `lr_after` is the applied learning rate.

The conditional workflow and final report generation are:

```bash
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/run_additional.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/run_additional_evaluations.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/verify.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/analyze_additional.py
# Review all results and write logs/appo_g_ablation/conclusions.md first.
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/finalize_artifacts.py
../sharpa_rl_unilab-issue-2/.venv/bin/python experiments/appo_g_ablation/write_report.py
```

`finalize_artifacts.py` verifies frozen production/dependency/training-entrypoint sources
and archives hashes for checkpoints and evaluation JSON. `write_report.py` preserves
journal sections 1–9 and writes the final results to section 10 and
`docs/validation/appo-g-ablation-20260916.md`, with compact JSON evidence alongside.
This is a single training seed experiment; evaluation scenes are not training repeats.

The user subsequently requested a separate N follow-up and explicitly selected restoring
only `log(std / old_std + 1e-5)` in the KL, retaining the new formula's normalized-square
operation order. `run_kl_log.py` runs this `N_kl_log_epsilon` condition and its offline
evaluations; `analyze_kl_log.py` verifies and compares it with N, R, and C. After reviewing
the results, `report_kl_log.py` appends journal section 11 from
`logs/appo_g_ablation/kl_log_conclusions.md`. The separate `kl_log_manifest.json` preserves
the completed original matrix and its four-run budget.

An earlier constant-bias interpretation was stopped at iteration 53 after the user's
clarification. Its `N_kl_bias_seed1` artifacts and `kl_bias_manifest.json` are marked aborted,
have no evaluations, and are excluded from comparisons. They are not a checkpoint source
for the final logarithmic-term intervention.

The later `train_zero_kl.py` experiment keeps the current production APPO defaults
and exact KL, disabling only the skip after reference synchronization. The first
zero KL therefore increases LR by 1.1, subject to the existing upper limit. It
records every scheduling decision without changing losses or drawing random numbers.
`--control` retains the production skip for an instrumentation check; `--iterations`
defaults to 501. Use a fresh output directory:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  taskset -c 0-127 ../sharpa_rl_unilab-issue-2/.venv/bin/python \
  experiments/appo_g_ablation/train_zero_kl.py logs/exact_zero_kl_run
```

This is a process-local experiment, not a production configuration change. The
checkpoint's `experiment` fields record provenance; reproducing the training
intervention requires this entrypoint. Evaluation uses the normal checkpoint loader.
The completed run and comparison artifacts are under
`logs/appo_exact_zero_increase_20260916/`.
