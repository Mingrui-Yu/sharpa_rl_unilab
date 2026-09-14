# FlashSAC HORA: native update-round training

The default rotation task now uses 2048 environments, 10000 completed update
rounds, and FP32 (`algo.use_amp=false`). Each round retains 14 critic updates and
7 actor/temperature updates, batch size 2048 and two collection steps per
environment. Warmup does not count as an update round. The nominal collection
volume is 40,960,000 transitions; actual collection is recorded independently.

`training/flashsac_runtime.py` injects `TeacherFlashLearner` into UniLab's
`DoubleBufferOffPolicyRunner`. The inherited `learn` method owns the collector
process, learner-side inference, bounded replay ingress, device replay, prefetch
and optimizer loop. Sharpa only adapts environment observations, fresh-data
statistics, checkpoint format and metrics. No dependency files are modified.

The transport has two groups: `obs=[actor147, current_priv9]` and `critic174`.
Both reset observations and terminal observations use this layout. Only new
committed replay rows update HORA normalization, including during warmup;
replayed minibatches do not update it. Collection and learning share one actor
and its buffers. `received` counts rows incorporated into these statistics;
`collected` follows the native committed replay counter, including tail rows
that may not have reached normalization at the final update. These counters can
differ, and neither is inferred from the iteration count.

Thread settings use the native `resolve_torch_thread_runtime`,
`apply_torch_thread_runtime` and collector startup environment context. The
legacy `hardware.torch_threads` is null for FlashSAC; use
`training.torch_threads.learner_num_threads` and `collector_num_threads` for
explicit tuning. Auto selects at most 8/4 operation threads, one interop thread
per role and at most two compile threads. Compilation remains disabled.

`budget.save_every` continues to mean transitions (default 1,000,000). The
adapter checks actual collection after every completed round, saving at the
first crossed threshold. Intermediate files are `teacher_iteration_N.pt` and
the final file is `teacher_final.pt`; both retain the HORA contract, model,
normalization buffers and native optimizer/scheduler state. Optional evaluation
snapshots are selected by transition thresholds and evaluated after the native
pipeline closes, avoiding collector timeouts during evaluation.

## Sampling-budget compatibility and dependency gaps

The installed native runner has no exact transition-budget stop/drain API.
Explicit `algo.max_iterations=null budget.transitions=N` therefore selects the
existing synchronous compatibility runner, which rounds only to a complete
vector environment step. The CLI announces this mode. `sharpa-compare` clears
`algo.max_iterations` for every algorithm and continues to use equal sampling
budgets. It does **not** benchmark the native FlashSAC pipeline. CPU-only
FlashSAC sampling tests also use this path; native device replay requires CUDA
or MPS.

An exact sampling budget in the native pipeline would require a collector
step limit and a runner stop/drain interface that handles warmup and partial
final rounds. Dividing a transition budget by nominal samples per round is
insufficient. These interfaces are not copied or patched into Sharpa.

The parameter-alignment command from journal 06 section 6 is now:

```bash
sharpa-train --algo flashsac \
  hardware.num_envs=1024 \
  algo.max_iterations=null \
  budget.transitions=10999808 \
  budget.save_every=1835008 \
  algo.use_amp=false
```

This explicitly uses the synchronous sampling-budget path. For the native
pipeline's nominal 5371-round comparison, use
`hardware.num_envs=1024 algo.max_iterations=5371 budget.save_every=1835008`;
actual collection can differ. Neither comparison restores main's original
actor architecture.

## Validation

`tests/test_flashsac_runtime.py` checks the defaults, inherited native loop,
terminal actor/privilege/critic transport, budget exclusivity and auto thread
resolution. CUDA integration runs three native rounds with eight environments
for both ordinary startup and extended warmup, checks actual scheduler update
counts and normalization counters in all saved checkpoints, and loads the
final model through evaluation and distillation. Existing pipeline tests cover
non-multiple transition budgets through the synchronous compatibility path.

Local validation on 2026-09-15: all four native tests and all eleven existing
teacher pipeline tests passed. Collector instrumentation measured 4 operation
threads, 1 interop thread and `OMP_NUM_THREADS=4`; the learner used 8/1. Ruff
lint, Mypy, Pyright and formatting checks for changed files passed. The broader
non-slow suite had 57 passes and 9 failures from pre-existing uncommitted
PPO/APPO iteration and playback defaults (501 versus old test expectations,
`record` versus `auto`). The full formatting check also reports nine unchanged
files. Those unrelated working-tree changes and formatting are preserved.
