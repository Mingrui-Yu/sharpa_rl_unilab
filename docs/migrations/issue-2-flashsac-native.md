# FlashSAC HORA: native update-round training

The default rotation task now uses 2048 environments, 3000 completed update
rounds, and FP32 (`algo.use_amp=false`). Each round retains 14 critic updates and
7 actor/temperature updates, batch size 2048 and two collection steps per
environment. Warmup does not count as an update round. The nominal collection
volume is 12,288,000 transitions; actual collection is recorded independently.

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

Thread settings use the shared `training.torch_threads` parser and collector
startup environment context. Both learner and collector default to 4 intra-op
and 1 inter-op threads; explicit values or `auto` can be used for tuning.
Compilation remains disabled.

Both FlashSAC paths use `CleanQ` to normalize clean174 observations before every
current/target Q call, including actor updates. Statistics are shared by current
and target Q, separate from actor statistics, and updated once per fresh current
observation. Actions and replay storage remain raw; Polyak updates affect only
Q parameters. Native internal normalization and reward normalization remain intact.

`algo.save_interval=50` saves every 50 completed update rounds, using
`teacher_iteration_N.pt`; 0 disables intermediate snapshots. The final drained
snapshot is `teacher_final.pt`. Snapshots retain normalization buffers and native
optimizer/scheduler state. Training does not schedule evaluation; use the separate
evaluation entry or `sharpa-compare`'s explicit final evaluation.

## Sampling-budget compatibility and dependency gaps

The installed native runner has no exact transition-budget stop/drain API.
Explicit `algo.max_iterations=null training.max_transitions=N` therefore selects the
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
  training.num_envs=1024 \
  algo.max_iterations=null \
  training.max_transitions=10999808 \
  algo.save_interval=896 \
  algo.use_amp=false
```

This explicitly uses the synchronous sampling-budget path. For the native
pipeline's nominal 5371-round comparison, use
`training.num_envs=1024 algo.max_iterations=5371 algo.save_interval=896`;
actual collection can differ. Neither comparison restores main's original
actor architecture.

## Validation

`tests/test_flashsac_runtime.py` checks the defaults, inherited native loop,
terminal actor/privilege/critic transport, budget exclusivity and thread
resolution. CUDA integration runs three native rounds with eight environments
for both ordinary startup and extended warmup, checks actual scheduler update
counts and normalization counters in all saved checkpoints, and loads the
final model through evaluation and distillation. Existing pipeline tests cover
non-multiple transition budgets through the synchronous compatibility path.

For the configuration migration and current verification scope, see
[RL alignment](issue-2-rl-align.md). Earlier smoke reports retain their historical settings.
