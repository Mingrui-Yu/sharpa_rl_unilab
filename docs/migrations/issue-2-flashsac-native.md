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

FlashSAC uses `CleanQ` to normalize clean174 observations before every
current/target Q call, including actor updates. Statistics are shared by current
and target Q, separate from actor statistics, and updated once per fresh current
observation. Actions and replay storage remain raw; Polyak updates affect only
Q parameters. Native internal normalization and reward normalization remain intact.

`algo.save_interval=50` saves every 50 completed update rounds, using
`teacher_iteration_N.pt`; 0 disables intermediate snapshots. The final drained
snapshot is `teacher_final.pt`. Snapshots retain normalization buffers and native
optimizer/scheduler state. Training does not schedule evaluation; use the separate
evaluation entry or `sharpa-compare`'s explicit final evaluation.

## Supported budgets and devices

FlashSAC requires a positive `algo.max_iterations` and
`training.max_transitions=null`. The synchronous sampling-budget runner and its
local replay have been removed. Sampling budgets are rejected before training;
they are not converted into nominal update rounds. Actual collected/received
counts remain in logs and checkpoints.

Teacher training requires CUDA or MPS for native device replay. Model unit tests,
checkpoint evaluation and student distillation can still run on CPU.
Existing supported v2 checkpoints remain loadable for evaluation and distillation,
including checkpoints created by the former synchronous runner.

`sharpa-compare` uses each algorithm's configured budget and evaluates on shared
scenes. It records actual sampling and wall time without enforcing equal costs.
`--smoke` uses two teacher update rounds, reduced batches/replay and 32 student
transitions; it exercises the native FlashSAC pipeline.

For a 5371-round run:

```bash
sharpa-train --algo flashsac \
  training.num_envs=1024 \
  algo.max_iterations=5371 \
  algo.save_interval=896 \
  algo.use_amp=false
```

Actual collection depends on the native pipeline; rounds do not specify an exact
transition count. PPO/APPO retain their optional sampling-budget mode.

## Validation

`tests/test_flashsac_runtime.py` checks the defaults, inherited native loop,
terminal actor/privilege/critic transport, unsupported-budget rejection and thread
resolution. CUDA integration runs three native rounds with eight environments
for both ordinary startup and extended warmup, checks actual scheduler update
counts and normalization counters in all saved checkpoints, and loads the
final model through evaluation and distillation. PPO/APPO pipeline tests retain
non-multiple transition-budget coverage. CPU learner tests retain gradient
isolation, raw privilege encoding, Q normalization and checkpoint round trips.

For the configuration migration and current verification scope, see
[RL alignment](issue-2-rl-align.md). Earlier smoke reports retain their historical settings.
