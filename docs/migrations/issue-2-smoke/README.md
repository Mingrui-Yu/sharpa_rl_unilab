# Pipeline smoke artifacts

These runs validate protocol-v2 execution, not algorithm performance. Each of
three algorithms uses training seeds 1, 2, 3; each teacher collects 128 new
transitions and each student collects 32. Evaluation uses the same eight fixed
scenes and a full 20-second window or real drop, without replacement episodes.

- `summary.json`: actual budgets, normalization counts, optimizer work, timings,
  parameter counts, checkpoint hashes, seed-level mean/std/bootstrap intervals,
  and per-scale aggregates. Training times include model/environment setup but
  exclude process startup, imports and CLI asset preparation.
- `episodes.csv`: all 144 original episode measurements, including signed angle,
  fixed-window speed, alive-time speed and raw return.
- `scenes.json`: frozen scale/cache row/DR values and episode/control-step RNG
  protocol. All algorithms and stages used this exact manifest hash.
- `learning_curves.png`: teacher/student displayed separately against collected
  transitions and wall time. Each plotted point belongs to one training seed;
  only final checkpoints were evaluated in this deliberately tiny smoke.

Full local checkpoints and configs are under `dist/issue-2-validated/`.
These are historical sampling-budget results. The current
`sharpa-compare --output NEW_DIRECTORY --device cuda:0 --smoke` uses two teacher
update rounds and native FlashSAC, so it does not reproduce these exact budgets.
Asynchronous APPO staging/reuse and timing can vary with scheduling. Three seeds
and these tiny budgets cannot support performance conclusions.
