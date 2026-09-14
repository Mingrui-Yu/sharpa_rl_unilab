# HORA teacher and student

PPO, APPO and FlashSAC now train the same HORA Actor by default. Each rotation
algorithm has one `mujoco.yaml`; no HORA profile is needed. A teacher consumes
147 measurable observation values plus a separate current 9-D privilege vector.
The independent critic receives 174 clean values. FlashSAC adapts the common
Actor while retaining its native distributional double Q and temperature losses.

```bash
uv run sharpa-train --algo appo
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

Distillation accepts all three teacher algorithms. It freezes the inherited
policy and base normalization, trains only the 30-frame history encoder by
latent MSE, and acts with the updated student's deterministic action after each
update. Deployment needs only the 147-D base observation and [N,30,49] history.

Checkpoints contain the full run configuration and normalization statistics.
Old shared HORA, flat PPO and native FlashSAC checkpoints require retraining.
Evaluation uses fixed scenes and complete 20-second windows; video replay is
not a quantitative evaluation.

See the [protocol and migration record](../migrations/issue-2.md).
