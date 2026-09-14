# Sharpa Wave in-hand rotation

The task is declared with UniLab's Manager-Based API. YAML owns the scene
entity, observation groups, incremental hand action, reset/event terms,
terminations and rewards; Python manager terms retain only task-specific state
and use the Entity facade rather than backend internals.

## Owners

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco.yaml`
- `conf/flashsac/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`

MuJoCo is the current production backend for the fixed object-size variant
plan. Grasp caches are bundled under `assets/caches/`; regenerate one only when
the grasp policy or model changes.

```bash
uv run sharpa-train --algo ppo --sim mujoco
uv run sharpa-train --algo appo --sim mujoco
uv run sharpa-train --algo flashsac --sim mujoco
```

## Fixed object scales

The default rotation catalog assigns environments round-robin to scales 0.8,
0.9, 1, 1.1, 1.2, 1.3, 1.4 and 1.5. Each scale has a compiled MJCF source.
A reset term samples the matching grasp cache and writes the object root
through the public Entity API.

Grasp generation intentionally uses one scale per run. The helper overrides the
single fixed variant and the recorder target:

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

## Validation

Useful checks include:

```bash
uv run pytest
uv run pyright
uv run sharpa-train --algo appo --sim mujoco --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

A one-iteration smoke train for both APPO and FlashSAC verifies the complete
collector/learner path before launching a full run.
