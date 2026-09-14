# Sharpa in-hand manipulation and HORA for UniLab

Sharpa Wave dexterous in-hand rotation now uses UniLab's current
**Manager-Based API**. The task owns action, observation, reset, event,
termination and reward terms while UniLab remains an external dependency and
owns the manager lifecycle and simulation backends. The former task-owned
legacy/direct `NpEnv` implementation and its compatibility factory have been
removed.

PPO, APPO and FlashSAC share the default HORA teacher setup and support the same
student distillation pipeline. Independent critics retain their V/Q structure.

## Install and validation

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco
uv run pytest
uv run pyright
```

Development uses the sibling checkout `../UniLab` as the `unilab` source; the
package dependency is still external (`unilab>=1.2.0,<1.3`), and
`unilab-rl>=1.2.0,<1.3` resolves from its release.

About 40 MB of robot XML, meshes and grasp caches are bundled in Git and the
Python package. `uv run sharpa-assets` prepares a writable local cache;
`SHARPA_RL_UNILAB_ASSET_CACHE` can select its directory.

## Train and evaluate

| Algorithm | Owner |
| --- | --- |
| PPO | `ppo/task/sharpa_inhand/mujoco.yaml` |
| APPO | `appo/task/sharpa_inhand/mujoco.yaml` |
| FlashSAC | `flashsac/task/sharpa_inhand/mujoco.yaml` |
| Grasp generation | `ppo/task/sharpa_inhand_grasp/mujoco.yaml` |

```bash
uv run sharpa-train --algo appo
uv run sharpa-train --algo flashsac
uv run sharpa-train --algo ppo
```

Append Hydra overrides for tuning; `--cfg` prints the composed Manager-Based
configuration. All algorithms share physical settings, observation preprocessing,
and fixed-scene quantitative evaluation. APPO defaults to 2048 environments,
305 learner updates and a checkpoint every 51 updates; its collector follows the
automatically selected learner device. PPO defaults to 2048 environments and
301 updates of 8 steps per environment (4,931,584 transitions). Its checkpoints
still use `budget.save_every`, defaulting to 1,000,000 transitions.
For a PPO or APPO sampling budget, explicitly set `algo.max_iterations=null
budget.transitions=N`; APPO checkpoints still use `algo.save_interval`.
FlashSAC uses a transition budget.
`sharpa-compare` explicitly selects sampling budgets for comparisons. `--nodr` uses
one common override; `+preset=throughput` selects a separate throughput experiment.

```bash
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
uv run sharpa-compare --output logs/comparison --seeds 1 2 3
```

Teacher and student runs show a UniLab Rich terminal panel and save TensorBoard
events alongside the full `metrics.jsonl`. Run `uv run tensorboard --logdir logs`
to view curves indexed by newly received transitions. Set `training.logger=none`
for the panel and JSONL only, or `training.logger=no_print` for JSONL only.
APPO logs every learner update and shows iteration progress/ETA; sampling counters
remain separate. Other runs use `budget.log_every` in transitions. Native TensorBoard
logging keeps slash metrics as-is and prefixes flat metrics with `train/`.
See [component reuse and validation](docs/migrations/issue-2-simplify.md).

Use `uv sync --extra mujoco --extra evaluation` for comparison figures. Add
`--smoke` to `sharpa-compare` for a short pipeline check. Old checkpoints require
retraining. The v2 runner supports one learner device and fresh training runs;
see the [protocol, migration and validation limits](docs/migrations/issue-2.md).

## Manager-Based correctness notes

- Object size DR uses UniLab's fixed model-variant catalog
  (`scene_scale_0.8.xml` … `scene_scale_1.5.xml`), not mutable geom-size DR.
- Every fixed-variant body geom has a unique name, which is required by
  `mjbatch.VariantPack`.
- Each variant retains `simple="false"` on the free object. This avoids the
  earlier reset-time mass/CoM `mj_setConst` sameframe crash.
- Tactile smoothing/latency, privileged state, incremental position targets,
  randomized actuator gains, object mass/CoM/friction/gravity and decaying
  object forces are explicit manager terms using the Entity facade.

See [architecture](docs/ARCHITECTURE.md), [validation](docs/VALIDATION.md) and
[HORA](docs/en/7-hora.md).
