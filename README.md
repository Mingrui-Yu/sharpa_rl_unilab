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

Append Hydra overrides for tuning; `--cfg` prints the composed configuration.
`conf/common/sharpa_inhand.yaml` defines the shared task, model and runtime settings.
All three teachers use 2048 environments, `training.device=cuda:0`, and 4 Torch
intra-op / 1 inter-op threads per learner or independent collector process.
APPO's collector follows the learner unless `algo.collector_device` is set.
Use `training.device=null` for automatic CUDA → MPS → CPU selection.

APPO/PPO default to 501 update rounds, FlashSAC to 3000. Every teacher saves each
50 rounds (`algo.save_interval`; 0 disables intermediate saves), logs every round
and writes `teacher_final.pt` on completion. Round counts do not imply equal
sample counts or compute. For a sampling limit, set
`algo.max_iterations=null training.max_transitions=N`; saving remains by round.
`sharpa-compare` explicitly selects sampling limits and evaluates final models.
Training itself does not schedule quantitative evaluation.

FlashSAC uses UniLab's DoubleBuffer runner with AMP disabled. A sampling limit
selects its synchronous compatibility runner; see
[FlashSAC runtime details](docs/migrations/issue-2-flashsac-native.md).
Both paths normalize clean174 Q inputs using fresh-sample statistics shared by
current and target Q networks. V/Q architectures and algorithm settings remain distinct.

Tune resources or individual randomization settings with explicit overrides:

```bash
uv run sharpa-train --algo flashsac training.num_envs=1024 training.torch_threads.learner_num_threads=8
uv run sharpa-train --algo appo training.device=cuda:0 algo.collector_device=cpu
```

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
Teachers log every round; iteration-limited runs show iteration progress/ETA.
Student logs use `distillation.log_every` (default 10,000 transitions).
Native TensorBoard logging keeps slash metrics as-is and prefixes flat metrics with `train/`.
See [component reuse and validation](docs/migrations/issue-2-simplify.md).

Use `uv sync --extra mujoco --extra evaluation` for comparison figures. Add
`--smoke` to `sharpa-compare` for a short pipeline check. Supported v2 checkpoints migrate their configuration
paths on load and retain saved model/environment semantics. Older incompatible
formats require retraining. The runner supports one learner device and fresh training runs;
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
