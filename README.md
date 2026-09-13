# Sharpa in-hand manipulation and HORA for UniLab

Sharpa Wave dexterous in-hand rotation now uses UniLab's current
**Manager-Based API**. The task owns action, observation, reset, event,
termination and reward terms while UniLab remains an external dependency and
owns the manager lifecycle and simulation backends. The former task-owned
legacy/direct `NpEnv` implementation and its compatibility factory have been
removed.

Supported teacher entrypoints are PPO, HORA APPO and FlashSAC. HORA student
distillation remains available for an APPO teacher.

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
| HORA APPO | `appo/task/sharpa_inhand/mujoco_hora.yaml` |
| FlashSAC | `flashsac/task/sharpa_inhand/mujoco.yaml` |
| Grasp generation | `ppo/task/sharpa_inhand_grasp/mujoco.yaml` |

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo flashsac --sim mujoco training.no_play=true
uv run sharpa-train --algo ppo --sim mujoco training.no_play=true
```

Append Hydra overrides for tuning; `--cfg` prints the composed Manager-Based
configuration. For evaluation, point `algo.load_run` at a run directory and use
`algo.checkpoint=-1` for its newest checkpoint.

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run algo.checkpoint=-1
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=/absolute/path/to/flashsac/run algo.checkpoint=-1
```

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
