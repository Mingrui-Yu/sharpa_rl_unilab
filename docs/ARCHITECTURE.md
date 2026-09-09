# Architecture and extraction record

Roadmap: [UniLab #1547](https://github.com/unilabsim/UniLab/issues/1547).
This repository owns the complete Sharpa Wave in-hand task family and the HORA
teacher/student algorithms. UniLab supplies the general environment framework
and training adapters; unilab-rl supplies shared PPO/APPO/off-policy runtime
infrastructure; unisim supplies the backend interface and simulator
implementations.

## Ownership

Paths in the destination column are relative to
`src/sharpa_rl_unilab/`.

| Extracted concern | Destination |
| --- | --- |
| Sharpa in-hand rotation/grasp envs, rewards and domain randomization | `tasks/sharpa_inhand/` |
| HORA actor/critic models, PPO/APPO/SAC runtimes, distillation trainer | `algos/hora/` |
| PPO, APPO, SAC and HORA-distill owner YAMLs | `conf/ppo/`, `conf/appo/`, `conf/sac/`, `conf/hora_distill/` |
| HORA APPO play orchestration, distill config/entry, stage-2 checkpoint helpers, HORA playback sessions | `training/` |
| Grasp-cache collection script and init-DR construction benchmark | `tools/` |
| Sharpa Wave XML, 40 STL meshes and 9 grasp caches | `assets/` |

UniLab and unilab_rl remove the task-specific implementation, configurations,
tools, tests, registrations and documentation from their maintained trees.
They provide no compatibility copy or alias for the removed namespaces
(`unilab.tasks.manipulation.sharpa_inhand`, `unilab.training.hora_distill_config`,
`unilab.scripts.play_hora_appo`, `uni_rl.algos.hora`).

What stays at the original owners:

- Generic env/reset/step contracts, registry bootstrap and backend adapters
  remain in UniLab (`unilab.base`, `unilab.dr`, `unilab.training`).
- Generic PPO/APPO/SAC training scripts (`train_rsl_rl.py`, `train_appo.py`,
  `train_offpolicy.py`) and the interactive playback session cores
  (`unilab.visualization.interactive_playback`) remain in UniLab.
- Shared PPO and the generic off-policy runtime remain in unilab-rl. In
  particular, unilab-rl's `uni_rl.offpolicy.actor_adapter.OffPolicyActorAdapter`
  is the generic seam that lets an external package adapt a custom off-policy
  actor (grouped/privileged observations, custom actor builders) to the stock
  SAC runtime: the owner YAML's `algo.runtime_resolver` returns the adapter and
  the generic worker/learner call only the hooks it provides. HORA-SAC is one
  consumer of that mechanism, not a special case inside unilab-rl.

## Integration contracts

- The `unilab.tasks` entry point registers the Sharpa envs
  (`SharpaInhandRotation`, `SharpaInhandRotationGrasp`). Importing the package
  does not require a hard-coded Sharpa import in UniLab, and spawn-based
  workers rediscover the registration from entry-point metadata.
- The local CLI selects an owner configuration and preserves task/backend
  identity. PPO delegates to UniLab's shared `train_rsl_rl` launcher, APPO to
  `train_appo`, SAC to `train_offpolicy`; the HORA variants are selected purely
  through the owner YAML `runtime_resolver`/`class_name` fields.
- `sharpa-distill` composes the packaged `conf/hora_distill` tree and resolves
  teacher checkpoints through the teacher owner YAMLs, which stay the single
  source of truth for teacher hyperparameters.
- Robot XML, meshes and grasp caches ship in Git and the wheel. Asset
  preparation copies and sha256-verifies files in a writable cache; no Hugging
  Face runtime dependency is needed for this robot.

The integration follows UniLab's existing decisions:

- [ADR-0001: runtime and layer boundaries](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0001-runtime-model-and-layer-boundaries.md)
- [ADR-0003: task owner and config composition](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md)
- [ADR-0004: registry bootstrap](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0004-registry-bootstrap-contract.md)
- [ADR-0005: observation, critic, environment and IPC contracts](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md)

## Provenance and dependency selection

`MIGRATION_MANIFEST.json` records source commits, source paths and
pre-extraction SHA-256 hashes. Those hashes describe the source files before
import and integration changes; they are not hashes of the edited destination
files.

The separate [asset manifest](../src/sharpa_rl_unilab/assets/manifest.json)
records the pinned robot-data revisions and packaged file hashes.
[NOTICE.md](../NOTICE.md) preserves attribution.

During development, [pyproject.toml](../pyproject.toml) resolves UniLab and
unilab-rl through local editable path sources (`../UniLab`, `../unilab_rl`).
