# Architecture

`sharpa_rl_unilab` is a third-party UniLab task package, not a fork of the
simulation runtime.

| Responsibility | Owner |
| --- | --- |
| Manager lifecycle, Entity facade, fixed variants, backends | external UniLab |
| Hand constants and term validation | `tasks/sharpa_inhand/terms/{constants,validation,types}.py` |
| Incremental hand action | `tasks/sharpa_inhand/terms/action.py` |
| Hand/object reset and grasp-cache sampling | `tasks/sharpa_inhand/terms/{reset,cache}.py` |
| Physical DR and privileged state | `tasks/sharpa_inhand/terms/randomization.py` |
| Observations and termination state | `tasks/sharpa_inhand/terms/{observation,termination}.py` |
| Rewards and external-force disturbance | `tasks/sharpa_inhand/terms/{rewards,disturbance}.py` |
| Grasp-generation quality/recorder | `tasks/sharpa_inhand/terms/grasp.py` |
| Hydra owners and entity declarations | `conf/{ppo,appo,flashsac}/task/...` |
| HORA actor/critic and distillation algorithms | `algos/hora`, `training` |
| Explicit observation groups and terminal snapshots | `tasks/sharpa_inhand/teacher_env.py` |
| Native PPO/APPO/FlashSAC loss integration, collection and step budgets | `training/teacher_runtime.py` |
| Fixed scene manifests, quantitative evaluation and seed aggregation | `training/{evaluation,compare}.py` |
| Common task, observation, model and experiment definitions | `conf/common/` |
| Robot meshes, MJCF variants and grasp caches | `assets/` |

The registered config classes extend `ManagerBasedRlEnvCfg`; registry factories
call UniLab's `make_manager_based_rl_env`. There is no task-owned direct/legacy
environment subclass and no `adapt_legacy_factory` seam.

The Sharpa teacher adapter wraps this environment with underlying autoreset
disabled. It captures all four observation groups before resetting finished
rows. Policy inputs stay in observation dictionaries, never in `info`. RSL-RL
and uni_rl retain ownership of optimization losses; Sharpa owns their input
mapping, raw replay, normalization updates and versioned checkpoints. See the
[issue 2 protocol](migrations/issue-2.md) for the complete contract.

Hydra references concrete term modules such as `terms.action` and
`terms.observation`; `terms/__init__.py` deliberately exports no compatibility
facade.

Object scales are immutable fixed model variants. The manager XML gives every
body geom a unique name for `mjbatch.VariantPack`, and every variant marks the
free object body `simple="false"` so reset-time mass/CoM writes do not invalidate
MuJoCo's compiled sameframe assumptions.
