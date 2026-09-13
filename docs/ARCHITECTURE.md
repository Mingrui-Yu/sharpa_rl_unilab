# Architecture

`sharpa_rl_unilab` is a third-party UniLab task package, not a fork of the
simulation runtime.

| Responsibility | Owner |
| --- | --- |
| Manager lifecycle, Entity facade, fixed variants, backends | external UniLab |
| Sharpa action / observation / reset / event / reward terms | `tasks/sharpa_inhand/manager_terms.py` |
| Hydra owners and entity declarations | `conf/{ppo,appo,flashsac}/task/...` |
| HORA actor/critic and distillation algorithms | `algos/hora`, `training` |
| Robot meshes, MJCF variants and grasp caches | `assets/` |

The registered config classes extend `ManagerBasedRlEnvCfg`; registry factories
call UniLab's `make_manager_based_rl_env`. There is no task-owned direct/legacy
environment subclass and no `adapt_legacy_factory` seam.

Object scales are immutable fixed model variants. The manager XML gives every
body geom a unique name for `mjbatch.VariantPack`, and every variant marks the
free object body `simple="false"` so reset-time mass/CoM writes do not invalidate
MuJoCo's compiled sameframe assumptions.
