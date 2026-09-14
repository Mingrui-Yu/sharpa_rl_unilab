# Architecture

`sharpa_rl_unilab` is an independent UniLab task package, not a fork of the
simulation runtime or its documentation tree.

| Responsibility | Owner |
| --- | --- |
| Manager lifecycle, Entity facade, fixed variants, backends | external UniLab |
| Task config classes and registries | `tasks/sharpa_inhand/config.py`, `*_registry.py` |
| Manager terms | `tasks/sharpa_inhand/terms/` |
| Hydra owners | `conf/{ppo,appo,flashsac}/task/` |
| HORA models and distillation | `algos/hora/`, `training/` |
| Robot MJCF variants, meshes, grasp caches | `assets/` |
| Documentation | `docs/` |

The registered classes extend `ManagerBasedRlEnvCfg`; registry factories call
UniLab's `make_manager_based_rl_env`. There is no task-owned direct/legacy
environment subclass and no `adapt_legacy_factory` seam.

Hydra references concrete term modules such as `terms.action` and
`terms.observation`; `terms/__init__.py` deliberately exports no compatibility
facade.

Object scales are immutable fixed model variants. The manager XML gives every
body geom a unique name for `mjbatch.VariantPack`, and every variant marks the
free object body `simple="false"` so reset-time mass/CoM writes do not invalidate
MuJoCo's compiled sameframe assumptions.
