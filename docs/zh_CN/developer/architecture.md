# 架构

`sharpa_rl_unilab` 是独立的 UniLab 任务包，不是仿真 runtime 或其文档树的
fork。

| 责任 | Owner |
| --- | --- |
| Manager 生命周期、Entity facade、fixed variants、backends | external UniLab |
| 任务 config 类与注册 | `tasks/sharpa_inhand/config.py`、`*_registry.py` |
| Manager terms | `tasks/sharpa_inhand/terms/` |
| Hydra owners | `conf/{ppo,appo,flashsac}/task/` |
| HORA 模型与蒸馏 | `algos/hora/`、`training/` |
| Robot MJCF variants、meshes、grasp caches | `assets/` |
| 文档 | `docs/` |

注册类扩展 `ManagerBasedRlEnvCfg`；registry factory 调用 UniLab 的
`make_manager_based_rl_env`。没有任务自有的 direct/legacy 环境子类，也没有
`adapt_legacy_factory` seam。

Hydra 引用具体 term 模块，例如 `terms.action` 与 `terms.observation`；
`terms/__init__.py` 刻意不导出兼容 facade。

物体尺度是 immutable fixed model variants。manager XML 为每个 body geom 提供
`mjbatch.VariantPack` 所需的唯一名称；每个 variant 都将自由物体标为
`simple="false"`，避免 reset-time mass/CoM 写入破坏 MuJoCo 编译后的
sameframe 假设。
