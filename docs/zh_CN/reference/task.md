# Sharpa 手内操作任务参考

## 任务身份

- 注册 rotation 环境：`SharpaInhandRotation`
- 注册 grasp-generation 环境：`SharpaInhandRotationGrasp`
- 生产后端：MuJoCo
- 动作维度：22 个手部关节
- Actor 历史：3 × 49 维 frame = 147 维
- HORA 特权 critic 历史：3 × 58 维 frame = 174 维
- Flat APPO 观测：3 × 58 维 frame = 174 维

## Manager-Based 所有权

注册类派生自 UniLab `ManagerBasedRlEnvCfg`，并调用
`make_manager_based_rl_env`。任务不派生环境子类，也不检查 backend model 或
qpos 布局。

| 关注点 | Term 模块 |
| --- | --- |
| 关节/执行器/传感器契约 | `terms/constants.py` |
| Term 参数校验 | `terms/validation.py` |
| 增量位置目标 | `terms/action.py` |
| 手/物体 reset | `terms/reset.py` |
| Grasp cache 加载 | `terms/cache.py` |
| 物理域随机化 | `terms/randomization.py` |
| 触觉与特权观测 | `terms/observation.py` |
| Drop termination | `terms/termination.py` |
| 旋转 reward | `terms/rewards.py` |
| 持续物体力 | `terms/disturbance.py` |
| Grasp 质量与记录 | `terms/grasp.py` |

Hydra owner 声明 scene entity、observation groups、events、actions、
terminations、rewards 与 recorders。Python 模块只包含有状态 term 逻辑。

## Fixed object variants

默认 catalog 会在以下尺度间轮询分配环境：

```text
0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

每个尺度都有 MJCF source，位于
`assets/robots/sharpa_wave/scene_scale_<scale>.xml`。reset 时，term 会采样
匹配的内置 grasp cache，并通过 Entity root state API 写入物体。

所有 variant 的自由物体都保留 `simple="false"`。manager robot XML 还为每个
body geom 提供唯一非空名称，这是 `mjbatch.VariantPack` 的要求。

## 随机化与观测

Reset events 负责：

- 手部执行器 P/D 增益倍率；
- 物体质量与质心偏移；
- 物体/elastomer/metal 摩擦 profile；
- 固定长度、均匀方向的 gravity；
- 通过 variant plan 固定的物体尺度身份。

Step event 负责质量缩放的衰减随机物体力。观测负责关节噪声、触觉力
clipping/smoothing/latency、actor/critic 历史以及特权物体属性。

包边界见[架构](../developer/architecture.md)。
