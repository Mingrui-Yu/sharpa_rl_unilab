# 任务与环境指南

## 机器人与目标

任务控制一只 22 关节 Sharpa Wave 手。被操作物体是自由圆柱。策略在保持物
体稳定的同时，让物体绕目标轴旋转。

注册环境为：

- `SharpaInhandRotation`：策略训练与评估。
- `SharpaInhandRotationGrasp`：抓取状态收集。

## 观测与动作

| 契约 | 大小 |
| --- | ---: |
| 手部动作 | 22 |
| Actor frame | 49 |
| Actor 历史 | 147 |
| Flat APPO 观测 | 174 |
| HORA 特权 critic 历史 | 174 |

Actor 观测包含手部状态、位置目标历史与触觉历史。特权 critic 还观察部署
actor 不可用的物体与域状态。

## 训练变化

每个 episode 随机采样：

- 手部执行器 P/D 增益倍率；
- 物体质量与质心偏移；
- 物体、elastomer 与 metal 摩擦；
- 重力方向；
- 衰减的外部物体力。

物体尺寸不在运行时修改。环境对这些尺度使用固定 MJCF variants：

```text
0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

这能避免 mass/CoM 随机化期间的 MuJoCo sameframe 错误，并为策略提供稳定
的多尺度训练课程。

## Reward 行为

Reward 鼓励目标轴物体旋转，并惩罚：

- 物体线速度；
- 手部姿态偏差；
- 估计力矩；
- 机械功；
- 物体偏离锚点。

当物体掉出 reset-height 区间或到达时间限制时，episode 结束。

## 内置资产

包内包含 Sharpa Wave MJCF、碰撞 mesh、视觉 mesh，以及每个物体尺度一个
grasp cache。需要时 `uv run sharpa-assets` 会准备可写修复缓存；常规运行
不会下载资产。

设置 `SHARPA_RL_UNILAB_ASSET_CACHE` 可以选择自定义 asset-cache 目录。
