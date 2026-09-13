# Sharpa Wave 手内旋转

任务使用 UniLab Manager-Based API 声明：YAML 拥有 scene entity、observation
groups、增量手部 action、reset/event terms、termination 与 reward；Python term
只保留任务状态，并通过 Entity facade 访问仿真，不读取 backend 内部对象。

## Owner

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/flashsac/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`

当前 fixed object-size variant 生产后端为 MuJoCo。抓取缓存随包提供，仅当
抓取策略或模型变更时需要重建。

```bash
uv run sharpa-train --algo ppo --sim mujoco training.no_play=true
uv run sharpa-train --algo appo --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo flashsac --sim mujoco training.no_play=true
```

## 固定物体尺度

默认 rotation catalog 将环境轮询分配到 0.8、0.9、1、1.1、1.2、1.3、1.4、1.5。
每个尺度都有独立 MJCF source；reset term 采样对应 grasp cache，并通过公开
Entity API 写物体 root state。

抓取生成每次只运行一个尺度：

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

## 校验

```bash
uv run pytest
uv run pyright
uv run sharpa-train --algo appo --sim mujoco --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

完整训练前，先对 APPO 与 FlashSAC 各执行一次 one-iteration smoke train，覆盖
collector/learner 全链路。
