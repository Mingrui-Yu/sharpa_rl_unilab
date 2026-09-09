# HORA

已提交的 HORA 路径是 Sharpa 手内（in-hand）teacher/student 流程。teacher owner 位
于 PPO、APPO 与 SAC 的 task 树下，通过 `sharpa_inhand` 的 `hora` profile 选择；
student 蒸馏使用 `sharpa-distill` 入口和
`src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml`。

## Teacher

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
uv run sharpa-train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --task sharpa_inhand --sim mujoco training.no_play=true
```

HORA PPO owner 设置 `algo.algo_log_name=hora_ppo`，并通过
`sharpa_rl_unilab.algos.hora.rsl_rl:resolve_hora_ppo_runtime` 解析运行时。APPO 变体设置
`algo.algo_log_name=hora_appo`；SAC teacher 设置 `algo.algo_log_name=hora_sac`，并解析
`sharpa_rl_unilab.algos.hora.sac:resolve_hora_sac_runtime`。

## Student 蒸馏

student 蒸馏由 `src/sharpa_rl_unilab/training/train_hora_distill.py` 实现，并由
`src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=-1
```

teacher 检查点的解析在 `src/sharpa_rl_unilab/training/hora_distill_config.py` 中实现。
student 日志族为 `hora_distill`。
