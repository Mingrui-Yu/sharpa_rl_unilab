# 灵巧手内操作

本页介绍由本仓库维护的 Sharpa Wave 手内操作路径。通过 `--task` 和 `--sim` 选择后端；不要单独覆盖 `training.sim_backend`。owner YAML 始终是哪些组合被配置的内部证据。

Allegro 手内任务仍保留在 [UniLab](https://github.com/unilabsim/UniLab) 仓库中，不属于本包。

## Sharpa

Sharpa 旋转使用已注册的 env `SharpaInhandRotation`。旋转 owner 是 `sharpa_inhand`，抓取缓存生成使用 `sharpa_inhand_grasp`。

Owner 证据（位于 `src/sharpa_rl_unilab/conf/`）：

- `ppo/task/sharpa_inhand/mujoco.yaml`
- `ppo/task/sharpa_inhand/mujoco_hora.yaml`
- `ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `appo/task/sharpa_inhand/mujoco.yaml`
- `appo/task/sharpa_inhand/mujoco_hora.yaml`
- `sac/task/sharpa_inhand/mujoco_hora.yaml`
- `hora_distill/task/sharpa_inhand/mujoco.yaml`

完整 HORA 流程分三个阶段：

1. 生成 grasp cache。
2. 训练 teacher policy。
3. 需要时再蒸馏出 student policy。

完整 HORA teacher / student 流程以 MuJoCo owner 为主。Motrix 路径当前只承担 phase-1 PPO rotation 和 grasp cache 采集，不是完整 HORA 能力等价路径。

### Grasp cache 与 scale

默认 cache 随本包（及其 wheel）一起发布，位于 `src/sharpa_rl_unilab/assets/caches/`；无需 Hugging Face 下载，运行时也不需要联网。`uv run sharpa-assets` 会把全部内置资产复制并校验到可写缓存目录。

如需为自定义 scale 采集 cache，或在本地重新生成，可按每个 scale 分别运行 grasp 任务（cache 文件命名为 `<prefix>_<scale>.npy`）。生成的文件会落到可写资产缓存根目录，后续训练能直接命中，**无需额外配置；但重新生成耗时较长**。

辅助脚本会按顺序采集每个 scale：

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

<sub>等价的逐 scale 调用：`uv run sharpa-train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true`（对 `[1.0]`、`[1.2]` 等同理重复）。</sub>

Motrix 也可以采集 grasp cache（仅 phase-1 范围）：

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand_grasp --sim motrix \
  'env.domain_rand.scale_list=[1.0]' \
  env.grasp_collection_target=1000 \
  training.no_play=true
```

要使用自定义 cache 前缀，override `env.grasp_cache_path`：

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco \
  env.grasp_cache_path=caches/my_sharpa_grasp_cache
```

### Teacher 与 student

用 `hora` profile 训练 HORA teacher（PPO、APPO 或 SAC）：

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --task sharpa_inhand --sim mujoco training.no_play=true
```

SAC owner `sac/task/sharpa_inhand/mujoco_hora.yaml` 是 SAC 的默认 task，本身已选择 `hora_sac` runtime，因此不需要 `--profile` 参数。

用 `sharpa-eval --profile hora algo.load_run=-1` 回放 teacher run：

```bash
uv run sharpa-eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora algo.load_run=-1
uv run sharpa-eval --algo appo --task sharpa_inhand --sim mujoco --profile hora algo.load_run=-1
```

Student 蒸馏由 `src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置，通过专用的 `sharpa-distill` 入口运行：

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=-1
```

需要从 APPO teacher 蒸馏时，使用 `task=sharpa_inhand/mujoco_nodr`，或在该 owner YAML 中设置 `teacher.algo_family=appo`。teacher 检查点的解析在 `src/sharpa_rl_unilab/training/hora_distill_config.py` 中实现。

常见日志目录：

- `logs/hora_ppo/SharpaInhandRotation/`
- `logs/hora_appo/SharpaInhandRotation/`
- `logs/hora_distill/SharpaInhandRotation/`
