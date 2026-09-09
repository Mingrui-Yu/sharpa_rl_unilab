# Dexterous In-Hand Manipulation

This page covers the Sharpa Wave in-hand manipulation path owned by this
repository. Select backends with `--task` and `--sim`; do not override
`training.sim_backend` alone. The owner YAMLs remain the internal evidence for
which combinations are configured.

The Allegro in-hand tasks remain in the
[UniLab](https://github.com/unilabsim/UniLab) repository and are not part of
this package.

## Sharpa

Sharpa rotation uses the registered env `SharpaInhandRotation`. The rotation
owner is `sharpa_inhand`, and grasp-cache generation uses
`sharpa_inhand_grasp`.

Owner evidence (under `src/sharpa_rl_unilab/conf/`):

- `ppo/task/sharpa_inhand/mujoco.yaml`
- `ppo/task/sharpa_inhand/mujoco_hora.yaml`
- `ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `appo/task/sharpa_inhand/mujoco.yaml`
- `appo/task/sharpa_inhand/mujoco_hora.yaml`
- `sac/task/sharpa_inhand/mujoco_hora.yaml`
- `hora_distill/task/sharpa_inhand/mujoco.yaml`

The full HORA path is three stages:

1. Generate the grasp cache.
2. Train the teacher policy.
3. Distill a student policy when needed.

The full HORA teacher/student path is MuJoCo-owner-primary. The Motrix path
currently covers only phase-1 PPO rotation and grasp-cache collection; it is not
a full HORA capability-equivalent path.

### Grasp cache and scale

The default caches ship inside this package (and its wheel) under
`src/sharpa_rl_unilab/assets/caches/`; no Hugging Face download or runtime
network access is needed. `uv run sharpa-assets` pre-copies and verifies all
bundled assets into a writable cache directory.

To collect caches for custom scales — or to regenerate them locally — run the
grasp task once per scale (cache files are named `<prefix>_<scale>.npy`).
Generated files land under the writable asset cache root, so subsequent
training resolves them without further configuration. Regeneration is **slow**.

The helper script collects each scale sequentially:

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

<sub>Equivalent per-scale invocations: `uv run sharpa-train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true` (repeat for `[1.0]`, `[1.2]`, …).</sub>

Motrix can also collect a grasp cache (phase-1 scope only):

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand_grasp --sim motrix \
  'env.domain_rand.scale_list=[1.0]' \
  env.grasp_collection_target=1000 \
  training.no_play=true
```

To use a custom cache prefix, override `env.grasp_cache_path`:

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco \
  env.grasp_cache_path=caches/my_sharpa_grasp_cache
```

### Teacher and student

Train the HORA teacher with the `hora` profile (PPO, APPO or SAC):

```bash
uv run sharpa-train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --task sharpa_inhand --sim mujoco training.no_play=true
```

The SAC owner `sac/task/sharpa_inhand/mujoco_hora.yaml` is the default SAC task
and already selects the `hora_sac` runtime, so it takes no `--profile` flag.

Replay a teacher run with `sharpa-eval --profile hora algo.load_run=-1`:

```bash
uv run sharpa-eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora algo.load_run=-1
uv run sharpa-eval --algo appo --task sharpa_inhand --sim mujoco --profile hora algo.load_run=-1
```

Student distillation is configured by
`src/sharpa_rl_unilab/conf/hora_distill/task/sharpa_inhand/mujoco.yaml` and run
through the dedicated `sharpa-distill` entrypoint:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=-1
```

To distill from an APPO teacher, use `task=sharpa_inhand/mujoco_nodr` or set
`teacher.algo_family=appo` in the task owner YAML. Teacher checkpoint
resolution is implemented in
`src/sharpa_rl_unilab/training/hora_distill_config.py`.

Common log directories:

- `logs/hora_ppo/SharpaInhandRotation/`
- `logs/hora_appo/SharpaInhandRotation/`
- `logs/hora_distill/SharpaInhandRotation/`
