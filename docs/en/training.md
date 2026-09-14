# Training guide

## Choose a training recipe

| Goal | Command family |
| --- | --- |
| Train a standard on-policy baseline | `--algo ppo` |
| Train a flat asymmetric APPO baseline | `--algo appo` |
| Train a HORA APPO teacher | `--algo appo --profile hora` |
| Train a FlashSAC teacher | `--algo flashsac` |
| Regenerate grasp states | `--algo ppo --task sharpa_inhand_grasp` |

All production runs in this guide use MuJoCo.

## Train a teacher

HORA APPO teacher:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

FlashSAC teacher:

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

PPO baseline:

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  algo.seed=1 training.no_play=true
```

Leave `training.log_root` unset and run from the repository root. Runs are
written to the canonical tree:

```text
logs/<algorithm-log-name>/SharpaInhandRotation/<timestamp>/
```

Examples:

```text
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
logs/rsl_rl_ppo/SharpaInhandRotation/<timestamp>/
```

## Inspect a run

A completed run contains:

- `model_<iteration>.pt`: policy checkpoints.
- `run_config.json`: the exact resolved configuration.
- `run_summary.json`: final reward, episode length, steps, and checkpoint path.
- `events.out.tfevents.*`: TensorBoard curves.

Watch all active runs:

```bash
uv run tensorboard --logdir ./logs --port 6006
```

Open:

```text
http://localhost:6006
```

## Useful controls

| Control | Purpose |
| --- | --- |
| `algo.seed=1` | Make a run reproducible |
| `training.no_play=true` | Exit after training instead of opening playback |
| `training.play_render_mode=none` | Disable post-training playback |
| `training.play_render_mode=record` | Record a video after training |
| `training.device=cuda:0` | Select a learner device where supported |
| `algo.max_iterations=N` | Set the training schedule length |
| `algo.save_interval=N` | Set the checkpoint interval |
| `training.log_dir=/absolute/run` | Use one exact output directory |

Algorithm-specific examples:

```text
PPO/APPO: algo.num_envs, algo.steps_per_env or algo.num_steps_per_env
FlashSAC: algo.batch_size, algo.replay_buffer_n, algo.updates_per_step
```

## Continue PPO or APPO

Use the same repository root and select a previous run:

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  algo.seed=1 algo.load_run=-1 training.no_play=true
```

PPO can select a checkpoint iteration with `algo.checkpoint=<iteration>`.
The current FlashSAC launcher does not resume training from `algo.load_run`;
keep completed FlashSAC checkpoints or start a new seeded run.

If a run was created with `training.log_dir`, evaluate it by passing that run or
checkpoint path directly to `algo.load_run`.

## Regenerate grasp states

The repository ships production grasp caches. Regenerate them only after
changing the hand, object, or grasp-generation policy:

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

Useful environment variables:

```text
SHARPA_GRASP_TARGET=<number of saved grasps>
SHARPA_GRASP_NUM_ENVS=<parallel environments>
SHARPA_GRASP_CACHE_PATH=<output prefix>
```
