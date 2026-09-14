# Training workflows

## Choose an owner

| CLI selection | Owner file | Use |
| --- | --- | --- |
| `--algo ppo --sim mujoco` | `conf/ppo/task/sharpa_inhand/mujoco.yaml` | On-policy PPO baseline |
| `--algo appo --sim mujoco` | `conf/appo/task/sharpa_inhand/mujoco.yaml` | Flat asymmetric APPO baseline |
| `--algo appo --sim mujoco --profile hora` | `conf/appo/task/sharpa_inhand/mujoco_hora.yaml` | HORA teacher with separated actor/critic observations |
| `--algo flashsac --sim mujoco` | `conf/flashsac/task/sharpa_inhand/mujoco.yaml` | FlashSAC asymmetric-critic teacher |
| `--algo ppo --task sharpa_inhand_grasp` | `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml` | Regenerate a one-scale grasp cache |

MuJoCo is the production backend for the fixed-variant task.

## Preview the effective configuration

Hydra composes the algorithm root, the task owner, and command-line overrides.
Print the result before allocating a simulation environment:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

Any overrides shown by `--cfg` may be appended to the training command.

## Start a full run

Use `training.no_play=true` for unattended training. Stay in the repository root
and leave `training.log_root` unset; each launcher then writes to the canonical
`./logs/<algo_log_name>/<task>/<timestamp>/` tree.

PPO:

```bash
uv run sharpa-train --algo ppo --sim mujoco \
  training.no_play=true
```

HORA APPO:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  training.no_play=true
```

APPO baseline:

```bash
uv run sharpa-train --algo appo --sim mujoco \
  training.no_play=true
```

FlashSAC:

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  training.no_play=true
```

For a reproducible run, set the seed explicitly:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true
```

## Useful training overrides

| Override | Meaning |
| --- | --- |
| `training.no_play=true` | Do not enter playback after training |
| unset `training.log_root` | Canonical `./logs/<algo_log_name>/<task>/<run>/` tree (recommended) |
| `training.log_dir=/absolute/run` | Use one exact run directory instead of generating a timestamp |
| `algo.seed=1` | Set the training seed |
| `training.device=cuda:0` | Select the learner device where the algorithm supports it |
| `algo.num_envs=N` | Set vectorized environment count |
| `algo.max_iterations=N` | Set schedule length |
| `algo.save_interval=N` | Checkpoint interval |
| `training.play_render_mode=none` | Disable playback after training |
| `training.logger=tensorboard` | Use the configured TensorBoard logger |

Algorithm-specific controls:

| Algorithm | Common controls |
| --- | --- |
| PPO / APPO | `num_steps_per_env` or `steps_per_env`, learning-rate and PPO parameters |
| FlashSAC | `batch_size`, `replay_buffer_n`, `updates_per_step`, `learning_starts` |

The exact defaults live in `src/sharpa_rl_unilab/conf/<algorithm>/`.

## Inspect output

A completed run directory contains:

- `run_config.json`: fully resolved configuration snapshot.
- `events.out.tfevents.*`: TensorBoard scalar data.
- `model_<iteration>.pt`: checkpoints.
- `run_summary.json`: completion status, environment steps, final/best return,
  episode length, checkpoint path, and wall time (written by the current APPO
  and FlashSAC launchers).
- FlashSAC evaluation may also write `policy.onnx` and `play_video.mp4`.

Default roots by owner:

```text
logs/rsl_rl_ppo/SharpaInhandRotation/<timestamp>/
logs/appo/SharpaInhandRotation/<timestamp>/
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
```

Monitor a root with TensorBoard:

```bash
uv run tensorboard --logdir ./logs --port 6006
```

## Select or resume a checkpoint

For PPO and APPO, set `algo.load_run` to a run directory or checkpoint file
under the configured log tree. `-1` means “latest run”.

```bash
# PPO example: continue the latest run under ./logs/rsl_rl_ppo
uv run sharpa-train --algo ppo --sim mujoco \
  algo.load_run=-1 training.no_play=true

# APPO example: continue an exact run directory
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/hora-appo/run \
  training.no_play=true
```

PPO can additionally select an iteration with `algo.checkpoint=<iteration>`.
The current FlashSAC launcher does not use `algo.load_run` to resume training;
keep complete FlashSAC runs and checkpoints intact or start a new seeded run.

`training.log_dir` is the portable way to assign one exact run directory for a
new training job. `training.log_root` has launcher-specific semantics in the
current UniLab entrypoints; avoid it for the documented default workflow. If a
run was created with `training.log_dir`, evaluate it by passing that run or its
checkpoint path directly to `algo.load_run`.

## Regenerate grasp caches

Grasp generation is a one-scale utility workflow, not part of normal teacher
training. The helper sets the fixed variant and invokes the PPO grasp owner:

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

Environment variables:

- `SHARPA_GRASP_TARGET`: number of successful grasps to save.
- `SHARPA_GRASP_NUM_ENVS`: vectorized environment count.
- `SHARPA_GRASP_CACHE_PATH`: output prefix.

The repository already bundles production caches, so this is needed only after
changing the hand/object model or grasp-generation policy.
