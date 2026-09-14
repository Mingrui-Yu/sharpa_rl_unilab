# Sharpa RL for UniLab

Sharpa Wave in-hand manipulation and HORA algorithms, packaged as an independent
UniLab task repository. The task uses UniLab's current Manager-Based API: it
owns action, observation, reset, event, reward, termination and recorder terms,
while UniLab remains the external dependency that owns managers and simulation
backends.

[中文说明](README_zh.md) · [Documentation index](docs/README.md)

## Supported workflows

| Workflow | Command family |
| --- | --- |
| PPO baseline | `sharpa-train --algo ppo --sim mujoco` |
| APPO baseline | `sharpa-train --algo appo --sim mujoco` |
| HORA APPO teacher | `sharpa-train --algo appo --sim mujoco --profile hora` |
| FlashSAC teacher | `sharpa-train --algo flashsac --sim mujoco` |
| Grasp-cache generation | `sharpa-train --algo ppo --task sharpa_inhand_grasp` |
| HORA student distillation | `sharpa-distill` |
| Evaluation/playback | `sharpa-eval` |

## Install

UniLab is consumed as an external dependency, but development metadata resolves
it from a sibling checkout. Clone the repositories side by side:

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab

uv sync --extra mujoco --extra export
uv run sharpa-assets
```

Verify the installation:

```bash
uv run ruff check src tests
uv run pytest -q
uv run pyright
```

## Preview a configuration

`--cfg` prints the fully composed Hydra configuration without allocating a
simulation environment:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

## Smoke-train both teacher paths

These are intentionally tiny one-iteration runs. They verify the complete
collector, learner, checkpoint and summary path.

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-smoke

uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-smoke
```

Each successful run writes `model_1.pt` and `run_summary.json`.

## Full training

```bash
# HORA APPO
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.seed=1 training.no_play=true

# FlashSAC
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true

# PPO baseline
uv run sharpa-train --algo ppo --sim mujoco \
  algo.seed=1 training.no_play=true
```

Default output roots:

```text
logs/hora_appo/SharpaInhandRotation/<timestamp>/
logs/flash_sac/SharpaInhandRotation/<timestamp>/
logs/rsl_rl_ppo/SharpaInhandRotation/<timestamp>/
```

A run contains its resolved config, TensorBoard events, checkpoints, and a
completion summary. Monitor all runs with:

```bash
uv run tensorboard --logdir ./logs --port 6006
```

## Evaluate a checkpoint

Run from the same repository root used for training. `-1` selects the latest run
and its latest checkpoint in the canonical log tree.

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 \
  training.play_render_mode=record

uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 \
  training.play_render_mode=record
```

To load an exact checkpoint, pass its file path:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=/absolute/path/to/run/model_305.pt \
  training.play_render_mode=record
```

For a headless video, use `training.play_render_mode=record`; FlashSAC also
exports and verifies `policy.onnx` beside the checkpoint by default. The
complete workflow is documented in [evaluation and playback](docs/en/user-guide/evaluation.md).

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/en/getting-started.md)
- [Training workflows](docs/en/user-guide/training.md)
- [Evaluation and playback](docs/en/user-guide/evaluation.md)
- [HORA teacher/student](docs/en/user-guide/hora.md)
- [Task reference](docs/en/reference/task.md)
- [Architecture](docs/en/developer/architecture.md)
- [Validation record](docs/en/developer/validation.md)

## Manager-Based correctness notes

- Object size randomization uses immutable UniLab fixed model variants from
  `scene_scale_0.8.xml` through `scene_scale_1.5.xml`; it does not mutate geom
  sizes at runtime.
- Every fixed-variant body geom has a unique nonempty name for
  `mjbatch.VariantPack`.
- Every variant keeps `simple="false"` on the free object, avoiding the earlier
  reset-time mass/CoM `mj_setConst` sameframe crash.
- Tactile latency/smoothing, privileged state, actuator gains, object physical
  parameters, gravity and decaying external forces are explicit manager terms
  using the Entity facade.
