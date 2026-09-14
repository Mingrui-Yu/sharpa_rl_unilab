# Getting started

## What you need

- Linux with an NVIDIA GPU is recommended for training.
- [uv](https://docs.astral.sh/uv/) 0.12 or newer.
- Git access to UniLab and this repository.
- A display is not required; evaluation can record an offscreen video.

## Install side by side with UniLab

The package treats UniLab as an external task runtime. Development metadata
resolves UniLab from a sibling checkout:

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco --extra export
```

Prepare the bundled robot, scene, and grasp-cache assets:

```bash
uv run sharpa-assets
```

No Hugging Face download is required for normal training and evaluation.

## Check the installation

Print two complete Hydra configurations without starting physics:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

Then run the package test suite:

```bash
uv run pytest -q
```

## Run a two-minute sanity check

The smallest complete training run uses four environments and one iteration.
It verifies asset loading, simulation, collection, learning, checkpointing, and
run summaries.

HORA APPO:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-check
```

FlashSAC:

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-check
```

Each successful run writes a checkpoint and `run_summary.json` under its
configured `training.log_dir`.

## Next

- Start a full teacher run: [training guide](training.md)
- Load a checkpoint and record a video: [evaluation guide](evaluation.md)
- Understand the task, observations, and randomization: [task guide](task.md)
