# Getting started

## Prerequisites

- Linux with an NVIDIA GPU is recommended for training.
- [uv](https://docs.astral.sh/uv/) 0.12 or newer.
- Git access to UniLab and this repository.
- MuJoCo native rendering packages are installed automatically by the `mujoco`
  extra; no system MuJoCo installation is needed.

The package treats UniLab as an external dependency. Development metadata in
`pyproject.toml` resolves that dependency from a sibling `../UniLab` checkout,
so clone the repositories side by side:

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
```

Install the locked environment:

```bash
uv sync --extra mujoco --extra export
```

Prepare and verify the bundled robot/grasp assets:

```bash
uv run sharpa-assets
```

The assets are package-owned. A writable repair cache is created only when
needed; `SHARPA_RL_UNILAB_ASSET_CACHE` may point it to another directory.

## Verify the installation

Run the package checks and inspect one composed Manager-Based configuration:

```bash
uv run ruff check src tests
uv run pytest -q
uv run pyright
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
```

The configuration printer exits before loading simulation assets. It is the
quickest way to inspect Hydra overrides and confirm that an owner exists.

## Run a one-iteration smoke train

Use a small environment count and an explicit temporary run directory. These
commands exercise the complete collector, learner, checkpoint, and summary path
without starting a benchmark-length run.

HORA APPO:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-smoke
```

FlashSAC:

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-smoke
```

A successful smoke run writes `model_1.pt` and `run_summary.json` in its
`training.log_dir`.

## Where to go next

- Launch a benchmark run: [training workflows](user-guide/training.md)
- Load a checkpoint: [evaluation and playback](user-guide/evaluation.md)
- Understand terms and fixed object variants: [task reference](reference/task.md)
