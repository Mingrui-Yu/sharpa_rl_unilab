# Sharpa RL UniLab

[中文](README_zh.md) | [Documentation](docs/README.md)

## Task overview

An independent UniLab task package for Sharpa Wave in-hand manipulation. It
provides a MuJoCo-ready hand/object task, bundled robot assets, grasp caches,
and training entrypoints for PPO, APPO, HORA APPO, FlashSAC, and HORA student
distillation.

## Highlights and demonstration

- Train policies that rotate a free cylinder inside a 22-DoF Sharpa Wave hand.
- Use tactile history, privileged critic information, and object-scale
  randomization.
- Export FlashSAC actors to ONNX and record evaluation videos.
- Train a HORA teacher and distill an actor-observation student.

**Demonstration:** GIF placeholder.

## Installation

Clone UniLab and this task package as sibling repositories:

```bash
mkdir ~/ws/unilab-tasks
cd ~/ws/unilab-tasks
git clone https://github.com/Motphys/UniLab.git
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco --extra export
uv run sharpa-assets
```

NVIDIA CUDA training is recommended. Evaluation can run headlessly with
`training.play_render_mode=record`.

## Quickstart

### 1. Smoke-test the installation

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora \
  algo.num_envs=4 algo.steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-hora-appo-check
```

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.num_envs=4 algo.batch_size=8 algo.replay_buffer_n=16 \
  algo.updates_per_step=1 algo.learning_starts=1 algo.max_iterations=1 \
  algo.save_interval=1 training.no_play=true \
  training.log_dir=/tmp/sharpa-flashsac-check
```

### 2. Train a policy

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

The APPO reference result in [the results guide](docs/en/results.md) uses the
baseline owner without `--profile hora`.

### 3. Evaluate a checkpoint

Evaluate the newest HORA APPO checkpoint:

```bash
uv run sharpa-eval --algo appo --sim mujoco --profile hora \
  algo.load_run=-1 training.play_render_mode=record
```

Evaluate the newest FlashSAC checkpoint:

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

Videos are written beside the selected checkpoint. FlashSAC also writes and
verifies `policy.onnx`.

### 4. Reproduce the reference benchmark

```bash
uv run sharpa-train --algo appo --sim mujoco \
  algo.seed=1 training.no_play=true
```

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

Reference metrics are recorded in [docs/en/results.md](docs/en/results.md).

## Documentation

- [Getting started](docs/en/getting-started.md)
- [Training guide](docs/en/training.md)
- [Evaluation guide](docs/en/evaluation.md)
- [HORA guide](docs/en/hora.md)
- [Task and environment guide](docs/en/task.md)
- [Reference results](docs/en/results.md)

## License

Code is licensed under the [Apache License 2.0](LICENSE). Asset sources and
third-party notices are described in [NOTICE.md](NOTICE.md).
