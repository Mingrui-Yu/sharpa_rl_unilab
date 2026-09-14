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

| Real robot | APPO | FlashSAC |
| --- | --- | --- |
| ![Real-robot deployment rollout](docs/media/sharpa-real-eval.gif) | ![APPO evaluation rollout](docs/media/sharpa-appo-eval.gif) | ![FlashSAC evaluation rollout](docs/media/sharpa-flashsac-eval.gif) |

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

### Train APPO

```bash
uv run sharpa-train --algo appo --sim mujoco \
  algo.seed=1 training.no_play=true
```

### Train FlashSAC

```bash
uv run sharpa-train --algo flashsac --sim mujoco \
  algo.seed=1 training.no_play=true
```

### Evaluate checkpoints

APPO:

```bash
uv run sharpa-eval --algo appo --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

FlashSAC:

```bash
uv run sharpa-eval --algo flashsac --sim mujoco \
  algo.load_run=-1 training.play_render_mode=record
```

Videos are written beside the selected checkpoint. FlashSAC also writes and
verifies `policy.onnx`.

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
