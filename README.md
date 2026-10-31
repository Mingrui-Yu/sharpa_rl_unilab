# Sharpa in-hand manipulation and HORA for UniLab

[中文](README_zh.md) | [Documentation](docs/README.md)

## Task overview

An independent UniLab task package for Sharpa Wave in-hand manipulation. It
provides a MuJoCo-ready hand/object task, bundled robot assets and grasp caches.
SuperMimic, PPO, APPO and FlashSAC default to the HORA teacher/student pipeline.
SuperMimic also offers an opt-in single-stage pipeline that trains a deployable
observation-only actor directly through a learned world model.

## Highlights and demonstration

- Train policies that rotate a free cylinder inside a 22-DoF Sharpa Wave hand.
- Use tactile history, privileged critic information, and object-scale
  randomization.
- Evaluate versioned teacher/student checkpoints and record videos.
- Train a HORA teacher and distill an actor-observation student.

### SuperMimic training highlight

The optional direct path removes teacher/student distillation while keeping a
privileged critic for training. In a one-seed MJWarp validation run on one RTX
4090, it finished 3,000 iterations (3,971,072 transitions) in 686 seconds. Both
policies below were evaluated on the same fixed 240-scene manifest:

| SuperMimic path | Return | Fixed-window rotation speed (rad/s) | Rotation angle (rad) | Survival (s) | Drop rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Privileged HORA teacher (default) | 45.33 | 0.784 | 15.68 | 19.56 | 1.25% |
| Observation-only direct actor (optional) | 43.59 | 0.671 | 13.42 | 19.41 | 2.08% |

These are task-specific, single-seed simulation results. As with the community
reports collected in [Awesome SuperMimic](https://github.com/miao-org/awesome-supermimic),
the reward, task horizon and evaluation protocol must be considered before
making comparisons across projects.

| Real robot | APPO | FlashSAC |
| --- | --- | --- |
| ![Real-robot deployment rollout](docs/media/sharpa-real-eval.gif) | ![APPO evaluation rollout](docs/media/sharpa-appo-eval.gif) | ![FlashSAC evaluation rollout](docs/media/sharpa-flashsac-eval.gif) |

## Manager-Based API

Sharpa Wave dexterous in-hand rotation uses UniLab's **Manager-Based API**. The
task owns action, observation, reset, event, termination and reward terms while
UniLab remains an external dependency and owns the manager lifecycle and
simulation backends.

All four algorithms share the HORA teacher policy and student distillation pipeline
by default. SuperMimic additionally supports an observation-only single-stage actor
with a privileged training critic. Critics retain the V/Q structure required by each algorithm.

## Installation and validation

UniLab 1.5.4 and its CUDA Torch wheels are resolved from the configured package
indexes.

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync
uv run pytest
uv run pyright
```

To install the source tree with pip instead, use `pip install .` in a separate
virtual environment. MuJoCo and MJWarp are installed by default; MJWarp requires
CUDA, and all four teacher algorithms accept `--sim mjwarp`. Training uses
UniLab's integrated tensor-native RL runtime: PPO/APPO collection and FlashSAC
inference/replay keep observations, actions, rewards, and done flags on Torch
tensors, and CUDA runs keep the training hot path device-resident.

`uv run pytest` runs fast regression tests by default; use
`uv run pytest -m slow` for simulation and training smoke tests.
The FlashSAC smoke test requires CUDA.

## Quick start

After installation, run from the repository root to train SuperMimic, PPO, APPO
and FlashSAC in sequence for Sharpa Wave in-hand cylinder rotation:

```bash
uv run sharpa-compare
```

In another terminal, start TensorBoard and open <http://localhost:6006> to view training curves:

```bash
uv run tensorboard --logdir logs/compare
```

Each algorithm saves its playback video after training to
`logs/compare/seed_<seed>_<timestamp>/<algorithm>/play_video.mp4`.

## Train, evaluate and distill

```bash
uv run sharpa-train --algo supermimic
# Optional single-stage SuperMimic without teacher/student distillation:
uv run sharpa-train --algo supermimic algo.single_stage=true
uv run sharpa-train --algo appo
uv run sharpa-train --algo flashsac
uv run sharpa-train --algo ppo
```

`--cfg` prints the composed Manager-Based configuration. Append Hydra overrides
to the command for tuning.

Default training saves `teacher_final.pt`; optional single-stage SuperMimic saves
a directly evaluable `policy_final.pt`. Training records `play_video.mp4` in the same
directory. Headless Linux machines use EGL or OSMesa for offscreen rendering; on
Ubuntu/Debian, install `libegl1` with a working NVIDIA graphics driver, or `libosmesa6`
for software rendering. `uv sync` does not install these system dependencies; set
`training.no_play=true` if recording is not needed.

For evaluation and HORA distillation, specify the checkpoint file directly:

```bash
uv run sharpa-eval --checkpoint /absolute/path/to/policy_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-distill --checkpoint /absolute/path/to/teacher_final.pt
uv run sharpa-eval --checkpoint /absolute/path/to/student_final.pt
```

`uv run sharpa-compare` trains the four algorithms sequentially using their
default configurations; append `--distill --eval --num-seeds 3` to run distillation
and evaluation across multiple seeds. Each algorithm uses its own training budget; equal sample counts or compute costs
are not guaranteed.

See [in-hand rotation](docs/en/task.md) for task background and the
[training guide](docs/en/training.md) for detailed steps, runtime limitations
and checkpoint compatibility.

## Task contract

- Object-scale randomization uses the fixed model-variant catalog
  `scene_scale_0.8.xml` … `scene_scale_1.5.xml`, keeping scene topology stable
  across scale samples.
- Fixed-variant body geoms have unique names for reliable
  `mjbatch.VariantPack` selection.
- Each variant retains `simple="false"` on the free object so reset-time mass
  and center-of-mass sampling remains consistent.
- Tactile smoothing/latency, privileged state, incremental position targets,
  randomized actuator gains, object mass/CoM/friction/gravity and decaying
  object forces are explicit manager terms that access state through the Entity
  facade.

See [architecture](docs/en/architecture.md) and [validation](docs/VALIDATION.md).
