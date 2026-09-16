# Training, distillation and evaluation

Install the environment as described in the
[README](../../README.md#installation-and-validation).
Run the following commands from the repository root; see
[in-hand rotation](task.md) for task background.

## 1. Prepare grasp caches

The bundled caches work without configuration. To regenerate all eight scales:

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
```

The script runs an independent PPO grasp task for each scale and writes to
`$XDG_CACHE_HOME/sharpa-rl-unilab/generated/caches/sharpa_grasp_linspace_<scale>.npy`
by default (`XDG_CACHE_HOME` defaults to `~/.cache`). Setting
`SHARPA_RL_UNILAB_ASSET_CACHE` places `generated/` under that custom directory.
Generated data takes precedence over bundled caches and survives asset repair and
manifest updates. Progress and the stop target count individual grasps. To use a custom location,
set the output prefix with the `SHARPA_GRASP_CACHE_PATH` environment variable,
then set `env.events.reset.params.grasp_cache_path` to the same prefix when
training the teacher.

## 2. Train a teacher

For APPO, run the following command; replace `--algo appo` with `--algo ppo` or
`--algo flashsac` to switch algorithms. PPO, APPO and FlashSAC share the same
HORA actor and Gaussian distribution.

```bash
uv run sharpa-train --algo appo
```

The default output directory is `logs/<algorithm>/seed_<seed>_<timestamp>/`.
The final model, `teacher_final.pt`, includes the run configuration and
normalization statistics. Configuration snapshots, `metrics.jsonl` and
TensorBoard logs are saved in the same directory.

## 3. Distill a student

Load a teacher and train a history encoder to match its privileged representation.
The teacher's policy and base normalization statistics remain frozen.

```bash
uv run sharpa-distill --checkpoint /path/to/teacher_final.pt
```

Replace `/path/to/teacher_final.pt` with the actual file path.
The default output directory is
`logs/hora_distill/<algorithm>_seed_<seed>_<timestamp>/`, with the final model
saved as `student_final.pt`.

## 4. Evaluate and compare algorithms

Specify the checkpoint directly for evaluation:

```bash
uv run sharpa-eval --checkpoint /path/to/teacher_final.pt
```

Replace the path with the actual file; use `student_final.pt` to evaluate a student.

By default, evaluation uses 3 evaluation seeds for each of the eight scales,
with 10 episodes per seed. Each episode lasts at most 20 seconds; episodes that
end early in a drop are not rerun. Results give equal weight to each scale.
Recorded metrics include return, survival time, drop rate, signed rotation angle
and rotation speed calculated over both the fixed window and actual survival
time. Results are saved beside the checkpoint as `<model_name>.evaluation.json`,
with the scene manifest in `scenes.json`.

To train all three algorithms:

```bash
uv run sharpa-compare
```

By default, this trains one teacher per algorithm. Append `--distill` for
distillation or `--eval` for evaluation. `--num-seeds N` selects N consecutive
seeds starting from each algorithm's configured seed. When the corresponding
stages are enabled, all teachers finish before student training begins, and
evaluation starts after all training finishes.

Outputs go to `logs/compare/seed_<seed>_<timestamp>/<algorithm>/`, with students
in its `student/` subdirectory. All evaluations share the scene manifest in the
first seed directory. Each algorithm uses its own budget; equal sample counts
or elapsed time are not guaranteed. The command does not aggregate multiple
seeds or create plots. For comparisons across seeds, treat training seeds as
independent replicates, rather than treating episodes as independent training runs.

Optional evaluation diagnostics are enabled with `evaluation.diagnostics=true`.
They report visited-state std, action saturation (`abs(action)>0.99`), adjacent
action RMS differences, joint-position second-difference RMS, and target-limit
frequency. Each episode is measured separately, including its terminal step;
scale-weighted summaries average episode metrics without joining reset boundaries.
The diagnostics use no policy random samples and leave evaluation trajectories
unchanged.

## 5. Common settings and compatibility

Append Hydra overrides to training commands. Use
`uv run sharpa-train --algo appo --cfg` to inspect the full merged configuration.
See the [common configuration](../../src/sharpa_rl_unilab/conf/common/sharpa_inhand.yaml)
for shared parameters and the [configuration directory](../../src/sharpa_rl_unilab/conf)
for algorithm parameters.

For example, reduce the number of teacher environments:

```bash
uv run sharpa-train --algo appo training.num_envs=1024
```

| Setting | Parameter and default |
| --- | --- |
| Teacher environments | `training.num_envs=2048` |
| Output directory | Created automatically by default; use `training.log_dir=/path/to/new_run` to specify a directory that does not yet exist |
| Device | `training.device=cuda:0`; `null` selects CUDA, MPS or CPU automatically; APPO's `algo.collector_device=null` follows the learner |
| Threads per process | `training.torch_threads.*`: 4 intra-op and 1 inter-op thread each for learner and collector |
| Teacher iterations | `algo.max_iterations`: 501 for PPO/APPO, 2000 for FlashSAC |
| Teacher save interval | `algo.save_interval`: 50 for PPO/APPO, 500 for FlashSAC; 0 disables intermediate saves, but the final checkpoint is still saved on normal completion |
| FlashSAC precision | `algo.use_amp=true`; set to `false` to disable mixed precision |
| Student budget | `distillation.transitions=100000000`, `distillation.num_envs=4096` |
| Student learning rate | `distillation.learning_rate=0.0003` |
| Student saves and logs | `distillation.save_every=10000000`, `distillation.log_every=10000`, in transitions |
| Logging | Teacher logs every iteration; `training.logger=none` disables TensorBoard, while `no_print` keeps only JSONL |
| Video | Saves `play_video.mp4` after training by default; `training.no_play=true` disables recording |

Use `uv run tensorboard --logdir logs` to view curves, with the number of newly
received transitions on the horizontal axis. Intermediate teacher checkpoints
are named `teacher_iteration_N.pt`. Sample counts and data reuse are recorded
separately; iteration counts cannot determine the actual number of samples.

Video is for inspecting behavior; quantitative evaluation must be run separately.
Headless Linux machines require EGL or OSMesa; see the
[README](../../README.md#train-evaluate-and-distill) for installation instructions.

See the [architecture](architecture.md) for implementation contracts and the
[validation record](../VALIDATION.md) for validation coverage.

The current layout requires tactile inputs, friction privilege, no gravity privilege,
three actor/critic history frames and 30 proprioceptive history frames. Incompatible
combinations fail during construction; this interface does not support variable dimensions.
