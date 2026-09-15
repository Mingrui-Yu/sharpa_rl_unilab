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
`caches/sharpa_grasp_linspace_<scale>.npy` by default. To use a custom location,
set the output prefix with the `SHARPA_GRASP_CACHE_PATH` environment variable,
then set `env.events.reset.params.grasp_cache_path` to the same prefix when
training the teacher.

## 2. Train a teacher

For APPO, run the following command; replace `--algo appo` with `--algo ppo` or
`--algo flashsac` to switch algorithms.

```bash
uv run sharpa-train --algo appo
```

The default output directory is `logs/<algorithm>/seed_<seed>_<timestamp>/`.
The final model, `teacher_final.pt`, includes the run configuration and
normalization statistics. Configuration snapshots, `metrics.jsonl` and
TensorBoard logs are saved in the same directory.

FlashSAC teacher training requires CUDA/MPS and supports stopping only by update
iteration count. PPO/APPO also support a sampling budget: set both
`algo.max_iterations=null training.max_transitions=N`. The sample count rounds
up to a vector step, collecting at most one fewer extra transition than the
number of environments. Checkpoint saves are still triggered by iteration count.

### Policy distribution

PPO, APPO and FlashSAC share the same HORA actor and Gaussian distribution.
New training defaults to `model.std_mode=state_independent`: one learnable
log-std per joint, initialized with `model.initial_std=1.0`. Set
`model.std_mode=state_dependent` to learn log-std from the actor trunk instead.
The output head initially has zero weights and bias `log(initial_std)`, so both
modes start with the specified actual standard deviation.

PPO/APPO default to `model.action_mapping=clip`; `tanh` is also supported.
FlashSAC requires `tanh`. For example:

```bash
uv run sharpa-train --algo appo model.action_mapping=tanh \
  model.std_mode=state_independent model.initial_std=1.0 \
  algo.algorithm.entropy_coef=0.01
```

`model.std_parameterization=log` uses `std=exp(log_std)`.
`model.log_std_bounds=null` leaves log-std unconstrained; optionally set a finite
increasing pair such as `model.log_std_bounds=[-10,2]` to clamp it before exp.
The initial std must lie within those bounds. Clamping changes gradients outside
the interval. Network computation retains the algorithm's AMP settings; sampling
and density, entropy and KL calculations use FP32. A broader AMP numerical audit
remains a separate TODO.

For tanh, deterministic actions are `tanh(mean)` and the entropy bonus estimates
transformed action entropy using fresh current-policy samples. For clip, the
training density, entropy and KL refer to the latent Gaussian. PPO/APPO always
store the raw sample before mapping, together with its behavior log-prob and
mean/std. Existing learning-rate schedules use `KL(old || current)`; APPO's old
distribution for this schedule is its target policy. The shared analytic KL
removes the upstream additive numerical offsets: identical policies now have
zero KL and do not trigger a learning-rate increase. Thresholds and adjustment
factors are unchanged.

FlashSAC collection keeps its per-environment Gaussian noise for a sampled
number of steps, controlled by `algo.exploration.noise_zeta_mu=2.0` and
`algo.exploration.noise_zeta_max=16`. A maximum of 1 refreshes every step.
The runner owns this state on the learner device; actor/critic updates and
deterministic evaluation do not advance it. Collection keeps the native IPC and
inference/update schedule.

The new defaults change FlashSAC from its former state-dependent bounded std
to state-independent log-std initialized at 1. These are new training settings,
not a reproduction of the old training trajectory. Existing protocol-v2 teacher
and student checkpoints load through an explicit compatibility path that keeps
their original scalar/tanh std parameterization and action mapping. Evaluation
and distillation cannot override the checkpoint's model configuration.

## 3. Distill a student

Load a teacher and train a history encoder to match its privileged representation.
The teacher's policy and base normalization statistics remain frozen.

```bash
uv run sharpa-distill --checkpoint /path/to/teacher_final.pt
```

Replace `/path/to/teacher_final.pt` with the actual file path.
The default output directory is
`logs/hora_distill/<algorithm>_seed_<seed>_<timestamp>/`, with the final model
saved as `student_final.pt`. The student inherits the teacher's task and grasp
cache configuration. You can override `distillation.*`, `evaluation.*`, device,
thread, logging and video settings; `env.*` and model configuration overrides
are rejected. Evaluation and distillation can both run on CPU with
`training.device=cpu`.

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

Evaluation restores the task from the checkpoint and allows only `evaluation.*`
and `training.device` overrides. You can also specify the checkpoint through
`algo.checkpoint`. Reusing a scene manifest validates the task configuration,
grasps and randomization values.

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

Current support covers evaluation of this project's v2 teacher/student
checkpoints and distillation of v2 teachers. Loading older v2 configurations
automatically migrates fields while preserving task, model and normalization
settings. Supported v2 checkpoints from the former synchronous FlashSAC runtime
can also be loaded. Incompatible legacy shared HORA, flat PPO and native
FlashSAC formats require retraining. Current entrypoints support only a single
learner and training from scratch; resuming training and JIT/ONNX export of v2
models are not supported.

See the [architecture](architecture.md) for implementation contracts and the
[validation record](../VALIDATION.md) for validation coverage.
