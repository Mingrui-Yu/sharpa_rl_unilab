# Architecture

This document is for code changes. See [in-hand rotation](task.md) for task
background and the [training guide](training.md) for usage instructions.

## Responsibilities

Sharpa is a task package for UniLab. UniLab manages simulation, scenes and the
manager lifecycle; RSL-RL and uni_rl provide PPO, APPO and FlashSAC algorithm
components. Sharpa defines the task, HORA models, observation adaptation, and
training, distillation and evaluation pipelines.

The task is registered through `ManagerBasedRlEnvCfg`, and the UniLab factory
creates the environment. YAML declares scene entities and manager terms;
Python terms access simulation state through the Entity API. Grasp generation
uses a separate configuration and the native PPO entrypoint, while rotation
training uses the common HORA configuration.

## Code navigation

Paths below are relative to [`src/sharpa_rl_unilab/`](../../src/sharpa_rl_unilab):

| Area to change | Location |
| --- | --- |
| CLI arguments and configuration merging | `cli.py` |
| Common task, model and runtime parameters | `conf/common/sharpa_inhand.yaml` |
| Algorithm parameters and grasp configuration | `conf/{ppo,appo,flashsac}/` |
| Task registration, fixed scales and time steps | `tasks/sharpa_inhand/{config,rotation_registry,grasp_registry}.py` |
| Actions, observations, rewards, resets and randomization | `tasks/sharpa_inhand/terms/` |
| Four observation groups and terminal snapshots | `tasks/sharpa_inhand/teacher_env.py` |
| HORA models, V/Q networks and algorithm adaptation | `algos/hora/{models,teacher}.py` |
| PPO/APPO training and common checkpoints | `training/teacher_runtime.py` |
| Native FlashSAC runner adaptation | `training/flashsac_runtime.py` |
| Student distillation | `training/student_runtime.py` |
| Devices, threads and checkpoint configuration migration | `training/configuration.py` |
| Logs, run metadata and configuration snapshots | `training/logging.py` |
| Fixed-scene evaluation and multi-algorithm orchestration | `training/evaluation.py`, `tools/compare_rl_algo.py` |
| Video playback and rendering checks | `training/{playback,rendering}.py` |
| Model assets, caches and grasp generation script | `assets/`, `tools/sharpa_collect_grasps.sh` |

## Observations and data flow

`SharpaTeacherEnv` converts manager observations into four explicit inputs.
N is the number of environments:

| Input | Shape | Contents and consumers |
| --- | --- | --- |
| `obs` | `[N,147]` | Three noisy base frames of 49 values each; shared by teacher/student actors |
| `critic` | `[N,174]` | Three frames, each with 49 clean base values + 9 privileged values; critic only |
| `priv_info` | `[N,9]` | Current raw privileged information; teacher encoder |
| `proprio_hist` | `[N,30,49]` | Noisy history from the same source as `obs`, oldest to newest; student encoder |

The teacher encodes privileged information, concatenates it with base
observations and produces actions. The student replaces the privilege encoder
with a history encoder and inherits the rest of the teacher's policy. PPO/APPO
use independent V networks; FlashSAC uses distributional double Q and target Q
networks. Network depth and optimizer parameters are defined in the configuration
and model source code.

Sensors are read only once per control step. Clean tactile values retain
magnitude computation and deterministic clipping, without the actor's smoothing,
latency or noise. Clean observations do not represent a complete Markov state.

## Algorithm integration

| Algorithm | Reused components | Sharpa adaptation |
| --- | --- | --- |
| PPO | RSL-RL `PPO`, `RolloutStorage` | Four inputs, terminal bootstrapping, short rollouts and budget scheduling |
| APPO | uni_rl APPO/V-trace, `RolloutStagingPool` | Separate collector, bounded queue, policy versions, history reuse and budget scheduling |
| FlashSAC | `DoubleBufferOffPolicyRunner`, native replay and learner losses | Observation transport, new-data statistics, common checkpoints and counters |

PPO/APPO retain local sampling schedules to handle terminal observations and
short final batches at the sampling budget boundary. FlashSAC inherits native
`learn()`, which manages the collector, inference, replay, prefetching and
optimization loop; warmup does not count toward update iterations. See the
[training guide](training.md#2-train-a-teacher) for device and budget limits.

FlashSAC transports `obs=[actor147,current_priv9]` and `critic174`; reset and
terminal snapshots use the same layout. Replay stores raw observations and
actions, with current normalization statistics and privilege encoding applied
after sampling. Collection inference and learning share the actor. `CleanQ`
applies independent critic input normalization before the native Q network.

Training reuses `OffPolicyLogger`, with Sharpa adding JSONL, stage information
and actual sample counts. `sharpa-compare` only orchestrates stage commands;
evaluation algorithms and statistical functions live in `training/evaluation.py`.

## Required contracts

- **Termination and reset:** underlying autoreset is disabled. Save all four
  terminal observation groups before resetting the affected environments.
  Fill a new episode's history with its first frame without affecting other
  environments. A pure timeout bootstraps from the old episode's next
  observation; a simultaneous drop and timeout counts as a true termination.
- **Normalization:** accumulate the current observation only once per newly
  received transition. Next observations, repeated epochs, staging and replay
  resampling do not accumulate statistics; evaluation freezes all statistics.
  Actor and critic statistics are independent. FlashSAC's current and target
  Q networks share statistics, and target parameter soft updates do not blend
  statistics.
- **Behavior policy:** PPO/APPO store raw Gaussian actions and their log
  probabilities at sampling time; the environment executes clipped actions.
  APPO synchronizes version numbers, weights and statistics together. The
  collector installs new versions only at rollout boundaries. Each rollout
  retains its own final frame to avoid joining different trajectories.
- **Gradients and distillation:** Q updates do not modify the actor. Actor
  updates retain Q gradients with respect to actions without updating Q
  parameters. The student trains only the history encoder, with base statistics
  frozen and history statistics updated on new batches. Each step first updates
  latent MSE, then acts using the updated student.
- **Actual counts:** distinguish collected, received and training_samples.
  At the end of FlashSAC training, some submitted tail data may not yet have
  contributed to statistics, so collected and received can differ.
- **Fixed scales:** object scales use separate MJCF files; geom sizes must not
  change at runtime. Each variant has unique geom names, and the free object
  retains `simple="false"` to avoid MuJoCo `sameframe` errors when resets change
  mass or center of mass. Task terms use the Entity API without accessing
  backend internals.

See [compatibility](training.md#5-common-settings-and-compatibility) for
checkpoint support and the [validation record](../VALIDATION.md) for historical
tests and experiments.
