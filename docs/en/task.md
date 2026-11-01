# Sharpa Wave in-hand rotation

## Task objective

Control the hand's joint position targets to rotate an object continuously
around a specified axis while maintaining a stable grasp. The reward encourages
rotation and penalizes object translation, deviations in hand posture and control
effort. Each episode lasts at most 20 seconds; leaving the height range around
the initial grasp ends the episode as a drop.

## Environment setup

The task currently uses MuJoCo. Each reset samples hand joint positions and an
object pose from a stable grasp cache. Caches are bundled with the package and
can be regenerated after changing the model or grasp policy.

Objects use eight fixed scales: 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4 and 1.5.
Each parallel environment uses one scale and its matching cache.

Training randomizes object mass, center of mass, friction, gravity direction
and joint control gains, and applies external force disturbances. Policy
observations include joint and tactile information, with configurable tactile
smoothing, latency and noise. These variations train the policy to adapt to
different objects and sensor conditions.

## Direct policies and HORA

All algorithms default to the HORA teacher/student pipeline, where the student estimates
the teacher's privileged representation from history and inherits its policy. SuperMimic
also supports an opt-in single stage: its direct actor depends only on measurable
observation history, while a privileged critic assists world-model and policy optimization.
The resulting policy is directly evaluable and deployable without distillation. Evaluation
checks rotation speed, survival time and drop rate in fixed scenes. Each algorithm uses
its own training budget; comparisons must account for actual samples and elapsed time.

See the [training guide](training.md) for instructions and the
[architecture](architecture.md) for input shapes and code responsibilities.
