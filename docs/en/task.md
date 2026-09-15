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

## Teacher and student

PPO, APPO and FlashSAC share the HORA teacher/student pipeline:

1. **Teacher:** learns a rotation policy using measurable observations and
   privileged simulation information, such as mass and friction.
2. **Student:** estimates the teacher's privileged representation from observation
   history, then uses the inherited policy to produce actions.
3. **Evaluation:** checks rotation speed, survival time and drop rate in fixed scenes.

Student inference requires only measurable observations and their history,
without privileged simulation information. Each algorithm uses its own training
budget; comparisons must account for actual sample counts and elapsed time.

See the [training guide](training.md) for instructions and the
[architecture](architecture.md) for input shapes and code responsibilities.
