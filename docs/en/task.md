# Task and environment guide

## Robot and objective

The task controls a Sharpa Wave hand with 22 actuated joints. The manipulated
object is a free cylinder. The policy rotates the object around the target axis
while keeping it stable in hand.

The registered environments are:

- `SharpaInhandRotation`: policy training and evaluation.
- `SharpaInhandRotationGrasp`: grasp-state collection.

## Observations and actions

| Contract | Size |
| --- | ---: |
| Hand action | 22 |
| Actor frame | 49 |
| Actor history | 147 |
| Flat APPO observation | 174 |
| HORA privileged critic history | 174 |

The actor observation contains hand state, position-target history, and tactile
history. The privileged critic additionally observes object and domain state
that is not available to the deployed actor.

## Training variations

Each episode randomly samples:

- hand actuator P/D gain multipliers;
- object mass and center-of-mass offset;
- object, elastomer, and metal friction;
- gravity direction;
- decaying external object force.

Object size is not mutated at runtime. The environment uses fixed MJCF variants
for these scales:

```text
0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

This avoids MuJoCo sameframe errors during mass/CoM randomization and gives the
policy a stable multi-scale training curriculum.

## Reward behavior

The reward favors target-axis object rotation and penalizes:

- object linear velocity;
- hand pose deviation;
- estimated torque;
- mechanical work;
- object displacement from its anchor.

Episodes end when the object drops outside the reset-height band or reaches the
time limit.

## Bundled assets

The package includes the Sharpa Wave MJCF, collision meshes, visual meshes, and
one grasp cache per object scale. `uv run sharpa-assets` prepares a writable
repair cache when needed; normal runs do not download assets.

Set `SHARPA_RL_UNILAB_ASSET_CACHE` to choose a custom asset-cache directory.
