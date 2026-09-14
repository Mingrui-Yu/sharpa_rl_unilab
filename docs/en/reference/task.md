# Sharpa in-hand task reference

## Task identity

- Registered rotation environment: `SharpaInhandRotation`
- Registered grasp-generation environment: `SharpaInhandRotationGrasp`
- Production backend: MuJoCo
- Action dimension: 22 hand joints
- Actor history: 3 × 49-dimensional frames = 147 dimensions
- HORA privileged critic history: 3 × 58-dimensional frames = 174 dimensions
- Flat APPO observation: 3 × 58-dimensional frames = 174 dimensions

## Manager-Based ownership

The registered classes derive from UniLab `ManagerBasedRlEnvCfg` and call
`make_manager_based_rl_env`. The task does not subclass the environment and does
not inspect backend model or qpos layouts.

| Concern | Term module |
| --- | --- |
| Joint/actuator/sensor contracts | `terms/constants.py` |
| Term parameter validation | `terms/validation.py` |
| Incremental position targets | `terms/action.py` |
| Hand/object reset | `terms/reset.py` |
| Grasp cache loading | `terms/cache.py` |
| Physical domain randomization | `terms/randomization.py` |
| Tactile and privileged observations | `terms/observation.py` |
| Drop termination | `terms/termination.py` |
| Rotation rewards | `terms/rewards.py` |
| Persistent object force | `terms/disturbance.py` |
| Grasp quality and recording | `terms/grasp.py` |

Hydra owners declare the scene entity, observation groups, events, actions,
terminations, rewards, and recorders. Python modules contain only stateful term
logic.

## Fixed object variants

The default catalog round-robins environments across scales:

```text
0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
```

Each scale has an MJCF source under
`assets/robots/sharpa_wave/scene_scale_<scale>.xml`. At reset, the term samples
the matching bundled grasp cache and writes the object through the Entity root
state API.

All variants retain `simple="false"` on the free object. The manager robot XML
also gives every body geom a unique nonempty name, as required by
`mjbatch.VariantPack`.

## Randomization and observations

Reset events own:

- hand actuator P/D gain multipliers;
- object mass and center-of-mass offset;
- object/elastomer/metal friction profiles;
- fixed-magnitude, uniformly directed gravity;
- fixed object scale identity through the variant plan.

A step event owns a mass-scaled decaying random object force. Observations own
joint noise, tactile force clipping/smoothing/latency, actor/critic history, and
privileged object properties.

See [architecture](../developer/architecture.md) for package boundaries.
