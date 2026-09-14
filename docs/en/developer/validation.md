# Validation record

## Manager-Based migration

Date: 2026-09-14

- UniLab source: `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`.
- `uv run ruff check src tests`: passed.
- `uv run pytest -q`: 54 passed.
- `uv run pyright`: 0 errors.
- `uv build`: source distribution and wheel built successfully.
- APPO one-iteration smoke (`4` envs): collector and learner completed and wrote
  `model_1.pt`.
- FlashSAC one-iteration smoke (`4` envs): collector and learner completed and
  wrote `model_1.pt`.
- Manager observation contracts: APPO flat observation 174; FlashSAC actor 147
  and privileged critic 174.
- Fixed variants: eight XML sources compile with unique geom names and retain
  `simple="false"`.

## Full training benchmark

Date: 2026-09-14

Both historical runs used MuJoCo, seed 1, and `training.no_play=true` on the
designated training host.

| Metric | APPO | FlashSAC |
| --- | ---: | ---: |
| Status / iterations | completed / 301 | completed / 5,371 |
| Environment steps | 39,452,672 | 11,000,832 |
| Training wall time | 1 h 13 m 13 s | 41 m 23 s |
| Final mean episode return | 48.8223 | 34.8279 |
| Best mean episode return | 49.2989 | 38.9166 |
| Final mean episode length | 390.35 | 351.57 |
| Final timeout rate | 0.9636 | 0.8333 |
| Final collector throughput | 9,755.5 env steps/s | 4,483.6 env steps/s |
| Final learner throughput | — | 62,770.9 samples/s |
| Final policy / actor loss | -0.00131 | -1.21809 |
| Final value / critic loss | 0.03056 | 1.74715 |

APPO obtained the higher final/best return and preserved objects longer.
FlashSAC completed its full off-policy schedule with a higher-sample-efficiency
learner, but its final return remained below APPO on this fixed benchmark.

## Documentation workflow verification

Date: 2026-09-14

Documentation commands are treated as executable interfaces. The following
checks were executed from the repository root after restructuring this guide.

Configuration composition:

```bash
uv run sharpa-train --algo appo --sim mujoco --profile hora --cfg
uv run sharpa-train --algo flashsac --sim mujoco --cfg
uv run sharpa-train --algo ppo --sim mujoco --cfg
uv run sharpa-distill task=sharpa_inhand/mujoco --cfg
```

Language parity:

```bash
uv run pytest tests/test_documentation_parity.py -q
```

Exact-directory smoke training:

- HORA APPO: 4 environments, 2 steps/env, 1 iteration; completed and wrote
  `model_1.pt`, `run_summary.json`, and later `play_video.mp4`.
- FlashSAC: 4 environments, batch 8, replay 16, 1 update/step, 1 iteration;
  completed and wrote `model_1.pt`, `run_summary.json`, `policy.onnx`, and
  `play_video.mp4`. ONNX/PyTorch max output difference was `9.13e-07`.
- PPO: 4 environments, 2 steps/env, 1 iteration; completed and wrote
  `model_0.pt`, `run_summary.json`, and later `play_video.mp4`.

Canonical default-tree training and latest-run evaluation:

- Trained HORA APPO, FlashSAC, and PPO one-iteration runs without setting
  `training.log_root`; outputs appeared under
  `logs/{hora_appo,flash_sac,rsl_rl_ppo}/SharpaInhandRotation/`.
- Evaluated all three with `algo.load_run=-1` and
  `training.play_render_mode=record`.
- Each evaluation loaded the expected latest checkpoint and produced
  `play_video.mp4`; FlashSAC also exported and verified `policy.onnx`.
