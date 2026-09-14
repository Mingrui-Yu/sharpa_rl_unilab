# Validation record

The runs below use the pre-issue-2 observation and training protocols. Their
Actor inputs, sample budgets and evaluation procedures differ. They are
historical execution records, not an algorithm ranking or evidence of relative
sample efficiency. See [the issue 2 migration record](migrations/issue-2.md).

## 2026-09-14 Manager-Based migration

- UniLab source: `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c` (`origin/main`).
- Package contract: `54 passed` with `uv run pytest -q`.
- Static typing: `0 errors` with `uv run pyright`.
- APPO one-iteration smoke (`4` envs): collector and learner completed, checkpoint
  `model_1.pt` written.
- FlashSAC one-iteration smoke (`4` envs): collector and learner completed,
  checkpoint `model_1.pt` written.
- Manager observation contracts: APPO flattened actor/critic obs `174`; FlashSAC
  actor obs `147`, privileged critic obs `174`.
- Fixed variants: eight XML sources compile with unique geom names and retain
  `simple="false"`.

## 2026-09-14 full training on pc823

Both runs used the synced checkout at
`/home/pc823/ws/unilabsim/sharpa_rl_unilab`, UniLab `b4e6b58f`, MuJoCo, seed
`1`, and `training.no_play=true`.

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
| Checkpoint | `logs/full_runs/appo/model_301.pt` | `logs/full_runs/flashsac/model_5371.pt` |

Both runs completed their respective schedules. These measurements do not
establish a fair APPO–FlashSAC comparison.
