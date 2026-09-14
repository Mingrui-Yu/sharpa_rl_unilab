# Reference training results

These completed runs used MuJoCo, seed 1, and `training.no_play=true`.

| Metric | APPO | FlashSAC |
| --- | ---: | ---: |
| Completed iterations | 301 | 5,371 |
| Environment steps | 39,452,672 | 11,000,832 |
| Training wall time | 1 h 13 m 13 s | 41 m 23 s |
| Final mean episode return | 48.8223 | 34.8279 |
| Best mean episode return | 49.2989 | 38.9166 |
| Final mean episode length | 390.35 | 351.57 |
| Final timeout rate | 0.9636 | 0.8333 |
| Final collector throughput | 9,755.5 steps/s | 4,483.6 steps/s |

APPO preserved the object longer and achieved the higher return on this fixed
benchmark. FlashSAC completed the off-policy schedule with higher learner sample
throughput. These are reference numbers, not a claim of universal superiority.
