# Validation record

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
- Full production training is executed on `pc823`, not on the local development
  machine; run summaries are reported separately in the task handoff.
