# Extraction validation — UniLab #1547

Status: **local gates green; behavior-equivalence replay and end-to-end training
smoke still pending on a healthy machine (see Known limitations).**
This file records migration checks, not a convergence benchmark or a new
platform claim.

## Coordinated source revisions

| Repository | Branch | Commit | Base |
| --- | --- | --- | --- |
| UniLab | `dev/issue-1547-remove-sharpa-hora` | `e8b61d24` | main `9e3bb6b8` |
| unilab_rl | `dev/issue-1547-remove-hora` (hook + removal) | `ae811c3` | main `441fa2b` |
| sharpa_rl_unilab | — | (initial commit, see git log) | — |

Pre-removal provenance for every migrated file (source path + sha256 at the
base commit) is recorded in [MIGRATION_MANIFEST.json](../MIGRATION_MANIFEST.json).

## Local gates

| Check | Result |
| --- | --- |
| unilab_rl `make format` / `mypy` / `pyright` (commit `ae811c3`) | green |
| unilab_rl `uv run pytest -q` (commit `ae811c3`) | 397 passed, 8 skipped, 3 deselected |
| unilab_rl `rg -i hora src/ tests/` (commit `ae811c3`) | zero hits (only CHANGELOG history entries) |
| UniLab `make check` (commit `e8b61d24`) | green |
| UniLab affected pytest subset (commit `e8b61d24`) | 1001 passed, 19 skipped, 1 xfailed |
| UniLab slow env/train-script config combinations (commit `e8b61d24`) | 49 passed, 2 skipped |
| UniLab Sphinx clean build (commit `e8b61d24`) | zero WARNING |
| UniLab `rg -i 'sharpa\|hora'` full-tree audit (commit `e8b61d24`) | zero hits (excluding `.git`/`.venv`/lock/log files) |
| sharpa_rl_unilab `uv sync --extra mujoco` | success (development-time local path sources) |
| sharpa_rl_unilab `make check` | green (ruff, format, mypy, pyright: 0 errors) |
| sharpa_rl_unilab `uv run pytest tests/ -q` (full suite incl. slow MuJoCo physics) | 70 passed, 4 skipped (motrix extra not installed) |
| sharpa_rl_unilab `uv build` | success; wheel contains `conf/` and `assets/` (manifest, caches, meshes, XML) |

## Entrypoint smoke checks

All run against the local `uv sync --extra mujoco` environment:

- `sharpa-train --algo {ppo,appo,sac} --sim mujoco --cfg` — composed config prints
  correctly for every owner combination.
- `sharpa-distill --cfg` — composed distillation config prints correctly.
- `sharpa-assets` — prepares the writable local asset cache.
- Spawn-subprocess registration, offline asset loading with sha256 verification,
  and owner combination identity are covered by `tests/test_package_contract.py`
  (passing in the gate above).

## Behavior equivalence

_pending: pre/post-extraction env replay comparison (reset/step observations,
rewards, termination flags) at a fixed seed with per-array comparison, for the
PPO/MuJoCo and APPO/MuJoCo owners. Suggested method: the
`docs/validation/compare_extraction.py` replay harness used for the
legged-manipulation extraction, pointed at the pre-removal UniLab/unilab_rl
commits recorded in MIGRATION_MANIFEST.json and at this repository._

## Training, checkpoints and playback

_partial: `--cfg` composition smoke for all owners passed (above). One-iteration
rollout/update smoke runs per owner, checkpoint resume, and
`sharpa-eval`/`sharpa-distill` playback checks are pending a healthy machine
(see Known limitations)._

## Installed assets and entrypoints

_passed: `uv build` wheel bundles `conf/` and `assets/` (manifest, grasp caches,
meshes, scene/robot XML); `sharpa-assets` cache preparation and offline asset
loading with sha256 verification are covered by `tests/test_package_contract.py`._

## Source-tree cleanup

_passed: `rg -i hora src/ tests/` in unilab_rl at `ae811c3` and full-tree
`rg -i 'sharpa|hora'` in UniLab at `e8b61d24` both return zero hits outside
excluded metadata (`.git`, `.venv`, lock/log files); the only remaining mention
is unilab_rl's CHANGELOG history entry._

## Known limitations / follow-ups

- **Slow MuJoCo physics tests: root cause found and fixed.** The segfault
  (exit 139 in `mujoco_uni` `BatchEnvPool.reset`) was caused by the `object`
  body in `scene.xml` being compiled with MuJoCo's `simple`/`sameframe`
  optimization: reset-time com/mass domain randomization then trips
  `mj_setConst` ("body 24 is compiled as simple but sameframe no longer
  holds") and the native batch runtime corrupts the heap on that error path.
  Fixed by `simple="false"` on the `object` body (see
  `src/sharpa_rl_unilab/assets/manifest.json` notes). The same defect exists
  in the pre-migration UniLab XML. A second pre-existing failure surfaced once
  the crash was gone: `test_sharpa_mujoco_interval_force_plan_matches_decay_and_mass_scaled_resample`
  asserted legacy `IntervalRandomizationPlan` fields while the provider
  (unchanged from UniLab HEAD) builds ops-based plans; the stale test was
  updated to the ops contract and the full suite passes (70 passed).
- **Behavior-equivalence replay and end-to-end training smoke are still
  pending**; run them on a healthy machine using the approach described above.
- **pyproject dependency pinning is development-time only.** Dependencies
  currently resolve through local path sources plus `override-dependencies`.
  After the source-repo PRs merge, switch to git pins (aligned with the
  legged-manipulation repo) and remove the override.
