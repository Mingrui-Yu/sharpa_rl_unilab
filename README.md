# Sharpa in-hand manipulation and HORA for UniLab

Sharpa Wave dexterous in-hand rotation with PPO, APPO and SAC teachers plus
HORA student distillation. This repository is the sole owner of the task, its
configuration, robot assets and grasp caches, and the HORA implementation
extracted from UniLab and unilab_rl.

[中文](README_zh.md) · [Roadmap: UniLab #1547](https://github.com/unilabsim/UniLab/issues/1547)

## Install

```bash
git clone https://github.com/unilabsim/sharpa_rl_unilab.git
cd sharpa_rl_unilab
uv sync --extra mujoco
```

Use `uv sync --extra motrix` for Motrix, or select both extras to install both
backends. During development UniLab and unilab-rl resolve through the local
editable path sources in [pyproject.toml](pyproject.toml) (`../UniLab`,
`../unilab_rl`).

Approximately 40 MB of robot XML, meshes and grasp caches are bundled in Git
and the Python package. Robot asset loading requires no Hugging Face download
or runtime network access. `uv run sharpa-assets` prepares a writable local
cache; `SHARPA_RL_UNILAB_ASSET_CACHE` can select its directory.

## Train and evaluate

| Algorithm | Owner configurations |
| --- | --- |
| PPO | MuJoCo, Motrix, MuJoCo HORA |
| APPO | MuJoCo, Motrix, MuJoCo HORA |
| SAC (HORA teacher) | MuJoCo |
| HORA distill | MuJoCo |

These are the available owner configurations, not a training-quality benchmark.

```bash
uv run sharpa-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run sharpa-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run sharpa-train --algo appo --sim mujoco --profile hora training.no_play=true
uv run sharpa-train --algo sac --sim mujoco training.no_play=true
```

The task defaults to `sharpa_inhand`; grasp-cache collection uses
`--task sharpa_inhand_grasp`, and the HORA teacher variants use
`--profile hora`. Append Hydra overrides for tuning; `--cfg` prints the
composed configuration. Select the backend with `--sim`.

For evaluation, point `algo.load_run` at the run directory containing your
checkpoint; `algo.checkpoint=-1` selects its latest checkpoint.

```bash
uv run sharpa-eval --algo ppo --sim mujoco training.log_root=./logs/eval-ppo algo.load_run=/absolute/path/to/ppo/run algo.checkpoint=-1
uv run sharpa-eval --algo appo --sim mujoco --profile hora algo.load_run=/absolute/path/to/hora-appo/run algo.checkpoint=-1
```

Student distillation runs through the dedicated entrypoint:

```bash
uv run sharpa-distill task=sharpa_inhand/mujoco algo.load_run=/absolute/path/to/teacher/run
```

See [HORA](docs/en/7-hora.md) for the teacher/student flow and
[Dexterous in-hand manipulation](docs/en/1-dexterous_inhand.md) for grasp-cache
collection and per-scale overrides.

## Tools and documentation

```bash
bash src/sharpa_rl_unilab/tools/sharpa_collect_grasps.sh 0.8 1.0 1.2
uv run python -m sharpa_rl_unilab.tools.benchmark_sharpa_init_dr_construct
```

The collect script drives `sharpa-train` grasp owners once per scale; the
benchmark measures init-DR construction cost versus variant count.

- [Task owners and commands](docs/en/1-dexterous_inhand.md)
- [HORA teacher/student](docs/en/7-hora.md)
- [Architecture and extraction record](docs/ARCHITECTURE.md)
- [Executed migration checks](docs/VALIDATION.md)
- [Source provenance](MIGRATION_MANIFEST.json) and [license notices](NOTICE.md)
