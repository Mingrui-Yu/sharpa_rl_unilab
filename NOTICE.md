# Source and license notices

Task, configuration and tooling code was extracted from UniLab (Apache-2.0).
The HORA modules were extracted from unilab_rl (Apache-2.0) and implement the
HORA teacher/student method ("In-Hand Object Rotation via Rapid Motor
Adaptation", Haozhi Qi et al., https://github.com/HaozhiQi/hora); any original
license headers carried by the extracted files are preserved in place.
`MIGRATION_MANIFEST.json` records source commits and pre-migration file hashes.
See [the extraction record](docs/ARCHITECTURE.md) for the ownership boundary.

Robot meshes, the Sharpa Wave XML scene and the grasp caches are bundled from
the pinned `unilabsim/unilab-robots` and `unilabsim/unilab-caches` datasets.
[`src/sharpa_rl_unilab/assets/manifest.json`](src/sharpa_rl_unilab/assets/manifest.json)
records the source revisions and file hashes; this package does not relicense
that data. These assets ship with the repository and package; no dataset fetch
is needed at runtime.
