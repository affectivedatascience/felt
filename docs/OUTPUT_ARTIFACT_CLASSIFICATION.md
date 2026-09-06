# Output artifact classification

The authoritative manifest is
[`OUTPUT_ARTIFACT_CLASSIFICATION.csv`](OUTPUT_ARTIFACT_CLASSIFICATION.csv).
It inventories the mixed material currently under `01_data/02_output/logs/`,
`migration_smoke/`, `plots/`, and `qc/` without moving or deleting anything.
Counts and byte sizes were recorded on 2026-09-06.

## Action definitions

- `keep`: canonical release evidence, required packaging input, or scientific
  provenance that should remain in the production output tree.
- `archive`: experiment, preview, benchmark, or superseded operational record
  worth preserving outside the production tree.
- `delete`: reproducible cache data that may be removed only after the stated
  verification condition is satisfied.

The two `delete` recommendations are limited to the 2,452 cutoff-estimation
checkpoints and 2,452 smoothing resume checkpoints. They are not raw data,
final smoothed data, or QC summaries. No deletion is authorized merely by this
manifest.

## Important finding

The expected canonical file
`01_data/02_output/qc/missing_value_correction_audit.csv` is absent. The
manifest records it as a zero-file `canonical_missing` item. Regenerate it with
stage 2 and verify it before performing output cleanup.

Archive destinations in the CSV are proposals under `01_data/03_archive/`.
Before any move, confirm that this destination is excluded from production
discovery and release packaging, and generate a checksum inventory so every
move can be verified.
