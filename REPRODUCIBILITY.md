# FELT v2 reproducibility contract

## Scientific profile

FELT v2 uses Py-Feat 2.0.3 `Detectorv2`. The selected corpus comprises 2,452
full audiovisual RAVDESS trials (1,440 speech and 1,012 song). The official
release applies the tracked missing-output correction and smooths geometry,
all Action Units, and all blendshapes. Seven visualization products are
generated for every trial.

The authoritative executable sequence is [PIPELINE.md](PIPELINE.md). The
development and preservation plan is in
[`docs/REPRODUCIBILITY_WORK_PLAN.md`](docs/REPRODUCIBILITY_WORK_PLAN.md).
Completed acceptance evidence is summarized in
[`docs/REPRODUCIBILITY_VALIDATION.md`](docs/REPRODUCIBILITY_VALIDATION.md).

## Locked software and external runtime

`02_code/pyproject.toml` and `02_code/uv.lock` define the Python environment.
The principal versions are:

- Python 3.11–3.13;
- Py-Feat 2.0.3;
- PyTorch 2.13.0;
- TorchVision 0.28.0;
- TorchCodec 0.14.0;
- NumPy below 2; and
- SciPy below 1.12.

FFmpeg is an external runtime dependency. Record its full `ffmpeg -version`
and `ffprobe -version` output for a release run, together with the operating
system, CPU, GPU model, GPU driver, CUDA runtime, Python version, worker counts,
and Git revision.

## Approved tracked artifacts

- `02_code/config/missing_value_corrections_v2.csv` identifies the sole
  reviewed Detectorv2 repair.
- `02_code/config/master_cutoffs_v2.json` contains the approved geometry, AU,
  and blendshape cutoffs plus calibration-corpus provenance.
- `02_code/uv.lock` resolves the Python dependency graph.

Normal release production consumes these artifacts. It does not re-estimate
cutoffs or reuse the historical Py-Feat 0.6.2 multi-face decisions.

## Expected corpus and output contract

| Component | Required result |
| --- | ---: |
| Restored RAVDESS MP4s | 4,904 |
| Selected full audiovisual trials | 2,452 |
| Speech trials | 1,440 |
| Song trials | 1,012 |
| Raw Detectorv2 CSVs | 2,452 |
| Frame rows | 299,854 |
| Smoothed CSVs | 2,452 |
| Canonical views | 7 |
| MP4s per view | 2,452 |
| Total canonical MP4s | 17,164 |

Released CSV source references use `Actor_XX/filename.mp4`, not extraction-host
absolute paths. The 2026-09-06 cleanup migrated this metadata token in the
existing raw and smoothed trees without parsing or reserializing any numerical
column. Its audit is
`01_data/02_output/qc/source_reference_normalization.csv` in the working data
tree. Protected pre-cleanup archive hashes remain documented in
`docs/REPRODUCIBILITY_BASELINE.md`.

## Equality and tolerances

Inventory, RAVDESS identities, CSV schema, frame indices, row counts,
correction targets, smoothing configuration, and view types must match
exactly. Published normalized raw archives can also be checked byte-for-byte
with the strict smoothing preflight flag.

A fresh Detectorv2 run on a different supported GPU, driver, or operating
system may exhibit small floating-point differences. Such a run is
scientifically reproducible when the structural gates pass and numeric values
are equivalent within explicitly reported tolerances; it is not required to
reproduce the published ZIP SHA-256. Do not silently choose a tolerance after
seeing differences: record absolute and relative tolerances before comparison,
then report maximum error and affected-cell counts by column family.

The repository comparator predeclares `atol=1e-6` and `rtol=1e-5`; override
them only when a study protocol declares different values before inspection:

```powershell
uv run python src/tools/qc_compare_motion_corpora.py `
  <candidate-root> <reference-root> `
  --report comparison.csv --summary comparison.json
```

For a predeclared smoke subset, add `--allow-reference-superset`; unexpected
candidate files, schema differences, metadata differences, and out-of-tolerance
numeric cells still fail.

Video container bytes may differ across compatible FFmpeg builds. Acceptance
therefore uses product identity, decoded frame count, dimensions, frame rate,
codec/profile, and successful decoding. Archive SHA-256 values identify one
packaged release build, while `content_manifest_sha256` identifies the ordered
source payloads used by the packager.

## Current verified evidence

The normalized post-correction audit reports:

- 2,452 files and 299,854 rows;
- zero schema or required-column failures;
- zero missing and infinite numeric cells;
- zero nonportable or unresolved source references;
- zero frame gaps and duplicate-frame files; and
- 2,452/2,452 matches to decoded source-video frame counts.

Detectorv2 identity labels fragment in 1,124 files, but each frame still has a
single output row; this is recorded as a model/tracker diagnostic rather than a
multiple-face correction list. The 24-file multi-face review and 125 removed
rows belong only to the historical Py-Feat 0.6.2 dataset under `062/`.

The approved smoothing configuration signature is
`abc92430897d564c9fc027431d7f283cae11fa731016996dbd50dcf1bac973e0`.
The current video inventory contains 17,164 non-empty MP4s, exactly 2,452 for
each canonical view. `qc_visualization_outputs.py` performs the independent
decoded-frame and stream-property acceptance audit, and stage 5 enforces the
passing QC summaries and inventory again before packaging. The 2026-09-06
full decode audit passed all 17,164 files.

## Reproducibility limits

RAVDESS licensing and redistribution terms govern the source MP4s; they are
not tracked in Git or included in FELT derivative archives. A reproducer must
obtain and arrange those inputs separately. GPU inference and video encoding
are the two main sources of cross-machine byte variation. Long full-corpus
extraction and seven-view rendering also require substantial compute time and
temporary disk capacity.
