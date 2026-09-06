# FELT repository classification

This inventory separates the canonical Py-Feat 2.0.3 production workflow from
supporting utilities, development experiments, and Py-Feat 0.6.2 history. It
records intended status before files are moved or renamed.

Generated output artifacts are classified separately in
`docs/OUTPUT_ARTIFACT_CLASSIFICATION.csv`, with interpretation in
`docs/OUTPUT_ARTIFACT_CLASSIFICATION.md`.

## Canonical production stages

| File | Status | Intended role |
| --- | --- | --- |
| `02_code/src/1_extract_raw_tracking.py` | Production | Extract the 2,452 Detectorv2 raw-motion CSVs. |
| `02_code/src/2_fill_missing_values.py` | Production | Apply the approved manifest-driven Detectorv2 missing-value correction. |
| `02_code/src/3_clean_signals.py` | Production | Apply approved Challis cutoffs to geometry, AUs, and blendshapes. |
| `02_code/src/4_generate_visualizations.py` | Production | Canonical numbered entry point for all seven video products. |
| `02_code/src/5_package_release.py` | Production | Validate inventories and create the three canonical v2 archives. |
| `02_code/src/tools/generate_all_felt_visualizations.py` | Production dependency | Resumable full-corpus renderer used by stage 4. |
| `02_code/src/tools/create_release_archives.py` | Production dependency | Deterministic archive builder and verifier used by stage 5. |

## Production dependencies and QC gates

| File | Status | Intended role |
| --- | --- | --- |
| `02_code/src/utils/felt_paths.py` | Production dependency | Shared portable paths, logging, and RAVDESS filename parsing. |
| `02_code/src/utils/ffmpeg_runtime.py` | Production dependency | Resolve shared FFmpeg 4–8 and register Windows DLLs before Py-Feat/TorchCodec imports. |
| `02_code/src/utils/challis_smoothing.py` | Production dependency | Schema, cutoff, filtering, validation, clipping, and provenance helpers. |
| `02_code/src/utils/video_rendering.py` | Production dependency | H.264/FFmpeg video writer shared by renderers. |
| `02_code/src/tools/qc_motion_csv_frame_counts.py` | Production QC | Compare CSV rows and frame indices with decoded source-video frames. |
| `02_code/src/tools/qc_pyfeat_v2_extraction.py` | Production QC | Validate Detectorv2 schema, values, identities, frames, and source linkage. |
| `02_code/src/tools/qc_missing_values.py` | Production QC | Preserve the pre-correction missing-value audit. |
| `02_code/src/tools/qc_visualization_outputs.py` | Production QC | Validate all seven MP4s per trial by decoding and checking stream properties. |
| `02_code/src/tools/qc_compare_motion_corpora.py` | Reproduction QC | Compare reproduced CSV structure, metadata, and numeric values against a reference with declared tolerances. |
| `02_code/src/tools/validate_ravdess_inputs.py` | Production QC | Validate exact restored source counts, naming, actor layout, and non-empty MP4s. |
| `02_code/src/tools/estimate_challis_cutoffs.py` | Calibration/reproduction QC | Re-estimate cutoffs; normal release reproduction uses the approved tracked artifact. |
| `02_code/src/tools/generate_felt_visualization_set.py` | Production dependency | Render the seven views for one CSV. |
| `02_code/src/tools/animate_au_mesh_csv.py` | Production dependency | Generate the AU-to-canonical-mesh view. |
| `02_code/src/tools/plot_felt_sequence.py` | Production dependency | Supply mesh topology and sequence plotting used by canonical views. |
| `02_code/src/tools/download_ravdess_archives.py` | Input preparation | Obtain RAVDESS archives; revise to verify source checksums and layout. |
| `02_code/src/tools/monitor_full_extraction.ps1` | Operational support | Monitor a long production extraction run. |
| `02_code/src/tools/plot_raw_smooth.py` | QC support | Produce raw-versus-smoothed diagnostic plots. |

## Superseded visualization scripts

These scripts implement the older three-product video workflow and are not
canonical for FELT v2 because the seven-view renderer is the approved output:

- `02_code/src/legacy/generate_au_videos_v1.py`
- `02_code/src/legacy/generate_landmark_videos_v1.py`
- `02_code/src/legacy/generate_overlay_videos_v1.py`

They are preserved under `src/legacy/`; the seven-view batch renderer is the
numbered production pipeline.

## Historical Py-Feat 0.6.2 review tooling

The following multi-face review workflow was used for the historical `062`
dataset. Detectorv2 QC found zero duplicate-frame files, so these scripts and
decisions are not part of the normal v2 run:

- `02_code/src/legacy/pyfeat_062/prepare_multi_face_review.py`
- `02_code/src/legacy/pyfeat_062/apply_multi_face_review_decisions.py`

They may remain available as archived diagnostics, but v2 must stop on any new
duplicate-frame exception rather than silently applying the old 24-file
decision set.

## Development and experimental tools

| File | Status | Reason retained |
| --- | --- | --- |
| `02_code/src/experimental/profile_extract_raw_tracking.py` | Experiment | Documents extraction bottleneck investigation. |
| `02_code/src/experimental/smoke_predecoded_frame_detection.py` | Experiment | Tests an alternative predecoded-frame extraction path. |
| `02_code/src/experimental/smoke_pyfeat_v2_migration.py` | Migration smoke tool | Verifies installation and one-image Py-Feat 2 behavior. |
| `02_code/src/experimental/run_batch1_parallel_remaining.py` | Superseded compatibility wrapper | Replaced by the maintained extraction CLI. |
| `02_code/src/experimental/plot_action_units.py` | Visualization experiment | Exploratory AU plotting, not a canonical release view. |
| `02_code/src/legacy/development_run_helpers/run_phase3_full_overwrite_workers5.ps1` | Historical run helper | Records a specific development-machine run, not a portable entry point. |

## Tests

Files under `02_code/tests/` are production safety infrastructure. They are not
pipeline stages, but they must pass in a clean environment before a release is
accepted.

## Documentation classification

### Active release documentation

- `README.md`
- `PIPELINE.md`
- `REPRODUCIBILITY.md`
- `DATA_DICTIONARY.md`
- `HISTORY.md`
- `TROUBLESHOOTING.md`
- `docs/REPRODUCIBILITY_WORK_PLAN.md`
- `docs/REPRODUCIBILITY_BASELINE.md`
- `docs/REPOSITORY_CLASSIFICATION.md`
- `docs/DATASET_PROCESSING_WRITEUP.md`
- `docs/PYFEAT_V2_QC.md`
- `docs/CHALLIS_SMOOTHING_PROCEDURE.md`

### Public historical documentation

- `docs/archive/OUTPUT_COLUMN_SEMANTICS_PYFEAT_062.md`

Private development conversations, supervisor material, manuscript drafts,
and locally held reference PDFs are intentionally excluded from the public
repository. The canonical documentation cites external publications by DOI
instead of redistributing local copies.

## Ignored data classification

| Location | Status |
| --- | --- |
| `01_data/01_input/` | Restored RAVDESS source input; excluded from Git and derivative release. |
| `01_data/02_output/01_raw_motion/` | Canonical Detectorv2 raw output. |
| `01_data/02_output/02_smoothed_motion/` | Canonical smoothed output. |
| `01_data/02_output/03_smoothed_video/` | Canonical seven-view video output. |
| `01_data/02_output/qc/` | Canonical Detectorv2 and smoothing QC evidence. |
| `01_data/02_output/062/` | Historical Py-Feat 0.6.2 outputs and review artifacts. |
| `01_data/reproducibility_smoke/` | Isolated cleanup-validation output; never package as production data. |

The approved cutoff artifact, portable smoothing preflight, decoded video QC,
and experimental/legacy separation have now been implemented. This inventory
is descriptive; the numbered scripts and `PIPELINE.md` remain authoritative.
