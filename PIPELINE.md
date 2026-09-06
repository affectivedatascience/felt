# FELT v2 production pipeline

This is the canonical command sequence for producing FELT v2 from restored
RAVDESS audiovisual videos. Run commands from `02_code/`. Do not use material
under `01_data/02_output/062/`; it belongs to the historical Py-Feat 0.6.2
workflow.

## 0. Install and test

```powershell
cd 02_code
uv sync --frozen
uv run python -m pytest -q -p no:cacheprovider
```

The locked environment uses Python 3.11–3.13, Py-Feat 2.0.3 `Detectorv2`,
PyTorch 2.13, TorchVision 0.28, and TorchCodec 0.14. A compatible FFmpeg shared
build (major version 4–8) is required for video decoding. The pipeline checks
`--ffmpeg-bin`, `FFMPEG_DIR`, `PATH`, and Scoop's `ffmpeg-shared` installation,
in that order.

## 1. Verify input selection

Place RAVDESS MP4s at `01_data/01_input/Actor_XX/*.mp4`. The complete local
source tree contains 4,904 files; the pipeline selects 2,452 full audiovisual
files: 1,440 speech and 1,012 song trials.

```powershell
uv run python src/tools/validate_ravdess_inputs.py

uv run python src/1_extract_raw_tracking.py `
  --dry-run `
  --report ../01_data/02_output/logs/extraction_dry_run.csv
```

The validator must pass all exact modality/channel counts. The extraction dry
run must report 2,452 selected tasks. It ignores video-only and
audio-only RAVDESS files and does not delete source material.

## 2. Extract raw Detectorv2 motion

CUDA production run:

```powershell
uv run python src/1_extract_raw_tracking.py `
  --device cuda --batch-size 1 --workers 1 --pyfeat-num-workers 0 `
  --ffmpeg-bin D:\ffmpeg\bin
```

Omit `--ffmpeg-bin` when one of the automatic locations works. Use `--device
cpu` on a CPU-only system. Existing non-empty CSVs are skipped;
use `--overwrite` only for an intentional replacement. The extractor writes
portable source references such as `Actor_01/01-01-01-01-01-01-01.mp4`.

For a long Windows run, `src/tools/monitor_full_extraction.ps1` can monitor the
existing extraction process and output counts. It does not start extraction.

## 3. Preserve pre-correction extraction QC

First compare CSV rows with decoded source-video frames:

```powershell
uv run python src/tools/qc_motion_csv_frame_counts.py `
  --csv-root ../01_data/02_output/01_raw_motion `
  --video-root ../01_data/01_input `
  --count-mode decoded --workers 8 `
  --report ../01_data/02_output/qc/pyfeat_v2_pre_correction/frame_count.csv
```

Then run the Detectorv2 integrity audit and detailed missing-value audit:

```powershell
uv run python src/tools/qc_pyfeat_v2_extraction.py `
  --csv-root ../01_data/02_output/01_raw_motion `
  --video-root ../01_data/01_input `
  --qc-dir ../01_data/02_output/qc/pyfeat_v2_pre_correction `
  --frame-report ../01_data/02_output/qc/pyfeat_v2_pre_correction/frame_count.csv

uv run python src/tools/qc_missing_values.py `
  --csv-root ../01_data/02_output/01_raw_motion `
  --qc-dir ../01_data/02_output/qc/pyfeat_v2_pre_correction/missing
```

The approved Detectorv2 exception is exactly one incomplete row: frame 36 of
`song/Actor_10/01-02-05-01-01-01-10.csv`, containing 2,177 blank cells and a
zero `FaceScore` failure sentinel. There
must be zero schema mismatches, infinite numeric cells, frame gaps, decoded
frame-count mismatches, and duplicate-frame files. Identity-label
fragmentation is reported but does not mean that duplicate face rows exist.

## 4. Apply the approved raw correction and re-audit

Preview and then apply the tracked correction manifest:

```powershell
uv run python src/2_fill_missing_values.py --dry-run
uv run python src/2_fill_missing_values.py
```

Re-run integrity QC into a separate post-correction directory:

```powershell
uv run python src/tools/qc_pyfeat_v2_extraction.py `
  --csv-root ../01_data/02_output/01_raw_motion `
  --video-root ../01_data/01_input `
  --qc-dir ../01_data/02_output/qc/pyfeat_v2_post_correction `
  --frame-report ../01_data/02_output/qc/pyfeat_v2_pre_correction/frame_count.csv
```

Post-correction acceptance requires 2,452 files and 299,854 rows, with zero
missing or infinite cells, schema mismatches, nonportable source references,
missing source paths, frame gaps, duplicate frames, and decoded frame-count
mismatches.

## 5. Apply the approved smoothing profile

```powershell
uv run python src/3_clean_signals.py --workers 12
```

The default tracked artifact is `config/master_cutoffs_v2.json`. It filters
geometry, all 20 Action Units, and all 52 blendshapes using corrected
zero-phase Butterworth filters. It checks the exact 2,452-trial inventory before
processing. When starting from the published normalized raw archive, add
`--require-calibration-content-match` to require its exact approved bytes.

Do not use `--no-filter-action-units`, `--no-filter-blendshapes`, or
`--allow-input-manifest-mismatch` for the official release. Acceptance requires
2,452 smoothed CSVs, no rows in `smoothing_errors.csv`, and a run manifest whose
processed plus checkpoint-reused count is 2,452.

Cutoff re-estimation is a scientific verification workflow, not a production
step. See `docs/CHALLIS_SMOOTHING_PROCEDURE.md` if it must be repeated.

## 6. Generate the seven canonical video products

```powershell
uv run python src/4_generate_visualizations.py --dry-run --list-pending
uv run python src/4_generate_visualizations.py --workers 2 --ffmpeg-bin D:\ffmpeg\bin
uv run python src/4_generate_visualizations.py --dry-run
```

The first run inventories missing products. Each pending trial is rendered in
a private staging directory, checked for non-empty outputs and decoded frame
counts, and atomically promoted. Re-running resumes missing views. Final dry-run
acceptance is 2,452 complete trials and zero pending trials, totaling 17,164
MP4s—2,452 in each of the seven view directories.

Run an independent full-corpus decode audit before packaging:

```powershell
uv run python src/tools/qc_visualization_outputs.py --workers 16
```

This checks every MP4 against its smoothed CSV row count and enforces H.264,
30 fps, and the approved 720×720 or 1280×720 dimensions. Use
`--inventory-only` only for a quick interim check, not final acceptance.

## 7. Validate and package the release

```powershell
uv run python src/5_package_release.py --dry-run
uv run python src/5_package_release.py
```

The dry run fails unless the exact raw, smoothed, and video inventories are
present and non-empty. It also requires passing RAVDESS input,
post-correction raw, smoothing-run, and decoded-video JSON summaries. The full
command writes deterministic archives under
`01_data/02_output/04_release_archives/`:

- `01_raw_motion.zip` — 2,452 CSVs;
- `02_smoothed_motion.zip` — 2,452 CSVs;
- `03_smoothed_video.zip` — 17,164 MP4s;
- `release_manifest.json`; and
- `SHA256SUMS.txt`.

Existing archives are verified and retained unless `--overwrite` is supplied.

## Isolated roots

Production defaults can be redirected without editing source code:

```powershell
$env:FELT_INPUT_DIR = "D:\RAVDESS\input"
$env:FELT_OUTPUT_DIR = "D:\FELT\output"
```

Individual stages also expose explicit root, report, QC, worker, FFmpeg,
resume, overwrite, and dry-run options through `--help`.
