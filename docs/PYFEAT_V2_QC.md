# Py-Feat 2 Extraction QC

Run QC from `02_code/` after raw extraction is complete.

## 1. Decoded frame counts and frame sequences

On Windows, make the shared FFmpeg tools available first:

```powershell
$ffmpegBin = "$env:USERPROFILE\scoop\apps\ffmpeg-shared\current\bin"
$env:PATH = "$ffmpegBin;$env:PATH"
```

Then compare every raw CSV with the decoded source-video frame count:

```powershell
uv run python src\tools\qc_motion_csv_frame_counts.py `
  --csv-root ..\01_data\02_output\01_raw_motion `
  --video-root ..\01_data\01_input `
  --report ..\01_data\02_output\qc\pyfeat_v2_pre_correction\frame_count.csv `
  --workers 12 `
  --count-mode decoded
```

The frame-count tool reports container metadata disagreements separately from
actual CSV-versus-decoded-frame mismatches.

## 2. Detectorv2 integrity and missing values

Run the streaming extraction audit and incorporate the frame report:

```powershell
uv run python src\tools\qc_pyfeat_v2_extraction.py `
  --csv-root ..\01_data\02_output\01_raw_motion `
  --video-root ..\01_data\01_input `
  --qc-dir ..\01_data\02_output\qc\pyfeat_v2_pre_correction `
  --frame-report ..\01_data\02_output\qc\pyfeat_v2_pre_correction\frame_count.csv
```

This checks missing and infinite values, schema consistency, required column
families, frame/index sequences, duplicate frames, source-input linkage,
identity fragmentation, face-score ranges, positive face boxes, and expected
frame dimensions. It processes one CSV at a time to remain practical with the
2,184-column Detectorv2 on-disk schema.

The main outputs are:

- `qc_summary.json`
- `extraction_integrity_file_summary.csv`
- `missing_values_file_summary.csv`
- `missing_values_column_summary.csv`
- `identity_fragmentation_files.csv`
- `identity_fragmentation_distribution.csv`
- `no_face_rows.csv`
- `qc_exceptions.csv`

Use `--limit N` to run a small smoke audit in a separate QC directory.

Preserve this directory before applying the correction manifest. After stage 2,
run the same integrity command with
`--qc-dir ../01_data/02_output/qc/pyfeat_v2_post_correction`. Final acceptance
requires zero missing cells, duplicate frames, schema failures, unresolved
source paths, and nonportable source references. See `../PIPELINE.md` for the
complete pre/post gate and expected counts.
