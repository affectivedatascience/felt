# FELT dataset processing scripts

This repository contains the Python scripts used to generate the FELT facial-tracking outputs from the RAVDESS audiovisual video files. The pipeline runs Py-Feat on RAVDESS videos, checks missing values, filters and smooths selected tracking columns, generates visualization videos, and packages the final release archives.

For the exact release command sequence, use [PIPELINE.md](PIPELINE.md). See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the fixed scientific profile,
acceptance counts, provenance requirements, and cross-machine tolerances. The
[data dictionary](DATA_DICTIONARY.md) describes the 2,184-column Detectorv2
schema. See [HISTORY.md](HISTORY.md) for the v1/v2 boundary and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for runtime issues.
The mixed generated logs, QC, previews, and caches are inventoried—without
moving them—in the
[output artifact classification](docs/OUTPUT_ARTIFACT_CLASSIFICATION.md).

The pipeline is non-destructive. It does not delete RAVDESS source files. It selectively processes only full audiovisual RAVDESS files whose filenames begin with `01-`. Video-only files beginning with `02-` and audio-only files beginning with `03-` are ignored during task construction.

## Repository structure

```text
face-tracking-2024/
├── 01_data/
│   ├── 00_downloads/
│   │   └── ravdess/
│   ├── 01_input/
│   │   ├── Actor_01/
│   │   ├── Actor_02/
│   │   └── ...
│   └── 02_output/
│       ├── 01_raw_motion/
│       ├── 02_smoothed_motion/
│       ├── 03_smoothed_video/
│       ├── 04_release_archives/
│       ├── logs/
│       └── plots/
└── 02_code/
    ├── pyproject.toml
    ├── uv.lock
    └── src/
        ├── 1_extract_raw_tracking.py
        ├── 2_fill_missing_values.py
        ├── 3_clean_signals.py
        ├── 4_generate_visualizations.py
        ├── 5_package_release.py
        ├── experimental/
        ├── legacy/
        ├── tools/
        │   ├── create_release_archives.py
        │   ├── download_ravdess_archives.py
        │   └── plot_raw_smooth.py
        └── utils/
            ├── __init__.py
            ├── ffmpeg_runtime.py
            ├── felt_paths.py
            └── video_rendering.py
```

## RAVDESS input structure

Download and unzip the RAVDESS video archives so that actor folders are placed under:

```text
01_data/01_input/
├── Actor_01/
├── Actor_02/
├── ...
└── Actor_24/
```

Each RAVDESS filename contains seven hyphen-separated fields:

```text
modality-vocal_channel-emotion-intensity-statement-repetition-actor
```

Relevant fields:

```text
modality:      01 = full audiovisual, 02 = video-only, 03 = audio-only
vocal channel: 01 = speech, 02 = song
```

The FELT extraction scripts process only full audiovisual speech/song files. Actor 18 has no song recordings in RAVDESS.

## Downloading RAVDESS archives

The helper script downloads the RAVDESS `Video_Speech_Actor_XX.zip` and `Video_Song_Actor_XX.zip` archives from Zenodo into:

```text
01_data/00_downloads/ravdess/
```

Run:

```bash
cd 02_code
uv run python src/tools/download_ravdess_archives.py
```

The script skips the missing Actor 18 song archive and uses `.part` files during download so interrupted downloads are not mistaken for complete ZIP files.

After downloading, unzip the actor archives into:

```text
01_data/01_input/
```

## Python environment

This project uses `uv` for Python environment management. The project file is located in `02_code/pyproject.toml`, and the resolved environment is recorded in `02_code/uv.lock`.
The Py-Feat 2 environment requires Python 3.11-3.13.

From the repository root:

```bash
cd 02_code
uv sync
```

All commands below assume you are running from `02_code/`. By default, `uv sync` creates the project environment at `02_code/.venv/`.
The lockfile installs Py-Feat 2.0.3 with PyTorch 2.13, TorchVision 0.28, and
TorchCodec 0.14 from the CUDA 13.0 PyTorch index.

On Windows, TorchCodec also requires an FFmpeg 4–8 shared build whose directory
contains DLLs such as `avcodec-62.dll`. The static FFmpeg executable supplied
by Scoop or `imageio-ffmpeg` is not sufficient. One isolated option is:

```powershell
conda create --prefix "$env:LOCALAPPDATA\felt-ffmpeg" -c conda-forge "ffmpeg>=8,<9" -y
$env:FFMPEG_DIR = "$env:LOCALAPPDATA\felt-ffmpeg\Library\bin"
uv run python src/experimental/smoke_pyfeat_v2_migration.py --ffmpeg-bin $env:FFMPEG_DIR
```

For CPU extraction, pass `--device cpu`; no source edit is required. See
`TROUBLESHOOTING.md` for CUDA and TorchCodec checks.

Run scripts with:

```bash
uv run python src/<script_name>.py
```

Example:

```bash
uv run python src/1_extract_raw_tracking.py
```

The raw extraction stage now uses `Detectorv2`. This is a scientific model
change from the Py-Feat 0.6.2 modular detector used for FELT v1.0.0, not merely
a dependency update. Validate the new outputs before replacing a published
FELT dataset.

## Historical Py-Feat local patches

The original FELT processing environment used Py-Feat 0.6.2 with two local
edits. They are provenance notes only and must not be applied to Py-Feat 2.0.3.

### 1. Identity tensor detach patch

The old environment changed the `detect_identity()` return statement from:

```python
return self._convert_detector_output(facebox, face_embeddings.numpy())
```

to:

```python
return self._convert_detector_output(facebox, face_embeddings.detach().numpy())
```

This prevented the PyTorch runtime error:

```text
RuntimeError: Can't call numpy() on Tensor that requires grad.
```

### 2. Overlay landmark colour patch

The old environment changed the overlay landmark colour from white to blue so landmarks remained visible over the original RAVDESS frames:

```python
color = "w"
```

to:

```python
color = "b"
```

## Pipeline scripts

Run the numbered scripts sequentially.

### 1. Extract raw tracking

```bash
cd 02_code
uv run python src/1_extract_raw_tracking.py
```

This runs Py-Feat on valid full audiovisual RAVDESS speech/song videos and writes raw CSV tracking files to:

```text
01_data/02_output/01_raw_motion/
├── speech/
│   ├── Actor_01/
│   └── ...
└── song/
    ├── Actor_01/
    └── ...
```

The script skips existing output files by default.

### 2. Fill missing values

```bash
cd 02_code
uv run python src/2_fill_missing_values.py
```

This applies only the versioned corrections listed in
`02_code/config/missing_value_corrections_v2.csv`. It verifies the expected
blank-cell count before replacing the failed model output from the preceding
frame while retaining source/frame metadata, and
writes an audit record. This stage modifies the listed raw file in place under:

```text
01_data/02_output/01_raw_motion/
```

Despite older terminology, this stage is not numerical interpolation. It is forward-fill missing-value handling.

### 3. Clean signals

The normal reproduction path uses the approved cutoff artifact tracked at
`02_code/config/master_cutoffs_v2.json`:

```bash
cd 02_code
uv run python src/3_clean_signals.py --workers 12
```

The stage verifies all 2,452 raw files against the artifact's path-independent
size and content manifests before processing. Action Units and blendshapes are
filtered by default in the official release profile. The switches
`--no-filter-action-units` and `--no-filter-blendshapes` exist only for
diagnostic derivatives.

To reproduce the scientific cutoff-estimation analysis rather than the release
pipeline, run:

```bash
cd 02_code
uv run python src/tools/estimate_challis_cutoffs.py \
  --workers 2 --include-action-units --include-blendshapes
```

Review its generated artifact and QC tables under
`01_data/02_output/qc/challis_smoothing/`; re-estimation does not replace the
tracked approved artifact automatically.

This applies group-specific, corrected zero-phase Butterworth filters to face
mesh, 68-point landmarks, head rotation, head translation, gaze, and face-box
trajectories, and independent approved cutoffs to each Action Unit and
blendshape column. Filtering all recoverable families preserves the option to
create a geometry-only derivative later by restoring raw AU and blendshape
columns; obtaining smoothed versions later would require rerunning the filters.
The revised pipeline does not apply a second Savitzky-Golay stage.

Outputs are written to:

```text
01_data/02_output/02_smoothed_motion/
├── speech/
│   ├── Actor_01/
│   └── ...
└── song/
    ├── Actor_01/
    └── ...
```

### 4. Generate the seven canonical visualization products

```bash
cd 02_code
uv run python src/4_generate_visualizations.py --dry-run
uv run python src/4_generate_visualizations.py --workers 2
```

The renderer resumes safely: each trial is generated in private staging,
checked for non-empty files and decoded frame counts, and atomically promoted.
The seven H.264 products are:

1. Action Unit region heatmap;
2. blendshape region heatmap;
3. AU-to-canonical-mesh animation;
4. landmark-only contour mesh;
5. landmark-only tessellation mesh;
6. landmark-overlay contour mesh; and
7. landmark-overlay tessellation mesh.

The two overlay products require the original RAVDESS MP4s under
`01_data/01_input/Actor_XX/`. The other five derive from the smoothed CSVs.
All products are written beneath:

```text
01_data/02_output/03_smoothed_video/felt_visualization_set/
├── AU_animation/
├── landmark_only/
└── Landmark_overlay/
```

The older three-product renderers are preserved under `src/legacy/` for
provenance and are not part of FELT v2 reproduction.

## Tools

Standalone helper scripts are stored in:

```text
02_code/src/tools/
```

### Download RAVDESS archives

```bash
cd 02_code
uv run python src/tools/download_ravdess_archives.py
```

Downloads RAVDESS speech and song video archives into:

```text
01_data/00_downloads/ravdess/
```

### Plot raw versus smoothed signals

```bash
cd 02_code
uv run python src/tools/plot_raw_smooth.py
```

Loads one raw CSV and its matching smoothed CSV, then plots a selected column for inspection. Output plots are written to:

```text
01_data/02_output/plots/
```

### Create release archives

```bash
cd 02_code
uv run python src/5_package_release.py --dry-run
uv run python src/5_package_release.py
```

The dry run fails unless the exact canonical inventories are present: 2,452 raw
CSVs, 2,452 smoothed CSVs, and 17,164 MP4s. The full command creates and
verifies three unified release ZIP files:

```text
01_raw_motion.zip
02_smoothed_motion.zip
03_smoothed_video.zip
release_manifest.json
SHA256SUMS.txt
```

Motion archives use maximum ZIP compression. MP4 members are stored without
additional compression. Fixed member timestamps and ordering make repeated
builds deterministic for unchanged inputs; the manifest also records a
path-independent content digest for every component.

Outputs are written to:

```text
01_data/02_output/04_release_archives/
```

## Output structure

After running the full pipeline, the output directory should contain:

```text
01_data/02_output/
├── 01_raw_motion/
│   ├── speech/
│   │   ├── Actor_01/
│   │   └── ...
│   └── song/
│       ├── Actor_01/
│       └── ...
├── 02_smoothed_motion/
│   ├── speech/
│   │   ├── Actor_01/
│   │   └── ...
│   └── song/
│       ├── Actor_01/
│       └── ...
├── 03_smoothed_video/
│   └── felt_visualization_set/
│       ├── AU_animation/
│       ├── landmark_only/
│       └── Landmark_overlay/
├── 04_release_archives/
│   ├── 01_raw_motion.zip
│   ├── 02_smoothed_motion.zip
│   ├── 03_smoothed_video.zip
│   ├── release_manifest.json
│   └── SHA256SUMS.txt
├── logs/
└── plots/
```

## Logs

Each script writes a log file to:

```text
01_data/02_output/logs/
```

Logs are useful for identifying extraction and smoothing errors. The release
manifest and checksums are the authoritative packaging inventory.

## Notes on removed helper scripts

Older helper scripts that manually split speech/song files or deleted video-only RAVDESS files are no longer part of the active codebase. Speech/song separation is now handled during task construction, and video-only files are ignored rather than deleted.
