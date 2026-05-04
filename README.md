# FELT dataset processing scripts

This repository contains the Python scripts used to generate the FELT facial-tracking outputs from the RAVDESS audiovisual video files. The pipeline runs Py-Feat on RAVDESS videos, checks missing values, filters and smooths selected tracking columns, generates visualization videos, and packages the final release archives.

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
        ├── 4_generate_au_videos.py
        ├── 5_generate_landmark_videos.py
        ├── 6_generate_overlay_videos.py
        ├── tools/
        │   ├── create_release_archives.py
        │   ├── download_ravdess_archives.py
        │   └── plot_raw_smooth.py
        └── utils/
            ├── __init__.py
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

From the repository root:

```bash
cd 02_code
uv sync
```

All commands below assume you are running from `02_code/`. By default, `uv sync` creates the project environment at `02_code/.venv/`.

Run scripts with:

```bash
uv run python src/<script_name>.py
```

Example:

```bash
uv run python src/1_extract_raw_tracking.py
```

The project pins `torch==2.2.0`, `torchvision==0.17.0`, and `torchaudio==2.2.0`. The original Windows/CUDA environment used CUDA 12.1 PyTorch wheels. CPU/macOS installs may resolve through the default package index. Exact reproduction of the original CUDA environment may require installing PyTorch from the appropriate PyTorch wheel index before running the pipeline.

## Py-Feat local patches

The original FELT processing environment used Py-Feat 0.6.2. Two local Py-Feat edits were used.

### 1. Identity tensor detach patch

In the installed Py-Feat `detector.py`, modify the return statement inside `detect_identity()` from:

```python
return self._convert_detector_output(facebox, face_embeddings.numpy())
```

to:

```python
return self._convert_detector_output(facebox, face_embeddings.detach().numpy())
```

This prevents the PyTorch runtime error:

```text
RuntimeError: Can't call numpy() on Tensor that requires grad.
```

### 2. Overlay landmark colour patch

In the installed Py-Feat `feat/data.py`, inside `plot_detections()`, change the overlay landmark colour from white to blue so landmarks remain visible over the original RAVDESS frames:

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

This checks raw tracking CSV files for null values and applies pandas forward-fill when needed. This stage modifies files in place under:

```text
01_data/02_output/01_raw_motion/
```

Despite older terminology, this stage is not numerical interpolation. It is forward-fill missing-value handling.

### 3. Clean signals

```bash
cd 02_code
uv run python src/3_clean_signals.py
```

This applies a low-pass Butterworth filter followed by Savitzky-Golay smoothing to selected tracking columns, including face box coordinates, landmarks, pose, and Action Units.

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

### 4. Generate Action Unit videos

```bash
cd 02_code
uv run python src/4_generate_au_videos.py
```

This generates Action Unit activation videos from smoothed CSV files. Some frames may raise Py-Feat plotting errors; these frames are omitted before the video is written.

Outputs are written to:

```text
01_data/02_output/03_smoothed_video/action_unit_activation/
├── speech/
└── song/
```

### 5. Generate landmark plot videos

```bash
cd 02_code
uv run python src/5_generate_landmark_videos.py
```

This generates videos showing landmarks, face bounding box, and head pose without rendering the original source video frame.

Outputs are written to:

```text
01_data/02_output/03_smoothed_video/landmark_plot/
├── speech/
└── song/
```

### 6. Generate landmark overlay videos

```bash
cd 02_code
uv run python src/6_generate_overlay_videos.py
```

This generates videos showing landmarks, face bounding box, and head pose over the original RAVDESS video frame.

Outputs are written to:

```text
01_data/02_output/03_smoothed_video/landmark_overlay/
├── speech/
└── song/
```

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
uv run python src/tools/create_release_archives.py
```

Creates the six release ZIP files:

```text
raw_motion_speech.zip
raw_motion_song.zip
smoothed_motion_speech.zip
smoothed_motion_song.zip
smoothed_video_speech.zip
smoothed_video_song.zip
```

Motion archives contain CSV files and use maximum ZIP compression. Video archives contain MP4 files and are stored without additional compression.

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
│   ├── action_unit_activation/
│   │   ├── speech/
│   │   └── song/
│   ├── landmark_plot/
│   │   ├── speech/
│   │   └── song/
│   └── landmark_overlay/
│       ├── speech/
│       └── song/
├── 04_release_archives/
├── logs/
└── plots/
```

## Logs

Each script writes a log file to:

```text
01_data/02_output/logs/
```

Logs are useful for identifying skipped files, missing folders, Py-Feat errors, plotting-frame omissions, and release archive counts.

## Notes on removed helper scripts

Older helper scripts that manually split speech/song files or deleted video-only RAVDESS files are no longer part of the active codebase. Speech/song separation is now handled during task construction, and video-only files are ignored rather than deleted.
