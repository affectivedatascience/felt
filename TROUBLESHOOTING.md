# FELT v2 troubleshooting

## Environment installation

Run from `02_code/`:

```powershell
uv sync --frozen
uv run python -m pytest -q -p no:cacheprovider
```

Use a Python version accepted by `pyproject.toml` (3.11–3.13). `--frozen`
prevents an unnoticed dependency resolution from changing `uv.lock`.

## CUDA and CPU extraction

The locked environment uses the CUDA 13 PyTorch wheel source, but extraction
can still be requested on CPU:

```powershell
uv run python src/1_extract_raw_tracking.py --device cpu --workers 1
```

For CUDA, verify availability before a full run:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Start with `--first` and an isolated output root. Do not increase file workers
until GPU memory and decoder stability have been observed. Batch size one is
part of the accepted extraction profile.

## TorchCodec cannot load FFmpeg

On Windows, TorchCodec needs a shared FFmpeg build with DLLs such as
`avcodec-*.dll`, not only a standalone `ffmpeg.exe`. Stages 1 and 4 register
the directory before importing Py-Feat/TorchCodec. Pass it explicitly:

```powershell
uv run python src/1_extract_raw_tracking.py --ffmpeg-bin D:\ffmpeg\bin --first
uv run python src/4_generate_visualizations.py --ffmpeg-bin D:\ffmpeg\bin --dry-run
```

If the flag is omitted, resolution checks `FFMPEG_DIR` (an installation root
or `bin` directory), `PATH`, and Scoop's `ffmpeg-shared` versions. An explicit
or configured directory fails early if it is missing, is not shared on
Windows, or reports a major version outside 4–8.

Record `ffmpeg -version` and `ffprobe -version` in release provenance. The
batch renderer accepts FFmpeg major versions 4–8 for the current TorchCodec
integration.

## Use data on another drive

No source edit is required. Set roots before invoking Python so imported path
constants see them:

```powershell
$env:FELT_INPUT_DIR = "D:\RAVDESS\input"
$env:FELT_OUTPUT_DIR = "D:\FELT\output"
```

Most stages also accept explicit `--input-root`, `--output-root`, or related
flags. Use `--help` for the exact interface.

## Interrupted extraction or rendering

Extraction and visualization skip existing outputs by default. Stage 4 uses a
private `.partial` staging tree and promotes a trial only after all newly
rendered products pass checks. Re-run the same command to resume. Use
`--dry-run --list-pending` to see incomplete trials.

Do not use `--overwrite` merely to resume. It intentionally replaces accepted
outputs and greatly increases runtime.

## Smoothing rejects the input corpus

The default preflight requires the exact 2,452 relative trial paths in the
approved calibration corpus. Check for missing files, inclusion of `062/`, or
an incorrect raw root. `--require-calibration-content-match` additionally
checks exact published normalized raw bytes.

`--allow-input-manifest-mismatch` is for a documented non-release application,
not a way to bypass a failed official run. If a fresh Detectorv2 run has the
correct trial inventory but different floating-point bytes, omit the strict
content flag and compare numerical output using predeclared tolerances.

## Identity fragmentation appears in QC

Detectorv2 generated multiple within-video `Identity` labels in 1,124 trials.
This is not the same as duplicate frame rows. The v2 acceptance gate requires
zero duplicate-frame files; it records identity fragmentation as a tracker
diagnostic. Do not apply the historical 0.6.2 multi-face removal manifest.

## Stage 5 refuses to package

Packaging requires four default JSON evidence files:

- `qc/ravdess_input_summary.json`;
- `qc/pyfeat_v2_post_correction/qc_summary.json`;
- `qc/challis_smoothing/smoothing_run_manifest.json`; and
- `qc/visualization_outputs_summary.json`.

Run the corresponding commands in `PIPELINE.md`. Stage 5 also requires exact
counts and non-empty files. An existing archive is verified and retained; pass
`--overwrite` only when deliberately rebuilding it.

## Disk and runtime planning

The current uncompressed outputs are approximately 6.46 GiB raw CSVs,
10.46 GiB smoothed CSVs, and 11.17 GiB canonical MP4s. Building release ZIPs
requires additional space roughly equal to their final archive sizes, plus
temporary space for the archive currently being written. Keep protected
backups outside the working output tree.

Full Detectorv2 extraction and seven-view rendering are long-running compute
jobs. Run a one-file smoke, then a small actor/channel subset, before committing
to the full corpus.
