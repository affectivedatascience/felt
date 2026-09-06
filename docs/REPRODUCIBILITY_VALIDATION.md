# FELT v2 reproducibility validation

This record summarizes the acceptance work completed on 2026-09-06. Generated
smoke artifacts remain outside Git under
`01_data/reproducibility_smoke/end_to_end/`.

## Validated environment

- Windows, Python 3.11 environment resolved by `02_code/uv.lock`;
- Py-Feat 2.0.3 `Detectorv2`;
- Torch 2.13.0+cu130 and TorchCodec 0.14.0+cu130;
- NVIDIA GeForce RTX 5070 Ti; and
- FFmpeg 8.1.2 full-shared, passed explicitly with `--ffmpeg-bin`.

The machine's `FFMPEG_DIR` pointed to unsupported FFmpeg 9.0.1 during this
test. The runtime helper rejected it before Py-Feat import, and the explicit
8.1.2 path completed decoding successfully.

## Two-trial end-to-end smoke

The selected trials were:

- speech: `Actor_10/01-01-05-01-01-01-10.mp4` (117 frames); and
- song: `Actor_10/01-02-05-01-01-01-10.mp4` (139 frames).

Both were extracted from restored MP4s with CUDA, batch size one, and
Detectorv2. Decoded video QC matched every extracted frame. Pre-correction QC
reproduced the approved song exception exactly: frame 36 contained 2,177 blank
cells and a zero `FaceScore` failure sentinel.

The manifest-driven repair retained source/frame metadata and copied all model
outputs from frame 35. Post-correction QC found zero missing or infinite cells,
schema failures, unresolved/nonportable source references, frame gaps,
duplicate frames, or decoded frame-count mismatches.

Stage 3 then processed both trials with the approved configuration signature
`abc92430897d564c9fc027431d7f283cae11fa731016996dbd50dcf1bac973e0`.
Geometry, all 20 Action Units, and all 52 blendshapes were filtered. The
subset-only `--allow-input-manifest-mismatch` flag was used because a two-trial
smoke intentionally differs from the 2,452-file calibration inventory.

Stage 4 produced all seven full-length views for both trials. Independent
decoded-video QC passed all 14 MP4s, including frame counts, H.264 codec,
30 fps, and canonical dimensions. A final stage-4 dry run reported two
complete trials and zero pending.

The fresh raw and smoothed CSVs were compared with their corresponding release
files. Both schemas and shapes matched, and all numeric cells were exactly
equal: maximum absolute difference zero and zero differing cells.

## Full-corpus evidence

The existing normalized production tree separately passed the full acceptance
gates:

- restored RAVDESS input: 4,904 MP4s, with exactly 2,452 selected full-AV
  trials (1,440 speech and 1,012 song);
- corrected raw motion: 2,452 CSVs and 299,854 rows, with zero structural,
  missing/infinite-value, source-linkage, or decoded frame-count failures;
- smoothed motion: 2,452 CSVs, no smoothing failures, approved configuration
  signature; and
- canonical visualization output: 17,164 MP4s, all 17,164 independently
  decoded and accepted (2,452 in each of seven views).

Stage 5 dry-run accepted all required JSON summaries and exact inventories for
the raw, smoothed, and video products. Full archive rebuilding was deliberately
not repeated during cleanup because protected copies already exist in a
separate local backup; their hashes are recorded in
`docs/REPRODUCIBILITY_BASELINE.md`.

## Clean-checkout verification

A detached disposable worktree at commit `19a750a` was created after the
cleanup commits. Within that clean checkout, `uv sync --frozen` created a new
Python 3.12.13 environment from the lockfile and installed Py-Feat 2.0.3 and
the declared CUDA/TorchCodec stack. All 59 tests passed there.

Using only the clean checkout's scripts and explicit data roots, the input
validator accepted 4,904 MP4s, the extraction dry run selected 2,452 tasks,
the correction dry run recognized the released target as already corrected,
and the visualization dry run found 2,452 complete trials with zero pending.
Stage 5 dry-run independently accepted 2,452 raw CSVs, 2,452 smoothed CSVs,
17,164 videos, and all required QC summaries. The disposable worktree and its
generated environment were then removed; the `reproducible-v2` worktree
remained clean.
