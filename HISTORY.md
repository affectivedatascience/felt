# FELT processing history

## FELT v1 / Py-Feat 0.6.2

The original workflow used Py-Feat 0.6.2 with a modular detector chain and two
local package edits: an identity-tensor detach fix and a landmark-overlay colour
change. It produced three older visualization types. A manual review covered
24 files with duplicate face rows and removed 125 rows.

Those choices belong only to the historical outputs under
`01_data/02_output/062/`. The old renderers are preserved under
`02_code/src/legacy/`, and the old schema note is under `docs/archive/`.
Neither is part of FELT v2 production.

## Detectorv2 migration and experiments

The migration evaluated Py-Feat 2 behavior, batch-size effects, predecoded
frames, extraction throughput, and alternative visualizations. The resulting
scientific decisions were:

- Py-Feat 2.0.3 `Detectorv2` is the final extraction method;
- extraction uses batch size one to preserve decoded-frame alignment;
- the release smooths geometry, all 20 AUs, and all 52 blendshapes; and
- seven organized visualization products replace the older three-product set.

Development profiling, smoke results, and alternative designs remain in the
files classified as development history by
`docs/REPOSITORY_CLASSIFICATION.md`. They explain decisions but are not steps a
reproducer must execute.

## FELT v2 production result

The full Detectorv2 run produced 2,452 raw CSVs and 299,854 frame rows. QC found
zero decoded-frame mismatches and zero duplicate-frame files. One repeatable
incomplete model-output row—frame 36 of
`song/Actor_10/01-02-05-01-01-01-10.csv`—was repaired using the tracked,
targeted forward-fill manifest. The repair retains source/frame metadata and
carries forward the complete preceding model output, including replacement of
the zero `FaceScore` failure sentinel.

Challis residual-autocorrelation analysis on the full corpus selected the
approved filter cutoffs. One corrected forward/backward Butterworth stage was
used; the previous fixed 6 Hz plus Savitzky–Golay cascade was retired. The
production smoothing run completed all 2,452 files and generated the seven
canonical H.264 videos for every trial, totaling 17,164 MP4s.

## Reproducibility cleanup

Before cleanup, the mixed experiment/production tree was captured in branch
`snapshot/pre-repro-cleanup-v2`, commit `00c9485`, and tag
`pre-reproducibility-cleanup-v2`. A separate reference worktree preserves that
state.

The cleanup established five numbered production entry points, tracked the
approved correction and cutoff artifacts, added unit and full-corpus QC gates,
replaced v1 packaging, and separated legacy renderers. On 2026-09-06, extraction
machine absolute paths in the raw and smoothed `input` columns were changed to
portable `Actor_XX/filename.mp4` identifiers using a byte-only replacement.
Protected pre-cleanup archives remain in a separately stored local backup
according to the baseline record.

See [PIPELINE.md](PIPELINE.md) for commands and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the acceptance contract.
