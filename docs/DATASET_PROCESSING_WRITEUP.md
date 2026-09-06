# FELT dataset processing write-up

## Scope and reproducibility

FELT is a derivative facial-motion dataset generated from the audiovisual
speech and song videos in RAVDESS. The processing unit is one RAVDESS trial:
one source video produces one raw tracking CSV, one smoothed tracking CSV, and
one file in each requested visualization view.

The full RAVDESS source-video corpus has been restored locally under
`01_data/01_input/`, but it is excluded from Git and is not part of the FELT
derivative release. A fresh extraction or source-background overlay render
requires users to obtain RAVDESS separately and recreate this input tree.

## Script-to-pipeline mapping

The numbered Python files directly under `02_code/src/` are the one-to-one
pipeline stages:

| Stage | Script | Function |
| --- | --- | --- |
| 1 | `1_extract_raw_tracking.py` | Extract framewise Py-Feat tracking from RAVDESS videos. |
| 2 | `2_fill_missing_values.py` | Repair the known missing raw model-output row by forward-fill. |
| 3 | `3_clean_signals.py` | Apply the approved smoothing configuration and write smoothed CSVs. |
| 4 | `4_generate_visualizations.py` | Generate the seven canonical visualization products. |
| 5 | `5_package_release.py` | Validate inventories and package the three release archives. |

The scripts under `02_code/src/tools/` support or extend those stages. They
include frame-count and extraction QC, missing-value audits, multi-face review,
Challis cutoff estimation, diagnostic plots, archive creation, and the newer
seven-view batch visualization renderer. In particular,
`tools/estimate_challis_cutoffs.py` records the scientific calibration workflow;
normal reproduction uses the approved tracked master artifact consumed by
`3_clean_signals.py`. In addition,
`4_generate_visualizations.py` is the canonical numbered entry point;
`tools/generate_all_felt_visualizations.py` implements its resumable batch
renderer.

## Dataset selection and organization

RAVDESS filenames contain seven fields:

```text
modality-vocal_channel-emotion-intensity-statement-repetition-actor
```

The pipeline selects only full audiovisual files (`modality = 01`) and divides
them by vocal channel:

| Subset | Number of trials |
| --- | ---: |
| Speech | 1,440 |
| Song | 1,012 |
| Total | 2,452 |

Video-only (`02-*`) and audio-only (`03-*`) files are ignored during task
construction. They are not deleted by the pipeline. Actor 18 has no RAVDESS
song recordings, which accounts for the unequal speech/song totals.

The seven filename-coded variables are retained in QC manifests and can be
used to summarize results by actor, vocal channel, emotion, intensity,
statement, and repetition. Output files are arranged as:

```text
01_data/02_output/
├── 01_raw_motion/{speech,song}/Actor_XX/*.csv
├── 02_smoothed_motion/{speech,song}/Actor_XX/*.csv
├── 03_smoothed_video/felt_visualization_set/
│   ├── AU_animation/
│   ├── landmark_only/
│   └── Landmark_overlay/
└── qc/
```

## Processing overview

```text
RAVDESS audiovisual videos
          ↓
Py-Feat Detectorv2 framewise extraction
          ↓
Raw tracking CSVs
          ↓
Frame, schema, value, face, identity, and multi-face QC
          ↓
Targeted forward-fill of the reproducible missing model-output row
          ↓
Challis cutoff estimation on the complete raw corpus
          ↓
Zero-phase Butterworth smoothing
          ↓
Smoothing and bounded-score QC
          ↓
Seven visualization videos per trial
```

Raw and smoothed CSVs are kept separately. This preserves the unsmoothed
tracking output for alternative preprocessing and makes the effect of the
chosen filter measurable.

## 1. Raw motion extraction

Raw tracking was performed with Py-Feat 2.0.3 using `Detectorv2`. This is a
scientific model change from the earlier FELT v1.0.0 processing, which used
Py-Feat 0.6.2; it is not merely a Python dependency update. The maintained
environment uses Python 3.11 or newer and the extraction is configured for a
CUDA device when available.

The extraction configuration is:

- output dimensions: 1,280 × 720 pixels;
- identity model: ArcFace;
- face-detection threshold: 0.83;
- face-identity threshold: 0.8;
- nominal sampling rate: approximately 29.97 Hz; and
- `BATCH_SIZE = 1` for the production rerun.

Batch size one was retained because earlier batch-size-five extraction could
produce fewer output rows than decoded video frames. A batch-size-one smoke
run produced exact row/frame agreement for both a speech trial (98 rows) and a
song trial (126 rows). The full rerun completed all 2,452 tasks successfully:
1,440 speech CSVs and 1,012 song CSVs.

Each Detectorv2 CSV contains 2,184 columns on disk, including the CSV index and
metadata fields. The tracked content includes:

- face rectangle and face-detection score;
- 68 two-dimensional landmarks (136 coordinate columns);
- a 478-vertex three-dimensional face mesh (1,434 coordinates);
- head rotation and translation;
- gaze direction;
- 20 Action Unit scores;
- seven emotion scores plus valence and arousal;
- 52 blendshape scores; and
- 512-dimensional identity output plus an identity label.

The `Unnamed: 0`, `frame`, `approx_time`, `FrameHeight`, `FrameWidth`, and `input`
fields provide temporal and source-file provenance. The source path in `input`
is normalized to a portable `Actor_XX/filename.mp4` reference. The cleanup
migration changed only that repeated metadata token in existing CSVs and did
not reserialize numerical values.

## 2. QC during and after extraction

### Frame and schema integrity

The full-corpus Detectorv2 audit found:

- 2,452/2,452 CSVs present and structurally readable;
- 299,854 extracted rows in total;
- no schema-mismatch files;
- no missing required-column files;
- no infinite numeric values;
- no invalid frame values;
- no non-monotonic frame steps; and
- no missing frame gaps.

CSV row counts matched decoded video-frame counts for all 2,452 trials. There
were 832 cases where container metadata disagreed with decoded frame counts;
these were recorded separately and did not represent CSV-versus-decoded-frame
mismatches.

### Multiple detections in a frame

The full Detectorv2 audit found zero files with duplicate frame rows. No
multiple-face row-removal decision was therefore applied to the Py-Feat 2.0.3
corpus. Duplicate-frame detection remains part of the production QC gate
because a file can have complete source-frame coverage while still containing
more than one detected face on some frames.

A separate 24-file multi-face review, including removal of 125 duplicate rows,
belongs to the historical Py-Feat 0.6.2 material under `062/`. Those decisions
must not be described as Detectorv2 processing or applied to FELT v2 output.

### Identity and face-quality diagnostics

The Detectorv2 audit also reported identity fragmentation as a diagnostic. A
total of 1,124 files (45.84%) contained more than one identity label across
their rows, with a maximum of 13 labels in one file. Identity-label changes are
not duplicate frame rows and should not be interpreted as proof that multiple
people were present without frame-level visual review.

Before missing-value repair, one file contained one incomplete model-output
row. The same row had `FaceScore = 0` and a non-positive face box, so the
low-confidence and face-box flags identify the same event rather than separate
failures.

## 3. Missing-value handling

Missingness was audited before modifying any raw CSV. The affected file was:

```text
song/Actor_10/01-02-05-01-01-01-10.csv
```

The incomplete row was source frame 36, approximately one second into the
trial. It contained 2,177 blank model-output cells while frame and source
metadata remained present. The event was reproducible when the source was
rerun with batch size one, so it was not treated as a transient extraction
failure.

The repair script is deliberately narrow. It targets this known file and
frame, retains its source/frame metadata, and replaces the failed model-output
fields from the preceding frame. This includes replacing the nonblank
`FaceScore = 0` failure sentinel as well as filling the 2,177 blank fields. It
then verifies that the repaired model output exactly matches the preceding
frame. It is forward-fill handling, not interpolation, and it modifies the raw
CSV in place. The resulting raw corpus was then finite and suitable for
smoothing.

The pre-repair audit and the repair log should be reported together. Reporting
only the post-repair state would hide the only observed incomplete model-output
event.

## 4. Data-driven smoothing

The previous processing approach used a fixed 6 Hz Butterworth filter followed
by Savitzky–Golay smoothing. The revised pipeline uses one data-driven,
zero-phase Butterworth filter and removes the second Savitzky–Golay stage.

Cutoffs were estimated using the residual-autocorrelation procedure described
by Challis (1999). For each trajectory, candidate final cutoffs from 1.0 to
12.0 Hz were evaluated in 0.25 Hz increments. At each candidate, the residual
was defined as raw minus filtered signal. The selection criterion was the sum
of squared residual autocorrelations over the first 10 nonzero lags. Nonfinite,
near-constant, degenerate, and search-boundary estimates were excluded.

For connected geometric signals, coordinates were pooled by the median within
each video and then by the median across all 2,452 videos. The same cutoff was
therefore applied to every coordinate in a geometric family, avoiding
coordinate-specific attenuation that could warp the face or head geometry.
Action Units and blendshapes were not pooled with geometry; each received an
independent feature-specific cutoff.

The approved geometric master cutoffs were:

| Signal family | Cutoff |
| --- | ---: |
| Face mesh | 3.75 Hz |
| 68-point landmarks | 3.75 Hz |
| Head rotation | 3.75 Hz |
| Head translation | 3.25 Hz |
| Eye gaze | 3.50 Hz |
| Face bounding box | 3.25 Hz |

Filtering used `scipy.signal.sosfiltfilt` with a second-order Butterworth per
pass. Forward-backward application gives zero phase and an effective fourth-
order magnitude response. The design frequency was corrected so the reported
cutoff corresponds to the final zero-phase −3 dB frequency. Because this is an
offline filter, it is noncausal and uses information from both sides of each
frame.

The estimator completed all 2,452 files without file failures. The approved
master was then applied to all 2,452 files, with geometry, AUs, and blendshapes
enabled. The final smoothing run completed without processing errors.

Filtering can move bounded scores slightly outside their theoretical range.
The post-filter clipping audit found 445,802 clipped values across 71 of 72
bounded AU/blendshape columns: 2.06% of the 21,589,488 bounded values audited.
Approximately 96.1% of clipped values were negative excursions; the maximum
undershoot was 0.080 and the maximum overshoot was 0.069. Values were clipped
back to [0, 1] in the smoothed output. This should be reported as a documented
post-filter correction, not described simply as if no out-of-range values had
occurred.

Smoothing QC showed strong raw/smoothed agreement across geometric families:
median correlations were approximately 0.965–0.998, median normalized RMSE
was approximately 0.020–0.083, and median first-difference RMS ratios were
approximately 0.45–0.76. These summaries indicate reduced frame-to-frame
roughness while retaining the broad trajectory shapes.

## 5. Visualization videos

The completed visualization set contains seven H.264 MP4 products per trial,
rendered at 30 frames per second with one output frame for each selected CSV
row. No expression frames are synthesized or interpolated.

The seven views are:

1. Action Unit region heatmap;
2. blendshape region heatmap;
3. AU-to-canonical-mesh animation;
4. landmark-only contour mesh;
5. landmark-only tessellation mesh;
6. landmark-overlay contour mesh; and
7. landmark-overlay tessellation mesh.

The landmark-only videos use recorded frame dimensions and a blank background.
The landmark-overlay videos decode the original RAVDESS frames and draw the
tracking result over them, including the mesh, face box, gaze, head-pose axes,
and source-frame labels. Therefore, source videos are required to regenerate
the overlay views, whereas the score and blank-background views can be
regenerated from the smoothed CSVs plus the visualization models.

The current visualization tree contains 17,164 MP4 files, exactly seven per
trial (7 × 2,452). Existing videos were rendered successfully. The RAVDESS
source MP4s were later restored locally for reproducibility validation, but
they are not tracked in Git or included in derivative release archives.

## 6. Provenance and evidence artifacts

The main v2 evidence is stored under `01_data/02_output/qc/` and includes:

- `ravdess_input_manifest.csv` and `ravdess_input_summary.json`;
- `pyfeat_v2_full/`: immutable pre-correction extraction evidence;
- `pyfeat_v2_post_correction/`: accepted corrected extraction evidence;
- `missing_value_correction_audit.csv`;
- `source_reference_normalization.csv`;
- `challis_smoothing/video_cutoff_estimates.csv` and cutoff summaries;
- `challis_smoothing/smoothing_run_manifest.json`;
- `challis_smoothing/smoothing_validation.csv`; and
- `visualization_outputs.csv` and `visualization_outputs_summary.json`.

The historical 0.6.2 multi-face review manifest and backups are retained under
`01_data/02_output/062/`; they are not v2 evidence.

The processing scripts and environment lockfile are under `02_code/`. The
numbered scripts document the core stages, while
`4_generate_visualizations.py` invokes the resumable batch renderer for the
seven-view visualization set.

## Related technical records

- [Repository README](../README.md)
- [Canonical production pipeline](../PIPELINE.md)
- [Reproducibility contract](../REPRODUCIBILITY.md)
- [Validation record](REPRODUCIBILITY_VALIDATION.md)
- [Processing history](../HISTORY.md)
- [Troubleshooting](../TROUBLESHOOTING.md)
- [Py-Feat v2 extraction QC](PYFEAT_V2_QC.md)
- [Challis smoothing procedure](CHALLIS_SMOOTHING_PROCEDURE.md)
- [FELT v2 data dictionary](../DATA_DICTIONARY.md)
- [Historical Py-Feat 0.6.2 column semantics](archive/OUTPUT_COLUMN_SEMANTICS_PYFEAT_062.md)
