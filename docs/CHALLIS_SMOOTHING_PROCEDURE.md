# Challis Cutoff Estimation and Smoothing Plan

## Status and scope

This document is the implementation source of truth for replacing FELT's fixed
6 Hz Butterworth plus Savitzky-Golay smoothing cascade with empirically selected
cutoff frequencies based on the residual-autocorrelation procedure described by
Challis (1999).

It supersedes the exploratory implementation notes used during development.
Those private working notes are not required to understand or reproduce this
procedure.

The goals are to:

1. estimate reproducible cutoff frequencies from FELT raw-motion trajectories;
2. use one master cutoff for each connected geometric group;
3. optionally estimate and apply independent cutoffs to action units and
   blendshapes;
4. apply one zero-phase Butterworth filter, without a second Savitzky-Golay
   stage; and
5. preserve complete provenance and QC records for estimation, filtering, and
   bounded-value clipping.

Implementation status (2026-08-19):

- shared schema, filter-design, residual-autocorrelation, integrity, and
  clipping utilities are implemented in
  `02_code/src/utils/challis_smoothing.py`;
- the resumable estimator is implemented in
  `02_code/src/tools/estimate_challis_cutoffs.py`;
- `02_code/src/3_clean_signals.py` consumes a versioned master artifact and
  emits smoothing/clipping QC;
- unit and end-to-end tests are in
  `02_code/tests/test_challis_smoothing.py`; and
- one-file geometric, ten-file optional-family, and two-worker estimator smoke
  checks have passed.

Full-corpus estimation completed on 2026-08-19:

- all 2,452 raw-motion CSVs completed with zero file failures;
- two-worker estimation took 604 seconds, and aggregation/reporting brought
  total wall time to 690 seconds;
- all six geometric groups and all 72 optional AU/blendshape columns received
  usable master cutoffs;
- geometric master cutoffs were `face_mesh=3.75`, `landmarks_68=3.75`,
  `head_rotation=3.75`, `head_translation=3.25`, `eye_gaze=3.50`, and
  `face_box=3.25` Hz; and
- the official calibration artifacts are under
  `01_data/02_output/qc/challis_smoothing/`.

The approved master was applied to the complete corpus on 2026-08-19 with
geometry, action units, and blendshapes all enabled:

- all 2,452 files completed with zero errors in 314 seconds using two workers;
- an identical 12-worker overwrite run completed with zero errors in 95.8
  seconds, a 3.27x end-to-end speedup, and produced the same clipping totals;
- outputs were written to `01_data/02_output/02_smoothed_motion/`, comprising
  1,440 speech files and 1,012 song files with exact path parity to the raw
  corpus;
- smoothing affected all 2,452 files and post-filter clipping affected 71 of
  the 72 bounded AU/blendshape columns, for 445,802 clipped values in total;
- detailed validation, clipping, error, checkpoint, and run-manifest artifacts
  are under `01_data/02_output/qc/challis_smoothing/`.

The canonical output filters all recoverable signal families. This preserves
the option to construct a geometry-only derivative later by copying the raw AU
and blendshape columns into the smoothed files. The reverse is not possible
without rerunning the filters if AU/blendshape smoothing is omitted now.

References:

- Challis, J. H. (1999). *A procedure for the automatic determination of filter
  cutoff frequency for the processing of biomechanical data*. Journal of
  Applied Biomechanics, 15(3), 303-317.
  <https://doi.org/10.1123/jab.15.3.303>
- Davis, D. J., & Challis, J. H. (2020). *Automatic segment filtering procedure
  for processing non-stationary signals*. Journal of Biomechanics, 101, 109619.
  <https://doi.org/10.1016/j.jbiomech.2020.109619>

The Davis and Challis segment-specific extension is not part of this
implementation. FELT will use fixed master cutoffs for dataset consistency. The
extension may be considered later as a sensitivity analysis.

## Decisions

### Validated input integrity

Previous FELT QC confirmed that the current `01_raw_motion` dataset has:

- continuous frame sequences;
- no missing frames or values in the smoothable trajectories;
- no `FaceScore == 0` rows; and
- the expected one-row-per-frame structure after multi-face QC.

The estimator and smoother will treat these properties as required preconditions.
They will recheck them before processing and fail with a clear file-level error
if a future input violates them. They will not interpolate, mask, or silently
fill data.

### Calibration corpus

The preferred calibration corpus is the complete raw-motion dataset (currently
2,452 CSV files), rather than the 613-file development subset. This removes the
subset's fixed statement-01/repetition-01 limitation and includes all actors,
channels, emotions, intensities, statements, and repetitions available in
RAVDESS.

Before the full run, benchmark a small configurable number of files and report:

- files and trajectories processed;
- candidate frequencies evaluated;
- wall time and peak memory;
- estimated full-dataset wall time; and
- failure and search-boundary rates.

Use the full dataset when the projected wall time is practical. Retain
`01_raw_motion_subset` only for development, correctness tests, and a fallback
calibration run if the full computation is impractical. Any fallback must be
identified in the master-cutoff metadata.

### Filter definition

- Sampling frequency: use the configured nominal RAVDESS rate of 29.97 Hz. The
  current Detectorv2 CSV `approx_time` field has only whole-second resolution,
  so it cannot independently estimate or validate a 29.97 Hz rate. Record the
  configured source explicitly in the master artifact.
- Design: second-order Butterworth per forward pass, represented as
  second-order sections.
- Application: forward-backward `scipy.signal.sosfiltfilt`, producing zero
  phase and an effective fourth-order magnitude response.
- Cutoff meaning: every reported cutoff is the final zero-phase -3 dB cutoff.
  Correct the single-pass design cutoff to account for the squared magnitude
  response of forward-backward filtering.
- Savitzky-Golay: removed from the revised pipeline.

Using the same cutoff within a geometric group gives all coordinates the same
amplitude response. Zero-phase filtering already removes relative phase delay;
the shared cutoff prevents unequal attenuation from distorting geometry.

### Column groups

The following groups receive one shared master cutoff each:

| Key | Columns | Count |
| --- | --- | ---: |
| `face_mesh` | `mesh_x_0..477`, `mesh_y_0..477`, `mesh_z_0..477` | 1,434 |
| `landmarks_68` | `x_0..67`, `y_0..67` | 136 |
| `head_rotation` | `Pitch`, `Roll`, `Yaw` | 3 |
| `head_translation` | `X`, `Y`, `Z` | 3 |
| `eye_gaze` | `gaze_pitch`, `gaze_yaw`, `gaze_angle` | 3 |
| `face_box` | `FaceRectX`, `FaceRectY`, `FaceRectWidth`, `FaceRectHeight` | 4 |

The 20 `AUxx` columns and 52 Detectorv2 blendshape columns each receive their
own master cutoff; they are not pooled with one another. They are
model-generated bounded scores rather than spatial coordinates. Their
estimation remains separately selectable for calibration experiments, but the
approved FELT v2 artifact contains all 72 independent cutoffs and the release
smoothing command filters both families by default.

`FaceScore`, emotion probabilities, valence/arousal, identity embeddings,
frame/timing fields, input paths, and other metadata are never smoothed.

## Challis residual-autocorrelation procedure

### Per-column, per-video calculation

For every included trajectory `x` and candidate final cutoff `f`:

1. apply the corrected zero-phase Butterworth filter to obtain `y_f`;
2. calculate the removed component `e_f = x - y_f`;
3. mean-center `e_f`;
4. calculate its normalized autocorrelation `rho_f(k)` at nonzero lags; and
5. calculate the residual-autocorrelation criterion

   `A(f) = sum(rho_f(k)^2), k = 1..K`.

Lower `A` means that the removed component more closely resembles temporally
uncorrelated noise. Select the candidate with the lowest finite `A`; break exact
ties in favor of the lower cutoff.

Initial benchmark defaults are:

- `K = 10` nonzero lags (about 334 ms at 29.97 fps);
- final effective cutoff grid from 1.0 through 12.0 Hz; and
- 0.25 Hz grid spacing.

All three values must be CLI/config parameters and stored in output metadata.
The benchmark must evaluate whether the grid is sufficiently broad and fine.
If boundary selections are material, expand or revise the grid before the full
run. Do not silently accept boundary estimates into pooling.

### Invalid estimates

A column-level estimate is invalid, excluded from medians, and recorded when:

- the source trajectory contains non-finite values;
- its length is insufficient for the filter padding or autocorrelation lags;
- it is constant or near-constant under a scale-aware tolerance;
- residual energy is numerically degenerate;
- all candidate scores are non-finite;
- the selected candidate lies on either search boundary; or
- filtering raises a numerical error.

Median pooling is not a substitute for detecting these cases. Report invalid
counts and percentages by reason, file, column, group, actor, channel, emotion,
and intensity. If a group has too few valid estimates to support a stable
median, stop rather than create its master cutoff.

An insufficient geometric group blocks creation of the master artifact. An
insufficient optional AU/blendshape does not invalidate otherwise valid
geometric cutoffs: omit that independent cutoff, record it under
`unavailable_independent_cutoffs`, and refuse later application of an optional
family unless every requested column has an available master cutoff.

### Double pooling

Pooling follows the original two-level plan:

1. **Within each video:** take the median of valid per-column selected cutoffs
   for each geometric group. Keep each enabled AU and blendshape cutoff
   independent.
2. **Across videos:** take the median of the valid video-level cutoffs for each
   geometric group and each enabled AU/blendshape.

This makes every video contribute at most one estimate to each final master
cutoff. Save the complete column-level and video-level distributions, not only
the final medians. The summary must include count, invalid count, median, IQR,
minimum, maximum, and search-boundary count.

## Master cutoff artifact

Write a versioned JSON artifact containing:

- schema and procedure version;
- creation timestamp;
- input root and corpus type (`full` or `subset`);
- input file count and a deterministic manifest digest;
- sampling-frequency validation results;
- filter order and zero-phase cutoff convention;
- cutoff grid and autocorrelation-lag settings;
- enabled optional families;
- master cutoff for every geometric group and enabled independent column;
- contributing and invalid estimate counts;
- code revision when available; and
- paths to detailed QC artifacts.

The smoothing stage must consume this file rather than duplicating cutoff
constants in source code. It must reject incompatible or incomplete artifacts.

## Final smoothing and clipping

For every raw CSV:

1. validate schema, timing, continuity, and finite values;
2. copy all columns unchanged;
3. filter each enabled group with its master cutoff;
4. optionally filter each enabled AU/blendshape with its independent cutoff;
5. clip enabled bounded scores to `[0, 1]` only after final filtering;
6. write the smoothed CSV to a separate output tree; and
7. record file-level processing and clipping QC.

Clipping must never occur during cutoff estimation because it would change the
residual-autocorrelation objective. For final application, record at least:

- file and column;
- number and percentage below zero before clipping;
- number and percentage above one before clipping;
- minimum pre-clip value and maximum pre-clip value;
- maximum undershoot and overshoot; and
- number of values actually clipped.

Write both detailed file/column records and aggregate summaries by column,
family, actor, vocal channel, emotion, and intensity. The normal processing log
must state the total number of affected files, columns, and values. Zero clipping
is also a reportable QC result.

## Validation

Before accepting master cutoffs, summarize their distributions by actor, vocal
channel, emotion, intensity, statement, and repetition. In particular, test
whether strong-intensity clips or speech/song produce systematically different
video-level cutoff distributions.

For selected raw-versus-filtered trajectories and for aggregate samples,
calculate:

- RMSE and robust-range NRMSE;
- mean absolute difference;
- correlation;
- variance ratio;
- first-difference RMS ratio;
- high-frequency power reduction;
- peak amplitude change; and
- edge-versus-interior error summaries to expose padding artifacts.

Generate diagnostic plots for representative low, median, and high cutoff
estimates, plus invalid and boundary cases. Inspect expressive peaks and rapid
movements visually before approving global application.

Segment-specific filtering from Davis and Challis (2020) is intentionally out
of scope. Nonstationarity is addressed here through stratified summaries and
peak-preservation validation while retaining fixed master cutoffs.

## Implementation plan

### 1. Shared filtering and schema utilities

Add a focused module under `02_code/src/utils/` containing:

- Detectorv2 column-family definitions;
- corrected zero-phase Butterworth design;
- matrix-oriented SOS filtering;
- residual-autocorrelation scoring;
- scale-aware stationary-series detection;
- input-integrity validation; and
- bounded-score clipping statistics.

Keep numerical functions independent of file I/O so they can be tested with
synthetic arrays.

### 2. Cutoff estimator

Add `02_code/src/tools/estimate_challis_cutoffs.py` with CLI options for:

- input root and output/QC root;
- full dataset, subset, explicit file list, and file limit;
- candidate grid, maximum lag, sampling rate, and filter order;
- AU and blendshape inclusion;
- worker count;
- benchmark-only mode; and
- resume/checkpoint behavior.

Process one CSV at a time to bound memory. For performance, filter an entire
eligible column matrix for each candidate frequency and vectorize residual
autocorrelation across columns instead of calling the filter separately for
every column. Parallelize at the file level only after a single-process
reference run passes.

Write checkpointed per-file results so an interrupted full run can resume
without recomputing completed files. Assemble deterministic detailed QC tables,
summaries, and the master JSON only after all requested files succeed.

### 3. Revise the smoothing stage

Update `02_code/src/3_clean_signals.py` to:

- accept configurable input root, output root, and master JSON paths;
- replace fixed `CUTOFF_FREQ` and Savitzky-Golay settings;
- apply master group/column cutoffs with SOS zero-phase filtering;
- require complete AU and blendshape cutoffs for the official profile and
  filter both families by default;
- perform preflight integrity checks;
- generate clipping QC and aggregate summaries; and
- preserve skip-existing/overwrite and multiprocessing behavior safely.

### 4. Tests

Create an initial `02_code/tests/` suite using Python's built-in `unittest` so
no new test dependency is required. Cover:

- known effective -3 dB response after forward-backward correction;
- residual-autocorrelation score behavior on synthetic signal-plus-noise data;
- stationary and degenerate trajectory rejection;
- boundary selection and tie handling;
- identical filtering within a geometric group;
- double-pooling calculations;
- master JSON validation and round-trip loading;
- integrity-check failures;
- optional AU/blendshape behavior; and
- clipping counts, extrema, and unchanged in-range values.

Also run a one-file end-to-end smoke test that writes only to a temporary output
directory.

### 5. Execution gates

Proceed in this order:

1. unit tests and synthetic numerical checks;
2. one-file estimator and smoothing smoke test;
3. 10-25 file performance benchmark;
4. review projected full-dataset runtime, memory, invalid rates, and boundary
   rates;
5. full-dataset cutoff estimation if practical;
6. review cutoff distributions and validation plots;
7. freeze/version the approved master JSON; and
8. run global smoothing and clipping QC.

Do not overwrite `01_raw_motion`. Do not start the full smoothing run until the
master cutoff artifact and validation report have been reviewed.

## Commands

From `02_code/`, run the test suite:

```text
uv run python -B -m unittest discover -s tests -v
```

Benchmark 25 full-corpus files without creating an official master artifact:

```text
uv run python -B src/tools/estimate_challis_cutoffs.py \
  --benchmark-only \
  --limit 25 \
  --workers 2
```

Estimate geometric cutoffs from the complete raw-motion corpus:

```text
uv run python -B src/tools/estimate_challis_cutoffs.py --workers 2
```

To estimate optional independent AU and blendshape cutoffs as well, add:

```text
--include-action-units --include-blendshapes
```

Apply the tracked, approved FELT v2 artifact to the verified raw corpus:

```text
uv run python -B src/3_clean_signals.py --workers 2
```

The default artifact is `config/master_cutoffs_v2.json`. The stage validates
the complete raw corpus using path-independent size and content digests. For a
diagnostic derivative only, either score family can be left raw with:

```text
--no-filter-action-units --no-filter-blendshapes
```

## Planned output layout

```text
01_data/02_output/
├── 01_raw_motion/
├── 01_raw_motion_subset/
├── 02_smoothed_motion/
├── logs/
│   ├── challis_cutoff_estimation.log
│   └── 3_clean_signals.log
└── qc/
    └── challis_smoothing/
        ├── benchmark.json
        ├── master_cutoffs.json
        ├── column_cutoff_estimates.csv.gz
        ├── video_cutoff_estimates.csv
        ├── cutoff_summary.csv
        ├── cutoff_stratified_summary.csv
        ├── invalid_estimates.csv
        ├── estimation_errors.csv
        ├── smoothing_validation.csv
        ├── clipping_by_file_column.csv
        ├── clipping_summary.csv
        ├── clipping_stratified_summary.csv
        ├── smoothing_errors.csv
        ├── smoothing_run_manifest.json
        ├── checkpoints/
        ├── smoothing_checkpoints/
        └── plots/
```

Large detailed tables may be compressed CSV files to avoid adding another
storage-format dependency. Filenames and schemas should be versioned before the
first full run so future reruns remain comparable.
