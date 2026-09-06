# FELT v2 data dictionary

Each raw and smoothed FELT v2 CSV represents one full audiovisual RAVDESS
trial, with one row per decoded source frame. The on-disk schema has 2,184
columns. Column order is stable and is checked across all 2,452 files.

## File and row identity

Output paths follow
`{speech|song}/Actor_XX/<RAVDESS-seven-field-stem>.csv`. The filename fields are
modality, vocal channel, emotion, intensity, statement, repetition, and actor.

| Column | Count | Meaning |
| --- | ---: | --- |
| `Unnamed: 0` | 1 | Saved zero-based pandas row index. It equals `frame` in accepted files. |
| `frame` | 1 | Zero-based decoded source-frame number; contiguous with no gaps or duplicates. |
| `approx_time` | 1 | Py-Feat display timestamp for the frame. Use `frame` and the video rate for precise alignment. |
| `input` | 1 | Portable source identifier, `Actor_XX/filename.mp4`; not a bundled file path. |
| `FrameHeight`, `FrameWidth` | 2 | Processed frame dimensions in pixels; 720 and 1,280 for this release. |

The full release contains 299,854 frame rows. CSV row count equals decoded
source-video frame count for every trial.

## Py-Feat interoperability

FELT CSV files retain the Py-Feat 2.0.3 `Detectorv2` tabular schema and can be
loaded for measurement-based Py-Feat analyses. The `input` column contains a
portable RAVDESS-relative identifier rather than an absolute path from the
extraction machine. This makes the CSVs relocatable, but functions that reopen
the original image or video require the identifier to be rebased to the user's
local RAVDESS directory after loading.

The raw FELT CSVs are compatible Detectorv2 tables with the documented missing
output repair and portable source-reference normalization. The smoothed FELT
CSVs retain that schema but contain FELT-derived filtered measurements; they
are not unmodified Py-Feat detector output.

## Face detection and 2D geometry

| Columns | Count | Meaning and units | Smoothed? |
| --- | ---: | --- | --- |
| `FaceRectX`, `FaceRectY` | 2 | Face-box upper-left position in frame pixels. | Yes, shared face-box cutoff |
| `FaceRectWidth`, `FaceRectHeight` | 2 | Face-box dimensions in pixels. | Yes, shared face-box cutoff |
| `FaceScore` | 1 | Detector confidence score in `[0,1]`; do not treat it as a calibrated probability. | No |
| `x_0..x_67`, `y_0..y_67` | 136 | Detectorv2-derived 68-point landmark coordinates in frame pixels. | Yes, one shared landmark cutoff |

Landmarks are derived by Detectorv2 from its 478-point mesh topology. Slight
out-of-frame coordinates can occur near image boundaries and are diagnostic,
not automatically invalid.

## Head pose and gaze

| Columns | Count | Meaning and units | Smoothed? |
| --- | ---: | --- | --- |
| `Pitch`, `Roll`, `Yaw` | 3 | Head rotation in radians in the Detectorv2 output convention. | Yes, shared rotation cutoff |
| `X`, `Y`, `Z` | 3 | Detectorv2 head-translation/model-space values; no physical distance unit is asserted. | Yes, shared translation cutoff |
| `gaze_pitch`, `gaze_yaw` | 2 | Gaze angles in radians. | Yes, shared gaze cutoff |
| `gaze_angle` | 1 | Scalar angular gaze displacement in radians. | Yes, shared gaze cutoff |

Coordinate signs follow Py-Feat/Detectorv2 conventions. Consumers should not
reinterpret model-space translation as millimetres without an external
calibration.

## Action Units

The 20 AU columns are continuous model scores bounded to `[0,1]`, not FACS
intensity grades and not mutually exclusive probabilities:

```text
AU01 AU02 AU04 AU05 AU06 AU07 AU09 AU10 AU11 AU12
AU14 AU15 AU17 AU20 AU23 AU24 AU25 AU26 AU28 AU43
```

Every AU is filtered independently using its approved cutoff. Forward/backward
filtering can create small boundary excursions; smoothed AU values are clipped
to `[0,1]`, and clipping counts are retained in smoothing QC.

## Affect outputs

| Columns | Count | Meaning | Smoothed? |
| --- | ---: | --- | --- |
| `Neutral`, `Happy`, `Sad`, `Surprise`, `Fear`, `Disgust`, `Anger` | 7 | Detectorv2 emotion scores in `[0,1]`; row sums are approximately one within output precision. | No |
| `valence`, `arousal` | 2 | Continuous Detectorv2 affect estimates. | No |

These are model estimates and should not be treated as RAVDESS ground-truth
labels. Ground-truth emotion and intensity are encoded in the filename.

## Identity outputs

| Columns | Count | Meaning | Smoothed? |
| --- | ---: | --- | --- |
| `Identity_1..Identity_512` | 512 | ArcFace embedding dimensions; finite model-space values with no probability interpretation. | No |
| `Identity` | 1 | Within-video tracker label such as `Person_0`; not the RAVDESS actor ID. | No |

Identity-label fragmentation is a tracker diagnostic. In this release, 1,124
trials contain more than one tracker label even though every frame has one CSV
row. Use the filename actor field—not `Identity`—as participant identity.

## 478-point mesh

| Columns | Count | Meaning and units | Smoothed? |
| --- | ---: | --- | --- |
| `mesh_x_0..mesh_x_477` | 478 | Mesh horizontal coordinates in frame pixels. | Yes |
| `mesh_y_0..mesh_y_477` | 478 | Mesh vertical coordinates in frame pixels. | Yes |
| `mesh_z_0..mesh_z_477` | 478 | Detectorv2 relative-depth/model-space coordinates. | Yes |

All 1,434 mesh coordinates share one cutoff to avoid differential attenuation
that would distort geometry. The vertex numbering follows MediaPipe topology.

## Blendshapes

The 52 Detectorv2/ARKit-style blendshape scores are bounded to `[0,1]`:

```text
_neutral
browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight
eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft
eyeLookInRight eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight
eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight
jawForward jawLeft jawOpen jawRight
mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight
mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft
mouthPressRight mouthPucker mouthRight mouthRollLower mouthRollUpper
mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight
mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight
noseSneerLeft noseSneerRight
```

Each blendshape is filtered independently and clipped to `[0,1]` after
filtering. These scores describe detected facial configuration; they are not
RAVDESS annotations.

## Raw versus smoothed files

Raw CSVs preserve Detectorv2 values except for the one approved missing-output
repair and portable source-reference normalization. Smoothed CSVs change only
the geometric families, AUs, and blendshapes listed above. Other model outputs,
frame identity, source metadata, and embeddings remain unchanged.

The exact cutoffs are in `02_code/config/master_cutoffs_v2.json`. Procedures,
clipping behavior, and QC statistics are documented in
`docs/CHALLIS_SMOOTHING_PROCEDURE.md` and
`docs/DATASET_PROCESSING_WRITEUP.md`.
