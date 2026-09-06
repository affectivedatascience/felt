# Historical Py-Feat 0.6.2 output-column semantics

This archived note documents the old FELT/Py-Feat 0.6.2 schema. It does not
describe the Detectorv2 FELT v2 release; use `../../DATA_DICTIONARY.md` for the
active schema.

The FELT pipeline currently uses Py-Feat `0.6.2` with the modular v1 detector configuration:

- `face_model="img2pose"`
- `landmark_model="mobilenet"`
- `au_model="xgb"`
- `emotion_model="resmasknet"`
- `facepose_model="img2pose-c"`
- `identity_model="facenet"`
- `output_size=(720, 1280)`

Py-Feat documentation now names this modular model-chain `Detectorv1`. The current Py-Feat docs state that Detectorv1 produces classic 68-point landmarks and runs separate models such as XGBoost for AUs and ResMaskNet for emotions. The local package source is still the controlling source for this project because the environment is pinned to Py-Feat `0.6.2`.

## Sources Used

- Py-Feat project docs and model list: https://py-feat.org/pages/models.html
- Py-Feat video tutorial, including output examples and batching behavior: https://py-feat.org/basic_tutorials/Detecting_Videos/
- Py-Feat repository: https://github.com/cosanlab/py-feat
- Py-Feat paper: Cheong et al., 2023, *Py-Feat: Python Facial Expression Analysis Toolbox*, https://doi.org/10.1007/s42761-023-00191-4
- Local installed source:
  - `02_code/.venv/Lib/site-packages/feat/utils/__init__.py`
  - `02_code/.venv/Lib/site-packages/feat/detector.py`
  - `02_code/.venv/Lib/site-packages/feat/facepose_detectors/img2pose/img2pose_test.py`
  - `02_code/.venv/Lib/site-packages/feat/au_detectors/StatLearning/SL_test.py`
  - `02_code/.venv/Lib/site-packages/feat/emo_detectors/ResMaskNet/resmasknet_test.py`
  - `02_code/.venv/Lib/site-packages/feat/identity_detectors/facenet/facenet_test.py`
  - `02_code/.venv/Lib/site-packages/feat/landmark_detectors/mobilefacenet_test.py`
- Relevant model cards:
  - XGB AU: https://huggingface.co/py-feat/xgb_au
  - ResMaskNet: https://huggingface.co/py-feat/resmasknet
  - FaceNet: https://huggingface.co/py-feat/facenet

## Column Groups

### `frame`

Py-Feat defines `FEAT_TIME_COLUMNS = ["frame"]`. In this pipeline, `feat_prediction.to_csv(csv_path)` writes the DataFrame index as the first CSV column. Because the Py-Feat output also contains a `frame` column, reading the CSV with pandas commonly yields `frame` and `frame.1`.

QC interpretation:

- Treat the first CSV column as the source-frame row index emitted by Py-Feat.
- Treat the later `frame`/`frame.1` field as Py-Feat's frame counter.
- For the `BATCH_SIZE=1` rerun, both should be integer, monotonic, nonduplicated, gap-free, and should match each other.
- Expected range is `0` to `source_video_decoded_frame_count - 1`.

### Face Box and `FaceScore`

Py-Feat defines face-box columns as:

```text
FaceRectX, FaceRectY, FaceRectWidth, FaceRectHeight, FaceScore
```

For the configured `img2pose` face model, local source returns detection boxes as `[x1, y1, x2, y2, score]`, then Py-Feat converts them to `FaceRectX`, `FaceRectY`, width, height, and `FaceScore`. The `img2pose` wrapper filters detections by `detection_threshold`; FELT passes `face_detection_threshold=0.83`.

QC interpretation:

- `FaceRectX` and `FaceRectY` are pixel coordinates for the face-box origin in the output frame.
- `FaceRectWidth` and `FaceRectHeight` are pixel dimensions.
- `FaceScore` is the face-detection score/confidence used for thresholding. It should not be described as a calibrated probability unless a model-specific source supports calibration.
- Expected source-truth range for `FaceScore` is `[0, 1]` because it is a detection score used in threshold comparisons and emitted by the detection model as `pred["scores"]`.
- Box bounds are pragmatic QC checks: `FaceRectWidth > 0`, `FaceRectHeight > 0`, and the box should usually lie within `[0, 1280] x [0, 720]` for FELT outputs.

### Landmarks

Py-Feat uses 68 2D landmark points with columns:

```text
x_0 ... x_67, y_0 ... y_67
```

The local source defines these as `openface_2d_landmark_columns`. The `mobilenet` landmark model is loaded from Py-Feat's landmark model family. The current Py-Feat docs describe Detectorv1 landmarks as 68-point dlib-style landmarks, and the model page says `mobilenet` weights are adapted from `cunjian/pytorch_face_landmark`.

QC interpretation:

- Values are pixel coordinates after Py-Feat maps model output back into the video/output frame.
- For FELT `output_size=(720, 1280)`, pragmatic expected ranges are `x_*` within `[0, 1280]` and `y_*` within `[0, 720]`.
- Slight out-of-frame values are diagnostic rather than automatically invalid if the face box itself is near or outside an image edge, but they should be counted and reviewed.

### Head Pose

Py-Feat defines 3D face-pose columns as:

```text
Pitch, Roll, Yaw
```

The local `img2pose` wrapper converts the model rotation vector to Euler angles using `degrees=True`, then returns `[pitch, roll, yaw]`. The `Img2Pose` class docstring states the constrained model is optimized for front-facing faces in the `[-90, 90]` degree range; the unconstrained model can detect faces at any angle. FELT uses the constrained `img2pose` path.

QC interpretation:

- Values are Euler angles in degrees.
- `abs(Pitch/Roll/Yaw) > 90` should be treated as an extreme-pose diagnostic, not automatically a data error.
- A conservative hard sanity range is `[-180, 180]` for Euler angles; values outside that range should be investigated.

### Action Units

Configured model: `au_model="xgb"`.

The local XGBoost AU detector returns `classifier.predict_proba(... )[:, 1]` for each AU. The 20 emitted AUs are:

```text
AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU11, AU12,
AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU28, AU43
```

The current Py-Feat model docs state that XGB AU values are continuous probability predictions. They also note a specific caveat for AU07 in current Py-Feat docs: AU07 may appear more binary in practice and should be interpreted as the proportion of decision trees with a detection rather than average decision-tree confidence. The local `0.6.2` source still confirms the general XGBoost probability-output path.

QC interpretation:

- Values are AU presence probabilities/scores, not FACS intensity ratings.
- Expected range is `[0, 1]`.
- Rowwise sums across AUs are not meaningful because AUs are separate binary/multi-label detections, not mutually exclusive classes.

### Emotions

Configured model: `emotion_model="resmasknet"`.

Py-Feat emotion columns are:

```text
anger, disgust, fear, happiness, sadness, surprise, neutral
```

The local ResMaskNet source applies `torch.softmax(..., dim=1)` and returns a probability array. Its docstring says it returns predicted emotions in probability order `[angry, disgust, fear, happy, sad, surprise, neutral]`.

QC interpretation:

- Values are softmax probabilities for seven mutually exclusive emotion classes.
- Expected range is `[0, 1]`.
- Rowwise emotion sums are expected to be approximately `1`, allowing small floating-point tolerance.

### Identity

Configured model: `identity_model="facenet"`.

Py-Feat defines identity columns as:

```text
Identity, Identity_1 ... Identity_512
```

In local detector output assembly, `Identity` is populated with `NaN` and the remaining 512 columns receive the FaceNet embedding vector. The model card states that FaceNet exposes a 512-dimensional latent facial embedding space.

QC interpretation:

- `Identity_1` through `Identity_512` are embedding dimensions, not probabilities or confidence scores.
- No fixed finite min/max range should be enforced for identity embeddings.
- Check only for numeric finite values after successful face/identity detection, unless identity is intentionally unused.
- The leading `Identity` column is expected to be `NaN` in this Py-Feat `0.6.2` path and should not be counted as a missing-value defect without special handling.

### `input`

The `input` column stores the path to the source image or video used for detection. It is a string/path field and should not be part of numeric range checks.

### `approx_time`

Py-Feat adds `approx_time` for video outputs. It is a timestamp-like field derived from the frame counter/video timing, not a model prediction. It should be used for temporal-integrity checks, not numeric feature range checks.

## Practical QC Rules

Use these source-truth rules in later QC tools:

- Treat `FaceScore`, AUs, and emotions as bounded `[0, 1]` model scores/probabilities.
- Describe `FaceScore` as detection confidence/score, not a calibrated probability.
- Describe XGB AUs as AU presence probabilities/scores, not FACS intensity.
- Describe ResMaskNet emotions as softmax probabilities; check rowwise sums against `1`.
- Treat landmarks and face boxes as pixel coordinates/dimensions in the `1280 x 720` output frame.
- Treat `Pitch`, `Roll`, and `Yaw` as degrees; flag `abs(value) > 90` as extreme pose and values outside `[-180, 180]` as stronger sanity-check failures.
- Exclude `Identity` from ordinary missing-value defect counts unless explicitly reviewing identity output.
- Do not run numeric range checks on `input` or timestamp parsing checks as if they were model scores.
- For RAVDESS, expect one true actor face per source frame. Duplicate rows with the same `frame` value should be classified as multiple-face detection artifacts requiring visual adjudication, not as missing-frame defects. FaceScore can support a suggested keep candidate, but the final `Person_0` / `Person_1` decision should come from overlay review.
