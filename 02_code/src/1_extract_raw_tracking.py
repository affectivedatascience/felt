"""
Run Py-Feat facial tracking on RAVDESS audiovisual videos.

This script generates the raw FELT facial-tracking CSV files from the RAVDESS
full-audiovisual speech and song videos. It iterates over actor folders, selects
valid full-AV speech/song files using the RAVDESS filename convention, runs
Py-Feat on each selected video, and writes one raw tracking CSV per source file.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   ├── 01_input/
│   │   ├── Actor_01/
│   │   ├── Actor_02/
│   │   └── ...
│   └── 02_output/
│       ├── 01_raw_motion/
│       │   ├── speech/
│       │   │   ├── Actor_01/
│       │   │   └── ...
│       │   └── song/
│       │       ├── Actor_01/
│       │       └── ...
│       └── logs/
└── 02_code/
    └── src/
        ├── 1_extract_raw_tracking.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py

RAVDESS filename convention
---------------------------
Each trial filename contains seven hyphen-separated fields:

    modality-vocal_channel-emotion-intensity-statement-repetition-actor

Relevant fields:
    modality:      01 = full-AV, 02 = video-only, 03 = audio-only
    vocal channel: 01 = speech, 02 = song

This script processes full-AV speech/song files only. Video-only and audio-only
files are skipped. Actor 18 has no song recordings in RAVDESS.
"""

# =============================================================================
# Original Py-Feat environment note
# =============================================================================
#
# FELT v1.0.0 was generated with Py-Feat 0.6.2 under Python 3.9.
# The original working environment used PyTorch 2.2.0 / torchvision 0.17.0 /
# torchaudio 2.2.0 with CUDA 12.1 wheels.
#
# A local patch was applied to feat/detector.py in Py-Feat 0.6.2. In the
# detect_identity() function, the return statement was changed from:
#
#     return self._convert_detector_output(facebox, face_embeddings.numpy())
#
# to:
#
#     return self._convert_detector_output(facebox, face_embeddings.detach().numpy())
#
# This patch detaches the PyTorch tensor before conversion to NumPy. Exact
# reproduction of the released FELT tracking files may require the same patched
# Py-Feat 0.6.2 environment.

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Make the local utils/ package importable when this script is run directly.
# =============================================================================

CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")


# =============================================================================
# Py-Feat import configuration
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning,
)

# Py-Feat device. Use "cuda" on a CUDA-enabled GPU; otherwise use "cpu".
DEVICE = "cpu"

# Limit native-library threading only for CPU execution.
# This avoids XGBoost/OpenMP stalls observed with Py-Feat 0.6.2 on macOS CPU.
# IMPORTANT: these must be set before importing Py-Feat / XGBoost / Torch.
if DEVICE == "cpu":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from feat import Detector

from utils.felt_paths import (
    INPUT_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    actor_name,
    configure_logging,
    parse_ravdess_stem,
    should_process_full_av_speech_song,
    vocal_channel_folder,
)


# =============================================================================
# User-editable configuration
# =============================================================================

# Input RAVDESS videos.
RAVDESS_INPUT_DIR = INPUT_DIR

# Logs.
LOG_FILE = LOG_DIR / "1_extract_raw_tracking.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Optional test mode: process only the first valid full-AV speech/song video found.
PROCESS_FIRST_VIDEO_ONLY = False

# Existing output CSV files are skipped so interrupted runs can resume.
SKIP_EXISTING = True


# =============================================================================
# Py-Feat detector configuration
# =============================================================================

FACE_MODEL = "img2pose"
LANDMARK_MODEL = "mobilenet"
AU_MODEL = "xgb"
EMOTION_MODEL = "resmasknet"
FACEPOSE_MODEL = "img2pose-c"
IDENTITY_MODEL = "facenet"

# Py-Feat video-detection parameters used for FELT.
# Py-Feat output size is specified as (height, width).
OUTPUT_SIZE = (720, 1280)
BATCH_SIZE = 5
NUM_WORKERS = 0
PIN_MEMORY = False
FACE_DETECTION_THRESHOLD = 0.83
FACE_IDENTITY_THRESHOLD = 0.8


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class DetectionTask:
    """A source video and its corresponding output CSV path."""

    video_path: Path
    csv_path: Path


# =============================================================================
# Pipeline functions
# =============================================================================

def create_detector() -> Detector:
    """Create the Py-Feat detector used for FELT tracking."""
    logging.info("Loading Py-Feat Detector.")

    detector = Detector(
        face_model=FACE_MODEL,
        landmark_model=LANDMARK_MODEL,
        au_model=AU_MODEL,
        emotion_model=EMOTION_MODEL,
        facepose_model=FACEPOSE_MODEL,
        identity_model=IDENTITY_MODEL,
        device=DEVICE,
        n_jobs=1,
        verbose=False,
    )

    logging.info("Py-Feat Detector loaded.")
    return detector


def build_tasks() -> list[DetectionTask]:
    """Build detection tasks for valid RAVDESS full-AV speech/song videos."""
    tasks: list[DetectionTask] = []

    for actor_id in range(START_ACTOR, END_ACTOR + 1):
        current_actor_name = actor_name(actor_id)
        actor_video_dir = RAVDESS_INPUT_DIR / current_actor_name

        if not actor_video_dir.exists():
            logging.warning("Actor input directory not found; skipping: %s", actor_video_dir)
            continue

        for video_path in sorted(actor_video_dir.glob("*.mp4")):
            try:
                code = parse_ravdess_stem(video_path)

                if not should_process_full_av_speech_song(code):
                    logging.debug("Skipping non-target file: %s", video_path)
                    continue

                channel_folder = vocal_channel_folder(code)

            except ValueError as exc:
                logging.warning(
                    "Skipping file with invalid RAVDESS filename: %s; %s",
                    video_path,
                    exc,
                )
                continue

            actor_csv_dir = RAW_MOTION_DIR / channel_folder / current_actor_name
            actor_csv_dir.mkdir(parents=True, exist_ok=True)

            csv_path = actor_csv_dir / f"{video_path.stem}.csv"
            tasks.append(DetectionTask(video_path=video_path, csv_path=csv_path))

            if PROCESS_FIRST_VIDEO_ONLY:
                logging.info("Single-file test mode enabled.")
                logging.info("Selected test video: %s", video_path)
                logging.info("Selected output CSV: %s", csv_path)
                return tasks

    logging.info("Prepared %d detection tasks.", len(tasks))
    return tasks


def save_prediction_csv(feat_prediction, csv_path: Path) -> None:
    """Save a Py-Feat prediction dataframe to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the previous script's default pandas behavior.
    # This writes the DataFrame index unless Py-Feat suppresses it internally.
    feat_prediction.to_csv(csv_path)

    logging.info("Output saved to %s", csv_path)
    print(f"Output saved to: {csv_path}")


def run_detection(detector: Detector, task: DetectionTask) -> None:
    """Run Py-Feat detection for one video file and save the CSV output."""
    if SKIP_EXISTING and task.csv_path.exists():
        logging.info("Output already exists; skipping: %s", task.csv_path)
        print(f"Output already exists; skipping: {task.csv_path}")
        return

    try:
        logging.info("Running detection for file: %s", task.video_path)
        print(f"Running detection for file: {task.video_path}")

        video_prediction = detector.detect_video(
            str(task.video_path),
            skip_frames=None,
            output_size=OUTPUT_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            face_detection_threshold=FACE_DETECTION_THRESHOLD,
            face_identity_threshold=FACE_IDENTITY_THRESHOLD,
        )

        save_prediction_csv(video_prediction, task.csv_path)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        raise

    except Exception:
        logging.exception("Error processing file: %s", task.video_path)
        print(f"Error processing file. See log: {LOG_FILE}")


def main() -> None:
    """Run raw facial-tracking extraction."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting Py-Feat RAVDESS raw tracking extraction.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("RAVDESS_INPUT_DIR: %s", RAVDESS_INPUT_DIR)
    logging.info("RAW_MOTION_DIR: %s", RAW_MOTION_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("DEVICE: %s", DEVICE)

    detector = create_detector()
    tasks = build_tasks()

    if not tasks:
        logging.warning("No detection tasks were prepared. Check RAVDESS_INPUT_DIR and filenames.")
        print("No detection tasks were prepared. Check RAVDESS_INPUT_DIR and filenames.")
        return

    logging.info("Starting detection loop for %d task(s).", len(tasks))
    print(f"Starting detection loop for {len(tasks)} task(s).")

    for index, task in enumerate(tasks, start=1):
        logging.info("Processing task %d/%d: %s", index, len(tasks), task.video_path)
        print(f"[{index}/{len(tasks)}] {task.video_path}")
        run_detection(detector, task)

    logging.info("Raw tracking extraction complete.")
    print("Raw tracking extraction complete.")


if __name__ == "__main__":
    main()