"""
Generate FELT landmark overlay videos.

This script reads smoothed FELT tracking CSV files and generates one landmark
overlay video per CSV file. Each output video renders facial landmarks, face
bounding box, and head pose over the original source video frames.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   └── 02_output/
│       ├── 02_smoothed_motion/
│       │   ├── speech/
│       │   │   ├── Actor_01/
│       │   │   └── ...
│       │   └── song/
│       │       ├── Actor_01/
│       │       └── ...
│       ├── 03_smoothed_video/
│       │   └── landmark_overlay/
│       │       ├── speech/
│       │       └── song/
│       └── logs/
└── 02_code/
    └── src/
        ├── 6_generate_overlay_videos.py
        └── utils/
            ├── __init__.py
            ├── felt_paths.py
            └── video_rendering.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from feat.utils.io import read_feat


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

from utils.felt_paths import (
    LANDMARK_OVERLAY_VIDEO_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    SMOOTHED_MOTION_DIR,
    VOCAL_CHANNELS,
    actor_name,
    configure_logging,
)
from utils.video_rendering import write_figures_to_video


# =============================================================================
# User-editable configuration
# =============================================================================

LOG_FILE = LOG_DIR / "6_generate_overlay_videos.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Existing video files are skipped so interrupted runs can resume.
SKIP_EXISTING = True

# Multiprocessing. Original overlay script used 8 processes.
NUM_PROCESSES = 8

# Video rendering settings.
FPS = 30
DPI = 100
CODEC = "libx264"

# Original RAVDESS video dimensions.
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class LandmarkOverlayTask:
    """One smoothed tracking CSV and its corresponding landmark overlay video path."""

    smoothed_csv_path: Path
    overlay_video_path: Path
    vocal_channel: str
    actor_name: str


# =============================================================================
# Pipeline functions
# =============================================================================

def build_tasks() -> list[LandmarkOverlayTask]:
    """Build one landmark-overlay video task per smoothed tracking CSV file."""
    tasks: list[LandmarkOverlayTask] = []

    for vocal_channel in VOCAL_CHANNELS:
        smoothed_channel_dir = SMOOTHED_MOTION_DIR / vocal_channel
        video_channel_dir = LANDMARK_OVERLAY_VIDEO_DIR / vocal_channel

        if not smoothed_channel_dir.exists():
            logging.warning(
                "Smoothed-motion channel directory not found; skipping: %s",
                smoothed_channel_dir,
            )
            continue

        for actor_id in range(START_ACTOR, END_ACTOR + 1):
            current_actor_name = actor_name(actor_id)
            smoothed_actor_dir = smoothed_channel_dir / current_actor_name
            video_actor_dir = video_channel_dir / current_actor_name

            if not smoothed_actor_dir.exists():
                logging.warning(
                    "Smoothed-motion actor directory not found; skipping: %s",
                    smoothed_actor_dir,
                )
                continue

            video_actor_dir.mkdir(parents=True, exist_ok=True)

            for smoothed_csv_path in sorted(smoothed_actor_dir.glob("*.csv")):
                overlay_video_path = video_actor_dir / f"{smoothed_csv_path.stem}.mp4"

                tasks.append(
                    LandmarkOverlayTask(
                        smoothed_csv_path=smoothed_csv_path,
                        overlay_video_path=overlay_video_path,
                        vocal_channel=vocal_channel,
                        actor_name=current_actor_name,
                    )
                )

    logging.info("Prepared %d landmark overlay video tasks.", len(tasks))
    return tasks


def generate_overlay_video_from_csv(task: LandmarkOverlayTask) -> None:
    """Generate one landmark overlay video from one smoothed CSV file."""
    if SKIP_EXISTING and task.overlay_video_path.exists():
        logging.info("File already processed, skipping: %s", task.overlay_video_path)
        return

    logging.info("Loading smoothed CSV: %s", task.smoothed_csv_path)

    video_prediction = read_feat(str(task.smoothed_csv_path))

    logging.info("Generating landmark overlay figures for: %s", task.smoothed_csv_path)

    figures = video_prediction.plot_detections(
        faces="landmarks",
        faceboxes=True,
        muscles=False,
        poses=True,
        gazes=False,
        add_titles=False,
        au_barplot=False,
        emotion_barplot=False,
        plot_original_image=True,
    )

    logging.info("Writing landmark overlay video: %s", task.overlay_video_path)

    write_figures_to_video(
        figures=figures,
        output_path=task.overlay_video_path,
        fps=FPS,
        dpi=DPI,
        codec=CODEC,
        set_size=(VIDEO_WIDTH, VIDEO_HEIGHT),
    )

    logging.info("Landmark overlay video saved to %s", task.overlay_video_path)


def main() -> None:
    """Generate landmark overlay videos from smoothed tracking files."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT landmark overlay video generation.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("SMOOTHED_MOTION_DIR: %s", SMOOTHED_MOTION_DIR)
    logging.info("LANDMARK_OVERLAY_VIDEO_DIR: %s", LANDMARK_OVERLAY_VIDEO_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("Vocal channels: %s", ", ".join(VOCAL_CHANNELS))
    logging.info("NUM_PROCESSES: %d", NUM_PROCESSES)
    logging.info("FPS: %d", FPS)
    logging.info("DPI: %d", DPI)
    logging.info("CODEC: %s", CODEC)
    logging.info("Video size: %dx%d", VIDEO_WIDTH, VIDEO_HEIGHT)

    tasks = build_tasks()

    if not tasks:
        logging.warning("No smoothed CSV files were found. Check SMOOTHED_MOTION_DIR.")
        print("No smoothed CSV files were found. Check SMOOTHED_MOTION_DIR.")
        return

    print(f"Starting landmark overlay video generation for {len(tasks)} files.")

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(generate_overlay_video_from_csv, tasks)

    logging.info("Landmark overlay video generation complete.")
    print("Landmark overlay video generation complete.")


if __name__ == "__main__":
    main()