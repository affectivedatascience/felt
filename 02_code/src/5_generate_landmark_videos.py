"""
Generate FELT landmark plot videos.

This script reads smoothed FELT tracking CSV files and generates one landmark
plot video per CSV file. Each output video visualizes facial landmarks, face
bounding box, and head pose without rendering the original source video frame.

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
│       │   └── landmark_plot/
│       │       ├── speech/
│       │       └── song/
│       └── logs/
└── 02_code/
    └── src/
        ├── 5_generate_landmark_videos.py
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
    LANDMARK_PLOT_VIDEO_DIR,
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

LOG_FILE = LOG_DIR / "5_generate_landmark_videos.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Existing video files are skipped so interrupted runs can resume.
SKIP_EXISTING = True

# Multiprocessing. Original visualization script used 10 processes.
NUM_PROCESSES = 10

# Video rendering settings.
FPS = 30
DPI = 100
CODEC = "libx264"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class LandmarkPlotTask:
    """One smoothed tracking CSV and its corresponding landmark plot video path."""

    smoothed_csv_path: Path
    landmark_video_path: Path
    vocal_channel: str
    actor_name: str


# =============================================================================
# Pipeline functions
# =============================================================================

def build_tasks() -> list[LandmarkPlotTask]:
    """Build one landmark-plot video task per smoothed tracking CSV file."""
    tasks: list[LandmarkPlotTask] = []

    for vocal_channel in VOCAL_CHANNELS:
        smoothed_channel_dir = SMOOTHED_MOTION_DIR / vocal_channel
        video_channel_dir = LANDMARK_PLOT_VIDEO_DIR / vocal_channel

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
                landmark_video_path = video_actor_dir / f"{smoothed_csv_path.stem}.mp4"

                tasks.append(
                    LandmarkPlotTask(
                        smoothed_csv_path=smoothed_csv_path,
                        landmark_video_path=landmark_video_path,
                        vocal_channel=vocal_channel,
                        actor_name=current_actor_name,
                    )
                )

    logging.info("Prepared %d landmark plot video tasks.", len(tasks))
    return tasks


def generate_landmark_video_from_csv(task: LandmarkPlotTask) -> None:
    """Generate one landmark plot video from one smoothed CSV file."""
    if SKIP_EXISTING and task.landmark_video_path.exists():
        logging.info("File already processed, skipping: %s", task.landmark_video_path)
        return

    logging.info("Loading smoothed CSV: %s", task.smoothed_csv_path)

    video_prediction = read_feat(str(task.smoothed_csv_path))

    logging.info("Generating landmark plot figures for: %s", task.smoothed_csv_path)

    figures = video_prediction.plot_detections(
        faces="landmarks",
        faceboxes=True,
        muscles=False,
        poses=True,
        gazes=False,
        add_titles=False,
        au_barplot=False,
        emotion_barplot=False,
        plot_original_image=False,
    )

    logging.info("Writing landmark plot video: %s", task.landmark_video_path)

    write_figures_to_video(
        figures=figures,
        output_path=task.landmark_video_path,
        fps=FPS,
        dpi=DPI,
        codec=CODEC,
    )

    logging.info("Landmark plot video saved to %s", task.landmark_video_path)


def main() -> None:
    """Generate landmark plot videos from smoothed tracking files."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT landmark plot video generation.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("SMOOTHED_MOTION_DIR: %s", SMOOTHED_MOTION_DIR)
    logging.info("LANDMARK_PLOT_VIDEO_DIR: %s", LANDMARK_PLOT_VIDEO_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("Vocal channels: %s", ", ".join(VOCAL_CHANNELS))
    logging.info("NUM_PROCESSES: %d", NUM_PROCESSES)
    logging.info("FPS: %d", FPS)
    logging.info("DPI: %d", DPI)
    logging.info("CODEC: %s", CODEC)

    tasks = build_tasks()

    if not tasks:
        logging.warning("No smoothed CSV files were found. Check SMOOTHED_MOTION_DIR.")
        print("No smoothed CSV files were found. Check SMOOTHED_MOTION_DIR.")
        return

    print(f"Starting landmark plot video generation for {len(tasks)} files.")

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(generate_landmark_video_from_csv, tasks)

    logging.info("Landmark plot video generation complete.")
    print("Landmark plot video generation complete.")


if __name__ == "__main__":
    main()