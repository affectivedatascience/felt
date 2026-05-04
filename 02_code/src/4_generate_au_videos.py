"""
Generate FELT Action Unit activation videos.

This script reads smoothed FELT tracking CSV files and generates one Action Unit
activation video per CSV file. Each output video visualizes frame-level AU values
as a Py-Feat face heatmap animation.

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
│       │   └── action_unit_activation/
│       │       ├── speech/
│       │       └── song/
│       └── logs/
└── 02_code/
    └── src/
        ├── 4_generate_au_videos.py
        └── utils/
            ├── __init__.py
            ├── felt_paths.py
            └── video_rendering.py

Note
----
Some frames may raise plotting errors in Py-Feat. These frames are omitted
before the remaining rendered frames are compiled into the output video.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from feat.plotting import plot_face
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
    AU_ACTIVATION_VIDEO_DIR,
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

LOG_FILE = LOG_DIR / "4_generate_au_videos.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Existing video files are skipped so interrupted runs can resume.
SKIP_EXISTING = True

# Keep AU video generation serial by default.
# Py-Feat's AU visualization model is loaded from an HDF5 file, and multiprocessing
# can trigger file-loading/cache errors.
NUM_PROCESSES = 1

# Video rendering settings.
FPS = 30
DPI = 100
CODEC = "libx264"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class AUVideoTask:
    """One smoothed tracking CSV and its corresponding AU activation video path."""

    smoothed_csv_path: Path
    au_video_path: Path
    vocal_channel: str
    actor_name: str


# =============================================================================
# Pipeline functions
# =============================================================================

def build_tasks() -> list[AUVideoTask]:
    """Build one AU-video generation task per smoothed tracking CSV file."""
    tasks: list[AUVideoTask] = []

    for vocal_channel in VOCAL_CHANNELS:
        smoothed_channel_dir = SMOOTHED_MOTION_DIR / vocal_channel
        video_channel_dir = AU_ACTIVATION_VIDEO_DIR / vocal_channel

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
                au_video_path = video_actor_dir / f"{smoothed_csv_path.stem}.mp4"

                tasks.append(
                    AUVideoTask(
                        smoothed_csv_path=smoothed_csv_path,
                        au_video_path=au_video_path,
                        vocal_channel=vocal_channel,
                        actor_name=current_actor_name,
                    )
                )

    logging.info("Prepared %d AU activation video tasks.", len(tasks))
    return tasks


def iter_au_plot_figures(input_prediction, error_frames: list[int]) -> Iterator:
    """Yield Py-Feat AU activation figures from one Fex dataframe.

    Frames that fail during Py-Feat plotting are omitted. This preserves the
    original pipeline behaviour while avoiding accumulation of many open figures.

    Failures come specifically from the AU heatmap overlay (``draw_muscles``),
    not from landmark prediction or the face wireframe (``draw_lineface``).
    """
    total_frames = input_prediction.aus.shape[0]

    for frame in range(total_frames):
        try:
            aus = np.array(input_prediction.aus.iloc[frame])
            muscles = {"all": "heatmap"}
            ax = plot_face(au=aus, muscles=muscles, title=f"Frame {frame}")
            fig = ax.get_figure()
            yield fig

        except ValueError as exc:
            logging.warning("ValueError at frame %d: %s", frame, exc)
            error_frames.append(frame)
            plt.close("all")

        except OSError as exc:
            logging.error("OSError at frame %d while plotting AU face: %s", frame, exc)
            error_frames.append(frame)
            plt.close("all")

        except Exception as exc:
            logging.exception("Unexpected plotting error at frame %d: %s", frame, exc)
            error_frames.append(frame)
            plt.close("all")


def generate_au_video_from_csv(task: AUVideoTask) -> None:
    """Generate one Action Unit activation video from one smoothed CSV file."""
    if SKIP_EXISTING and task.au_video_path.exists():
        logging.info("File already processed, skipping: %s", task.au_video_path)
        return

    logging.info("Loading smoothed CSV: %s", task.smoothed_csv_path)

    video_prediction = read_feat(str(task.smoothed_csv_path))

    logging.info("Generating AU activation video: %s", task.au_video_path)

    error_frames: list[int] = []

    write_figures_to_video(
        figures=iter_au_plot_figures(video_prediction, error_frames),
        output_path=task.au_video_path,
        fps=FPS,
        dpi=DPI,
        codec=CODEC,
    )

    if error_frames:
        logging.warning(
            "AU video saved with %d omitted frames: %s",
            len(error_frames),
            error_frames,
        )
    else:
        logging.info("AU video saved with no omitted frames.")

    logging.info("AU activation video saved to %s", task.au_video_path)


def main() -> None:
    """Generate Action Unit activation videos from smoothed tracking files."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT AU activation video generation.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("SMOOTHED_MOTION_DIR: %s", SMOOTHED_MOTION_DIR)
    logging.info("AU_ACTIVATION_VIDEO_DIR: %s", AU_ACTIVATION_VIDEO_DIR)
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

    print(f"Starting AU activation video generation for {len(tasks)} files.")

    if NUM_PROCESSES == 1:
        for index, task in enumerate(tasks, start=1):
            logging.info("Processing AU task %d/%d: %s", index, len(tasks), task.smoothed_csv_path)
            print(f"[{index}/{len(tasks)}] {task.smoothed_csv_path}")
            generate_au_video_from_csv(task)
    else:
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.map(generate_au_video_from_csv, tasks)

    logging.info("AU activation video generation complete.")
    print("AU activation video generation complete.")


if __name__ == "__main__":
    main()