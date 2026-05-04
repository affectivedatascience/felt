"""
Filter and smooth FELT raw tracking CSV files.

This script reads raw Py-Feat tracking CSV files, applies a low-pass Butterworth
filter followed by Savitzky-Golay smoothing to selected tracking columns, and
writes one smoothed CSV file per raw CSV.

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
│       ├── 02_smoothed_motion/
│       │   ├── speech/
│       │   └── song/
│       └── logs/
└── 02_code/
    └── src/
        ├── 3_clean_signals.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from feat.utils.io import read_feat
from scipy.signal import butter, filtfilt, savgol_filter


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
    LOG_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    SMOOTHED_MOTION_DIR,
    VOCAL_CHANNELS,
    actor_name,
    configure_logging,
)


# =============================================================================
# User-editable configuration
# =============================================================================

# Logs.
LOG_FILE = LOG_DIR / "3_clean_signals.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Existing smoothed CSV files are skipped so interrupted runs can resume.
SKIP_EXISTING = True

# Multiprocessing. The original script used 10 processes.
NUM_PROCESSES = 10

# Low-pass Butterworth filter parameters.
CUTOFF_FREQ = 6
SAMPLING_FREQ = 29.97
BUTTERWORTH_ORDER = 5

# Savitzky-Golay filter parameters.
SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLY_ORDER = 5


# =============================================================================
# Columns to filter
# =============================================================================

COLUMNS_TO_FILTER = [
    "FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight",
    "x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9", "x_10",
    "x_11", "x_12", "x_13", "x_14", "x_15", "x_16", "x_17", "x_18", "x_19", "x_20",
    "x_21", "x_22", "x_23", "x_24", "x_25", "x_26", "x_27", "x_28", "x_29", "x_30",
    "x_31", "x_32", "x_33", "x_34", "x_35", "x_36", "x_37", "x_38", "x_39", "x_40",
    "x_41", "x_42", "x_43", "x_44", "x_45", "x_46", "x_47", "x_48", "x_49", "x_50",
    "x_51", "x_52", "x_53", "x_54", "x_55", "x_56", "x_57", "x_58", "x_59", "x_60",
    "x_61", "x_62", "x_63", "x_64", "x_65", "x_66", "x_67",
    "y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6", "y_7", "y_8", "y_9", "y_10",
    "y_11", "y_12", "y_13", "y_14", "y_15", "y_16", "y_17", "y_18", "y_19", "y_20",
    "y_21", "y_22", "y_23", "y_24", "y_25", "y_26", "y_27", "y_28", "y_29", "y_30",
    "y_31", "y_32", "y_33", "y_34", "y_35", "y_36", "y_37", "y_38", "y_39", "y_40",
    "y_41", "y_42", "y_43", "y_44", "y_45", "y_46", "y_47", "y_48", "y_49", "y_50",
    "y_51", "y_52", "y_53", "y_54", "y_55", "y_56", "y_57", "y_58", "y_59", "y_60",
    "y_61", "y_62", "y_63", "y_64", "y_65", "y_66", "y_67",
    "Pitch", "Roll", "Yaw",
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class SmoothTask:
    """One raw tracking CSV and its corresponding smoothed output CSV."""

    raw_csv_path: Path
    smoothed_csv_path: Path
    vocal_channel: str
    actor_name: str


# =============================================================================
# Pipeline functions
# =============================================================================

def apply_lowpass_filter(
    column: pd.Series,
    cutoff_freq: float,
    sampling_freq: float,
    order: int = 5,
):
    """Apply a low-pass Butterworth filter to one column of data."""
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, column)
    return filtered_data


def apply_savgol(column: pd.Series, window_length: int, poly_order: int):
    """Apply a Savitzky-Golay filter to one column of data.

    Preserves original behaviour: if the column is too short for the window,
    return the original column.
    """
    if len(column) > window_length:
        return savgol_filter(column, window_length, poly_order)
    return column


def build_tasks() -> list[SmoothTask]:
    """Build one smoothing task per raw tracking CSV file."""
    tasks: list[SmoothTask] = []

    for vocal_channel in VOCAL_CHANNELS:
        raw_channel_dir = RAW_MOTION_DIR / vocal_channel
        smoothed_channel_dir = SMOOTHED_MOTION_DIR / vocal_channel

        if not raw_channel_dir.exists():
            logging.warning("Raw-motion channel directory not found; skipping: %s", raw_channel_dir)
            continue

        for actor_id in range(START_ACTOR, END_ACTOR + 1):
            current_actor_name = actor_name(actor_id)
            raw_actor_dir = raw_channel_dir / current_actor_name
            smoothed_actor_dir = smoothed_channel_dir / current_actor_name

            if not raw_actor_dir.exists():
                logging.warning("Raw-motion actor directory not found; skipping: %s", raw_actor_dir)
                continue

            smoothed_actor_dir.mkdir(parents=True, exist_ok=True)

            for raw_csv_path in sorted(raw_actor_dir.glob("*.csv")):
                smoothed_csv_path = smoothed_actor_dir / raw_csv_path.name

                tasks.append(
                    SmoothTask(
                        raw_csv_path=raw_csv_path,
                        smoothed_csv_path=smoothed_csv_path,
                        vocal_channel=vocal_channel,
                        actor_name=current_actor_name,
                    )
                )

    logging.info("Prepared %d smoothing tasks.", len(tasks))
    return tasks


def filter_and_smooth(task: SmoothTask) -> None:
    """Filter and smooth one raw CSV, then save the smoothed CSV."""
    raw_csv_path = task.raw_csv_path
    smoothed_csv_path = task.smoothed_csv_path

    if SKIP_EXISTING and smoothed_csv_path.exists():
        logging.info("File already processed, skipping: %s", smoothed_csv_path)
        return

    logging.info("Running smoothing for file: %s", raw_csv_path)

    input_prediction = read_feat(str(raw_csv_path))
    df_smooth = input_prediction.copy()

    missing_columns = [col for col in COLUMNS_TO_FILTER if col not in df_smooth.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns in {raw_csv_path}: {missing_columns}")

    # Preserve the original pipeline behaviour:
    # 1. low-pass Butterworth filter
    # 2. Savitzky-Golay smoothing
    df_smooth[COLUMNS_TO_FILTER] = df_smooth[COLUMNS_TO_FILTER].apply(
        lambda x: apply_lowpass_filter(
            x,
            cutoff_freq=CUTOFF_FREQ,
            sampling_freq=SAMPLING_FREQ,
            order=BUTTERWORTH_ORDER,
        )
    )

    df_smooth[COLUMNS_TO_FILTER] = df_smooth[COLUMNS_TO_FILTER].apply(
        apply_savgol,
        args=(SAVGOL_WINDOW_LENGTH, SAVGOL_POLY_ORDER),
    )

    smoothed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_smooth.to_csv(smoothed_csv_path, index=False)

    logging.info("Output saved to %s", smoothed_csv_path)


def main() -> None:
    """Run filtering and smoothing for raw tracking files."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT filtering and smoothing.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("RAW_MOTION_DIR: %s", RAW_MOTION_DIR)
    logging.info("SMOOTHED_MOTION_DIR: %s", SMOOTHED_MOTION_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("Vocal channels: %s", ", ".join(VOCAL_CHANNELS))
    logging.info("NUM_PROCESSES: %d", NUM_PROCESSES)
    logging.info(
        "Butterworth: cutoff=%s Hz, sampling_freq=%s Hz, order=%s",
        CUTOFF_FREQ,
        SAMPLING_FREQ,
        BUTTERWORTH_ORDER,
    )
    logging.info(
        "Savitzky-Golay: window_length=%s, poly_order=%s",
        SAVGOL_WINDOW_LENGTH,
        SAVGOL_POLY_ORDER,
    )

    tasks = build_tasks()

    if not tasks:
        logging.warning("No raw CSV files were found. Check RAW_MOTION_DIR.")
        print("No raw CSV files were found. Check RAW_MOTION_DIR.")
        return

    print(f"Starting filtering and smoothing for {len(tasks)} files.")

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(filter_and_smooth, tasks)

    logging.info("Filtering and smoothing complete.")
    print("Filtering and smoothing complete.")


if __name__ == "__main__":
    main()