"""
Check FELT raw tracking CSV files for missing values and apply forward-fill.

This script checks raw Py-Feat tracking CSV files for null values, applies pandas
forward-fill (`ffill()`) when null values are present, writes the filled file
back to the same path, and raises an error if null values remain.

Note
----
Despite the original script name, this procedure is not numerical
interpolation. It is forward-fill missing-value handling. This stage modifies 
raw tracking CSV files in place. It does not create a separate output directory.

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
        ├── 2_fill_missing_values.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
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
    LOG_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    VOCAL_CHANNELS,
    actor_name,
    configure_logging,
)


# =============================================================================
# User-editable configuration
# =============================================================================

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Logs.
LOG_FILE = LOG_DIR / "2_fill_missing_values.log"


# =============================================================================
# Data structures and errors
# =============================================================================

@dataclass(frozen=True)
class FillTask:
    """One raw tracking CSV file to check and optionally forward-fill."""

    csv_path: Path
    vocal_channel: str
    actor_name: str


class FillNaNError(Exception):
    """Raised when ffill() fails to remove all null values from a DataFrame."""

    pass


# =============================================================================
# Pipeline functions
# =============================================================================

def count_null_values(df: pd.DataFrame) -> int:
    """Return the total number of null values in a DataFrame."""
    return int(df.isnull().sum().sum())


def build_tasks() -> list[FillTask]:
    """Build one missing-value check task per raw tracking CSV file."""
    tasks: list[FillTask] = []

    for vocal_channel in VOCAL_CHANNELS:
        channel_dir = RAW_MOTION_DIR / vocal_channel

        if not channel_dir.exists():
            logging.warning("Raw-motion channel directory not found; skipping: %s", channel_dir)
            continue

        for actor_id in range(START_ACTOR, END_ACTOR + 1):
            current_actor_name = actor_name(actor_id)
            actor_dir = channel_dir / current_actor_name

            if not actor_dir.exists():
                logging.warning("Actor raw-motion directory not found; skipping: %s", actor_dir)
                continue

            for csv_path in sorted(actor_dir.glob("*.csv")):
                tasks.append(
                    FillTask(
                        csv_path=csv_path,
                        vocal_channel=vocal_channel,
                        actor_name=current_actor_name,
                    )
                )

    logging.info("Prepared %d missing-value check tasks.", len(tasks))
    return tasks


def process_file(task: FillTask) -> str | None:
    """Check one CSV for null values and apply ffill() if needed.

    Returns
    -------
    str | None
        The CSV filename stem if null values were found and successfully
        forward-filled; otherwise None.
    """
    csv_path = task.csv_path
    video_basename = csv_path.stem

    logging.info("Now loading file: %s", csv_path)

    fex_dataframe = read_feat(str(csv_path))
    n_null_before = count_null_values(fex_dataframe)

    if n_null_before == 0:
        logging.info("No null values found: %s", csv_path)
        return None

    logging.info(
        "Null values found in %s: %d null values. Applying ffill().",
        csv_path,
        n_null_before,
    )

    # Preserve the original pipeline behaviour: forward-fill only.
    fex_dataframe = fex_dataframe.ffill()

    n_null_after = count_null_values(fex_dataframe)

    if n_null_after == 0:
        # Preserve the original script's behaviour: write back to the same file.
        fex_dataframe.to_csv(csv_path)
        logging.info("ffill() completed and saved to: %s", csv_path)
        return f"{task.vocal_channel}/{task.actor_name}/{video_basename}"

    logging.error(
        "Failed to ffill() file %s. Null values remaining: %d",
        csv_path,
        n_null_after,
    )
    raise FillNaNError(f"Failed to ffill() file {csv_path}")


def main() -> None:
    """Run missing-value checking and forward-fill for raw tracking files."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT raw-motion missing-value check.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("RAW_MOTION_DIR: %s", RAW_MOTION_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("Actors: %02d-%02d", START_ACTOR, END_ACTOR)
    logging.info("Vocal channels: %s", ", ".join(VOCAL_CHANNELS))

    tasks = build_tasks()

    if not tasks:
        logging.warning("No CSV files were found. Check RAW_MOTION_DIR.")
        print("No CSV files were found. Check RAW_MOTION_DIR.")
        return

    csv_contain_null: list[str] = []

    for index, task in enumerate(tasks, start=1):
        logging.info(
            "Processing task %d/%d: %s",
            index,
            len(tasks),
            task.csv_path,
        )
        print(f"[{index}/{len(tasks)}] Checking {task.csv_path}")

        modified_basename = process_file(task)
        if modified_basename is not None:
            csv_contain_null.append(modified_basename)

    logging.info("Missing-value check complete.")
    logging.info("Files containing null values: %d", len(csv_contain_null))

    print(f"There were {len(csv_contain_null)} files containing null values", end="")
    if csv_contain_null:
        print(": ", end="")
        print(csv_contain_null)
        print("ffill() has been successfully applied to those files.")
    else:
        print(".")


if __name__ == "__main__":
    main()