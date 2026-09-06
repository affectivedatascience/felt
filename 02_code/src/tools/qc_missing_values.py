"""Profile missing values in raw FELT tracking CSV files before forward-fill.

This tool is the Phase 4 pre-fill audit. It answers where missing-value runs
occur, how long they are, and which columns are affected before any in-place
missing-value handling modifies raw tracking outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")

from utils.felt_paths import (  # noqa: E402
    OUTPUT_DIR,
    RAW_MOTION_DIR,
    VOCAL_CHANNELS,
    configure_logging,
    parse_ravdess_stem,
)


QC_DIR = OUTPUT_DIR / "qc"
DEFAULT_LOG = OUTPUT_DIR / "logs" / "qc_missing_values.log"


@dataclass(frozen=True)
class CsvTask:
    """One raw CSV to audit."""

    csv_path: Path
    vocal_channel: str
    actor: str


@dataclass(frozen=True)
class MissingRun:
    """One contiguous missing-value sequence for one file and column."""

    vocal_channel: str
    actor: str
    stem: str
    csv_path: str
    modality_code: str
    vocal_channel_code: str
    emotion_code: str
    intensity_code: str
    statement_code: str
    repetition_code: str
    actor_code: int
    column: str
    column_group: str
    start_row: int
    end_row: int
    length: int
    n_rows: int
    start_frame: int | str
    end_frame: int | str
    start_position: float
    end_position: float
    midpoint_position: float
    position_region: str
    start_decile: int
    midpoint_decile: int


def column_group(column: str) -> str:
    """Group raw Py-Feat columns into broad semantic families."""
    if column in {"frame", "frame.1", "approx_time", "input"}:
        return "metadata"
    if column == "FaceScore" or column.startswith("FaceRect"):
        return "face_detection"
    if column.startswith("x_") or column.startswith("y_"):
        return "landmark"
    if column in {"Pitch", "Roll", "Yaw"}:
        return "pose"
    if column.startswith("AU"):
        return "action_unit"
    if column in {
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
        "Anger",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    }:
        return "emotion"
    if column in {"valence", "arousal"}:
        return "affect"
    if column.startswith("gaze_"):
        return "gaze"
    if column.startswith(("mesh_x_", "mesh_y_", "mesh_z_")):
        return "face_mesh"
    if column == "_neutral" or column.startswith(
        ("brow", "cheek", "eye", "jaw", "mouth", "nose", "tongue")
    ):
        return "blendshape"
    if column == "Identity" or column.startswith("Identity_"):
        return "identity"
    return "other"


def position_region(start_position: float, end_position: float) -> str:
    """Classify a run as beginning, end, middle, or spanning multiple regions."""
    if start_position <= 0.05:
        if end_position >= 0.95:
            return "entire_file"
        return "beginning"
    if end_position >= 0.95:
        return "end"
    return "middle"


def decile(position: float) -> int:
    """Return 1-10 decile for a normalized position."""
    return min(10, max(1, int(np.floor(position * 10)) + 1))


def source_frame_column(df: pd.DataFrame) -> str | None:
    """Return the source-frame column when present."""
    if "frame.1" in df.columns:
        return "frame.1"
    if "frame" in df.columns:
        return "frame"
    return None


def build_tasks(csv_root: Path) -> list[CsvTask]:
    """Build raw CSV audit tasks."""
    tasks: list[CsvTask] = []
    for vocal_channel in VOCAL_CHANNELS:
        channel_dir = csv_root / vocal_channel
        if not channel_dir.exists():
            logging.warning("Raw-motion channel directory not found: %s", channel_dir)
            continue
        for actor_dir in sorted(channel_dir.glob("Actor_*")):
            if not actor_dir.is_dir():
                continue
            for csv_path in sorted(actor_dir.glob("*.csv")):
                tasks.append(
                    CsvTask(
                        csv_path=csv_path,
                        vocal_channel=vocal_channel,
                        actor=actor_dir.name,
                    )
                )
    return tasks


def missing_runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Return inclusive row-index ranges for contiguous True runs."""
    values = mask.to_numpy(dtype=bool)
    if len(values) == 0 or not values.any():
        return []

    padded = np.concatenate(([False], values, [False]))
    transitions = np.diff(padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def frame_value(df: pd.DataFrame, frame_col: str | None, row_index: int) -> int | str:
    """Return frame value for a row when available."""
    if frame_col is None:
        return ""
    value = df.iloc[row_index][frame_col]
    if pd.isna(value):
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def parse_codes(stem: str) -> dict[str, object]:
    """Parse RAVDESS filename codes for grouping."""
    code = parse_ravdess_stem(stem)
    return {
        "modality_code": code.modality,
        "vocal_channel_code": code.vocal_channel,
        "emotion_code": code.emotion,
        "intensity_code": code.intensity,
        "statement_code": code.statement,
        "repetition_code": code.repetition,
        "actor_code": code.actor,
    }


def audit_file(
    task: CsvTask,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Audit one CSV and return file, column, and run rows."""
    df = pd.read_csv(task.csv_path)
    n_rows = len(df)
    n_cells = int(n_rows * len(df.columns))
    stem = task.csv_path.stem
    codes = parse_codes(stem)
    frame_col = source_frame_column(df)

    null_counts = df.isna().sum()
    total_missing = int(null_counts.sum())
    columns_with_missing = null_counts[null_counts > 0]

    file_row = {
        "vocal_channel": task.vocal_channel,
        "actor": task.actor,
        "stem": stem,
        "csv_path": str(task.csv_path),
        **codes,
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "n_cells": n_cells,
        "n_missing": total_missing,
        "prop_missing": total_missing / n_cells if n_cells else 0.0,
        "n_columns_with_missing": int((null_counts > 0).sum()),
        "prop_columns_with_missing": float((null_counts > 0).mean())
        if len(df.columns)
        else 0.0,
        "has_missing": total_missing > 0,
    }

    column_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for column, n_missing in columns_with_missing.items():
        group = column_group(str(column))
        col_mask = df[column].isna()
        runs = missing_runs(col_mask)

        column_rows.append(
            {
                "vocal_channel": task.vocal_channel,
                "actor": task.actor,
                "stem": stem,
                "csv_path": str(task.csv_path),
                **codes,
                "column": column,
                "column_group": group,
                "n_rows": n_rows,
                "n_missing": int(n_missing),
                "prop_missing": int(n_missing) / n_rows if n_rows else 0.0,
                "n_missing_runs": len(runs),
                "min_run_length": min(
                    (end - start + 1 for start, end in runs), default=0
                ),
                "max_run_length": max(
                    (end - start + 1 for start, end in runs), default=0
                ),
            }
        )

        for start, end in runs:
            length = end - start + 1
            denom = max(n_rows - 1, 1)
            start_pos = start / denom
            end_pos = end / denom
            midpoint_pos = ((start + end) / 2) / denom

            run_rows.append(
                asdict(
                    MissingRun(
                        vocal_channel=task.vocal_channel,
                        actor=task.actor,
                        stem=stem,
                        csv_path=str(task.csv_path),
                        **codes,
                        column=str(column),
                        column_group=group,
                        start_row=start,
                        end_row=end,
                        length=length,
                        n_rows=n_rows,
                        start_frame=frame_value(df, frame_col, start),
                        end_frame=frame_value(df, frame_col, end),
                        start_position=start_pos,
                        end_position=end_pos,
                        midpoint_position=midpoint_pos,
                        position_region=position_region(start_pos, end_pos),
                        start_decile=decile(start_pos),
                        midpoint_decile=decile(midpoint_pos),
                    )
                )
            )

    return file_row, column_rows, run_rows


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing-value runs by column and position."""
    if runs.empty:
        return pd.DataFrame()

    group_cols = ["column_group", "column", "position_region"]
    grouped = runs.groupby(group_cols, dropna=False)
    return grouped.agg(
        n_runs=("length", "size"),
        n_files=("csv_path", "nunique"),
        total_missing_cells=("length", "sum"),
        min_run_length=("length", "min"),
        max_run_length=("length", "max"),
        mean_run_length=("length", "mean"),
        median_run_length=("length", "median"),
        mean_start_position=("start_position", "mean"),
        median_start_position=("start_position", "median"),
        mean_midpoint_position=("midpoint_position", "mean"),
        median_midpoint_position=("midpoint_position", "median"),
    ).reset_index()


def summarize_columns(columns: pd.DataFrame, total_files: int) -> pd.DataFrame:
    """Summarize missingness by column across files."""
    if columns.empty:
        return pd.DataFrame()

    grouped = columns.groupby(["column_group", "column"], dropna=False)
    summary = grouped.agg(
        n_files_with_missing=("csv_path", "nunique"),
        total_missing_cells=("n_missing", "sum"),
        min_file_prop_missing=("prop_missing", "min"),
        max_file_prop_missing=("prop_missing", "max"),
        mean_file_prop_missing=("prop_missing", "mean"),
        median_file_prop_missing=("prop_missing", "median"),
        total_missing_runs=("n_missing_runs", "sum"),
        max_run_length=("max_run_length", "max"),
    ).reset_index()
    summary["prop_files_with_missing"] = summary["n_files_with_missing"] / total_files
    return summary.sort_values(
        ["total_missing_cells", "n_files_with_missing"], ascending=False
    )


def summarize_positions(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize run positions overall and by decile."""
    if runs.empty:
        return pd.DataFrame()

    grouped = runs.groupby(["position_region", "midpoint_decile"], dropna=False)
    return grouped.agg(
        n_runs=("length", "size"),
        n_files=("csv_path", "nunique"),
        total_missing_cells=("length", "sum"),
        min_run_length=("length", "min"),
        max_run_length=("length", "max"),
        mean_run_length=("length", "mean"),
        median_run_length=("length", "median"),
    ).reset_index()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    """Run the missing-value audit."""
    args = parse_args()
    configure_logging(args.log_file)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.csv_root)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    logging.info("Prepared %d CSV files for missing-value QC.", len(tasks))
    file_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for index, task in enumerate(tasks, start=1):
        if index % 100 == 0 or index == len(tasks):
            print(f"Processed {index}/{len(tasks)}")
        file_row, task_column_rows, task_run_rows = audit_file(task)
        file_rows.append(file_row)
        column_rows.extend(task_column_rows)
        run_rows.extend(task_run_rows)

    files = pd.DataFrame(file_rows)
    columns = pd.DataFrame(column_rows)
    runs = pd.DataFrame(run_rows)

    files.to_csv(args.qc_dir / "missing_values_file_summary.csv", index=False)
    columns.to_csv(args.qc_dir / "missing_values_column_file_summary.csv", index=False)
    runs.to_csv(args.qc_dir / "missing_value_runs_long.csv", index=False)
    summarize_columns(columns, total_files=len(files)).to_csv(
        args.qc_dir / "missing_values_column_summary.csv", index=False
    )
    summarize_runs(runs).to_csv(
        args.qc_dir / "missing_value_run_summary.csv", index=False
    )
    summarize_positions(runs).to_csv(
        args.qc_dir / "missing_value_position_summary.csv", index=False
    )

    print("Missing-value QC complete.")
    print(f"Files checked: {len(files)}")
    print(
        f"Files with missing values: {int(files['has_missing'].sum()) if not files.empty else 0}"
    )
    print(f"Missing cells: {int(files['n_missing'].sum()) if not files.empty else 0}")
    print(f"Missing runs: {len(runs)}")
    print(f"Outputs written to: {args.qc_dir}")


if __name__ == "__main__":
    main()
