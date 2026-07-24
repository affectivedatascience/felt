"""Run scalable integrity and missing-value QC on Py-Feat 2 extraction CSVs.

The audit streams one CSV at a time so the full Detectorv2 schema does not
expand into millions of in-memory file/column records. It checks schema,
required column families, frame sequences, missing and infinite values, input
linkage, identity fragmentation, face scores, face boxes, and frame dimensions.
When a frame-count report is available, it also writes a combined QC summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

from utils.felt_paths import OUTPUT_DIR, RAW_MOTION_DIR  # noqa: E402


DEFAULT_QC_DIR = OUTPUT_DIR / "qc" / "pyfeat_v2_full"
DEFAULT_FACE_THRESHOLD = 0.83
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

EXPECTED_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
EXPECTED_EMOTIONS = [
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger",
]


def required_columns() -> set[str]:
    """Return core Detectorv2 columns required by the FELT pipeline and QC."""
    return set(
        [
            "FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight",
            "FaceScore", "Pitch", "Roll", "Yaw", "X", "Y", "Z",
            "gaze_pitch", "gaze_yaw", "gaze_angle", "valence", "arousal",
            "FrameHeight", "FrameWidth", "input", "frame", "approx_time",
            "Identity",
        ]
        + [f"x_{index}" for index in range(68)]
        + [f"y_{index}" for index in range(68)]
        + [f"mesh_x_{index}" for index in range(478)]
        + [f"mesh_y_{index}" for index in range(478)]
        + [f"mesh_z_{index}" for index in range(478)]
        + EXPECTED_AUS
        + EXPECTED_EMOTIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument(
        "--frame-report",
        type=Path,
        help="Frame-count report to incorporate; defaults to QC_DIR/frame_count.csv.",
    )
    parser.add_argument("--face-threshold", type=float, default=DEFAULT_FACE_THRESHOLD)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--limit", type=int, help="Audit only the first N CSVs.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.csv_root.is_dir():
        raise FileNotFoundError(f"CSV root does not exist: {args.csv_root}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")
    if not 0 <= args.face_threshold <= 1:
        raise ValueError("--face-threshold must be between 0 and 1")


def bool_series(values: pd.Series) -> pd.Series:
    """Convert CSV booleans back to booleans after a report is reloaded."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def audit_extraction(
    csv_root: Path,
    paths: list[Path],
    *,
    face_threshold: float,
    frame_width: int,
    frame_height: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Audit CSVs sequentially and return file, missing, exception, and summary data."""
    reference_columns: tuple[str, ...] | None = None
    required = required_columns()
    file_rows: list[dict[str, Any]] = []
    no_face_rows: list[dict[str, Any]] = []
    missing_by_column: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "cells": 0, "max_in_file": 0}
    )
    identity_files: Counter[str] = Counter()
    total_missing = 0
    total_infinite = 0

    for file_index, path in enumerate(paths, start=1):
        df = pd.read_csv(path, low_memory=False)
        columns = tuple(str(column) for column in df.columns)
        if reference_columns is None:
            reference_columns = columns

        issues: list[str] = []
        missing = df.isna().sum()
        missing_cells = int(missing.sum())
        total_missing += missing_cells
        for column, count in missing[missing > 0].items():
            stats = missing_by_column[str(column)]
            stats["files"] += 1
            stats["cells"] += int(count)
            stats["max_in_file"] = max(stats["max_in_file"], int(count))
        if missing_cells:
            issues.append("missing_values")

        numeric = df.select_dtypes(include=[np.number])
        numeric_missing = int(numeric.isna().sum().sum())
        nonfinite = int((~np.isfinite(numeric.to_numpy())).sum())
        infinite_cells = max(0, nonfinite - numeric_missing)
        total_infinite += infinite_cells
        if infinite_cells:
            issues.append("infinite_values")

        expected_frames = np.arange(len(df))
        frames = pd.to_numeric(df.get("frame"), errors="coerce")
        frame_sequence_ok = (
            not frames.isna().any()
            and np.array_equal(frames.to_numpy(), expected_frames)
        )
        duplicate_frame_rows = int(frames.duplicated(keep=False).sum())
        if not frame_sequence_ok:
            issues.append("frame_sequence")
        if duplicate_frame_rows:
            issues.append("duplicate_frames")

        saved_index_ok = True
        if "Unnamed: 0" in df.columns:
            saved_index = pd.to_numeric(df["Unnamed: 0"], errors="coerce")
            saved_index_ok = (
                not saved_index.isna().any()
                and np.array_equal(saved_index.to_numpy(), expected_frames)
            )
            if not saved_index_ok:
                issues.append("saved_index_sequence")

        schema_matches = columns == reference_columns
        missing_required = sorted(required - set(columns))
        if not schema_matches:
            issues.append("schema_mismatch")
        if missing_required:
            issues.append("required_columns_missing")

        input_values = (
            sorted(df["input"].dropna().astype(str).unique().tolist())
            if "input" in df else []
        )
        input_stem_match = bool(input_values) and all(
            Path(value).stem == path.stem for value in input_values
        )
        input_paths_exist = bool(input_values) and all(
            Path(value).is_file() for value in input_values
        )
        if not input_stem_match:
            issues.append("input_stem_mismatch")
        if not input_paths_exist:
            issues.append("input_path_missing")

        identities = (
            sorted(df["Identity"].dropna().astype(str).unique().tolist())
            if "Identity" in df else []
        )
        identity_files.update(identities)
        if identities != ["Person_0"]:
            issues.append("identity_unexpected")

        face_score = pd.to_numeric(df.get("FaceScore"), errors="coerce")
        face_score_min = float(face_score.min())
        face_score_max = float(face_score.max())
        below_threshold = int((face_score < face_threshold).sum())
        if face_score.isna().any() or not face_score.between(
            0, 1, inclusive="both"
        ).all():
            issues.append("face_score_range")
        if below_threshold:
            issues.append("face_score_below_threshold")

        facebox_positive = bool(
            (pd.to_numeric(df.get("FaceRectWidth"), errors="coerce") > 0).all()
            and (pd.to_numeric(df.get("FaceRectHeight"), errors="coerce") > 0).all()
        )
        if not facebox_positive:
            issues.append("facebox_nonpositive")

        dimensions_ok = bool(
            (pd.to_numeric(df.get("FrameWidth"), errors="coerce") == frame_width).all()
            and (pd.to_numeric(df.get("FrameHeight"), errors="coerce") == frame_height).all()
        )
        if not dimensions_ok:
            issues.append("frame_dimensions")

        row_missing = df.isna().sum(axis=1)
        for row_index in np.flatnonzero(row_missing.to_numpy() > 0):
            row = df.iloc[int(row_index)]
            no_face_rows.append(
                {
                    "relative_csv_path": str(path.relative_to(csv_root)),
                    "row_index": int(row_index),
                    "frame": row.get("frame", ""),
                    "approx_time": row.get("approx_time", ""),
                    "missing_cells": int(row_missing.iloc[int(row_index)]),
                    "face_score": row.get("FaceScore", ""),
                    "identity": row.get("Identity", ""),
                    "input": row.get("input", ""),
                }
            )

        file_rows.append(
            {
                "status": "ok" if not issues else ";".join(sorted(set(issues))),
                "relative_csv_path": str(path.relative_to(csv_root)),
                "rows": len(df),
                "columns_on_disk": len(df.columns),
                "missing_cells": missing_cells,
                "infinite_cells": infinite_cells,
                "frame_sequence_ok": frame_sequence_ok,
                "saved_index_sequence_ok": saved_index_ok,
                "duplicate_frame_rows": duplicate_frame_rows,
                "schema_matches_reference": schema_matches,
                "missing_required_column_count": len(missing_required),
                "missing_required_columns": ";".join(missing_required),
                "input_stem_match": input_stem_match,
                "input_paths_exist": input_paths_exist,
                "identity_count": len(identities),
                "identities": ";".join(identities),
                "face_score_min": face_score_min,
                "face_score_max": face_score_max,
                "face_scores_below_threshold": below_threshold,
                "facebox_positive": facebox_positive,
                "frame_dimensions_expected": dimensions_ok,
            }
        )
        if file_index % progress_every == 0 or file_index == len(paths):
            print(f"Audited {file_index}/{len(paths)}")

    files = pd.DataFrame(file_rows)
    missing_rows = [
        {
            "column": column,
            "files_with_missing": stats["files"],
            "missing_cells": stats["cells"],
            "max_missing_in_one_file": stats["max_in_file"],
        }
        for column, stats in sorted(missing_by_column.items())
    ]
    missing_columns = pd.DataFrame(
        missing_rows,
        columns=[
            "column", "files_with_missing", "missing_cells",
            "max_missing_in_one_file",
        ],
    )
    issue_counts: Counter[str] = Counter()
    for status in files.loc[files["status"] != "ok", "status"]:
        issue_counts.update(str(status).split(";"))
    summary = {
        "files": len(files),
        "ok_files": int((files["status"] == "ok").sum()),
        "files_with_issues": int((files["status"] != "ok").sum()),
        "total_rows": int(files["rows"].sum()),
        "reference_columns_on_disk": len(reference_columns or ()),
        "total_missing_cells": total_missing,
        "columns_with_missing": len(missing_by_column),
        "total_infinite_cells": total_infinite,
        "identity_file_counts": dict(identity_files),
        "global_face_score_min": float(files["face_score_min"].min()),
        "global_face_score_max": float(files["face_score_max"].max()),
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    return files, missing_columns, no_face_rows, summary


def write_reports(
    qc_dir: Path,
    files: pd.DataFrame,
    missing_columns: pd.DataFrame,
    no_face_rows: list[dict[str, Any]],
    integrity_summary: dict[str, Any],
    frame_report: Path,
) -> dict[str, Any]:
    """Write detailed reports and return the combined QC summary."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    files.to_csv(qc_dir / "extraction_integrity_file_summary.csv", index=False)
    files[["relative_csv_path", "rows", "missing_cells", "status"]].to_csv(
        qc_dir / "missing_values_file_summary.csv", index=False
    )
    missing_columns.to_csv(qc_dir / "missing_values_column_summary.csv", index=False)
    files[files["status"] != "ok"].to_csv(qc_dir / "qc_exceptions.csv", index=False)
    files[files["identity_count"] > 1].to_csv(
        qc_dir / "identity_fragmentation_files.csv", index=False
    )
    (
        files.groupby("identity_count", dropna=False).size().rename("file_count")
        .reset_index().sort_values("identity_count")
        .to_csv(qc_dir / "identity_fragmentation_distribution.csv", index=False)
    )
    pd.DataFrame(
        no_face_rows,
        columns=[
            "relative_csv_path", "row_index", "frame", "approx_time",
            "missing_cells", "face_score", "identity", "input",
        ],
    ).to_csv(qc_dir / "no_face_rows.csv", index=False)
    (qc_dir / "extraction_integrity_summary.json").write_text(
        json.dumps(integrity_summary, indent=2), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "files_checked": int(len(files)),
        "total_extracted_rows": int(files["rows"].sum()),
        "schema_mismatch_files": int((~bool_series(files["schema_matches_reference"])).sum()),
        "missing_required_column_files": int(
            (files["missing_required_column_count"] > 0).sum()
        ),
        "input_path_missing_files": int((~bool_series(files["input_paths_exist"])).sum()),
        "duplicate_frame_files": int((files["duplicate_frame_rows"] > 0).sum()),
        "files_with_missing_values": int((files["missing_cells"] > 0).sum()),
        "total_missing_cells": int(files["missing_cells"].sum()),
        "rows_with_missing_values": len(no_face_rows),
        "infinite_numeric_cells": int(files["infinite_cells"].sum()),
        "identity_fragmented_files": int((files["identity_count"] > 1).sum()),
        "identity_fragmented_file_percent": round(
            float((files["identity_count"] > 1).mean() * 100), 3
        ),
        "maximum_identities_in_one_file": int(files["identity_count"].max()),
        "global_face_score_min": float(files["face_score_min"].min()),
        "global_face_score_max": float(files["face_score_max"].max()),
        "files_with_face_score_below_threshold": int(
            (files["face_scores_below_threshold"] > 0).sum()
        ),
    }
    if frame_report.is_file():
        frame = pd.read_csv(frame_report)
        summary.update(
            {
                "csv_decoded_frame_count_matches": int(
                    (frame["delta_csv_minus_video"] == 0).sum()
                ),
                "csv_decoded_frame_count_mismatches": int(
                    (frame["delta_csv_minus_video"] != 0).sum()
                ),
                "metadata_decoded_count_disagreements": int(
                    (frame["delta_metadata_minus_decoded"] != 0).sum()
                ),
                "invalid_frame_values": int(frame["csv_invalid_frame_values"].sum()),
                "non_monotonic_frame_steps": int(
                    frame["csv_non_monotonic_steps"].sum()
                ),
                "missing_frame_gaps": int(frame["csv_missing_frame_gaps"].sum()),
            }
        )
    (qc_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    csv_root = args.csv_root.resolve()
    qc_dir = args.qc_dir.resolve()
    frame_report = (
        args.frame_report.resolve() if args.frame_report else qc_dir / "frame_count.csv"
    )
    paths = sorted(csv_root.rglob("*.csv"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No CSV files found under {csv_root}")

    print(f"Auditing {len(paths)} CSV files under {csv_root}")
    files, missing_columns, no_face_rows, integrity_summary = audit_extraction(
        csv_root,
        paths,
        face_threshold=args.face_threshold,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        progress_every=args.progress_every,
    )
    summary = write_reports(
        qc_dir,
        files,
        missing_columns,
        no_face_rows,
        integrity_summary,
        frame_report,
    )
    print(json.dumps(summary, indent=2))
    print(f"QC reports written to: {qc_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
