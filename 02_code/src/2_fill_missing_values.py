"""Apply approved, manifest-driven missing-value corrections to FELT raw CSVs.

The Detectorv2 production run contained one reproducible incomplete model-
output row. This stage reads the reviewed correction manifest, verifies that
the observed missing-cell count matches the approved expectation, copies the
preceding frame's model outputs while retaining source/frame metadata, and
writes an audit table.

This is targeted forward-fill handling, not numerical interpolation. The raw
CSV is modified in place so this stage must run only after pre-correction QC.
The operation is idempotent: an already-corrected target is reported without
being rewritten.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SRC_ROOT = SCRIPT_PATH.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.felt_paths import RAW_MOTION_DIR, configure_logging  # noqa: E402

DEFAULT_MANIFEST = (
    SCRIPT_PATH.parents[1] / "config" / "missing_value_corrections_v2.csv"
)
SUPPORTED_METHOD = "forward_fill_failed_model_output"
PRESERVED_METADATA_COLUMNS = {
    "",
    "FrameHeight",
    "FrameWidth",
    "approx_time",
    "frame",
    "input",
}


@dataclass(frozen=True)
class FillTask:
    """One reviewed missing-value correction."""

    relative_csv_path: str
    csv_path: Path
    frame: int
    method: str
    expected_missing_cells: int
    rationale: str


@dataclass(frozen=True)
class FillResult:
    """Auditable result from one correction task."""

    relative_csv_path: str
    frame: int
    method: str
    expected_missing_cells: int
    missing_cells_before: int
    missing_cells_after: int
    overwritten_nonblank_cells: int
    status: str
    rationale: str


class FillNaNError(ValueError):
    """Raised when an approved correction cannot be safely applied."""


def resolve_manifest_target(raw_motion_root: Path, relative_path: str) -> Path:
    """Resolve one manifest path while preventing traversal outside the root."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FillNaNError(f"Unsafe relative_csv_path: {relative_path}")

    root = raw_motion_root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise FillNaNError(f"Correction target is outside raw-motion root: {target}")
    return target


def load_tasks(manifest_path: Path, raw_motion_root: Path) -> list[FillTask]:
    """Load and validate the approved correction manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Correction manifest not found: {manifest_path}")

    required = {
        "relative_csv_path",
        "frame",
        "method",
        "expected_missing_cells",
        "rationale",
    }
    tasks: list[FillTask] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required - set(reader.fieldnames or ())
        if missing_fields:
            raise FillNaNError(
                f"Correction manifest is missing fields: {sorted(missing_fields)}"
            )

        for row_number, row in enumerate(reader, start=2):
            method = row["method"].strip()
            if method != SUPPORTED_METHOD:
                raise FillNaNError(
                    f"Unsupported correction method on row {row_number}: {method}"
                )
            relative_csv_path = row["relative_csv_path"].strip().replace("\\", "/")
            tasks.append(
                FillTask(
                    relative_csv_path=relative_csv_path,
                    csv_path=resolve_manifest_target(
                        raw_motion_root, relative_csv_path
                    ),
                    frame=int(row["frame"]),
                    method=method,
                    expected_missing_cells=int(row["expected_missing_cells"]),
                    rationale=row["rationale"].strip(),
                )
            )

    if not tasks:
        raise FillNaNError(f"Correction manifest contains no tasks: {manifest_path}")
    return tasks


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """Read CSV lines and parsed rows while preserving the original line endings."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        original_lines = handle.readlines()
    rows = list(csv.reader(original_lines))
    if len(rows) != len(original_lines):
        raise FillNaNError(
            f"Cannot safely preserve lines containing embedded newlines: {csv_path}"
        )
    if not rows:
        raise FillNaNError(f"Correction target is empty: {csv_path}")
    return original_lines, rows


def target_row_number(rows: list[list[str]], frame: int, csv_path: Path) -> int:
    """Return the unique data-row number for a source frame."""
    header = rows[0]
    if "frame" not in header:
        raise FillNaNError(f"No frame column found in {csv_path}")
    frame_column = header.index("frame")
    matches = [
        row_number
        for row_number, row in enumerate(rows[1:], start=1)
        if row[frame_column] == str(frame)
    ]
    if len(matches) != 1:
        raise FillNaNError(
            f"Expected one row for frame {frame} in {csv_path}; found {len(matches)}"
        )
    return matches[0]


def model_output_columns(header: list[str]) -> list[int]:
    """Return fields produced by the model rather than source/frame metadata."""
    return [
        index
        for index, column in enumerate(header)
        if column not in PRESERVED_METADATA_COLUMNS
    ]


def copy_previous_model_output(rows: list[list[str]], row_number: int) -> int:
    """Replace a failed row's model outputs with the preceding frame's values."""
    if row_number <= 1:
        raise FillNaNError("Cannot forward-fill the first data row.")
    columns = model_output_columns(rows[0])
    previous = rows[row_number - 1]
    target = rows[row_number]
    if any(previous[index] == "" for index in columns):
        raise FillNaNError("The preceding row contains blank model outputs.")

    overwritten_nonblank = sum(
        target[index] != "" and target[index] != previous[index] for index in columns
    )
    for index in columns:
        target[index] = previous[index]
    return overwritten_nonblank


def model_output_matches_previous(rows: list[list[str]], row_number: int) -> bool:
    """Return whether an already repaired row matches its preceding model output."""
    if row_number <= 1:
        return False
    columns = model_output_columns(rows[0])
    return all(
        rows[row_number][index] == rows[row_number - 1][index] for index in columns
    )


def write_replacement_row(
    csv_path: Path,
    original_lines: list[str],
    rows: list[list[str]],
    row_number: int,
) -> None:
    """Replace one row while preserving the file's other bytes and line ending."""
    original_line = original_lines[row_number]
    line_ending = (
        "\r\n"
        if original_line.endswith("\r\n")
        else "\n"
        if original_line.endswith("\n")
        else ""
    )
    replacement_line = io.StringIO(newline="")
    csv.writer(replacement_line, lineterminator=line_ending).writerow(rows[row_number])
    original_lines[row_number] = replacement_line.getvalue()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(original_lines)


def process_file(task: FillTask, dry_run: bool = False) -> FillResult:
    """Verify and optionally apply one reviewed correction."""
    if not task.csv_path.is_file():
        raise FileNotFoundError(f"Correction target not found: {task.csv_path}")

    original_lines, rows = read_csv_rows(task.csv_path)
    row_number = target_row_number(rows, task.frame, task.csv_path)
    missing_before = sum(value == "" for value in rows[row_number])

    if missing_before == 0:
        if not model_output_matches_previous(rows, row_number):
            raise FillNaNError(
                f"Correction target has no blanks but does not match the approved "
                f"repaired row: {task.relative_csv_path} frame {task.frame}"
            )
        return FillResult(
            relative_csv_path=task.relative_csv_path,
            frame=task.frame,
            method=task.method,
            expected_missing_cells=task.expected_missing_cells,
            missing_cells_before=0,
            missing_cells_after=0,
            overwritten_nonblank_cells=0,
            status="already_corrected",
            rationale=task.rationale,
        )

    if missing_before != task.expected_missing_cells:
        raise FillNaNError(
            f"Unexpected missing-cell count for {task.relative_csv_path} frame "
            f"{task.frame}: expected {task.expected_missing_cells}, found "
            f"{missing_before}"
        )

    if dry_run:
        return FillResult(
            relative_csv_path=task.relative_csv_path,
            frame=task.frame,
            method=task.method,
            expected_missing_cells=task.expected_missing_cells,
            missing_cells_before=missing_before,
            missing_cells_after=missing_before,
            overwritten_nonblank_cells=1,
            status="would_fill",
            rationale=task.rationale,
        )

    overwritten_nonblank = copy_previous_model_output(rows, row_number)
    missing_after = sum(value == "" for value in rows[row_number])
    if missing_after:
        raise FillNaNError(
            f"Forward-fill left {missing_after} blank cells in "
            f"{task.relative_csv_path} frame {task.frame}"
        )
    write_replacement_row(task.csv_path, original_lines, rows, row_number)
    return FillResult(
        relative_csv_path=task.relative_csv_path,
        frame=task.frame,
        method=task.method,
        expected_missing_cells=task.expected_missing_cells,
        missing_cells_before=missing_before,
        missing_cells_after=missing_after,
        overwritten_nonblank_cells=overwritten_nonblank,
        status="filled",
        rationale=task.rationale,
    )


def write_audit(audit_path: Path, results: list[FillResult]) -> None:
    """Write the correction audit table."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-motion-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.raw_motion_root = args.raw_motion_root.resolve()
    output_root = args.raw_motion_root.parent
    args.audit = (
        args.audit.resolve()
        if args.audit
        else output_root / "qc" / "missing_value_correction_audit.csv"
    )
    args.log_file = (
        args.log_file.resolve()
        if args.log_file
        else output_root / "logs" / "2_fill_missing_values.log"
    )
    configure_logging(args.log_file)

    tasks = load_tasks(args.manifest, args.raw_motion_root)
    logging.info("Loaded %d approved missing-value correction(s).", len(tasks))
    results = [process_file(task, dry_run=args.dry_run) for task in tasks]
    write_audit(args.audit, results)

    for result in results:
        print(
            f"{result.status}: {result.relative_csv_path} frame {result.frame} "
            f"({result.missing_cells_before} -> {result.missing_cells_after} blanks)"
        )
    print(f"Audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
