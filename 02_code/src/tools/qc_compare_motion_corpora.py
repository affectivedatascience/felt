"""Compare a reproduced FELT CSV corpus with a reference corpus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5


@dataclass(frozen=True)
class ComparisonResult:
    relative_csv_path: str
    status: str
    rows: int
    columns: int
    numeric_cells: int
    out_of_tolerance_cells: int
    non_numeric_mismatches: int
    max_absolute_error: float
    max_relative_error: float
    error: str = ""


def discover(root: Path) -> dict[str, Path]:
    """Return a portable relative-path inventory for a corpus root."""
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.csv"))
    }


def compare_file(
    relative_path: str,
    candidate_path: Path,
    reference_path: Path,
    *,
    atol: float,
    rtol: float,
) -> ComparisonResult:
    """Compare one CSV structurally, textually, and numerically."""
    candidate = pd.read_csv(candidate_path, low_memory=False)
    reference = pd.read_csv(reference_path, low_memory=False)
    if list(candidate.columns) != list(reference.columns):
        return ComparisonResult(
            relative_path, "failed", len(candidate), len(candidate.columns), 0, 0, 0, 0.0, 0.0,
            "column names or order differ",
        )
    if candidate.shape != reference.shape:
        return ComparisonResult(
            relative_path, "failed", len(candidate), len(candidate.columns), 0, 0, 0, 0.0, 0.0,
            f"shape differs: candidate={candidate.shape}, reference={reference.shape}",
        )

    numeric_columns = [
        column
        for column in candidate.columns
        if pd.api.types.is_numeric_dtype(candidate[column])
        and pd.api.types.is_numeric_dtype(reference[column])
    ]
    text_columns = [column for column in candidate.columns if column not in numeric_columns]
    candidate_numeric = candidate[numeric_columns].to_numpy(dtype=float)
    reference_numeric = reference[numeric_columns].to_numpy(dtype=float)
    close = np.isclose(
        candidate_numeric,
        reference_numeric,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    finite_pairs = np.isfinite(candidate_numeric) & np.isfinite(reference_numeric)
    absolute = np.zeros_like(candidate_numeric)
    np.subtract(candidate_numeric, reference_numeric, out=absolute, where=finite_pairs)
    absolute = np.abs(absolute)
    denominator = np.maximum(np.abs(reference_numeric), np.finfo(float).tiny)
    relative = np.divide(
        absolute,
        denominator,
        out=np.zeros_like(absolute),
        where=finite_pairs,
    )
    text_mismatches = sum(
        int(
            np.count_nonzero(
                candidate[column].fillna("<NA>").astype(str).to_numpy()
                != reference[column].fillna("<NA>").astype(str).to_numpy()
            )
        )
        for column in text_columns
    )
    numeric_mismatches = int(np.size(close) - np.count_nonzero(close))
    passed = numeric_mismatches == 0 and text_mismatches == 0
    return ComparisonResult(
        relative_csv_path=relative_path,
        status="ok" if passed else "failed",
        rows=len(candidate),
        columns=len(candidate.columns),
        numeric_cells=int(candidate_numeric.size),
        out_of_tolerance_cells=numeric_mismatches,
        non_numeric_mismatches=text_mismatches,
        max_absolute_error=float(absolute.max(initial=0.0)),
        max_relative_error=float(relative.max(initial=0.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--allow-reference-superset",
        action="store_true",
        help="Compare a candidate subset without failing on other reference files.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Tolerances must be non-negative.")
    candidate_root = args.candidate_root.resolve()
    reference_root = args.reference_root.resolve()
    candidate = discover(candidate_root)
    reference = discover(reference_root)
    reference_only = sorted(set(reference) - set(candidate))
    missing = [] if args.allow_reference_superset else reference_only
    unexpected = sorted(set(candidate) - set(reference))
    common = sorted(set(candidate) & set(reference))
    results = [
        compare_file(
            relative_path,
            candidate[relative_path],
            reference[relative_path],
            atol=args.atol,
            rtol=args.rtol,
        )
        for relative_path in common
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ComparisonResult.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    summary = {
        "schema_version": 1,
        "candidate_root": str(candidate_root),
        "reference_root": str(reference_root),
        "absolute_tolerance": args.atol,
        "relative_tolerance": args.rtol,
        "candidate_files": len(candidate),
        "reference_files": len(reference),
        "compared_files": len(results),
        "missing_files": missing,
        "reference_only_file_count": len(reference_only),
        "unexpected_files": unexpected,
        "failed_files": [r.relative_csv_path for r in results if r.status != "ok"],
        "out_of_tolerance_cells": sum(r.out_of_tolerance_cells for r in results),
        "non_numeric_mismatches": sum(r.non_numeric_mismatches for r in results),
        "max_absolute_error": max((r.max_absolute_error for r in results), default=0.0),
        "max_relative_error": max((r.max_relative_error for r in results), default=0.0),
    }
    summary["passed"] = not (
        missing
        or unexpected
        or summary["failed_files"]
        or summary["out_of_tolerance_cells"]
        or summary["non_numeric_mismatches"]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
