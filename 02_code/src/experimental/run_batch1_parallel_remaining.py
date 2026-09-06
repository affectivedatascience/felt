"""Superseded compatibility wrapper for raw-tracking extraction.

Historically this helper used ``runpy`` to mutate globals inside
``1_extract_raw_tracking.py``. The maintained Phase 1 interface now lives in the
extraction script itself. This wrapper keeps the old command name available for
existing notes while delegating to the new CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTRACT_SCRIPT = PROJECT_ROOT / "02_code" / "src" / "1_extract_raw_tracking.py"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "01_data"
    / "02_output"
    / "logs"
    / "batch_size_1_parallel_remaining_report.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BATCH_SIZE=1 extraction via the maintained extraction CLI."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=10,
        help="Skip this many leading selected tasks.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report tasks without processing videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSVs instead of resuming around them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--batch-size",
        "1",
        "--workers",
        str(args.workers),
        "--skip-first",
        str(args.skip_first),
        "--report",
        str(args.report),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
