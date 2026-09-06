"""Normalize legacy absolute CSV ``input`` values without reserializing data."""

from __future__ import annotations

import argparse
import csv
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizationResult:
    csv_path: str
    status: str
    rows: int
    replacements: int
    old_reference: str
    new_reference: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-root",
        action="append",
        required=True,
        type=Path,
        help="CSV tree to normalize; repeat for raw and smoothed roots.",
    )
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def canonical_reference(csv_path: Path) -> str:
    actor = csv_path.parent.name
    if not actor.startswith("Actor_") or not actor[-2:].isdigit():
        raise ValueError(f"CSV is not under an Actor_XX directory: {csv_path}")
    return f"{actor}/{csv_path.stem}.mp4"


def first_input_value(payload: bytes) -> str:
    """Read the header and first data row only to identify the input token."""
    lines = payload.splitlines(keepends=False)
    if len(lines) < 2:
        raise ValueError("CSV has no data rows.")
    header = next(csv.reader([lines[0].decode("utf-8-sig")]))
    try:
        input_index = header.index("input")
    except ValueError as exc:
        raise ValueError("CSV has no input column.") from exc
    first_row = next(csv.reader([lines[1].decode("utf-8")]))
    if input_index >= len(first_row) or not first_row[input_index]:
        raise ValueError("First data row has no input reference.")
    return first_row[input_index]


def normalize_file(path: Path, *, dry_run: bool) -> NormalizationResult:
    """Replace only the repeated source-reference bytes in one CSV."""
    try:
        payload = path.read_bytes()
        rows = len(payload.splitlines()) - 1
        old = first_input_value(payload)
        new = canonical_reference(path)
        if old == new:
            return NormalizationResult(str(path), "already_portable", rows, 0, old, new)

        old_bytes = old.encode("utf-8")
        replacements = payload.count(old_bytes)
        if replacements != rows:
            raise ValueError(
                f"Expected the input token once in each of {rows} rows; "
                f"found {replacements} occurrences."
            )
        updated = payload.replace(old_bytes, new.encode("utf-8"))
        if not dry_run:
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(updated)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return NormalizationResult(
            str(path),
            "would_normalize" if dry_run else "normalized",
            rows,
            replacements,
            old,
            new,
        )
    except Exception as exc:
        return NormalizationResult(str(path), "error", 0, 0, "", "", repr(exc))


def write_audit(path: Path, results: list[NormalizationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NormalizationResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    args = parse_args()
    paths = sorted(
        {
            path.resolve()
            for root in args.csv_root
            for path in root.resolve().rglob("*.csv")
            if path.is_file()
        },
        key=str,
    )
    if not paths:
        raise ValueError("No CSV files found under the requested roots.")
    results = [normalize_file(path, dry_run=args.dry_run) for path in paths]
    write_audit(args.audit.resolve(), results)
    errors = [result for result in results if result.status == "error"]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"Source-reference normalization: {counts}")
    print(f"Audit: {args.audit.resolve()}")
    if errors:
        for result in errors[:10]:
            print(f"ERROR {result.csv_path}: {result.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
