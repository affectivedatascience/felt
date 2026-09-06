"""Apply historical Py-Feat 0.6.2 reviewed keep-person decisions.

This tool backs up affected raw CSV files, then removes duplicate same-frame
candidate rows that do not match the reviewed keep-person decision. It preserves
the original CSV header line, including Py-Feat's duplicate ``frame`` column.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")

from utils.felt_paths import OUTPUT_DIR, PROJECT_ROOT, configure_logging  # noqa: E402


QC_DIR = OUTPUT_DIR / "qc"
DEFAULT_MANIFEST = QC_DIR / "multi_face_review_manifest.csv"
DEFAULT_CANDIDATES = QC_DIR / "multi_face_candidates_long.csv"
DEFAULT_AUDIT = QC_DIR / "multi_face_review_application_audit.csv"
DEFAULT_BACKUP_ROOT = QC_DIR / "raw_motion_backups"
DEFAULT_LOG = OUTPUT_DIR / "logs" / "apply_multi_face_review_decisions.log"


@dataclass(frozen=True)
class ReviewDecision:
    """Reviewed keep-person decision for one affected raw CSV."""

    vocal_channel: str
    relative_csv_path: str
    raw_csv_path: Path
    keep_person: str
    affected_frames: set[int]
    recommended_person: str
    recommendation_basis: str
    reviewer_notes: str


def project_relative(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_frame_list(value: str) -> set[int]:
    """Parse semicolon-delimited frame numbers from the manifest."""
    frames: set[int] = set()
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        frames.add(int(part))
    return frames


def source_frame_column(df: pd.DataFrame) -> str:
    """Pick the source-frame column from Py-Feat CSV output."""
    if "frame.1" in df.columns:
        return "frame.1"
    if "frame" in df.columns:
        return "frame"
    raise ValueError("CSV has no frame column")


def read_original_header(path: Path) -> str:
    """Read the raw CSV header exactly as written."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readline()


def write_csv_preserving_header(df: pd.DataFrame, path: Path, header_line: str) -> None:
    """Write dataframe rows beneath the original raw CSV header line."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header_line)
        df.to_csv(handle, index=False, header=False, lineterminator="\n")
    tmp_path.replace(path)


def update_manifest_decisions(
    manifest_path: Path,
    keep_person: str,
    reviewer_note: str,
    force: bool,
) -> pd.DataFrame:
    """Fill reviewer decision fields in the manifest."""
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")

    if "reviewer_keep_person" not in manifest.columns:
        manifest["reviewer_keep_person"] = ""
    if "reviewer_notes" not in manifest.columns:
        manifest["reviewer_notes"] = ""

    existing = manifest["reviewer_keep_person"].str.strip()
    conflicting = manifest[(existing != "") & (existing != keep_person)]
    if not conflicting.empty and not force:
        examples = ", ".join(conflicting["relative_csv_path"].head(3).tolist())
        raise ValueError(
            "Manifest already contains conflicting reviewer_keep_person values. "
            f"Use --force to overwrite. Examples: {examples}"
        )

    manifest["reviewer_keep_person"] = keep_person
    manifest["reviewer_notes"] = manifest.apply(
        lambda row: (
            reviewer_note
            if not str(row.get("reviewer_notes", "")).strip() or force
            else row["reviewer_notes"]
        ),
        axis=1,
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest


def load_decisions(manifest: pd.DataFrame) -> list[ReviewDecision]:
    """Convert manifest rows to typed decisions."""
    decisions: list[ReviewDecision] = []
    for _, row in manifest.iterrows():
        keep_person = str(row["reviewer_keep_person"]).strip()
        if not keep_person:
            raise ValueError(
                f"Missing reviewer_keep_person for {row['relative_csv_path']}"
            )

        decisions.append(
            ReviewDecision(
                vocal_channel=str(row.get("vocal_channel", "")),
                relative_csv_path=str(row["relative_csv_path"]),
                raw_csv_path=Path(str(row["raw_csv_path"])),
                keep_person=keep_person,
                affected_frames=parse_frame_list(str(row["affected_unique_frames"])),
                recommended_person=str(row.get("recommended_person", "")),
                recommendation_basis=str(row.get("recommendation_basis", "")),
                reviewer_notes=str(row.get("reviewer_notes", "")),
            )
        )
    return decisions


def choose_backup_dir(backup_root: Path, label: str) -> Path:
    """Create a unique backup directory for this application run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"{label}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def backup_raw_csv(decision: ReviewDecision, backup_dir: Path) -> Path:
    """Copy the original raw CSV to the backup directory."""
    backup_path = backup_dir / decision.vocal_channel / decision.relative_csv_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(decision.raw_csv_path, backup_path)
    return backup_path


def apply_decision(decision: ReviewDecision, backup_dir: Path) -> dict[str, object]:
    """Apply one keep-person decision to one raw CSV."""
    header_line = read_original_header(decision.raw_csv_path)
    df = pd.read_csv(decision.raw_csv_path)
    frame_col = source_frame_column(df)

    affected_mask = df[frame_col].astype(int).isin(decision.affected_frames)
    remove_mask = affected_mask & (df["Identity"].astype(str) != decision.keep_person)

    before_rows = len(df)
    before_duplicate_rows = int(df[frame_col].duplicated(keep=False).sum())
    rows_to_remove = int(remove_mask.sum())

    if rows_to_remove == 0:
        raise ValueError(f"No rows would be removed for {decision.relative_csv_path}")

    kept_df = df.loc[~remove_mask].copy()
    after_duplicate_frames = kept_df[frame_col][
        kept_df[frame_col].duplicated(keep=False)
    ]

    if not after_duplicate_frames.empty:
        frames = sorted(after_duplicate_frames.astype(int).unique().tolist())
        raise ValueError(
            f"Duplicate frames remain after applying decision for "
            f"{decision.relative_csv_path}: {frames[:10]}"
        )

    backup_path = backup_raw_csv(decision, backup_dir)
    write_csv_preserving_header(kept_df, decision.raw_csv_path, header_line)

    return {
        "vocal_channel": decision.vocal_channel,
        "relative_csv_path": decision.relative_csv_path,
        "raw_csv_path": str(decision.raw_csv_path),
        "backup_path": str(backup_path),
        "keep_person": decision.keep_person,
        "recommended_person": decision.recommended_person,
        "recommendation_basis": decision.recommendation_basis,
        "reviewer_notes": decision.reviewer_notes,
        "affected_frames": ";".join(
            str(frame) for frame in sorted(decision.affected_frames)
        ),
        "before_rows": before_rows,
        "after_rows": len(kept_df),
        "rows_removed": rows_to_remove,
        "before_duplicate_row_members": before_duplicate_rows,
        "after_duplicate_row_members": int(
            kept_df[frame_col].duplicated(keep=False).sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--keep-person", default="Person_0")
    parser.add_argument(
        "--reviewer-note",
        default=(
            "Reviewer visually verified overlay videos and approved keeping Person_0. "
            "This matches the FaceScore recommendation."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Apply manifest decisions to raw CSV files."""
    args = parse_args()
    configure_logging(args.log_file)

    if not args.candidates.exists():
        raise FileNotFoundError(f"Candidate table not found: {args.candidates}")

    manifest = update_manifest_decisions(
        manifest_path=args.manifest,
        keep_person=args.keep_person,
        reviewer_note=args.reviewer_note,
        force=args.force,
    )
    decisions = load_decisions(manifest)
    logging.info("Loaded %d reviewed decisions.", len(decisions))

    if args.dry_run:
        print(
            f"Dry run: would apply {len(decisions)} decisions from {project_relative(args.manifest)}"
        )
        return

    backup_dir = choose_backup_dir(
        args.backup_root, f"multi_face_keep_{args.keep_person.lower()}"
    )
    audit_rows = []
    for index, decision in enumerate(decisions, start=1):
        logging.info(
            "Applying decision %d/%d: %s -> %s",
            index,
            len(decisions),
            decision.relative_csv_path,
            decision.keep_person,
        )
        audit_rows.append(apply_decision(decision, backup_dir))

    audit = pd.DataFrame(audit_rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit, index=False)

    print(f"Applied {len(audit)} review decisions")
    print(f"Backed up raw CSVs under {project_relative(backup_dir)}")
    print(f"Wrote audit to {project_relative(args.audit)}")
    print(f"Rows removed: {int(audit['rows_removed'].sum())}")


if __name__ == "__main__":
    main()
