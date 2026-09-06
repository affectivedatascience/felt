"""Build and verify the three canonical FELT v2 release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.felt_paths import (  # noqa: E402
    OUTPUT_DIR,
    RAW_MOTION_DIR,
    RELEASE_ARCHIVE_DIR,
    SMOOTHED_MOTION_DIR,
    SMOOTHED_VIDEO_DIR,
)

RELEASE_SCHEMA_VERSION = 1
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
CANONICAL_VIDEO_ROOT = SMOOTHED_VIDEO_DIR / "felt_visualization_set"


@dataclass(frozen=True)
class ArchiveSpec:
    """One canonical archive and its required source inventory."""

    archive_name: str
    source_root: Path
    member_root: str
    suffix: str
    expected_file_count: int
    compression: int
    compresslevel: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--smoothed-root", type=Path, default=SMOOTHED_MOTION_DIR)
    parser.add_argument("--video-root", type=Path, default=CANONICAL_VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RELEASE_ARCHIVE_DIR)
    parser.add_argument(
        "--input-qc-summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "ravdess_input_summary.json",
    )
    parser.add_argument(
        "--raw-qc-summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "pyfeat_v2_post_correction" / "qc_summary.json",
    )
    parser.add_argument(
        "--smoothing-qc-summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "challis_smoothing" / "smoothing_run_manifest.json",
    )
    parser.add_argument(
        "--video-qc-summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "visualization_outputs_summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source inventories without hashing or writing archives.",
    )
    return parser.parse_args()


def build_archive_specs(args: argparse.Namespace) -> tuple[ArchiveSpec, ...]:
    """Return the approved unified FELT v2 archive specification."""
    return (
        ArchiveSpec(
            "01_raw_motion.zip",
            args.raw_root.resolve(),
            "01_raw_motion",
            ".csv",
            2452,
            zipfile.ZIP_DEFLATED,
            9,
        ),
        ArchiveSpec(
            "02_smoothed_motion.zip",
            args.smoothed_root.resolve(),
            "02_smoothed_motion",
            ".csv",
            2452,
            zipfile.ZIP_DEFLATED,
            9,
        ),
        ArchiveSpec(
            "03_smoothed_video.zip",
            args.video_root.resolve(),
            "03_smoothed_video/felt_visualization_set",
            ".mp4",
            17164,
            zipfile.ZIP_STORED,
            None,
        ),
    )


def collect_files(spec: ArchiveSpec) -> list[Path]:
    """Collect and validate one archive's deterministic source inventory."""
    if not spec.source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {spec.source_root}")
    files = sorted(
        (
            path
            for path in spec.source_root.rglob(f"*{spec.suffix}")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(spec.source_root).as_posix(),
    )
    if len(files) != spec.expected_file_count:
        raise ValueError(
            f"{spec.archive_name}: found {len(files)} {spec.suffix} files; "
            f"expected {spec.expected_file_count}."
        )
    empty = [path for path in files if path.stat().st_size == 0]
    if empty:
        raise ValueError(
            f"{spec.archive_name}: found {len(empty)} empty files; first is {empty[0]}"
        )
    return files


def load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read required QC evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"QC evidence must be a JSON object: {path}")
    return payload


def validate_qc_evidence(args: argparse.Namespace) -> dict[str, str]:
    """Require passing raw, smoothing, and decoded-video release gates."""
    input_path = args.input_qc_summary.resolve()
    raw_path = args.raw_qc_summary.resolve()
    smoothing_path = args.smoothing_qc_summary.resolve()
    video_path = args.video_qc_summary.resolve()
    inputs = load_json(input_path)
    raw = load_json(raw_path)
    smoothing = load_json(smoothing_path)
    video = load_json(video_path)

    if inputs.get("passed") is not True or inputs.get("counts") != {
        "all_mp4": 4904,
        "full_av": 2452,
        "full_av_song": 1012,
        "full_av_speech": 1440,
        "video_only": 2452,
        "video_only_song": 1012,
        "video_only_speech": 1440,
    }:
        raise ValueError("RAVDESS input QC did not pass the exact source contract.")

    raw_expected = {
        "files_checked": 2452,
        "total_extracted_rows": 299854,
        "schema_mismatch_files": 0,
        "missing_required_column_files": 0,
        "input_path_missing_files": 0,
        "nonportable_input_reference_files": 0,
        "duplicate_frame_files": 0,
        "files_with_missing_values": 0,
        "total_missing_cells": 0,
        "infinite_numeric_cells": 0,
        "csv_decoded_frame_count_mismatches": 0,
        "invalid_frame_values": 0,
        "non_monotonic_frame_steps": 0,
        "missing_frame_gaps": 0,
    }
    raw_mismatches = [
        f"{key}={raw.get(key)!r}, expected {expected!r}"
        for key, expected in raw_expected.items()
        if raw.get(key) != expected
    ]
    if raw_mismatches:
        raise ValueError("Raw QC failed: " + "; ".join(raw_mismatches))

    if smoothing.get("available_file_count") != 2452:
        raise ValueError("Smoothing QC does not cover 2,452 available files.")
    completed = int(smoothing.get("processed_count", 0)) + int(
        smoothing.get("checkpoint_reused_count", 0)
    )
    if completed != 2452:
        raise ValueError(f"Smoothing QC covers {completed} completed files, not 2,452.")

    if video.get("passed") is not True:
        raise ValueError("Decoded-video QC did not pass.")
    if video.get("checked_video_count") != 17164:
        raise ValueError("Decoded-video QC does not cover 17,164 files.")
    if video.get("status_counts") != {"ok": 17164}:
        raise ValueError("Decoded-video QC contains non-passing statuses.")

    return {
        "input_qc_summary": str(input_path),
        "raw_qc_summary": str(raw_path),
        "smoothing_qc_summary": str(smoothing_path),
        "video_qc_summary": str(video_path),
    }


def member_name(spec: ArchiveSpec, source_path: Path) -> str:
    """Return a portable POSIX member path rooted in the dataset component."""
    relative = source_path.resolve().relative_to(spec.source_root).as_posix()
    return f"{spec.member_root}/{relative}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_manifest_digest(spec: ArchiveSpec, files: list[Path]) -> str:
    """Hash ordered archive member names and source-file content digests."""
    digest = hashlib.sha256()
    for source_path in files:
        digest.update(member_name(spec, source_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(source_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def create_archive(
    spec: ArchiveSpec,
    files: list[Path],
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    """Write one archive through a private temporary file and verify it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / spec.archive_name
    expected_members = [member_name(spec, path) for path in files]
    if archive_path.exists() and not overwrite:
        verify_archive(archive_path, expected_members)
        return archive_path

    temporary = output_dir / f".{spec.archive_name}.{uuid.uuid4().hex}.tmp"
    try:
        kwargs = {
            "file": temporary,
            "mode": "w",
            "compression": spec.compression,
            "allowZip64": True,
        }
        if spec.compresslevel is not None:
            kwargs["compresslevel"] = spec.compresslevel
        with zipfile.ZipFile(**kwargs) as archive:
            for index, source_path in enumerate(files, start=1):
                info = _zip_info(member_name(spec, source_path), spec.compression)
                with source_path.open("rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if index == 1 or index % 500 == 0 or index == len(files):
                    print(f"  [{index}/{len(files)}] {info.filename}")
        verify_archive(temporary, expected_members)
        os.replace(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive_path


def verify_archive(archive_path: Path, expected_members: list[str]) -> None:
    """Verify member identity, non-empty payloads, and ZIP CRCs."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if names != expected_members:
                raise ValueError(
                    f"{archive_path}: member inventory or ordering differs from source."
                )
            empty = [member.filename for member in members if member.file_size == 0]
            if empty:
                raise ValueError(f"{archive_path}: empty member {empty[0]}")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"{archive_path}: CRC failure in {corrupt}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc


def write_release_metadata(
    output_dir: Path,
    records: list[dict[str, object]],
) -> None:
    """Write a stable release manifest and standard checksum file."""
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "dataset": "FELT v2",
        "archive_count": len(records),
        "archives": records,
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{record['archive_sha256']}  {record['archive_name']}" for record in records
    ]
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    evidence = validate_qc_evidence(args)
    specs = build_archive_specs(args)
    inventories = [(spec, collect_files(spec)) for spec in specs]
    for spec, files in inventories:
        total_bytes = sum(path.stat().st_size for path in files)
        print(
            f"{spec.archive_name}: {len(files)} files, "
            f"{total_bytes / (1024**3):.2f} GiB"
        )
    if args.dry_run:
        print("Release QC evidence and inventory valid; no archives written.")
        return

    output_dir = args.output_dir.resolve()
    records: list[dict[str, object]] = []
    for spec, files in inventories:
        print(f"Building {spec.archive_name}")
        archive_path = create_archive(
            spec,
            files,
            output_dir,
            overwrite=args.overwrite,
        )
        records.append(
            {
                "archive_name": spec.archive_name,
                "archive_sha256": sha256_file(archive_path),
                "archive_size_bytes": archive_path.stat().st_size,
                "compression": (
                    "deflate-9"
                    if spec.compression == zipfile.ZIP_DEFLATED
                    else "stored"
                ),
                "content_manifest_algorithm": (
                    "sha256(member_path,NUL,file_sha256,newline)"
                ),
                "content_manifest_sha256": content_manifest_digest(spec, files),
                "file_count": len(files),
                "member_root": spec.member_root,
                "source_total_bytes": sum(path.stat().st_size for path in files),
            }
        )
        print(f"Verified {archive_path}")
    write_release_metadata(output_dir, records)
    (output_dir / "qc_evidence_paths.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Release metadata: {output_dir / 'release_manifest.json'}")
    print(f"Checksums: {output_dir / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
