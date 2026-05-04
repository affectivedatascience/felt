"""
Create FELT release ZIP archives.

This utility packages the generated FELT outputs into the six archive files used
for dataset release:

    raw_motion_speech.zip
    raw_motion_song.zip
    smoothed_motion_speech.zip
    smoothed_motion_song.zip
    smoothed_video_speech.zip
    smoothed_video_song.zip

Motion archives contain plaintext CSV files and are compressed with maximum
DEFLATE compression. Video archives contain H.264 MP4 files and are stored
without additional compression.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   └── 02_output/
│       ├── 01_raw_motion/
│       │   ├── speech/
│       │   └── song/
│       ├── 02_smoothed_motion/
│       │   ├── speech/
│       │   └── song/
│       ├── 03_smoothed_video/
│       │   ├── action_unit_activation/
│       │   │   ├── speech/
│       │   │   └── song/
│       │   ├── landmark_plot/
│       │   │   ├── speech/
│       │   │   └── song/
│       │   └── landmark_overlay/
│       │       ├── speech/
│       │       └── song/
│       ├── 04_release_archives/
│       └── logs/
└── 02_code/
    └── src/
        ├── tools/
        │   └── create_release_archives.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py
"""

from __future__ import annotations

import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


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
    AU_ACTIVATION_VIDEO_DIR,
    LANDMARK_OVERLAY_VIDEO_DIR,
    LANDMARK_PLOT_VIDEO_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    RELEASE_ARCHIVE_DIR,
    SMOOTHED_MOTION_DIR,
    configure_logging,
)


# =============================================================================
# User-editable configuration
# =============================================================================

LOG_FILE = LOG_DIR / "create_release_archives.log"

OVERWRITE_EXISTING = False

# Maximum compression for plaintext CSV archives.
CSV_COMPRESSION = zipfile.ZIP_DEFLATED
CSV_COMPRESSLEVEL = 9

# MP4 files are already compressed; storing avoids wasted CPU time.
VIDEO_COMPRESSION = zipfile.ZIP_STORED
VIDEO_COMPRESSLEVEL = None

# Expected FELT v1 release counts.
EXPECTED_COUNTS = {
    "raw_motion_speech_csv": 1440,
    "raw_motion_song_csv": 1012,
    "smoothed_motion_speech_csv": 1440,
    "smoothed_motion_song_csv": 1012,
    "au_activation_speech_mp4": 1440,
    "au_activation_song_mp4": 1012,
    "landmark_plot_speech_mp4": 1440,
    "landmark_plot_song_mp4": 1012,
    "landmark_overlay_speech_mp4": 1440,
    "landmark_overlay_song_mp4": 1012,
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class ArchiveSpec:
    """Specification for one release archive."""

    archive_name: str
    source_paths: tuple[Path, ...]
    include_patterns: tuple[str, ...]
    compression: int
    compresslevel: int | None


# =============================================================================
# Archive specifications
# =============================================================================

def build_archive_specs() -> list[ArchiveSpec]:
    """Build the six FELT release archive specifications."""
    return [
        ArchiveSpec(
            archive_name="raw_motion_speech.zip",
            source_paths=(RAW_MOTION_DIR / "speech",),
            include_patterns=("*.csv",),
            compression=CSV_COMPRESSION,
            compresslevel=CSV_COMPRESSLEVEL,
        ),
        ArchiveSpec(
            archive_name="raw_motion_song.zip",
            source_paths=(RAW_MOTION_DIR / "song",),
            include_patterns=("*.csv",),
            compression=CSV_COMPRESSION,
            compresslevel=CSV_COMPRESSLEVEL,
        ),
        ArchiveSpec(
            archive_name="smoothed_motion_speech.zip",
            source_paths=(SMOOTHED_MOTION_DIR / "speech",),
            include_patterns=("*.csv",),
            compression=CSV_COMPRESSION,
            compresslevel=CSV_COMPRESSLEVEL,
        ),
        ArchiveSpec(
            archive_name="smoothed_motion_song.zip",
            source_paths=(SMOOTHED_MOTION_DIR / "song",),
            include_patterns=("*.csv",),
            compression=CSV_COMPRESSION,
            compresslevel=CSV_COMPRESSLEVEL,
        ),
        ArchiveSpec(
            archive_name="smoothed_video_speech.zip",
            source_paths=(
                AU_ACTIVATION_VIDEO_DIR / "speech",
                LANDMARK_PLOT_VIDEO_DIR / "speech",
                LANDMARK_OVERLAY_VIDEO_DIR / "speech",
            ),
            include_patterns=("*.mp4",),
            compression=VIDEO_COMPRESSION,
            compresslevel=VIDEO_COMPRESSLEVEL,
        ),
        ArchiveSpec(
            archive_name="smoothed_video_song.zip",
            source_paths=(
                AU_ACTIVATION_VIDEO_DIR / "song",
                LANDMARK_PLOT_VIDEO_DIR / "song",
                LANDMARK_OVERLAY_VIDEO_DIR / "song",
            ),
            include_patterns=("*.mp4",),
            compression=VIDEO_COMPRESSION,
            compresslevel=VIDEO_COMPRESSLEVEL,
        ),
    ]


# =============================================================================
# File collection and compression
# =============================================================================

def collect_files(source_paths: tuple[Path, ...], include_patterns: tuple[str, ...]) -> list[Path]:
    """Collect files matching one or more patterns from one or more source paths."""
    files: list[Path] = []

    for source_path in source_paths:
        if not source_path.exists():
            logging.warning("Source path not found; skipping: %s", source_path)
            continue

        for pattern in include_patterns:
            files.extend(sorted(source_path.rglob(pattern)))

    return sorted(set(files))


def archive_member_name(file_path: Path, spec: ArchiveSpec) -> Path:
    """Return the path to use for a file inside an archive.

    For motion archives, files are stored as:
        Actor_01/file.csv

    For video archives, files are stored as:
        action_unit_activation/Actor_01/file.mp4
        landmark_plot/Actor_01/file.mp4
        landmark_overlay/Actor_01/file.mp4
    """
    for source_path in spec.source_paths:
        try:
            relative_path = file_path.relative_to(source_path)
        except ValueError:
            continue

        if len(spec.source_paths) == 1:
            return relative_path

        # For combined video archives, include the video-type folder name.
        video_type_name = source_path.parent.name
        return Path(video_type_name) / relative_path

    raise ValueError(f"File is not inside any source path: {file_path}")


def create_archive(spec: ArchiveSpec) -> None:
    """Create one release ZIP archive."""
    RELEASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RELEASE_ARCHIVE_DIR / spec.archive_name

    if archive_path.exists() and not OVERWRITE_EXISTING:
        logging.info("Archive already exists; skipping: %s", archive_path)
        print(f"Archive already exists; skipping: {archive_path}")
        return

    files = collect_files(spec.source_paths, spec.include_patterns)

    if not files:
        logging.warning("No files found for archive: %s", spec.archive_name)
        print(f"No files found for archive: {spec.archive_name}")
        return

    logging.info("Creating archive: %s", archive_path)
    logging.info("Files to archive: %d", len(files))
    print(f"Creating {archive_path}")
    print(f"Files: {len(files)}")

    zip_kwargs = {
        "file": archive_path,
        "mode": "w",
        "compression": spec.compression,
    }

    if spec.compresslevel is not None:
        zip_kwargs["compresslevel"] = spec.compresslevel

    with zipfile.ZipFile(**zip_kwargs) as zipf:
        for index, file_path in enumerate(files, start=1):
            arcname = archive_member_name(file_path, spec)
            zipf.write(file_path, arcname)

            if index == 1 or index % 500 == 0 or index == len(files):
                print(f"  [{index}/{len(files)}] {arcname}")

    logging.info("Archive created: %s", archive_path)
    print(f"Created: {archive_path}")


# =============================================================================
# Validation summary
# =============================================================================

def summarize_expected_counts() -> None:
    """Print and log quick counts for the release inputs."""
    counts = {
        "raw_motion_speech_csv": len(list((RAW_MOTION_DIR / "speech").rglob("*.csv"))),
        "raw_motion_song_csv": len(list((RAW_MOTION_DIR / "song").rglob("*.csv"))),
        "smoothed_motion_speech_csv": len(list((SMOOTHED_MOTION_DIR / "speech").rglob("*.csv"))),
        "smoothed_motion_song_csv": len(list((SMOOTHED_MOTION_DIR / "song").rglob("*.csv"))),
        "au_activation_speech_mp4": len(list((AU_ACTIVATION_VIDEO_DIR / "speech").rglob("*.mp4"))),
        "au_activation_song_mp4": len(list((AU_ACTIVATION_VIDEO_DIR / "song").rglob("*.mp4"))),
        "landmark_plot_speech_mp4": len(list((LANDMARK_PLOT_VIDEO_DIR / "speech").rglob("*.mp4"))),
        "landmark_plot_song_mp4": len(list((LANDMARK_PLOT_VIDEO_DIR / "song").rglob("*.mp4"))),
        "landmark_overlay_speech_mp4": len(list((LANDMARK_OVERLAY_VIDEO_DIR / "speech").rglob("*.mp4"))),
        "landmark_overlay_song_mp4": len(list((LANDMARK_OVERLAY_VIDEO_DIR / "song").rglob("*.mp4"))),
    }

    print("Input file counts:")
    for key, value in counts.items():
        expected = EXPECTED_COUNTS.get(key)

        if expected is None:
            print(f"  {key}: {value}")
            logging.info("%s: %d", key, value)
            continue

        if value == expected:
            print(f"  {key}: {value}")
            logging.info("%s: %d", key, value)
        else:
            print(f"  {key}: {value}  WARNING: expected {expected}")
            logging.warning(
                "%s count differs from expected: observed=%d expected=%d",
                key,
                value,
                expected,
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Create the six FELT release ZIP archives."""
    configure_logging(LOG_FILE)

    print(f"Writing log to: {LOG_FILE}")

    logging.info("Starting FELT release archive creation.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("RELEASE_ARCHIVE_DIR: %s", RELEASE_ARCHIVE_DIR)
    logging.info("LOG_FILE: %s", LOG_FILE)
    logging.info("OVERWRITE_EXISTING: %s", OVERWRITE_EXISTING)

    summarize_expected_counts()

    specs = build_archive_specs()

    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {spec.archive_name}")
        logging.info("Processing archive %d/%d: %s", index, len(specs), spec.archive_name)
        create_archive(spec)

    logging.info("FELT release archive creation complete.")
    print("\nFELT release archive creation complete.")


if __name__ == "__main__":
    main()