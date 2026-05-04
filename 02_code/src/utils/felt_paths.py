"""
Shared path, logging, and RAVDESS filename helpers for the FELT pipeline.

This module contains infrastructure used by the numbered pipeline scripts. It
does not perform tracking, missing-value filling, smoothing, or visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Project paths
# =============================================================================

def find_project_root() -> Path:
    """Find the project root by walking upward from the current file.

    The project root is defined as the folder containing 02_code/.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "02_code").exists():
            return parent
    raise RuntimeError("Could not find project root containing 02_code/.")


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "01_data"
INPUT_DIR = DATA_DIR / "01_input"
OUTPUT_DIR = DATA_DIR / "02_output"

RAW_MOTION_DIR = OUTPUT_DIR / "01_raw_motion"
SMOOTHED_MOTION_DIR = OUTPUT_DIR / "02_smoothed_motion"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"

SMOOTHED_VIDEO_DIR = OUTPUT_DIR / "03_smoothed_video"

AU_ACTIVATION_VIDEO_DIR = SMOOTHED_VIDEO_DIR / "action_unit_activation"
LANDMARK_PLOT_VIDEO_DIR = SMOOTHED_VIDEO_DIR / "landmark_plot"
LANDMARK_OVERLAY_VIDEO_DIR = SMOOTHED_VIDEO_DIR / "landmark_overlay"

RELEASE_ARCHIVE_DIR = OUTPUT_DIR / "04_release_archives"

# =============================================================================
# RAVDESS constants
# =============================================================================

# RAVDESS modality codes: first filename field.
FULL_AV_MODALITY = "01"
VIDEO_ONLY_MODALITY = "02"
AUDIO_ONLY_MODALITY = "03"

# RAVDESS vocal-channel codes: second filename field.
SPEECH_CHANNEL = "01"
SONG_CHANNEL = "02"

# Folder names used in FELT outputs.
SPEECH_FOLDER = "speech"
SONG_FOLDER = "song"
VOCAL_CHANNELS = (SPEECH_FOLDER, SONG_FOLDER)
VALID_VOCAL_CHANNEL_CODES = {SPEECH_CHANNEL, SONG_CHANNEL}

# RAVDESS song stimuli are not available for Actor 18.
ACTOR_WITHOUT_SONG = 18


def actor_name(actor_id: int) -> str:
    """Return the standard RAVDESS actor folder name."""
    return f"Actor_{actor_id:02}"


# =============================================================================
# Logging
# =============================================================================

def configure_logging(log_file: Path) -> None:
    """Configure file and console logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# =============================================================================
# RAVDESS filename helpers
# =============================================================================

@dataclass(frozen=True)
class RavdessCode:
    """Parsed seven-field RAVDESS filename code."""

    modality: str
    vocal_channel: str
    emotion: str
    intensity: str
    statement: str
    repetition: str
    actor: int


def parse_ravdess_stem(path_or_stem: str | Path) -> RavdessCode:
    """Parse a RAVDESS filename stem into its seven coded fields.

    Example
    -------
    01-01-06-01-02-01-12
    """
    stem = Path(path_or_stem).stem
    parts = stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename stem: {stem}")

    try:
        actor = int(parts[6])
    except ValueError as exc:
        raise ValueError(f"Invalid RAVDESS actor field in stem: {stem}") from exc

    return RavdessCode(
        modality=parts[0],
        vocal_channel=parts[1],
        emotion=parts[2],
        intensity=parts[3],
        statement=parts[4],
        repetition=parts[5],
        actor=actor,
    )


def vocal_channel_folder(code: RavdessCode) -> str:
    """Return the output folder name for the RAVDESS vocal-channel code."""
    if code.vocal_channel == SPEECH_CHANNEL:
        return SPEECH_FOLDER
    if code.vocal_channel == SONG_CHANNEL:
        return SONG_FOLDER
    raise ValueError(f"Unexpected RAVDESS vocal-channel code: {code.vocal_channel}")


def should_process_full_av_speech_song(code: RavdessCode) -> bool:
    """Return True for RAVDESS full-audiovisual speech/song files.

    FELT is non-destructive: video-only files (`02-*`) and audio-only files
    (`03-*`) are ignored during task construction rather than deleted from the
    source RAVDESS directory.
    """
    if code.modality != FULL_AV_MODALITY:
        return False

    if code.vocal_channel not in VALID_VOCAL_CHANNEL_CODES:
        return False

    if code.actor == ACTOR_WITHOUT_SONG and code.vocal_channel == SONG_CHANNEL:
        return False

    return True