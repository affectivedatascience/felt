"""Locate and register an FFmpeg build usable by FELT on Windows."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

_FFMPEG_DLL_HANDLE: Any = None


def _version_key(path: Path) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in path.name.split("."))
    except ValueError:
        return ()


def ffmpeg_major(bin_dir: Path) -> int | None:
    """Return the installed FFmpeg major version, or ``None`` if unavailable."""
    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    executable = bin_dir / executable_name
    if not executable.is_file():
        return None
    try:
        result = subprocess.run(
            [str(executable), "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.splitlines()[0]
        version = first_line.split("version", maxsplit=1)[1].strip().split()[0]
        return int(version.split(".", maxsplit=1)[0])
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None


def _is_shared_build(bin_dir: Path) -> bool:
    return os.name != "nt" or any(bin_dir.glob("avcodec-*.dll"))


def _validate_bin(bin_dir: Path) -> Path:
    candidate = bin_dir.expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"FFmpeg bin directory not found: {candidate}")
    major = ffmpeg_major(candidate)
    if major is None or not 4 <= major <= 8:
        raise ValueError(
            f"TorchCodec requires FFmpeg 4-8; found {major or 'unknown'} "
            f"in {candidate}"
        )
    if not _is_shared_build(candidate):
        raise ValueError(f"FFmpeg shared DLLs were not found in {candidate}")
    return candidate


def resolve_ffmpeg_bin(explicit: Path | None = None) -> Path | None:
    """Locate a shared FFmpeg 4-8 build suitable for TorchCodec.

    Resolution order is an explicit CLI path, ``FFMPEG_DIR``, the executable
    on ``PATH``, and the versioned Scoop ``ffmpeg-shared`` installation.
    ``FFMPEG_DIR`` may name either the installation root or its ``bin`` folder.
    """
    if explicit is not None:
        return _validate_bin(explicit)

    configured = os.environ.get("FFMPEG_DIR")
    if configured:
        root = Path(configured)
        candidate = root / "bin" if (root / "bin").is_dir() else root
        return _validate_bin(candidate)

    executable = shutil.which("ffmpeg")
    if executable:
        candidate = Path(executable).resolve().parent
        major = ffmpeg_major(candidate)
        if _is_shared_build(candidate) and major is not None and 4 <= major <= 8:
            return candidate

    if os.name == "nt":
        scoop_root = Path.home() / "scoop" / "apps" / "ffmpeg-shared"
        candidates: list[tuple[tuple[int, ...], Path]] = []
        if scoop_root.is_dir():
            for version_dir in scoop_root.iterdir():
                key = _version_key(version_dir)
                bin_dir = version_dir / "bin"
                if key and 4 <= key[0] <= 8 and _is_shared_build(bin_dir):
                    candidates.append((key, bin_dir))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1].resolve()
    return None


def configure_ffmpeg_dlls(ffmpeg_bin: str | Path | None) -> None:
    """Register FFmpeg DLLs before importing TorchCodec/Py-Feat on Windows."""
    global _FFMPEG_DLL_HANDLE
    if os.name == "nt" and ffmpeg_bin:
        _FFMPEG_DLL_HANDLE = os.add_dll_directory(str(ffmpeg_bin))


def resolve_ffprobe(ffmpeg_bin: Path | None) -> Path | None:
    """Locate ffprobe alongside FFmpeg, falling back to ``PATH``."""
    executable_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    if ffmpeg_bin is not None:
        candidate = ffmpeg_bin / executable_name
        if candidate.is_file():
            return candidate
    executable = shutil.which("ffprobe")
    return Path(executable).resolve() if executable else None
