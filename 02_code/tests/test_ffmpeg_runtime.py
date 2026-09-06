from __future__ import annotations

from pathlib import Path

import pytest
from utils import ffmpeg_runtime


def test_explicit_bin_is_resolved_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setattr(ffmpeg_runtime, "ffmpeg_major", lambda _path: 7)
    monkeypatch.setattr(ffmpeg_runtime, "_is_shared_build", lambda _path: True)

    assert ffmpeg_runtime.resolve_ffmpeg_bin(bin_dir) == bin_dir.resolve()


def test_ffmpeg_dir_accepts_installation_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "ffmpeg"
    bin_dir = install_root / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("FFMPEG_DIR", str(install_root))
    monkeypatch.setattr(ffmpeg_runtime, "ffmpeg_major", lambda _path: 6)
    monkeypatch.setattr(ffmpeg_runtime, "_is_shared_build", lambda _path: True)

    assert ffmpeg_runtime.resolve_ffmpeg_bin() == bin_dir.resolve()


def test_rejects_unsupported_explicit_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(ffmpeg_runtime, "ffmpeg_major", lambda _path: 9)

    with pytest.raises(ValueError, match="requires FFmpeg 4-8"):
        ffmpeg_runtime.resolve_ffmpeg_bin(bin_dir)
