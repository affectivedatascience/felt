"""Tests for Detectorv2 extraction source-linkage QC."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.qc_pyfeat_v2_extraction import source_reference_status  # noqa: E402


def test_portable_source_reference_resolves_under_video_root(tmp_path: Path) -> None:
    source = tmp_path / "Actor_01" / "trial.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")

    status = source_reference_status("Actor_01/trial.mp4", "trial", tmp_path)

    assert status == (True, True, True)


def test_absolute_source_reference_is_nonportable(tmp_path: Path) -> None:
    source = tmp_path / "Actor_01" / "trial.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")

    stem_match, portable, exists = source_reference_status(
        str(source), "trial", tmp_path
    )

    assert stem_match
    assert not portable
    assert exists
