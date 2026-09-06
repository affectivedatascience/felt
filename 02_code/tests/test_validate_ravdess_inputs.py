"""Tests for RAVDESS input inventory validation."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.validate_ravdess_inputs import audit_inputs  # noqa: E402


def test_audit_classifies_full_and_video_only_inputs(tmp_path: Path) -> None:
    actor = tmp_path / "Actor_01"
    actor.mkdir()
    (actor / "01-01-01-01-01-01-01.mp4").write_bytes(b"full")
    (actor / "02-01-01-01-01-01-01.mp4").write_bytes(b"video")

    rows, summary = audit_inputs(tmp_path)

    assert len(rows) == 2
    assert summary["counts"]["full_av_speech"] == 1
    assert summary["counts"]["video_only_speech"] == 1
    assert not summary["passed"]


def test_audit_rejects_filename_actor_folder_mismatch(tmp_path: Path) -> None:
    actor = tmp_path / "Actor_02"
    actor.mkdir()
    (actor / "01-01-01-01-01-01-01.mp4").write_bytes(b"full")

    _, summary = audit_inputs(tmp_path)

    assert any("does not match folder" in error for error in summary["errors"])
