"""Tests for byte-preserving source-reference normalization."""

from __future__ import annotations

from pathlib import Path

from tools.normalize_source_references import normalize_file


def test_normalize_file_changes_only_absolute_input_tokens(tmp_path: Path) -> None:
    path = tmp_path / "speech" / "Actor_01" / "trial.csv"
    path.parent.mkdir(parents=True)
    before = (
        b"frame,input,value\r\n"
        b"0,E:\\old\\Actor_01\\trial.mp4,1.234567890123\r\n"
        b"1,E:\\old\\Actor_01\\trial.mp4,9.876543210987\r\n"
    )
    path.write_bytes(before)

    result = normalize_file(path, dry_run=False)

    assert result.status == "normalized"
    assert result.replacements == 2
    assert path.read_bytes() == before.replace(
        b"E:\\old\\Actor_01\\trial.mp4", b"Actor_01/trial.mp4"
    )


def test_normalize_file_dry_run_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "song" / "Actor_02" / "trial.csv"
    path.parent.mkdir(parents=True)
    before = b"frame,input\n0,C:\\old\\Actor_02\\trial.mp4\n"
    path.write_bytes(before)

    result = normalize_file(path, dry_run=True)

    assert result.status == "would_normalize"
    assert path.read_bytes() == before
