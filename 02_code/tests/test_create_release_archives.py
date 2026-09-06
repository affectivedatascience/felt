"""Tests for deterministic FELT v2 release packaging."""

from __future__ import annotations

import json
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
from tools.create_release_archives import (
    ArchiveSpec,
    collect_files,
    create_archive,
    member_name,
    sha256_file,
    validate_qc_evidence,
    verify_archive,
)


def sample_spec(root: Path, archive_name: str = "sample.zip") -> ArchiveSpec:
    return ArchiveSpec(
        archive_name=archive_name,
        source_root=root.resolve(),
        member_root="component",
        suffix=".csv",
        expected_file_count=2,
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


def make_sample_files(root: Path) -> None:
    for relative, content in (
        (Path("song/Actor_01/b.csv"), "frame,value\n0,2\n"),
        (Path("speech/Actor_01/a.csv"), "frame,value\n0,1\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_collect_files_requires_exact_nonempty_inventory(tmp_path: Path) -> None:
    make_sample_files(tmp_path)
    spec = sample_spec(tmp_path)

    files = collect_files(spec)

    assert [path.name for path in files] == ["b.csv", "a.csv"]
    assert [member_name(spec, path) for path in files] == [
        "component/song/Actor_01/b.csv",
        "component/speech/Actor_01/a.csv",
    ]


def test_collect_files_rejects_wrong_count(tmp_path: Path) -> None:
    (tmp_path / "only.csv").write_text("frame\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"found 1 \.csv files; expected 2"):
        collect_files(sample_spec(tmp_path))


def test_archive_bytes_are_deterministic_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_sample_files(source)
    files = collect_files(sample_spec(source))
    first_spec = sample_spec(source, "first.zip")
    second_spec = sample_spec(source, "second.zip")

    first = create_archive(first_spec, files, tmp_path / "out", overwrite=False)
    second = create_archive(second_spec, files, tmp_path / "out", overwrite=False)

    assert sha256_file(first) == sha256_file(second)
    expected = [member_name(first_spec, path) for path in files]
    verify_archive(first, expected)


def test_validate_qc_evidence_requires_all_release_gates(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    raw_path = tmp_path / "raw.json"
    smooth_path = tmp_path / "smooth.json"
    video_path = tmp_path / "video.json"
    input_path.write_text(
        json.dumps(
            {
                "passed": True,
                "counts": {
                    "all_mp4": 4904,
                    "full_av": 2452,
                    "full_av_song": 1012,
                    "full_av_speech": 1440,
                    "video_only": 2452,
                    "video_only_song": 1012,
                    "video_only_speech": 1440,
                },
            }
        ),
        encoding="utf-8",
    )
    raw_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    smooth_path.write_text(
        json.dumps(
            {
                "available_file_count": 2452,
                "processed_count": 2400,
                "checkpoint_reused_count": 52,
            }
        ),
        encoding="utf-8",
    )
    video_path.write_text(
        json.dumps(
            {
                "passed": True,
                "checked_video_count": 17164,
                "status_counts": {"ok": 17164},
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_qc_evidence(
        Namespace(
            input_qc_summary=input_path,
            raw_qc_summary=raw_path,
            smoothing_qc_summary=smooth_path,
            video_qc_summary=video_path,
        )
    )

    assert evidence["video_qc_summary"] == str(video_path.resolve())
