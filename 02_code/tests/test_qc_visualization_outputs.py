"""Tests for canonical visualization-output QC."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.qc_visualization_outputs import (  # noqa: E402
    EXPECTED_VIDEO_PROPERTIES,
    check_video,
    count_csv_rows,
    write_reports,
)


def test_count_csv_rows_excludes_header(tmp_path: Path) -> None:
    path = tmp_path / "trial.csv"
    path.write_bytes(b"frame,value\r\n0,1\r\n1,2\r\n")

    assert count_csv_rows(path) == 2


def test_inventory_check_rejects_empty_video(tmp_path: Path) -> None:
    csv_path = tmp_path / "csv" / "speech" / "Actor_01" / "trial.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("frame\n0\n", encoding="utf-8")
    video = tmp_path / "empty.mp4"
    video.touch()

    check = check_video(
        csv_path,
        tmp_path / "csv",
        "au_region_heatmap",
        video,
        1,
        None,
    )

    assert check.status == "missing_or_empty"


def test_expected_properties_cover_all_canonical_views() -> None:
    assert len(EXPECTED_VIDEO_PROPERTIES) == 7
    assert EXPECTED_VIDEO_PROPERTIES["au_to_mesh"]["width"] == 720
    assert EXPECTED_VIDEO_PROPERTIES["landmark_overlay_contours"]["width"] == 1280


def test_report_fails_when_a_product_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "csv" / "speech" / "Actor_01" / "trial.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("frame\n0\n", encoding="utf-8")
    check = check_video(
        csv_path,
        tmp_path / "csv",
        "au_region_heatmap",
        tmp_path / "missing.mp4",
        1,
        None,
    )

    summary = write_reports(tmp_path / "report.csv", tmp_path / "summary.json", [check], 1)

    assert not summary["passed"]
