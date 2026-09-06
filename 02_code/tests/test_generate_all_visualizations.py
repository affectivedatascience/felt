"""Tests for the canonical seven-view batch renderer."""

from __future__ import annotations

from pathlib import Path

from tools.generate_all_felt_visualizations import (
    DEFAULT_VIEW_NAMES,
    discover_csvs,
    missing_views,
)
from tools.generate_felt_visualization_set import organized_output_paths


def test_discover_csvs_is_deterministic_and_filters_actor(tmp_path: Path) -> None:
    expected = []
    for channel in ("speech", "song"):
        for actor in (2, 1):
            path = tmp_path / channel / f"Actor_{actor:02d}" / f"{channel}-{actor}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frame\n0\n", encoding="utf-8")
            if actor == 1:
                expected.append(path)

    discovered = discover_csvs(tmp_path, None, [1])

    assert discovered == expected


def test_missing_views_requires_all_seven_nonempty_products(tmp_path: Path) -> None:
    input_csv = tmp_path / "speech" / "Actor_01" / "trial-01.csv"
    output_root = tmp_path / "video"
    paths = organized_output_paths(input_csv, output_root)

    assert missing_views(input_csv, output_root, overwrite=False) == DEFAULT_VIEW_NAMES

    first_view = DEFAULT_VIEW_NAMES[0]
    paths[first_view].parent.mkdir(parents=True)
    paths[first_view].write_bytes(b"video")
    assert first_view not in missing_views(input_csv, output_root, overwrite=False)

    paths[first_view].write_bytes(b"")
    assert first_view in missing_views(input_csv, output_root, overwrite=False)


def test_overwrite_selects_every_canonical_view(tmp_path: Path) -> None:
    input_csv = tmp_path / "song" / "Actor_01" / "trial-01.csv"

    assert missing_views(input_csv, tmp_path / "video", overwrite=True) == DEFAULT_VIEW_NAMES
