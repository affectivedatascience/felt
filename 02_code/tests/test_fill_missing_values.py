from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "2_fill_missing_values.py"
SPEC = importlib.util.spec_from_file_location("fill_missing_values_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fill
SPEC.loader.exec_module(fill)


def write_sample(path: Path, target_value: str = "", face_score: str = "0.0") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",value,FaceScore,FrameHeight,FrameWidth,input,frame,approx_time\n"
        "0,10,0.9,720.0,1280.0,Actor_01/test.mp4,0,00:00\n"
        f"1,{target_value},{face_score},720.0,1280.0,Actor_01/test.mp4,1,00:00\n",
        encoding="utf-8",
    )


def task_for(path: Path, expected_missing_cells: int = 1):
    return fill.FillTask(
        relative_csv_path="song/Actor_01/test.csv",
        csv_path=path,
        frame=1,
        method=fill.SUPPORTED_METHOD,
        expected_missing_cells=expected_missing_cells,
        rationale="test",
    )


def test_process_file_replaces_failed_model_outputs_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "test.csv"
    write_sample(csv_path)

    result = fill.process_file(task_for(csv_path))

    assert result.status == "filled"
    assert result.missing_cells_before == 1
    assert result.missing_cells_after == 0
    assert result.overwritten_nonblank_cells == 1
    assert csv_path.read_text(encoding="utf-8").splitlines()[2] == (
        "1,10,0.9,720.0,1280.0,Actor_01/test.mp4,1,00:00"
    )


def test_process_file_is_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    write_sample(csv_path, target_value="10", face_score="0.9")
    original = csv_path.read_bytes()

    result = fill.process_file(task_for(csv_path))

    assert result.status == "already_corrected"
    assert csv_path.read_bytes() == original


def test_process_file_rejects_incomplete_prior_repair(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    write_sample(csv_path, target_value="10", face_score="0.0")

    with pytest.raises(fill.FillNaNError, match="does not match"):
        fill.process_file(task_for(csv_path))


def test_process_file_rejects_unexpected_missing_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    write_sample(csv_path)

    with pytest.raises(fill.FillNaNError, match="expected 2, found 1"):
        fill.process_file(task_for(csv_path, expected_missing_cells=2))


def test_resolve_manifest_target_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(fill.FillNaNError, match="Unsafe"):
        fill.resolve_manifest_target(tmp_path, "../outside.csv")
