from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "tools" / "animate_au_mesh_csv.py"
SPEC = importlib.util.spec_from_file_location("animate_au_mesh_csv", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_rows_preserves_observed_rows() -> None:
    data = pd.DataFrame({"frame": np.arange(10), "AU01": np.arange(10) / 10})

    selected = MODULE.select_rows(data, stride=2, max_frames=3)

    assert selected["frame"].tolist() == [0, 4, 8]
    assert selected.index.tolist() == [0, 4, 8]


def test_au_array_uses_model_column_order() -> None:
    data = pd.DataFrame({"AU02": [0.2], "AU01": [0.1]})

    values = MODULE.au_array(data, ["AU01", "AU02"])

    np.testing.assert_allclose(values, [[0.1, 0.2]])


def test_au_array_rejects_invalid_values() -> None:
    data = pd.DataFrame({"AU01": [0.1], "AU02": ["bad"]}, index=[17])

    with pytest.raises(ValueError, match=r"row 17, column AU02"):
        MODULE.au_array(data, ["AU01", "AU02"])


def test_project_mesh_2d_flips_vertical_axis_like_pyfeat_atlas() -> None:
    mesh = np.zeros((478, 3), dtype=np.float32)
    mesh[10, 1] = -2
    mesh[152, 1] = 3

    projected = MODULE.project_mesh_2d(mesh)

    assert projected[10, 1] == 2
    assert projected[152, 1] == -3


def test_project_mesh_2d_rejects_wrong_topology() -> None:
    with pytest.raises(ValueError, match="expected mesh shape"):
        MODULE.project_mesh_2d(np.zeros((68, 3), dtype=np.float32))


def test_organized_au_to_mesh_path_matches_other_products(tmp_path: Path) -> None:
    generator_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tools"
        / "generate_felt_visualization_set.py"
    )
    spec = importlib.util.spec_from_file_location("generate_visualizations", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    input_csv = tmp_path / "song" / "Actor_01" / "01-02-01-01-01-01-01.csv"

    output = generator.organized_output_paths(input_csv, tmp_path / "output")

    assert output["au_to_mesh"] == (
        tmp_path
        / "output"
        / "AU_animation"
        / "au_to_mesh"
        / "Actor_01"
        / "au_to_mesh_01-02-01-01-01-01-01.mp4"
    )


def test_source_video_fallback_uses_configured_input_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tools"
        / "generate_felt_visualization_set.py"
    )
    spec = importlib.util.spec_from_file_location("generate_source_lookup", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    input_root = tmp_path / "ravdess"
    csv_path = tmp_path / "smooth" / "speech" / "Actor_03" / "trial-03.csv"
    source = input_root / "Actor_03" / "trial-03.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(generator, "INPUT_DIR", input_root)

    assert generator.infer_source_video(pd.DataFrame(), csv_path) == source
