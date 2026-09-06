from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.qc_compare_motion_corpora import compare_file  # noqa: E402


def write_csv(path: Path, value: float, label: str = "Person_0") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [0, 1], "x_0": [value, 2.0], "Identity": [label, label]}).to_csv(
        path, index=False
    )


def test_compare_file_accepts_declared_numeric_tolerance(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    write_csv(candidate, 1.000001)
    write_csv(reference, 1.0)

    result = compare_file("trial.csv", candidate, reference, atol=2e-6, rtol=0.0)

    assert result.status == "ok"
    assert result.out_of_tolerance_cells == 0
    assert result.max_absolute_error > 0


def test_compare_file_rejects_numeric_and_text_differences(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    write_csv(candidate, 1.1, "Person_1")
    write_csv(reference, 1.0)

    result = compare_file("trial.csv", candidate, reference, atol=1e-6, rtol=1e-5)

    assert result.status == "failed"
    assert result.out_of_tolerance_cells == 1
    assert result.non_numeric_mismatches == 2
