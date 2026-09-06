from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "1_extract_raw_tracking.py"
SPEC = importlib.util.spec_from_file_location("extract_raw_tracking_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def test_build_tasks_supports_separate_input_and_output_roots(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "raw"
    actor_dir = input_root / "Actor_01"
    actor_dir.mkdir(parents=True)
    target = actor_dir / "01-01-01-01-01-01-01.mp4"
    target.touch()
    (actor_dir / "02-01-01-01-01-01-01.mp4").touch()

    filters = extractor.TaskFilters(
        start_actor=1,
        end_actor=1,
        actors=(),
        vocal_channel="all",
        stems=(),
        file_list=None,
        limit=None,
        first=False,
    )

    tasks = extractor.build_tasks(filters, input_root, output_root)

    assert tasks == [
        extractor.DetectionTask(
            video_path=target,
            csv_path=output_root
            / "speech"
            / "Actor_01"
            / "01-01-01-01-01-01-01.csv",
        )
    ]


def test_cpu_device_configures_single_thread_native_libraries(
    monkeypatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    extractor.configure_compute_environment("cpu")

    assert extractor.os.environ["OMP_NUM_THREADS"] == "1"
    assert extractor.os.environ["MKL_NUM_THREADS"] == "1"
    assert extractor.os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert extractor.os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_save_prediction_csv_canonicalizes_source_path(tmp_path: Path) -> None:
    prediction = pd.DataFrame(
        {
            "input": [r"E:\old\machine\Actor_03\trial.mp4"],
            "frame": [0],
        }
    )
    output = tmp_path / "raw" / "trial.csv"

    extractor.save_prediction_csv(
        prediction,
        output,
        tmp_path / "input" / "Actor_03" / "trial.mp4",
    )

    saved = pd.read_csv(output)
    assert saved["input"].tolist() == ["Actor_03/trial.mp4"]
