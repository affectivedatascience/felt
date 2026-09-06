"""Run the retained one-image smoke test from the Py-Feat 2 migration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.ffmpeg_runtime import (  # noqa: E402
    configure_ffmpeg_dlls,
    resolve_ffmpeg_bin,
)

PYFEAT_VERSION = "2.0.3"
EXPECTED_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
EXPECTED_EMOTIONS = [
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger",
]
EXPECTED_GAZE = ["gaze_pitch", "gaze_yaw", "gaze_angle"]
EXPECTED_POSE = ["Pitch", "Roll", "Yaw", "X", "Y", "Z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Py-Feat 2.0.3 Detectorv2 on one face image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Face image to detect; defaults to Py-Feat's bundled test image.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        help="Directory containing a shared FFmpeg 4-8 build.",
    )
    return parser.parse_args()


def require_columns(actual: set[str], expected: list[str], family: str) -> None:
    missing = [column for column in expected if column not in actual]
    if missing:
        raise RuntimeError(f"Detectorv2 output is missing {family} columns: {missing}")


def main() -> int:
    args = parse_args()

    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    configure_ffmpeg_dlls(ffmpeg_bin)

    installed_version = importlib.metadata.version("py-feat")
    if installed_version != PYFEAT_VERSION:
        raise RuntimeError(
            f"Expected py-feat {PYFEAT_VERSION}, found {installed_version}."
        )

    import torch
    import torchcodec
    from feat import Detectorv2
    from feat.utils import MP_BLENDSHAPE_NAMES
    from feat.utils.io import get_test_data_path

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested, but torch {torch.__version__} cannot access it."
        )

    image_path = args.image or Path(get_test_data_path()) / "single_face.jpg"
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Smoke-test image not found: {image_path}")

    started = time.perf_counter()
    detector = Detectorv2(device=args.device, identity_model="arcface")
    detector_load_seconds = time.perf_counter() - started

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    prediction = detector.detect(
        str(image_path),
        data_type="image",
        progress_bar=False,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    if prediction.empty:
        raise RuntimeError(f"Detectorv2 found no face in {image_path}")

    columns = set(prediction.columns)
    require_columns(columns, EXPECTED_AUS, "action-unit")
    require_columns(columns, EXPECTED_EMOTIONS, "emotion")
    require_columns(columns, EXPECTED_GAZE, "gaze")
    require_columns(columns, EXPECTED_POSE, "pose")
    require_columns(columns, ["valence", "arousal"], "affect")
    require_columns(columns, [f"x_{i}" for i in range(68)], "x-landmark")
    require_columns(columns, [f"y_{i}" for i in range(68)], "y-landmark")
    require_columns(columns, [f"mesh_x_{i}" for i in range(478)], "mesh-x")
    require_columns(columns, [f"mesh_y_{i}" for i in range(478)], "mesh-y")
    require_columns(columns, [f"mesh_z_{i}" for i in range(478)], "mesh-z")
    require_columns(columns, list(MP_BLENDSHAPE_NAMES), "blendshape")

    result = {
        "status": "ok",
        "py_feat": installed_version,
        "torch": torch.__version__,
        "torchcodec": torchcodec.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "detector_load_seconds": round(detector_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_gpu_mib": (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            if args.device == "cuda"
            else None
        ),
        "rows": len(prediction),
        "columns": len(prediction.columns),
        "face_score": round(float(prediction.iloc[0]["FaceScore"]), 6),
        "image": str(image_path),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
