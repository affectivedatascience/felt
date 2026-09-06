"""Experimentally test Py-Feat detection from predecoded video frames.

This compares the current ``detect_video(..., batch_size=1)`` path against a
candidate path that decodes frames with ffmpeg, loads the ordered frame PNGs,
runs the same detector waterfall, and checks count/schema/value equivalence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = SRC_ROOT / "1_extract_raw_tracking.py"
FRAME_RE = re.compile(r"frame_(\d+)\.png$")


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    stream_frames: int | None
    decoded_frames: int | None


def load_extractor_module() -> Any:
    spec = importlib.util.spec_from_file_location("felt_extract_raw_tracking", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extractor module from {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_video(input_dir: Path, stem: str) -> Path:
    matches = sorted(input_dir.rglob(f"{stem}.mp4"))
    if not matches:
        raise FileNotFoundError(f"No video found for stem {stem!r} under {input_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple videos found for stem {stem!r}: {matches}")
    return matches[0]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def read_video_metadata(video_path: Path) -> VideoMetadata:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    return VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        stream_frames=int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        decoded_frames=(
            int(stream["nb_read_frames"]) if stream.get("nb_read_frames") else None
        ),
    )


def decode_frames(
    video_path: Path, frame_dir: Path, overwrite: bool
) -> tuple[list[Path], float]:
    start = time.perf_counter()
    if overwrite and frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("frame_*.png"))
    if not existing:
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-start_number",
                "0",
                str(frame_dir / "frame_%06d.png"),
            ]
        )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frame PNGs in {frame_dir}")
    return frames, time.perf_counter() - start


def frame_number_from_path(path_value: Any) -> int:
    match = FRAME_RE.search(Path(str(path_value)).name)
    if not match:
        raise ValueError(f"Could not parse frame number from {path_value!r}")
    return int(match.group(1))


def format_approx_time(frame: int, fps: float) -> str:
    seconds = int(frame / fps)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def normalize_predecoded_prediction(prediction: pd.DataFrame, fps: float) -> pd.DataFrame:
    output = prediction.copy()
    output["frame"] = output["input"].map(frame_number_from_path)
    output["approx_time"] = output["frame"].map(lambda frame: format_approx_time(frame, fps))
    return output.set_index("frame", drop=False).sort_index()


def run_video_detection(detector: Any, extractor: Any, video_path: Path) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    prediction = detector.detect_video(
        str(video_path),
        output_size=extractor.OUTPUT_SIZE,
        batch_size=1,
        num_workers=0,
        pin_memory=extractor.PIN_MEMORY,
        face_detection_threshold=extractor.FACE_DETECTION_THRESHOLD,
        face_identity_threshold=extractor.FACE_IDENTITY_THRESHOLD,
    )
    return prediction.sort_index(), time.perf_counter() - start


def run_predecoded_detection(
    detector: Any,
    extractor: Any,
    frame_paths: list[Path],
    fps: float,
) -> tuple[pd.DataFrame, float]:
    from feat.data import ImageDataset
    from torch.utils.data import DataLoader

    start = time.perf_counter()
    data_loader = DataLoader(
        ImageDataset(
            [str(path) for path in frame_paths],
            output_size=extractor.OUTPUT_SIZE,
            preserve_aspect_ratio=True,
            padding=False,
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=extractor.PIN_MEMORY,
        shuffle=False,
    )
    batch_output = []
    for batch_data in data_loader:
        (
            faces,
            landmarks,
            poses,
            aus,
            emotions,
            identities,
        ) = detector._run_detection_waterfall(
            batch_data,
            extractor.FACE_DETECTION_THRESHOLD,
            {},
            {},
            {},
            {},
            {},
            {},
        )
        frames = [frame_number_from_path(path) for path in batch_data["FileNames"]]
        output = detector._create_fex(
            faces,
            landmarks,
            poses,
            aus,
            emotions,
            identities,
            batch_data["FileNames"],
            frames,
        )
        batch_output.append(output)
    prediction = pd.concat(batch_output)
    prediction.reset_index(drop=True, inplace=True)
    prediction.compute_identities(threshold=extractor.FACE_IDENTITY_THRESHOLD, inplace=True)
    normalized = normalize_predecoded_prediction(prediction, fps=fps)
    return normalized, time.perf_counter() - start


def run_detect_image_reference(
    detector: Any,
    extractor: Any,
    frame_paths: list[Path],
    fps: float,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    prediction = detector.detect_image(
        [str(path) for path in frame_paths],
        output_size=extractor.OUTPUT_SIZE,
        batch_size=1,
        num_workers=0,
        pin_memory=extractor.PIN_MEMORY,
        frame_counter=0,
        face_detection_threshold=extractor.FACE_DETECTION_THRESHOLD,
        face_identity_threshold=extractor.FACE_IDENTITY_THRESHOLD,
    )
    normalized = normalize_predecoded_prediction(prediction, fps=fps)
    return normalized, time.perf_counter() - start


def numeric_delta_report(
    video_prediction: pd.DataFrame,
    image_prediction: pd.DataFrame,
    tolerance: float,
) -> list[dict[str, Any]]:
    comparable = sorted(
        (set(video_prediction.columns) & set(image_prediction.columns))
        - {"input", "frame", "approx_time"}
    )
    rows: list[dict[str, Any]] = []
    video_aligned = video_prediction.sort_index()
    image_aligned = image_prediction.sort_index()
    for column in comparable:
        video_series = pd.to_numeric(video_aligned[column], errors="coerce")
        image_series = pd.to_numeric(image_aligned[column], errors="coerce")
        both_nan = video_series.isna() & image_series.isna()
        diff = (video_series - image_series).abs()
        valid_diff = diff[~both_nan]
        if valid_diff.empty:
            max_abs = 0.0
            mean_abs = 0.0
            above_tol = 0
        else:
            max_abs = float(valid_diff.max(skipna=True))
            mean_abs = float(valid_diff.mean(skipna=True))
            above_tol = int((valid_diff > tolerance).sum())
        rows.append(
            {
                "column": column,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "values_above_tolerance": above_tol,
            }
        )
    rows.sort(key=lambda row: row["max_abs_diff"], reverse=True)
    return rows


def write_delta_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "column",
                "max_abs_diff",
                "mean_abs_diff",
                "values_above_tolerance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def compare_first_frame_pixels(video_path: Path, frame_path: Path) -> dict[str, Any]:
    import av
    import torch
    from torchvision.io import read_image

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    first_frame = next(container.decode(stream))
    pyav_frame = torch.from_numpy(first_frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
    container.close()

    predecoded_frame = read_image(str(frame_path))
    diff = (pyav_frame.to(torch.int16) - predecoded_frame.to(torch.int16)).abs()
    return {
        "pyav_shape": list(pyav_frame.shape),
        "predecoded_shape": list(predecoded_frame.shape),
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": float(diff.float().mean()),
        "nonzero_values": int((diff != 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare detect_video with ffmpeg-predecoded detect_image on one video."
    )
    parser.add_argument("--stem", default="01-01-01-01-01-01-01")
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "01_data" / "02_output" / "qc" / "predecoded_smoke",
    )
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extractor = load_extractor_module()
    video_path = args.video or find_video(extractor.INPUT_DIR, args.stem)
    metadata = read_video_metadata(video_path)
    frame_dir = args.work_dir / video_path.stem / "frames"
    frame_paths, decode_seconds = decode_frames(
        video_path, frame_dir, overwrite=args.overwrite_frames
    )

    video_detector = extractor.create_detector()
    video_prediction, video_seconds = run_video_detection(video_detector, extractor, video_path)

    predecoded_detector = extractor.create_detector()
    predecoded_prediction, predecoded_seconds = run_predecoded_detection(
        predecoded_detector, extractor, frame_paths, fps=metadata.fps
    )

    video_csv = args.work_dir / video_path.stem / "detect_video.csv"
    predecoded_csv = args.work_dir / video_path.stem / "detect_predecoded_video_like.csv"
    delta_csv = args.work_dir / video_path.stem / "numeric_deltas.csv"
    summary_json = args.work_dir / video_path.stem / "summary.json"
    video_prediction.to_csv(video_csv, index=False)
    predecoded_prediction.to_csv(predecoded_csv, index=False)

    columns_only_in_video = sorted(
        set(video_prediction.columns) - set(predecoded_prediction.columns)
    )
    columns_only_in_predecoded = sorted(
        set(predecoded_prediction.columns) - set(video_prediction.columns)
    )
    expected_frames = set(range(len(frame_paths)))
    video_frames = set(int(x) for x in video_prediction["frame"].dropna().unique())
    predecoded_frames = set(
        int(x) for x in predecoded_prediction["frame"].dropna().unique()
    )
    deltas: list[dict[str, Any]] = []
    comparable_values = (
        len(video_prediction) == len(predecoded_prediction)
        and list(video_prediction["frame"]) == list(predecoded_prediction["frame"])
    )
    if comparable_values:
        deltas = numeric_delta_report(
            video_prediction, predecoded_prediction, args.tolerance
        )
        write_delta_report(delta_csv, deltas)
    columns_above_tolerance = [
        row["column"] for row in deltas if row["values_above_tolerance"] > 0
    ]
    first_frame_pixel_delta = compare_first_frame_pixels(video_path, frame_paths[0])

    summary = {
        "video_path": str(video_path),
        "metadata": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "stream_frames": metadata.stream_frames,
            "decoded_frames_ffprobe": metadata.decoded_frames,
        },
        "decoded_png_frames": len(frame_paths),
        "detect_video_rows": int(len(video_prediction)),
        "predecoded_rows": int(len(predecoded_prediction)),
        "detect_video_unique_frames": len(video_frames),
        "predecoded_unique_frames": len(predecoded_frames),
        "ffmpeg_decode_seconds": decode_seconds,
        "detect_video_seconds": video_seconds,
        "predecoded_detection_seconds": predecoded_seconds,
        "predecoded_total_seconds": decode_seconds + predecoded_seconds,
        "columns_only_in_video": columns_only_in_video,
        "columns_only_in_predecoded": columns_only_in_predecoded,
        "detect_video_missing_frames": sorted(expected_frames - video_frames),
        "predecoded_missing_frames": sorted(expected_frames - predecoded_frames),
        "frame_sequence_equal": comparable_values,
        "max_abs_numeric_diff": max(
            (float(row["max_abs_diff"]) for row in deltas),
            default=None,
        ),
        "columns_above_tolerance_count": len(columns_above_tolerance),
        "columns_above_tolerance_sample": columns_above_tolerance[:25],
        "top_numeric_deltas": deltas[:25],
        "first_frame_pixel_delta": first_frame_pixel_delta,
        "artifacts": {
            "frames": str(frame_dir),
            "detect_video_csv": str(video_csv),
            "predecoded_csv": str(predecoded_csv),
            "numeric_deltas_csv": str(delta_csv) if comparable_values else None,
            "summary_json": str(summary_json),
        },
    }
    write_summary(summary_json, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
