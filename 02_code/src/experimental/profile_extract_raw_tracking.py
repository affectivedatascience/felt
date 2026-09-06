"""Experimentally profile Py-Feat extraction on a small RAVDESS subset.

This tool is intentionally separate from the production extractor. It monkey
patches Py-Feat at runtime to measure the expensive boundaries we care about:
video frame loading, detector stages, XGBoost model reloads, and img2pose retry
behavior. Use ``num_workers=0`` when you need frame-loading timings because
PyTorch DataLoader worker processes do not share this process-local profiler.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = SRC_ROOT / "1_extract_raw_tracking.py"


@dataclass
class TimerStat:
    calls: int = 0
    seconds: float = 0.0
    extras: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds


class Profiler:
    def __init__(self) -> None:
        self.stats: dict[str, TimerStat] = defaultdict(TimerStat)

    def add(self, name: str, seconds: float) -> None:
        self.stats[name].add(seconds)

    def increment(self, name: str, key: str, value: int = 1) -> None:
        self.stats[name].extras[key] += value

    def wrap_callable(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.add(name, time.perf_counter() - start)

        return wrapped


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


def patch_pyfeat(profiler: Profiler) -> None:
    import feat.data
    import xgboost as xgb

    if not hasattr(feat.data.VideoDataset, "_felt_original_get_item"):
        feat.data.VideoDataset._felt_original_get_item = feat.data.VideoDataset.__getitem__
    if not hasattr(feat.data.VideoDataset, "_felt_original_load_frame"):
        feat.data.VideoDataset._felt_original_load_frame = feat.data.VideoDataset.load_frame
    if not hasattr(xgb.XGBClassifier, "_felt_original_load_model"):
        xgb.XGBClassifier._felt_original_load_model = xgb.XGBClassifier.load_model

    original_get_item = feat.data.VideoDataset._felt_original_get_item
    original_load_frame = feat.data.VideoDataset._felt_original_load_frame
    original_load_model = xgb.XGBClassifier._felt_original_load_model

    def timed_get_item(self: Any, idx: int) -> Any:
        start = time.perf_counter()
        try:
            return original_get_item(self, idx)
        finally:
            profiler.add("data.video_dataset.__getitem__", time.perf_counter() - start)

    def timed_load_frame(self: Any, idx: int) -> Any:
        start = time.perf_counter()
        try:
            return original_load_frame(self, idx)
        finally:
            profiler.add("data.video_dataset.load_frame", time.perf_counter() - start)

    def timed_load_model(self: Any, fname: str) -> Any:
        start = time.perf_counter()
        try:
            return original_load_model(self, fname)
        finally:
            profiler.add("xgb.load_model", time.perf_counter() - start)

    feat.data.VideoDataset.__getitem__ = timed_get_item
    feat.data.VideoDataset.load_frame = timed_load_frame
    xgb.XGBClassifier.load_model = timed_load_model


def patch_detector(detector: Any, profiler: Profiler) -> None:
    for method_name in [
        "detect_faces",
        "detect_landmarks",
        "detect_facepose",
        "detect_aus",
        "detect_emotions",
        "detect_identity",
        "_batch_hog",
        "_create_fex",
        "_run_detection_waterfall",
    ]:
        if not hasattr(detector, method_name):
            continue
        bound = getattr(detector, method_name)
        wrapped = profiler.wrap_callable(f"detector.{method_name}", bound)
        setattr(detector, method_name, wrapped)

    if getattr(detector, "au_model", None) is not None and hasattr(detector.au_model, "detect_au"):
        bound = detector.au_model.detect_au
        wrapped = profiler.wrap_callable("model.au.detect_au", bound)
        detector.au_model.detect_au = wrapped

    if getattr(detector, "emotion_model", None) is not None and hasattr(
        detector.emotion_model, "detect_emo"
    ):
        bound = detector.emotion_model.detect_emo
        wrapped = profiler.wrap_callable("model.emotion.detect_emo", bound)
        detector.emotion_model.detect_emo = wrapped

    if getattr(detector, "identity_model", None) is not None and callable(detector.identity_model):
        original_call = detector.identity_model.__call__

        def timed_identity(model_self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_call(*args, **kwargs)
            finally:
                profiler.add("model.identity.__call__", time.perf_counter() - start)

        detector.identity_model.__call__ = MethodType(timed_identity, detector.identity_model)

    for attr_name in ["face_detector", "facepose_detector"]:
        model = getattr(detector, attr_name, None)
        if model is None:
            continue
        if hasattr(model, "predict"):
            original_predict = model.predict

            def make_predict_wrapper(
                label: str, original: Callable[..., Any]
            ) -> Callable[..., Any]:
                @wraps(original)
                def wrapped_predict(*args: Any, **kwargs: Any) -> Any:
                    start = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        profiler.add(f"{label}.predict", time.perf_counter() - start)

                return wrapped_predict

            model.predict = make_predict_wrapper(attr_name, original_predict)
        if hasattr(model, "scale_and_predict"):
            original_scale_and_predict = model.scale_and_predict

            def make_scale_wrapper(
                label: str, original: Callable[..., Any]
            ) -> Callable[..., Any]:
                @wraps(original)
                def wrapped_scale(*args: Any, **kwargs: Any) -> Any:
                    before = profiler.stats[f"{label}.predict"].calls
                    start = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        elapsed = time.perf_counter() - start
                        after = profiler.stats[f"{label}.predict"].calls
                        predict_calls = after - before
                        profiler.add(f"{label}.scale_and_predict", elapsed)
                        profiler.increment(
                            f"{label}.scale_and_predict",
                            "extra_predict_calls",
                            max(0, predict_calls - 1),
                        )

                return wrapped_scale

            model.scale_and_predict = make_scale_wrapper(attr_name, original_scale_and_predict)


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_stem",
        "frames",
        "batch_size",
        "num_workers",
        "stage",
        "calls",
        "seconds",
        "seconds_per_call",
        "percent_detect_video",
        "extra_predict_calls",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_video(
    detector: Any,
    extractor: Any,
    video_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[Any, Profiler, float]:
    profiler = Profiler()
    patch_pyfeat(profiler)
    patch_detector(detector, profiler)

    start = time.perf_counter()
    prediction = detector.detect_video(
        str(video_path),
        output_size=extractor.OUTPUT_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        face_detection_threshold=extractor.FACE_DETECTION_THRESHOLD,
        face_identity_threshold=extractor.FACE_IDENTITY_THRESHOLD,
    )
    elapsed = time.perf_counter() - start
    profiler.add("detector.detect_video.total", elapsed)
    return prediction, profiler, elapsed


def render_rows(
    video_stem: str,
    frames: int,
    batch_size: int,
    num_workers: int,
    profiler: Profiler,
) -> list[dict[str, Any]]:
    total = profiler.stats["detector.detect_video.total"].seconds
    rows = []
    for stage, stat in sorted(
        profiler.stats.items(), key=lambda item: item[1].seconds, reverse=True
    ):
        rows.append(
            {
                "video_stem": video_stem,
                "frames": frames,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "stage": stage,
                "calls": stat.calls,
                "seconds": f"{stat.seconds:.6f}",
                "seconds_per_call": f"{stat.seconds / stat.calls:.6f}" if stat.calls else "",
                "percent_detect_video": f"{(stat.seconds / total * 100):.2f}" if total else "",
                "extra_predict_calls": stat.extras.get("extra_predict_calls", 0),
            }
        )
    return rows


def print_summary(rows: list[dict[str, Any]], top: int) -> None:
    print("Top profiling rows:")
    for row in rows[:top]:
        print(
            f"{row['stage']}: {row['seconds']}s, calls={row['calls']}, "
            f"{row['percent_detect_video']}% of detect_video"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Py-Feat raw tracking extraction on selected RAVDESS videos."
    )
    parser.add_argument(
        "--stem",
        action="append",
        default=[],
        help="RAVDESS video stem to profile. Can be repeated.",
    )
    parser.add_argument("--video", action="append", default=[], help="Explicit MP4 path to profile.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "01_data" / "02_output" / "logs" / "profile_extract_raw_tracking.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extractor = load_extractor_module()
    stems = args.stem or ["01-01-01-01-01-01-01"]
    videos = [Path(path) for path in args.video]
    videos.extend(find_video(extractor.INPUT_DIR, stem) for stem in stems)

    all_rows: list[dict[str, Any]] = []
    for video_path in videos:
        print(f"Profiling {video_path}")
        detector = extractor.create_detector()
        prediction, profiler, elapsed = profile_video(
            detector=detector,
            extractor=extractor,
            video_path=video_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        frames = len(prediction)
        print(f"Completed {video_path.name}: {frames} rows in {elapsed:.3f}s")
        rows = render_rows(
            video_path.stem, frames, args.batch_size, args.num_workers, profiler
        )
        print_summary(rows, args.top)
        all_rows.extend(rows)

    write_report(args.report, all_rows)
    print(f"Wrote profiling report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
