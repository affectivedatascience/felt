"""Validate all seven canonical FELT visualization products."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.generate_all_felt_visualizations import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    discover_csvs,
)
from tools.generate_felt_visualization_set import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    organized_output_paths,
)
from utils.felt_paths import OUTPUT_DIR, SMOOTHED_MOTION_DIR  # noqa: E402

EXPECTED_VIDEO_PROPERTIES = {
    view: {
        "codec": "h264",
        "width": 720 if view in DEFAULT_VIEW_NAMES[:3] else 1280,
        "height": 720,
        "frame_rate": "30/1",
    }
    for view in DEFAULT_VIEW_NAMES
}
REPORT_COLUMNS = (
    "status",
    "view",
    "relative_csv_path",
    "video_path",
    "expected_frames",
    "decoded_frames",
    "codec",
    "width",
    "height",
    "frame_rate",
    "size_bytes",
    "error",
)


@dataclass(frozen=True)
class VideoCheck:
    status: str
    view: str
    relative_csv_path: str
    video_path: str
    expected_frames: int
    decoded_frames: int | None
    codec: str
    width: int | None
    height: int | None
    frame_rate: str
    size_bytes: int
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=SMOOTHED_MOTION_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_DIR / "qc" / "visualization_outputs.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "visualization_outputs_summary.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Check paths and non-empty files without decoding every MP4.",
    )
    parser.add_argument("--ffprobe", type=Path)
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    if line_count < 2:
        raise ValueError(f"Smoothed CSV has no data rows: {path}")
    return line_count - 1


def resolve_ffprobe(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"ffprobe not found: {candidate}")
        return candidate
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("ffprobe is required for decoded video QC.")
    return Path(executable).resolve()


def probe_video(path: Path, ffprobe: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream; found {len(streams)}")
    return streams[0]


def check_video(
    csv_path: Path,
    csv_root: Path,
    view: str,
    video_path: Path,
    expected_frames: int,
    ffprobe: Path | None,
) -> VideoCheck:
    relative = csv_path.relative_to(csv_root).as_posix()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return VideoCheck(
            "missing_or_empty",
            view,
            relative,
            str(video_path),
            expected_frames,
            None,
            "",
            None,
            None,
            "",
            video_path.stat().st_size if video_path.is_file() else 0,
        )
    size = video_path.stat().st_size
    if ffprobe is None:
        return VideoCheck(
            "inventory_ok",
            view,
            relative,
            str(video_path),
            expected_frames,
            None,
            "",
            None,
            None,
            "",
            size,
        )
    try:
        probe = probe_video(video_path, ffprobe)
        actual = {
            "codec": str(probe.get("codec_name", "")),
            "width": int(probe["width"]),
            "height": int(probe["height"]),
            "frame_rate": str(probe.get("r_frame_rate", "")),
            "frames": int(probe["nb_read_frames"]),
        }
        expected = EXPECTED_VIDEO_PROPERTIES[view]
        errors = []
        if actual["frames"] != expected_frames:
            errors.append(f"frames {actual['frames']} != {expected_frames}")
        for key in ("codec", "width", "height", "frame_rate"):
            if actual[key] != expected[key]:
                errors.append(f"{key} {actual[key]} != {expected[key]}")
        return VideoCheck(
            "ok" if not errors else "mismatch",
            view,
            relative,
            str(video_path),
            expected_frames,
            actual["frames"],
            actual["codec"],
            actual["width"],
            actual["height"],
            actual["frame_rate"],
            size,
            "; ".join(errors),
        )
    except Exception as exc:
        return VideoCheck(
            "probe_error",
            view,
            relative,
            str(video_path),
            expected_frames,
            None,
            "",
            None,
            None,
            "",
            size,
            repr(exc),
        )


def write_reports(
    report_path: Path,
    summary_path: Path,
    checks: list[VideoCheck],
    trial_count: int,
) -> dict[str, object]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(check) for check in checks)
    status_counts: dict[str, int] = {}
    view_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1
        if check.status in {"ok", "inventory_ok"}:
            view_counts[check.view] = view_counts.get(check.view, 0) + 1
    summary = {
        "schema_version": 1,
        "trial_count": trial_count,
        "expected_view_count": len(DEFAULT_VIEW_NAMES),
        "expected_video_count": trial_count * len(DEFAULT_VIEW_NAMES),
        "checked_video_count": len(checks),
        "status_counts": status_counts,
        "passing_count_by_view": view_counts,
        "passed": len(checks) == trial_count * len(DEFAULT_VIEW_NAMES)
        and set(status_counts).issubset({"ok", "inventory_ok"}),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    csv_root = args.csv_root.resolve()
    video_root = args.video_root.resolve()
    csv_paths = discover_csvs(csv_root, None, None)
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise ValueError(f"No smoothed CSVs found under {csv_root}")
    ffprobe = None if args.inventory_only else resolve_ffprobe(args.ffprobe)

    tasks = []
    for csv_path in csv_paths:
        frames = count_csv_rows(csv_path)
        output_paths = organized_output_paths(csv_path, video_root)
        tasks.extend(
            (csv_path, csv_root, view, output_paths[view], frames, ffprobe)
            for view in DEFAULT_VIEW_NAMES
        )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        checks = list(executor.map(lambda values: check_video(*values), tasks))
    summary = write_reports(
        args.report.resolve(),
        args.summary.resolve(),
        checks,
        len(csv_paths),
    )
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.report.resolve()}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
