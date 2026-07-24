"""Check raw-motion CSV frame counts against source videos at scale.

The script matches each CSV to a source video by relative path first, then by
unique filename stem as a fallback. It writes one row per CSV plus a small
summary to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CSV_ROOT = Path(r"U:\felt\raw_motion_song")
DEFAULT_VIDEO_ROOT = Path(r"E:\github_repos\felt\01_data\01_all_input")
DEFAULT_REPORT = Path(
    r"E:\github_repos\felt\01_data\02_output\logs\raw_motion_song_frame_qc.csv"
)
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


@dataclass(frozen=True)
class CsvStats:
    line_count: int
    data_rows: int
    first_frame: int | None
    last_frame: int | None
    invalid_frame_values: int
    non_monotonic_steps: int
    missing_frame_gaps: int


@dataclass(frozen=True)
class VideoStats:
    metadata_frames: int | None
    decoded_frames: int | None
    duration: str
    avg_frame_rate: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC raw-motion CSV frame counts against source videos."
    )
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, (os.cpu_count() or 4)),
        help="Number of concurrent ffprobe workers.",
    )
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated video suffixes to consider.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("decoded", "metadata"),
        default="decoded",
        help="Video count used for pass/fail comparison.",
    )
    return parser.parse_args()


def iter_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    normalized = tuple(s.lower() for s in suffixes)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized:
            yield path


def build_stem_index(video_paths: list[Path]) -> dict[str, Path | None]:
    index: dict[str, Path | None] = {}
    for video_path in video_paths:
        key = video_path.stem.lower()
        if key in index:
            index[key] = None
        else:
            index[key] = video_path
    return index


def find_video(
    csv_path: Path,
    csv_root: Path,
    video_root: Path,
    video_suffixes: tuple[str, ...],
    stem_index: dict[str, Path | None],
) -> tuple[Path | None, str]:
    relative = csv_path.relative_to(csv_root).with_suffix("")
    for suffix in video_suffixes:
        candidate = video_root / relative.with_suffix(suffix)
        if candidate.exists():
            return candidate, "relative_path"

    fallback = stem_index.get(csv_path.stem.lower())
    if fallback is not None:
        return fallback, "unique_stem"
    if csv_path.stem.lower() in stem_index:
        return None, "ambiguous_stem"
    return None, "missing"


def read_csv_stats(csv_path: Path) -> CsvStats:
    line_count = 0
    data_rows = 0
    first_frame: int | None = None
    last_frame: int | None = None
    invalid_frame_values = 0
    non_monotonic_steps = 0
    missing_frame_gaps = 0

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for line in f:
            line_count += 1
            if line_count == 1:
                continue

            data_rows += 1
            raw_frame = line.split(",", 1)[0].strip()
            try:
                frame = int(raw_frame)
            except ValueError:
                invalid_frame_values += 1
                continue

            if first_frame is None:
                first_frame = frame
            if last_frame is not None:
                if frame <= last_frame:
                    non_monotonic_steps += 1
                elif frame > last_frame + 1:
                    missing_frame_gaps += frame - last_frame - 1
            last_frame = frame

    return CsvStats(
        line_count=line_count,
        data_rows=data_rows,
        first_frame=first_frame,
        last_frame=last_frame,
        invalid_frame_values=invalid_frame_values,
        non_monotonic_steps=non_monotonic_steps,
        missing_frame_gaps=missing_frame_gaps,
    )


def parse_optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def ffprobe_video_stats(video_path: Path) -> VideoStats:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return VideoStats(None, None, "", "", str(exc))

    if completed.returncode != 0:
        return VideoStats(None, None, "", "", completed.stderr.strip())

    try:
        payload = json.loads(completed.stdout)
        stream = payload.get("streams", [{}])[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        return VideoStats(None, None, "", "", f"Could not parse ffprobe JSON: {exc}")

    return VideoStats(
        metadata_frames=parse_optional_int(stream.get("nb_frames")),
        decoded_frames=parse_optional_int(stream.get("nb_read_frames")),
        duration=str(stream.get("duration") or ""),
        avg_frame_rate=str(stream.get("avg_frame_rate") or ""),
        error="",
    )


def qc_one(
    csv_path: Path,
    csv_root: Path,
    video_root: Path,
    video_suffixes: tuple[str, ...],
    stem_index: dict[str, Path | None],
    count_mode: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "csv_path": str(csv_path),
        "relative_csv_path": str(csv_path.relative_to(csv_root)),
        "video_path": "",
        "match_method": "",
        "status": "",
        "csv_lines": "",
        "csv_data_rows": "",
        "csv_first_frame": "",
        "csv_last_frame": "",
        "csv_invalid_frame_values": "",
        "csv_non_monotonic_steps": "",
        "csv_missing_frame_gaps": "",
        "video_metadata_frames": "",
        "video_decoded_frames": "",
        "delta_metadata_minus_decoded": "",
        "video_count_used": "",
        "delta_csv_minus_video": "",
        "video_duration": "",
        "video_avg_frame_rate": "",
        "error": "",
    }

    try:
        csv_stats = read_csv_stats(csv_path)
    except OSError as exc:
        row["status"] = "csv_error"
        row["error"] = str(exc)
        return row

    row.update(
        {
            "csv_lines": csv_stats.line_count,
            "csv_data_rows": csv_stats.data_rows,
            "csv_first_frame": csv_stats.first_frame if csv_stats.first_frame is not None else "",
            "csv_last_frame": csv_stats.last_frame if csv_stats.last_frame is not None else "",
            "csv_invalid_frame_values": csv_stats.invalid_frame_values,
            "csv_non_monotonic_steps": csv_stats.non_monotonic_steps,
            "csv_missing_frame_gaps": csv_stats.missing_frame_gaps,
        }
    )
    video_path, match_method = find_video(
        csv_path, csv_root, video_root, video_suffixes, stem_index
    )
    row["match_method"] = match_method
    if video_path is None:
        row["status"] = "missing_video" if match_method == "missing" else "ambiguous_video"
        return row

    row["video_path"] = str(video_path)
    video_stats = ffprobe_video_stats(video_path)
    row.update(
        {
            "video_metadata_frames": video_stats.metadata_frames
            if video_stats.metadata_frames is not None
            else "",
            "video_decoded_frames": video_stats.decoded_frames
            if video_stats.decoded_frames is not None
            else "",
            "video_duration": video_stats.duration,
            "video_avg_frame_rate": video_stats.avg_frame_rate,
        }
    )
    if (
        video_stats.metadata_frames is not None
        and video_stats.decoded_frames is not None
    ):
        row["delta_metadata_minus_decoded"] = (
            video_stats.metadata_frames - video_stats.decoded_frames
        )
    if video_stats.error:
        row["status"] = "ffprobe_error"
        row["error"] = video_stats.error
        return row

    video_count = (
        video_stats.decoded_frames if count_mode == "decoded" else video_stats.metadata_frames
    )
    row["video_count_used"] = video_count if video_count is not None else ""
    if video_count is None:
        row["status"] = "no_video_frame_count"
        return row

    delta = csv_stats.data_rows - video_count
    row["delta_csv_minus_video"] = delta

    has_csv_frame_issue = (
        csv_stats.invalid_frame_values > 0
        or csv_stats.non_monotonic_steps > 0
        or csv_stats.missing_frame_gaps > 0
    )
    has_video_count_disagreement = (
        video_stats.metadata_frames is not None
        and video_stats.decoded_frames is not None
        and video_stats.metadata_frames != video_stats.decoded_frames
    )
    if delta != 0:
        row["status"] = "frame_count_mismatch"
    elif has_csv_frame_issue:
        row["status"] = "csv_frame_index_issue"
    elif has_video_count_disagreement:
        row["status"] = "video_frame_count_disagreement"
    else:
        row["status"] = "ok"

    return row


def write_report(report_path: Path, rows: list[dict[str, object]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "relative_csv_path",
        "csv_data_rows",
        "video_count_used",
        "delta_csv_minus_video",
        "csv_first_frame",
        "csv_last_frame",
        "csv_invalid_frame_values",
        "csv_non_monotonic_steps",
        "csv_missing_frame_gaps",
        "video_metadata_frames",
        "video_decoded_frames",
        "delta_metadata_minus_decoded",
        "video_duration",
        "video_avg_frame_rate",
        "match_method",
        "csv_path",
        "video_path",
        "csv_lines",
        "error",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    csv_root = args.csv_root.resolve()
    video_root = args.video_root.resolve()
    video_suffixes = tuple(
        suffix.strip().lower()
        for suffix in args.video_extensions.split(",")
        if suffix.strip()
    )

    if not csv_root.exists():
        print(f"CSV root does not exist: {csv_root}", file=sys.stderr)
        return 2
    if not video_root.exists():
        print(f"Video root does not exist: {video_root}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    csv_paths = sorted(iter_files(csv_root, (".csv",)))
    video_paths = sorted(iter_files(video_root, video_suffixes))
    stem_index = build_stem_index(video_paths)

    print(f"Found {len(csv_paths)} CSV files under {csv_root}")
    print(f"Found {len(video_paths)} candidate videos under {video_root}")
    print(f"Writing report to {args.report}")

    rows: list[dict[str, object]] = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                qc_one,
                csv_path,
                csv_root,
                video_root,
                video_suffixes,
                stem_index,
                args.count_mode,
            )
            for csv_path in csv_paths
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(csv_paths):
                print(f"Processed {completed_count}/{len(csv_paths)}")

    rows.sort(key=lambda row: str(row["relative_csv_path"]))
    write_report(args.report, rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    print("QC summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
