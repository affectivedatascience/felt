"""Generate the organized seven-video set for all smoothed FELT CSV files.

Each trial is rendered into a private staging directory, validated, and then
atomically promoted into the shared output tree. Existing non-empty outputs are
skipped by default, so interrupted runs can safely resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.generate_felt_visualization_set import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    generate_visualization_set,
    organized_output_paths,
)
from utils.felt_paths import SMOOTHED_MOTION_DIR, SMOOTHED_VIDEO_DIR  # noqa: E402
from utils.ffmpeg_runtime import (  # noqa: E402
    configure_ffmpeg_dlls,
    resolve_ffmpeg_bin,
    resolve_ffprobe,
)

DEFAULT_OUTPUT_DIR = SMOOTHED_VIDEO_DIR / "felt_visualization_set"
MANIFEST_COLUMNS = (
    "timestamp",
    "status",
    "input_csv",
    "views",
    "frames",
    "elapsed_seconds",
    "error",
)


@dataclass(frozen=True)
class RenderTask:
    input_csv: Path
    output_dir: Path
    views: tuple[str, ...]
    fps: float
    stride: int
    max_frames: int | None
    ffprobe: Path | None


@dataclass(frozen=True)
class RenderResult:
    input_csv: Path
    status: str
    views: tuple[str, ...]
    frames: int | None
    elapsed_seconds: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=SMOOTHED_MOTION_DIR,
        help=f"Smoothed CSV root (default: {SMOOTHED_MOTION_DIR}).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Visualization root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--channel",
        action="append",
        choices=("speech", "song"),
        help="Process only this channel; repeat to select both. Default: both.",
    )
    parser.add_argument(
        "--actor",
        action="append",
        type=int,
        help="Process only this actor number; repeat for multiple actors.",
    )
    parser.add_argument("--limit", type=int, help="Process at most N discovered CSVs.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and atomically replace all seven outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Summarize the work without rendering or writing a manifest.",
    )
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="With --dry-run, list every pending CSV and its missing views.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        help="Directory containing FFmpeg shared DLLs and ffprobe.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Append results here (default: <output_dir>/batch_manifest.csv).",
    )
    return parser.parse_args()


def discover_csvs(
    input_dir: Path,
    channels: list[str] | None,
    actors: list[int] | None,
) -> list[Path]:
    """Discover CSVs in deterministic channel/actor/filename order."""
    selected_channels = channels or ["speech", "song"]
    selected_actors = set(actors) if actors else None
    csv_paths: list[Path] = []
    for channel in selected_channels:
        channel_dir = input_dir / channel
        if not channel_dir.is_dir():
            continue
        for actor_dir in sorted(channel_dir.glob("Actor_[0-9][0-9]")):
            try:
                actor_id = int(actor_dir.name[-2:])
            except ValueError:
                continue
            if selected_actors is not None and actor_id not in selected_actors:
                continue
            csv_paths.extend(sorted(actor_dir.glob("*.csv")))
    return csv_paths


def configure_worker(ffmpeg_bin: str | None) -> None:
    """Configure a worker and leave Ctrl-C handling to the parent process."""
    configure_ffmpeg_dlls(ffmpeg_bin)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def probe_frame_count(video_path: Path, ffprobe: Path | None) -> int:
    """Return the decoded video-frame count using ffprobe or imageio fallback."""
    if ffprobe is not None:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())

    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    try:
        return reader.count_frames()
    finally:
        reader.close()


def missing_views(input_csv: Path, output_dir: Path, overwrite: bool) -> tuple[str, ...]:
    paths = organized_output_paths(input_csv, output_dir)
    if overwrite:
        return DEFAULT_VIEW_NAMES
    return tuple(
        view
        for view in DEFAULT_VIEW_NAMES
        if not paths[view].is_file() or paths[view].stat().st_size == 0
    )


def _safe_remove_stage(stage_dir: Path, partial_root: Path) -> None:
    if not stage_dir.exists():
        return
    resolved_stage = stage_dir.resolve()
    resolved_root = partial_root.resolve()
    if not resolved_stage.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing to remove staging path outside {resolved_root}")
    shutil.rmtree(resolved_stage)


def render_task(task: RenderTask) -> RenderResult:
    """Render, validate, and atomically promote one CSV's missing products."""
    started = time.perf_counter()
    partial_root = task.output_dir / ".partial"
    stage_dir = partial_root / str(os.getpid()) / f"{task.input_csv.stem}-{uuid.uuid4().hex}"
    try:
        stage_paths, frame_count = generate_visualization_set(
            task.input_csv,
            stage_dir,
            fps=task.fps,
            stride=task.stride,
            max_frames=task.max_frames,
            views=task.views,
            print_paths=False,
        )
        final_paths = organized_output_paths(task.input_csv, task.output_dir)
        for view in task.views:
            staged_path = stage_paths[view]
            if not staged_path.is_file() or staged_path.stat().st_size == 0:
                raise RuntimeError(f"Renderer did not create a non-empty file: {staged_path}")
            actual_frames = probe_frame_count(staged_path, task.ffprobe)
            if actual_frames != frame_count:
                raise RuntimeError(
                    f"Frame-count mismatch for {view}: expected {frame_count}, "
                    f"found {actual_frames}"
                )

        for view in task.views:
            destination = final_paths[view]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_paths[view], destination)

        return RenderResult(
            input_csv=task.input_csv,
            status="generated",
            views=task.views,
            frames=frame_count,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception:
        return RenderResult(
            input_csv=task.input_csv,
            status="failed",
            views=task.views,
            frames=None,
            elapsed_seconds=time.perf_counter() - started,
            error=traceback.format_exc(),
        )
    finally:
        _safe_remove_stage(stage_dir, partial_root)


def append_manifest(path: Path, result: RenderResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": result.status,
                "input_csv": str(result.input_csv),
                "views": ";".join(result.views),
                "frames": result.frames if result.frames is not None else "",
                "elapsed_seconds": f"{result.elapsed_seconds:.3f}",
                "error": result.error,
            }
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be at least 1")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.stride < 1:
        raise ValueError("stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if args.actor and any(actor < 1 or actor > 99 for actor in args.actor):
        raise ValueError("actor values must be between 1 and 99")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Smoothed CSV root not found: {input_dir}")

    csv_paths = discover_csvs(input_dir, args.channel, args.actor)
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        print("No matching smoothed CSV files found.")
        return

    tasks: list[RenderTask] = []
    skipped: list[RenderResult] = []
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    ffprobe = resolve_ffprobe(ffmpeg_bin)
    for input_csv in csv_paths:
        views = missing_views(input_csv, output_dir, args.overwrite)
        if views:
            tasks.append(
                RenderTask(
                    input_csv=input_csv,
                    output_dir=output_dir,
                    views=views,
                    fps=args.fps,
                    stride=args.stride,
                    max_frames=args.max_frames,
                    ffprobe=ffprobe,
                )
            )
        else:
            skipped.append(
                RenderResult(input_csv, "skipped", (), None, 0.0)
            )

    print(f"Discovered: {len(csv_paths)} CSVs")
    print(f"Pending:    {len(tasks)} CSVs")
    print(f"Complete:   {len(skipped)} CSVs")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {output_dir}")
    if ffmpeg_bin is not None:
        print(f"FFmpeg DLLs: {ffmpeg_bin}")

    if args.dry_run:
        if args.list_pending:
            for task in tasks:
                print(f"PENDING {task.input_csv} ({', '.join(task.views)})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest or output_dir / "batch_manifest.csv").resolve()
    for result in skipped:
        append_manifest(manifest_path, result)

    failures = 0
    completed = 0
    interrupted = False
    ffmpeg_bin_value = str(ffmpeg_bin) if ffmpeg_bin is not None else None

    def record_result(result: RenderResult) -> None:
        nonlocal completed, failures
        completed += 1
        append_manifest(manifest_path, result)
        if result.status == "failed":
            failures += 1
        print(
            f"[{completed}/{len(tasks)}] {result.status.upper()} "
            f"{result.input_csv.name} ({result.elapsed_seconds:.1f}s)"
        )
        if result.error:
            print(result.error, file=sys.stderr)

    if args.workers == 1:
        configure_ffmpeg_dlls(ffmpeg_bin_value)
        active_task: RenderTask | None = None
        active_started = 0.0
        try:
            for active_task in tasks:
                active_started = time.perf_counter()
                record_result(render_task(active_task))
                active_task = None
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl-C received; cleaned the active staging directory.")
            if active_task is not None:
                append_manifest(
                    manifest_path,
                    RenderResult(
                        active_task.input_csv,
                        "interrupted",
                        active_task.views,
                        None,
                        time.perf_counter() - active_started,
                    ),
                )
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=configure_worker,
            initargs=(ffmpeg_bin_value,),
        )
        futures = {executor.submit(render_task, task): task for task in tasks}
        handled = set()
        try:
            for future in as_completed(futures):
                record_result(future.result())
                handled.add(future)
        except KeyboardInterrupt:
            interrupted = True
            cancelled = sum(future.cancel() for future in futures if future not in handled)
            print(
                "\nCtrl-C received; cancelling queued trials and allowing "
                f"active trials to finish ({cancelled} cancelled)."
            )
            executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future in handled or future.cancelled() or not future.done():
                    continue
                record_result(future.result())
                handled.add(future)
        else:
            executor.shutdown(wait=True)

    print(f"Finished: {completed - failures} generated, {len(skipped)} skipped, {failures} failed")
    print(f"Manifest: {manifest_path}")
    if interrupted:
        print("Interrupted safely. Run the same command again to resume.")
        raise SystemExit(130)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
