"""
Run Py-Feat facial tracking on RAVDESS audiovisual videos.

This script generates the raw FELT facial-tracking CSV files from the RAVDESS
full-audiovisual speech and song videos. It iterates over actor folders, selects
valid full-AV speech/song files using the RAVDESS filename convention, runs
Py-Feat on each selected video, and writes one raw tracking CSV per source file.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   ├── 01_input/
│   │   ├── Actor_01/
│   │   ├── Actor_02/
│   │   └── ...
│   └── 02_output/
│       ├── 01_raw_motion/
│       │   ├── speech/
│       │   │   ├── Actor_01/
│       │   │   └── ...
│       │   └── song/
│       │       ├── Actor_01/
│       │       └── ...
│       └── logs/
└── 02_code/
    └── src/
        ├── 1_extract_raw_tracking.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py

RAVDESS filename convention
---------------------------
Each trial filename contains seven hyphen-separated fields:

    modality-vocal_channel-emotion-intensity-statement-repetition-actor

Relevant fields:
    modality:      01 = full-AV, 02 = video-only, 03 = audio-only
    vocal channel: 01 = speech, 02 = song

This script processes full-AV speech/song files only. Video-only and audio-only
files are skipped. Actor 18 has no song recordings in RAVDESS.
"""

# =============================================================================
# Original Py-Feat environment note
# =============================================================================
#
# FELT v1.0.0 was generated with Py-Feat 0.6.2 under Python 3.9.
# The original working environment used PyTorch 2.2.0 / torchvision 0.17.0 /
# torchaudio 2.2.0 with CUDA 12.1 wheels.
#
# The maintained rerun environment uses Py-Feat 2.0.3 with Detectorv2 and
# Python 3.11+. Detectorv2 changes the scientific model and expands the raw
# output schema while retaining the 68-point landmarks and 20 AU columns used
# by the existing FELT smoothing and visualization stages.
# The local Py-Feat 0.6.2 identity patch is no longer required.

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Make the local utils/ package importable when this script is run directly.
# =============================================================================

CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")


# =============================================================================
# Py-Feat import configuration
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning,
)

# Py-Feat device. Use "cuda" on a CUDA-enabled GPU; otherwise use "cpu".
DEVICE = "cuda"

# Limit native-library threading only for CPU execution.
# This avoids XGBoost/OpenMP stalls observed with Py-Feat 0.6.2 on macOS CPU.
# IMPORTANT: these must be set before importing Py-Feat / XGBoost / Torch.
if DEVICE == "cpu":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from utils.felt_paths import (  # noqa: E402
    INPUT_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    actor_name,
    configure_logging,
    parse_ravdess_stem,
    should_process_full_av_speech_song,
    vocal_channel_folder,
)


# =============================================================================
# User-editable configuration
# =============================================================================

# Input RAVDESS videos.
RAVDESS_INPUT_DIR = INPUT_DIR

# Logs.
LOG_FILE = LOG_DIR / "1_extract_raw_tracking.log"

# Actor range, inclusive.
START_ACTOR = 1
END_ACTOR = 24

# Existing output CSV files are skipped so interrupted runs can resume.
SKIP_EXISTING = True


# =============================================================================
# Py-Feat detector configuration
# =============================================================================

IDENTITY_MODEL = "arcface"

# Py-Feat video-detection parameters used for FELT.
# Py-Feat output size is specified as (height, width).
OUTPUT_SIZE = (720, 1280)
BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = False
FACE_DETECTION_THRESHOLD = 0.83
FACE_IDENTITY_THRESHOLD = 0.8


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class DetectionTask:
    """A source video and its corresponding output CSV path."""

    video_path: Path
    csv_path: Path


@dataclass(frozen=True)
class RunConfig:
    """Runtime controls for a raw tracking extraction run."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    skip_existing: bool
    face_detection_threshold: float
    face_identity_threshold: float
    log_file: Path


@dataclass(frozen=True)
class TaskResult:
    """A per-file extraction result row."""

    worker_id: int
    task_index: int
    task_count: int
    status: str
    video_path: Path
    csv_path: Path
    elapsed_seconds: float
    batch_size: int
    error: str = ""


@dataclass(frozen=True)
class TaskFilters:
    """Selection filters for development subsets and full runs."""

    start_actor: int
    end_actor: int
    actors: tuple[int, ...]
    vocal_channel: str
    stems: tuple[str, ...]
    file_list: Path | None
    limit: int | None
    first: bool


# =============================================================================
# Pipeline functions
# =============================================================================


def create_detector() -> Any:
    """Create the Py-Feat detector used for FELT tracking."""
    from feat import Detectorv2

    logging.info("Loading Py-Feat Detectorv2.")

    detector = Detectorv2(
        identity_model=IDENTITY_MODEL,
        device=DEVICE,
    )

    logging.info("Py-Feat Detectorv2 loaded.")
    return detector


def load_requested_stems(file_list: Path | None) -> set[str]:
    """Load requested video stems from a plain-text file list."""
    if file_list is None:
        return set()

    stems: set[str] = set()
    with file_list.open("r", encoding="utf-8-sig") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            stems.add(Path(value).stem)
    return stems


def include_task(video_path: Path, filters: TaskFilters, file_list_stems: set[str]) -> bool:
    """Return True when a source video matches the CLI subset filters."""
    code = parse_ravdess_stem(video_path)
    channel_folder = vocal_channel_folder(code)

    if filters.actors and code.actor not in filters.actors:
        return False

    if filters.vocal_channel != "all" and channel_folder != filters.vocal_channel:
        return False

    requested_stems = set(filters.stems) | file_list_stems
    if requested_stems and video_path.stem not in requested_stems:
        return False

    return True


def build_tasks(filters: TaskFilters | None = None) -> list[DetectionTask]:
    """Build detection tasks for valid RAVDESS full-AV speech/song videos."""
    if filters is None:
        filters = TaskFilters(
            start_actor=START_ACTOR,
            end_actor=END_ACTOR,
            actors=(),
            vocal_channel="all",
            stems=(),
            file_list=None,
            limit=None,
            first=False,
        )

    tasks: list[DetectionTask] = []
    file_list_stems = load_requested_stems(filters.file_list)

    for actor_id in range(filters.start_actor, filters.end_actor + 1):
        current_actor_name = actor_name(actor_id)
        actor_video_dir = RAVDESS_INPUT_DIR / current_actor_name

        if not actor_video_dir.exists():
            logging.warning(
                "Actor input directory not found; skipping: %s", actor_video_dir
            )
            continue

        for video_path in sorted(actor_video_dir.glob("*.mp4")):
            try:
                code = parse_ravdess_stem(video_path)

                if not should_process_full_av_speech_song(code):
                    logging.debug("Skipping non-target file: %s", video_path)
                    continue

                if not include_task(video_path, filters, file_list_stems):
                    logging.debug("Skipping file outside selected subset: %s", video_path)
                    continue

                channel_folder = vocal_channel_folder(code)

            except ValueError as exc:
                logging.warning(
                    "Skipping file with invalid RAVDESS filename: %s; %s",
                    video_path,
                    exc,
                )
                continue

            actor_csv_dir = RAW_MOTION_DIR / channel_folder / current_actor_name
            actor_csv_dir.mkdir(parents=True, exist_ok=True)

            csv_path = actor_csv_dir / f"{video_path.stem}.csv"
            tasks.append(DetectionTask(video_path=video_path, csv_path=csv_path))

            if filters.first:
                logging.info("Single-file test mode enabled.")
                logging.info("Selected test video: %s", video_path)
                logging.info("Selected output CSV: %s", csv_path)
                return tasks

            if filters.limit is not None and len(tasks) >= filters.limit:
                logging.info("Task limit reached: %d", filters.limit)
                return tasks

    logging.info("Prepared %d detection tasks.", len(tasks))
    return tasks


def save_prediction_csv(feat_prediction, csv_path: Path) -> None:
    """Save a Py-Feat prediction dataframe to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the previous script's default pandas behavior.
    # This writes the DataFrame index unless Py-Feat suppresses it internally.
    feat_prediction.to_csv(csv_path)

    logging.info("Output saved to %s", csv_path)
    print(f"Output saved to: {csv_path}")


def run_detection(
    detector: Any,
    task: DetectionTask,
    config: RunConfig,
    *,
    worker_id: int = 0,
    task_index: int = 1,
    task_count: int = 1,
) -> TaskResult:
    """Run Py-Feat detection for one video file and save the CSV output."""
    started_at = time.perf_counter()

    if config.skip_existing and task.csv_path.exists():
        logging.info("Output already exists; skipping: %s", task.csv_path)
        print(f"Output already exists; skipping: {task.csv_path}")
        return TaskResult(
            worker_id=worker_id,
            task_index=task_index,
            task_count=task_count,
            status="skipped_existing",
            video_path=task.video_path,
            csv_path=task.csv_path,
            elapsed_seconds=time.perf_counter() - started_at,
            batch_size=config.batch_size,
        )

    try:
        logging.info("Running detection for file: %s", task.video_path)
        print(f"Running detection for file: {task.video_path}")

        video_prediction = detector.detect(
            str(task.video_path),
            data_type="video",
            skip_frames=None,
            output_size=OUTPUT_SIZE,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            face_detection_threshold=config.face_detection_threshold,
            face_identity_threshold=config.face_identity_threshold,
        )

        save_prediction_csv(video_prediction, task.csv_path)
        return TaskResult(
            worker_id=worker_id,
            task_index=task_index,
            task_count=task_count,
            status="ok",
            video_path=task.video_path,
            csv_path=task.csv_path,
            elapsed_seconds=time.perf_counter() - started_at,
            batch_size=config.batch_size,
        )

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        raise

    except Exception as exc:
        logging.exception("Error processing file: %s", task.video_path)
        print(f"Error processing file. See log: {config.log_file}")
        return TaskResult(
            worker_id=worker_id,
            task_index=task_index,
            task_count=task_count,
            status="error",
            video_path=task.video_path,
            csv_path=task.csv_path,
            elapsed_seconds=time.perf_counter() - started_at,
            batch_size=config.batch_size,
            error=repr(exc),
        )


def split_evenly(items: list[DetectionTask], workers: int) -> list[list[DetectionTask]]:
    """Split tasks across worker shards while preserving deterministic order."""
    shards = [[] for _ in range(workers)]
    for index, item in enumerate(items):
        shards[index % workers].append(item)
    return [shard for shard in shards if shard]


def run_task_shard(
    worker_id: int,
    tasks: list[DetectionTask],
    config: RunConfig,
) -> list[TaskResult]:
    """Run one process-local detector over a shard of files."""
    worker_log = config.log_file.with_name(
        f"{config.log_file.stem}_worker_{worker_id}{config.log_file.suffix}"
    )
    configure_logging(worker_log)
    logging.info("Worker %d starting with %d task(s).", worker_id, len(tasks))

    detector = create_detector()
    results: list[TaskResult] = []
    for index, task in enumerate(tasks, start=1):
        logging.info("Worker %d processing %d/%d: %s", worker_id, index, len(tasks), task.video_path)
        print(f"[worker {worker_id} {index}/{len(tasks)}] {task.video_path}")
        result = run_detection(
            detector,
            task,
            config,
            worker_id=worker_id,
            task_index=index,
            task_count=len(tasks),
        )
        results.append(result)
    return results


def write_run_report(report_path: Path, rows: list[TaskResult]) -> None:
    """Write a CSV manifest for dry-run or extraction results."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "worker_id",
        "task_index",
        "task_count",
        "status",
        "batch_size",
        "elapsed_seconds",
        "video_path",
        "csv_path",
        "error",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "worker_id": row.worker_id,
                    "task_index": row.task_index,
                    "task_count": row.task_count,
                    "status": row.status,
                    "batch_size": row.batch_size,
                    "elapsed_seconds": f"{row.elapsed_seconds:.3f}",
                    "video_path": row.video_path,
                    "csv_path": row.csv_path,
                    "error": row.error,
                }
            )


def write_task_report(report_path: Path, tasks: list[DetectionTask], batch_size: int) -> None:
    """Write a dry-run task manifest."""
    rows = [
        TaskResult(
            worker_id=0,
            task_index=index,
            task_count=len(tasks),
            status="planned",
            video_path=task.video_path,
            csv_path=task.csv_path,
            elapsed_seconds=0.0,
            batch_size=batch_size,
        )
        for index, task in enumerate(tasks, start=1)
    ]
    write_run_report(report_path, rows)


def parse_args() -> argparse.Namespace:
    """Parse raw extraction command-line options."""
    parser = argparse.ArgumentParser(
        description="Run Py-Feat raw tracking on RAVDESS full-AV speech/song videos."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of file-level worker processes.",
    )
    parser.add_argument(
        "--pyfeat-num-workers",
        type=int,
        default=NUM_WORKERS,
        help="num_workers passed to Py-Feat detect_video within each process.",
    )
    parser.add_argument("--start-actor", type=int, default=START_ACTOR)
    parser.add_argument("--end-actor", type=int, default=END_ACTOR)
    parser.add_argument(
        "--actor",
        type=int,
        action="append",
        default=[],
        help="Restrict to one actor. Repeat for multiple actors.",
    )
    parser.add_argument(
        "--vocal-channel",
        choices=("all", "speech", "song"),
        default="all",
    )
    parser.add_argument(
        "--stem",
        action="append",
        default=[],
        help="Restrict to one RAVDESS filename stem. Repeat for multiple files.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Plain-text list of filename stems or paths to process.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N selected tasks.")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N selected tasks before dry-run or processing.",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Process only the first selected task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSVs instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report tasks without loading Py-Feat or processing videos.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=LOG_DIR / "1_extract_raw_tracking_report.csv",
        help="CSV task/result report path.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_FILE,
        help="Main log file path.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    """Validate CLI arguments and return a shell-style status code."""
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2
    if args.pyfeat_num_workers < 0:
        print("--pyfeat-num-workers must be >= 0", file=sys.stderr)
        return 2
    if args.start_actor < 1 or args.end_actor > 24 or args.start_actor > args.end_actor:
        print("--start-actor/--end-actor must describe an actor range within 1-24", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    if args.skip_first < 0:
        print("--skip-first must be >= 0", file=sys.stderr)
        return 2
    if args.file_list is not None and not args.file_list.exists():
        print(f"--file-list does not exist: {args.file_list}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    """Run raw facial-tracking extraction."""
    args = parse_args()
    validation_status = validate_args(args)
    if validation_status:
        return validation_status

    configure_logging(args.log_file)

    print(f"Writing log to: {args.log_file}")

    logging.info("Starting Py-Feat RAVDESS raw tracking extraction.")
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("RAVDESS_INPUT_DIR: %s", RAVDESS_INPUT_DIR)
    logging.info("RAW_MOTION_DIR: %s", RAW_MOTION_DIR)
    logging.info("LOG_FILE: %s", args.log_file)
    logging.info("REPORT: %s", args.report)
    logging.info("Actors: %02d-%02d", args.start_actor, args.end_actor)
    logging.info("Selected actors: %s", args.actor or "all")
    logging.info("Selected vocal channel: %s", args.vocal_channel)
    logging.info("DEVICE: %s", DEVICE)
    logging.info("Batch size: %d", args.batch_size)
    logging.info("File-level workers: %d", args.workers)
    logging.info("Py-Feat num_workers: %d", args.pyfeat_num_workers)
    logging.info("Skip existing: %s", not args.overwrite)

    filters = TaskFilters(
        start_actor=args.start_actor,
        end_actor=args.end_actor,
        actors=tuple(sorted(set(args.actor))),
        vocal_channel=args.vocal_channel,
        stems=tuple(args.stem),
        file_list=args.file_list,
        limit=args.limit,
        first=args.first,
    )
    tasks = build_tasks(filters)
    if args.skip_first:
        tasks = tasks[args.skip_first :]
        logging.info("Skipped first %d selected task(s).", args.skip_first)

    if not tasks:
        logging.warning(
            "No detection tasks were prepared. Check RAVDESS_INPUT_DIR and filenames."
        )
        print(
            "No detection tasks were prepared. Check RAVDESS_INPUT_DIR and filenames."
        )
        return 1

    if args.dry_run:
        write_task_report(args.report, tasks, args.batch_size)
        logging.info("Dry run complete. Prepared %d task(s).", len(tasks))
        print(f"Dry run complete. Prepared {len(tasks)} task(s).")
        print(f"Task report: {args.report}")
        return 0

    logging.info("Starting detection loop for %d task(s).", len(tasks))
    print(f"Starting detection loop for {len(tasks)} task(s).")

    config = RunConfig(
        batch_size=args.batch_size,
        num_workers=args.pyfeat_num_workers,
        pin_memory=PIN_MEMORY,
        skip_existing=not args.overwrite,
        face_detection_threshold=FACE_DETECTION_THRESHOLD,
        face_identity_threshold=FACE_IDENTITY_THRESHOLD,
        log_file=args.log_file,
    )

    results: list[TaskResult] = []
    if args.workers == 1:
        detector = create_detector()
        for index, task in enumerate(tasks, start=1):
            logging.info("Processing task %d/%d: %s", index, len(tasks), task.video_path)
            print(f"[{index}/{len(tasks)}] {task.video_path}")
            results.append(
                run_detection(
                    detector,
                    task,
                    config,
                    worker_id=1,
                    task_index=index,
                    task_count=len(tasks),
                )
            )
            write_run_report(args.report, results)
    else:
        shards = split_evenly(tasks, args.workers)
        with ProcessPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(run_task_shard, worker_id, shard, config)
                for worker_id, shard in enumerate(shards, start=1)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
                results.sort(key=lambda row: str(row.csv_path))
                write_run_report(args.report, results)
                print(f"Completed {len(results)}/{len(tasks)} task result(s).")

    results.sort(key=lambda row: str(row.csv_path))
    write_run_report(args.report, results)

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    logging.info("Raw tracking extraction complete. Status counts: %s", status_counts)
    print("Raw tracking extraction complete.")
    print(f"Result report: {args.report}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    return 1 if any(result.status == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

