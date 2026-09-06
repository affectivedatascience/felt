"""Validate the restored RAVDESS MP4 inventory used by FELT v2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.felt_paths import (  # noqa: E402
    FULL_AV_MODALITY,
    INPUT_DIR,
    OUTPUT_DIR,
    VIDEO_ONLY_MODALITY,
    actor_name,
    parse_ravdess_stem,
    vocal_channel_folder,
)

EXPECTED_COUNTS = {
    "all_mp4": 4904,
    "full_av": 2452,
    "video_only": 2452,
    "full_av_speech": 1440,
    "full_av_song": 1012,
    "video_only_speech": 1440,
    "video_only_song": 1012,
}
MANIFEST_FIELDS = (
    "relative_path",
    "size_bytes",
    "modality",
    "vocal_channel",
    "actor",
    "selected_for_felt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_DIR)
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_DIR / "qc" / "ravdess_input_summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_DIR / "qc" / "ravdess_input_manifest.csv",
    )
    return parser.parse_args()


def audit_inputs(input_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return a file manifest and exact FELT/RAVDESS inventory checks."""
    input_root = input_root.resolve()
    paths = sorted(input_root.glob("Actor_*/*.mp4"))
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    errors: list[str] = []
    stems: Counter[str] = Counter()

    for path in paths:
        relative = path.relative_to(input_root).as_posix()
        try:
            code = parse_ravdess_stem(path)
            channel = vocal_channel_folder(code)
        except ValueError as exc:
            errors.append(f"{relative}: {exc}")
            continue
        expected_actor = actor_name(code.actor)
        if path.parent.name != expected_actor:
            errors.append(
                f"{relative}: filename actor {expected_actor} does not match folder"
            )
        if code.modality not in {FULL_AV_MODALITY, VIDEO_ONLY_MODALITY}:
            errors.append(f"{relative}: unexpected MP4 modality {code.modality}")
        if code.actor == 18 and channel == "song":
            errors.append(f"{relative}: RAVDESS has no Actor_18 song corpus")
        if path.stat().st_size == 0:
            errors.append(f"{relative}: file is empty")

        stems[path.stem] += 1
        counts["all_mp4"] += 1
        modality_name = (
            "full_av" if code.modality == FULL_AV_MODALITY else "video_only"
        )
        counts[modality_name] += 1
        counts[f"{modality_name}_{channel}"] += 1
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "modality": code.modality,
                "vocal_channel": channel,
                "actor": code.actor,
                "selected_for_felt": code.modality == FULL_AV_MODALITY,
            }
        )

    duplicate_stems = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicate_stems:
        errors.append(f"duplicate filename stems: {duplicate_stems[:10]}")
    count_mismatches = {
        key: {"observed": counts[key], "expected": expected}
        for key, expected in EXPECTED_COUNTS.items()
        if counts[key] != expected
    }
    errors.extend(
        f"{key}: observed {values['observed']}, expected {values['expected']}"
        for key, values in count_mismatches.items()
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "input_root": str(input_root),
        "counts": {key: counts[key] for key in EXPECTED_COUNTS},
        "expected_counts": EXPECTED_COUNTS,
        "error_count": len(errors),
        "errors": errors,
        "passed": not errors,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"RAVDESS input root not found: {input_root}")
    rows, summary = audit_inputs(input_root)
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {manifest_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
