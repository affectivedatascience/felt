"""Prepare historical Py-Feat 0.6.2 multiple-face review artifacts.

The Phase 3 raw tracking rerun produced complete unique-frame coverage, but a
small set of files contains duplicate rows for the same source frame. RAVDESS is
expected to contain one actor face per frame, so these duplicate rows need
reviewer adjudication before downstream analysis collapses to one row per frame.

This tool creates:

- ``multi_face_candidates_long.csv``: one row per duplicate-frame candidate.
- ``multi_face_review_manifest.csv``: one row per affected file, with suggested
  keep candidate and blank reviewer decision fields.
- optional overlay MP4s with source frame numbers and candidate labels.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")

from utils.felt_paths import (  # noqa: E402
    OUTPUT_DIR,
    PROJECT_ROOT,
    configure_logging,
    parse_ravdess_stem,
)


QC_DIR = OUTPUT_DIR / "qc"
DEFAULT_FRAME_SUMMARY = QC_DIR / "frame_integrity_file_summary.csv"
DEFAULT_MANIFEST = QC_DIR / "multi_face_review_manifest.csv"
DEFAULT_CANDIDATES = QC_DIR / "multi_face_candidates_long.csv"
DEFAULT_OVERLAY_DIR = QC_DIR / "multi_face_overlays"
DEFAULT_LOG = OUTPUT_DIR / "logs" / "prepare_multi_face_review.log"

PERSON_COLORS = {
    "Person_0": (35, 170, 70),
    "Person_1": (220, 70, 65),
    "Person_2": (55, 120, 220),
    "Person_3": (220, 150, 35),
}


@dataclass(frozen=True)
class ReviewFile:
    """One file that needs same-frame duplicate-face review."""

    vocal_channel: str
    relative_csv_path: str
    raw_csv_path: Path
    source_video_path: Path
    duplicate_frame_rows: int

    @property
    def stem(self) -> str:
        return self.raw_csv_path.stem

    @property
    def actor_name(self) -> str:
        return self.raw_csv_path.parent.name


def project_relative(path: Path) -> str:
    """Return a project-relative path if possible, otherwise the absolute path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def infer_vocal_channel(row: pd.Series) -> str:
    """Infer speech/song from QC source report or filename code."""
    source_report = str(row.get("source_report", ""))
    if "speech" in source_report:
        return "speech"
    if "song" in source_report:
        return "song"

    csv_path = Path(str(row["csv_path"]))
    try:
        code = parse_ravdess_stem(csv_path.stem)
    except ValueError:
        return ""
    return "speech" if code.vocal_channel == "01" else "song"


def load_review_files(frame_summary_path: Path) -> list[ReviewFile]:
    """Load the file-level duplicate-frame worklist from frame-integrity QC."""
    summary = pd.read_csv(frame_summary_path)
    duplicate_files = summary[
        summary["csv_duplicate_frame_rows"].fillna(0).astype(int) > 0
    ]

    files: list[ReviewFile] = []
    for _, row in duplicate_files.sort_values("relative_csv_path").iterrows():
        files.append(
            ReviewFile(
                vocal_channel=infer_vocal_channel(row),
                relative_csv_path=str(row["relative_csv_path"]),
                raw_csv_path=Path(str(row["csv_path"])),
                source_video_path=Path(str(row["video_path"])),
                duplicate_frame_rows=int(row["csv_duplicate_frame_rows"]),
            )
        )
    return files


def source_frame_column(df: pd.DataFrame) -> str:
    """Pick the source-frame column from Py-Feat CSV output."""
    if "frame.1" in df.columns:
        return "frame.1"
    if "frame" in df.columns:
        return "frame"
    raise ValueError("CSV has no frame column")


def numeric_or_blank(value: Any) -> Any:
    """Convert pandas/numpy scalar values to plain values, preserving blanks."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and math.isfinite(value):
        return value
    return value


def get_candidate_person(row: pd.Series, rank_in_frame: int) -> str:
    """Return the detected person label or a stable fallback candidate label."""
    identity = row.get("Identity", "")
    if isinstance(identity, str) and identity.startswith("Person_"):
        return identity
    return f"Candidate_{rank_in_frame}"


def candidate_rows_for_file(review_file: ReviewFile) -> list[dict[str, Any]]:
    """Extract one row per duplicate-frame candidate for an affected file."""
    df = pd.read_csv(review_file.raw_csv_path)
    frame_col = source_frame_column(df)
    frame_counts = df[frame_col].value_counts(dropna=False)
    duplicate_frames = sorted(
        int(frame) for frame, count in frame_counts.items() if count > 1
    )

    rows: list[dict[str, Any]] = []
    for frame in duplicate_frames:
        frame_df = df[df[frame_col] == frame].copy()
        frame_df = frame_df.sort_values(
            "FaceScore", ascending=False, na_position="last"
        )
        best_score = pd.to_numeric(frame_df["FaceScore"], errors="coerce").max()

        scores = (
            pd.to_numeric(frame_df["FaceScore"], errors="coerce")
            .dropna()
            .sort_values(ascending=False)
        )
        score_margin = ""
        if len(scores) >= 2:
            score_margin = float(scores.iloc[0] - scores.iloc[1])

        for rank, (row_index, row) in enumerate(frame_df.iterrows()):
            face_score = pd.to_numeric(
                pd.Series([row.get("FaceScore")]), errors="coerce"
            ).iloc[0]
            candidate_person = get_candidate_person(row, rank)
            is_score_recommended = bool(
                pd.notna(face_score)
                and pd.notna(best_score)
                and face_score == best_score
            )

            face_rect_x = numeric_or_blank(row.get("FaceRectX", ""))
            face_rect_y = numeric_or_blank(row.get("FaceRectY", ""))
            face_rect_width = numeric_or_blank(row.get("FaceRectWidth", ""))
            face_rect_height = numeric_or_blank(row.get("FaceRectHeight", ""))
            face_area = ""
            if face_rect_width != "" and face_rect_height != "":
                face_area = float(face_rect_width) * float(face_rect_height)

            rows.append(
                {
                    "vocal_channel": review_file.vocal_channel,
                    "actor": review_file.actor_name,
                    "stem": review_file.stem,
                    "relative_csv_path": review_file.relative_csv_path,
                    "raw_csv_path": str(review_file.raw_csv_path),
                    "source_video_path": str(review_file.source_video_path),
                    "source_frame": frame,
                    "csv_row_index": int(row_index),
                    "candidate_rank_by_facescore": rank,
                    "candidate_person": candidate_person,
                    "is_facescore_recommended": is_score_recommended,
                    "facescore": numeric_or_blank(face_score),
                    "facescore_margin_to_next": score_margin if rank == 0 else "",
                    "face_rect_x": face_rect_x,
                    "face_rect_y": face_rect_y,
                    "face_rect_width": face_rect_width,
                    "face_rect_height": face_rect_height,
                    "face_area": face_area,
                }
            )

    return rows


def summarize_recommendation(file_candidates: pd.DataFrame) -> dict[str, str]:
    """Create file-level recommendation fields from candidate rows."""
    affected_frames = sorted(
        file_candidates["source_frame"].astype(int).unique().tolist()
    )
    candidate_persons = sorted(
        file_candidates["candidate_person"].astype(str).unique().tolist()
    )

    top_rows = file_candidates[file_candidates["is_facescore_recommended"].astype(bool)]
    top_by_frame = (
        top_rows.groupby("source_frame")["candidate_person"].first().to_dict()
    )
    top_counts = Counter(str(person) for person in top_by_frame.values())

    recommended_person = ""
    recommendation_basis = ""
    if top_counts:
        person, count = top_counts.most_common(1)[0]
        if len(top_counts) == 1:
            recommended_person = person
            recommendation_basis = (
                f"highest FaceScore on {count}/{len(affected_frames)} affected frames"
            )
        else:
            recommended_person = "MIXED"
            parts = ", ".join(f"{p}: {c}" for p, c in top_counts.most_common())
            recommendation_basis = f"highest FaceScore differs by frame ({parts})"

    return {
        "affected_unique_frames": ";".join(str(frame) for frame in affected_frames),
        "candidate_persons": ";".join(candidate_persons),
        "recommended_person": recommended_person,
        "recommendation_basis": recommendation_basis,
    }


def load_existing_manifest_decisions(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Preserve reviewer fields from an existing manifest when regenerating."""
    if not manifest_path.exists():
        return {}

    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    decisions: dict[str, dict[str, str]] = {}
    for _, row in manifest.iterrows():
        key = str(row.get("relative_csv_path", ""))
        decisions[key] = {
            "reviewer_keep_person": str(row.get("reviewer_keep_person", "")),
            "reviewer_notes": str(row.get("reviewer_notes", "")),
        }
    return decisions


def write_candidate_artifacts(
    review_files: list[ReviewFile],
    candidates_path: Path,
    manifest_path: Path,
    overlay_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write candidate-long and file-level manifest CSVs."""
    all_candidate_rows: list[dict[str, Any]] = []
    for review_file in review_files:
        all_candidate_rows.extend(candidate_rows_for_file(review_file))

    candidates = pd.DataFrame(all_candidate_rows)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_path, index=False)

    existing_decisions = load_existing_manifest_decisions(manifest_path)
    manifest_rows: list[dict[str, Any]] = []
    for review_file in review_files:
        file_candidates = candidates[
            candidates["relative_csv_path"] == review_file.relative_csv_path
        ]
        summary = summarize_recommendation(file_candidates)
        overlay_path = (
            overlay_dir
            / review_file.vocal_channel
            / review_file.actor_name
            / f"{review_file.stem}.mp4"
        )
        decisions = existing_decisions.get(review_file.relative_csv_path, {})

        manifest_rows.append(
            {
                "vocal_channel": review_file.vocal_channel,
                "relative_csv_path": review_file.relative_csv_path,
                "raw_csv_path": str(review_file.raw_csv_path),
                "source_video_path": str(review_file.source_video_path),
                "overlay_video_path": str(overlay_path),
                "duplicate_frame_rows": review_file.duplicate_frame_rows,
                "affected_unique_frames": summary["affected_unique_frames"],
                "candidate_persons": summary["candidate_persons"],
                "recommended_person": summary["recommended_person"],
                "recommendation_basis": summary["recommendation_basis"],
                "reviewer_keep_person": decisions.get("reviewer_keep_person", ""),
                "reviewer_notes": decisions.get("reviewer_notes", ""),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    return candidates, manifest


def frame_to_detections(
    candidates: pd.DataFrame, raw_csv_path: Path
) -> dict[int, list[dict[str, Any]]]:
    """Map source frame number to all detection rows for overlay drawing."""
    df = pd.read_csv(raw_csv_path)
    frame_col = source_frame_column(df)

    affected_frames = set(candidates["source_frame"].astype(int).tolist())
    detections: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in df.iterrows():
        frame_value = row.get(frame_col)
        if pd.isna(frame_value):
            continue
        frame = int(frame_value)

        candidate_match = candidates[
            (candidates["source_frame"].astype(int) == frame)
            & (candidates["csv_row_index"].astype(int) == int(row_index))
        ]

        if frame in affected_frames or not candidate_match.empty:
            candidate_person = (
                str(candidate_match.iloc[0]["candidate_person"])
                if not candidate_match.empty
                else get_candidate_person(row, 0)
            )
            is_recommended = (
                bool(candidate_match.iloc[0]["is_facescore_recommended"])
                if not candidate_match.empty
                else False
            )
            detections[frame].append(
                {
                    "candidate_person": candidate_person,
                    "facescore": row.get("FaceScore", ""),
                    "is_recommended": is_recommended,
                    "x": row.get("FaceRectX", ""),
                    "y": row.get("FaceRectY", ""),
                    "w": row.get("FaceRectWidth", ""),
                    "h": row.get("FaceRectHeight", ""),
                    "landmarks_x": [row.get(f"x_{i}", "") for i in range(68)],
                    "landmarks_y": [row.get(f"y_{i}", "") for i in range(68)],
                }
            )

    return detections


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw readable label text with a solid background."""
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 4
    bg = (0, 0, 0)
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg
    )
    draw.text(xy, text, fill=color, font=font)


def draw_detection(draw: ImageDraw.ImageDraw, detection: dict[str, Any]) -> None:
    """Draw one candidate face box, label, and landmarks."""
    person = str(detection["candidate_person"])
    color = PERSON_COLORS.get(person, (230, 230, 80))
    width = 5 if detection["is_recommended"] else 3

    try:
        x = float(detection["x"])
        y = float(detection["y"])
        w = float(detection["w"])
        h = float(detection["h"])
    except (TypeError, ValueError):
        return

    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)

    score = detection["facescore"]
    try:
        score_text = f"{float(score):.3f}"
    except (TypeError, ValueError):
        score_text = str(score)

    marker = " suggested" if detection["is_recommended"] else ""
    draw_label(
        draw,
        (int(x), max(0, int(y) - 18)),
        f"{person} FaceScore={score_text}{marker}",
        color,
    )

    for lx, ly in zip(detection["landmarks_x"], detection["landmarks_y"]):
        try:
            px = float(lx)
            py = float(ly)
        except (TypeError, ValueError):
            continue
        r = 2
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)


def render_overlay_video(
    review_file: ReviewFile,
    candidates: pd.DataFrame,
    overlay_path: Path,
    overwrite: bool,
) -> None:
    """Render one full source video with duplicate-face review annotations."""
    if overlay_path.exists() and not overwrite:
        logging.info("Overlay exists, skipping: %s", overlay_path)
        return

    file_candidates = candidates[
        candidates["relative_csv_path"] == review_file.relative_csv_path
    ]
    affected_frames = set(file_candidates["source_frame"].astype(int).tolist())
    detections_by_frame = frame_to_detections(file_candidates, review_file.raw_csv_path)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(review_file.source_video_path)
    metadata = reader.get_meta_data()
    fps = metadata.get("fps", 30)

    writer = imageio.get_writer(
        overlay_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=None,
    )

    try:
        for frame_index, frame in enumerate(reader):
            image = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(image)

            is_affected = frame_index in affected_frames
            header = f"source frame {frame_index}"
            if is_affected:
                header += " | MULTI-FACE REVIEW"
            draw_label(
                draw,
                (16, 16),
                header,
                (255, 255, 255) if not is_affected else (255, 230, 80),
            )

            for detection in detections_by_frame.get(frame_index, []):
                draw_detection(draw, detection)

            writer.append_data(np.asarray(image))

    finally:
        writer.close()
        reader.close()

    logging.info("Wrote overlay: %s", overlay_path)


def render_overlays(
    review_files: list[ReviewFile],
    candidates: pd.DataFrame,
    overlay_dir: Path,
    overwrite: bool,
    limit: int | None,
) -> None:
    """Render review overlays for affected files."""
    files = review_files[:limit] if limit is not None else review_files
    for index, review_file in enumerate(files, start=1):
        overlay_path = (
            overlay_dir
            / review_file.vocal_channel
            / review_file.actor_name
            / f"{review_file.stem}.mp4"
        )
        logging.info(
            "Rendering overlay %d/%d: %s",
            index,
            len(files),
            review_file.relative_csv_path,
        )
        render_overlay_video(review_file, candidates, overlay_path, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-summary", type=Path, default=DEFAULT_FRAME_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--overlays", action="store_true", help="Render review overlay videos."
    )
    parser.add_argument("--overwrite-overlays", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Limit overlay rendering to the first N files."
    )
    return parser.parse_args()


def main() -> None:
    """Prepare candidate and optional video-review artifacts."""
    args = parse_args()
    configure_logging(args.log_file)

    review_files = load_review_files(args.frame_summary)
    logging.info("Loaded %d multiple-face review files.", len(review_files))

    candidates, manifest = write_candidate_artifacts(
        review_files=review_files,
        candidates_path=args.candidates,
        manifest_path=args.manifest,
        overlay_dir=args.overlay_dir,
    )

    logging.info("Wrote candidate rows: %s (%d rows)", args.candidates, len(candidates))
    logging.info("Wrote review manifest: %s (%d rows)", args.manifest, len(manifest))
    print(
        f"Wrote {len(candidates)} candidate rows to {project_relative(args.candidates)}"
    )
    print(f"Wrote {len(manifest)} review rows to {project_relative(args.manifest)}")

    if args.overlays:
        render_overlays(
            review_files=review_files,
            candidates=candidates,
            overlay_dir=args.overlay_dir,
            overwrite=args.overwrite_overlays,
            limit=args.limit,
        )
        rendered_count = (
            min(len(review_files), args.limit)
            if args.limit is not None
            else len(review_files)
        )
        print(
            f"Rendered review overlays for {rendered_count} files under {project_relative(args.overlay_dir)}"
        )


if __name__ == "__main__":
    main()
