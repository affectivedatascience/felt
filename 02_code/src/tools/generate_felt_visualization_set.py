"""Generate seven organized FELT v2 visualization videos for one tracking CSV.

Each H.264 video contains one output frame per selected FELT sequence row; no
expression frames are synthesized or interpolated.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.animate_au_mesh_csv import (  # noqa: E402
    frame_labels as au_mesh_frame_labels,
)
from tools.animate_au_mesh_csv import (  # noqa: E402
    iter_figures as iter_au_mesh_figures,
)
from tools.animate_au_mesh_csv import (  # noqa: E402
    mesh_edges as au_mesh_edges,
)
from tools.animate_au_mesh_csv import (  # noqa: E402
    predict_sequence as predict_au_mesh_sequence,
)
from tools.animate_au_mesh_csv import (  # noqa: E402
    project_mesh_2d,
)
from tools.plot_felt_sequence import (  # noqa: E402
    create_sequence_figure,
    mesh_connections,
    select_rows,
)
from utils.felt_paths import INPUT_DIR  # noqa: E402
from utils.video_rendering import write_figures_to_video  # noqa: E402

DEFAULT_VIEW_NAMES = (
    "au_region_heatmap",
    "blendshape_region_heatmap",
    "au_to_mesh",
    "landmark_only_contours",
    "landmark_only_tessellation",
    "landmark_overlay_contours",
    "landmark_overlay_tessellation",
)

# Retain the exploratory views as explicit opt-ins, but keep the default output
# focused on the seven organized MP4 products.
VIEW_NAMES = DEFAULT_VIEW_NAMES + (
    "mesh_contours",
    "mesh_tessellation",
    "mesh_overlay_contours",
    "mesh_overlay_tessellation",
    "legacy_au_muscle_face",
)


def actor_name_from_csv(csv_path: Path) -> str:
    """Return an Actor_XX folder name from a FELT/RAVDESS CSV path."""
    parent_name = csv_path.parent.name
    if (
        parent_name.startswith("Actor_")
        and len(parent_name) == len("Actor_00")
        and parent_name[-2:].isdigit()
    ):
        return parent_name

    actor_id = csv_path.stem.rsplit("-", maxsplit=1)[-1]
    if len(actor_id) == 2 and actor_id.isdigit():
        return f"Actor_{actor_id}"
    raise ValueError(
        "Could not determine actor from the CSV parent directory or filename: "
        f"{csv_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Original video; inferred from the CSV input column when omitted.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--faceboxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw cyan face bounding boxes on mesh-overlay videos (default: enabled).",
    )
    parser.add_argument(
        "--gazes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw yellow gaze arrows on mesh-overlay videos (default: enabled).",
    )
    parser.add_argument(
        "--poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw red/green/blue head-pose axes on mesh-overlay videos "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--original-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set the background for the legacy mesh_overlay_* views. The "
            "organized landmark-only and landmark-overlay backgrounds are fixed."
        ),
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=VIEW_NAMES,
        help=(
            "Generate only this view; repeat for multiple views. "
            "Default: the seven organized MP4 products."
        ),
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=("inline", "cdn"),
        default="inline",
    )
    return parser.parse_args()


def infer_source_video(data: pd.DataFrame, csv_path: Path) -> Path:
    if "input" in data and data["input"].notna().any():
        candidate = Path(str(data.loc[data["input"].notna(), "input"].iloc[0]))
        if candidate.exists():
            return candidate

    actor_name = csv_path.parent.name
    candidate = INPUT_DIR / actor_name / f"{csv_path.stem}.mp4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not find the original video. Pass it explicitly with --source-video."
    )


def detected_xy(data: pd.DataFrame) -> np.ndarray:
    columns = [f"mesh_x_{i}" for i in range(478)] + [f"mesh_y_{i}" for i in range(478)]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"CSV lacks Detectorv2 mesh coordinates: {missing[:5]}")
    x = data[[f"mesh_x_{i}" for i in range(478)]].to_numpy(float)
    y = data[[f"mesh_y_{i}" for i in range(478)]].to_numpy(float)
    return np.stack([x, y], axis=-1)


def _draw_mesh_overlay(
    frame: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    *,
    color: tuple[int, int, int, int],
    width: int,
    label: str,
    facebox: np.ndarray | None = None,
    gaze: np.ndarray | None = None,
    pose: np.ndarray | None = None,
    draw_facebox: bool = True,
) -> np.ndarray:
    base = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    finite = np.isfinite(points).all(axis=1)
    for start, end in edges:
        if finite[start] and finite[end]:
            draw.line(
                [tuple(points[start]), tuple(points[end])],
                fill=color,
                width=width,
            )

    if facebox is not None and np.isfinite(facebox).all():
        x, y, box_width, box_height = facebox
        if draw_facebox:
            draw.rectangle(
                (x, y, x + box_width, y + box_height),
                outline=(0, 255, 255, 255),
                width=3,
            )

        if pose is not None and np.isfinite(pose).all():
            pitch, roll, yaw = pose
            yaw = -yaw
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            size = min(box_width, box_height) / 2.0

            # Match Py-Feat's draw_facepose projection and axis colors.
            x_axis = (
                center_x + size * np.cos(yaw) * np.cos(roll),
                center_y
                + size
                * (
                    np.cos(pitch) * np.sin(roll)
                    + np.cos(roll) * np.sin(pitch) * np.sin(yaw)
                ),
            )
            y_axis = (
                center_x - size * np.cos(yaw) * np.sin(roll),
                center_y
                + size
                * (
                    np.cos(pitch) * np.cos(roll)
                    - np.sin(pitch) * np.sin(yaw) * np.sin(roll)
                ),
            )
            z_axis = (
                center_x + size * np.sin(yaw),
                center_y - size * np.cos(yaw) * np.sin(pitch),
            )
            for endpoint, axis_color in (
                (x_axis, (255, 0, 0, 255)),
                (y_axis, (0, 255, 0, 255)),
                (z_axis, (0, 80, 255, 255)),
            ):
                draw.line(
                    (center_x, center_y, *endpoint),
                    fill=axis_color,
                    width=4,
                )

        if gaze is not None and np.isfinite(gaze).all():
            pitch, yaw = gaze
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            length = min(box_width, box_height) * 1.1
            delta_x = length * np.sin(yaw) * np.cos(pitch)
            delta_y = -length * np.sin(pitch)
            tip_x = center_x + delta_x
            tip_y = center_y + delta_y
            draw.line(
                (center_x, center_y, tip_x, tip_y),
                fill=(255, 255, 0, 255),
                width=4,
            )

            arrow_length = float(np.hypot(delta_x, delta_y))
            head_length = min(arrow_length * 0.35, length * 0.08)
            if arrow_length > 0 and head_length > 0:
                unit_x = delta_x / arrow_length
                unit_y = delta_y / arrow_length
                perpendicular_x = -unit_y
                perpendicular_y = unit_x
                head_width = head_length * 0.75
                base_x = tip_x - head_length * unit_x
                base_y = tip_y - head_length * unit_y
                draw.polygon(
                    (
                        (tip_x, tip_y),
                        (
                            base_x + head_width * perpendicular_x / 2.0,
                            base_y + head_width * perpendicular_y / 2.0,
                        ),
                        (
                            base_x - head_width * perpendicular_x / 2.0,
                            base_y - head_width * perpendicular_y / 2.0,
                        ),
                    ),
                    fill=(255, 255, 0, 255),
                )
    draw.rounded_rectangle((12, 12, 270, 46), radius=7, fill=(0, 0, 0, 160))
    draw.text((22, 21), label, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def write_mesh_overlay_videos(
    data: pd.DataFrame,
    source_video: Path | None,
    contour_path: Path | None,
    tessellation_path: Path | None,
    fps: float,
    *,
    faceboxes: bool = True,
    gazes: bool = True,
    poses: bool = True,
    original_image: bool = True,
) -> None:
    contour_edges = mesh_connections("contours") if contour_path else None
    tessellation_edges = mesh_connections("tessellation") if tessellation_path else None
    points = detected_xy(data)
    facebox_columns = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
    required_columns = facebox_columns.copy() if faceboxes or gazes or poses else []
    if gazes:
        required_columns += ["gaze_pitch", "gaze_yaw"]
    if poses:
        required_columns += ["Pitch", "Roll", "Yaw"]
    missing = [column for column in required_columns if column not in data]
    if missing:
        raise ValueError(f"CSV lacks requested overlay columns: {missing}")
    facebox_values = (
        data[facebox_columns].to_numpy(float)
        if faceboxes or gazes or poses
        else None
    )
    gaze_values = (
        data[["gaze_pitch", "gaze_yaw"]].to_numpy(float) if gazes else None
    )
    pose_values = data[["Pitch", "Roll", "Yaw"]].to_numpy(float) if poses else None
    source_frames = (
        data["frame"].round().astype(int).to_numpy()
        if "frame" in data
        else np.arange(len(data))
    )
    for path in (contour_path, tessellation_path):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    contour_writer = (
        imageio.get_writer(contour_path, fps=fps, codec="libx264", macro_block_size=None)
        if contour_path
        else None
    )
    tessellation_writer = (
        imageio.get_writer(tessellation_path, fps=fps, codec="libx264", macro_block_size=None)
        if tessellation_path
        else None
    )
    written = 0

    def append_frame(row_index: int, frame: np.ndarray) -> None:
        nonlocal written
        source_frame = source_frames[row_index]
        facebox = (
            facebox_values[row_index] if facebox_values is not None else None
        )
        gaze = gaze_values[row_index] if gazes else None
        pose = pose_values[row_index] if poses else None
        if contour_writer is not None and contour_edges is not None:
            contour_writer.append_data(
                _draw_mesh_overlay(
                    frame,
                    points[row_index],
                    contour_edges,
                    color=(0, 220, 255, 235),
                    width=3,
                    label=f"mesh contours | source frame {source_frame}",
                    facebox=facebox,
                    gaze=gaze,
                    pose=pose,
                    draw_facebox=faceboxes,
                )
            )
        if tessellation_writer is not None and tessellation_edges is not None:
            tessellation_writer.append_data(
                _draw_mesh_overlay(
                    frame,
                    points[row_index],
                    tessellation_edges,
                    color=(70, 255, 150, 125),
                    width=1,
                    label=f"mesh tessellation | source frame {source_frame}",
                    facebox=facebox,
                    gaze=gaze,
                    pose=pose,
                    draw_facebox=faceboxes,
                )
            )
        written += 1

    reader = None
    try:
        if original_image:
            if source_video is None:
                raise ValueError("source_video is required when original_image=True")
            requested = {
                int(frame): index for index, frame in enumerate(source_frames)
            }
            reader = imageio.get_reader(source_video)
            for decoded_index, frame in enumerate(reader):
                row_index = requested.get(decoded_index)
                if row_index is not None:
                    append_frame(row_index, frame)
                if written == len(data):
                    break
        else:
            dimension_columns = ["FrameHeight", "FrameWidth"]
            missing_dimensions = [
                column for column in dimension_columns if column not in data
            ]
            if missing_dimensions:
                raise ValueError(
                    "CSV lacks frame dimensions required for mesh-only rendering: "
                    f"{missing_dimensions}"
                )
            dimensions = data[dimension_columns].to_numpy(float)
            for row_index, (height, width) in enumerate(dimensions):
                if not np.isfinite([height, width]).all() or height <= 0 or width <= 0:
                    raise ValueError(
                        f"Invalid frame dimensions at row {row_index}: "
                        f"height={height}, width={width}"
                    )
                blank_frame = np.full(
                    (round(height), round(width), 3), 255, dtype=np.uint8
                )
                append_frame(row_index, blank_frame)
    finally:
        if reader is not None:
            reader.close()
        if contour_writer is not None:
            contour_writer.close()
        if tessellation_writer is not None:
            tessellation_writer.close()

    if written != len(data):
        raise RuntimeError(f"Wrote {written} overlay frames; expected {len(data)}")


def _add_score_colorbar(fig, ax, cmap: str, label: str) -> None:
    scale = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    scale.set_array([])
    colorbar = fig.colorbar(scale, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(label)


def _add_canonical_face_outline(ax) -> None:
    """Draw the MediaPipe face/features contour over a canonical region map."""
    from feat.utils import region_maps
    from feat.utils.mp_plotting import FaceLandmarksConnections

    vertices, _ = region_maps._canonical_geometry()
    xy = region_maps.project_xy(vertices)
    edges = np.asarray(
        [
            (edge.start, edge.end)
            for edge in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
        ],
        dtype=int,
    )
    ax.add_collection(
        LineCollection(
            xy[edges],
            colors="#333333",
            linewidths=1.15,
            alpha=0.9,
            zorder=6,
        )
    )


def iter_region_figures(data: pd.DataFrame, kind: str) -> Iterable:
    from feat.plotting import plot_face_regions
    from feat.utils import region_maps

    if kind == "au":
        names = list(region_maps.load_au_region_map())
        cmap = "Reds"
        title = "AU region heatmap"
    else:
        names = list(region_maps.load_blendshape_region_map())
        cmap = "Blues"
        title = "Blendshape region heatmap"

    source_frames = data["frame"].to_numpy() if "frame" in data else np.arange(len(data))
    for (_, row), source_frame in zip(data.iterrows(), source_frames, strict=True):
        values = {name: float(row.get(name, 0.0)) for name in names}
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        plot_face_regions(
            values=values,
            kind=kind,
            ax=ax,
            cmap=cmap,
            alpha=0.92,
            mesh=True,
            title=f"{title} | source frame {int(source_frame)}",
        )
        _add_canonical_face_outline(ax)
        _add_score_colorbar(fig, ax, cmap, "Model score")
        fig.tight_layout()
        yield fig


def iter_legacy_muscle_figures(data: pd.DataFrame) -> Iterable:
    from feat.plotting import load_viz_model, plot_face

    model = load_viz_model()
    model_columns = list(model.au_columns)
    missing = [column for column in model_columns if column not in data]
    if missing:
        raise ValueError(f"CSV lacks legacy visualization AU inputs: {missing}")
    source_frames = data["frame"].to_numpy() if "frame" in data else np.arange(len(data))
    for (_, row), source_frame in zip(data.iterrows(), source_frames, strict=True):
        ax = plot_face(
            au=row[model_columns].to_numpy(dtype=float),
            model=model,
            muscles={"all": "heatmap"},
            cmap="Reds",
            title=f"Legacy AU muscle face | source frame {int(source_frame)}",
        )
        yield ax.figure


def write_3d_html(
    csv_path: Path,
    output_path: Path,
    mode: str,
    *,
    fps: float,
    stride: int,
    max_frames: int | None,
    include_plotlyjs: str,
) -> None:
    figure = create_sequence_figure(
        csv_path,
        source="detected",
        mode=mode,
        fps=fps,
        stride=stride,
        max_frames=max_frames,
    )
    figure.update_layout(title={"text": f"FELT detected mesh — {mode}", "x": 0.5})
    figure.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_play=False,
    )


def organized_output_paths(input_csv: Path, output_dir: Path) -> dict[str, Path]:
    """Return the seven default output paths for one input CSV."""
    csv_stem = input_csv.stem
    actor_name = actor_name_from_csv(input_csv)
    return {
        "au_region_heatmap": (
            output_dir
            / "AU_animation"
            / "au_region_heatmap"
            / actor_name
            / f"au_region_heatmap_{csv_stem}.mp4"
        ),
        "blendshape_region_heatmap": (
            output_dir
            / "AU_animation"
            / "blendshape_region"
            / actor_name
            / f"blendshape_region_heatmap_{csv_stem}.mp4"
        ),
        "au_to_mesh": (
            output_dir
            / "AU_animation"
            / "au_to_mesh"
            / actor_name
            / f"au_to_mesh_{csv_stem}.mp4"
        ),
        "landmark_only_contours": (
            output_dir
            / "landmark_only"
            / "mesh_overlay_contours"
            / actor_name
            / f"mesh_overlay_contours_{csv_stem}.mp4"
        ),
        "landmark_only_tessellation": (
            output_dir
            / "landmark_only"
            / "mesh_overlay_tessellation"
            / actor_name
            / f"mesh_overlay_tessellation_{csv_stem}.mp4"
        ),
        "landmark_overlay_contours": (
            output_dir
            / "Landmark_overlay"
            / "mesh_overlay_contours"
            / actor_name
            / f"mesh_overlay_contours_{csv_stem}.mp4"
        ),
        "landmark_overlay_tessellation": (
            output_dir
            / "Landmark_overlay"
            / "mesh_overlay_tessellation"
            / actor_name
            / f"mesh_overlay_tessellation_{csv_stem}.mp4"
        ),
    }


def _generate_from_args(args: argparse.Namespace) -> tuple[dict[str, Path], int]:
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    requested = set(args.view or DEFAULT_VIEW_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = args.input_csv.stem
    output_paths = organized_output_paths(args.input_csv, args.output_dir)

    full_data = pd.read_csv(args.input_csv)
    data = select_rows(full_data, args.stride, args.max_frames)

    if "mesh_contours" in requested:
        write_3d_html(
            args.input_csv,
            args.output_dir / "mesh_contours.html",
            "contours",
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            include_plotlyjs=args.include_plotlyjs,
        )
    if "mesh_tessellation" in requested:
        write_3d_html(
            args.input_csv,
            args.output_dir / "mesh_tessellation.html",
            "tessellation",
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            include_plotlyjs=args.include_plotlyjs,
        )

    overlay_contours = "mesh_overlay_contours" in requested
    overlay_tessellation = "mesh_overlay_tessellation" in requested
    if overlay_contours or overlay_tessellation:
        source_video = (
            args.source_video or infer_source_video(full_data, args.input_csv)
            if args.original_image
            else None
        )
        write_mesh_overlay_videos(
            data,
            source_video,
            args.output_dir / f"mesh_overlay_contours_{csv_stem}.mp4"
            if overlay_contours
            else None,
            args.output_dir / f"mesh_overlay_tessellation_{csv_stem}.mp4"
            if overlay_tessellation
            else None,
            args.fps,
            faceboxes=args.faceboxes,
            gazes=args.gazes,
            poses=args.poses,
            original_image=args.original_image,
        )

    landmark_only_contours = "landmark_only_contours" in requested
    landmark_only_tessellation = "landmark_only_tessellation" in requested
    if landmark_only_contours or landmark_only_tessellation:
        write_mesh_overlay_videos(
            data,
            None,
            output_paths["landmark_only_contours"]
            if landmark_only_contours
            else None,
            output_paths["landmark_only_tessellation"]
            if landmark_only_tessellation
            else None,
            args.fps,
            faceboxes=args.faceboxes,
            gazes=args.gazes,
            poses=args.poses,
            original_image=False,
        )

    landmark_overlay_contours = "landmark_overlay_contours" in requested
    landmark_overlay_tessellation = "landmark_overlay_tessellation" in requested
    if landmark_overlay_contours or landmark_overlay_tessellation:
        source_video = args.source_video or infer_source_video(full_data, args.input_csv)
        write_mesh_overlay_videos(
            data,
            source_video,
            output_paths["landmark_overlay_contours"]
            if landmark_overlay_contours
            else None,
            output_paths["landmark_overlay_tessellation"]
            if landmark_overlay_tessellation
            else None,
            args.fps,
            faceboxes=args.faceboxes,
            gazes=args.gazes,
            poses=args.poses,
            original_image=True,
        )

    if "au_region_heatmap" in requested:
        write_figures_to_video(
            iter_region_figures(data, "au"),
            output_paths["au_region_heatmap"],
            fps=round(args.fps),
            set_size=(720, 720),
        )
    if "blendshape_region_heatmap" in requested:
        write_figures_to_video(
            iter_region_figures(data, "blendshape"),
            output_paths["blendshape_region_heatmap"],
            fps=round(args.fps),
            set_size=(720, 720),
        )
    if "au_to_mesh" in requested:
        meshes_3d, neutral_3d, _ = predict_au_mesh_sequence(data)
        write_figures_to_video(
            iter_au_mesh_figures(
                project_mesh_2d(meshes_3d),
                project_mesh_2d(neutral_3d),
                au_mesh_edges("tesselation"),
                au_mesh_frame_labels(data),
            ),
            output_paths["au_to_mesh"],
            fps=round(args.fps),
            set_size=(720, 720),
        )
    if "legacy_au_muscle_face" in requested:
        write_figures_to_video(
            iter_legacy_muscle_figures(data),
            args.output_dir / f"legacy_au_muscle_face_{csv_stem}.mp4",
            fps=round(args.fps),
            set_size=(720, 720),
        )

    if getattr(args, "print_paths", True):
        for name in DEFAULT_VIEW_NAMES:
            if name in requested:
                print(output_paths[name])

        for name in (
            "mesh_contours",
            "mesh_tessellation",
            "mesh_overlay_contours",
            "mesh_overlay_tessellation",
            "legacy_au_muscle_face",
        ):
            if name in {"mesh_contours", "mesh_tessellation"}:
                path = args.output_dir / f"{name}.html"
            else:
                path = args.output_dir / f"{name}_{csv_stem}.mp4"
            if name in requested:
                print(path)

    return output_paths, len(data)


def generate_visualization_set(
    input_csv: Path,
    output_dir: Path,
    *,
    source_video: Path | None = None,
    fps: float = 30.0,
    stride: int = 1,
    max_frames: int | None = None,
    faceboxes: bool = True,
    gazes: bool = True,
    poses: bool = True,
    views: Iterable[str] | None = None,
    print_paths: bool = True,
) -> tuple[dict[str, Path], int]:
    """Generate selected organized views and return their paths and frame count."""
    requested = list(views) if views is not None else None
    unknown = set(requested or ()) - set(DEFAULT_VIEW_NAMES)
    if unknown:
        raise ValueError(f"Unsupported organized views: {sorted(unknown)}")
    args = argparse.Namespace(
        input_csv=Path(input_csv),
        output_dir=Path(output_dir),
        source_video=Path(source_video) if source_video is not None else None,
        fps=fps,
        stride=stride,
        max_frames=max_frames,
        faceboxes=faceboxes,
        gazes=gazes,
        poses=poses,
        original_image=True,
        view=requested,
        include_plotlyjs="inline",
        print_paths=print_paths,
    )
    return _generate_from_args(args)


def main() -> None:
    _generate_from_args(parse_args())


if __name__ == "__main__":
    main()
