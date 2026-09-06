"""Create an interactive Plotly view of a FELT v2 facial sequence.

Unlike Py-Feat's ``animate_face_mesh_plotly(start, end)``, this tool does not
interpolate between two expressions. Each animation frame comes from one row
of a FELT CSV. The 3D view is linked to an AU heatmap and a moving time cursor.

Two mesh sources are useful for different questions:

``detected``
    The 478-point mesh measured by Detectorv2. Translation and face scale are
    removed per frame, while expression and head rotation are retained.

``au``
    A pose-canonical mesh reconstructed from the 20 AU scores. This removes
    identity and head motion and shows only the geometry implied by the AUs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


AU_COLUMNS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate an observed FELT v2 sequence as an interactive 3D mesh."
    )
    parser.add_argument("input_csv", type=Path, help="Raw or smoothed FELT v2 CSV.")
    parser.add_argument("output_html", type=Path, help="Standalone Plotly HTML output.")
    parser.add_argument(
        "--source",
        choices=("detected", "au"),
        default="detected",
        help="Use the observed Detectorv2 mesh or reconstruct a canonical mesh from AUs.",
    )
    parser.add_argument(
        "--mode",
        choices=("contours", "tesselation", "tessellation"),
        default="contours",
        help="Contours are compact; the full tessellation is much larger.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth CSV row.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Evenly sample at most this many frames after applying --stride.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=("cdn", "inline"),
        default="cdn",
        help="CDN makes a smaller HTML; inline works fully offline.",
    )
    parser.add_argument(
        "--muscle-heatmap",
        action="store_true",
        help="Shade the v2 AU muscle regions on the animated 3D mesh.",
    )
    return parser.parse_args(argv)


def select_rows(data: pd.DataFrame, stride: int, max_frames: int | None) -> pd.DataFrame:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    selected = data.iloc[::stride]
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        if len(selected) > max_frames:
            positions = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
            selected = selected.iloc[np.unique(positions)]
    return selected.reset_index(drop=True)


def detected_meshes(data: pd.DataFrame) -> np.ndarray:
    """Return face-centred meshes with comparable x/y/z units."""
    x_cols = [f"mesh_x_{i}" for i in range(478)]
    y_cols = [f"mesh_y_{i}" for i in range(478)]
    z_cols = [f"mesh_z_{i}" for i in range(478)]
    required = x_cols + y_cols + z_cols + ["FaceRectWidth", "FaceRectHeight"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"CSV is missing Detectorv2 mesh columns: {missing[:5]}")

    x = data[x_cols].to_numpy(dtype=np.float32)
    y = data[y_cols].to_numpy(dtype=np.float32)
    z = data[z_cols].to_numpy(dtype=np.float32)
    scale = data[["FaceRectWidth", "FaceRectHeight"]].max(axis=1).to_numpy(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)

    # X/Y were decoded to source-image pixels, while Z remains normalized to
    # the square face crop. Dividing X/Y by the crop scale puts all axes in the
    # normalized units already used by Z. Per-frame centring removes camera
    # translation without erasing expression or head rotation.
    x = x / scale[:, None]
    y = -y / scale[:, None]
    x -= np.nanmedian(x, axis=1, keepdims=True)
    y -= np.nanmedian(y, axis=1, keepdims=True)
    z -= np.nanmedian(z, axis=1, keepdims=True)
    return np.stack([x, z, y], axis=-1)


def au_meshes(data: pd.DataFrame) -> np.ndarray:
    missing = [column for column in AU_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"CSV is missing AU columns: {missing}")

    from feat.plotting import load_face_mesh_viz_model, predict_face_mesh

    model = load_face_mesh_viz_model()
    model_columns = list(model.au_columns)
    aus = data[model_columns].to_numpy(dtype=np.float32)
    meshes = np.asarray(predict_face_mesh(aus, model=model), dtype=np.float32)
    # Match Py-Feat's upright Plotly convention: data (X,Y,Z) -> (X,Z,Y).
    return meshes[:, :, [0, 2, 1]]


def mesh_connections(mode: str) -> np.ndarray:
    from feat.utils.mp_plotting import FaceLandmarksConnections

    if mode in ("tesselation", "tessellation"):
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    else:
        connections = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    return np.asarray([(edge.start, edge.end) for edge in connections], dtype=int)


def segment_coordinates(mesh: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    """Encode all mesh edges as NaN-separated coordinates for one trace."""
    segments = np.full((len(edges), 3, 3), np.nan, dtype=np.float32)
    segments[:, 0, :] = mesh[edges[:, 0]]
    segments[:, 1, :] = mesh[edges[:, 1]]
    flattened = segments.reshape(-1, 3)
    return flattened[:, 0], flattened[:, 1], flattened[:, 2]


def frame_values(data: pd.DataFrame) -> np.ndarray:
    if "frame" in data:
        return data["frame"].to_numpy()
    return np.arange(len(data))


def time_values(data: pd.DataFrame, fps: float) -> np.ndarray:
    if "approx_time" in data:
        values = pd.to_numeric(data["approx_time"], errors="coerce").to_numpy(float)
        if np.isfinite(values).all():
            return values
    return frame_values(data).astype(float) / fps


def mesh_trace(mesh: np.ndarray, edges: np.ndarray) -> go.Scatter3d:
    x, y, z = segment_coordinates(mesh, edges)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": "#3767b1", "width": 2},
        hoverinfo="skip",
        showlegend=False,
    )


def muscle_region_assets() -> tuple[np.ndarray, dict[str, list[int]]]:
    """Load Py-Feat's non-overlapping AU-to-mesh triangle partition."""
    from feat.utils import region_maps

    assets = region_maps.render_assets("au")
    return np.asarray(assets["tris"], dtype=int), assets["region_tris"]


def _rgba_red(value: float) -> str:
    """Map a [0, 1] AU score to Py-Feat-like red heatmap color."""
    from plotly.colors import sample_colorscale

    clipped = float(np.clip(value, 0, 1))
    rgb = sample_colorscale("Reds", [clipped])[0]
    channels = rgb.removeprefix("rgb(").removesuffix(")")
    alpha = 0.08 + 0.84 * clipped
    return f"rgba({channels},{alpha:.3f})"


def muscle_trace(
    mesh: np.ndarray,
    au_values: pd.Series,
    triangles: np.ndarray,
    region_triangles: dict[str, list[int]],
) -> go.Mesh3d:
    """Render AU scores on Py-Feat's v2 facial muscle-region partition."""
    face_colors = np.full(len(triangles), "rgba(255,255,255,0)", dtype=object)
    for au_name, triangle_indices in region_triangles.items():
        value = float(au_values.get(au_name, 0.0))
        face_colors[np.asarray(triangle_indices, dtype=int)] = _rgba_red(value)

    return go.Mesh3d(
        x=mesh[:, 0],
        y=mesh[:, 1],
        z=mesh[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        facecolor=face_colors.tolist(),
        flatshading=True,
        hoverinfo="skip",
        showscale=False,
        showlegend=False,
        name="AU muscle regions",
    )


def cursor_trace(time_value: float) -> go.Scatter:
    return go.Scatter(
        x=[time_value, time_value],
        y=[-0.5, len(AU_COLUMNS) - 0.5],
        mode="lines",
        line={"color": "#111111", "width": 2},
        hoverinfo="skip",
        showlegend=False,
    )


def build_figure(
    data: pd.DataFrame,
    meshes: np.ndarray,
    edges: np.ndarray,
    *,
    fps: float,
    source: str,
    muscle_heatmap: bool = False,
) -> go.Figure:
    if len(data) != len(meshes):
        raise ValueError("data and meshes must contain the same number of frames")
    if len(data) == 0:
        raise ValueError("the selected sequence is empty")
    if not np.isfinite(meshes).all():
        raise ValueError("selected frames contain missing or infinite mesh coordinates")

    times = time_values(data, fps)
    frames_in_source = frame_values(data)
    au_values = data[AU_COLUMNS].to_numpy(dtype=float).T

    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )
    figure.add_trace(mesh_trace(meshes[0], edges), row=1, col=1)
    muscle_assets = muscle_region_assets() if muscle_heatmap else None
    if muscle_assets is not None:
        triangles, region_triangles = muscle_assets
        figure.add_trace(
            muscle_trace(
                meshes[0], data.loc[0, AU_COLUMNS], triangles, region_triangles
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Heatmap(
            x=times,
            y=AU_COLUMNS,
            z=au_values,
            zmin=0,
            zmax=max(1.0, float(np.nanmax(au_values))),
            colorscale="Viridis",
            colorbar={"title": "AU score", "len": 0.26, "y": 0.13},
            hovertemplate="%{y}<br>%{x:.3f} s<br>%{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    cursor_index = len(figure.data)
    figure.add_trace(cursor_trace(times[0]), row=2, col=1)

    lower = meshes.reshape(-1, 3).min(axis=0)
    upper = meshes.reshape(-1, 3).max(axis=0)
    centre = (lower + upper) / 2
    half_extent = max(float(np.max(upper - lower)) / 2, 1e-6) * 1.04
    axis_ranges = [
        [float(value - half_extent), float(value + half_extent)] for value in centre
    ]

    duration_ms = max(1, round(1000 / fps))
    animation_frames = []
    slider_steps = []
    for index, (mesh, time_value, source_frame) in enumerate(
        zip(meshes, times, frames_in_source, strict=True)
    ):
        name = str(index)
        frame_data: list[go.BaseTraceType] = [mesh_trace(mesh, edges)]
        frame_trace_indices = [0]
        if muscle_assets is not None:
            triangles, region_triangles = muscle_assets
            frame_data.append(
                muscle_trace(
                    mesh,
                    data.loc[index, AU_COLUMNS],
                    triangles,
                    region_triangles,
                )
            )
            frame_trace_indices.append(1)
        frame_data.append(cursor_trace(float(time_value)))
        frame_trace_indices.append(cursor_index)
        animation_frames.append(
            go.Frame(
                name=name,
                data=frame_data,
                traces=frame_trace_indices,
            )
        )
        slider_steps.append(
            {
                "label": str(int(source_frame)),
                "method": "animate",
                "args": [[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            }
        )
    figure.frames = animation_frames

    figure.update_layout(
        title={
            "text": (
                f"FELT sequence — {source} mesh"
                + (" with AU muscle heatmap" if muscle_heatmap else "")
            ),
            "x": 0.5,
        },
        height=820,
        margin={"l": 80, "r": 50, "t": 60, "b": 125},
        scene={
            "xaxis": {"visible": False, "range": axis_ranges[0]},
            "yaxis": {"visible": False, "range": axis_ranges[1]},
            "zaxis": {"visible": False, "range": axis_ranges[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 0, "y": -2.4, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}},
        },
        xaxis={"title": "Time (s)", "range": [float(times.min()), float(times.max())]},
        yaxis={"title": None, "autorange": "reversed"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.13,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"fromcurrent": True, "frame": {"duration": duration_ms, "redraw": True}, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Source frame "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )
    return figure


def create_sequence_figure(
    csv_path: Path,
    *,
    source: str = "detected",
    mode: str = "contours",
    fps: float = 30.0,
    stride: int = 1,
    max_frames: int | None = None,
    muscle_heatmap: bool = False,
) -> go.Figure:
    data = select_rows(pd.read_csv(csv_path), stride, max_frames)
    meshes = detected_meshes(data) if source == "detected" else au_meshes(data)
    return build_figure(
        data,
        meshes,
        mesh_connections(mode),
        fps=fps,
        source=source,
        muscle_heatmap=muscle_heatmap,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    figure = create_sequence_figure(
        args.input_csv,
        source=args.source,
        mode=args.mode,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        muscle_heatmap=args.muscle_heatmap,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output_html,
        include_plotlyjs=args.include_plotlyjs,
        full_html=True,
        auto_play=False,
    )
    print(f"Saved interactive sequence: {args.output_html}")


if __name__ == "__main__":
    main()
