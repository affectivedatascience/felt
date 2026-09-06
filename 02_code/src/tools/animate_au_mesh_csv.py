"""Animate CSV Action Units as Py-Feat canonical displacement-colored meshes.

Each selected CSV row is passed directly to Py-Feat's ``au_to_mesh`` model.
The resulting pose-canonical mesh is drawn over the canonical neutral mesh and
colored by per-vertex displacement from neutral, following the Py-Feat AU atlas
example. No detected mesh columns are read and no AU frames are interpolated.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.ffmpeg_runtime import (  # noqa: E402
    configure_ffmpeg_dlls,
    resolve_ffmpeg_bin,
)
from utils.video_rendering import write_figures_to_video  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing Py-Feat AU columns.")
    parser.add_argument("output_mp4", type=Path, help="Destination H.264 MP4.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth CSV row.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Evenly sample at most this many frames after applying --stride.",
    )
    parser.add_argument(
        "--mode",
        choices=("contours", "tesselation", "tessellation"),
        default="tesselation",
        help="Use the full atlas tessellation (default) or lighter contours.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Fixed displacement color maximum; default is the sequence maximum.",
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        help="Windows FFmpeg shared-build bin directory; normally auto-detected.",
    )
    return parser.parse_args(argv)


def select_rows(
    data: pd.DataFrame, stride: int, max_frames: int | None
) -> pd.DataFrame:
    """Select observed rows without synthesizing or interpolating AU values."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    selected = data.iloc[::stride]
    if max_frames is not None and len(selected) > max_frames:
        positions = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
        selected = selected.iloc[np.unique(positions)]
    if selected.empty:
        raise ValueError("the selected CSV sequence is empty")
    return selected.copy()


def au_array(data: pd.DataFrame, au_columns: Sequence[str]) -> np.ndarray:
    """Return finite AU values in the visualization model's required order."""
    missing = [column for column in au_columns if column not in data]
    if missing:
        raise ValueError(f"CSV is missing AU columns: {missing}")

    numeric = data[list(au_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    invalid = np.argwhere(~np.isfinite(values))
    if len(invalid):
        row_position, column_position = invalid[0]
        source_row = data.index[int(row_position)]
        column = au_columns[int(column_position)]
        raise ValueError(
            f"CSV contains a missing or non-numeric AU value at row "
            f"{source_row}, column {column}"
        )
    return values


def project_mesh_2d(meshes: np.ndarray) -> np.ndarray:
    """Apply the front projection and orientation check used by Py-Feat's atlas."""
    values = np.asarray(meshes, dtype=np.float32)
    single_mesh = values.ndim == 2
    if single_mesh:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (478, 3):
        raise ValueError(f"expected mesh shape (478, 3) or (N, 478, 3), got {values.shape}")

    projected = values[:, :, :2].copy()
    flip = projected[:, 10, 1] < projected[:, 152, 1]
    projected[flip, :, 1] *= -1
    return projected[0] if single_mesh else projected


def mesh_edges(mode: str) -> np.ndarray:
    from feat.utils.mp_plotting import FaceLandmarksConnections

    if mode in ("tesselation", "tessellation"):
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    else:
        connections = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    return np.asarray([(edge.start, edge.end) for edge in connections], dtype=int)


@lru_cache(maxsize=1)
def _face_mesh_model():
    """Load the AU-to-mesh model once per batch worker process."""
    from feat.plotting import load_face_mesh_viz_model

    return load_face_mesh_viz_model()


def predict_sequence(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict activated and neutral canonical meshes from CSV AU rows."""
    from feat.plotting import predict_face_mesh

    model = _face_mesh_model()
    columns = list(model.au_columns)
    values = au_array(data, columns)
    meshes = np.asarray(predict_face_mesh(values, model=model), dtype=np.float32)
    neutral = np.asarray(
        predict_face_mesh(np.zeros(len(columns), dtype=np.float32), model=model),
        dtype=np.float32,
    )
    return meshes, neutral, columns


def _plot_limits(meshes: np.ndarray, neutral: np.ndarray) -> tuple[tuple[float, float], ...]:
    all_points = np.concatenate([meshes.reshape(-1, 2), neutral], axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.04, 1e-6)
    bounds = zip(lower, upper, padding, strict=True)
    return tuple((float(lo - pad), float(hi + pad)) for lo, hi, pad in bounds)


def iter_figures(
    meshes: np.ndarray,
    neutral: np.ndarray,
    edges: np.ndarray,
    labels: Sequence[str],
    *,
    vmax: float | None = None,
) -> Iterator[plt.Figure]:
    """Yield atlas-style frames with one color scale shared by the sequence."""
    displacement = np.linalg.norm(meshes - neutral[None, :, :], axis=2)
    color_max = float(np.max(displacement)) if vmax is None else float(vmax)
    if not np.isfinite(color_max) or color_max <= 0:
        color_max = 1.0
    limits = _plot_limits(meshes, neutral)

    for mesh, magnitude, label in zip(meshes, displacement, labels, strict=True):
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.add_collection(
            LineCollection(
                neutral[edges], colors="lightgray", linewidths=0.35, alpha=0.55
            )
        )
        activated_collection = LineCollection(
            mesh[edges],
            array=magnitude[edges].mean(axis=1),
            cmap="plasma",
            # Do not share one Normalize instance across short-lived figures.
            # Matplotlib attaches weak callbacks to it, and concurrent cleanup
            # can otherwise emit a harmless CallbackRegistry KeyError.
            norm=Normalize(vmin=0.0, vmax=color_max),
            linewidths=0.7,
            alpha=0.95,
        )
        axis.add_collection(activated_collection)
        axis.set(xlim=limits[0], ylim=limits[1])
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(f"AU → canonical mesh — {label}")
        figure.colorbar(
            activated_collection,
            ax=axis,
            fraction=0.045,
            pad=0.02,
            label="Displacement from neutral",
        )
        figure.tight_layout()
        yield figure


def frame_labels(data: pd.DataFrame) -> list[str]:
    if "frame" in data:
        return [f"source frame {value}" for value in data["frame"]]
    return [f"CSV row {index}" for index in data.index]


def create_video(
    input_csv: Path,
    output_mp4: Path,
    *,
    fps: float = 30.0,
    stride: int = 1,
    max_frames: int | None = None,
    mode: str = "tesselation",
    vmax: float | None = None,
    dpi: int = 120,
    codec: str = "libx264",
    ffmpeg_bin: Path | None = None,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if vmax is not None and vmax <= 0:
        raise ValueError("vmax must be positive")

    configure_ffmpeg_dlls(resolve_ffmpeg_bin(ffmpeg_bin))
    data = select_rows(pd.read_csv(input_csv), stride, max_frames)
    meshes_3d, neutral_3d, _ = predict_sequence(data)
    meshes_2d = project_mesh_2d(meshes_3d)
    neutral_2d = project_mesh_2d(neutral_3d)
    write_figures_to_video(
        iter_figures(
            meshes_2d,
            neutral_2d,
            mesh_edges(mode),
            frame_labels(data),
            vmax=vmax,
        ),
        output_mp4,
        fps=fps,
        dpi=dpi,
        codec=codec,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    create_video(
        args.input_csv,
        args.output_mp4,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        mode=args.mode,
        vmax=args.vmax,
        dpi=args.dpi,
        codec=args.codec,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    print(f"Saved canonical AU mesh video: {args.output_mp4}")


if __name__ == "__main__":
    main()
