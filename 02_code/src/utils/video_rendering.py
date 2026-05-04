"""
Shared video-rendering helpers for FELT visualization scripts.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def write_figures_to_video(
    figures: Iterable,
    output_path: Path,
    fps: int = 30,
    dpi: int = 100,
    codec: str = "libx264",
    set_size: tuple[int, int] | None = None,
) -> None:
    """Write a sequence of matplotlib figures to an H.264 mp4 video.

    Parameters
    ----------
    figures
        Iterable of matplotlib figure objects.
    output_path
        Output .mp4 path.
    fps
        Output video frame rate.
    dpi
        Figure rendering DPI.
    codec
        Video codec passed to imageio.
    set_size
        Optional `(width_px, height_px)` used to resize figures before rendering.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=codec,
        format="FFMPEG",
        macro_block_size=None,
    )

    try:
        for fig in figures:
            buf = io.BytesIO()

            try:
                if set_size is not None:
                    width_px, height_px = set_size
                    fig.set_size_inches(width_px / dpi, height_px / dpi)

                fig.savefig(buf, format="png", dpi=dpi)
                buf.seek(0)

                image = imageio.imread(buf)
                writer.append_data(image)

            finally:
                buf.close()
                plt.close(fig)

    finally:
        writer.close()