"""Exploratory plotting of all FELT action-unit time series from one CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


AU_COLUMNS = [
    "AU01", "AU02", "AU04", "AU05",
    "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15",
    "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 5-by-4 panel plot of the 20 FELT action units."
    )
    parser.add_argument("input_csv", type=Path, help="Raw or smoothed FELT CSV.")
    parser.add_argument("output_png", type=Path, help="Destination PNG path.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data = pd.read_csv(args.input_csv)

    missing = [column for column in AU_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"CSV is missing AU columns: {missing}")

    frames = data.index.to_numpy()
    figure, axes = plt.subplots(5, 4, figsize=(20, 10))

    for axis, column in zip(axes.flat, AU_COLUMNS, strict=True):
        values = pd.to_numeric(data[column], errors="coerce")
        axis.plot(
            frames,
            values,
            linestyle="--",
            marker="o",
            linewidth=1.5,
            markersize=3,
        )
        axis.set_title(column, fontsize=10)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output_png, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved AU plot: {args.output_png}")


if __name__ == "__main__":
    main()
