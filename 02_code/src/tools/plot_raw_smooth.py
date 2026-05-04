"""
Plot one FELT raw tracking column against its smoothed equivalent.

This utility loads one raw tracking CSV and the matching smoothed CSV, then
plots the selected column over frames. Paths are resolved dynamically from the
project structure.

Expected project layout
-----------------------
face-tracking-2024/
├── 01_data/
│   └── 02_output/
│       ├── 01_raw_motion/
│       │   ├── speech/
│       │   │   └── Actor_01/
│       │   └── song/
│       ├── 02_smoothed_motion/
│       │   ├── speech/
│       │   │   └── Actor_01/
│       │   └── song/
│       └── plots/
└── 02_code/
    └── src/
        ├── tools/
        │   └── plot_raw_smooth.py
        └── utils/
            ├── __init__.py
            └── felt_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

from utils.felt_paths import PLOT_DIR, RAW_MOTION_DIR, SMOOTHED_MOTION_DIR


# =============================================================================
# User-editable settings
# =============================================================================

FILE_NAME = "01-01-01-01-01-01-01.csv"
VOCAL_CHANNEL = "speech"      # "speech" or "song"
ACTOR_NAME = "Actor_01"
COLUMN_NAME = "x_0"


# =============================================================================
# Paths
# =============================================================================

RAW_CSV = RAW_MOTION_DIR / VOCAL_CHANNEL / ACTOR_NAME / FILE_NAME
SMOOTHED_CSV = SMOOTHED_MOTION_DIR / VOCAL_CHANNEL / ACTOR_NAME / FILE_NAME
OUTPUT_PNG = PLOT_DIR / f"raw_vs_smoothed_{FILE_NAME.replace('.csv', '')}_{COLUMN_NAME}.png"


# =============================================================================
# Plotting
# =============================================================================

def load_column(csv_path: Path, column_name: str, label: str) -> pd.DataFrame:
    """Load one numeric column from a FELT CSV file for plotting."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available first columns: {df.columns[:25].tolist()}"
        )

    if "frame" in df.columns:
        frame = pd.to_numeric(df["frame"], errors="coerce")
    else:
        frame = pd.Series(range(len(df)), name="frame")

    value = pd.to_numeric(df[column_name], errors="coerce")

    plot_df = pd.DataFrame(
        {
            "frame": frame,
            "value": value,
            "series": label,
        }
    )

    return plot_df.dropna(subset=["frame", "value"])


def main() -> None:
    """Plot one raw column against its smoothed equivalent."""
    raw = load_column(RAW_CSV, COLUMN_NAME, "Raw")
    smoothed = load_column(SMOOTHED_CSV, COLUMN_NAME, "Smoothed")

    plot_df = pd.concat([raw, smoothed], ignore_index=True)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="frame",
        y="value",
        hue="series",
        palette={"Raw": "blue", "Smoothed": "red"},
        linewidth=1.5,
    )

    ax.set_title(f"{COLUMN_NAME}: raw vs smoothed\n{FILE_NAME}")
    ax.set_xlabel("Frame")
    ax.set_ylabel(COLUMN_NAME)
    ax.legend(title="")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()

    print(f"Raw file:      {RAW_CSV}")
    print(f"Smoothed file: {SMOOTHED_CSV}")
    print(f"Saved plot:    {OUTPUT_PNG}")


if __name__ == "__main__":
    main()