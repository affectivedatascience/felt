"""Apply approved Challis master cutoffs to FELT raw-motion CSV files.

The script reads a versioned ``master_cutoffs.json`` artifact, applies one
corrected zero-phase Butterworth filter to each enabled trajectory group, and
writes smoothed CSVs to a separate output tree. The official FELT v2 profile
filters geometry, action units, and blendshapes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve()
for parent in CODE_ROOT.parents:
    if (parent / "utils").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not find code root containing utils/.")

from utils.challis_smoothing import (  # noqa: E402
    ACTION_UNIT_COLUMNS,
    BLENDSHAPE_COLUMNS,
    GEOMETRIC_COLUMN_GROUPS,
    apply_zero_phase_filter,
    clip_bounded,
    configuration_digest,
    content_manifest_digest,
    load_master_cutoffs,
    manifest_digest,
    path_manifest_digest,
    validate_input_dataframe,
    write_json,
)
from utils.felt_paths import (  # noqa: E402
    LOG_DIR,
    OUTPUT_DIR,
    RAW_MOTION_DIR,
    SMOOTHED_MOTION_DIR,
    VOCAL_CHANNELS,
    configure_logging,
    parse_ravdess_stem,
)

DEFAULT_MASTER_PATH = CODE_ROOT.parents[1] / "config" / "master_cutoffs_v2.json"
DEFAULT_QC_ROOT = OUTPUT_DIR / "qc" / "challis_smoothing"
DEFAULT_LOG_FILE = LOG_DIR / "3_clean_signals.log"
SMOOTHING_CHECKPOINT_SCHEMA_VERSION = 2

CLIPPING_FIELDS = (
    "relative_path",
    "vocal_channel",
    "actor_folder",
    "stem",
    "emotion_code",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "actor_code",
    "family",
    "column",
    "value_count",
    "below_zero_count",
    "above_one_count",
    "below_zero_percent",
    "above_one_percent",
    "minimum_preclip",
    "maximum_preclip",
    "maximum_undershoot",
    "maximum_overshoot",
    "clipped_count",
)

VALIDATION_FIELDS = (
    "relative_path",
    "vocal_channel",
    "actor_folder",
    "stem",
    "emotion_code",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "actor_code",
    "family",
    "column_count",
    "cutoff_hz",
    "rmse_median",
    "nrmse_median",
    "mae_median",
    "correlation_median",
    "variance_ratio_median",
    "first_difference_rms_ratio_median",
    "high_frequency_power_ratio_median",
    "peak_to_peak_ratio_median",
    "edge_rmse_median",
    "interior_rmse_median",
)


@dataclass(frozen=True)
class SmoothingConfig:
    """Immutable settings supplied to each smoothing worker."""

    master_digest: str
    master_semantic_digest: str
    sampling_frequency_hz: float
    order_per_pass: int
    geometric_cutoffs_hz: Mapping[str, float]
    independent_cutoffs_hz: Mapping[str, float]
    filter_action_units: bool
    filter_blendshapes: bool

    @property
    def signature(self) -> str:
        return configuration_digest(
            {
                "checkpoint_schema_version": SMOOTHING_CHECKPOINT_SCHEMA_VERSION,
                "master_semantic_digest": self.master_semantic_digest,
                "sampling_frequency_hz": self.sampling_frequency_hz,
                "order_per_pass": self.order_per_pass,
                "geometric_cutoffs_hz": dict(self.geometric_cutoffs_hz),
                "independent_cutoffs_hz": dict(self.independent_cutoffs_hz),
                "filter_action_units": self.filter_action_units,
                "filter_blendshapes": self.filter_blendshapes,
            }
        )


@dataclass(frozen=True)
class SmoothTask:
    """One raw input and its separate smoothed output/checkpoint paths."""

    input_path: Path
    output_path: Path
    checkpoint_path: Path
    input_root: Path
    config: SmoothingConfig
    overwrite: bool


@dataclass(frozen=True)
class SmoothResult:
    """File-level status returned by a smoothing worker."""

    relative_path: str
    output_path: str
    checkpoint_path: str
    status: str
    elapsed_seconds: float
    error: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_csv_list(input_root: Path) -> list[Path]:
    paths: list[Path] = []
    for channel in VOCAL_CHANNELS:
        channel_dir = input_root / channel
        if not channel_dir.exists():
            logging.warning("Input channel directory is absent: %s", channel_dir)
            continue
        for actor_dir in sorted(channel_dir.glob("Actor_*")):
            if actor_dir.is_dir():
                paths.extend(sorted(actor_dir.glob("*.csv")))
    return paths


def load_file_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    stems: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            value = line.strip()
            if value and not value.startswith("#"):
                stems.add(Path(value).stem)
    return stems


def _metadata(input_path: Path, input_root: Path) -> dict[str, Any]:
    code = parse_ravdess_stem(input_path)
    relative = input_path.relative_to(input_root)
    return {
        "relative_path": relative.as_posix(),
        "vocal_channel": relative.parts[0],
        "actor_folder": input_path.parent.name,
        "stem": input_path.stem,
        "emotion_code": code.emotion,
        "intensity_code": code.intensity,
        "statement_code": code.statement,
        "repetition_code": code.repetition,
        "actor_code": code.actor,
    }


def _requested_independent_columns(config: SmoothingConfig) -> dict[str, str]:
    columns: dict[str, str] = {}
    if config.filter_action_units:
        columns.update({column: "action_unit" for column in ACTION_UNIT_COLUMNS})
    if config.filter_blendshapes:
        columns.update({column: "blendshape" for column in BLENDSHAPE_COLUMNS})
    return columns


def _filter_batches(config: SmoothingConfig) -> dict[float, list[str]]:
    """Group columns sharing an exact cutoff so each matrix is filtered once."""
    batches: dict[float, list[str]] = {}
    for group, columns in GEOMETRIC_COLUMN_GROUPS.items():
        cutoff = float(config.geometric_cutoffs_hz[group])
        batches.setdefault(cutoff, []).extend(columns)
    for column in _requested_independent_columns(config):
        cutoff = float(config.independent_cutoffs_hz[column])
        batches.setdefault(cutoff, []).append(column)
    return batches


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else math.nan


def validation_metrics(
    raw: np.ndarray,
    final: np.ndarray,
    sampling_frequency_hz: float,
    edge_rows: int = 9,
) -> dict[str, float]:
    """Return median trajectory-distortion metrics for one logical family."""
    raw = np.asarray(raw, dtype=float)
    final = np.asarray(final, dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    if final.ndim == 1:
        final = final[:, None]
    difference = final - raw
    rmse = np.sqrt(np.mean(difference * difference, axis=0))
    mae = np.mean(np.abs(difference), axis=0)
    robust_range = np.percentile(raw, 95, axis=0) - np.percentile(raw, 5, axis=0)
    nrmse = np.divide(
        rmse,
        robust_range,
        out=np.full_like(rmse, np.nan),
        where=robust_range > np.finfo(float).eps,
    )

    raw_centered = raw - np.mean(raw, axis=0)
    final_centered = final - np.mean(final, axis=0)
    correlation_denominator = np.sqrt(
        np.sum(raw_centered * raw_centered, axis=0)
        * np.sum(final_centered * final_centered, axis=0)
    )
    correlation = np.divide(
        np.sum(raw_centered * final_centered, axis=0),
        correlation_denominator,
        out=np.full(raw.shape[1], np.nan),
        where=correlation_denominator > np.finfo(float).eps,
    )
    raw_variance = np.var(raw, axis=0)
    variance_ratio = np.divide(
        np.var(final, axis=0),
        raw_variance,
        out=np.full(raw.shape[1], np.nan),
        where=raw_variance > np.finfo(float).eps,
    )

    raw_diff_rms = np.sqrt(np.mean(np.diff(raw, axis=0) ** 2, axis=0))
    final_diff_rms = np.sqrt(np.mean(np.diff(final, axis=0) ** 2, axis=0))
    diff_ratio = np.divide(
        final_diff_rms,
        raw_diff_rms,
        out=np.full(raw.shape[1], np.nan),
        where=raw_diff_rms > np.finfo(float).eps,
    )

    frequencies = np.fft.rfftfreq(raw.shape[0], d=1.0 / sampling_frequency_hz)
    high_mask = frequencies > 10.0
    if high_mask.any():
        raw_power = np.sum(np.abs(np.fft.rfft(raw, axis=0)[high_mask]) ** 2, axis=0)
        final_power = np.sum(
            np.abs(np.fft.rfft(final, axis=0)[high_mask]) ** 2,
            axis=0,
        )
        high_power_ratio = np.divide(
            final_power,
            raw_power,
            out=np.full(raw.shape[1], np.nan),
            where=raw_power > np.finfo(float).eps,
        )
    else:
        high_power_ratio = np.full(raw.shape[1], np.nan)

    raw_peak_to_peak = np.ptp(raw, axis=0)
    peak_ratio = np.divide(
        np.ptp(final, axis=0),
        raw_peak_to_peak,
        out=np.full(raw.shape[1], np.nan),
        where=raw_peak_to_peak > np.finfo(float).eps,
    )

    edge_rows = min(edge_rows, max(1, raw.shape[0] // 4))
    edge_difference = np.concatenate(
        [difference[:edge_rows], difference[-edge_rows:]],
        axis=0,
    )
    edge_rmse = np.sqrt(np.mean(edge_difference * edge_difference, axis=0))
    interior = difference[edge_rows:-edge_rows]
    interior_rmse = (
        np.sqrt(np.mean(interior * interior, axis=0))
        if len(interior)
        else np.full(raw.shape[1], np.nan)
    )

    return {
        "rmse_median": _finite_median(rmse),
        "nrmse_median": _finite_median(nrmse),
        "mae_median": _finite_median(mae),
        "correlation_median": _finite_median(correlation),
        "variance_ratio_median": _finite_median(variance_ratio),
        "first_difference_rms_ratio_median": _finite_median(diff_ratio),
        "high_frequency_power_ratio_median": _finite_median(high_power_ratio),
        "peak_to_peak_ratio_median": _finite_median(peak_ratio),
        "edge_rmse_median": _finite_median(edge_rmse),
        "interior_rmse_median": _finite_median(interior_rmse),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def smooth_one_file(task: SmoothTask) -> SmoothResult:
    """Filter one CSV, write its output, and checkpoint its QC records."""
    started = time.perf_counter()
    relative = task.input_path.relative_to(task.input_root).as_posix()
    try:
        input_sha256 = sha256_file(task.input_path)
        if task.output_path.exists() and not task.overwrite:
            if not task.checkpoint_path.exists():
                raise ValueError(
                    "Output exists without a revised-smoothing checkpoint; use --overwrite."
                )
            checkpoint = _load_checkpoint(task.checkpoint_path)
            if checkpoint.get("config_signature") != task.config.signature:
                raise ValueError(
                    "Output checkpoint uses a different master/configuration; "
                    "use --overwrite or a different output root."
                )
            if checkpoint.get("input_sha256") != input_sha256:
                raise ValueError(
                    "Raw input content changed after the output checkpoint was "
                    "created; use --overwrite."
                )
            if checkpoint.get("output_sha256") != sha256_file(task.output_path):
                raise ValueError(
                    "Smoothed output content no longer matches its checkpoint; "
                    "use --overwrite."
                )
            return SmoothResult(
                relative,
                str(task.output_path),
                str(task.checkpoint_path),
                "checkpoint_reused",
                time.perf_counter() - started,
            )

        dataframe = pd.read_csv(task.input_path, low_memory=False)
        required_columns = [
            column for columns in GEOMETRIC_COLUMN_GROUPS.values() for column in columns
        ] + list(_requested_independent_columns(task.config))
        validate_input_dataframe(dataframe, required_columns)

        raw_groups: dict[str, np.ndarray] = {
            group: dataframe.loc[:, columns].to_numpy(dtype=float, copy=True)
            for group, columns in GEOMETRIC_COLUMN_GROUPS.items()
        }
        requested_independent = _requested_independent_columns(task.config)
        for family in set(requested_independent.values()):
            columns = [
                column
                for column, column_family in requested_independent.items()
                if column_family == family
            ]
            raw_groups[family] = dataframe.loc[:, columns].to_numpy(
                dtype=float,
                copy=True,
            )

        for cutoff_hz, columns in _filter_batches(task.config).items():
            filtered = apply_zero_phase_filter(
                dataframe.loc[:, columns].to_numpy(dtype=float, copy=True),
                cutoff_hz,
                task.config.sampling_frequency_hz,
                task.config.order_per_pass,
            )
            dataframe.loc[:, columns] = filtered

        metadata = _metadata(task.input_path, task.input_root)
        clipping_rows: list[dict[str, Any]] = []
        for column, family in requested_independent.items():
            clipped, stats = clip_bounded(dataframe[column].to_numpy(float))
            dataframe.loc[:, column] = clipped
            clipping_rows.append(
                {
                    **metadata,
                    "family": family,
                    "column": column,
                    "value_count": stats.count,
                    "below_zero_count": stats.below_count,
                    "above_one_count": stats.above_count,
                    "below_zero_percent": 100.0 * stats.below_count / stats.count,
                    "above_one_percent": 100.0 * stats.above_count / stats.count,
                    "minimum_preclip": stats.minimum,
                    "maximum_preclip": stats.maximum,
                    "maximum_undershoot": stats.maximum_undershoot,
                    "maximum_overshoot": stats.maximum_overshoot,
                    "clipped_count": stats.clipped_count,
                }
            )

        validation_rows: list[dict[str, Any]] = []
        logical_groups: dict[str, Sequence[str]] = dict(GEOMETRIC_COLUMN_GROUPS)
        for family in set(requested_independent.values()):
            logical_groups[family] = tuple(
                column
                for column, column_family in requested_independent.items()
                if column_family == family
            )
        for family, columns in logical_groups.items():
            if family in GEOMETRIC_COLUMN_GROUPS:
                cutoff_values = [float(task.config.geometric_cutoffs_hz[family])]
            else:
                cutoff_values = [
                    float(task.config.independent_cutoffs_hz[column])
                    for column in columns
                ]
            validation_rows.append(
                {
                    **metadata,
                    "family": family,
                    "column_count": len(columns),
                    "cutoff_hz": (
                        cutoff_values[0]
                        if len(set(cutoff_values)) == 1
                        else "independent"
                    ),
                    **validation_metrics(
                        raw_groups[family],
                        dataframe.loc[:, columns].to_numpy(dtype=float, copy=False),
                        task.config.sampling_frequency_hz,
                    ),
                }
            )

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = task.output_path.with_name(task.output_path.name + ".tmp")
        dataframe.to_csv(temporary_output, index=False)
        temporary_output.replace(task.output_path)
        output_sha256 = sha256_file(task.output_path)
        _write_checkpoint(
            task.checkpoint_path,
            {
                "checkpoint_schema_version": SMOOTHING_CHECKPOINT_SCHEMA_VERSION,
                "config_signature": task.config.signature,
                "relative_path": relative,
                "input_sha256": input_sha256,
                "output_path": str(task.output_path.resolve()),
                "output_sha256": output_sha256,
                "clipping_rows": clipping_rows,
                "validation_rows": validation_rows,
            },
        )
        return SmoothResult(
            relative,
            str(task.output_path),
            str(task.checkpoint_path),
            "processed",
            time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - report file-level worker failures
        return SmoothResult(
            relative,
            str(task.output_path),
            str(task.checkpoint_path),
            "error",
            time.perf_counter() - started,
            f"{type(exc).__name__}: {exc}",
        )


def run_tasks(tasks: Sequence[SmoothTask], workers: int) -> list[SmoothResult]:
    if workers == 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            result = smooth_one_file(task)
            results.append(result)
            logging.info(
                "Smoothing file %d/%d: %s (%s, %.2fs)",
                index,
                len(tasks),
                result.relative_path,
                result.status,
                result.elapsed_seconds,
            )
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(smooth_one_file, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            logging.info(
                "Smoothing file %d/%d: %s (%s, %.2fs)",
                completed,
                len(tasks),
                result.relative_path,
                result.status,
                result.elapsed_seconds,
            )
    return sorted(results, key=lambda value: value.relative_path)


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_qc(results: Sequence[SmoothResult], qc_root: Path) -> dict[str, int]:
    clipping_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for result in results:
        checkpoint = _load_checkpoint(Path(result.checkpoint_path))
        clipping_rows.extend(checkpoint.get("clipping_rows", []))
        validation_rows.extend(checkpoint.get("validation_rows", []))

    _write_csv(qc_root / "clipping_by_file_column.csv", clipping_rows, CLIPPING_FIELDS)
    _write_csv(qc_root / "smoothing_validation.csv", validation_rows, VALIDATION_FIELDS)
    write_validation_plots(validation_rows, qc_root)

    if clipping_rows:
        clipping = pd.DataFrame(clipping_rows)
        summary_rows = []
        for (family, column), group in clipping.groupby(
            ["family", "column"], sort=True
        ):
            summary_rows.append(
                {
                    "family": family,
                    "column": column,
                    "file_count": len(group),
                    "affected_file_count": int((group["clipped_count"] > 0).sum()),
                    "value_count": int(group["value_count"].sum()),
                    "below_zero_count": int(group["below_zero_count"].sum()),
                    "above_one_count": int(group["above_one_count"].sum()),
                    "clipped_count": int(group["clipped_count"].sum()),
                    "minimum_preclip": float(group["minimum_preclip"].min()),
                    "maximum_preclip": float(group["maximum_preclip"].max()),
                    "maximum_undershoot": float(group["maximum_undershoot"].max()),
                    "maximum_overshoot": float(group["maximum_overshoot"].max()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(qc_root / "clipping_summary.csv", index=False)

        stratified_rows = []
        dimensions = (
            "vocal_channel",
            "actor_code",
            "emotion_code",
            "intensity_code",
        )
        for dimension in dimensions:
            for (family, stratum), group in clipping.groupby(
                ["family", dimension], sort=True
            ):
                stratified_rows.append(
                    {
                        "family": family,
                        "stratifier": dimension,
                        "stratum": str(stratum),
                        "file_column_count": len(group),
                        "affected_file_column_count": int(
                            (group["clipped_count"] > 0).sum()
                        ),
                        "value_count": int(group["value_count"].sum()),
                        "clipped_count": int(group["clipped_count"].sum()),
                        "maximum_undershoot": float(group["maximum_undershoot"].max()),
                        "maximum_overshoot": float(group["maximum_overshoot"].max()),
                    }
                )
        pd.DataFrame(stratified_rows).to_csv(
            qc_root / "clipping_stratified_summary.csv",
            index=False,
        )

        affected_files = len(
            {
                row["relative_path"]
                for row in clipping_rows
                if int(row["clipped_count"]) > 0
            }
        )
        affected_columns = len(
            {row["column"] for row in clipping_rows if int(row["clipped_count"]) > 0}
        )
        clipped_values = sum(int(row["clipped_count"]) for row in clipping_rows)
    else:
        _write_csv(
            qc_root / "clipping_summary.csv",
            [],
            (
                "family",
                "column",
                "file_count",
                "affected_file_count",
                "value_count",
                "below_zero_count",
                "above_one_count",
                "clipped_count",
                "minimum_preclip",
                "maximum_preclip",
                "maximum_undershoot",
                "maximum_overshoot",
            ),
        )
        affected_files = affected_columns = clipped_values = 0

    return {
        "affected_file_count": affected_files,
        "affected_column_count": affected_columns,
        "clipped_value_count": clipped_values,
    }


def write_validation_plots(
    validation_rows: Sequence[Mapping[str, Any]],
    qc_root: Path,
) -> None:
    """Render compact smoothing-distortion plots from per-file QC rows."""
    if not validation_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataframe = pd.DataFrame(validation_rows)
    families = sorted(dataframe["family"].unique())
    plot_root = qc_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("nrmse_median", "Median robust-range NRMSE"),
        (
            "first_difference_rms_ratio_median",
            "Median first-difference RMS ratio",
        ),
    )
    for metric, ylabel in metrics:
        values = [
            pd.to_numeric(
                dataframe.loc[dataframe["family"] == family, metric],
                errors="coerce",
            )
            .dropna()
            .to_numpy(float)
            for family in families
        ]
        if not any(len(value) for value in values):
            continue
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.boxplot(values, tick_labels=families, showfliers=True)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Smoothing validation: {metric}")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / f"{metric}.png", dpi=160)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--output-root", type=Path, default=SMOOTHED_MOTION_DIR)
    parser.add_argument("--master-cutoffs", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--qc-root", type=Path, default=DEFAULT_QC_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--filter-action-units",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter action units (official v2 default).",
    )
    parser.add_argument(
        "--filter-blendshapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter blendshapes (official v2 default).",
    )
    parser.add_argument(
        "--require-calibration-content-match",
        action="store_true",
        help="Require exact approved raw CSV byte content, not only trial inventory.",
    )
    parser.add_argument(
        "--allow-input-manifest-mismatch",
        action="store_true",
        help="Allow smoothing a corpus that differs from the calibration corpus.",
    )
    parser.add_argument(
        "--allow-input-root-mismatch",
        dest="allow_input_manifest_mismatch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")


def build_config(
    master: Mapping[str, Any],
    master_digest: str,
    filter_action_units: bool,
    filter_blendshapes: bool,
) -> SmoothingConfig:
    independent = {
        str(key): float(value)
        for key, value in master.get("independent_cutoffs_hz", {}).items()
    }
    if filter_action_units:
        missing = sorted(set(ACTION_UNIT_COLUMNS) - set(independent))
        if missing:
            raise ValueError(
                "Action-unit filtering was requested but the master artifact is "
                f"missing cutoffs: {missing}"
            )
    if filter_blendshapes:
        missing = sorted(set(BLENDSHAPE_COLUMNS) - set(independent))
        if missing:
            raise ValueError(
                "Blendshape filtering was requested but the master artifact is "
                f"missing cutoffs: {missing}"
            )

    return SmoothingConfig(
        master_digest=master_digest,
        master_semantic_digest=configuration_digest(
            {
                "schema_version": master["schema_version"],
                "procedure_version": master["procedure_version"],
                "filter": master["filter"],
                "geometric_cutoffs_hz": master["geometric_cutoffs_hz"],
                "independent_cutoffs_hz": independent,
            }
        ),
        sampling_frequency_hz=float(master["filter"]["sampling_frequency_hz"]),
        order_per_pass=int(master["filter"]["order_per_pass"]),
        geometric_cutoffs_hz={
            str(key): float(value)
            for key, value in master["geometric_cutoffs_hz"].items()
        },
        independent_cutoffs_hz=independent,
        filter_action_units=filter_action_units,
        filter_blendshapes=filter_blendshapes,
    )


def validate_calibration_corpus(
    master: Mapping[str, Any],
    paths: Sequence[Path],
    input_root: Path,
    *,
    allow_mismatch: bool,
    require_content_match: bool = False,
) -> dict[str, str]:
    """Verify that raw inputs are the path-independent calibration corpus."""
    input_metadata = master.get("input")
    if not isinstance(input_metadata, dict):
        raise ValueError("Master artifact has no input-corpus metadata.")

    expected_count = input_metadata.get("available_file_count")
    expected_paths = input_metadata.get("path_manifest_digest")
    expected_manifest = input_metadata.get("manifest_digest")
    expected_content = input_metadata.get("content_manifest_digest")
    if (
        expected_count is None
        or not expected_paths
        or not expected_manifest
        or not expected_content
    ):
        raise ValueError(
            "Master artifact must record the calibration file count, path "
            "manifest, and both content provenance digests."
        )

    actual_paths = path_manifest_digest(paths, input_root)
    mismatches = []
    if int(expected_count) != len(paths):
        mismatches.append(f"file count {len(paths)} != {expected_count}")
    if actual_paths != str(expected_paths):
        mismatches.append("trial-path manifest digest differs")

    actual_manifest = "not-computed"
    actual_content = "not-computed"
    if require_content_match and not mismatches:
        actual_manifest = manifest_digest(paths, input_root)
        actual_content = content_manifest_digest(paths, input_root)
        if actual_manifest != str(expected_manifest):
            mismatches.append("path/size manifest digest differs")
        if actual_content != str(expected_content):
            mismatches.append("content manifest digest differs")

    if mismatches:
        message = "Calibration corpus mismatch: " + "; ".join(mismatches)
        if not allow_mismatch:
            raise ValueError(
                message
                + ". Use --allow-input-manifest-mismatch only for an intentional "
                "non-release application."
            )
        logging.warning("%s; override accepted.", message)

    return {
        "path_manifest_digest": actual_paths,
        "manifest_digest": actual_manifest,
        "content_manifest_digest": actual_content,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    qc_root = args.qc_root.resolve()
    master_path = args.master_cutoffs.resolve()

    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError(
            "Output root must be separate from and outside the raw input root."
        )

    configure_logging(args.log_file.resolve())
    master = load_master_cutoffs(master_path)
    available = build_csv_list(input_root)
    calibration_manifest = validate_calibration_corpus(
        master,
        available,
        input_root,
        allow_mismatch=args.allow_input_manifest_mismatch,
        require_content_match=args.require_calibration_content_match,
    )
    config = build_config(
        master,
        sha256_file(master_path),
        args.filter_action_units,
        args.filter_blendshapes,
    )

    requested = load_file_list(args.file_list)
    paths = [path for path in available if not requested or path.stem in requested]
    if requested:
        found = {path.stem for path in paths}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested stems were not found: {missing[:10]}")
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No input CSV files found under {input_root}")

    checkpoint_root = qc_root / "smoothing_checkpoints"
    tasks = []
    for input_path in paths:
        relative = input_path.relative_to(input_root)
        checkpoint_name = "__".join(relative.with_suffix("").parts) + ".json"
        tasks.append(
            SmoothTask(
                input_path=input_path,
                output_path=output_root / relative,
                checkpoint_path=checkpoint_root / checkpoint_name,
                input_root=input_root,
                config=config,
                overwrite=args.overwrite,
            )
        )

    logging.info("Revised FELT smoothing starting.")
    logging.info("Input root: %s", input_root)
    logging.info("Output root: %s", output_root)
    logging.info("Master artifact: %s", master_path)
    logging.info("Selected files: %d of %d available", len(paths), len(available))
    logging.info("Smoothing configuration signature: %s", config.signature)
    logging.info("Geometric cutoffs: %s", dict(config.geometric_cutoffs_hz))
    logging.info("Action-unit filtering: %s", config.filter_action_units)
    logging.info("Blendshape filtering: %s", config.filter_blendshapes)

    started = time.perf_counter()
    results = run_tasks(tasks, args.workers)
    errors = [asdict(result) for result in results if result.status == "error"]
    _write_csv(
        qc_root / "smoothing_errors.csv",
        errors,
        tuple(SmoothResult.__dataclass_fields__),
    )
    if errors:
        for error in errors[:10]:
            logging.error(
                "Smoothing failure: %s: %s", error["relative_path"], error["error"]
            )
        raise RuntimeError(f"{len(errors)} files failed during smoothing.")

    clipping_totals = aggregate_qc(results, qc_root)
    elapsed = time.perf_counter() - started
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "master_cutoffs_path": str(master_path),
        "master_digest": config.master_digest,
        "master_semantic_digest": config.master_semantic_digest,
        "configuration_signature": config.signature,
        "selected_file_count": len(paths),
        "available_file_count": len(available),
        "input_manifest_digest": manifest_digest(paths, input_root),
        "calibration_corpus_manifest_digest": calibration_manifest[
            "manifest_digest"
        ],
        "calibration_corpus_path_manifest_digest": calibration_manifest[
            "path_manifest_digest"
        ],
        "calibration_corpus_content_manifest_digest": calibration_manifest[
            "content_manifest_digest"
        ],
        "workers": args.workers,
        "wall_seconds": elapsed,
        "processed_count": sum(result.status == "processed" for result in results),
        "checkpoint_reused_count": sum(
            result.status == "checkpoint_reused" for result in results
        ),
        **clipping_totals,
    }
    write_json(qc_root / "smoothing_run_manifest.json", run_manifest)
    logging.info(
        "Clipping QC: %d files, %d columns, %d values affected.",
        clipping_totals["affected_file_count"],
        clipping_totals["affected_column_count"],
        clipping_totals["clipped_value_count"],
    )
    logging.info("Revised smoothing complete: %d files in %.2fs.", len(paths), elapsed)
    print(f"Smoothing complete. Outputs: {output_root}")
    print(f"QC: {qc_root}")


if __name__ == "__main__":
    main()
