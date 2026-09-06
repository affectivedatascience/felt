"""Estimate FELT master smoothing cutoffs with the Challis procedure.

The tool processes raw-motion CSVs one file at a time, checkpoints per-column
cutoff estimates, performs within-video and across-video median pooling, and
writes a versioned master-cutoff JSON plus detailed QC tables.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import logging
import math
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    GEOMETRIC_COLUMN_GROUPS,
    MASTER_SCHEMA_VERSION,
    PROCEDURE_VERSION,
    build_cutoff_grid,
    configuration_digest,
    estimate_cutoffs_matrix,
    manifest_digest,
    median_summary,
    selected_column_groups,
    validate_input_dataframe,
    write_json,
)
from utils.felt_paths import (  # noqa: E402
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_MOTION_DIR,
    VOCAL_CHANNELS,
    configure_logging,
    parse_ravdess_stem,
)


DEFAULT_QC_ROOT = OUTPUT_DIR / "qc" / "challis_smoothing"
DEFAULT_LOG_FILE = OUTPUT_DIR / "logs" / "challis_cutoff_estimation.log"
CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_MIN_VALID_FRACTION = 0.5

COLUMN_RESULT_FIELDS = (
    "checkpoint_schema_version",
    "config_signature",
    "input_sha256",
    "relative_path",
    "csv_path",
    "vocal_channel",
    "actor_folder",
    "stem",
    "modality_code",
    "vocal_channel_code",
    "emotion_code",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "actor_code",
    "column",
    "family",
    "pool_key",
    "selected_cutoff_hz",
    "criterion_a",
    "valid",
    "reason",
)

VIDEO_RESULT_FIELDS = (
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
    "pool_key",
    "video_cutoff_hz",
    "valid",
    "reason",
    "valid_column_count",
    "total_column_count",
    "invalid_column_count",
    "boundary_column_count",
)


@dataclass(frozen=True)
class EstimatorConfig:
    """Numerical settings shared with estimator workers."""

    sampling_frequency_hz: float
    order_per_pass: int
    max_lag: int
    candidate_cutoffs_hz: tuple[float, ...]
    include_action_units: bool
    include_blendshapes: bool
    reject_boundaries: bool
    min_valid_fraction: float

    def signature_payload(self) -> dict[str, Any]:
        """Return only settings that affect per-file checkpoint contents."""
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "procedure_version": PROCEDURE_VERSION,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "order_per_pass": self.order_per_pass,
            "max_lag": self.max_lag,
            "candidate_cutoffs_hz": list(self.candidate_cutoffs_hz),
            "include_action_units": self.include_action_units,
            "include_blendshapes": self.include_blendshapes,
            "reject_boundaries": self.reject_boundaries,
        }

    @property
    def signature(self) -> str:
        return configuration_digest(self.signature_payload())


@dataclass(frozen=True)
class ProcessResult:
    """Status returned by one file-level estimator worker."""

    relative_path: str
    checkpoint_path: str
    status: str
    elapsed_seconds: float
    error: str = ""


def build_csv_list(input_root: Path) -> list[Path]:
    """Return raw-motion CSVs in deterministic channel/actor/file order."""
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
    """Load requested filename stems from a plain-text file list."""
    if path is None:
        return set()
    requested: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            value = line.strip()
            if value and not value.startswith("#"):
                requested.add(Path(value).stem)
    return requested


def checkpoint_path_for(
    csv_path: Path, input_root: Path, checkpoint_root: Path
) -> Path:
    """Return a collision-resistant checkpoint path for one input CSV."""
    relative = csv_path.relative_to(input_root)
    name = "__".join(relative.with_suffix("").parts) + ".csv.gz"
    return checkpoint_root / name


def checkpoint_identity(path: Path) -> tuple[str, str]:
    """Read a checkpoint's configuration and input-content signatures."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        first = next(csv.DictReader(file), None)
    if first is None:
        raise ValueError(f"Checkpoint is empty: {path}")
    return first.get("config_signature", ""), first.get("input_sha256", "")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_column_specs(
    include_action_units: bool,
    include_blendshapes: bool,
) -> list[tuple[str, str, str]]:
    """Return ``(column, family, pool_key)`` specifications in stable order."""
    groups, independent = selected_column_groups(
        include_action_units,
        include_blendshapes,
    )
    specs: list[tuple[str, str, str]] = []
    for group, columns in groups.items():
        specs.extend((column, group, group) for column in columns)
    specs.extend((column, family, column) for column, family in independent.items())
    return specs


def _metadata(csv_path: Path, input_root: Path) -> dict[str, Any]:
    code = parse_ravdess_stem(csv_path)
    relative = csv_path.relative_to(input_root)
    return {
        "relative_path": relative.as_posix(),
        "csv_path": str(csv_path.resolve()),
        "vocal_channel": relative.parts[0],
        "actor_folder": csv_path.parent.name,
        "stem": csv_path.stem,
        "modality_code": code.modality,
        "vocal_channel_code": code.vocal_channel,
        "emotion_code": code.emotion,
        "intensity_code": code.intensity,
        "statement_code": code.statement,
        "repetition_code": code.repetition,
        "actor_code": code.actor,
    }


def estimate_one_file(
    csv_path: Path,
    input_root: Path,
    checkpoint_root: Path,
    config: EstimatorConfig,
    overwrite_checkpoint: bool,
) -> ProcessResult:
    """Estimate and checkpoint all requested column cutoffs for one CSV."""
    started = time.perf_counter()
    relative = csv_path.relative_to(input_root).as_posix()
    checkpoint = checkpoint_path_for(csv_path, input_root, checkpoint_root)
    try:
        input_sha256 = sha256_file(csv_path)
        if checkpoint.exists() and not overwrite_checkpoint:
            existing_signature, checkpoint_input_sha256 = checkpoint_identity(
                checkpoint
            )
            if existing_signature != config.signature:
                raise ValueError(
                    "Existing checkpoint uses different estimator settings; "
                    "use --overwrite-checkpoints or a different QC root."
                )
            if checkpoint_input_sha256 != input_sha256:
                raise ValueError(
                    "Input CSV content changed after this checkpoint was created; "
                    "use --overwrite-checkpoints."
                )
            return ProcessResult(
                relative,
                str(checkpoint),
                "checkpoint_reused",
                time.perf_counter() - started,
            )

        specs = ordered_column_specs(
            config.include_action_units,
            config.include_blendshapes,
        )
        columns = tuple(spec[0] for spec in specs)
        required = {"frame", "FaceScore", *columns}
        dataframe = pd.read_csv(
            csv_path,
            usecols=lambda column: column in required,
            low_memory=False,
        )
        validate_input_dataframe(dataframe, columns)
        values = dataframe.loc[:, columns].to_numpy(dtype=float, copy=True)
        estimates = estimate_cutoffs_matrix(
            values,
            config.candidate_cutoffs_hz,
            config.sampling_frequency_hz,
            config.order_per_pass,
            config.max_lag,
            reject_boundaries=config.reject_boundaries,
        )

        metadata = _metadata(csv_path, input_root)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMN_RESULT_FIELDS)
            writer.writeheader()
            for index, (column, family, pool_key) in enumerate(specs):
                selected = estimates.selected_cutoffs_hz[index]
                score = estimates.selected_scores[index]
                writer.writerow(
                    {
                        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "config_signature": config.signature,
                        "input_sha256": input_sha256,
                        **metadata,
                        "column": column,
                        "family": family,
                        "pool_key": pool_key,
                        "selected_cutoff_hz": (
                            f"{selected:.12g}" if np.isfinite(selected) else ""
                        ),
                        "criterion_a": f"{score:.12g}" if np.isfinite(score) else "",
                        "valid": bool(estimates.valid[index]),
                        "reason": estimates.reasons[index],
                    }
                )
        temporary.replace(checkpoint)
        return ProcessResult(
            relative,
            str(checkpoint),
            "processed",
            time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - worker must report file-level failure
        return ProcessResult(
            relative,
            str(checkpoint),
            "error",
            time.perf_counter() - started,
            f"{type(exc).__name__}: {exc}",
        )


def run_estimation(
    paths: Sequence[Path],
    input_root: Path,
    checkpoint_root: Path,
    config: EstimatorConfig,
    overwrite_checkpoints: bool,
    workers: int,
) -> list[ProcessResult]:
    """Run file-level estimation serially or in a process pool."""
    if workers == 1:
        results = []
        for index, path in enumerate(paths, start=1):
            result = estimate_one_file(
                path,
                input_root,
                checkpoint_root,
                config,
                overwrite_checkpoints,
            )
            results.append(result)
            logging.info(
                "Estimator file %d/%d: %s (%s, %.2fs)",
                index,
                len(paths),
                result.relative_path,
                result.status,
                result.elapsed_seconds,
            )
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                estimate_one_file,
                path,
                input_root,
                checkpoint_root,
                config,
                overwrite_checkpoints,
            ): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            logging.info(
                "Estimator file %d/%d: %s (%s, %.2fs)",
                completed,
                len(paths),
                result.relative_path,
                result.status,
                result.elapsed_seconds,
            )
    return sorted(results, key=lambda item: item.relative_path)


def _read_checkpoint(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def pool_one_video(
    rows: Sequence[dict[str, str]],
    min_valid_fraction: float,
) -> list[dict[str, Any]]:
    """Pool column cutoffs within one video."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["pool_key"], []).append(row)

    pooled: list[dict[str, Any]] = []
    for pool_key, group_rows in grouped.items():
        first = group_rows[0]
        valid_rows = [
            row
            for row in group_rows
            if _is_true(row["valid"]) and row["selected_cutoff_hz"]
        ]
        total = len(group_rows)
        required = max(1, math.ceil(total * min_valid_fraction))
        boundary_count = sum(row["reason"] == "search_boundary" for row in group_rows)
        enough = len(valid_rows) >= required
        cutoff = (
            float(np.median([float(row["selected_cutoff_hz"]) for row in valid_rows]))
            if enough
            else math.nan
        )
        pooled.append(
            {
                "relative_path": first["relative_path"],
                "vocal_channel": first["vocal_channel"],
                "actor_folder": first["actor_folder"],
                "stem": first["stem"],
                "emotion_code": first["emotion_code"],
                "intensity_code": first["intensity_code"],
                "statement_code": first["statement_code"],
                "repetition_code": first["repetition_code"],
                "actor_code": int(first["actor_code"]),
                "family": first["family"],
                "pool_key": pool_key,
                "video_cutoff_hz": cutoff,
                "valid": enough,
                "reason": "" if enough else "insufficient_valid_columns",
                "valid_column_count": len(valid_rows),
                "total_column_count": total,
                "invalid_column_count": total - len(valid_rows),
                "boundary_column_count": boundary_count,
            }
        )
    return pooled


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_checkpoints(
    checkpoint_paths: Sequence[Path],
    qc_root: Path,
    config: EstimatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble detailed output and within-video pooled results."""
    detailed_path = qc_root / "column_cutoff_estimates.csv.gz"
    invalid_path = qc_root / "invalid_estimates.csv"
    detailed_path.parent.mkdir(parents=True, exist_ok=True)

    video_rows: list[dict[str, Any]] = []
    with (
        gzip.open(detailed_path, "wt", encoding="utf-8", newline="") as detail_file,
        invalid_path.open("w", encoding="utf-8", newline="") as invalid_file,
    ):
        detail_writer = csv.DictWriter(detail_file, fieldnames=COLUMN_RESULT_FIELDS)
        invalid_writer = csv.DictWriter(invalid_file, fieldnames=COLUMN_RESULT_FIELDS)
        detail_writer.writeheader()
        invalid_writer.writeheader()

        for checkpoint in checkpoint_paths:
            rows = _read_checkpoint(checkpoint)
            if not rows:
                raise ValueError(f"Checkpoint contains no rows: {checkpoint}")
            if rows[0]["config_signature"] != config.signature:
                raise ValueError(f"Checkpoint configuration mismatch: {checkpoint}")
            detail_writer.writerows(rows)
            invalid_writer.writerows(row for row in rows if not _is_true(row["valid"]))
            video_rows.extend(pool_one_video(rows, config.min_valid_fraction))

    _write_csv(qc_root / "video_cutoff_estimates.csv", video_rows, VIDEO_RESULT_FIELDS)
    video_dataframe = pd.DataFrame(video_rows)
    if video_dataframe.empty:
        raise ValueError("No video-level cutoff estimates were produced.")

    overall_rows: list[dict[str, Any]] = []
    for pool_key, group in video_dataframe.groupby("pool_key", sort=True):
        valid = group.loc[group["valid"], "video_cutoff_hz"].to_numpy(float)
        summary = median_summary(valid)
        overall_rows.append(
            {
                "pool_key": pool_key,
                "family": group.iloc[0]["family"],
                "video_count": len(group),
                "valid_video_count": int(group["valid"].sum()),
                "invalid_video_count": int((~group["valid"]).sum()),
                "boundary_column_count": int(group["boundary_column_count"].sum()),
                **summary,
            }
        )
    summary_dataframe = pd.DataFrame(overall_rows)
    summary_dataframe.to_csv(qc_root / "cutoff_summary.csv", index=False)
    write_stratified_summaries(video_dataframe, qc_root)
    write_diagnostic_plots(video_dataframe, summary_dataframe, qc_root)
    return video_dataframe, summary_dataframe


def write_stratified_summaries(video_dataframe: pd.DataFrame, qc_root: Path) -> None:
    """Write video-level cutoff summaries across RAVDESS factors."""
    rows: list[dict[str, Any]] = []
    dimensions = (
        "vocal_channel",
        "actor_code",
        "emotion_code",
        "intensity_code",
        "statement_code",
        "repetition_code",
    )
    for dimension in dimensions:
        for (pool_key, stratum), group in video_dataframe.groupby(
            ["pool_key", dimension], sort=True
        ):
            valid = group.loc[group["valid"], "video_cutoff_hz"].to_numpy(float)
            rows.append(
                {
                    "pool_key": pool_key,
                    "family": group.iloc[0]["family"],
                    "stratifier": dimension,
                    "stratum": str(stratum),
                    "video_count": len(group),
                    "valid_video_count": int(group["valid"].sum()),
                    "invalid_video_count": int((~group["valid"]).sum()),
                    **median_summary(valid),
                }
            )
    pd.DataFrame(rows).to_csv(qc_root / "cutoff_stratified_summary.csv", index=False)


def write_diagnostic_plots(
    video_dataframe: pd.DataFrame,
    summary_dataframe: pd.DataFrame,
    qc_root: Path,
) -> None:
    """Render compact cutoff-distribution and estimator-QC plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = qc_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    geometric = video_dataframe[
        video_dataframe["pool_key"].isin(GEOMETRIC_COLUMN_GROUPS)
        & video_dataframe["valid"]
    ]
    ordered_groups = [
        group
        for group in GEOMETRIC_COLUMN_GROUPS
        if group in set(geometric["pool_key"])
    ]
    if ordered_groups:
        values = [
            geometric.loc[
                geometric["pool_key"] == group,
                "video_cutoff_hz",
            ].to_numpy(float)
            for group in ordered_groups
        ]
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.boxplot(values, tick_labels=ordered_groups, showfliers=True)
        axis.set_ylabel("Selected final cutoff (Hz)")
        axis.set_title("Valid video-level Challis cutoffs by geometric group")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / "geometric_video_cutoff_boxplot.png", dpi=160)
        plt.close(figure)

    if not summary_dataframe.empty:
        labels = summary_dataframe["pool_key"].astype(str).tolist()
        invalid = summary_dataframe["invalid_video_count"].to_numpy(int)
        boundaries = summary_dataframe["boundary_column_count"].to_numpy(int)
        if len(labels) > 25:
            order = np.argsort(-(invalid + boundaries))[:25]
            labels = [labels[index] for index in order]
            invalid = invalid[order]
            boundaries = boundaries[order]
        positions = np.arange(len(labels))
        figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.35), 5))
        axis.bar(positions, invalid, label="Invalid video estimates")
        axis.bar(
            positions,
            boundaries,
            bottom=invalid,
            label="Boundary column estimates",
        )
        axis.set_xticks(positions, labels, rotation=60, ha="right")
        axis.set_ylabel("Count")
        axis.set_title("Challis estimation QC (top 25 pools when needed)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / "estimation_invalid_boundary_counts.png", dpi=160)
        plt.close(figure)


def git_provenance() -> dict[str, Any]:
    """Return best-effort repository revision and dirty-state metadata."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_revision": revision, "git_worktree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_revision": None, "git_worktree_dirty": None}


def checkpoint_content_manifest_digest(checkpoint_paths: Sequence[Path]) -> str:
    """Hash ordered input relative paths and their content digests."""
    digest = hashlib.sha256()
    identities: list[tuple[str, str]] = []
    for checkpoint in checkpoint_paths:
        rows = _read_checkpoint(checkpoint)
        if not rows:
            raise ValueError(f"Checkpoint contains no rows: {checkpoint}")
        identities.append((rows[0]["relative_path"], rows[0]["input_sha256"]))
    for relative_path, input_sha256 in sorted(identities):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(input_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_master_artifact(
    paths: Sequence[Path],
    all_available_count: int,
    input_root: Path,
    qc_root: Path,
    config: EstimatorConfig,
    summary_dataframe: pd.DataFrame,
    corpus_type: str,
    content_manifest_digest: str | None = None,
) -> dict[str, Any]:
    """Create and validate the across-video master cutoff mapping."""
    geometric: dict[str, float] = {}
    independent: dict[str, float] = {}
    unavailable_independent: dict[str, dict[str, Any]] = {}
    contribution: dict[str, dict[str, int]] = {}

    for row in summary_dataframe.to_dict(orient="records"):
        video_count = int(row["video_count"])
        valid_count = int(row["valid_video_count"])
        required = max(1, math.ceil(video_count * config.min_valid_fraction))
        pool_key = str(row["pool_key"])
        contribution[pool_key] = {
            "video_count": video_count,
            "valid_video_count": valid_count,
            "invalid_video_count": int(row["invalid_video_count"]),
            "boundary_column_count": int(row["boundary_column_count"]),
        }
        if valid_count < required or not np.isfinite(row["median"]):
            if pool_key in GEOMETRIC_COLUMN_GROUPS:
                raise ValueError(
                    f"Geometric pool {pool_key} has {valid_count}/{video_count} "
                    f"valid video estimates; at least {required} are required."
                )
            unavailable_independent[pool_key] = {
                "reason": "insufficient_valid_videos",
                "video_count": video_count,
                "valid_video_count": valid_count,
                "required_valid_video_count": required,
            }
            continue
        if pool_key in GEOMETRIC_COLUMN_GROUPS:
            geometric[pool_key] = float(row["median"])
        else:
            independent[pool_key] = float(row["median"])

    missing_groups = sorted(set(GEOMETRIC_COLUMN_GROUPS) - set(geometric))
    if missing_groups:
        raise ValueError(
            f"Cannot create master artifact; missing groups: {missing_groups}"
        )

    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "procedure_version": PROCEDURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "root": str(input_root.resolve()),
            "corpus_type": corpus_type,
            "selected_file_count": len(paths),
            "available_file_count": all_available_count,
            "manifest_digest_algorithm": "sha256(relative_path,NUL,size,newline)",
            "manifest_digest": manifest_digest(paths, input_root),
            "content_manifest_digest_algorithm": (
                "sha256(relative_path,NUL,file_sha256,newline)"
            ),
            "content_manifest_digest": content_manifest_digest,
        },
        "filter": {
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sampling_frequency_source": "configured_ravdess_nominal_rate",
            "order_per_pass": config.order_per_pass,
            "application": "scipy.signal.sosfiltfilt",
            "effective_order": config.order_per_pass * 2,
            "reported_cutoff_convention": "final_zero_phase_minus_3db_hz",
            "single_pass_cutoff_corrected": True,
        },
        "estimation": {
            "criterion": "sum_squared_residual_autocorrelation",
            "max_lag": config.max_lag,
            "candidate_cutoffs_hz": list(config.candidate_cutoffs_hz),
            "reject_search_boundaries": config.reject_boundaries,
            "min_valid_fraction": config.min_valid_fraction,
            "pooling": "median_within_video_then_median_across_videos",
            "config_signature": config.signature,
        },
        "optional_families": {
            "action_units_enabled": config.include_action_units,
            "blendshapes_enabled": config.include_blendshapes,
        },
        "geometric_cutoffs_hz": geometric,
        "independent_cutoffs_hz": independent,
        "unavailable_independent_cutoffs": unavailable_independent,
        "contributing_counts": contribution,
        "qc_files": {
            "column_estimates": "column_cutoff_estimates.csv.gz",
            "video_estimates": "video_cutoff_estimates.csv",
            "summary": "cutoff_summary.csv",
            "stratified_summary": "cutoff_stratified_summary.csv",
            "invalid_estimates": "invalid_estimates.csv",
        },
        **git_provenance(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=RAW_MOTION_DIR)
    parser.add_argument("--qc-root", type=Path, default=DEFAULT_QC_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-action-units", action="store_true")
    parser.add_argument("--include-blendshapes", action="store_true")
    parser.add_argument("--sampling-frequency", type=float, default=29.97)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--cutoff-min", type=float, default=1.0)
    parser.add_argument("--cutoff-max", type=float, default=12.0)
    parser.add_argument("--cutoff-step", type=float, default=0.25)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--allow-boundary-estimates", action="store_true")
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument(
        "--corpus-type",
        choices=("auto", "full", "subset"),
        default="auto",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.filter_order < 1:
        raise ValueError("--filter-order must be at least 1.")
    if args.max_lag < 1:
        raise ValueError("--max-lag must be at least 1.")
    if not 0 < args.min_valid_fraction <= 1:
        raise ValueError("--min-valid-fraction must lie in (0, 1].")
    if args.benchmark_only and args.limit is None:
        args.limit = 10


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = args.input_root.resolve()
    qc_root = args.qc_root.resolve()
    configure_logging(args.log_file.resolve())

    candidates = build_cutoff_grid(args.cutoff_min, args.cutoff_max, args.cutoff_step)
    if candidates[-1] >= args.sampling_frequency / 2.0:
        raise ValueError("--cutoff-max must be below Nyquist.")
    config = EstimatorConfig(
        sampling_frequency_hz=args.sampling_frequency,
        order_per_pass=args.filter_order,
        max_lag=args.max_lag,
        candidate_cutoffs_hz=tuple(float(value) for value in candidates),
        include_action_units=args.include_action_units,
        include_blendshapes=args.include_blendshapes,
        reject_boundaries=not args.allow_boundary_estimates,
        min_valid_fraction=args.min_valid_fraction,
    )

    available = build_csv_list(input_root)
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
        raise ValueError(f"No raw-motion CSV files found under {input_root}")

    corpus_type = args.corpus_type
    if corpus_type == "auto":
        corpus_type = "subset" if "subset" in input_root.name.lower() else "full"

    logging.info("Challis cutoff estimation starting.")
    logging.info("Input root: %s", input_root)
    logging.info("QC root: %s", qc_root)
    logging.info("Selected files: %d of %d available", len(paths), len(available))
    logging.info("Estimator config: %s", asdict(config))
    logging.info("Configuration signature: %s", config.signature)

    qc_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = qc_root / "checkpoints"
    started = time.perf_counter()
    tracemalloc.start()
    results = run_estimation(
        paths,
        input_root,
        checkpoint_root,
        config,
        args.overwrite_checkpoints,
        args.workers,
    )
    elapsed_estimation = time.perf_counter() - started
    _, coordinator_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    error_rows = [asdict(result) for result in results if result.status == "error"]
    _write_csv(
        qc_root / "estimation_errors.csv",
        error_rows,
        tuple(ProcessResult.__dataclass_fields__),
    )
    if error_rows:
        for row in error_rows[:10]:
            logging.error(
                "Estimator failure: %s: %s", row["relative_path"], row["error"]
            )
        raise RuntimeError(
            f"{len(error_rows)} input files failed. No master cutoff was created."
        )

    checkpoint_paths = [Path(result.checkpoint_path) for result in results]
    video_dataframe, summary_dataframe = aggregate_checkpoints(
        checkpoint_paths,
        qc_root,
        config,
    )
    total_elapsed = time.perf_counter() - started
    fresh_results = [result for result in results if result.status == "processed"]
    processed_rate = (
        len(fresh_results) / elapsed_estimation
        if len(fresh_results) == len(results) and elapsed_estimation > 0
        else None
    )
    projected_full_seconds = (
        len(available) / processed_rate if processed_rate is not None else None
    )
    benchmark = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": args.benchmark_only,
        "input_root": str(input_root),
        "selected_file_count": len(paths),
        "available_file_count": len(available),
        "trajectory_count_per_file": len(
            ordered_column_specs(
                config.include_action_units,
                config.include_blendshapes,
            )
        ),
        "candidate_frequency_count": len(config.candidate_cutoffs_hz),
        "workers": args.workers,
        "estimation_wall_seconds": elapsed_estimation,
        "fresh_file_count": len(fresh_results),
        "checkpoint_reused_count": sum(
            result.status == "checkpoint_reused" for result in results
        ),
        "total_wall_seconds": total_elapsed,
        "files_per_second": processed_rate,
        "projected_available_corpus_seconds": projected_full_seconds,
        "coordinator_python_peak_memory_bytes": coordinator_peak_bytes,
        "memory_note": (
            "tracemalloc reports coordinator Python allocations only; worker and "
            "native NumPy/SciPy allocations are not included"
        ),
        "config_signature": config.signature,
        "invalid_video_estimate_count": int((~video_dataframe["valid"]).sum()),
        "boundary_column_count": int(video_dataframe["boundary_column_count"].sum()),
    }
    write_json(qc_root / "benchmark.json", benchmark)

    if not args.benchmark_only:
        master = build_master_artifact(
            paths,
            len(available),
            input_root,
            qc_root,
            config,
            summary_dataframe,
            corpus_type,
            checkpoint_content_manifest_digest(checkpoint_paths),
        )
        write_json(qc_root / "master_cutoffs.json", master)
        logging.info("Master cutoff artifact: %s", qc_root / "master_cutoffs.json")
    else:
        logging.info("Benchmark-only mode: master_cutoffs.json was not written.")

    if projected_full_seconds is None:
        logging.info(
            "Challis estimation complete: %d files in %.2fs; all checkpoints "
            "were reused, so no fresh runtime projection was calculated.",
            len(paths),
            total_elapsed,
        )
    else:
        logging.info(
            "Challis estimation complete: %d files in %.2fs (projected %.2fs for %d).",
            len(paths),
            total_elapsed,
            projected_full_seconds,
            len(available),
        )
    print(f"Challis estimation complete. QC written to: {qc_root}")


if __name__ == "__main__":
    main()
