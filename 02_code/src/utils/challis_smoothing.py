"""Numerical and schema utilities for Challis-based FELT smoothing.

This module intentionally does not import Py-Feat. Cutoff estimation and final
smoothing operate on already-extracted CSV files and require only NumPy, pandas,
and SciPy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

PROCEDURE_VERSION = "1.0"
MASTER_SCHEMA_VERSION = 1
EXPECTED_SAMPLING_FREQUENCY_HZ = 29.97
DEFAULT_FILTER_ORDER_PER_PASS = 2
DEFAULT_MAX_LAG = 10
DEFAULT_STATIONARY_ATOL = 1e-12
DEFAULT_STATIONARY_RTOL = 1e-8

FACE_BOX_COLUMNS = (
    "FaceRectX",
    "FaceRectY",
    "FaceRectWidth",
    "FaceRectHeight",
)
LANDMARK_COLUMNS = tuple(
    [f"x_{index}" for index in range(68)] + [f"y_{index}" for index in range(68)]
)
HEAD_ROTATION_COLUMNS = ("Pitch", "Roll", "Yaw")
HEAD_TRANSLATION_COLUMNS = ("X", "Y", "Z")
GAZE_COLUMNS = ("gaze_pitch", "gaze_yaw", "gaze_angle")
MESH_COLUMNS = tuple(
    f"mesh_{axis}_{index}" for axis in ("x", "y", "z") for index in range(478)
)

GEOMETRIC_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "face_mesh": MESH_COLUMNS,
    "landmarks_68": LANDMARK_COLUMNS,
    "head_rotation": HEAD_ROTATION_COLUMNS,
    "head_translation": HEAD_TRANSLATION_COLUMNS,
    "eye_gaze": GAZE_COLUMNS,
    "face_box": FACE_BOX_COLUMNS,
}

ACTION_UNIT_COLUMNS = (
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU11",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU24",
    "AU25",
    "AU26",
    "AU28",
    "AU43",
)

# Py-Feat 2.0.3 feat.utils.MP_BLENDSHAPE_NAMES. Keeping this local avoids
# importing Py-Feat (and therefore its video-decoder runtime) during smoothing.
BLENDSHAPE_COLUMNS = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)


class IntegrityError(ValueError):
    """Raised when a raw-motion CSV violates required integrity conditions."""


class MasterCutoffError(ValueError):
    """Raised when a master-cutoff artifact is missing or incompatible."""


@dataclass(frozen=True)
class CutoffEstimateResult:
    """Per-column results from one candidate-cutoff search."""

    selected_cutoffs_hz: np.ndarray
    selected_scores: np.ndarray
    valid: np.ndarray
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ClippingStats:
    """Pre-clipping bounds diagnostics for one numeric trajectory."""

    count: int
    below_count: int
    above_count: int
    minimum: float
    maximum: float
    maximum_undershoot: float
    maximum_overshoot: float

    @property
    def clipped_count(self) -> int:
        return self.below_count + self.above_count


def all_geometric_columns() -> tuple[str, ...]:
    """Return geometric columns in stable group order."""
    return tuple(
        column for columns in GEOMETRIC_COLUMN_GROUPS.values() for column in columns
    )


def selected_column_groups(
    include_action_units: bool = False,
    include_blendshapes: bool = False,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return pooled geometric groups and independent optional columns.

    The independent mapping is ``column -> family``.
    """
    independent: dict[str, str] = {}
    if include_action_units:
        independent.update({column: "action_unit" for column in ACTION_UNIT_COLUMNS})
    if include_blendshapes:
        independent.update({column: "blendshape" for column in BLENDSHAPE_COLUMNS})
    return dict(GEOMETRIC_COLUMN_GROUPS), independent


def build_cutoff_grid(
    minimum_hz: float, maximum_hz: float, step_hz: float
) -> np.ndarray:
    """Build an inclusive, strictly increasing cutoff grid."""
    if not np.isfinite([minimum_hz, maximum_hz, step_hz]).all():
        raise ValueError("Cutoff-grid values must be finite.")
    if minimum_hz <= 0:
        raise ValueError("minimum_hz must be positive.")
    if maximum_hz <= minimum_hz:
        raise ValueError("maximum_hz must be greater than minimum_hz.")
    if step_hz <= 0:
        raise ValueError("step_hz must be positive.")

    count = int(math.floor((maximum_hz - minimum_hz) / step_hz + 1e-12))
    grid = minimum_hz + step_hz * np.arange(count + 1, dtype=float)
    if grid[-1] < maximum_hz - 1e-10:
        grid = np.append(grid, maximum_hz)
    else:
        grid[-1] = maximum_hz
    return grid


def corrected_design_cutoff(
    target_hz: float,
    sampling_frequency_hz: float = EXPECTED_SAMPLING_FREQUENCY_HZ,
    order_per_pass: int = DEFAULT_FILTER_ORDER_PER_PASS,
) -> float:
    """Return the single-pass cutoff for a final zero-phase -3 dB target.

    ``sosfiltfilt`` squares the magnitude response. This calculation prewarps
    the digital target and adjusts the Butterworth critical frequency so the
    combined forward-backward magnitude is 1/sqrt(2) at ``target_hz``.
    """
    if order_per_pass < 1:
        raise ValueError("order_per_pass must be at least 1.")
    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive.")
    nyquist_hz = sampling_frequency_hz / 2.0
    if not 0 < target_hz < nyquist_hz:
        raise ValueError(
            f"target_hz must lie between 0 and Nyquist ({nyquist_hz:g} Hz)."
        )

    ratio = (np.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order_per_pass))
    omega_target = (
        2.0 * sampling_frequency_hz * np.tan(np.pi * target_hz / sampling_frequency_hz)
    )
    omega_design = omega_target / ratio
    design_hz = (sampling_frequency_hz / np.pi) * np.arctan(
        omega_design / (2.0 * sampling_frequency_hz)
    )
    if not 0 < design_hz < nyquist_hz:
        raise ValueError("Corrected design cutoff is outside the valid range.")
    return float(design_hz)


def design_filter_sos(
    target_hz: float,
    sampling_frequency_hz: float = EXPECTED_SAMPLING_FREQUENCY_HZ,
    order_per_pass: int = DEFAULT_FILTER_ORDER_PER_PASS,
) -> np.ndarray:
    """Design the corrected low-pass filter as second-order sections."""
    design_hz = corrected_design_cutoff(
        target_hz,
        sampling_frequency_hz,
        order_per_pass,
    )
    return butter(
        order_per_pass,
        design_hz,
        btype="lowpass",
        fs=sampling_frequency_hz,
        output="sos",
    )


def sos_default_padlen(sos: np.ndarray) -> int:
    """Return SciPy's default ``sosfiltfilt`` pad length for an SOS array."""
    sos = np.asarray(sos)
    zeros_at_origin = min(
        int(np.sum(sos[:, 2] == 0.0)),
        int(np.sum(sos[:, 5] == 0.0)),
    )
    return 3 * (2 * len(sos) + 1 - zeros_at_origin)


def apply_zero_phase_filter(
    values: np.ndarray | Sequence[float],
    cutoff_hz: float,
    sampling_frequency_hz: float = EXPECTED_SAMPLING_FREQUENCY_HZ,
    order_per_pass: int = DEFAULT_FILTER_ORDER_PER_PASS,
) -> np.ndarray:
    """Apply the corrected zero-phase Butterworth filter down axis 0."""
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("values must be a one- or two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite entries.")
    sos = design_filter_sos(cutoff_hz, sampling_frequency_hz, order_per_pass)
    padlen = sos_default_padlen(sos)
    if array.shape[0] <= padlen:
        raise ValueError(
            f"Trajectory has {array.shape[0]} rows; more than {padlen} are required."
        )
    return sosfiltfilt(sos, array, axis=0, padlen=padlen)


def near_constant_columns(
    values: np.ndarray,
    atol: float = DEFAULT_STATIONARY_ATOL,
    rtol: float = DEFAULT_STATIONARY_RTOL,
) -> np.ndarray:
    """Return a boolean mask for constant or numerically near-constant columns."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    finite = np.isfinite(array).all(axis=0)
    result = np.ones(array.shape[1], dtype=bool)
    if finite.any():
        finite_values = array[:, finite]
        span = np.ptp(finite_values, axis=0)
        scale = np.max(np.abs(finite_values), axis=0)
        result[finite] = span <= (atol + rtol * scale)
    return result


def residual_autocorrelation_scores(
    raw: np.ndarray,
    filtered: np.ndarray,
    max_lag: int = DEFAULT_MAX_LAG,
) -> np.ndarray:
    """Calculate Challis criterion A for every column of a matrix."""
    raw_array = np.asarray(raw, dtype=float)
    filtered_array = np.asarray(filtered, dtype=float)
    if raw_array.ndim == 1:
        raw_array = raw_array[:, None]
    if filtered_array.ndim == 1:
        filtered_array = filtered_array[:, None]
    if raw_array.shape != filtered_array.shape:
        raise ValueError("raw and filtered arrays must have identical shapes.")
    if raw_array.ndim != 2:
        raise ValueError("raw and filtered must be one- or two-dimensional.")
    if max_lag < 1 or len(raw_array) <= max_lag:
        raise ValueError("max_lag must be positive and smaller than the row count.")

    residual = raw_array - filtered_array
    residual -= np.mean(residual, axis=0, keepdims=True)
    denominator = np.sum(residual * residual, axis=0)
    raw_scale = np.max(np.abs(raw_array), axis=0)
    energy_tolerance = (
        np.finfo(float).eps
        * raw_array.shape[0]
        * np.maximum(1.0, raw_scale * raw_scale)
    )
    valid_energy = np.isfinite(denominator) & (denominator > energy_tolerance)
    scores = np.full(raw_array.shape[1], np.nan, dtype=float)
    if not valid_energy.any():
        return scores

    accumulated = np.zeros(raw_array.shape[1], dtype=float)
    for lag in range(1, max_lag + 1):
        numerator = np.sum(residual[:-lag] * residual[lag:], axis=0)
        rho = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=valid_energy,
        )
        accumulated += np.where(np.isfinite(rho), rho * rho, 0.0)
    scores[valid_energy] = accumulated[valid_energy]
    return scores


def select_best_candidates(
    score_matrix: np.ndarray,
    candidate_cutoffs_hz: Sequence[float],
    reject_boundaries: bool = True,
) -> CutoffEstimateResult:
    """Select the minimum finite score per column, preferring lower-cutoff ties."""
    scores = np.asarray(score_matrix, dtype=float)
    candidates = np.asarray(candidate_cutoffs_hz, dtype=float)
    if scores.ndim != 2:
        raise ValueError("score_matrix must have shape (candidates, columns).")
    if len(candidates) != scores.shape[0]:
        raise ValueError("candidate count does not match score_matrix rows.")
    if len(candidates) < 2 or np.any(np.diff(candidates) <= 0):
        raise ValueError("candidate_cutoffs_hz must be strictly increasing.")

    finite = np.isfinite(scores)
    any_finite = finite.any(axis=0)
    safe_scores = np.where(finite, scores, np.inf)
    indices = np.argmin(safe_scores, axis=0)
    selected = np.full(scores.shape[1], np.nan, dtype=float)
    selected_scores = np.full(scores.shape[1], np.nan, dtype=float)
    valid = any_finite.copy()
    reasons = ["" if value else "all_scores_nonfinite" for value in any_finite]

    columns = np.arange(scores.shape[1])
    selected[any_finite] = candidates[indices[any_finite]]
    selected_scores[any_finite] = scores[indices[any_finite], columns[any_finite]]

    if reject_boundaries:
        boundary = any_finite & ((indices == 0) | (indices == len(candidates) - 1))
        valid[boundary] = False
        for column in np.flatnonzero(boundary):
            reasons[column] = "search_boundary"

    return CutoffEstimateResult(
        selected_cutoffs_hz=selected,
        selected_scores=selected_scores,
        valid=valid,
        reasons=tuple(reasons),
    )


def estimate_cutoffs_matrix(
    values: np.ndarray,
    candidate_cutoffs_hz: Sequence[float],
    sampling_frequency_hz: float = EXPECTED_SAMPLING_FREQUENCY_HZ,
    order_per_pass: int = DEFAULT_FILTER_ORDER_PER_PASS,
    max_lag: int = DEFAULT_MAX_LAG,
    stationary_atol: float = DEFAULT_STATIONARY_ATOL,
    stationary_rtol: float = DEFAULT_STATIONARY_RTOL,
    reject_boundaries: bool = True,
) -> CutoffEstimateResult:
    """Estimate one Challis cutoff for every column of a trajectory matrix."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")

    candidates = np.asarray(candidate_cutoffs_hz, dtype=float)
    if len(candidates) < 2 or np.any(np.diff(candidates) <= 0):
        raise ValueError("candidate_cutoffs_hz must be strictly increasing.")
    if candidates[-1] >= sampling_frequency_hz / 2.0:
        raise ValueError("All candidate cutoffs must be below Nyquist.")

    column_count = array.shape[1]
    nonfinite = ~np.isfinite(array).all(axis=0)
    stationary = near_constant_columns(array, stationary_atol, stationary_rtol)
    eligible = ~(nonfinite | stationary)
    scores = np.full((len(candidates), column_count), np.nan, dtype=float)

    if eligible.any():
        eligible_values = array[:, eligible]
        for index, cutoff_hz in enumerate(candidates):
            filtered = apply_zero_phase_filter(
                eligible_values,
                float(cutoff_hz),
                sampling_frequency_hz,
                order_per_pass,
            )
            scores[index, eligible] = residual_autocorrelation_scores(
                eligible_values,
                filtered,
                max_lag,
            )

    selected = select_best_candidates(scores, candidates, reject_boundaries)
    valid = selected.valid.copy()
    reasons = list(selected.reasons)
    for column in np.flatnonzero(nonfinite):
        valid[column] = False
        reasons[column] = "nonfinite_source"
    for column in np.flatnonzero(stationary & ~nonfinite):
        valid[column] = False
        reasons[column] = "near_constant_source"

    return CutoffEstimateResult(
        selected_cutoffs_hz=selected.selected_cutoffs_hz,
        selected_scores=selected.selected_scores,
        valid=valid,
        reasons=tuple(reasons),
    )


def validate_input_dataframe(
    dataframe: pd.DataFrame,
    required_numeric_columns: Iterable[str],
    *,
    require_positive_face_score: bool = True,
) -> None:
    """Validate FELT frame continuity and required numeric trajectories."""
    required = tuple(dict.fromkeys(required_numeric_columns))
    missing = [column for column in ("frame", *required) if column not in dataframe]
    if missing:
        raise IntegrityError(f"Missing required columns: {missing}")
    if dataframe.empty:
        raise IntegrityError("CSV contains no data rows.")

    frame = pd.to_numeric(dataframe["frame"], errors="coerce").to_numpy(float)
    if not np.isfinite(frame).all():
        raise IntegrityError("frame contains non-numeric or non-finite values.")
    if not np.equal(frame, np.floor(frame)).all():
        raise IntegrityError("frame contains non-integer values.")
    if len(frame) > 1 and not np.equal(np.diff(frame), 1.0).all():
        raise IntegrityError("frame is not strictly continuous with step 1.")

    if require_positive_face_score:
        if "FaceScore" not in dataframe:
            raise IntegrityError("Missing required FaceScore column.")
        face_score = pd.to_numeric(dataframe["FaceScore"], errors="coerce").to_numpy(
            float
        )
        if not np.isfinite(face_score).all():
            raise IntegrityError("FaceScore contains non-finite values.")
        if np.any(face_score <= 0.0):
            raise IntegrityError(
                "FaceScore contains values less than or equal to zero."
            )

    try:
        numeric = dataframe.loc[:, required].to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("A required trajectory column is non-numeric.") from exc
    if not np.isfinite(numeric).all():
        bad_mask = ~np.isfinite(numeric).all(axis=0)
        bad_columns = [required[index] for index in np.flatnonzero(bad_mask)]
        raise IntegrityError(
            f"Required trajectory columns contain non-finite values: {bad_columns[:10]}"
        )


def clipping_stats(
    values: np.ndarray | Sequence[float],
    lower: float = 0.0,
    upper: float = 1.0,
) -> ClippingStats:
    """Summarize values outside a closed interval before clipping."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite entries.")
    below = array < lower
    above = array > upper
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return ClippingStats(
        count=int(array.size),
        below_count=int(np.count_nonzero(below)),
        above_count=int(np.count_nonzero(above)),
        minimum=minimum,
        maximum=maximum,
        maximum_undershoot=max(0.0, lower - minimum),
        maximum_overshoot=max(0.0, maximum - upper),
    )


def clip_bounded(
    values: np.ndarray | Sequence[float],
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[np.ndarray, ClippingStats]:
    """Clip a numeric trajectory and return its pre-clipping diagnostics."""
    array = np.asarray(values, dtype=float)
    stats = clipping_stats(array, lower, upper)
    return np.clip(array, lower, upper), stats


def manifest_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash stable relative paths and byte sizes for a calibration corpus."""
    root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside manifest root: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def path_manifest_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash the ordered relative paths in a corpus, independent of contents."""
    root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside manifest root: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def content_manifest_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash stable relative paths and SHA-256 file-content digests."""
    root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside manifest root: {path}") from exc
        file_digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for JSON-compatible configuration."""
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_master_cutoffs(path: Path) -> dict[str, Any]:
    """Load and validate the structural contract of a master cutoff artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MasterCutoffError(f"Cannot read master cutoff JSON: {path}") from exc

    if payload.get("schema_version") != MASTER_SCHEMA_VERSION:
        raise MasterCutoffError(
            f"Unsupported master schema version: {payload.get('schema_version')!r}"
        )
    if payload.get("procedure_version") != PROCEDURE_VERSION:
        raise MasterCutoffError(
            f"Unsupported procedure version: {payload.get('procedure_version')!r}"
        )

    filter_config = payload.get("filter")
    if not isinstance(filter_config, dict):
        raise MasterCutoffError("Master artifact has no filter configuration.")
    for key in ("sampling_frequency_hz", "order_per_pass"):
        if key not in filter_config:
            raise MasterCutoffError(f"Master filter configuration is missing {key}.")

    geometric = payload.get("geometric_cutoffs_hz")
    if not isinstance(geometric, dict):
        raise MasterCutoffError("Master artifact has no geometric_cutoffs_hz mapping.")
    missing_groups = sorted(set(GEOMETRIC_COLUMN_GROUPS) - set(geometric))
    if missing_groups:
        raise MasterCutoffError(f"Master artifact is missing groups: {missing_groups}")

    nyquist = float(filter_config["sampling_frequency_hz"]) / 2.0
    for key, value in {
        **geometric,
        **payload.get("independent_cutoffs_hz", {}),
    }.items():
        try:
            cutoff = float(value)
        except (TypeError, ValueError) as exc:
            raise MasterCutoffError(f"Invalid cutoff for {key}: {value!r}") from exc
        if not 0 < cutoff < nyquist:
            raise MasterCutoffError(f"Cutoff for {key} is outside (0, Nyquist).")
    return payload


def median_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return stable descriptive statistics for finite values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
        }
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {
        "count": int(len(array)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
