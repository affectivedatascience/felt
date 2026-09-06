"""Unit tests for the shared Challis smoothing utilities."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.signal import sosfreqz

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.estimate_challis_cutoffs import (  # noqa: E402
    EstimatorConfig,
    build_master_artifact,
    pool_one_video,
)
from utils.challis_smoothing import (  # noqa: E402
    ACTION_UNIT_COLUMNS,
    BLENDSHAPE_COLUMNS,
    GEOMETRIC_COLUMN_GROUPS,
    MASTER_SCHEMA_VERSION,
    PROCEDURE_VERSION,
    IntegrityError,
    apply_zero_phase_filter,
    build_cutoff_grid,
    clip_bounded,
    content_manifest_digest,
    corrected_design_cutoff,
    design_filter_sos,
    load_master_cutoffs,
    manifest_digest,
    near_constant_columns,
    path_manifest_digest,
    residual_autocorrelation_scores,
    select_best_candidates,
    validate_input_dataframe,
)


def load_clean_signals_module():
    """Load the numbered smoothing script as an importable test module."""
    path = SRC_ROOT / "3_clean_signals.py"
    spec = importlib.util.spec_from_file_location("felt_clean_signals", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FilterDesignTests(unittest.TestCase):
    def test_corrected_design_has_requested_final_minus_3db_point(self) -> None:
        sampling_frequency = 29.97
        target = 8.0
        sos = design_filter_sos(target, sampling_frequency, order_per_pass=2)
        _, response = sosfreqz(sos, worN=np.array([target]), fs=sampling_frequency)
        final_magnitude = abs(response[0]) ** 2
        self.assertAlmostEqual(final_magnitude, 1.0 / np.sqrt(2.0), places=10)
        self.assertGreater(
            corrected_design_cutoff(target, sampling_frequency, 2),
            target,
        )

    def test_filter_accepts_matrix_and_preserves_shape(self) -> None:
        time = np.arange(120) / 29.97
        values = np.column_stack([np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)])
        filtered = apply_zero_phase_filter(values, 8.0)
        self.assertEqual(filtered.shape, values.shape)

    def test_cutoff_grid_is_inclusive(self) -> None:
        np.testing.assert_allclose(
            build_cutoff_grid(1.0, 2.0, 0.25),
            [1.0, 1.25, 1.5, 1.75, 2.0],
        )


class ChallisScoreTests(unittest.TestCase):
    def test_white_residual_scores_below_correlated_residual(self) -> None:
        rng = np.random.default_rng(12)
        raw = np.zeros((500, 2))
        white = rng.normal(size=500)
        correlated = np.empty(500)
        correlated[0] = white[0]
        for index in range(1, len(correlated)):
            correlated[index] = 0.95 * correlated[index - 1] + white[index]
        filtered = np.column_stack([-white, -correlated])
        scores = residual_autocorrelation_scores(raw, filtered, max_lag=10)
        self.assertLess(scores[0], scores[1])

    def test_candidate_tie_prefers_lower_frequency(self) -> None:
        result = select_best_candidates(
            np.array([[0.4], [0.2], [0.2], [0.5]]),
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertTrue(result.valid[0])
        self.assertEqual(result.selected_cutoffs_hz[0], 2.0)

    def test_boundary_candidate_is_rejected(self) -> None:
        result = select_best_candidates(
            np.array([[0.1], [0.2], [0.3]]),
            [1.0, 2.0, 3.0],
        )
        self.assertFalse(result.valid[0])
        self.assertEqual(result.reasons[0], "search_boundary")

    def test_near_constant_detection_is_scale_aware(self) -> None:
        values = np.column_stack(
            [np.ones(20), 1e6 + np.linspace(0, 1e-4, 20), np.linspace(0, 1, 20)]
        )
        np.testing.assert_array_equal(
            near_constant_columns(values),
            [True, True, False],
        )


class IntegrityAndClippingTests(unittest.TestCase):
    def test_valid_input(self) -> None:
        dataframe = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "FaceScore": [0.9, 0.8, 0.95],
                "x": [1.0, 1.2, 1.4],
            }
        )
        validate_input_dataframe(dataframe, ["x"])

    def test_frame_gap_fails(self) -> None:
        dataframe = pd.DataFrame(
            {
                "frame": [0, 2],
                "FaceScore": [0.9, 0.8],
                "x": [1.0, 1.4],
            }
        )
        with self.assertRaises(IntegrityError):
            validate_input_dataframe(dataframe, ["x"])

    def test_nonpositive_face_score_fails(self) -> None:
        dataframe = pd.DataFrame(
            {
                "frame": [0, 1],
                "FaceScore": [0.9, 0.0],
                "x": [1.0, 1.4],
            }
        )
        with self.assertRaises(IntegrityError):
            validate_input_dataframe(dataframe, ["x"])

    def test_clipping_reports_overshoot(self) -> None:
        clipped, stats = clip_bounded([-0.2, 0.5, 1.3])
        np.testing.assert_allclose(clipped, [0.0, 0.5, 1.0])
        self.assertEqual(stats.below_count, 1)
        self.assertEqual(stats.above_count, 1)
        self.assertAlmostEqual(stats.maximum_undershoot, 0.2)
        self.assertAlmostEqual(stats.maximum_overshoot, 0.3)


class MasterArtifactTests(unittest.TestCase):
    def test_approved_v2_master_contains_all_release_cutoffs(self) -> None:
        path = SRC_ROOT.parent / "config" / "master_cutoffs_v2.json"
        artifact = load_master_cutoffs(path)
        self.assertEqual(
            set(artifact["independent_cutoffs_hz"]),
            set(ACTION_UNIT_COLUMNS) | set(BLENDSHAPE_COLUMNS),
        )
        self.assertEqual(artifact["input"]["available_file_count"], 2452)

    def test_master_round_trip(self) -> None:
        payload = {
            "schema_version": MASTER_SCHEMA_VERSION,
            "procedure_version": PROCEDURE_VERSION,
            "filter": {
                "sampling_frequency_hz": 29.97,
                "order_per_pass": 2,
            },
            "geometric_cutoffs_hz": {key: 8.0 for key in GEOMETRIC_COLUMN_GROUPS},
            "independent_cutoffs_hz": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_master_cutoffs(path)
        self.assertEqual(loaded["geometric_cutoffs_hz"]["face_mesh"], 8.0)

    def test_video_pooling_uses_valid_column_median(self) -> None:
        common = {
            "relative_path": "speech/Actor_01/example.csv",
            "vocal_channel": "speech",
            "actor_folder": "Actor_01",
            "stem": "01-01-01-01-01-01-01",
            "emotion_code": "01",
            "intensity_code": "01",
            "statement_code": "01",
            "repetition_code": "01",
            "actor_code": "1",
            "family": "face_box",
            "pool_key": "face_box",
        }
        rows = [
            {**common, "valid": "True", "selected_cutoff_hz": "2", "reason": ""},
            {**common, "valid": "True", "selected_cutoff_hz": "4", "reason": ""},
            {
                **common,
                "valid": "False",
                "selected_cutoff_hz": "12",
                "reason": "search_boundary",
            },
        ]
        pooled = pool_one_video(rows, min_valid_fraction=0.5)
        self.assertEqual(len(pooled), 1)
        self.assertTrue(pooled[0]["valid"])
        self.assertEqual(pooled[0]["video_cutoff_hz"], 3.0)
        self.assertEqual(pooled[0]["boundary_column_count"], 1)

    def test_unavailable_optional_column_does_not_block_geometric_master(self) -> None:
        config = EstimatorConfig(
            sampling_frequency_hz=29.97,
            order_per_pass=2,
            max_lag=10,
            candidate_cutoffs_hz=(1.0, 2.0, 3.0),
            include_action_units=True,
            include_blendshapes=False,
            reject_boundaries=True,
            min_valid_fraction=0.5,
        )
        rows = [
            {
                "pool_key": group,
                "family": group,
                "video_count": 10,
                "valid_video_count": 10,
                "invalid_video_count": 0,
                "boundary_column_count": 0,
                "median": 4.0,
            }
            for group in GEOMETRIC_COLUMN_GROUPS
        ]
        rows.append(
            {
                "pool_key": "AU11",
                "family": "action_unit",
                "video_count": 10,
                "valid_video_count": 2,
                "invalid_video_count": 8,
                "boundary_column_count": 1,
                "median": 3.0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "speech" / "Actor_01" / "01-01-01-01-01-01-01.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("frame\n0\n", encoding="utf-8")
            artifact = build_master_artifact(
                [csv_path],
                1,
                root,
                root / "qc",
                config,
                pd.DataFrame(rows),
                "subset",
            )
        self.assertEqual(artifact["geometric_cutoffs_hz"]["face_mesh"], 4.0)
        self.assertNotIn("AU11", artifact["independent_cutoffs_hz"])
        self.assertIn("AU11", artifact["unavailable_independent_cutoffs"])


class EndToEndSmoothingTests(unittest.TestCase):
    def test_release_cli_filters_optional_families_by_default(self) -> None:
        module = load_clean_signals_module()
        with mock.patch.object(sys, "argv", ["3_clean_signals.py"]):
            args = module.parse_args()
        self.assertTrue(args.filter_action_units)
        self.assertTrue(args.filter_blendshapes)
        self.assertEqual(
            args.master_cutoffs,
            SRC_ROOT.parent / "config" / "master_cutoffs_v2.json",
        )

    def test_calibration_validation_is_independent_of_root_path(self) -> None:
        module = load_clean_signals_module()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            relative = Path("speech/Actor_01/example.csv")
            for root in (first_root, second_root):
                path = root / relative
                path.parent.mkdir(parents=True)
                path.write_text("frame,value\n0,1\n", encoding="utf-8")

            first_paths = [first_root / relative]
            second_paths = [second_root / relative]
            master = {
                "input": {
                    "available_file_count": 1,
                    "path_manifest_digest": path_manifest_digest(
                        first_paths, first_root
                    ),
                    "manifest_digest": manifest_digest(first_paths, first_root),
                    "content_manifest_digest": content_manifest_digest(
                        first_paths, first_root
                    ),
                }
            }
            result = module.validate_calibration_corpus(
                master,
                second_paths,
                second_root,
                allow_mismatch=False,
            )
        self.assertEqual(
            result["path_manifest_digest"],
            master["input"]["path_manifest_digest"],
        )
        self.assertEqual(result["manifest_digest"], "not-computed")

    def test_strict_calibration_validation_rejects_content_change(self) -> None:
        module = load_clean_signals_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "speech" / "Actor_01" / "example.csv"
            path.parent.mkdir(parents=True)
            path.write_text("frame,value\n0,1\n", encoding="utf-8")
            paths = [path]
            master = {
                "input": {
                    "available_file_count": 1,
                    "path_manifest_digest": path_manifest_digest(paths, root),
                    "manifest_digest": manifest_digest(paths, root),
                    "content_manifest_digest": content_manifest_digest(paths, root),
                }
            }
            path.write_text("frame,value\n0,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content manifest digest differs"):
                module.validate_calibration_corpus(
                    master,
                    paths,
                    root,
                    allow_mismatch=False,
                    require_content_match=True,
                )

    def test_optional_scores_are_filtered_and_checkpointed(self) -> None:
        module = load_clean_signals_module()
        frame_count = 40
        phase = np.linspace(0.0, 4.0 * np.pi, frame_count)
        data: dict[str, np.ndarray] = {
            "frame": np.arange(frame_count),
            "FaceScore": np.full(frame_count, 0.99),
        }
        geometric_columns = [
            column for columns in GEOMETRIC_COLUMN_GROUPS.values() for column in columns
        ]
        for index, column in enumerate(geometric_columns):
            data[column] = index * 1e-3 + np.sin(phase + index * 1e-4)
        bounded_columns = (*ACTION_UNIT_COLUMNS, *BLENDSHAPE_COLUMNS)
        for index, column in enumerate(bounded_columns):
            data[column] = 0.5 + 0.45 * np.sin(phase + index * 0.01)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "raw"
            input_path = input_root / "speech" / "Actor_01" / "01-01-01-01-01-01-01.csv"
            output_path = root / "smooth" / "speech" / "Actor_01" / input_path.name
            checkpoint_path = root / "qc" / "checkpoint.json"
            input_path.parent.mkdir(parents=True)
            pd.DataFrame(data).to_csv(input_path, index=False)

            config = module.SmoothingConfig(
                master_digest="test",
                master_semantic_digest="test-semantic",
                sampling_frequency_hz=29.97,
                order_per_pass=2,
                geometric_cutoffs_hz={group: 4.0 for group in GEOMETRIC_COLUMN_GROUPS},
                independent_cutoffs_hz={column: 4.0 for column in bounded_columns},
                filter_action_units=True,
                filter_blendshapes=True,
            )
            result = module.smooth_one_file(
                module.SmoothTask(
                    input_path=input_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                    input_root=input_root,
                    config=config,
                    overwrite=False,
                )
            )
            self.assertEqual(result.status, "processed", result.error)
            output = pd.read_csv(output_path)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertTrue((output.loc[:, bounded_columns] >= 0.0).all().all())
        self.assertTrue((output.loc[:, bounded_columns] <= 1.0).all().all())
        self.assertFalse(np.allclose(output["x_0"], data["x_0"]))
        self.assertEqual(len(checkpoint["clipping_rows"]), len(bounded_columns))
        self.assertEqual(len(checkpoint["validation_rows"]), 8)


if __name__ == "__main__":
    unittest.main()
