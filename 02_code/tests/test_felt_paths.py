from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "utils" / "felt_paths.py"
SPEC = importlib.util.spec_from_file_location("felt_paths_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
felt_paths = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = felt_paths
SPEC.loader.exec_module(felt_paths)


def test_configured_path_uses_default_when_override_is_absent(tmp_path: Path) -> None:
    default = tmp_path / "default"

    result = felt_paths.configured_path("FELT_TEST_PATH", default, {})

    assert result == default.resolve()


def test_configured_path_uses_environment_override(tmp_path: Path) -> None:
    default = tmp_path / "default"
    override = tmp_path / "override"

    result = felt_paths.configured_path(
        "FELT_TEST_PATH",
        default,
        {"FELT_TEST_PATH": str(override)},
    )

    assert result == override.resolve()


def test_input_and_output_environment_names_are_distinct() -> None:
    assert felt_paths.INPUT_DIR_ENV == "FELT_INPUT_DIR"
    assert felt_paths.OUTPUT_DIR_ENV == "FELT_OUTPUT_DIR"
