"""End-to-end checks for the executable UK Gamma--Beta calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def calibration_module() -> ModuleType:
    """Load the calibration script with its neighbouring demo import visible."""
    repository_root = Path(__file__).parents[3]
    examples_directory = repository_root / "examples/basis"
    script = examples_directory / "dyadic_gamma_beta_calibration.py"
    sys.path.insert(0, str(examples_directory))
    try:
        spec = importlib.util.spec_from_file_location("_test_dyadic_gamma_beta_calibration", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load calibration module from {script}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(examples_directory))
    return module


def test_calibration_rebuilds_stable_topology_and_changes_land_only(
    calibration_module: ModuleType,
) -> None:
    """A small real-grid calibration reaches target without changing ocean root."""
    result = calibration_module.build_calibrated_case(
        data_directory=Path("tests/data"),
        topology_weight_mode="flat",
        target_relative_standard_deviation=0.2,
        draws=2,
        inner_regions=8,
        max_depth=2,
        seed=17,
    )
    variance_by_group = {group.name: group.root_variance for group in result.case.forest.groups}

    assert result.topology_iterations == 2
    assert result.aggregate.relative_standard_deviation == pytest.approx(0.2)
    assert variance_by_group["inner_land"] == pytest.approx(
        result.calibration.calibrated_root_variance
    )
    assert variance_by_group["inner_ocean"] == pytest.approx(0.25)
