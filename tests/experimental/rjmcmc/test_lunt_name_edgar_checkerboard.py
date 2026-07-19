"""Focused contracts for the data-backed NAME/EDGAR RJMCMC example."""

from __future__ import annotations

from collections.abc import Iterator
from importlib import util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def example_module() -> Iterator[ModuleType]:
    """Load the executable root example as an isolated module."""
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "examples" / "rjmcmc" / "lunt_name_edgar_checkerboard.py"
    module_name = "_test_lunt_name_edgar_checkerboard"
    specification = util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load example module from {script}.")
    module = util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def checkerboard_case(example_module: ModuleType) -> Any:
    """Build the local raw-data case once for the focused contracts."""
    return example_module.build_name_edgar_checkerboard_case()


def test_name_edgar_checkerboard_closes_raw_data_accounting(
    example_module: ModuleType,
    checkerboard_case: Any,
) -> None:
    """Inner and seven-column outer calculations should close exactly."""
    case = checkerboard_case

    assert case.site_observation_counts == (28, 28)
    assert case.problem.sensitivities.shape == (56, 48 * 56)
    assert (case.problem.k_min, case.problem.k_max) == (5, 100)
    assert case.fixed_outer_regions.shape == (56, 7)
    assert case.problem.fixed_block is not None
    assert case.problem.n_fixed_coefficients == 7
    np.testing.assert_array_equal(np.unique(case.crop_intem_labels), [6])
    np.testing.assert_array_equal(np.unique(case.truth), [0.5, 1.5])
    np.testing.assert_array_equal(
        case.truth.reshape(example_module.GRID_SHAPE)[:12, :28],
        np.repeat([[0.5] * 14 + [1.5] * 14], 12, axis=0),
    )
    np.testing.assert_allclose(
        case.fixed_outer_regions @ np.ones(7),
        case.fixed_outer,
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        case.problem.sensitivities @ case.truth,
        case.inner_noiseless,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        case.fixed_outer_regions.sum(axis=1),
        case.fixed_outer,
        rtol=0.0,
        atol=3.0e-13,
    )
    assert np.all(np.any(case.fixed_outer_regions > 0.0, axis=0))
    np.testing.assert_allclose(
        case.problem.fixed_block.design,
        case.fixed_outer_regions,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        case.problem.fixed_block.coefficient_prior_mean,
        np.ones(7),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        case.problem.fixed_block.coefficient_prior_sd,
        np.ones(7),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        case.problem.fixed_offset,
        np.zeros(56),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        case.problem.observations,
        case.full_observations,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        case.full_noiseless,
        case.problem.sensitivities @ case.truth + case.problem.fixed_block.design @ np.ones(7),
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        case.problem.observations,
        case.full_noiseless + case.noise,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.all(case.fixed_outer >= 0.0)
    assert np.median(case.inner_noiseless) > example_module.OBSERVATION_SD


def test_rhime_adapter_preserves_native_grid_order(
    example_module: ModuleType,
    checkerboard_case: Any,
) -> None:
    """The example should preserve row-major NAME order through the adapter."""
    problem = checkerboard_case.problem
    rows, columns = np.indices(example_module.GRID_SHAPE)

    assert problem.grid_coordinates.shape == (48 * 56, 2)
    np.testing.assert_array_equal(
        problem.grid_coordinates,
        np.column_stack((rows.reshape(-1), columns.reshape(-1))),
    )
    assert checkerboard_case.latitudes.shape == (48,)
    assert checkerboard_case.longitudes.shape == (56,)


def test_builder_opens_only_declared_non_boundary_inputs(
    example_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The example must not open the known-corrupt boundary-condition fixture."""
    expected_paths = example_module.CheckerboardInputPaths.repository_defaults().all_paths
    real_open_dataset = xr.open_dataset
    opened_paths: list[Path] = []

    def recording_open_dataset(path: Path, *args: object, **kwargs: object) -> xr.Dataset:
        """Record each opened path before delegating to xarray."""
        opened_paths.append(Path(path))
        return real_open_dataset(path, *args, **kwargs)

    monkeypatch.setattr(example_module.xr, "open_dataset", recording_open_dataset)
    case = example_module.build_name_edgar_checkerboard_case()

    assert tuple(opened_paths) == expected_paths
    assert case.input_paths.all_paths == expected_paths
    assert not any("boundary" in str(path).lower() or "bc_" in path.name.lower() for path in opened_paths)


def test_fixed_comparators_have_declared_dynamic_and_outer_blocks(
    example_module: ModuleType,
    checkerboard_case: Any,
) -> None:
    """Truth labels and non-oracle sensitivity labels should share the outer block."""
    oracle_labels = example_module.block_labels()
    oracle = example_module.oracle_fixed_problem(checkerboard_case)
    quadtree, quadtree_labels = example_module.quadtree_fixed_problem(checkerboard_case)

    assert oracle_labels.shape == (48 * 56,)
    assert quadtree_labels.shape == (48 * 56,)
    np.testing.assert_array_equal(np.unique(oracle_labels), np.arange(16))
    assert np.unique(quadtree_labels).size == 16
    assert oracle.sensitivities.shape == (56, 16)
    assert quadtree.sensitivities.shape == (56, 16)
    assert (oracle.k_min, oracle.k_max) == (16, 16)
    assert (quadtree.k_min, quadtree.k_max) == (16, 16)
    assert oracle.fixed_block is checkerboard_case.problem.fixed_block
    assert quadtree.fixed_block is checkerboard_case.problem.fixed_block
    assert oracle.n_fixed_coefficients == 7
    assert quadtree.n_fixed_coefficients == 7
    np.testing.assert_allclose(oracle.fixed_offset, np.zeros(56), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(quadtree.fixed_offset, np.zeros(56), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        oracle.fixed_block.design @ np.ones(7),
        checkerboard_case.fixed_outer,
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        quadtree.fixed_block.design @ np.ones(7),
        checkerboard_case.fixed_outer,
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        oracle.sensitivities,
        np.column_stack(
            [
                checkerboard_case.problem.sensitivities[:, oracle_labels == label].sum(axis=1)
                for label in range(16)
            ]
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_main_prints_machine_readable_provenance_and_results(
    example_module: ModuleType,
    checkerboard_case: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cheap CLI path should print JSON and make no output-file request."""
    fit = example_module.BenchmarkFit(
        name="test fit",
        prediction_rmse=1.25,
        mean_fixed_coefficients=(0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.95),
        visited_k=(15, 16),
        runtime_seconds=0.5,
    )
    summary = example_module.CheckerboardBenchmarkSummary(
        observation_count=56,
        grid_shape=(48, 56),
        site_observation_counts=(28, 28),
        observation_sd=5.0,
        iterations=12,
        start=4,
        thin=2,
        sampling_seed=17,
        initial_seed=18,
        prior_prediction_rmse=6.5,
        oracle=fit,
        quadtree=fit,
        rjmcmc=fit,
        input_provenance=({"path": "flux.nc", "size_bytes": 10, "sha256": "abc"},),
    )
    build_case = MagicMock(return_value=checkerboard_case)
    run_benchmark = MagicMock(return_value=summary)
    monkeypatch.setattr(example_module, "build_name_edgar_checkerboard_case", build_case)
    monkeypatch.setattr(example_module, "run_benchmark", run_benchmark)

    result = example_module.main(
        [
            "--iterations",
            "12",
            "--start",
            "4",
            "--thin",
            "2",
            "--sampling-seed",
            "17",
            "--initial-seed",
            "18",
            "--indent",
            "0",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output == summary.as_dict()
    assert output["scope"] == "implementation benchmark, not a paper reproduction"
    assert output["boundary_conditions"].startswith("excluded")
    assert output["sampler"]["fixed_coefficient_proposal_sd"] == 0.1
    assert output["sampler"]["schedule_id"].startswith("five_slot")
    assert output["fits"]["rjmcmc"]["mean_fixed_coefficients"] == [
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.0,
        0.95,
    ]
    assert output["inputs"] == [{"path": "flux.nc", "size_bytes": 10, "sha256": "abc"}]
    build_case.assert_called_once_with()
    run_benchmark.assert_called_once_with(
        checkerboard_case,
        iterations=12,
        start=4,
        thin=2,
        sampling_seed=17,
        initial_seed=18,
    )


def test_short_real_cli_runs_all_three_joint_inversions(
    example_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All three real joint fits should report seven outer means and total RMSE."""
    result = example_module.main(
        [
            "--iterations",
            "5",
            "--start",
            "0",
            "--thin",
            "1",
            "--sampling-seed",
            "17",
            "--initial-seed",
            "18",
            "--indent",
            "0",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["observations"]["count"] == 56
    assert output["sampler"]["iterations"] == 5
    assert output["sampler"]["schedule_id"].startswith("five_slot")
    for fit in output["fits"].values():
        assert len(fit["mean_fixed_coefficients"]) == 7
        assert np.all(np.isfinite(fit["mean_fixed_coefficients"]))
        assert fit["prediction_rmse"] >= 0.0
