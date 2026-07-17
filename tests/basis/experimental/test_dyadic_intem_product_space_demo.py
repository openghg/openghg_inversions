"""Focused regression tests for the synthetic InTEM product-space demo."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.product_space import (
    enumerate_partition_neighbors,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


@pytest.fixture(scope="module")
def demo_module() -> Iterator[ModuleType]:
    """Load the executable example as an isolated module for direct testing."""
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "examples/basis/dyadic_intem_product_space_demo.py"
    module_name = "_test_dyadic_intem_product_space_demo"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load example module from {script}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def synthetic_case(demo_module: ModuleType) -> Any:
    """Build the default deterministic synthetic case once for focused tests."""
    return demo_module.build_synthetic_case()


def test_packaged_europe_regions_and_rectangle_are_stable(demo_module: ModuleType) -> None:
    """The packaged EUROPE map should retain its labels, dimensions, and selected bounds."""
    regions = demo_module.load_intem_europe_regions()

    assert regions.dims == ("lat", "lon")
    assert regions.shape == (293, 391)
    assert regions.dtype.kind in "iu"
    np.testing.assert_array_equal(np.unique(regions), np.arange(7))
    assert demo_module.select_inner_rectangle(regions) == demo_module.InnerRectangle(
        row_start=175,
        row_stop=179,
        column_start=277,
        column_stop=281,
        latitude_min=51.679,
        latitude_max=52.381,
        longitude_min=-0.396,
        longitude_max=0.66,
    )


def test_synthetic_case_preserves_inner_outer_contract(
    demo_module: ModuleType,
    synthetic_case: Any,
) -> None:
    """The case should expose a 4x4 inner model and seven fixed outer coefficients."""
    case = synthetic_case

    assert case.rectangle.shape == (4, 4)
    assert case.outer_labels == tuple(range(7))
    assert case.inner_grid_design.shape == (24, 4, 4)
    assert case.outer_design.shape == (24, 7)
    assert case.inner_truth.shape == (4, 4)
    assert case.outer_truth.shape == (7,)
    assert case.observation_mean.shape == (24,)
    assert case.observations.shape == (24,)
    assert case.observation_covariance.shape == (24, 24)
    assert case.inner_prior_sd == 1.0
    assert case.outer_prior_sd == 0.5
    np.testing.assert_array_equal(case.target.outer_prior_covariance, np.eye(7) * 0.25)
    assert not any("boundary" in field.name for field in fields(demo_module.IntemSyntheticCase))


def test_inner_rectangle_and_outer_masks_cover_the_grid_once(synthetic_case: Any) -> None:
    """The carved rectangle and seven residual classes should be disjoint and complete."""
    case = synthetic_case
    region_values = np.asarray(case.regions.values)
    inner_mask = np.zeros(region_values.shape, dtype=bool)
    inner_mask[case.rectangle.row_slice, case.rectangle.column_slice] = True
    outer_masks = [(region_values == label) & ~inner_mask for label in case.outer_labels]
    membership = inner_mask.astype(int)
    for outer_mask in outer_masks:
        membership += outer_mask

    np.testing.assert_array_equal(membership, np.ones(region_values.shape, dtype=int))
    assert not np.any(outer_masks[6] & inner_mask)
    assert np.any(outer_masks[6])


def test_sensitivities_and_observations_follow_declared_equations(synthetic_case: Any) -> None:
    """Summed designs and seeded noise should reconstruct the synthetic observations."""
    case = synthetic_case
    aggregated_sensitivity = case.inner_grid_design.sum(axis=(1, 2)) + case.outer_design.sum(axis=1)
    np.testing.assert_allclose(aggregated_sensitivity, 1.35, rtol=0.0, atol=1e-14)

    noise = np.random.default_rng(20260717).normal(scale=0.08, size=case.observations.size)
    reconstructed = (
        case.observation_mean
        + np.einsum("ijk,jk->i", case.inner_grid_design, case.inner_truth, optimize=True)
        + case.outer_design @ case.outer_truth
        + noise
    )
    np.testing.assert_allclose(case.observations, reconstructed, rtol=0.0, atol=1e-12)


def test_default_seed_has_pinned_observations(synthetic_case: Any) -> None:
    """The documented seed should reproduce the exact synthetic observation vector."""
    expected = np.array(
        [
            1799.5070165621266,
            1799.6402032368801,
            1799.8222980430708,
            1800.0772617747743,
            1799.6344967349785,
            1800.0506646994972,
            1800.6647944321264,
            1800.3609182719872,
            1799.7460112278079,
            1800.3082472948083,
            1800.6873898698643,
            1800.3047865924414,
            1799.8629100348271,
            1800.3415101015737,
            1800.3227465859475,
            1800.1761298306037,
            1799.6176440271875,
            1799.8712974401467,
            1800.1593173010731,
            1800.3766675340014,
            1799.9175229283962,
            1800.3741880224429,
            1800.5660575429902,
            1800.408577930308,
        ]
    )

    np.testing.assert_allclose(synthetic_case.observations, expected, rtol=0.0, atol=1e-10)


def test_exact_partition_space_is_closed_and_has_pinned_probabilities(
    synthetic_case: Any,
) -> None:
    """All 677 partitions should form a closed oracle with deterministic probabilities."""
    case = synthetic_case
    partitions = case.partitions
    partition_set = set(partitions)

    assert len(partitions) == 677
    assert len(partition_set) == 677
    for partition in partitions:
        partition.validate(case.target.tree)
        assert {
            neighbor.partition for neighbor in enumerate_partition_neighbors(case.target.tree, partition)
        } <= partition_set

    probability_by_partition = case.target.partition_probabilities(partitions)
    probabilities = np.array([probability_by_partition[partition] for partition in partitions])
    expected = {
        0: 3.8148007892440524e-40,
        1: 1.0848636368780026e-37,
        100: 1.0960885687064343e-4,
        300: 4.766451768273937e-4,
        389: 1.6301009567895042e-2,
        500: 1.5152604639667451e-3,
        676: 2.7617158199417565e-3,
    }

    assert probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.argmax(probabilities) == 389
    for index, probability in expected.items():
        assert probabilities[index] == pytest.approx(probability, rel=1e-10, abs=0.0)


def test_main_prints_machine_readable_summary_json(
    demo_module: ModuleType,
    synthetic_case: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command should print the complete summary contract as JSON."""
    summary = demo_module.IntemDemoSummary(
        rectangle=synthetic_case.rectangle,
        observation_count=24,
        partition_count=677,
        outer_region_count=7,
        inner_prior_sd=1.0,
        outer_prior_sd=0.5,
        exact_map_index=389,
        exact_map_inner_regions=12,
        exact_map_probability=0.25,
        exact_expected_inner_regions=11.5,
        sampled_expected_inner_regions=11.25,
        sampled_unique_partitions=42,
        inner_region_count_total_variation_distance=0.1,
    )
    build_case = MagicMock(return_value=synthetic_case)
    run_demo = MagicMock(return_value=summary)
    monkeypatch.setattr(demo_module, "build_synthetic_case", build_case)
    monkeypatch.setattr(demo_module, "run_demo", run_demo)

    assert demo_module.main(["--draws", "8", "--tune", "3", "--seed", "41"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output == summary.as_dict()
    assert output["rectangle"] == {
        "row_start": 175,
        "row_stop": 179,
        "column_start": 277,
        "column_stop": 281,
        "latitude_min": 51.679,
        "latitude_max": 52.381,
        "longitude_min": -0.396,
        "longitude_max": 0.66,
    }
    assert not any("boundary" in key for key in output)
    build_case.assert_called_once_with(region_penalty=0.12, seed=41)
    run_demo.assert_called_once_with(synthetic_case, draws=8, tune=3, seed=41)


def test_run_demo_computes_inner_region_summaries_from_trace(
    demo_module: ModuleType,
    synthetic_case: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic fake trace should exercise sampled inner-K diagnostics."""
    sampled_indices = np.array([0, 1, 1, 2], dtype=int)
    fake_adapter = SimpleNamespace(
        model=nullcontext(),
        step_methods=MagicMock(return_value=()),
    )
    monkeypatch.setattr(
        demo_module,
        "build_pymc_product_space_model",
        MagicMock(return_value=fake_adapter),
    )
    monkeypatch.setattr(
        demo_module.pm,
        "sample",
        MagicMock(return_value=SimpleNamespace(posterior={"partition_index": sampled_indices})),
    )

    summary = demo_module.run_demo(synthetic_case, draws=4, tune=0, seed=12)

    exact_by_partition = synthetic_case.target.partition_probabilities(synthetic_case.partitions)
    exact = np.array([exact_by_partition[partition] for partition in synthetic_case.partitions])
    inner_region_counts = np.array(
        [len(partition.active) for partition in synthetic_case.partitions],
        dtype=int,
    )
    sampled = np.bincount(sampled_indices, minlength=len(synthetic_case.partitions)) / 4.0
    exact_by_count = np.bincount(inner_region_counts, weights=exact)
    sampled_by_count = np.bincount(inner_region_counts, weights=sampled)

    assert summary.sampled_unique_partitions == 3
    assert summary.sampled_expected_inner_regions == pytest.approx(sampled @ inner_region_counts)
    assert summary.inner_region_count_total_variation_distance == pytest.approx(
        0.5 * np.abs(sampled_by_count - exact_by_count).sum()
    )


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        (xr.Dataset(), "contain 'region'"),
        (xr.Dataset({"region": (("y", "x"), np.zeros((2, 2), dtype=int))}), "dimensions"),
        (xr.Dataset({"region": (("lat", "lon"), np.zeros((2, 2), dtype=float))}), "integers"),
        (xr.Dataset({"region": (("lat", "lon"), np.arange(6, dtype=int)[None, :])}), "0 through 6"),
    ],
    ids=("missing-region", "wrong-dimensions", "float-labels", "incomplete-labels"),
)
def test_packaged_region_validation_rejects_invalid_layouts(
    demo_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    dataset: xr.Dataset,
    message: str,
) -> None:
    """Packaged map validation should reject missing or malformed region arrays."""
    resource_root = MagicMock()
    resource_root.joinpath.return_value = Path("invalid-intem.nc")
    monkeypatch.setattr(demo_module, "files", MagicMock(return_value=resource_root))
    monkeypatch.setattr(
        demo_module,
        "as_file",
        MagicMock(return_value=nullcontext(Path("invalid-intem.nc"))),
    )
    monkeypatch.setattr(demo_module.xr, "open_dataset", MagicMock(return_value=dataset))

    with pytest.raises(ValueError, match=message):
        demo_module.load_intem_europe_regions()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (True, 4)}, "two integer sizes"),
        ({"shape": (0, 4)}, "non-empty rectangle"),
        ({"shape": (294, 4)}, "within regions"),
        ({"centre": (np.nan, 0.0)}, "finite"),
    ],
)
def test_rectangle_validation_rejects_invalid_requests(
    demo_module: ModuleType,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Rectangle selection should reject malformed sizes and centres."""
    regions = demo_module.load_intem_europe_regions()

    with pytest.raises(ValueError, match=message):
        demo_module.select_inner_rectangle(regions, **kwargs)


def test_rectangle_validation_rejects_wrong_dims_and_missing_window(demo_module: ModuleType) -> None:
    """Rectangle selection should require labelled dimensions and a complete inner window."""
    wrong_dims = xr.DataArray(np.zeros((2, 2), dtype=int), dims=("y", "x"))
    isolated_inner = xr.DataArray(
        [[1, 0], [0, 0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )

    with pytest.raises(ValueError, match="dimensions"):
        demo_module.select_inner_rectangle(wrong_dims)
    with pytest.raises(ValueError, match="No requested rectangle"):
        demo_module.select_inner_rectangle(isolated_inner, shape=(2, 2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"observation_count": True}, "must be an integer"),
        ({"observation_count": 0}, "must be positive"),
        ({"inner_prior_sd": 0.0}, "inner_prior_sd"),
        ({"outer_prior_sd": np.nan}, "outer_prior_sd"),
        ({"observation_error_sd": np.inf}, "observation_error_sd"),
        ({"region_penalty": -1.0}, "region_penalty"),
        ({"region_penalty": np.inf}, "region_penalty"),
    ],
)
def test_synthetic_case_validation_rejects_invalid_configuration(
    demo_module: ModuleType,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Synthetic case construction should fail before work for invalid numeric settings."""
    with pytest.raises(ValueError, match=message):
        demo_module.build_synthetic_case(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"draws": True}, "draws must be a positive integer"),
        ({"draws": 0}, "draws must be a positive integer"),
        ({"tune": True}, "tune must be a non-negative integer"),
        ({"tune": -1}, "tune must be a non-negative integer"),
    ],
)
def test_run_demo_validation_rejects_invalid_sample_counts(
    demo_module: ModuleType,
    synthetic_case: Any,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Sampling validation should reject invalid counts without starting PyMC."""
    with pytest.raises(ValueError, match=message):
        demo_module.run_demo(synthetic_case, **kwargs)
