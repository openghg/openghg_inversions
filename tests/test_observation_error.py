import json

import numpy as np
import dask.array as da
from dask import delayed
import pytest
import xarray as xr

from openghg_inversions.observation_error import (
    aggregation_error_as_low_rank,
    prepare_low_rank_aggregation_error,
    resolve_aggregation_error,
)


def _inputs() -> xr.Dataset:
    return xr.Dataset(coords={"nmeasure": np.arange(3)})


def test_dense_covariance_is_primary_over_matching_sd_diagnostic() -> None:
    covariance = np.array([[2.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 0.5]])
    data = _inputs()
    data["aggregation_error_covariance"] = (("nmeasure", "nmeasure_cov"), covariance)
    data["aggregation_error_sd"] = ("nmeasure", np.sqrt(np.diag(covariance)))

    result = resolve_aggregation_error(data)

    assert result.mode == "dense"
    np.testing.assert_allclose(result.marginal_variance, np.diag(covariance))


def test_dense_covariance_rejects_reordered_second_observation_axis() -> None:
    data = xr.Dataset(
        {
            "aggregation_error_covariance": (
                ("nmeasure", "nmeasure_cov"),
                np.array([[2.0, 0.4], [0.4, 1.0]]),
            )
        },
        coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["B", "A"]},
    )

    with pytest.raises(ValueError, match="nmeasure_cov"):
        resolve_aggregation_error(data)


def test_dense_covariance_labels_an_unlabelled_second_axis() -> None:
    data = xr.Dataset(coords={"nmeasure": ["A", "B"]})
    data["aggregation_error_covariance"] = (
        ("nmeasure", "nmeasure_cov"),
        np.array([[2.0, 0.4], [0.4, 1.0]]),
    )

    result = resolve_aggregation_error(data)

    assert result.covariance is not None
    np.testing.assert_array_equal(result.covariance["nmeasure_cov"], ["A", "B"])
def test_low_rank_covariance_uses_factor_and_residual_diagonal() -> None:
    factor = np.array([[1.0, 0.0], [0.5, 0.25], [0.0, 0.5]])
    residual = np.array([0.2, 0.3, 0.4])
    data = _inputs()
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), factor)
    data["diagonal_residual_variance"] = ("nmeasure", residual)

    result = resolve_aggregation_error(data)

    assert result.mode == "low_rank"
    np.testing.assert_allclose(result.marginal_variance, np.sum(factor**2, axis=1) + residual)


@pytest.mark.parametrize("mode", ["dense", "low_rank", "diagonal", "none"])
def test_aggregation_error_low_rank_conversion_preserves_covariance(mode: str) -> None:
    """The model-owned conversion retains each validated covariance exactly."""
    factor = np.array([[0.3], [0.1], [0.2]])
    diagonal = np.array([0.2, 0.3, 0.4])
    covariance = factor @ factor.T + np.diag(diagonal)
    data = _inputs()
    if mode == "dense":
        data["aggregation_error_covariance"] = (("nmeasure", "nmeasure_cov"), covariance)
    elif mode == "low_rank":
        data["low_rank_factor"] = (("nmeasure", "agg_rank"), factor)
        data["diagonal_residual_variance"] = ("nmeasure", diagonal)
    elif mode == "diagonal":
        data["aggregation_error_sd"] = ("nmeasure", np.sqrt(diagonal))

    result = resolve_aggregation_error(data, mode)
    converted_factor, converted_diagonal = aggregation_error_as_low_rank(result)
    expected = covariance if mode in ("dense", "low_rank") else np.diag(diagonal if mode == "diagonal" else np.zeros(3))
    np.testing.assert_allclose(converted_factor @ converted_factor.T + np.diag(converted_diagonal), expected)


def test_low_rank_payloads_materialize_together_and_remain_eager() -> None:
    executions = 0

    @delayed
    def shared_payload() -> np.ndarray:
        nonlocal executions
        executions += 1
        return np.array([1.0, 0.0, 0.5, 0.25, 0.2, 0.3])

    shared = da.from_delayed(shared_payload(), shape=(6,), dtype=float)
    data = xr.Dataset(coords={"nmeasure": ["A", "B"]})
    data["low_rank_factor"] = xr.DataArray(
        shared[:4].reshape((2, 2)),
        dims=("nmeasure", "agg_rank"),
    )
    data["diagonal_residual_variance"] = xr.DataArray(
        shared[4:],
        dims="nmeasure",
    )

    result = resolve_aggregation_error(data)

    assert executions == 1
    assert result.factor is not None
    assert result.diagonal_variance is not None
    assert isinstance(result.factor.data, np.ndarray)
    assert isinstance(result.diagonal_variance.data, np.ndarray)
    assert executions == 1


def test_low_rank_checks_both_payload_structures_before_materializing() -> None:
    executions = 0

    @delayed
    def factor_payload() -> np.ndarray:
        nonlocal executions
        executions += 1
        return np.ones((2, 1))

    data = xr.Dataset(coords={"nmeasure": ["A", "B"], "other": [0, 1]})
    data["low_rank_factor"] = xr.DataArray(
        da.from_delayed(factor_payload(), shape=(2, 1), dtype=float),
        dims=("nmeasure", "agg_rank"),
    )
    data["diagonal_residual_variance"] = xr.DataArray(
        da.ones(2, chunks=2),
        dims="other",
    )

    with pytest.raises(ValueError, match="diagonal_residual_variance.*dims"):
        resolve_aggregation_error(data)

    assert executions == 0


def test_auto_rejects_two_structured_representations() -> None:
    data = _inputs()
    data["aggregation_error_covariance"] = (
        ("nmeasure", "nmeasure_cov"),
        np.eye(3),
    )
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), np.ones((3, 1)))
    data["diagonal_residual_variance"] = ("nmeasure", np.ones(3))

    with pytest.raises(ValueError, match="both dense and low-rank"):
        resolve_aggregation_error(data)


@pytest.mark.parametrize(
    ("covariance", "match"),
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_dense_covariance_validation(covariance: np.ndarray, match: str) -> None:
    data = xr.Dataset(
        {"aggregation_error_covariance": (("nmeasure", "nmeasure_cov"), covariance)},
        coords={"nmeasure": np.arange(2)},
    )

    with pytest.raises(ValueError, match=match):
        resolve_aggregation_error(data)


def test_structured_covariance_rejects_inconsistent_sd_diagnostic() -> None:
    data = _inputs()
    data["aggregation_error_covariance"] = (
        ("nmeasure", "nmeasure_cov"),
        np.eye(3),
    )
    data["aggregation_error_sd"] = ("nmeasure", np.full(3, 2.0))

    with pytest.raises(ValueError, match="diagnostic.*square root"):
        resolve_aggregation_error(data)


def test_explicit_none_ignores_available_diagnostic() -> None:
    data = _inputs()
    data["aggregation_error_sd"] = ("nmeasure", np.ones(3))

    result = resolve_aggregation_error(data, "none")

    assert result.mode == "none"
    np.testing.assert_array_equal(result.marginal_variance, np.zeros(3))


def test_prepare_low_rank_aggregation_error_preserves_diagonal() -> None:
    covariance_values = np.array(
        [
            [2.0, 0.8, 0.3],
            [0.8, 1.5, 0.2],
            [0.3, 0.2, 0.7],
        ]
    )
    covariance = xr.DataArray(
        da.from_array(covariance_values, chunks=(2, 2)),
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": ["A", "B", "C"], "nmeasure_cov": ["A", "B", "C"]},
    )

    result = prepare_low_rank_aggregation_error(covariance, rank=2)

    aggregation_error = result.aggregation_error
    assert aggregation_error.mode == "low_rank"
    assert aggregation_error.factor is not None
    assert aggregation_error.diagonal_variance is not None
    assert isinstance(aggregation_error.factor.data, np.ndarray)
    assert aggregation_error.factor.dims == ("nmeasure", "agg_rank")
    assert aggregation_error.factor.shape == (3, 2)
    reconstructed = (
        aggregation_error.factor.values @ aggregation_error.factor.values.T
        + np.diag(aggregation_error.diagonal_variance.values)
    )
    np.testing.assert_allclose(np.diag(reconstructed), np.diag(covariance_values), atol=1e-12)
    assert result.source_covariance_sha256.startswith("sha256:")
    assert result.diagnostics["requested_rank"] == 2
    assert result.diagnostics["actual_rank"] == 2
    assert 0.0 < result.diagnostics["retained_positive_spectral_fraction"] <= 1.0
    assert result.diagnostics["diagonal_preservation_error"] < 1e-12
    json.dumps(result.diagnostics, allow_nan=False)


def test_full_rank_aggregation_error_approximation_is_exact() -> None:
    covariance_values = np.array([[2.0, 0.4], [0.4, 1.0]])
    covariance = xr.DataArray(
        covariance_values,
        dims=("observation", "observation_cov"),
        coords={"observation": ["A", "B"], "observation_cov": ["A", "B"]},
    )

    result = prepare_low_rank_aggregation_error(
        covariance,
        rank=2,
        output_dim="observation",
        covariance_dim="observation_cov",
    )

    aggregation_error = result.aggregation_error
    assert aggregation_error.factor is not None
    assert aggregation_error.diagonal_variance is not None
    reconstructed = (
        aggregation_error.factor.values @ aggregation_error.factor.values.T
        + np.diag(aggregation_error.diagonal_variance.values)
    )
    np.testing.assert_allclose(reconstructed, covariance_values, atol=1e-12)
    assert result.diagnostics["relative_frobenius_reconstruction_error"] < 1e-12


@pytest.mark.parametrize("rank", [0, 4, True, 1.5])
def test_prepare_low_rank_aggregation_error_rejects_invalid_rank(rank: object) -> None:
    covariance = xr.DataArray(
        np.eye(3),
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": ["A", "B", "C"], "nmeasure_cov": ["A", "B", "C"]},
    )

    with pytest.raises(ValueError, match="rank"):
        prepare_low_rank_aggregation_error(covariance, rank=rank)  # type: ignore[arg-type]


def test_prepare_low_rank_aggregation_error_requires_exact_covariance_labels() -> None:
    covariance = xr.DataArray(
        np.eye(2),
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["B", "A"]},
    )

    with pytest.raises(ValueError, match="same values in the same order"):
        prepare_low_rank_aggregation_error(covariance, rank=1)


@pytest.mark.parametrize(
    ("covariance_values", "match"),
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
    ],
)
def test_prepare_low_rank_aggregation_error_validates_values(
    covariance_values: np.ndarray,
    match: str,
) -> None:
    covariance = xr.DataArray(
        covariance_values,
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["A", "B"]},
    )

    with pytest.raises(ValueError, match=match):
        prepare_low_rank_aggregation_error(covariance, rank=1)


def test_low_rank_psd_tolerance_scales_with_covariance_magnitude() -> None:
    covariance = xr.DataArray(
        np.array([[1.0, 2.0], [2.0, 1.0]]) * 1.0e-20,
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["A", "B"]},
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        prepare_low_rank_aggregation_error(covariance, rank=1)
