import numpy as np
import dask.array as da
from dask import delayed
import pytest
import xarray as xr

from openghg_inversions.rhime.specs import RhimeModelSpec
from openghg_inversions.observation_error import (
    resolve_aggregation_error,
    validate_complete_observation_covariance,
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


def test_optional_complete_covariance_check_uses_lrpd_structure() -> None:
    data = xr.Dataset(coords={"nmeasure": ["A", "B"]})
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), np.eye(2))
    data["diagonal_residual_variance"] = ("nmeasure", np.zeros(2))

    validate_complete_observation_covariance(
        resolve_aggregation_error(data),
        np.zeros(2),
    )


def test_optional_complete_covariance_check_rejects_singular_lrpd() -> None:
    data = xr.Dataset(coords={"nmeasure": ["A", "B"]})
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), np.ones((2, 1)))
    data["diagonal_residual_variance"] = ("nmeasure", np.zeros(2))

    with pytest.raises(ValueError, match="positive definite"):
        validate_complete_observation_covariance(
            resolve_aggregation_error(data),
            np.zeros(2),
        )


def test_optional_complete_covariance_check_rejects_singular_dense() -> None:
    data = xr.Dataset(coords={"nmeasure": ["A", "B"]})
    data["aggregation_error_covariance"] = (
        ("nmeasure", "nmeasure_cov"),
        np.ones((2, 2)),
    )

    with pytest.raises(ValueError, match="positive definite"):
        validate_complete_observation_covariance(
            resolve_aggregation_error(data),
            np.zeros(2),
        )


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


def test_model_spec_rejects_unknown_aggregation_error_mode() -> None:
    with pytest.raises(ValueError, match="aggregation_error_mode.*dense.*low_rank"):
        RhimeModelSpec(
            species="ch4",
            domain="EUROPE",
            sectors=(),
            aggregation_error_mode="factorized",  # type: ignore[arg-type]
        )
