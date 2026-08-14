import numpy as np
import pytest
import xarray as xr

from openghg_inversions.models import RhimeModelSpec
from openghg_inversions.observation_error import resolve_aggregation_error


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


def test_low_rank_covariance_uses_factor_and_residual_diagonal() -> None:
    factor = np.array([[1.0, 0.0], [0.5, 0.25], [0.0, 0.5]])
    residual = np.array([0.2, 0.3, 0.4])
    data = _inputs()
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), factor)
    data["diagonal_residual_variance"] = ("nmeasure", residual)

    result = resolve_aggregation_error(data)

    assert result.mode == "low_rank"
    np.testing.assert_allclose(result.marginal_variance, np.sum(factor**2, axis=1) + residual)


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
