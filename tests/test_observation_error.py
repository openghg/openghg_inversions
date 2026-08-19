import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.specs import RhimeModelSpec
from openghg_inversions.observation_error import (
    AggregationError,
    resolve_aggregation_error,
    validate_aggregation_error_alignment,
    validate_observation_alignment,
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


def test_observation_alignment_uses_only_index_values_not_attrs_or_aux_coords() -> None:
    observations = xr.DataArray(
        [1.0, 2.0],
        dims="nmeasure",
        coords={"nmeasure": ["A", "B"], "qc_flag": ("nmeasure", [0, 1])},
    )
    candidate = xr.DataArray(
        [3.0, 4.0],
        dims="nmeasure",
        coords={"nmeasure": xr.DataArray(["A", "B"], dims="nmeasure", attrs={"source": "other"})},
    )

    validate_observation_alignment(
        observations,
        candidate,
        input_name="candidate",
        owner="test",
    )


@pytest.mark.parametrize(
    "aggregation_error",
    [
        AggregationError(
            mode="none",
            marginal_variance=np.ones(2),
        ),
        AggregationError(
            mode="dense",
            marginal_variance=np.ones(2),
            covariance=xr.DataArray(
                np.diag([1.0, 2.0]),
                dims=("nmeasure", "nmeasure_cov"),
                coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["A", "B"]},
            ),
        ),
        AggregationError(
            mode="low_rank",
            marginal_variance=np.ones(2),
            factor=xr.DataArray(
                [[1.0], [2.0]],
                dims=("nmeasure", "rank"),
                coords={"nmeasure": ["A", "B"]},
            ),
            diagonal_variance=xr.DataArray(
                [0.0, 0.0],
                dims="nmeasure",
                coords={"nmeasure": ["A", "B"]},
            ),
        ),
        AggregationError(
            mode="diagonal",
            marginal_variance=np.ones(2),
            covariance=xr.DataArray(
                np.eye(2),
                dims=("nmeasure", "nmeasure_cov"),
                coords={"nmeasure": ["A", "B"], "nmeasure_cov": ["A", "B"]},
            ),
            diagonal_variance=xr.DataArray(
                [1.0, 1.0],
                dims="nmeasure",
                coords={"nmeasure": ["A", "B"]},
            ),
        ),
    ],
)
def test_direct_aggregation_error_requires_coherent_mode_payload_and_marginal(
    aggregation_error: AggregationError,
) -> None:
    observations = xr.DataArray(
        [10.0, 11.0],
        dims="nmeasure",
        coords={"nmeasure": ["A", "B"]},
    )

    with pytest.raises(ValueError, match="payload|marginal variance"):
        validate_aggregation_error_alignment(
            observations,
            aggregation_error,
            owner="Test likelihood",
        )


def test_direct_low_rank_aggregation_error_requires_observations_on_first_axis() -> None:
    observations = xr.DataArray(
        [10.0, 11.0],
        dims="nmeasure",
        coords={"nmeasure": ["A", "B"]},
    )
    factor = xr.DataArray(
        np.eye(2),
        dims=("rank", "nmeasure"),
        coords={"nmeasure": ["A", "B"]},
    )
    aggregation_error = AggregationError(
        mode="low_rank",
        marginal_variance=np.ones(2),
        factor=factor,
        diagonal_variance=xr.zeros_like(observations),
    )

    with pytest.raises(ValueError, match="observation rows"):
        validate_aggregation_error_alignment(
            observations,
            aggregation_error,
            owner="Test likelihood",
        )


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
