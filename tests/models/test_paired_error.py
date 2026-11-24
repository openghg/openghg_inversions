"""Fixed pair correlations of independent errors are exact in every Gaussian likelihood."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.models import registered_model
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.paired_error import PairedErrorCorrelation
from openghg_inversions.observation_error import AggregationError

LABELS = ["a", "b", "c", "d", "e"]
SD = np.array([0.5, 0.4, 6.0, 5.0, 0.3])
PAIRS = PairedErrorCorrelation(
    first=np.array([0, 1]), second=np.array([2, 3]), correlation=np.array([-0.9, 0.6])
)
# A configured zero correlation keeps its pair labels but needs no factor column.
ZERO_PAIR = PairedErrorCorrelation(
    first=np.array([0, 1]), second=np.array([2, 3]), correlation=np.array([-0.9, 0.0])
)


def _aggregation(mode: str) -> tuple[AggregationError | None, np.ndarray]:
    diagonal = np.array([0.03, 0.02, 0.5, 0.4, 0.01])
    factor = np.array([[0.1], [0.05], [-0.8], [0.3], [0.02]])
    dense = factor @ factor.T + np.diag(diagonal)
    coords = {"nmeasure": LABELS}
    if mode == "none":
        return None, np.zeros((5, 5))
    if mode == "diagonal":
        return AggregationError(
            mode="diagonal",
            marginal_variance=diagonal,
            diagonal_variance=xr.DataArray(diagonal, dims="nmeasure", coords=coords),
        ), np.diag(diagonal)
    if mode == "low_rank":
        return AggregationError(
            mode="low_rank",
            marginal_variance=np.diag(dense),
            factor=xr.DataArray(factor, dims=("nmeasure", "rank"), coords=coords),
            diagonal_variance=xr.DataArray(diagonal, dims="nmeasure", coords=coords),
        ), dense
    return AggregationError(
        mode="dense",
        marginal_variance=np.diag(dense),
        covariance=xr.DataArray(
            dense, dims=("nmeasure", "nmeasure_cov"), coords={"nmeasure": LABELS, "nmeasure_cov": LABELS}
        ),
    ), dense


def _observations() -> tuple[xr.DataArray, xr.DataArray]:
    times = np.datetime64("2021-07-01T00") + np.array([0, 1, 0, 1, 2]).astype("timedelta64[h]")
    # OU groups by site, so each channel is its own group here.
    sites = ["HFD-co2", "HFD-co2", "HFD-o2", "HFD-o2", "HFD-co2"]
    coords = {"nmeasure": LABELS, "site": ("nmeasure", sites), "time": ("nmeasure", times)}
    observations = xr.DataArray([400.0, 401.0, -700.0, -705.0, 402.0], dims="nmeasure", coords=coords)
    return observations, observations.copy(data=SD)


def test_split_reproduces_the_paired_block_and_validates_pairs():
    reduction, factor = PAIRS.split(SD)
    np.testing.assert_allclose(
        np.diag(SD**2 - reduction) + factor @ factor.T, np.diag(SD**2) + PAIRS.covariance(SD), atol=1e-12
    )
    assert np.all(SD**2 - reduction > 0)
    np.testing.assert_array_equal(PAIRS.partner(5), [2, 3, 0, 1, -1])
    np.testing.assert_allclose(PAIRS.row_correlation(5), [-0.9, 0.6, -0.9, 0.6, 0.0])
    for first, second, rho in (([0], [0], [0.1]), ([0, 1], [2, 0], [0.1, 0.1]), ([0], [1], [1.0])):
        with pytest.raises(ValueError):
            PairedErrorCorrelation(np.array(first), np.array(second), np.array(rho))


def test_zero_correlation_pairs_add_no_factor_columns():
    """Only correlated pairs cost a factor column; the split stays exact."""
    from openghg_inversions.models.paired_error import RegisteredPairedError

    reduction, factor = ZERO_PAIR.split(SD)
    assert factor.shape == (5, 1)
    np.testing.assert_allclose(reduction, [0.9 * SD[0] ** 2, 0.0, 0.9 * SD[2] ** 2, 0.0, 0.0])
    np.testing.assert_allclose(
        np.diag(SD**2 - reduction) + factor @ factor.T,
        np.diag(SD**2) + ZERO_PAIR.covariance(SD),
        atol=1e-12,
    )
    registered = RegisteredPairedError(
        pairs=ZERO_PAIR, correlation=pm.math.constant(ZERO_PAIR.row_correlation(5))
    )
    np.testing.assert_allclose(registered.factor(pm.math.constant(SD)).eval(), factor)
    none = PairedErrorCorrelation(np.array([0]), np.array([2]), np.array([0.0]))
    assert none.split(SD)[1].shape == (5, 0)
    np.testing.assert_array_equal(none.partner(5), [2, -1, 0, -1, -1])


@pytest.mark.usefixtures("double_precision")
@pytest.mark.parametrize("pairs", [PAIRS, ZERO_PAIR], ids=["correlated", "zero-pair"])
@pytest.mark.parametrize("mode", ["none", "diagonal", "low_rank", "dense"])
def test_additive_likelihood_matches_dense_paired_covariance(mode, pairs):
    observations, error = _observations()
    aggregation, aggregation_covariance = _aggregation(mode)
    mean = observations.values + np.array([0.2, -0.1, 3.0, -2.0, 0.05])
    with registered_model() as model:
        add_additive_sigma_likelihood(
            observations=observations,
            observation_error=error,
            aggregation_error=aggregation,
            mean=pm.math.constant(mean),
            output_dim="nmeasure",
            paired_error_correlation=pairs,
        )
    covariance = aggregation_covariance + np.diag(SD**2) + pairs.covariance(SD)
    expected = multivariate_normal.logpdf(observations.values, mean=mean, cov=covariance)
    np.testing.assert_allclose(model.compile_logp()({}), expected, rtol=1e-10)
    epsilon = model.compile_fn(model["epsilon"])({})
    np.testing.assert_allclose(epsilon, np.sqrt(np.diag(covariance)), rtol=1e-12)
    np.testing.assert_allclose(model["paired_error_correlation"].get_value(), pairs.row_correlation(5))


@pytest.mark.usefixtures("double_precision")
@pytest.mark.parametrize("pairs", [PAIRS, ZERO_PAIR], ids=["correlated", "zero-pair"])
def test_fixed_ou_likelihood_matches_dense_paired_covariance(pairs):
    observations, error = _observations()
    aggregation, aggregation_covariance = _aggregation("dense")
    mean = observations.values + np.array([0.2, -0.1, 3.0, -2.0, 0.05])
    amplitude, tau = 0.7, 3.0
    with registered_model() as model:
        add_fixed_ou_gaussian_likelihood(
            observations=observations,
            observation_error=error,
            aggregation_error=aggregation,
            mean=pm.math.constant(mean),
            tau_hours=tau,
            fixed_site_amplitudes=amplitude,
            output_dim="nmeasure",
            paired_error_correlation=pairs,
        )
    hours = (observations.time.values - observations.time.values[0]) / np.timedelta64(1, "h")
    same_site = observations.site.values[:, None] == observations.site.values[None, :]
    ou = amplitude**2 * np.exp(-np.abs(hours[:, None] - hours[None, :]) / tau) * same_site
    covariance = aggregation_covariance + np.diag(SD**2) + pairs.covariance(SD) + ou
    expected = multivariate_normal.logpdf(observations.values, mean=mean, cov=covariance)
    np.testing.assert_allclose(model.compile_logp()({}), expected, rtol=1e-10)
    np.testing.assert_allclose(
        model.compile_fn(model["epsilon"])({}), np.sqrt(np.diag(covariance)), rtol=1e-12
    )
