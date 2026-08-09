from typing import Any, cast

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.likelihoods import add_gaussian_observation_likelihood
from openghg_inversions.models.rhime_likelihood import add_rhime_likelihood_component
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.sigma import SigmaAlignment


def _base_data() -> xr.Dataset:
    return xr.Dataset(
        {
            "mf": ("nmeasure", [1.0, 2.0, 3.0]),
            "mf_error": ("nmeasure", [0.2, 0.3, 0.4]),
            "min_error": ("nmeasure", [0.0, 0.0, 0.0]),
            "site_indicator": ("nmeasure", [0, 0, 0]),
        },
        coords={"nmeasure": np.arange(3)},
    )


def _low_rank_data() -> tuple[xr.Dataset, np.ndarray]:
    data = _base_data()
    factor = np.array([[0.5, 0.0], [0.2, 0.3], [0.0, 0.2]])
    residual = np.array([0.25, 0.27, 0.26])
    data["low_rank_factor"] = (("nmeasure", "agg_rank"), factor)
    data["diagonal_residual_variance"] = ("nmeasure", residual)
    return data, factor @ factor.T + np.diag(residual)


def _fixed_likelihood_model(data: xr.Dataset) -> pm.Model:
    aggregation_error = resolve_aggregation_error(data)
    with pm.Model(coords={"nmeasure": np.arange(3)}) as model:
        observed = pm.Data("Y", pm.floatX(data["mf"].values), dims="nmeasure")
        mean = pm.Data("mean", pm.floatX(np.array([0.8, 1.7, 2.9])), dims="nmeasure")
        independent_variance = pm.Data(
            "independent_variance",
            pm.floatX(data["mf_error"].values**2),
            dims="nmeasure",
        )
        add_gaussian_observation_likelihood(
            observed=observed,
            mean=mean,
            independent_variance=independent_variance,
            aggregation_error=aggregation_error,
            output_dim="nmeasure",
        )
    return model


def test_low_rank_logp_matches_dense_gaussian() -> None:
    low_rank_data, covariance = _low_rank_data()
    dense_data = _base_data()
    dense_data["aggregation_error_covariance"] = (
        ("nmeasure", "nmeasure_cov"),
        covariance,
    )
    low_rank_model = _fixed_likelihood_model(low_rank_data)
    dense_model = _fixed_likelihood_model(dense_data)
    expected = multivariate_normal.logpdf(
        low_rank_data["mf"].values,
        mean=np.array([0.8, 1.7, 2.9]),
        cov=covariance + np.diag(low_rank_data["mf_error"].values**2),
    )

    low_rank_logp = float(low_rank_model.compile_logp()(low_rank_model.initial_point()))
    dense_logp = float(dense_model.compile_logp()(dense_model.initial_point()))

    assert low_rank_logp == pytest.approx(expected, rel=1e-6)
    assert low_rank_logp == pytest.approx(dense_logp, rel=1e-6)


def test_low_rank_likelihood_retains_observed_y_and_predictive_sampling() -> None:
    data, _ = _low_rank_data()
    model = _fixed_likelihood_model(data)

    with model:
        predictive = pm.sample_prior_predictive(draws=2, var_names=["y"])

    assert [rv.name for rv in model.observed_RVs] == ["y"]
    assert predictive.prior_predictive["y"].shape == (1, 2, 3)


@pytest.mark.parametrize("mode", ["diagonal", "dense", "low_rank"])
def test_min_error_is_a_floor_on_total_marginal_sd(mode: str) -> None:
    data = _base_data()
    data["min_error"] = ("nmeasure", np.full(3, 2.0))
    aggregation_variance = np.array([0.5, 0.6, 0.7])
    if mode == "diagonal":
        data["aggregation_error_sd"] = ("nmeasure", np.sqrt(aggregation_variance))
    elif mode == "dense":
        data["aggregation_error_covariance"] = (
            ("nmeasure", "nmeasure_cov"),
            np.diag(aggregation_variance),
        )
    else:
        data["low_rank_factor"] = (("nmeasure", "agg_rank"), np.zeros((3, 1)))
        data["diagonal_residual_variance"] = ("nmeasure", aggregation_variance)

    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with pm.Model(coords={"nmeasure": np.arange(3)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(3), dims="nmeasure")
        add_rhime_likelihood_component(
            data,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.5, "upper": 0.5001},
            sigma_alignment=sigma_alignment,
            aggregation_error_mode=cast(Any, mode),
        )

    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), 2.0)
    assert "model_error" not in model.named_vars


def test_no_model_error_ignores_floor_but_includes_structured_marginal() -> None:
    data, covariance = _low_rank_data()
    data["min_error"] = ("nmeasure", np.full(3, 20.0))
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with pm.Model(coords={"nmeasure": np.arange(3)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(3), dims="nmeasure")
        add_rhime_likelihood_component(
            data,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=sigma_alignment,
            no_model_error=True,
        )

    expected = np.sqrt(data["mf_error"].values**2 + np.diag(covariance))
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)
    assert "model_error" not in model.named_vars
