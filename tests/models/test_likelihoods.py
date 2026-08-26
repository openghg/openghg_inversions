from typing import Any, cast

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.models.additive_sigma import (
    _resolve_site_sigma_prior,
    add_additive_sigma_gaussian_likelihood,
    additive_sigma_likelihood_builder,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.gaussian_likelihood import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.models.pollution_event import build_pollution_event_error
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.sigma import SigmaAlignment


def _add_pollution_event_likelihood(
    data: xr.Dataset,
    /,
    *,
    mu: Any,
    mu_bc: Any | None,
    sigprior: dict[str, Any],
    sigma_alignment: SigmaAlignment,
    offset: Any | None = None,
    power: dict[str, Any] | float = 1.99,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    retain_unused_sigma: bool = False,
    aggregation_error_mode: Any = "none",
) -> None:
    """Compose the ordinary mean and add the built-in error and distribution."""
    baseline = mu_bc
    if offset is not None:
        baseline = offset if baseline is None else baseline + offset
    mean = mu if baseline is None else mu + baseline
    state = build_pollution_event_error(
        observations=data["mf"],
        observation_error=data["mf_error"],
        minimum_error=data["min_error"],
        aggregation_error=resolve_aggregation_error(data, aggregation_error_mode),
        pollution_mean=mu,
        pollution_event_baseline=baseline,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigprior,
        power=power,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        retain_unused_sigma=retain_unused_sigma,
    )
    add_gaussian_observation_likelihood(
        observed=state.observed,
        mean=mean,
        independent_variance=state.independent_variance,
        aggregation_error=state.aggregation_error,
        output_dim="nmeasure",
    )


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
    with registered_model() as model:
        registered_aggregation_error = add_aggregation_error_data(
            aggregation_error,
            data["mf"],
            output_dim="nmeasure",
        )
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
            aggregation_error=registered_aggregation_error,
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


@pytest.mark.parametrize(
    ("mode", "payload_names"),
    [
        ("diagonal", {"diagonal_residual_variance"}),
        ("dense", {"aggregation_error_covariance"}),
        ("low_rank", {"low_rank_factor", "diagonal_residual_variance"}),
    ],
)
def test_aggregation_error_is_preserved_as_labelled_constant_data(
    mode: str,
    payload_names: set[str],
) -> None:
    """Every fixed aggregation term used by the graph remains inspectable."""
    data = _base_data()
    if mode == "diagonal":
        data["aggregation_error_sd"] = ("nmeasure", [0.1, 0.2, 0.3])
    elif mode == "dense":
        data["aggregation_error_covariance"] = (
            ("nmeasure", "nmeasure_cov"),
            np.diag([0.1, 0.2, 0.3]),
        )
    else:
        data, _ = _low_rank_data()

    model = _fixed_likelihood_model(data)
    with model:
        predictive = pm.sample_prior_predictive(draws=1, var_names=["y"])

    expected_names = payload_names | {"aggregation_error_marginal_variance"}
    assert expected_names.issubset(predictive.constant_data.data_vars)
    assert predictive.constant_data["aggregation_error_marginal_variance"].dims == (
        "nmeasure",
    )
    if mode == "dense":
        assert predictive.constant_data["aggregation_error_covariance"].dims == (
            "nmeasure",
            "nmeasure_cov",
        )


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
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        mu = pm.Data("mu_input", np.ones(3), dims="nmeasure")
        _add_pollution_event_likelihood(
            data,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.5, "upper": 0.5001},
            sigma_alignment=sigma_alignment,
            aggregation_error_mode=cast(Any, mode),
        )

    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), 2.0)
    assert "model_error" not in model.named_vars


def test_no_model_error_retains_legacy_sigma_graph_and_observation_error_floor() -> None:
    """Disabling model error preserves the legacy graph but epsilon uses only data error."""
    data = _base_data()
    data["mf_error"] = ("nmeasure", np.array([0.0, 0.3, 0.4]))
    data["min_error"] = ("nmeasure", np.full(3, 20.0))
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        mu = pm.Data("mu_input", np.ones(3), dims="nmeasure")
        _add_pollution_event_likelihood(
            data,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=sigma_alignment,
            no_model_error=True,
            retain_unused_sigma=True,
        )

    small_amount = 1e-12 * np.nanmean(data["mf"].values)
    expected = np.maximum(np.abs(data["mf_error"].values), small_amount)
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)
    assert "model_error" not in model.named_vars
    assert model["sigma"] in model.free_RVs
    assert {"sigma", "sigma_site_index", "sigma_period_index"}.issubset(model.named_vars)


def test_no_model_error_omits_unused_sigma_by_default() -> None:
    """The ordinary component keeps compatibility-only variables out of the graph."""
    data = _base_data()
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        mu = pm.Data("mu_input", np.ones(3), dims="nmeasure")
        _add_pollution_event_likelihood(
            data,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=sigma_alignment,
            no_model_error=True,
        )

    assert "sigma" not in model.named_vars


def test_pollution_event_validation_names_malformed_input_and_owner() -> None:
    """Malformed observation errors fail at their owning likelihood boundary."""
    data = _base_data()
    data["mf_error"] = ("nmeasure", [0.2, np.nan, 0.4])
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )

    with registered_model(coords={"nmeasure": np.arange(3)}):
        pollution_mean = pm.Data("pollution_mean", np.ones(3), dims="nmeasure")
        with pytest.raises(
            ValueError,
            match="Pollution-event likelihood input 'observation_error'.*finite",
        ):
            build_pollution_event_error(
                observations=data["mf"],
                observation_error=data["mf_error"],
                minimum_error=data["min_error"],
                aggregation_error=resolve_aggregation_error(data, "none"),
                pollution_mean=pollution_mean,
                pollution_event_baseline=None,
                sigma_alignment=sigma_alignment,
                sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                power=1.99,
                pollution_events_from_obs=False,
                no_model_error=False,
            )


def test_additive_sigma_validation_names_malformed_input_and_owner() -> None:
    """Malformed reported error fails before the additive component builds."""
    data = _base_data()
    malformed_error = data["mf_error"].rename(nmeasure="sample")
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )

    with registered_model(coords={"nmeasure": np.arange(3)}):
        with pytest.raises(
            ValueError,
            match="Additive-sigma likelihood input 'observation_error'.*dims",
        ):
            add_additive_sigma_gaussian_likelihood(
                observations=data["mf"],
                observation_error=malformed_error,
                aggregation_error=resolve_aggregation_error(data, "none"),
                mean=pm.math.constant(np.ones(3)),
                sigma_alignment=sigma_alignment,
                sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            )


@pytest.mark.parametrize("component", ["pollution-event", "additive-sigma"])
def test_likelihoods_reject_reordered_observation_error_coordinates(
    component: str,
) -> None:
    """The registry rejects a conflicting observation order during registration."""
    data = _base_data()
    reordered_error = data["mf_error"].isel(nmeasure=[2, 1, 0])
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )

    with registered_model(coords={"nmeasure": np.arange(3)}):
        with pytest.raises(
            ValueError,
            match="Conflicting coord registration for 'nmeasure': values differ",
        ):
            if component == "pollution-event":
                pollution_mean = pm.Data("pollution_mean", np.ones(3), dims="nmeasure")
                build_pollution_event_error(
                    observations=data["mf"],
                    observation_error=reordered_error,
                    minimum_error=data["min_error"],
                    aggregation_error=resolve_aggregation_error(data, "none"),
                    pollution_mean=pollution_mean,
                    pollution_event_baseline=None,
                    sigma_alignment=sigma_alignment,
                    sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                    power=1.99,
                    pollution_events_from_obs=False,
                    no_model_error=False,
                )
            else:
                add_additive_sigma_gaussian_likelihood(
                    observations=data["mf"],
                    observation_error=reordered_error,
                    aggregation_error=resolve_aggregation_error(data, "none"),
                    mean=pm.math.constant(np.ones(3)),
                    sigma_alignment=sigma_alignment,
                    sigma_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                )


def test_additive_sigma_likelihood_applies_minimum_error_floor() -> None:
    """The complete additive likelihood floors total marginal error."""
    data = _base_data()
    data["min_error"] = ("nmeasure", np.array([0.0, 1.0, 0.0]))
    data["aggregation_error_sd"] = ("nmeasure", np.array([0.1, 0.2, 0.3]))
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        add_additive_sigma_gaussian_likelihood(
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, "diagonal"),
            mean=pm.math.constant(np.ones(3)),
            sigma_alignment=sigma_alignment,
            sigma_prior={"pdf": "uniform", "lower": 0.5, "upper": 0.500001},
        )

    sigma = np.asarray(model.named_vars["sigma"].eval()).item()
    unconstrained_scale = np.sqrt(
        data["mf_error"].values ** 2
        + sigma**2
        + data["aggregation_error_sd"].values ** 2
    )
    expected_scale = np.maximum(unconstrained_scale, data["min_error"].values)
    np.testing.assert_allclose(model["epsilon"].eval(), expected_scale)


def test_additive_sigma_likelihood_omits_optional_sigma_and_minimum_error() -> None:
    """The fixed-error form has no disconnected optional variables."""
    data = _base_data()
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        add_additive_sigma_gaussian_likelihood(
            observations=data["mf"],
            observation_error=data["mf_error"],
            aggregation_error=resolve_aggregation_error(data, "none"),
            mean=pm.math.constant(np.ones(3)),
        )

    assert "sigma" not in model.named_vars
    assert "min_error" not in model.named_vars
    assert "Y" not in model.named_vars
    np.testing.assert_allclose(model["epsilon"].eval(), data["mf_error"].values)


def test_additive_sigma_gaussian_likelihood_uses_completed_mean() -> None:
    """The installed likelihood consumes the complete recipe-owned mean."""
    data = _base_data()
    data["mf_error"] = ("nmeasure", np.full(3, 0.5))
    completed_mean = np.array([0.75, 1.5, 2.75])
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        mean = pm.Data("completed_mean", completed_mean, dims="nmeasure")
        likelihood = add_additive_sigma_gaussian_likelihood(
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, "none"),
            mean=mean,
            sigma_alignment=sigma_alignment,
            sigma_prior={"pdf": "uniform", "lower": 0.2, "upper": 0.200001},
        )

    assert likelihood is model.named_vars["y"]
    np.testing.assert_allclose(model.named_vars["y"].owner.inputs[-2].eval(), completed_mean)
    expected_scale = np.sqrt(
        data["mf_error"].values ** 2
        + np.squeeze(np.asarray(model.named_vars["sigma"].eval())) ** 2
    )
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected_scale)


def test_additive_sigma_likelihood_builder_uses_only_common_inputs() -> None:
    """The model-owned builder derives sigma alignment without PEFO inputs."""
    data = _base_data()
    data = data.assign_coords(
        site=("nmeasure", ["MHD", "MHD", "TAC"]),
        time=(
            "nmeasure",
            np.array(["2019-01-01", "2019-01-02", "2019-01-03"], dtype="datetime64[ns]"),
        ),
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        mean = pm.Data("completed_mean", np.ones(3), dims="nmeasure")
        likelihood = additive_sigma_likelihood_builder(
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, "none"),
            mean=mean,
            sigma_prior={"pdf": "uniform", "lower": 0.2, "upper": 0.200001},
            sigma_freq="1d",
            sigma_per_site=False,
            output_dim="nmeasure",
        )

    assert likelihood is model.named_vars["y"]
    np.testing.assert_allclose(model.named_vars["y"].owner.inputs[-2].eval(), np.ones(3))


def test_additive_sigma_likelihood_builder_aligns_site_prior_scales() -> None:
    site = xr.DataArray(["TAC", "MHD", "TAC"], dims="nmeasure")

    resolved = _resolve_site_sigma_prior(
        {
            "pdf": "halfnormal",
            "sigma": {"MHD": 5.0, "unused": 9.0, "TAC": 2.0},
        },
        site,
        per_site=True,
    )

    assert resolved["pdf"] == "halfnormal"
    np.testing.assert_array_equal(resolved["sigma"], [[2.0], [5.0]])


def test_additive_sigma_likelihood_builder_rejects_incomplete_site_prior_scales() -> None:
    site = xr.DataArray(["MHD", "TAC"], dims="nmeasure")

    with pytest.raises(ValueError, match="missing retained site.*TAC"):
        _resolve_site_sigma_prior(
            {"pdf": "halfnormal", "sigma": {"MHD": 5.0}},
            site,
            per_site=True,
        )

    with pytest.raises(ValueError, match="sigma_per_site=True"):
        _resolve_site_sigma_prior(
            {"pdf": "halfnormal", "sigma": {"MHD": 5.0, "TAC": 2.0}},
            site,
            per_site=False,
        )


def test_observation_derived_pollution_event_subtracts_complete_baseline() -> None:
    """Boundary and offset terms are both excluded from the pollution event."""
    data = _base_data()
    data["mf_error"] = ("nmeasure", np.zeros(3))
    sigma_alignment = SigmaAlignment.from_frequency(
        data["site_indicator"], frequency=None, per_site=False
    )
    with registered_model(coords={"nmeasure": np.arange(3)}) as model:
        pollution_mean = pm.Data("mu_input", np.zeros(3), dims="nmeasure")
        boundary_mean = pm.Data("mu_bc_input", np.full(3, 0.25), dims="nmeasure")
        offset = pm.Data("offset_input", np.full(3, 0.5), dims="nmeasure")
        _add_pollution_event_likelihood(
            data,
            mu=pollution_mean,
            mu_bc=boundary_mean,
            offset=offset,
            sigprior={"pdf": "uniform", "lower": 0.5, "upper": 0.500001},
            sigma_alignment=sigma_alignment,
            pollution_events_from_obs=True,
            power=2.0,
        )

    sigma = np.asarray(model.named_vars["sigma"].eval()).item()
    expected = np.abs(data["mf"].values - 0.25 - 0.5) * sigma
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)
