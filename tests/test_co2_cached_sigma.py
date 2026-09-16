"""Named CO2 cached fixed-OU model and joint-output checks."""

from __future__ import annotations

import json
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_cached_sigma_model,
    run_rhime_co2_cached_sigma,
)
from openghg_inversions.rhime.co2 import co2_cached_sigma_runner
from openghg_inversions.rhime.co2 import co2_cached_sigma_model
from openghg_inversions.rhime.sampling import RhimeSampler


def _inputs() -> xr.Dataset:
    nmeasure = np.arange(4)
    state = np.asarray(["biosphere", "fossil"])
    design = np.asarray(
        [
            [0.8, 0.1],
            [0.4, 0.3],
            [0.2, 0.7],
            [0.5, 0.2],
        ]
    )
    fixed = np.asarray([0.1, 0.2, 0.1, 0.2])
    observations = fixed + design @ np.ones(2) + np.asarray([0.05, -0.04, 0.02, -0.03])
    result = xr.Dataset(
        {
            "H": (("nmeasure", "region"), design),
            "alpha_prior_mean": (("region",), np.ones(2)),
            "alpha_prior_covariance": (
                ("region", "region_cov"),
                np.asarray([[0.08, 0.01], [0.01, 0.06]]),
            ),
            "fixed_prior_contribution": (("nmeasure",), fixed),
            "aggregation_error_covariance": (
                ("nmeasure", "nmeasure_cov"),
                np.asarray(
                    [
                        [0.02, 0.004, 0.002, 0.0],
                        [0.004, 0.03, 0.0, 0.003],
                        [0.002, 0.0, 0.02, 0.004],
                        [0.0, 0.003, 0.004, 0.03],
                    ]
                ),
            ),
            "mf": (("nmeasure",), observations),
            "mf_error": (("nmeasure",), np.full(4, 0.1)),
        },
        coords={
            "nmeasure": nmeasure,
            "region": state,
            "site": ("nmeasure", ["AAA", "AAA", "BBB", "BBB"]),
            "time": (
                "nmeasure",
                np.asarray(
                    [
                        "2021-01-01T00",
                        "2021-01-01T03",
                        "2021-01-01T01",
                        "2021-01-01T05",
                    ],
                    dtype="datetime64[h]",
                ),
            ),
        },
    )
    for name in ("fixed_prior_contribution", "mf", "mf_error"):
        result[name].attrs["units"] = "ppm"
    return result


def _boundary_inputs() -> xr.Dataset:
    inputs = _inputs().assign_coords(
        basis_group=("region", ["inner", "outer"]),
        time=(
            "nmeasure",
            np.asarray(
                [
                    "2021-01-01T00",
                    "2021-02-01T03",
                    "2021-01-01T01",
                    "2021-02-01T05",
                ],
                dtype="datetime64[h]",
            ),
        ),
    )
    bc_index = pd.MultiIndex.from_product(
        [["north", "east", "south", "west"], ["2021-01", "2021-02"]],
        names=("bc_curtain", "bc_period"),
    )
    bc_coords = xr.Coordinates.from_pandas_multiindex(bc_index, "bc_region")
    boundary_design = np.arange(32, dtype=float).reshape(4, 8) / 50.0
    observation_period = np.asarray(["2021-01", "2021-02", "2021-01", "2021-02"])
    boundary_period = np.asarray(bc_index.get_level_values("bc_period"))
    boundary_design[observation_period[:, None] != boundary_period[None, :]] = 0.0
    inputs["H_bc"] = xr.DataArray(
        boundary_design,
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": inputs["nmeasure"], **bc_coords},
        attrs={"units": "ppm"},
    )
    return inputs


def _dense_fixed_ou_covariance(
    inputs: xr.Dataset,
    amplitude: np.ndarray,
    tau_hours: dict[str, float],
) -> np.ndarray:
    covariance = (
        np.asarray(inputs["aggregation_error_covariance"].values)
        + np.diag(np.square(inputs["mf_error"].values))
    )
    sites = np.asarray(inputs["site"].values)
    times = inputs["time"].values.astype("datetime64[m]").astype(np.int64) / 60.0
    for site_index, site in enumerate(("AAA", "BBB")):
        rows = np.flatnonzero(sites == site)
        lag = np.abs(times[rows, None] - times[None, rows])
        covariance[np.ix_(rows, rows)] += amplitude[site_index] ** 2 * np.exp(
            -lag / tau_hours[site]
        )
    return covariance


def _build(*, state_activity: StateActivity | None = None, **kwargs: Any) -> Any:
    inputs = _inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    return build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes={"BBB": 0.4, "unused": 9.0, "AAA": 0.3},
        state_activity=state_activity,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("options", "required_component"),
    [
        (
            {"bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1}},
            "boundary_sensitivity",
        ),
        ({"bc_state_activity": StateActivity()}, "boundary_sensitivity"),
        ({"offset_freq": "monthly"}, "offset_prior"),
        ({"offset_drop_first": True}, "offset_prior"),
        ({"offset_per_site": False}, "offset_prior"),
    ],
)
def test_cached_builder_rejects_options_for_absent_components(
    options: dict[str, Any],
    required_component: str,
) -> None:
    """Component options cannot silently disappear from the public builder."""
    with pytest.raises(ValueError, match=required_component):
        _build(**options)


def test_cached_input_names_do_not_auto_select_prepared_boundary() -> None:
    """Cached input selection leaves an unrequested prepared boundary untouched."""
    inputs = _boundary_inputs()

    class PreparedInputsStub:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

    names = co2_cached_sigma_runner.co2_cached_sigma_input_names(
        cast(Any, PreparedInputsStub()),
    )

    assert "H_bc" not in names


def test_named_cached_co2_model_owns_normalized_potential_and_sampler_inputs() -> None:
    cached = _build()

    assert {
        "flux_scaling_latent",
        "flux_scaling",
        "modelled_concentration",
        "ou_site_amplitude",
        "ou_site_index",
        "cached_fixed_ou_likelihood",
    } <= set(cached.model.named_vars)
    assert "y" not in cached.model.named_vars
    assert "cached_linear_state" not in cached.model.named_vars
    assert cached.target.n_state == 2
    assert cached.target.n_group == 2
    assert cached.target.prepared.site_labels == ("AAA", "BBB")
    np.testing.assert_allclose(cached.target.prepared.tau_hours_by_site, [3.0, 7.0])

    point = cached.model.initial_point()
    modelled_mean, actual = cached.model.compile_fn(
        cached.model.replace_rvs_by_values(
            [cached.modelled_mean, cached.model["cached_fixed_ou_likelihood"]]
        ),
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(point)
    expected = cached.target.log_likelihood_from_mean(modelled_mean, [0.3, 0.4])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)


def test_active_prior_is_prepared_once_for_graph_and_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_prior: dict[str, CorrelatedLognormalPrior] = {}
    original_add_state = (
        co2_cached_sigma_model._add_prepared_correlated_lognormal_state_with_activity
    )

    def capture_graph_prior(*args: Any, **kwargs: Any) -> Any:
        graph_prior["value"] = args[1]
        return original_add_state(*args, **kwargs)

    monkeypatch.setattr(
        co2_cached_sigma_model,
        "_add_prepared_correlated_lognormal_state_with_activity",
        capture_graph_prior,
    )
    cached = _build(
        state_activity=StateActivity(
            active=xr.DataArray(
                [True, False],
                dims="region",
                coords={"region": ["biosphere", "fossil"]},
            )
        )
    )
    step_args: dict[str, Any] = {}
    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "make_cached_sigma_compound_step",
        lambda **kwargs: step_args.update(kwargs) or object(),
    )

    co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(chains=1),
        cached_model=cached,
        sigma_target_accept=0.81,
        state_target_accept=0.92,
    )

    assert graph_prior["value"].mean.sizes == {"region_flux_scaling_active": 1}
    assert step_args["initial_cache"] is cached.initial_cache
    assert step_args["sigma_target_accept"] == 0.81
    assert step_args["state_target_accept"] == 0.92
    assert step_args["modelled_mean"] is cached.modelled_mean
    assert step_args["states"] == cached.states
    assert "state_location" not in step_args
    assert "state_cholesky" not in step_args


def test_cached_boundary_and_default_site_offset_match_dense_oracle_at_varied_points() -> None:
    """Cached boundary and offset terms match the dense likelihood and gradients."""
    inputs = _boundary_inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    bc_active = xr.DataArray(
        [True, False, True, False, True, False, True, False],
        dims="bc_region",
        coords={"bc_region": inputs["bc_region"]},
    )
    bc_fixed = xr.DataArray(
        np.linspace(0.8, 1.5, 8),
        dims="bc_region",
        coords={"bc_region": inputs["bc_region"]},
    )
    cached = build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes={"AAA": 0.3, "BBB": 0.4},
        state_activity=StateActivity(
            active=xr.DataArray(
                [True, False],
                dims="region",
                coords={"region": inputs["region"]},
            ),
            fixed_value=xr.DataArray(
                [1.0, 1.4],
                dims="region",
                coords={"region": inputs["region"]},
            ),
        ),
        boundary_sensitivity=inputs["H_bc"],
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        bc_state_activity=StateActivity(active=bc_active, fixed_value=bc_fixed),
        offset_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
    )

    value_vars = [cached.model.rvs_to_values[rv] for rv in cached.states]
    likelihood = cached.model.replace_rvs_by_values(
        [cached.model["cached_fixed_ou_likelihood"]]
    )[0]
    outputs = cached.model.replace_rvs_by_values(
        [
            cached.model["modelled_concentration"],
            cached.model["fixed_prior_contribution"],
            cached.model["co2_flux_contribution"],
            cached.model["mu_bc"],
            cached.model["offset"],
            cached.model["cached_fixed_ou_likelihood"],
            cached.model["flux_scaling"],
        ]
    )
    compiled = cached.model.compile_fn(
        [*outputs, *pt.grad(likelihood, value_vars)],
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )
    mean_fn = cached.model.compile_fn(
        outputs[0],
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )
    base_point = cached.model.initial_point()
    state_names = [cast(str, variable.name) for variable in value_vars]
    assert state_names == ["flux_scaling_active_latent", "bc_active", "offset_latent"]
    varied_values = [
        ([-0.4], [0.7, 1.1, 0.9, 1.3], [-0.2, 0.4]),
        ([0.2], [1.4, 0.6, 1.2, 0.8], [0.5, -0.1]),
        ([0.7], [0.9, 1.5, 0.5, 1.1], [0.15, 0.65]),
    ]
    covariance = _dense_fixed_ou_covariance(
        inputs,
        np.asarray([0.3, 0.4]),
        {"AAA": 3.0, "BBB": 7.0},
    )

    def varied_point(values: tuple[list[float], list[float], list[float]]) -> dict[str, Any]:
        point = {name: np.asarray(value).copy() for name, value in base_point.items()}
        for name, value in zip(state_names, values, strict=True):
            point[name] = np.asarray(value, dtype=point[name].dtype)
        return point

    def dense_logp(point: dict[str, Any]) -> float:
        return float(
            multivariate_normal.logpdf(
                inputs["mf"].values,
                mean=np.asarray(mean_fn(point)),
                cov=covariance,
            )
        )

    gradient_values: list[np.ndarray] | None = None
    gradient_point: dict[str, Any] | None = None
    for index, values in enumerate(varied_values):
        point = varied_point(values)
        evaluated = list(map(np.asarray, compiled(point)))
        total, fixed, flux, boundary, offset, cached_logp, flux_scaling = evaluated[:7]
        np.testing.assert_allclose(total, flux + boundary + offset)
        np.testing.assert_allclose(
            flux,
            fixed + inputs["H"].values @ flux_scaling,
        )
        logp_rtol = 2.0e-6 if np.asarray(cached_logp).dtype == np.float32 else 2.0e-10
        assert float(cached_logp) == pytest.approx(dense_logp(point), rel=logp_rtol)
        if index == 1:
            gradient_values = evaluated[7:]
            gradient_point = point

    assert gradient_values is not None and gradient_point is not None
    gradient_dtype = np.result_type(*(value.dtype for value in gradient_values))
    is_float32 = gradient_dtype == np.dtype("float32")
    step = 1.0e-3 if is_float32 else 1.0e-6
    gradient_rtol = 2.0e-3 if is_float32 else 2.0e-5
    gradient_atol = 2.0e-4 if is_float32 else 2.0e-6
    for name, actual in zip(state_names, gradient_values, strict=True):
        expected = np.empty_like(actual, dtype=np.float64)
        for index in np.ndindex(actual.shape):
            plus = {key: value.copy() for key, value in gradient_point.items()}
            minus = {key: value.copy() for key, value in gradient_point.items()}
            plus[name][index] += step
            minus[name][index] -= step
            expected[index] = (dense_logp(plus) - dense_logp(minus)) / (2.0 * step)
        np.testing.assert_allclose(actual, expected, rtol=gradient_rtol, atol=gradient_atol)

    assert cached.target.n_state == 1 + int(bc_active.sum()) + 2
    assert len(cached.states) == 3
    assert "cached_linear_state" not in cached.model.named_vars
    registry = get_coord_registry(cached.model)
    assert registry is not None
    assert registry.original_coords["bc_region"].equals(inputs.indexes["bc_region"])
    np.testing.assert_array_equal(cached.model["bc_is_active"].eval(), bc_active)


@pytest.mark.parametrize(
    (
        "offset_prior",
        "offset_per_site",
        "offset_freq",
        "expected_offset_states",
        "expected_sampler_name",
    ),
    [
        (
            {"pdf": "normal", "mu": 0.2, "sigma": 0.1},
            False,
            None,
            1,
            "offset_latent",
        ),
        (
            {"pdf": "normal", "mu": 0.2, "sigma": 0.1},
            True,
            "monthly",
            4,
            "offset_latent",
        ),
        (
            {
                "pdf": "lognormal",
                "mean": 0.2,
                "stdev": 0.1,
                "reparameterise": True,
            },
            True,
            None,
            2,
            "offset_latent_latent",
        ),
    ],
)
def test_cached_offset_modes_match_completed_mean_and_dense_likelihood(
    offset_prior: dict[str, Any],
    offset_per_site: bool,
    offset_freq: str | None,
    expected_offset_states: int,
    expected_sampler_name: str,
) -> None:
    """Each supported offset layout enters the completed cached likelihood once."""
    inputs = _boundary_inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    cached = build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        offset_prior=offset_prior,
        offset_per_site=offset_per_site,
        offset_freq=offset_freq,
    )

    variables = cached.model.replace_rvs_by_values(
        [
            cached.model["modelled_concentration"],
            cached.model["fixed_prior_contribution"],
            cached.model["co2_flux_contribution"],
            cached.model["flux_scaling"],
            cached.model["offset"],
            cached.model["cached_fixed_ou_likelihood"],
        ]
    )
    compiled = cached.model.compile_fn(
        variables,
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )
    point = cached.model.initial_point()
    for state in cached.states:
        value_name = cast(str, cached.model.rvs_to_values[state].name)
        point[value_name] = np.full_like(point[value_name], 0.25)
    total, fixed, flux, flux_scaling, offset, cached_logp = map(
        np.asarray,
        compiled(point),
    )
    covariance = _dense_fixed_ou_covariance(
        inputs,
        np.asarray([0.75, 0.75]),
        {"AAA": 3.0, "BBB": 7.0},
    )

    np.testing.assert_allclose(total, flux + offset)
    np.testing.assert_allclose(
        flux,
        fixed + inputs["H"].values @ flux_scaling,
    )
    logp_rtol = 2.0e-6 if np.asarray(cached_logp).dtype == np.float32 else 2.0e-10
    assert float(cached_logp) == pytest.approx(
        multivariate_normal.logpdf(inputs["mf"].values, mean=total, cov=covariance),
        rel=logp_rtol,
    )
    assert cached.target.n_state == inputs.sizes["region"] + expected_offset_states
    assert cached.states[-1].name == expected_sampler_name


@pytest.mark.parametrize("active_component", ["boundary", "offset"])
def test_cached_all_fixed_flux_uses_other_active_component(
    active_component: str,
) -> None:
    """A boundary or offset can carry the cache when every flux state is fixed."""
    inputs = _boundary_inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    fixed_flux = np.asarray([1.25, 0.75])
    kwargs: dict[str, Any]
    if active_component == "boundary":
        kwargs = {
            "boundary_sensitivity": inputs["H_bc"],
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        }
        expected_state_names = ("bc",)
        expected_state_count = inputs.sizes["bc_region"]
    else:
        kwargs = {
            "offset_prior": {"pdf": "normal", "mu": 0.2, "sigma": 0.1},
        }
        expected_state_names = ("offset_latent",)
        expected_state_count = 2

    cached = build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        state_activity=StateActivity(active=False, fixed_value=fixed_flux),
        **kwargs,
    )

    assert "flux_scaling_latent" not in cached.model.named_vars
    assert tuple(state.name for state in cached.states) == expected_state_names
    assert cached.target.n_state == expected_state_count
    np.testing.assert_allclose(
        cached.target.fixed_contribution,
        inputs["fixed_prior_contribution"].values + inputs["H"].values @ fixed_flux,
    )
    mean, potential = cached.model.compile_fn(
        cached.model.replace_rvs_by_values(
            [
                cached.model["modelled_concentration"],
                cached.model["cached_fixed_ou_likelihood"],
            ]
        ),
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(cached.model.initial_point())
    assert float(potential) == pytest.approx(
        cached.target.log_likelihood_from_mean(mean, np.asarray([0.75, 0.75]))
    )


def test_cached_rejects_model_without_active_affine_coefficients() -> None:
    """Reject a cached graph with no coefficient for its state transition."""
    inputs = _inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )

    with pytest.raises(ValueError, match="at least one active flux, boundary, or offset"):
        build_co2_cached_sigma_model(
            inputs["H"],
            retained_prior=prior,
            fixed_prior_contribution=inputs["fixed_prior_contribution"],
            observations=inputs["mf"],
            observation_error=inputs["mf_error"],
            aggregation_error=resolve_aggregation_error(inputs, "dense"),
            tau_hours={"AAA": 3.0, "BBB": 7.0},
            site_amplitude_prior_scale=0.75,
            state_activity=StateActivity(active=False),
        )


@pytest.mark.parametrize(
    ("offset_freq", "expected_offset_states", "expected_cached_states"),
    [
        (None, 2, 11),
        ("monthly", 4, 13),
    ],
)
def test_cached_state_dimensions_include_boundary_and_offset_coefficients(
    offset_freq: str | None,
    expected_offset_states: int,
    expected_cached_states: int,
) -> None:
    """The cached design includes every active boundary and offset coefficient."""
    inputs = _boundary_inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    cached = build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        state_activity=StateActivity(active=np.asarray([True, False])),
        boundary_sensitivity=inputs["H_bc"],
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        offset_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
        offset_freq=offset_freq,
    )

    assert cached.model["offset_latent"].shape.eval().item() == expected_offset_states
    assert cached.target.design.shape == (inputs.sizes["nmeasure"], expected_cached_states)
    assert cached.initial_cache.precision.shape == (
        expected_cached_states,
        expected_cached_states,
    )


def test_joint_outputs_are_exact_and_predict_complete_correlated_vectors() -> None:
    cached = _build()
    draws = 6_000
    state = np.broadcast_to(np.asarray([1.0, 1.0]), (1, draws, 2)).copy()
    sigma = np.broadcast_to(np.asarray([0.5, 0.8]), (1, draws, 2)).copy()
    inputs = _inputs()
    modelled_mean = np.broadcast_to(
        inputs["fixed_prior_contribution"].values + inputs["H"].values @ np.ones(2),
        (1, draws, 4),
    ).copy()
    trace = az.from_dict(
        posterior={
            "flux_scaling": state,
            "modelled_concentration": modelled_mean,
            "ou_site_amplitude": sigma,
        },
        dims={
            "flux_scaling": ["region"],
            "modelled_concentration": ["nmeasure"],
            "ou_site_amplitude": ["ou_site"],
        },
        coords={
            "region": ["biosphere", "fossil"],
            "nmeasure": inputs["nmeasure"].values,
            "ou_site": ["AAA", "BBB"],
        },
    )

    result = co2_cached_sigma_runner._append_joint_outputs(
        trace,
        cached_model=cached,
        observations=inputs["mf"],
        posterior_predictive=True,
        random_seed=42,
    )

    expected_logp = cached.target.log_likelihood_from_mean(
        modelled_mean[0, 0],
        [0.5, 0.8],
    )
    np.testing.assert_allclose(result.log_likelihood["y"], expected_logp)
    assert result.log_likelihood["y"].dims == ("chain", "draw")
    assert result.log_likelihood["y"].attrs["rhime_likelihood_scope"] == (
        "joint_observation_vector"
    )
    predictive = np.asarray(result.posterior_predictive["y"]).reshape(draws, 4)
    expected_covariance = (
        inputs["aggregation_error_covariance"].values
        + np.diag(inputs["mf_error"].values ** 2)
    )
    expected_covariance[:2, :2] += 0.5**2 * np.asarray(
        [[1.0, np.exp(-1.0)], [np.exp(-1.0), 1.0]]
    )
    expected_covariance[2:, 2:] += 0.8**2 * np.asarray(
        [[1.0, np.exp(-4.0 / 7.0)], [np.exp(-4.0 / 7.0), 1.0]]
    )
    np.testing.assert_allclose(
        np.cov(predictive, rowvar=False),
        expected_covariance,
        atol=0.035,
    )
    assert result.posterior_predictive["y"].attrs["rhime_predictive_scope"] == (
        "joint_observation_vector"
    )


def test_named_runner_samples_real_graph_and_labels_cached_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named runner samples the real graph and labels every cached output."""
    inputs = _boundary_inputs()
    step_settings: dict[str, float] = {}
    original_make_step = co2_cached_sigma_runner.make_cached_sigma_compound_step

    def capture_step_settings(**kwargs: Any) -> pm.CompoundStep:
        step_settings["sigma_target_accept"] = kwargs["sigma_target_accept"]
        step_settings["state_target_accept"] = kwargs["state_target_accept"]
        return original_make_step(**kwargs)

    class PreparedInputsStub:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

        def validated(self) -> "PreparedInputsStub":
            return self

    selected: list[tuple[str, ...]] = []

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        selected.append(variable_names)
        return inputs

    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "materialize_pymc_inputs",
        materialize,
    )
    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "make_cached_sigma_compound_step",
        capture_step_settings,
    )
    sampler = RhimeSampler(
        draws=2,
        tune=2,
        chains=1,
        progressbar=False,
        sample_kwargs={
            "random_seed": 147,
            "cores": 1,
            "compute_convergence_checks": False,
        },
        sample_prior_predictive=False,
        posterior_predictive_kwargs={"random_seed": 148},
    )

    result = run_rhime_co2_cached_sigma(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes=0.4,
        sampler=sampler,
        sigma_target_accept=0.82,
        state_target_accept=0.93,
        use_bc=True,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        bc_state_activity=StateActivity(
            active=np.asarray([True, False, False, False, False, False, False, False]),
            fixed_value=1.0,
        ),
        offset_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
    )

    assert step_settings == {
        "sigma_target_accept": 0.82,
        "state_target_accept": 0.93,
    }
    assert len(selected) == 1
    assert selected[0].count("H_bc") == 1
    assert result.posterior["flux_scaling"].shape == (1, 2, 2)
    assert result.posterior["bc"].shape == (1, 2, 8)
    assert result.posterior["offset"].shape == (1, 2, 4)
    assert result.posterior["offset_latent"].shape == (1, 2, 2)
    assert result.posterior["offset_latent"].dims == (
        "chain",
        "draw",
        "offset_term",
    )
    np.testing.assert_array_equal(
        result.posterior["offset_latent"].coords["offset_term"],
        ["AAA", "BBB"],
    )
    assert result.posterior["offset_latent"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].shape == (1, 2, 4)
    assert result.log_likelihood["y"].shape == (1, 2)
    roles = json.loads(result.attrs["rhime_variable_roles"])
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert roles["boundary_concentration"] == "mu_bc"
    assert roles["boundary_scale"] == "bc"
    assert roles["boundary_sensitivity"] == "hbc"
    assert "baseline_concentration" not in roles
    assert "baseline_scale" not in roles
    assert json.loads(result.posterior["bc"].attrs["rhime_scientific_roles"]) == [
        "boundary_scale"
    ]
    assert json.loads(result.posterior["mu_bc"].attrs["rhime_scientific_roles"]) == [
        "boundary_concentration"
    ]
    assert metadata["recipe"] == "co2_cached_sigma_fixed_ou"
    assert "sampler" not in metadata
    assert "numerical_preparation" not in metadata
    assert "rhime_sampler_provenance" not in result.attrs
    assert set(result.posterior["ou_site_amplitude"].coords["ou_site"].values) == {
        "AAA",
        "BBB",
    }
    assert result.posterior["ou_site_amplitude"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert json.loads(result.posterior_predictive["y"].attrs["rhime_scientific_roles"]) == [
        "concentration"
    ]
    assert "units" not in result.log_likelihood["y"].attrs
    assert json.loads(result.log_likelihood["y"].attrs["rhime_scientific_roles"]) == [
        "joint_log_likelihood"
    ]
    assert result.posterior.attrs["rhime_recipe"] == "co2_cached_sigma_fixed_ou"


def test_cached_runner_rejects_generic_target_accept() -> None:
    """The cached runner rejects one target acceptance rate for its two samplers."""
    class PreparedInputsStub:
        inv_inputs = _inputs()
        aggregation_error_mode = "dense"

        def validated(self) -> "PreparedInputsStub":
            return self

    prepared = PreparedInputsStub()
    prepared.rhime_inputs = cast(Any, prepared)

    with pytest.raises(
        ValueError,
        match="sigma_target_accept.*state_target_accept",
    ):
        run_rhime_co2_cached_sigma(
            prepared_inputs=cast(Any, prepared),
            tau_hours={"AAA": 3.0, "BBB": 7.0},
            site_amplitude_prior_scale=0.75,
            sampler=RhimeSampler(sample_kwargs={"target_accept": 0.95}),
        )
