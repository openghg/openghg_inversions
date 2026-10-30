"""Named CO2 cached fixed-OU model and joint-output checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
from pymc.step_methods import CompoundStep
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import halfnorm, multivariate_normal, norm
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn

from openghg_inversions.inference.cached_sigma import make_cached_sigma_compound_step
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.rhime.co2.inference import sample_co2_inference
from openghg_inversions.rhime.co2 import (
    build_co2_cached_sigma_model,
)
from openghg_inversions.rhime.co2 import co2_runner
from openghg_inversions.rhime.co2 import co2_cached_sigma_runner
from openghg_inversions.rhime.co2 import co2_cached_sigma_model
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import load_trace, save_trace


@pytest.fixture(autouse=True)
def dense_oracle_precision():
    """Keep strict double-precision oracles independent of package import order."""
    with pytensor.config.change_flags(floatX="float64"):
        yield


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


def _build(*, state_activity: StateActivity | None = None, inputs: xr.Dataset | None = None, **kwargs: Any) -> Any:
    inputs = _inputs() if inputs is None else inputs
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
        site_amplitude_prior=kwargs.pop("site_amplitude_prior", {"pdf": "halfnormal", "sigma": 0.75}),
        initial_site_amplitudes=kwargs.pop("initial_site_amplitudes", {"BBB": 0.4, "unused": 9.0, "AAA": 0.3}),
        state_activity=state_activity,
        **kwargs,
    )


def _initial_point(cached):
    return make_initial_point_fn(
        model=cached.model, overrides=cached.initial_values, return_transformed=True,
    )(769)


def _compound(cached):
    return make_cached_sigma_compound_step(
        model=cached.model, sigma=cached.amplitude, states=cached.states,
        modelled_mean=cached.modelled_mean, target=cached.target,
        shared_cache=cached.shared_cache, initial_cache=cached.initial_cache,
        cached_state_logp=cached.cached_state_logp, initial_point=_initial_point(cached),
        rng=np.random.default_rng(769),
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


def test_named_cached_co2_model_owns_one_observed_likelihood_and_sampler_inputs() -> None:
    cached = _build()

    assert {
        "flux_scaling_latent",
        "flux_scaling",
        "modelled_concentration",
        "ou_site_amplitude",
        "ou_site_index",
        "y",
    } <= set(cached.model.named_vars)
    assert not cached.model.potentials
    assert [rv.name for rv in cached.model.observed_RVs] == ["y"]
    assert "cached_fixed_ou_likelihood" not in cached.model.named_vars
    assert "cached_linear_state" not in cached.model.named_vars
    assert cached.target.n_state == 2
    assert cached.target.n_group == 2
    assert cached.target.prepared.site_labels == ("AAA", "BBB")
    np.testing.assert_allclose(cached.target.prepared.tau_hours_by_site, [3.0, 7.0])

    point = _initial_point(cached)
    modelled_mean, actual = cached.model.compile_fn(
        cached.model.replace_rvs_by_values(
            [cached.modelled_mean, cached.cached_state_logp]
        ),
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(point)
    expected = cached.target.log_likelihood_from_mean(modelled_mean, [0.3, 0.4])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(cached.model.compile_logp(vars=[cached.model["y"]])(point), expected)
    assert all(value is None for value in cached.model.rvs_to_initial_values.values())
    np.testing.assert_array_equal(cached.initial_values["ou_site_amplitude"], [0.3, 0.4])


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
    assert step_args["cached_state_logp"] is cached.cached_state_logp
    assert "state_location" not in step_args
    assert "state_cholesky" not in step_args


@pytest.mark.parametrize("positive_offset", [False, True])
def test_lifted_prior_starts_and_caller_transforms_preserve_complete_state_target(positive_offset):
    inputs = _boundary_inputs()
    cached = _build(
        inputs=inputs, boundary_sensitivity=inputs["H_bc"],
        bc_prior={"pdf": "lognormal", "mu": 0.0, "sigma": 0.1, "initval": np.full(8, 0.8)},
        offset_prior=(
            {"pdf": "lognormal", "mu": -1.6, "sigma": 0.3, "initval": "prior"}
            if positive_offset else {"pdf": "normal", "mu": 0.2, "sigma": 0.1, "initval": "prior"}
        ),
    )
    assert all(value is None for value in cached.model.rvs_to_initial_values.values())
    np.testing.assert_array_equal(cached.initial_values["bc"].eval(), np.full(8, 0.8))
    assert cached.initial_values["offset_latent"] == "prior"
    offset_value_name = cached.model.rvs_to_values[cached.states[-1]].name
    sampler = co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(chains=3, sample_kwargs={"initvals": [
            {"ou_site_amplitude_log__": np.log([0.6, 0.7]), "bc_log__": np.log(np.full(8, 0.9))},
            {"ou_site_amplitude": [0.2, 0.5], "bc": np.full(8, 1.1)},
            None,
        ]}),
        cached_model=cached, sigma_target_accept=0.8, state_target_accept=0.9,
    )
    starts = [make_initial_point_fn(
        model=cached.model, overrides=overrides, return_transformed=True,
    )(seed=seed) for overrides, seed in zip(sampler.sample_kwargs["initvals"], [769, 770, 771], strict=True)]
    np.testing.assert_allclose(starts[0]["ou_site_amplitude_log__"], np.log([0.6, 0.7]))
    np.testing.assert_allclose(starts[1]["ou_site_amplitude_log__"], np.log([0.2, 0.5]))
    np.testing.assert_allclose(starts[0]["bc_log__"], np.log(0.9))
    np.testing.assert_allclose(starts[1]["bc_log__"], np.log(1.1))
    np.testing.assert_allclose(starts[2]["ou_site_amplitude_log__"], np.log([0.3, 0.4]))
    np.testing.assert_allclose(starts[2]["bc_log__"], np.log(0.8))
    assert not np.array_equal(starts[0][offset_value_name], starts[1][offset_value_name])
    repeated = make_initial_point_fn(
        model=cached.model, overrides=sampler.sample_kwargs["initvals"][0], return_transformed=True,
    )(seed=769)
    np.testing.assert_array_equal(starts[0][offset_value_name], repeated[offset_value_name])
    state_step = sampler.sample_kwargs["step"].methods[1]
    mean = cached.model.compile_fn(
        cached.model.replace_rvs_by_values([cached.modelled_mean])[0],
        inputs=cached.model.value_vars, on_unused_input="ignore",
    )
    public_logp = cached.model.compile_logp()
    public_gradient = cached.model.compile_dlogp(vars=list(cached.states))
    for point in starts:
        amplitude = np.exp(point["ou_site_amplitude_log__"])
        cached.shared_cache.update(cached.target.refresh(amplitude))
        state_step._logp_dlogp_func.set_extra_values(point)
        actual, gradient = state_step._logp_dlogp_func(DictToArrayBijection.map(
            {name: point[name] for name in state_step.var_names}
        ))
        # LogNormal physical density plus the log transform Jacobian equals
        # this Normal density in log space; omitting either term changes it.
        prior = (
            norm.logpdf(point["flux_scaling_latent"]).sum()
            + norm.logpdf(point["bc_log__"], scale=0.1).sum()
            + norm.logpdf(
                point[offset_value_name], loc=-1.6 if positive_offset else 0.2,
                scale=0.3 if positive_offset else 0.1,
            ).sum()
            + halfnorm.logpdf(amplitude, scale=0.75).sum()
            + point["ou_site_amplitude_log__"].sum()
        )
        expected = prior + multivariate_normal.logpdf(
            inputs["mf"].values, mean=mean(point),
            cov=_dense_fixed_ou_covariance(inputs, amplitude, {"AAA": 3.0, "BBB": 7.0}),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        np.testing.assert_allclose(actual, public_logp(point), atol=1e-8)
        np.testing.assert_allclose(
            gradient, public_gradient(point), atol=1e-8,
        )


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
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
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
        [cached.cached_state_logp]
    )[0]
    outputs = cached.model.replace_rvs_by_values(
        [
            cached.model["modelled_concentration"],
            cached.model["fixed_prior_contribution"],
            cached.model["co2_flux_contribution"],
            cached.model["mu_bc"],
            cached.model["offset"],
            cached.cached_state_logp,
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
    base_point = _initial_point(cached)
    state_names = [cast(str, variable.name) for variable in value_vars]
    assert state_names == ["flux_scaling_active_latent", "bc_active", "offset_latent"]
    varied_values = [
        ([-0.4], [0.7, 1.1, 0.9, 1.3], [-0.2, 0.4]),
        ([0.2], [1.4, 0.6, 1.2, 0.8], [0.5, -0.1]),
        ([0.7], [0.9, 1.5, 0.5, 1.1], [0.15, 0.65]),
    ]
    state_step = _compound(cached).methods[1]
    public_logp = cached.model.compile_logp()
    public_gradient = cached.model.compile_dlogp(vars=list(cached.states))

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
                cov=_dense_fixed_ou_covariance(
                    inputs, np.exp(point["ou_site_amplitude_log__"]), {"AAA": 3.0, "BBB": 7.0},
                ),
            )
        )

    gradient_values: list[np.ndarray] | None = None
    gradient_point: dict[str, Any] | None = None
    for index, values in enumerate(varied_values):
        point = varied_point(values)
        amplitude = np.asarray([(0.3, 0.4), (0.6, 0.2), (0.25, 0.8)][index])
        point["ou_site_amplitude_log__"] = np.log(amplitude)
        cached.shared_cache.update(cached.target.refresh(amplitude))
        evaluated = list(map(np.asarray, compiled(point)))
        total, fixed, flux, boundary, offset, cached_logp, flux_scaling = evaluated[:7]
        np.testing.assert_allclose(total, flux + boundary + offset)
        np.testing.assert_allclose(
            flux,
            fixed + inputs["H"].values @ flux_scaling,
        )
        logp_rtol = 2.0e-6 if np.asarray(cached_logp).dtype == np.float32 else 2.0e-10
        assert float(cached_logp) == pytest.approx(dense_logp(point), rel=logp_rtol)
        prior = (
            norm.logpdf(point[state_names[0]]).sum()
            + norm.logpdf(point[state_names[1]], loc=1.0, scale=0.1).sum()
            + norm.logpdf(point[state_names[2]], loc=0.2, scale=0.1).sum()
            + halfnorm.logpdf(amplitude, scale=0.75).sum()
            + np.log(amplitude).sum()
        )
        state_step._logp_dlogp_func.set_extra_values(point)
        state_point = DictToArrayBijection.map({name: point[name] for name in state_step.var_names})
        state_logp, state_gradient = state_step._logp_dlogp_func(state_point)
        np.testing.assert_allclose(state_logp, dense_logp(point) + prior, atol=1e-8)
        np.testing.assert_allclose(state_logp, public_logp(point), atol=1e-8)
        np.testing.assert_allclose(state_gradient, public_gradient(point), atol=1e-8)
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
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
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
            cached.cached_state_logp,
        ]
    )
    compiled = cached.model.compile_fn(
        variables,
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )
    point = _initial_point(cached)
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
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
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
                cached.cached_state_logp,
            ]
        ),
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(_initial_point(cached))
    assert float(potential) == pytest.approx(
        cached.target.log_likelihood_from_mean(mean, np.asarray([0.75, 0.75]))
    )
    point = _initial_point(cached)
    state_step = _compound(cached).methods[1]
    state_step._logp_dlogp_func.set_extra_values(point)
    actual, gradient = state_step._logp_dlogp_func(DictToArrayBijection.map(
        {name: point[name] for name in state_step.var_names}
    ))
    prior_location = 1.0 if active_component == "boundary" else 0.2
    prior = sum(norm.logpdf(point[name], loc=prior_location, scale=0.1).sum() for name in state_step.var_names)
    prior += halfnorm.logpdf([0.75, 0.75], scale=0.75).sum() + np.log([0.75, 0.75]).sum()
    np.testing.assert_allclose(actual, float(potential) + prior, atol=1e-7)
    np.testing.assert_allclose(actual, cached.model.compile_logp()(point), atol=1e-8)
    np.testing.assert_allclose(gradient, cached.model.compile_dlogp(vars=list(cached.states))(point), atol=1e-8)


@pytest.mark.parametrize("centered", [False, True])
def test_cached_all_fixed_states_infer_amplitudes(centered) -> None:
    """A fixed mean still has an inferred covariance and exact amplitude transition."""
    inputs = _inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"], inputs["alpha_prior_covariance"], covariance_dim="region_cov",
    )
    boundary_options = {}
    shift = np.zeros(4)
    if centered:
        operands = xr.Dataset({
            "H_bc_anomaly": xr.DataArray(np.full((4,1), .03), dims=("nmeasure", "bc_region"), attrs={"units": "ppm"}),
            "H_bc_mass": xr.DataArray(np.full((4,1), .1), dims=("nmeasure", "bc_region"), attrs={"units": "1"}),
            "bc_reference": xr.DataArray(400., attrs={"units": "ppm"}),
        }, coords={"nmeasure": inputs.nmeasure, "bc_region": ["constant"]}, attrs={"boundary_representation": "centered-v1"})
        shift[:] = 40.
        inputs["mf"] = inputs.mf + shift
        inputs.mf.attrs["units"] = "ppm"
        boundary_options = dict(boundary_operands=operands, bc_state_activity=StateActivity(active=False, fixed_value=1.2))
    cached = build_co2_cached_sigma_model(
        inputs["H"], retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"], observations=inputs["mf"],
        observation_error=inputs["mf_error"], aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0, "DROPPED": 9.0},
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        initial_site_amplitudes={"AAA": 0.5, "BBB": 0.7, "DROPPED": 1.0},
        state_activity=StateActivity(active=False),
        **boundary_options,
    )
    assert cached.states == ()
    assert cached.target.design.shape == (4, 0)
    amplitude = np.asarray([0.5, 0.7])
    expected = multivariate_normal.logpdf(
        inputs["mf"].values-shift, mean=cached.modelled_mean.eval(),
        cov=_dense_fixed_ou_covariance(inputs, amplitude, {"AAA": 3.0, "BBB": 7.0}),
    )
    np.testing.assert_allclose(cached.target.refresh(amplitude).log_likelihood(np.empty(0)), expected)
    sampler = co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(draws=3, tune=2, chains=1, nuts_sampler="pymc"), cached_model=cached,
        sigma_target_accept=0.8, state_target_accept=0.9,
    )
    step = sampler.sample_kwargs["step"]
    assert len(step.methods) == 1
    point = make_initial_point_fn(model=cached.model, overrides=cached.initial_values, return_transformed=True)(seed=3)
    updated, _ = step.step(point)
    assert np.isfinite(updated["ou_site_amplitude_log__"]).all()


@pytest.mark.parametrize("name", ["tau_hours", "initial_site_amplitudes"])
@pytest.mark.parametrize("invalid", [True, "bad", 0.0, -1.0, float("nan")])
def test_cached_mapping_validates_dropped_sites(name, invalid):
    inputs = _inputs()
    options = {"tau_hours": {"AAA": 3.0, "BBB": 7.0},
               "initial_site_amplitudes": {"AAA": 0.5, "BBB": 0.7}}
    options[name]["DROPPED"] = invalid
    with pytest.raises(ValueError, match="numeric|positive"):
        build_co2_cached_sigma_model(
            inputs["H"], retained_prior=CorrelatedLognormalPrior(
                inputs["alpha_prior_mean"], inputs["alpha_prior_covariance"], covariance_dim="region_cov"),
            fixed_prior_contribution=inputs["fixed_prior_contribution"], observations=inputs["mf"],
            observation_error=inputs["mf_error"], aggregation_error=resolve_aggregation_error(inputs, "dense"),
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75}, **options,
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
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
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


def _assert_native_joint_outputs(cached, covariance):
    """Check both native predictions and density with deliberately stale cache."""
    cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 3.0)))
    with cached.model:
        prior = pm.sample_prior_predictive(draws=1000, random_seed=769)
        repeated = pm.sample_prior_predictive(draws=1000, random_seed=769)
    np.testing.assert_array_equal(prior.prior_predictive.y, repeated.prior_predictive.y)
    means = prior.prior[cached.modelled_mean.name].values[0]
    amplitudes = prior.prior[cached.amplitude.name].values[0]
    residual = prior.prior_predictive.y.values[0] - means
    whitened = np.stack([
        np.linalg.solve(np.linalg.cholesky(covariance(sigma)), noise)
        for noise, sigma in zip(residual, amplitudes, strict=True)
    ])
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.16)
    np.testing.assert_allclose(np.cov(whitened.T), np.eye(cached.target.n_obs), atol=0.18)
    chains = []
    for chain in range(2):
        sections = [
            prior.prior.to_dataset().isel(chain=0, draw=2 * chain + half, drop=True)
            .expand_dims(draw=np.arange(half * 500, (half + 1) * 500))
            for half in range(2)
        ]
        chains.append(xr.concat(sections, dim="draw"))
    posterior = xr.concat(chains, dim=xr.IndexVariable("chain", [0, 1]))
    trace = xr.DataTree.from_dict({"posterior": posterior})
    cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 4.0)))
    with cached.model:
        predictions = pm.sample_posterior_predictive(
            trace, var_names=["y", cached.modelled_mean.name], random_seed=770, progressbar=False,
        )
        repeated = pm.sample_posterior_predictive(
            trace, var_names=["y"], random_seed=770, progressbar=False,
        )
    np.testing.assert_array_equal(predictions.posterior_predictive.y, repeated.posterior_predictive.y)
    np.testing.assert_allclose(
        predictions.posterior_predictive[cached.modelled_mean.name], posterior[cached.modelled_mean.name],
    )
    assert predictions.posterior_predictive.y.shape == (2, 1000, cached.target.n_obs)
    assert not np.allclose(posterior[cached.modelled_mean.name][0], posterior[cached.modelled_mean.name][1])
    pm.compute_log_likelihood(trace, model=cached.model, progressbar=False)
    assert trace.log_likelihood.y.dims == ("chain", "draw")
    for chain in range(2):
        for half in range(2):
            section = slice(half * 500, (half + 1) * 500)
            sigma = posterior[cached.amplitude.name].values[chain, section.start]
            mean = posterior[cached.modelled_mean.name].values[chain, section.start]
            dense = covariance(sigma)
            expected = multivariate_normal.logpdf(cached.target.observations, mean=mean, cov=dense)
            np.testing.assert_allclose(trace.log_likelihood.y.values[chain, section], expected, atol=1e-8)
            noise = predictions.posterior_predictive.y.values[chain, section] - mean
            whitened = np.linalg.solve(np.linalg.cholesky(dense), noise.T).T
            np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.18)
            np.testing.assert_allclose(np.cov(whitened.T), np.eye(cached.target.n_obs), atol=0.27)
    alternate = pt.dvector("alternate_observations")
    density = cached.model.replace_rvs_by_values([pm.logp(cached.model["y"], alternate)])[0]
    evaluate = cached.model.compile_fn(
        density, inputs=[alternate, *cached.model.value_vars], point_fn=False, on_unused_input="ignore",
    )
    point = _initial_point(cached)
    mean_fn = cached.model.compile_fn(
        cached.model.replace_rvs_by_values([cached.modelled_mean])[0],
        inputs=cached.model.value_vars, on_unused_input="ignore",
    )
    value = cached.target.observations + np.linspace(-0.4, 0.7, cached.target.n_obs)
    sigma = np.exp(point[cached.model.rvs_to_values[cached.amplitude].name])
    expected = multivariate_normal.logpdf(value, mean=mean_fn(point), cov=covariance(sigma))
    np.testing.assert_allclose(
        evaluate(value, *[point[variable.name] for variable in cached.model.value_vars]), expected, atol=1e-8,
    )


def test_native_joint_outputs_use_each_draw_and_ignore_sampler_cache():
    inputs = _inputs()
    _assert_native_joint_outputs(
        _build(), lambda amplitude: _dense_fixed_ou_covariance(inputs, amplitude, {"AAA": 3.0, "BBB": 7.0}),
    )


@pytest.mark.parametrize("prior_predictive", [8, True, False])
def test_cached_sampler_preserves_prior_predictive_policy(
    monkeypatch: pytest.MonkeyPatch,
    prior_predictive: bool | int,
) -> None:
    """The cached adapter leaves draw-count policy to the generic sampler."""
    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "make_cached_sigma_compound_step",
        lambda **_: object(),
    )
    sampler = RhimeSampler(
        draws=3,
        burn=1,
        chains=2,
        sample_prior_predictive=prior_predictive,
    )

    configured = co2_cached_sigma_runner._sampler_for_cached_graph(
        sampler,
        cached_model=_build(),
        sigma_target_accept=0.8,
        state_target_accept=0.9,
    )

    assert configured.sample_prior_predictive == prior_predictive
    assert type(configured.sample_prior_predictive) is type(prior_predictive)


@pytest.mark.parametrize("mean_shift", [False, True])
def test_named_runner_samples_real_graph_and_labels_cached_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mean_shift: bool,
) -> None:
    """The named runner samples the real graph and labels every cached output."""
    inputs = _boundary_inputs()
    inputs["G_bc"] = xr.DataArray(np.where(inputs.H_bc.values > 0, 0.2, 0.0), dims=("nmeasure", "bc_region"), attrs={"units": "1"})
    inputs["bc_centering_weights"] = xr.DataArray(np.arange(1, 9), dims="bc_region", attrs={"units": "1"})
    from openghg_inversions.rhime.co2.co2_preparation import _independent_boundary_weights
    weights = _independent_boundary_weights(inputs.bc_centering_weights)
    inputs = inputs.drop_vars("bc_centering_weights").assign(bc_centering_weights=weights)
    step_settings: dict[str, float] = {}
    original_make_step = co2_cached_sigma_runner.make_cached_sigma_compound_step

    def capture_step_settings(**kwargs: Any) -> CompoundStep:
        step_settings["sigma_target_accept"] = kwargs["sigma_target_accept"]
        step_settings["state_target_accept"] = kwargs["state_target_accept"]
        return original_make_step(**kwargs)

    class PreparedInputsStub:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

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
        draws=3,
        burn=1,
        tune=2,
        chains=2,
        progressbar=False,
        sample_kwargs={
            "random_seed": 147,
            "cores": 1,
            "compute_convergence_checks": False,
        },
        sample_prior_predictive=8,
        sample_posterior_predictive=("concentration", "y", "concentration"),
        posterior_predictive_kwargs={"random_seed": 148},
    )

    result, actual = sample_co2_inference(
        cast(Any, PreparedInputsStub()),
        {
            "use_bc": True,
            "bc_prior": None if mean_shift else {"pdf": "normal", "mu": 1.0, "sigma": 0.1},
            "bc_mean_shift_prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.5} if mean_shift else None,
            "bc_anomaly_scale": 0.2 if mean_shift else None,
            "bc_state_activity": None if mean_shift else StateActivity(
                active=np.asarray([True, False, False, False, False, False, False, False]),
                fixed_value=1.0,
            ),
            "offset_prior": {"pdf": "normal", "mu": 0.2, "sigma": 0.1},
            "likelihood_builder": add_fixed_ou_gaussian_likelihood,
            "likelihood_kwargs": {
                "tau_hours": {"AAA": 3.0, "BBB": 7.0},
                "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                "initial_site_amplitudes": 0.4,
            },
        },
        sampler, sigma_target_accept=0.82, state_target_accept=0.93,
    )

    assert step_settings == {
        "sigma_target_accept": 0.82,
        "state_target_accept": 0.93,
    }
    assert actual["sigma_target_accept"] == 0.82
    assert actual["state_target_accept"] == 0.93
    assert json.loads(result.attrs["rhime_actual_inference"]) == actual
    assert len(selected) == 1
    assert selected[0].count("H_bc") == 1
    assert result.posterior["flux_scaling"].shape == (2, 2, 2)
    if not mean_shift:
        assert result.posterior["bc"].shape == (2, 2, 8)
    assert result.posterior["offset"].shape == (2, 2, 4)
    assert result.posterior["offset_latent"].shape == (2, 2, 2)
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
    assert result.posterior_predictive["y"].shape == (2, 2, 4)
    np.testing.assert_array_equal(result.posterior_predictive.draw, np.arange(2))
    assert result.posterior_predictive.attrs["burn"] == 1
    assert result.log_likelihood["y"].shape == (2, 2)
    roles = json.loads(result.attrs["rhime_variable_roles"])
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert roles["boundary_concentration"] == "mu_bc"
    if not mean_shift:
        assert roles["boundary_scale"] == "bc"
    assert roles["boundary_sensitivity"] == "hbc"
    assert "baseline_concentration" not in roles
    assert "baseline_scale" not in roles
    if not mean_shift:
        assert json.loads(result.posterior["bc"].attrs["rhime_scientific_roles"]) == ["boundary_scale"]
    assert json.loads(result.posterior["mu_bc"].attrs["rhime_scientific_roles"]) == ["boundary_concentration"]
    assert metadata["recipe"] == "co2"
    assert "sampler" not in metadata
    assert "numerical_preparation" not in metadata
    assert "rhime_sampler_provenance" not in result.attrs
    assert set(result.posterior["ou_site_amplitude"].coords["ou_site"].values) == {
        "AAA",
        "BBB",
    }
    assert result.posterior["ou_site_amplitude"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert json.loads(result.posterior_predictive["y"].attrs["rhime_scientific_roles"]) == ["concentration"]
    assert "units" not in result.log_likelihood["y"].attrs
    assert json.loads(result.log_likelihood["y"].attrs["rhime_scientific_roles"]) == ["joint_log_likelihood"]
    assert result.posterior.attrs["rhime_recipe"] == "co2"

    assert result.prior.sizes["draw"] == 8
    assert result.prior_predictive["y"].shape == (1, 8, 4)
    assert result.prior_predictive["y"].attrs["rhime_predictive_scope"] == "joint_observation_vector"
    assert result.prior_predictive["y"].attrs["units"] == "ppm"
    np.testing.assert_array_equal(result.prior["site"], inputs["site"])
    path = tmp_path / "cached-prior-posterior.nc"
    save_trace(result, path)
    replay = load_trace(path)
    xr.testing.assert_identical(replay.prior.to_dataset(), result.prior.to_dataset())
    xr.testing.assert_identical(replay.prior_predictive.to_dataset(), result.prior_predictive.to_dataset())
    assert replay.posterior.sizes["draw"] == 2

    if mean_shift:
        assert selected[0].count("G_bc") == 1
        assert result.posterior["bc_mean_shift"].dims == ("chain", "draw")
        assert result.posterior["bc_mean_shift"].attrs["units"] == "ppm"
        assert result.constant_data["G_bc"].attrs["units"] == "1"
        np.testing.assert_allclose(
            result.posterior["mu_bc_mean_shift"],
            result.posterior["bc_mean_shift"] * inputs.G_bc.sum("bc_region"),
        )
        np.testing.assert_allclose(
            result.posterior["mu_bc"],
            result.posterior["mu_bc_reference"] + result.posterior["mu_bc_mean_shift"] + result.posterior["mu_bc_anomaly"],
        )
        assert roles["boundary_mean_shift"] == "bc_mean_shift"
        assert "bc" not in result.posterior
        assert "boundary_scale" not in roles
        np.testing.assert_allclose(
            (result.posterior["bc_anomaly"] * result.constant_data["bc_centering_weights"]).sum("bc_region"),
            0.0, atol=1e-12,
        )
        assert result.posterior["bc_correction"].attrs["units"] == "ppm"
    else:
        assert "G_bc" not in selected[0]
        assert "bc_mean_shift" not in result.posterior


def test_cached_runner_rejects_generic_target_accept() -> None:
    """The cached runner rejects one target acceptance rate for its two samplers."""

    class PreparedInputsStub:
        inv_inputs = _inputs()
        aggregation_error_mode = "dense"

    prepared = PreparedInputsStub()
    prepared.rhime_inputs = cast(Any, prepared)

    with pytest.raises(
        ValueError,
        match="sigma_target_accept.*state_target_accept",
    ):
        sample_co2_inference(
            cast(Any, prepared),
            {
                "likelihood_builder": add_fixed_ou_gaussian_likelihood,
                "likelihood_kwargs": {
                    "tau_hours": {"AAA": 3.0, "BBB": 7.0},
                    "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                },
            },
            RhimeSampler(sample_kwargs={"target_accept": 0.95}),
        )


@pytest.mark.parametrize("options", [
    {"sample_posterior_predictive": ("flux_scaling",)},
    {"sample_posterior_predictive": False, "posterior_predictive_kwargs": {"var_names": ["y"]}},
])
def test_cached_predictive_options_fail_before_constructing_step(monkeypatch, options):
    monkeypatch.setattr(
        co2_cached_sigma_runner, "make_cached_sigma_compound_step",
        lambda **_: pytest.fail("Invalid predictive options must fail before constructing a sampler."),
    )
    with pytest.raises(ValueError, match="predict|random_seed"):
        co2_cached_sigma_runner._sampler_for_cached_graph(
            RhimeSampler(**options),
            cached_model=cast(Any, object()),
            sigma_target_accept=0.8,
            state_target_accept=0.9,
        )


def test_cached_sampler_honors_disabled_likelihood_and_posterior_prediction():
    cached = _build()
    sampler = co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(
            draws=3, tune=3, chains=1, burn=1, sample_prior_predictive=4, sample_posterior_predictive=False,
            sample_kwargs={"cores": 1, "random_seed": 769, "compute_convergence_checks": False,
                           "idata_kwargs": {"log_likelihood": False}},
        ),
        cached_model=cached, sigma_target_accept=0.8, state_target_accept=0.9,
    )
    trace = sampler.sample(cached.model, variable_roles={"concentration": "y"})
    assert "log_likelihood" not in trace.children
    assert "posterior_predictive" not in trace.children
    assert trace.prior_predictive.y.shape == (1, 4, 4)
    assert trace.posterior.sizes["draw"] == 2


@pytest.mark.parametrize("initial, expected", [(None, [0.8, 0.8]), (0.4, [0.4, 0.4]), ({"AAA": 0.3, "BBB": 0.7}, [0.3, 0.7])])
def test_bounded_amplitude_prior_resolves_support_point_and_explicit_starts(initial, expected):
    cached = _build(site_amplitude_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.5}, initial_site_amplitudes=initial)
    np.testing.assert_allclose(cached.initial_cache.sigma, expected)
    point = _initial_point(cached)
    assert "ou_site_amplitude_interval__" in point
    assert np.isfinite(point["ou_site_amplitude_interval__"]).all()
    assert np.isfinite(cached.model.compile_logp()(point))
    assert all(value is None for value in cached.model.rvs_to_initial_values.values())


@pytest.mark.parametrize("initial", [0.1, 1.5, 1.6, 0.0, float("nan")])
def test_bounded_amplitude_prior_rejects_invalid_or_endpoint_start(initial):
    with pytest.raises(ValueError, match="Initial site amplitudes"):
        _build(site_amplitude_prior={"pdf": "uniform", "lower": 0.1, "upper": 1.5}, initial_site_amplitudes=initial)


@pytest.mark.parametrize("parameter", [0.7, np.array([0.6, 0.8]), xr.DataArray([0.8, 0.6], dims="ou_site", coords={"ou_site": ["BBB", "AAA"]})])
def test_cached_positive_prior_uses_scalar_positional_and_labelled_parameters(parameter):
    from openghg_inversions.models.priors import prior_records

    cached = _build(site_amplitude_prior={"pdf": "gamma", "alpha": 2.0, "beta": parameter}, initial_site_amplitudes=None)
    expected_beta = np.array([0.7, 0.7]) if np.ndim(parameter) == 0 else np.array([0.6, 0.8])
    np.testing.assert_allclose(cached.initial_cache.sigma, 2.0 / expected_beta)
    assert prior_records(cached.model)["ou_site_amplitude"]["pdf"] == "gamma"


def test_cached_lognormal_normalizes_optional_reparameterization_without_mutating_prior():
    from openghg_inversions.models.priors import prior_records

    prior = {"pdf": "lognormal", "mean": 0.7, "stdev": 0.3, "reparameterise": True}
    with pytest.warns(UserWarning, match="reparameterise=True.*False.*physical prior"):
        cached = _build(site_amplitude_prior=prior, initial_site_amplitudes=None)
    assert prior == {"pdf": "lognormal", "mean": 0.7, "stdev": 0.3, "reparameterise": True}
    assert cached.amplitude in cached.model.free_RVs
    assert prior_records(cached.model)["ou_site_amplitude"] == {**prior, "reparameterise": False}
    assert np.isfinite(cached.model.compile_logp()(_initial_point(cached)))


@pytest.mark.parametrize("storage", ["sparse", "lazy"])
def test_cached_uniform_prior_materializes_bounds_together_once(storage):
    import dask.array as da
    from dask import delayed
    import sparse

    executions = []
    values = np.array([[0.0, 0.2], [1.0, 1.2]])
    if storage == "sparse":
        lower, upper = (sparse.COO.from_numpy(value) for value in values)
    else:
        @delayed
        def bounds():
            executions.append(1)
            return values

        payload = da.from_delayed(bounds(), shape=(2, 2), dtype=float)
        lower, upper = payload[0], payload[1]
    prior = {"pdf": "uniform", "lower": lower, "upper": upper}
    cached = _build(site_amplitude_prior=prior, initial_site_amplitudes=None)
    np.testing.assert_allclose(cached.initial_cache.sigma, [0.5, 0.7])
    assert executions == ([1] if storage == "lazy" else [])
    assert prior["lower"] is lower and prior["upper"] is upper
    assert isinstance(lower, sparse.COO if storage == "sparse" else da.Array)


@pytest.mark.parametrize("frequency,expected_states", [(None, 2), ("monthly", 4)])
@pytest.mark.parametrize("direct", [False, True])
def test_cached_runner_reuses_raw_boundary_periods(monkeypatch, frequency, expected_states, direct):
    inputs = _boundary_inputs().drop_dims("bc_region")
    inputs["H_bc"] = xr.DataArray(
        np.ones((4, 2)), dims=("nmeasure", "bc_region"),
        coords={"nmeasure": inputs.nmeasure, "bc_region": ["north", "south"]},
        attrs={"units": "ppm", "boundary_sensitivity_preparation": "raw-spatial"},
    )

    class PreparedInputsStub:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

    monkeypatch.setattr(co2_cached_sigma_runner, "materialize_pymc_inputs", lambda *a, **k: inputs)
    if direct:
        result = _build(inputs=inputs, boundary_sensitivity=inputs.H_bc, bc_freq=frequency)
    else:
        result = co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
            prepared_inputs=cast(Any, PreparedInputsStub()), use_bc=True, bc_freq=frequency,
            tau_hours=3., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75},
        )
    assert result.model["bc"].eval().size == expected_states
    assert inputs.H_bc.shape == (4, 2)


def test_cached_runner_ignores_disabled_centered_boundary(monkeypatch):
    inputs = _inputs().assign(H_bc_anomaly=xr.DataArray(0.))

    class PreparedInputsStub:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

    monkeypatch.setattr(co2_cached_sigma_runner, "materialize_pymc_inputs", lambda *a, **k: inputs)
    result = co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
        prepared_inputs=cast(Any, PreparedInputsStub()), use_bc=False,
        tau_hours=3., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75},
    )
    assert "bc" not in result.model


@pytest.mark.parametrize("all_fixed", [False, True])
def test_centered_cache_full_operands_dense_density_and_gradients(all_fixed):
    """Full operands survive zero-design pruning, physical fixed states and offsets."""
    inputs = _inputs()
    reference = 400.0
    mass = np.asarray([[.2, .1, .05], [.3, .1, .05], [.4, .1, .05], [.2, .1, .05]])
    anomaly = np.asarray([[.01, 0., -20.], [.03, 0., -20.], [-.02, 0., -20.], [.04, 0., -20.]])
    operands = xr.Dataset({
        "H_bc_anomaly": (("nmeasure", "bc_region"), anomaly),
        "H_bc_mass": (("nmeasure", "bc_region"), mass),
        "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
    }, coords={"nmeasure": inputs.nmeasure, "bc_region": ["varying", "constant", "zero"]})
    operands.attrs["boundary_representation"] = "centered-v1"
    operands.H_bc_anomaly.attrs["units"] = "ppm"
    operands.H_bc_mass.attrs["units"] = "1"
    shift = reference * mass.sum(axis=1)
    inputs["mf"] = inputs.mf + shift
    inputs.mf.attrs["units"] = "ppm"
    from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands
    bound = bind_boundary_operands(operands, inputs.mf)
    active = xr.DataArray([not all_fixed, not all_fixed, False], dims="bc_region", coords={"bc_region": bound.bc_region})
    fixed = xr.DataArray([.9, 1.2, .7], dims="bc_region", coords=active.coords)
    cached = _build(inputs=inputs, boundary_operands=operands,
        bc_state_activity=StateActivity(active=active, fixed_value=fixed),
        bc_prior={"pdf": "lognormal", "mu": 0., "sigma": .1},
        offset_prior={"pdf": "normal", "mu": .2, "sigma": .1})
    model = cached.model
    expr = model.replace_rvs_by_values([cached.cached_state_logp, cached.modelled_mean, model["modelled_concentration"], model["bc"], model["co2_flux_contribution"], model["offset"]])
    evaluate = model.compile_fn(expr, inputs=model.value_vars, on_unused_input="ignore")
    cache_grad = model.compile_fn(pt.grad(expr[0], [model.rvs_to_values[rv] for rv in cached.states]),
        inputs=model.value_vars, on_unused_input="ignore")
    public_likelihood = model.compile_logp(vars=model.observed_RVs)
    public_gradient = model.compile_dlogp()
    prior_logp = model.compile_logp(vars=model.free_RVs)
    point = _initial_point(cached)
    max_errors = np.zeros(3)
    for i, amplitude in enumerate(([.3, .4], [.6, .2], [.25, .8])):
        point["ou_site_amplitude_log__"] = np.log(amplitude)
        point["flux_scaling_latent"] = np.asarray([-.2 + i*.2, .1-i*.1])
        if not all_fixed:
            point["bc_active_log__"] = np.log([.95+i*.03, 1.1-i*.02])
        point["offset_latent"] = np.asarray([.1+i*.01, -.1])
        cached.shared_cache.update(cached.target.refresh(np.asarray(amplitude)))
        cache_logp, centered, absolute, b, flux, offset = evaluate(point)
        expected_mean = flux + offset + anomaly @ b + reference * (mass @ (b-1))
        np.testing.assert_allclose(centered, expected_mean, atol=1e-12)
        np.testing.assert_allclose(absolute, centered + shift, atol=1e-12)
        dense = multivariate_normal.logpdf(inputs.mf.values-shift, mean=expected_mean,
            cov=_dense_fixed_ou_covariance(inputs, np.asarray(amplitude), {"AAA": 3., "BBB": 7.}))
        np.testing.assert_allclose(cache_logp, dense, atol=1e-9)
        np.testing.assert_allclose(public_likelihood(point), dense, atol=1e-9)
        max_errors[0] = max(max_errors[0], abs(cache_logp-dense), abs(public_likelihood(point)-dense))
        # Independent dense finite differences include physical prior transforms/Jacobians.
        def oracle(p, include_prior=True):
            _, _, _, b0, flux0, offset0 = evaluate(p)
            mean0 = flux0+offset0+anomaly@b0+reference*(mass@(b0-1))
            a0 = np.exp(p["ou_site_amplitude_log__"])
            likelihood0 = multivariate_normal.logpdf(inputs.mf.values-shift, mean=mean0,
                cov=_dense_fixed_ou_covariance(inputs,a0,{"AAA":3.,"BBB":7.}))
            return likelihood0 + prior_logp(p) if include_prior else likelihood0
        numerical = []
        for value in model.value_vars:
            name = value.name
            for ix in np.ndindex(point[name].shape):
                plus={k:v.copy() for k,v in point.items()}
                minus={k:v.copy() for k,v in point.items()}
                plus[name][ix]+=1e-6
                minus[name][ix]-=1e-6
                numerical.append((oracle(plus)-oracle(minus))/2e-6)
        np.testing.assert_allclose(public_gradient(point), numerical, rtol=3e-5, atol=3e-5)
        max_errors[1] = max(max_errors[1], np.max(np.abs(public_gradient(point)-numerical)))
        for rv, gradient in zip(cached.states, cache_grad(point), strict=True):
            name = model.rvs_to_values[rv].name
            numerical_state = np.empty_like(point[name])
            for ix in np.ndindex(point[name].shape):
                plus={k:v.copy() for k,v in point.items()}
                minus={k:v.copy() for k,v in point.items()}
                plus[name][ix]+=1e-6
                minus[name][ix]-=1e-6
                numerical_state[ix]=(oracle(plus,False)-oracle(minus,False))/2e-6
            np.testing.assert_allclose(gradient, numerical_state, rtol=3e-5, atol=3e-5)
            max_errors[2] = max(max_errors[2], np.max(np.abs(gradient-numerical_state)))
    print(f"CO2 all_fixed_bc={all_fixed}: density max error={max_errors[0]:.3g}, full-gradient finite-difference error={max_errors[1]:.3g}, cache-state finite-difference error={max_errors[2]:.3g}")
    if not all_fixed:
        compound = _compound(cached)
        np.testing.assert_allclose(compound.methods[0].modelled_mean_fn(point), evaluate(point)[1])
        next_point, _ = compound.step(point)
        assert all(np.isfinite(value).all() for value in next_point.values())


@pytest.mark.parametrize("reference", [400.0, 200_000.0])
@pytest.mark.parametrize("anomaly_scale", [None, 0.0, 0.05])
def test_centered_stock_and_cached_boundary_match_dense_likelihood(reference, anomaly_scale, monkeypatch):
    """Both BC priors compose with stock/cached OU on the same centered residual."""
    additive = anomaly_scale is not None
    inputs = _inputs()
    anomaly = np.asarray([[.2, -.1], [.3, .05], [-.1, .2], [.15, -.05]])
    mass = np.asarray([[.7, .2], [.1, .8], [.3, .6], [.5, .4]])
    shift = reference * mass.sum(axis=1)
    inputs["mf"] = inputs.mf + shift
    inputs.mf.attrs["units"] = "ppm"
    operands = xr.Dataset({
        "H_bc_anomaly": (("nmeasure", "bc_region"), anomaly, {"units": "ppm"}),
        "H_bc_mass": (("nmeasure", "bc_region"), mass, {"units": "1"}),
        "bc_reference": ((), reference, {"units": "ppm"}),
    }, coords={"nmeasure": inputs.nmeasure, "bc_region": ["north", "south"]}, attrs={"boundary_representation": "centered-v1"})
    from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands
    bound = bind_boundary_operands(operands, inputs.mf)
    weights = xr.DataArray([1., 3.], dims="bc_region", coords={"bc_region": bound.bc_region}, attrs={"units": "1"})
    boundary_parameters = (
        {"bc_mean_shift_prior": {"pdf": "normal", "mu": .1, "sigma": .2}, "bc_anomaly_scale": anomaly_scale}
        if additive else {"bc_prior": {"pdf": "normal", "mu": 1., "sigma": .01}}
    )
    cached = _build(inputs=inputs, boundary_operands=operands,
        boundary_correction_sensitivity=operands.H_bc_mass if additive else None,
        bc_centering_weights=weights if additive else None, **boundary_parameters)
    model = cached.model
    # Build the stock OU route through both public materialization/build boundaries.
    from openghg_inversions.rhime.co2.co2_preparation import _independent_boundary_weights
    stock_inputs = inputs.assign(operands.data_vars).assign(
        bc_centering_weights=_independent_boundary_weights(weights),
    )
    stock_inputs.attrs.update(boundary_representation="centered-v1")
    class PreparedStub:
        inv_inputs = stock_inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"
    prepared = cast(Any, PreparedStub())
    monkeypatch.setattr(co2_runner, "_materialize_pymc_inputs_and_priors",
        lambda _prepared, *, variable_names, prior_parameters: (stock_inputs, prior_parameters))
    options = dict(use_bc=True, **boundary_parameters,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={"tau_hours": {"AAA":3., "BBB":7.},
            "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .75}})
    eager_inputs, priors = co2_runner.materialize_co2_inputs(prepared_inputs=prepared, **options)
    stock = co2_runner.build_co2_from_materialized(eager_inputs,
        prepared_inputs=prepared, priors=priors, **options).model
    stock_likelihood = stock.compile_logp(vars=stock.observed_RVs)
    names = sorted(value.name for value in model.value_vars)
    assert names == sorted(value.name for value in stock.value_vars)
    def full_gradient(graph):
        values = {value.name: value for value in graph.value_vars}
        return graph.compile_fn(pt.grad(graph.logp(), [values[name] for name in names]),
            inputs=graph.value_vars, on_unused_input="ignore")
    stock_gradient = full_gradient(stock)
    cached_gradient = full_gradient(model)
    boundary_states = [model["bc_mean_shift"], model["bc_anomaly"]] if additive else [model["bc"]]
    expressions = model.replace_rvs_by_values([cached.cached_state_logp, cached.modelled_mean,
        model["mu_bc"], model["co2_flux_contribution"], *boundary_states])
    evaluate = model.compile_fn(expressions, inputs=model.value_vars, on_unused_input="ignore")
    point = _initial_point(cached)
    likelihood_logp = model.compile_logp(vars=model.observed_RVs)
    state_names = ("bc",)
    if additive:
        state_names = ("bc_mean_shift", "bc_anomaly_latent") if anomaly_scale else ("bc_mean_shift",)
        assert ("bc_anomaly_latent" in model) == (anomaly_scale > 0)
        assert ("bc_anomaly_latent" in stock) == (anomaly_scale > 0)
        assert ("bc_anomaly_latent" in {state.name for state in cached.states}) == (anomaly_scale > 0)
    value_variables = [model.rvs_to_values[model[name]] for name in state_names]
    gradient = model.compile_fn(pt.grad(expressions[0], value_variables),
        inputs=model.value_vars, on_unused_input="ignore")
    covariance = _dense_fixed_ou_covariance(inputs, np.asarray([.3, .4]), {"AAA":3., "BBB":7.})
    design = anomaly + reference * mass
    if additive:
        design = mass.sum(axis=1, keepdims=True)
        if anomaly_scale:
            design = np.column_stack((design, mass @ model["bc_anomaly_basis"].eval()))
    for mean_shift, contrast in ((.1, .0), (-.15, .08), (.25, -.04)):
        if additive:
            point["bc_mean_shift"] = np.asarray(mean_shift)
            if anomaly_scale:
                point["bc_anomaly_latent"] = np.asarray([contrast])
        else:
            point["bc"] = 1. + 1.e-6 * np.asarray([mean_shift, contrast])
        logp, mean, absolute, flux, *states = evaluate(point)
        if additive:
            delta, deviations = states
            if anomaly_scale == 0:
                np.testing.assert_array_equal(deviations, np.zeros(2))
            centered_boundary = anomaly.sum(axis=1) + mass @ (delta + deviations)
        else:
            scaling = states[0]
            centered_boundary = anomaly @ scaling + reference * (mass @ (scaling - 1.))
        np.testing.assert_allclose(mean, flux + centered_boundary, atol=1e-12)
        np.testing.assert_allclose(absolute, centered_boundary + shift, atol=1e-10)
        dense = multivariate_normal.logpdf(inputs.mf.values-shift, mean=mean, cov=covariance)
        np.testing.assert_allclose(logp, dense, atol=1e-9)
        np.testing.assert_allclose(likelihood_logp(point), dense, atol=1e-9)
        np.testing.assert_allclose(stock_likelihood(point), dense, atol=1e-9)
        for ordinary, cached_derivative in zip(stock_gradient(point), cached_gradient(point), strict=True):
            np.testing.assert_allclose(ordinary, cached_derivative, atol=1e-9)
        expected_gradient = design.T @ np.linalg.solve(covariance, inputs.mf.values-shift-mean)
        np.testing.assert_allclose(np.concatenate([np.atleast_1d(g) for g in gradient(point)]),
            expected_gradient, atol=1e-10)
    assert ("bc" not in model.named_vars) == additive


def test_cached_multiplicative_boundary_ignores_passive_correction_periods():
    inputs = _boundary_inputs()
    boundary = xr.DataArray(np.full((4, 2), 200.), dims=("nmeasure", "bc_region"),
        coords={"nmeasure": inputs.nmeasure, "bc_region": ["north", "south"]},
        attrs={"units": "ppm", "boundary_sensitivity_preparation": "raw-spatial"})
    passive_correction = xr.ones_like(inputs.H_bc).assign_attrs(units="1")
    cached = _build(inputs=inputs, boundary_sensitivity=boundary, bc_freq="2h",
        boundary_correction_sensitivity=passive_correction)
    assert "bc" in cached.model.named_vars
    assert "G_bc" not in cached.model.named_vars
