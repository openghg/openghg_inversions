"""Dense two-tracer oracle and matched linked fixed-OU sampler contracts."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import xarray as xr
from openghg_inversions.models import StateActivity
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
from scipy.stats import halfnorm, norm
from pymc.blocking import DictToArrayBijection

from openghg_inversions.models.fixed_ou import FixedOuLowRank

from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_fixed_ou import linked_fixed_ou_alignment
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_runner import (
    run_rhime_co2_o2_cached_sigma_from_prepared_inputs,
)
from openghg_inversions.rhime.co2.co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs
from openghg_inversions.rhime.co2.co2_cached_sigma_runner import _sampler_for_cached_graph
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import save_trace, load_trace

from test_rhime_co2_o2 import _inputs, _independent_error
from test_cached_sigma_sampling import _nuts_stats
from test_co2_cached_sigma import _assert_native_joint_outputs, _compound


def _prepared():
    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    for channel, sites, times in [("co2", ["A", "A"], [4, 0]), ("o2", ["A", "B", "A"], [4, 2, 0])]:
        obs = inputs[f"{channel}_observations"]
        dim = obs.dims[0]
        inputs[f"{channel}_observations"] = obs.assign_coords(
            site=(dim, sites),
            time=(dim, np.datetime64("2021-01-01T00") + np.array(times).astype("timedelta64[h]")),
        )
    return prepare_co2_o2_inputs(**inputs)


def _kwargs(prepared, *, interleaved=False, native_multiindex=False):
    observations = prepared.observations
    fixed = prepared.fixed_prior_contribution
    error = _independent_error(prepared)
    aggregation = prepared.aggregation_error
    if interleaved:
        order = [2, 0, 3, 1, 4]
        observations, fixed, error = [array.isel(observation=order) for array in (observations, fixed, error)]
        covariance = aggregation.covariance
        covariance = covariance.isel({covariance.dims[0]: order, covariance.dims[1]: order})
        aggregation = replace(
            aggregation, covariance=covariance, marginal_variance=aggregation.marginal_variance[order]
        )
    result = dict(
        observations=observations,
        fixed_prior_contribution=fixed,
        co2_sensitivity=prepared.co2_sensitivity,
        o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=aggregation,
        retained_prior=prepared.retained_prior,
        independent_error_sd=error,
    )
    if native_multiindex:
        for channel in ("co2", "o2"):
            sensitivity = result[f"{channel}_sensitivity"]
            native_dim = sensitivity.dims[0]
            rows = prepared.observations.sel(species=channel)
            index = pd.MultiIndex.from_arrays([rows.site.values, rows.time.values], names=["site", "time"])
            result[f"{channel}_sensitivity"] = sensitivity.drop_vars(native_dim).assign_coords(
                xr.Coordinates.from_pandas_multiindex(index, native_dim)
            )
        index = pd.MultiIndex.from_arrays(
            [observations.species.values, observations.site.values, observations.time.values],
            names=["species", "site", "time"],
        )
        coords = xr.Coordinates.from_pandas_multiindex(index, "observation")
        for name in ("observations", "fixed_prior_contribution", "independent_error_sd"):
            array = result[name]
            result[name] = xr.DataArray(
                array.values, dims="observation", coords=coords, attrs=array.attrs, name=array.name
            ).assign_coords(observation_units=("observation", observations.observation_units.values))
        covariance_dim = str(aggregation.covariance.dims[1])
        covariance = xr.DataArray(
            aggregation.covariance.values,
            dims=("observation", covariance_dim),
            coords={
                **coords,
                **xr.Coordinates.from_pandas_multiindex(
                    index.set_names(["species_cov", "site_cov", "time_cov"]), covariance_dim
                ),
            },
        )
        result["aggregation_error"] = replace(aggregation, covariance=covariance)
    return result


def _dense(kwargs, amplitudes, tau):
    obs = kwargs["observations"]
    alignment = linked_fixed_ou_alignment(obs)
    result = kwargs["aggregation_error"].covariance.values.copy()
    result += np.diag(kwargs["independent_error_sd"].values ** 2)
    hours = obs.time.values.astype("datetime64[m]").astype(float) / 60
    for index, label in enumerate(alignment.site_labels.values):
        rows = np.flatnonzero(alignment.site_index.values == index)
        result[np.ix_(rows, rows)] += amplitudes[index] ** 2 * np.exp(
            -np.abs(hours[rows, None] - hours[None, rows]) / tau[label]
        )
    return result


def test_linked_native_joint_predictions_and_density_ignore_poisoned_cache():
    kwargs = _kwargs(_prepared(), interleaved=True, native_multiindex=True)
    tau = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}
    cached = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=tau, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    assert not cached.model.potentials
    assert [rv.name for rv in cached.model.observed_RVs] == ["y"]
    covariance = _dense(kwargs, np.zeros(3), tau)
    species = kwargs["observations"].species.values
    assert np.any(covariance[np.ix_(species == "co2", species == "o2")] != 0.0)
    _assert_native_joint_outputs(cached, lambda amplitude: _dense(kwargs, amplitude, tau))


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("native_multiindex", [False, True])
def test_linked_stock_cached_covariance_logp_and_gradients_match_dense(interleaved, native_multiindex):
    kwargs = _kwargs(_prepared(), interleaved=interleaved, native_multiindex=native_multiindex)
    alignment = linked_fixed_ou_alignment(kwargs["observations"])
    tau = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}
    sigma = np.array([0.4, 0.6, 0.8])
    cached = build_co2_o2_cached_sigma_model(
        **kwargs,
        tau_hours=tau,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8},
        initial_site_amplitudes=dict(zip(alignment.site_labels.values, sigma)),
    )
    stock = build_co2_o2_model(
        **kwargs, tau_hours=tau, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8}
    )
    covariance = _dense(kwargs, sigma, tau)
    species = kwargs["observations"].species.values
    aggregation = kwargs["aggregation_error"].covariance.values
    assert np.any(aggregation[np.ix_(species == "co2", species == "o2")] != 0.0)
    state = np.array([0.9, 1.1, 0.8, 1.2, 1.05])
    mean = cached.target.fixed_contribution + cached.target.design @ state
    residual = kwargs["observations"].values - mean
    expected = multivariate_normal.logpdf(residual, cov=covariance)
    np.testing.assert_allclose(
        cached.target.prepared.solve(np.eye(5), sigma).solution,
        np.linalg.inv(covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(cached.target.refresh(sigma).log_likelihood(state), expected, rtol=1e-11)
    amplitude = pt.dvector("amplitude")
    residual_var = pt.dvector("residual")
    stock_target = cached.target.prepared
    logp = stock_target.logp(residual_var, pt.zeros(5), amplitude)
    actual, residual_gradient, amplitude_gradient = pytensor.function(
        [residual_var, amplitude], [logp, pt.grad(logp, residual_var), pt.grad(logp, amplitude)]
    )(residual, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-11)
    np.testing.assert_allclose(residual_gradient, -np.linalg.solve(covariance, residual), rtol=1e-10)
    for index in range(len(sigma)):
        plus, minus = sigma.copy(), sigma.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        expected_gradient = (
            multivariate_normal.logpdf(residual, cov=_dense(kwargs, plus, tau))
            - multivariate_normal.logpdf(residual, cov=_dense(kwargs, minus, tau))
        ) / 2e-5
        np.testing.assert_allclose(amplitude_gradient[index], expected_gradient, rtol=1e-6)
    fixed = build_co2_o2_model(
        **kwargs, tau_hours=tau, fixed_site_amplitudes=dict(zip(alignment.site_labels.values, sigma))
    )
    fixed_point = fixed.initial_point()
    fixed_mean = fixed.compile_fn(
        fixed.replace_rvs_by_values([fixed["modelled_concentration"]]),
        inputs=fixed.value_vars,
        on_unused_input="ignore",
    )(fixed_point)[0]
    np.testing.assert_allclose(
        fixed.compile_logp(vars=[fixed["y"]])(fixed_point),
        cached.target.log_likelihood_from_mean(fixed_mean, sigma),
        rtol=2e-6,
    )
    # Compare the actual observed PyMC likelihood to the cached target at the
    # same flux/amplitude point (including transforms and correlated state).
    point = stock.initial_point()
    point["ou_site_amplitude_log__"] = np.log(sigma)
    stock_mean = stock.compile_fn(
        stock.replace_rvs_by_values([stock["modelled_concentration"]]),
        inputs=stock.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )(point)[0]
    np.testing.assert_allclose(
        stock.compile_logp(vars=[stock["y"]])(point),
        cached.target.log_likelihood_from_mean(stock_mean, sigma),
        rtol=2e-6,
    )


@pytest.fixture
def centered_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.mark.parametrize("centered", [False, True])
def test_linked_cache_accept_reject_order_and_chain_locality(monkeypatch, centered, centered_precision):
    kwargs = _kwargs(_prepared())
    if centered:
        sensitivity = kwargs["co2_sensitivity"]
        dim = str(sensitivity.dims[0])
        operands = xr.Dataset({
            "H_bc_anomaly": xr.DataArray(np.full((2,1),.01),dims=("nmeasure","bc_region"),attrs={"units":"ppm"}),
            "H_bc_mass": xr.DataArray(np.full((2,1),.2),dims=("nmeasure","bc_region"),attrs={"units":"1"}),
            "bc_reference": xr.DataArray(400.,attrs={"units":"ppm"}),
        },coords={"nmeasure":sensitivity[dim].values,"bc_region":["north"]},attrs={"boundary_representation":"centered-v1"})
        kwargs["boundary_operands"] = {"co2":operands}
    first = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    second = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    sampler = _sampler_for_cached_graph(
        RhimeSampler(nuts_sampler="pymc"),
        cached_model=first,
        sigma_target_accept=0.8,
        state_target_accept=0.9,
    )
    amplitude_step, state_step = sampler.sample_kwargs["step"].methods
    assert amplitude_step.target is first.target
    assert first.shared_cache.linear is not second.shared_cache.linear
    point = first.model.initial_point()
    proposed = point["ou_site_amplitude_log__"] + [0.05, -0.03, 0.1]
    monkeypatch.setattr(
        amplitude_step.nuts_step,
        "step",
        lambda _: ({"ou_site_amplitude_log__": proposed.copy()}, _nuts_stats()),
    )
    updated, stats = amplitude_step.step(point)
    assert stats[0]["cache_refreshes"] == 1
    np.testing.assert_allclose(
        first.shared_cache.linear.get_value(), first.target.refresh(np.exp(proposed)).linear, rtol=2e-6
    )
    np.testing.assert_allclose(second.shared_cache.linear.get_value(), second.initial_cache.linear)
    monkeypatch.setattr(
        amplitude_step.nuts_step,
        "step",
        lambda point: ({"ou_site_amplitude_log__": point["ou_site_amplitude_log__"].copy()}, _nuts_stats()),
    )
    _, stats = amplitude_step.step(updated)
    assert stats[0]["cache_refreshes"] == 0
    monkeypatch.setattr(first.target, "refresh", lambda _: pytest.fail("state step refreshed covariance"))
    monkeypatch.setattr(FixedOuLowRank, "evaluate", lambda *_: pytest.fail("state step evaluated covariance"))
    monkeypatch.setattr(np.linalg, "cholesky", lambda *_: pytest.fail("state step factored covariance"))
    state_step.step(updated)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_ou_rejects_unsupported_backend_before_sampling(cached):
    prepared = _prepared()
    runner = (
        run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        if cached
        else run_rhime_co2_o2_from_prepared_inputs
    )
    options = {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.8}} if cached else {"fixed_site_amplitudes": 0.8}
    with pytest.raises(ValueError, match="nuts_sampler='pymc'"):
        runner(
            prepared_inputs=prepared,
            independent_error_sd=_independent_error(prepared),
            tau_hours=6.0,
            sampler=RhimeSampler(nuts_sampler="numpyro"),
            **options,
        )


def test_linked_ou_requires_same_units():
    observations = _prepared().observations.assign_coords(
        observation_units=("observation", ["ppm", "ppm", "per meg", "per meg", "per meg"])
    )
    with pytest.raises(ValueError, match="same units"):
        linked_fixed_ou_alignment(observations)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_sampling_serializes_group_labels_and_joint_outputs(tmp_path, cached):
    """Replay saved errors and preserve joint outputs and provenance through trace storage."""
    prepared = _prepared()
    prepared = replace(prepared, independent_error_sd=_independent_error(prepared))
    prepared_path = tmp_path / "prepared.nc"
    prepared.save(prepared_path)
    prepared = Co2O2PreparedInputs.load(prepared_path)
    runner = (
        run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        if cached
        else run_rhime_co2_o2_from_prepared_inputs
    )
    options = {"site_amplitude_prior": {"pdf": "uniform", "lower": 0.1, "upper": 1.5}}
    trace = runner(
        prepared_inputs=prepared,
        tau_hours={"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0},
        **options,
        sampler=RhimeSampler(
            nuts_sampler="pymc",
            draws=5,
            tune=4,
            chains=2,
            burn=1,
            sample_kwargs={
                "cores": 1,
                "random_seed": 171,
                "progressbar": False,
                "compute_convergence_checks": False,
            },
            sample_posterior_predictive=True,
        ),
    )
    assert isinstance(trace, xr.DataTree)
    path = tmp_path / "linked.nc"
    save_trace(trace, path)
    with xr.open_datatree(path, decode_cf=False) as encoded:
        assert encoded.constant_data.ou_tau_hours.attrs["dtype"] == "timedelta64[ns]"
    restored = load_trace(path)
    assert isinstance(restored, xr.DataTree)
    assert json.loads(restored.attrs["rhime_model_metadata"])["independent_error"] == (
        "fixed labelled standard deviation resolved from prepared or explicit inputs"
    )
    assert restored.posterior.ou_site.values.tolist() == ["co2:A", "o2:A", "o2:B"]
    assert restored.posterior.ou_species.values.tolist() == ["co2", "o2", "o2"]
    assert restored.posterior.ou_station.values.tolist() == ["A", "A", "B"]
    assert restored.posterior.ou_site_amplitude.attrs["units"] == "ppm"
    assert restored.constant_data.ou_tau_hours.encoding["units"] == "hours"
    np.testing.assert_array_equal(
        restored.constant_data.ou_tau_hours.values, np.array([6, 8, 2], dtype="timedelta64[h]")
    )
    assert restored.log_likelihood.y.dims == ("chain", "draw")
    assert restored.log_likelihood.y.shape == (2, 4)
    assert restored.log_likelihood.y.attrs["rhime_likelihood_scope"] == "joint_observation_vector"
    assert restored.posterior_predictive.y.sizes["observation"] == 5
    assert restored.posterior_predictive.y.shape == (2, 4, 5)
    assert restored.prior_predictive.y.shape == (1, 4, 5)
    np.testing.assert_array_equal(restored.posterior_predictive.draw, np.arange(4))
    assert restored.posterior_predictive.attrs["burn"] == 1
    assert restored.prior_predictive.y.attrs["units"] == "ppm"
    assert restored.posterior_predictive.y.attrs["units"] == "ppm"
    assert restored.posterior_predictive.species.values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert "flux_scaling" in restored.prior


def test_padded_boundary_design_follows_reordered_native_labels():
    from openghg_inversions.rhime.co2.co2_o2_model import _pad_channel_design
    from test_rhime_co2_o2_baselines import _prepared as prepared_with_baselines

    prepared = prepared_with_baselines(("co2", "o2"))
    order = [1, 4, 0, 2, 3]
    observations = prepared.observations.isel(observation=order)
    for channel in ("co2", "o2"):
        design = prepared.boundary_sensitivity[channel]
        gathered = _pad_channel_design(design, observations, channel)
        expected = np.zeros((5, 2))
        selected = slice(0, 2) if channel == "co2" else slice(2, 5)
        expected[selected] = design.values
        np.testing.assert_array_equal(gathered, expected[order])
        assert gathered.indexes["observation"].equals(observations.indexes["observation"])


@pytest.mark.parametrize("native_multiindex", [False, True])
def test_cached_affine_design_includes_fixed_pruned_boundary_and_channel_offsets(native_multiindex):
    prepared = _prepared()
    kwargs = _kwargs(prepared, interleaved=True, native_multiindex=native_multiindex)
    boundaries = {}
    activities = {}
    for channel in ("co2", "o2"):
        sensitivity = kwargs[f"{channel}_sensitivity"]
        dim = sensitivity.dims[0]
        boundaries[channel] = xr.DataArray(
            np.column_stack(
                [
                    np.arange(1, sensitivity.shape[0] + 1),
                    np.ones(sensitivity.shape[0]),
                    np.zeros(sensitivity.shape[0]),
                ]
            ),
            dims=(dim, "boundary"),
            coords={dim: sensitivity[dim], "boundary": ["north", "south", "zero"]},
        )
        activities[channel] = StateActivity(
            active=xr.DataArray(
                [True, False, False], dims="boundary", coords={"boundary": ["north", "south", "zero"]}
            ),
            fixed_value=2.0,
        )
    kwargs.update(
        boundary_sensitivity=boundaries,
        bc_state_activity=activities,
        bc_prior={channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.1} for channel in boundaries},
        offset_prior={channel: {"pdf": "normal", "mu": 0.1, "sigma": 0.2} for channel in boundaries},
        offset_args={channel: {"per_site": native_multiindex} for channel in boundaries},
    )
    cached = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    stock = build_co2_o2_model(
        **kwargs, tau_hours=6.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8}
    )
    point = stock.initial_point()
    point["co2_bc_active"] = np.array([1.3])
    point["o2_bc_active"] = np.array([0.7])
    variables = [
        stock["modelled_concentration"],
        stock["flux_scaling"],
        stock["co2_bc"],
        stock["co2_offset_latent"],
        stock["o2_bc"],
        stock["o2_offset_latent"],
    ]
    values = stock.compile_fn(
        stock.replace_rvs_by_values(variables), inputs=stock.value_vars, on_unused_input="ignore"
    )(point)
    mean, flux, co2_bc, co2_offset, o2_bc, o2_offset = values
    coefficients = np.concatenate(
        [flux, co2_bc[:1], np.atleast_1d(co2_offset), o2_bc[:1], np.atleast_1d(o2_offset)]
    )
    np.testing.assert_allclose(
        cached.target.fixed_contribution + cached.target.design @ coefficients, mean, rtol=2e-6, atol=2e-6
    )
    sigma = np.exp(point["ou_site_amplitude_log__"])
    np.testing.assert_allclose(
        cached.target.refresh(sigma).log_likelihood(coefficients),
        stock.compile_logp(vars=[stock["y"]])(point),
        rtol=2e-6,
    )
    state_step = _compound(cached).methods[1]
    public_logp = cached.model.compile_logp()
    stock_logp = stock.compile_logp()
    public_gradient = cached.model.compile_dlogp(vars=list(cached.states))
    tau = {str(label): 6.0 for label in cached.target.prepared.site_labels}
    mean_fn = cached.model.compile_fn(
        cached.model.replace_rvs_by_values([cached.modelled_mean])[0],
        inputs=cached.model.value_vars, on_unused_input="ignore",
    )
    for shift in (-0.2, 0.3):
        varied = {name: value.copy() for name, value in point.items()}
        for name in state_step.var_names:
            varied[name] += shift
        amplitudes = np.linspace(0.3 + shift / 10, 0.9, cached.target.n_group)
        varied["ou_site_amplitude_log__"] = np.log(amplitudes)
        varied = {
            str(value.name): np.asarray(varied[str(value.name)], dtype=value.dtype)
            for value in cached.model.value_vars
        }
        amplitudes = np.exp(varied["ou_site_amplitude_log__"].astype(np.float64))
        cached.shared_cache.update(cached.target.refresh(amplitudes))
        state_step._logp_dlogp_func.set_extra_values(varied)
        actual, gradient = state_step._logp_dlogp_func(DictToArrayBijection.map(
            {name: varied[name] for name in state_step.var_names}
        ))
        prior = halfnorm.logpdf(amplitudes, scale=0.8).sum() + np.log(amplitudes).sum()
        for name in state_step.var_names:
            location, scale = (0.0, 1.0) if name == "flux_scaling_latent" else (
                (1.0, 0.1) if name.endswith("bc_active") else (0.1, 0.2)
            )
            prior += norm.logpdf(varied[name], loc=location, scale=scale).sum()
        expected = prior + multivariate_normal.logpdf(
            kwargs["observations"].values, mean=mean_fn(varied), cov=_dense(kwargs, amplitudes, tau),
        )
        tolerance = 2e-6 if pytensor.config.floatX == "float32" else 1e-8
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual, public_logp(varied), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual, stock_logp(varied), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(gradient, public_gradient(varied), rtol=tolerance, atol=tolerance)
    assert [rv.name for rv in cached.states] == [
        "flux_scaling_latent",
        "co2_bc_active",
        "co2_offset_latent",
        "o2_bc_active",
        "o2_offset_latent",
    ]


def test_linked_ou_rejects_missing_station_labels():
    obs = _prepared().observations.assign_coords(site=("observation", ["A", None, "A", "B", "A"]))
    with pytest.raises(ValueError, match="complete site labels"):
        linked_fixed_ou_alignment(obs)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("o2_reference", [209300., 1e12])
def test_centered_linked_cache_matches_independent_dense_gaussian(mixed, o2_reference):
    """Direct centered arithmetic survives large O2 backgrounds and native labels."""
    import pymc as pm
    from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands
    from openghg_inversions.array_ops import select_gathered_data_array

    with pytensor.config.change_flags(floatX="float64"):
        kwargs = _kwargs(_prepared(), interleaved=True)
        observations = kwargs["observations"]
        operands = {}
        raw_padded = {}
        policies = {}
        shifts = np.zeros(observations.size)
        for channel, labels, reference in (("co2", ["north", "constant"], 400.), ("o2", ["top", "fixed", "zero"], o2_reference)):
            sensitivity = kwargs[f"{channel}_sensitivity"]
            native_dim = str(sensitivity.dims[0])
            native = select_gathered_data_array(observations, key=channel, key_dim="species",
                ragged_dim="channel_observation", stack_dim="observation").rename({"observation": native_dim})
            n = native.size
            mass = np.full((n, len(labels)), .2)
            anomaly = np.zeros_like(mass)
            anomaly[:,0] = np.linspace(.01, .03, n)
            if channel == "o2":
                mass[:,-1] = 1./reference
                anomaly[:,-1] = -1.
            raw = xr.Dataset({
                "H_bc_anomaly": xr.DataArray(anomaly, dims=("nmeasure","bc_region"), attrs={"units":"ppm"}),
                "H_bc_mass": xr.DataArray(mass, dims=("nmeasure","bc_region"), attrs={"units":"1"}),
                "bc_reference": xr.DataArray(reference, attrs={"units":"ppm"}),
            },coords={"nmeasure":sensitivity[native_dim].values,"bc_region":labels},attrs={"boundary_representation":"centered-v1"})
            aligned = bind_boundary_operands(raw, native)
            active = [True, True] if channel == "co2" else [True, False, False]
            fixed = [1.,1.] if channel == "co2" else [1.,1.+2e-13,.7]
            policies[channel] = StateActivity(
                active=xr.DataArray(active,dims="bc_region",coords={"bc_region":aligned.bc_region}),
                fixed_value=xr.DataArray(fixed,dims="bc_region",coords={"bc_region":aligned.bc_region}),
            )
            rows = np.flatnonzero(observations.species.values == channel)
            # Native labels can differ from joint row order: map each label explicitly.
            joint_labels = observations.channel_observation.values[rows]
            native_order = [int(np.flatnonzero(sensitivity[native_dim].values == label)[0]) for label in joint_labels]
            padded_a = np.zeros((observations.size,len(labels)))
            padded_m = np.zeros_like(padded_a)
            padded_a[rows] = anomaly[native_order]
            padded_m[rows] = mass[native_order]
            raw_padded[channel] = (padded_a,padded_m,reference)
            if mixed and channel == "co2":
                kwargs["boundary_sensitivity"] = {"co2": (raw.H_bc_anomaly+reference*raw.H_bc_mass).rename({"nmeasure":native_dim}).transpose(native_dim,"bc_region")}
                # Ordinary boundary policies use their unexpanded labels.
                policies[channel] = StateActivity(active=True)
            else:
                operands[channel] = raw
                shifts += reference*padded_m.sum(axis=1)
        kwargs["observations"] = observations + xr.DataArray(shifts,dims="observation",coords={"observation":observations.observation})
        tau = {"co2:A":6.,"o2:A":8.,"o2:B":2.}
        cached = build_co2_o2_cached_sigma_model(**kwargs,boundary_operands=operands,
            bc_state_activity=policies,bc_prior={channel:{"pdf":"normal","mu":1.,"sigma":.1} for channel in ("co2","o2")},
            offset_prior={"co2":{"pdf":"normal","mu":0.,"sigma":.1}},
            tau_hours=tau,site_amplitude_prior={"pdf":"halfnormal","sigma":.8})
        model = cached.model
        oracle_mean = model["co2_o2_affine_contribution"] + model["co2_offset"]
        oracle_absolute = oracle_mean
        for channel,(a,m,c) in raw_padded.items():
            b = model[f"{channel}_bc"]
            oracle_mean = oracle_mean + pt.dot(a,b) + c*pt.dot(m,b-1 if channel in operands else b)
            oracle_absolute = oracle_absolute + pt.dot(a,b) + c*pt.dot(m,b)
        amplitude = model["ou_site_amplitude"]
        covariance = pt.as_tensor_variable(_dense(kwargs,np.zeros(3),tau))
        alignment = linked_fixed_ou_alignment(observations)
        hours = observations.time.values.astype("datetime64[m]").astype(float)/60
        for i,label in enumerate(alignment.site_labels.values):
            selected = alignment.site_index.values == i
            kernel = np.exp(-np.abs(hours[:,None]-hours[None,:])/tau[label])*selected[:,None]*selected[None,:]
            covariance = covariance + amplitude[i]**2*kernel
        oracle = pm.logp(pm.MvNormal.dist(mu=oracle_mean,cov=covariance),kwargs["observations"].values-shifts)
        public = model.logp(vars=model.observed_RVs)
        expressions = model.replace_rvs_by_values([oracle,public,cached.cached_state_logp,oracle_mean,cached.modelled_mean,model["modelled_concentration"],oracle_absolute])
        oracle_values,public_values,cache_values = expressions[:3]
        state_values = [model.rvs_to_values[rv] for rv in cached.states]
        evaluate = model.compile_fn([*expressions,pt.concatenate([g.ravel() for g in pt.grad(oracle_values,model.value_vars)]),pt.concatenate([g.ravel() for g in pt.grad(public_values,model.value_vars)]),
            pt.concatenate([g.ravel() for g in pt.grad(oracle_values,state_values)]),pt.concatenate([g.ravel() for g in pt.grad(cache_values,state_values)])],inputs=model.value_vars,on_unused_input="ignore")
        point = model.initial_point()
        maxima = np.zeros(3)
        relative_gradients = np.zeros(2)
        cancellation = 0.
        for i,sigma in enumerate(([.4,.6,.8],[.8,.3,.5],[.25,.9,.4])):
            point["ou_site_amplitude_log__"] = np.log(sigma)
            point["o2_bc_active"] = np.asarray([1.+(i+1)*1e-13])
            point["co2_bc"] = np.asarray([.9+i*.05,1.1-i*.03])
            point["co2_offset_latent"] = np.asarray([.03])
            cached.shared_cache.update(cached.target.refresh(np.asarray(sigma)))
            dense,observed,cache,oracle_mu,centered,absolute,absolute_oracle,dense_grad,public_grad,dense_state,cache_state = evaluate(point)
            np.testing.assert_allclose([observed,cache],dense,atol=1e-9,rtol=1e-10)
            np.testing.assert_allclose(centered,oracle_mu,atol=1e-12)
            np.testing.assert_allclose(absolute,absolute_oracle,atol=1e-10,rtol=1e-15)
            for actual,expected in zip(public_grad,dense_grad,strict=True):
                np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-10)
            for actual,expected in zip(cache_state,dense_state,strict=True):
                np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-10)
            maxima[0]=max(maxima[0],abs(cache-dense),abs(observed-dense))
            maxima[1]=max(maxima[1],max(np.max(np.abs(a-b)) for a,b in zip(public_grad,dense_grad,strict=True)))
            maxima[2]=max(maxima[2],max(np.max(np.abs(a-b)) for a,b in zip(cache_state,dense_state,strict=True)))
            relative_gradients[0]=max(relative_gradients[0],np.max(np.abs(public_grad-dense_grad)/np.maximum(1.,np.abs(dense_grad))))
            relative_gradients[1]=max(relative_gradients[1],np.max(np.abs(cache_state-dense_state)/np.maximum(1.,np.abs(dense_state))))
            cancellation=max(cancellation,np.max(np.abs((absolute-shifts)-centered)))
        print(f"centered linked O2={o2_reference:g} mixed={mixed}: density max error={maxima[0]:.3g}, full-gradient absolute error={maxima[1]:.3g}, cached-state absolute error={maxima[2]:.3g}; scaled full/cache gradient error={relative_gradients}; absolute subtraction loses {cancellation:.3g} ppm")
        assert cancellation > (1e-6 if o2_reference == 1e12 else 1e-12)


def test_centered_linked_all_fixed_boundary_cache(centered_precision):
    """Independent fixed channel baselines add no cache columns or sampled states."""
    kwargs = _kwargs(_prepared(), interleaved=True)
    operands = {}
    shift = np.zeros(5)
    centered_fixed = np.zeros(5)
    for channel, reference, fixed_value in (("co2",400.,1.2),("o2",209300.,.8)):
        sensitivity = kwargs[f"{channel}_sensitivity"]
        dim = str(sensitivity.dims[0])
        n = sensitivity.sizes[dim]
        operands[channel] = xr.Dataset({
            "H_bc_anomaly": xr.DataArray(np.full((n,1),.02),dims=("nmeasure","bc_region"),attrs={"units":"ppm"}),
            "H_bc_mass": xr.DataArray(np.full((n,1),.1),dims=("nmeasure","bc_region"),attrs={"units":"1"}),
            "bc_reference": xr.DataArray(reference,attrs={"units":"ppm"}),
        },coords={"nmeasure":sensitivity[dim].values,"bc_region":[f"{channel}_fixed"]},attrs={"boundary_representation":"centered-v1"})
        rows = kwargs["observations"].species.values == channel
        shift[rows] = .1*reference
        centered_fixed[rows] = .02*fixed_value+.1*reference*(fixed_value-1.)
    cached = build_co2_o2_cached_sigma_model(**kwargs,boundary_operands=operands,
        bc_state_activity={"co2":StateActivity(active=False,fixed_value=1.2),"o2":StateActivity(active=False,fixed_value=.8)},
        tau_hours=6.,site_amplitude_prior={"pdf":"halfnormal","sigma":.8})
    np.testing.assert_allclose(cached.target.fixed_contribution,kwargs["fixed_prior_contribution"].values+centered_fixed)
    assert len(cached.states) == 1
    assert cached.target.design.shape[1] == kwargs["co2_sensitivity"].sizes[kwargs["co2_sensitivity"].dims[1]]
    np.testing.assert_allclose(cached.target.observations,kwargs["observations"].values-shift)
