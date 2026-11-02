"""The same calibrated state in ordinary, cached, and linked recipes."""

import numpy as np
import pytest
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_model,
    build_co2_cached_sigma_model,
    build_co2_o2_model,
    build_co2_o2_cached_sigma_model,
)
from test_biosphere_prior import case
from test_co2_o2_fixed_ou import _prepared, _kwargs


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def state_and_mean(model, point):
    return model.compile_fn(
        model.replace_rvs_by_values([model.flux_scaling, model.modelled_concentration]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)


@pytest.mark.parametrize("family", ["lognormal", "gaussian"])
@pytest.mark.parametrize("full", [True, False])
def test_full_and_enhancement_ordinary_cached_prediction_density_and_no_default_change(family, full):
    baseline, inputs = case()
    calibration = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5, family=family)
    observation = xr.DataArray(
        [1.2, 0.4, 0.8, -0.5],
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "A", "B", "B"]),
            "time": ("nmeasure", np.datetime64("2021-01-01") + np.arange(4).astype("timedelta64[h]")),
        },
        attrs={"units": "ppm"},
    )
    h = xr.DataArray(
        [[-2, 1, -1, 2, 0.3], [-1, 0.8, -2, 1, 0.7], [0.1, 0.2, -0.3, 1, 0.4], [-0.5, 1, -0.7, 2, 0.8]],
        dims=("nmeasure", "state"),
        coords={"nmeasure": observation.nmeasure, **baseline.mean.coords},
    )
    kwargs = dict(
        flux_sensitivity=h,
        retained_prior=baseline,
        observations=observation,
        observation_error=xr.full_like(observation, 0.2),
        fixed_prior_contribution=xr.zeros_like(observation).rename("fixed_prior_contribution"),
        aggregation_error=resolve_aggregation_error(observation.to_dataset(name="mf"), "none"),
    )
    if full:
        kwargs["boundary_sensitivity"] = xr.DataArray(
            [[0.1], [0.2], [0.3], [0.4]],
            dims=("nmeasure", "bc_region"),
            coords={"nmeasure": observation.nmeasure, "bc_region": ["level"]},
        )
        kwargs["bc_prior"] = {"pdf": "normal", "mu": 0, "sigma": 0.2}
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    original = build_co2_model(
        **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=likelihood
    )
    identity = calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=1)
    same = build_co2_model(
        **kwargs,
        biosphere_prior=identity,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs=likelihood,
    )
    point = original.initial_point()
    np.testing.assert_allclose(original.compile_logp()(point), same.compile_logp()(point), rtol=1e-14)
    cached = build_co2_cached_sigma_model(**kwargs, biosphere_prior=calibration, **likelihood)
    ordinary = build_co2_model(
        **kwargs,
        biosphere_prior=calibration,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs=likelihood,
    )
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.5, -0.2, 0.4])
    point["ou_site_amplitude_log__"] = np.log([0.4, 0.6])
    cached_point = cached.model.initial_point()
    cached_point.update(point)
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, cached_point)
    np.testing.assert_allclose(x, xc, rtol=1e-14)
    np.testing.assert_allclose(mean, mc, rtol=1e-14)
    np.testing.assert_allclose(
        ordinary.compile_logp()(point), cached.model.compile_logp()(cached_point), rtol=1e-12
    )
    np.testing.assert_allclose(
        cached.target.log_likelihood_from_mean(mean, np.array([0.4, 0.6])),
        ordinary.compile_logp(vars=[ordinary.y])(point),
        rtol=1e-12,
    )
    assert len([rv for rv in ordinary.free_RVs if rv.name == "flux_scaling_latent"]) == 1
    # GPP and TER are still separate columns and states.
    assert x.shape == (5,) and h.shape == (4, 5)


@pytest.mark.parametrize("family", ["lognormal", "gaussian"])
def test_linked_species_share_one_calibration_and_density(family):
    prepared = _prepared()
    kwargs = _kwargs(prepared)
    baseline = prepared.retained_prior
    mean = baseline.mean
    weights = xr.DataArray(
        [[-10, 0, 0, 0, 0], [0, 8, 0, 0, 0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-10.0, 8.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    calibration = calibrate_biosphere_prior(
        baseline,
        biosphere=mean.copy(data=[True, True, False, False, False]),
        weights=weights,
        reference=reference,
        residual_sd_multiplier=0.5,
        family=family,
    )
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_co2_o2_model(**kwargs, biosphere_prior=calibration, **likelihood)
    cached = build_co2_o2_cached_sigma_model(**kwargs, biosphere_prior=calibration, **likelihood)
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.2, -0.1, 0.4])
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, point)
    np.testing.assert_allclose(x, xc)
    np.testing.assert_allclose(mean, mc)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(point), rtol=1e-12)
    assert len([rv for rv in cached.model.free_RVs if rv.name.startswith("flux_scaling")]) == 1
    assert cached.model.biosphere_prior is calibration


@pytest.mark.parametrize("family", ["lognormal", "gaussian"])
def test_linked_runner_prior_prediction_serializes_calibration_and_reconstructs(tmp_path, family):
    import json
    import pymc as pm
    from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2 import build_rhime_co2_o2, annotate_co2_o2_trace
    from openghg_inversions.serialization import save_trace, load_trace
    from test_rhime_co2_o2 import _independent_error

    prepared = _prepared()
    mean = prepared.retained_prior.mean
    weights = xr.DataArray(
        [[-10, 0, 0, 0, 0], [0, 8, 0, 0, 0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-10.0, 8.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    calibration = calibrate_biosphere_prior(
        prepared.retained_prior,
        biosphere=mean.copy(data=[True, True, False, False, False]),
        weights=weights,
        reference=reference,
        residual_sd_multiplier=0.5,
        family=family,
    )
    built = build_rhime_co2_o2(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared),
        biosphere_prior=calibration,
    )
    assert built.output_contract.metadata["prior"] == family
    with built.model:
        trace = pm.sample_prior_predictive(draws=4, random_seed=817)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    annotate_co2_o2_trace(trace, built=built)
    assert json.loads(trace.attrs["biosphere_prior"])["family"] == family
    save_trace(trace, tmp_path / "prior.nc")
    restored = load_trace(tmp_path / "prior.nc")
    np.testing.assert_array_equal(restored.biosphere_prior.covariance, calibration.data.covariance)
    x = restored.prior.flux_scaling
    expected = xr.dot(prepared.co2_sensitivity, x, dim="state")
    np.testing.assert_allclose(
        restored.prior.co2_o2_flux_contribution.sel(species="co2").to_numpy(),
        expected.transpose("chain", "draw", "co2_measure"),
    )
