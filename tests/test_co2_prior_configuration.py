"""Configured priors retain the explicit Python target and frozen replay values."""

from types import SimpleNamespace
from dataclasses import replace
import tomllib

from dask.callbacks import Callback
import numpy as np
import pymc as pm
import pytest
import pytensor
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    BoundaryCorrectionConfig,
    BiospherePriorConfig,
    SharedBaselineConfig,
    Co2ModelConfig,
    Co2O2ModelConfig,
    build_co2_model,
    build_rhime_co2_o2,
    build_shared_co2_from_config,
    build_shared_co2_model,
    annotate_shared_baseline_trace,
    load_boundary_correction,
    load_biosphere_prior,
    load_shared_baseline,
)
from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.execution import load_co2_params
from openghg_inversions.rhime.co2.inference import build_co2_inference
from openghg_inversions.serialization import save_trace, load_trace
from test_biosphere_prior import case as bio_case
from test_boundary_correction_priors import operands
from test_co2_cached_sigma import _inputs
from test_co2_o2_fixed_ou import _prepared, _kwargs
from test_centered_handoffs import _ordinary
from test_shared_baseline import case as shared_case


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def width(value, units="ppm"):
    return {"value": value, "units": units}


@pytest.mark.parametrize("family", ["level", "A", "curtain_contrasts", "independent_curtains"])
@pytest.mark.parametrize("period", [False, True])
def test_boundary_configuration_units_reference_periods_and_frozen_reload(tmp_path, family, period):
    obs = _inputs().mf
    options = {"family": family, "reference": "whole_window_transport_mean"}
    kwargs = {"method": family, "reference": "whole_window_transport_mean"}
    if family != "independent_curtains":
        options["level_sd"] = width(5000, "ppb")
        kwargs["level_sd"] = 5.0
    if family != "level":
        options["pattern_sd"] = width(0.7, "1" if family == "A" else "ppm")
        kwargs["pattern_sd"] = 0.7
    if period:
        options.update(
            frequency="2h", anchor_time="2021-01-01", level_by_period=family != "independent_curtains"
        )
        kwargs.update({k: options[k] for k in ("frequency", "anchor_time", "level_by_period")})
    config = BoundaryCorrectionConfig.from_params(options)
    with Callback(pretask=lambda *a: pytest.fail("preparation must stay lazy")):
        actual = config.prepare(operands(obs).chunk({"nmeasure": 2}), obs)
    expected = prepare_boundary_correction(operands(obs), obs, **kwargs)
    xr.testing.assert_allclose(actual, expected)
    path = tmp_path / "bc.nc"
    config.save(actual, path)
    restored = load_boundary_correction(path)
    xr.testing.assert_identical(restored, actual.compute())
    subset = restored.isel(nmeasure=[1, 3])
    xr.testing.assert_identical(subset.bc_reference, restored.bc_reference)
    assert subset.attrs == restored.attrs
    with pytest.raises(FileExistsError):
        config.save(actual, path)


@pytest.mark.parametrize("policy", ["residual_sd_multiplier", "target_residual_sd", "target_gross_sd"])
@pytest.mark.parametrize("family", ["lognormal", "gaussian"])
def test_biosphere_configuration_policies_functionals_and_saved_identity(tmp_path, policy, family):
    prior, inputs = bio_case()
    expected = calibrate_biosphere_prior(prior, **inputs, residual_sd_multiplier=0.5, family=family)
    target = (
        0.5
        if policy == "residual_sd_multiplier"
        else width(
            float(
                expected.data.functional_sd.sel(
                    prior="requested", functional="gross" if policy == "target_gross_sd" else "residual_gross"
                )
            )
            * 1000,
            "g/year",
        )
    )
    config = BiospherePriorConfig.from_params({"family": family, policy: target})
    actual = config.prepare(prior, **inputs)
    xr.testing.assert_allclose(actual.data.covariance, expected.data.covariance)
    path = tmp_path / "bio.nc"
    config.save(actual, path)
    reopened = load_biosphere_prior(path)
    xr.testing.assert_identical(reopened.data, actual.data)
    xr.testing.assert_identical(reopened.baseline.mean, prior.mean)
    xr.testing.assert_identical(reopened.baseline.arithmetic_covariance, prior.arithmetic_covariance)
    if reopened.positive_prior is not None:
        np.testing.assert_array_equal(reopened.positive_prior.arithmetic_covariance, actual.data.covariance)


@pytest.mark.parametrize("mode", ["fixed", "additive", "multiplicative", "multiplicative_shrunk"])
def test_configured_shared_preparation_matches_python_and_preserves_subset_means(tmp_path, mode):
    obs, h, prior, expected = shared_case(mode)
    correction = prepare_boundary_correction(
        operands(obs), obs, method="A", level_sd=5, pattern_sd=0.7, frequency="2D", anchor_time="2021-01-01"
    )
    kwargs = {}
    if mode == "additive":
        kwargs.update(additive_design=expected.additive_raw, additive_density_sd=expected.additive_density_sd)
    if mode == "multiplicative_shrunk":
        rename = {
            name: name.removeprefix("calibration_")
            for name in set(expected.variables) | set(expected.dims)
            if name.startswith("calibration_")
        }
        # Recover the exact reviewed input using the saved calibration codec.
        import json
        from openghg_inversions.biosphere_prior import BiospherePrior

        calibration = (
            expected[[name for name in rename if name in expected.data_vars]]
            .rename(rename)
            .assign_attrs(json.loads(expected.attrs["calibration_attrs"]))
        )
        kwargs["calibrated_prior"] = BiospherePrior.from_dataset(calibration)
    config = SharedBaselineConfig.from_params({"mode": mode, "delta_sd": width(5000, "ppb")})
    actual = config.prepare(
        h,
        prior,
        xr.full_like(obs, 0.2),
        correction,
        outer=prior.mean.copy(data=[False, True, True, True]),
        sampled_outer=prior.mean.copy(data=[False, True, True, False]),
        **kwargs,
    )
    xr.testing.assert_allclose(actual, expected)
    config.save(actual, tmp_path / "shared.nc")
    restored = load_shared_baseline(tmp_path / "shared.nc")
    xr.testing.assert_identical(restored, actual)
    xr.testing.assert_identical(restored.isel(nmeasure=[1, 3]).bc_pattern_mean, actual.bc_pattern_mean)


@pytest.mark.parametrize(
    "options",
    [
        {"family": "bogus"},
        {"family": "level", "level_sd": width(1), "pattern_sd": width(1)},
        {"family": "A", "level_sd": width(1), "pattern_sd": width(1, "kg")},
        {"family": "level", "level_sd": width(True)},
        {"family": "level", "level_sd": width(1), "level_by_period": True},
        {"family": "level", "level_sd": width(1), "reference": "rolling"},
        {"family": "level", "level_sd": width(1), "unknown": 1},
    ],
)
def test_invalid_boundary_options_fail(options):
    with pytest.raises((ValueError, TypeError)):
        BoundaryCorrectionConfig.from_params(options)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"residual_sd_multiplier": 0.5, "target_gross_sd": width(1, "kg/year")},
        {"residual_sd_multiplier": 0.5, "family": "truncated"},
        {"target_gross_sd": {"value": 1}},
        {"residual_sd_multiplier": 0.5, "recalibrate": True},
    ],
)
def test_invalid_biosphere_options_fail(options):
    with pytest.raises((ValueError, TypeError)):
        BiospherePriorConfig.from_params(options)


def _parity(left, right):
    assert [rv.name for rv in left.free_RVs] == [rv.name for rv in right.free_RVs]
    point = left.initial_point()
    for name, value in point.items():
        point[name] = value + np.linspace(0.1, 0.3, value.size).reshape(value.shape)
    np.testing.assert_allclose(left.compile_logp()(point), right.compile_logp()(point), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        left.compile_dlogp()(point), right.compile_dlogp()(point), rtol=1e-11, atol=1e-9
    )

    def prediction(model):
        return model.compile_fn(
            model.replace_rvs_by_values([model.modelled_concentration]),
            inputs=model.value_vars,
            on_unused_input="ignore",
        )(point)[0]

    np.testing.assert_allclose(prediction(left), prediction(right), rtol=1e-13, atol=1e-11)


@pytest.mark.parametrize("cached", [False, True])
def test_configured_co2_inference_uses_same_bc_calibration_and_nonzero_target(tmp_path, cached):
    prepared = _ordinary()
    inv = prepared.inv_inputs
    inv = inv.assign(
        alpha_prior_mean=inv.alpha_prior_mean.copy(data=np.ones(2)),
        alpha_prior_covariance=inv.alpha_prior_covariance.copy(data=np.array([[0.08, 0.01], [0.01, 0.06]])),
    )
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inv))
    prior = CorrelatedLognormalPrior(
        inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
    )
    mean = prior.mean
    weights = xr.DataArray(
        [[-2.0, 0], [0, 3.0]],
        dims=("process", "region"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-2.0, 3.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs=weights.attrs
    )
    bio_config = BiospherePriorConfig.from_params({"residual_sd_multiplier": 0.5})
    bio = bio_config.prepare(
        prior, biosphere=mean.copy(data=[True, True]), weights=weights, reference=reference
    )
    bc_config = BoundaryCorrectionConfig.from_params(
        {"family": "A", "level_sd": width(0.2), "pattern_sd": width(0.3, "1")}
    )
    bc = bc_config.prepare(operands(inv.mf), inv.mf)
    bio_config.save(bio, tmp_path / "bio.nc")
    bc_config.save(bc, tmp_path / "bc.nc")
    likelihood = (
        {"kind": "fixed_ou", "tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
        if cached
        else {"kind": "additive_sigma", "fixed_model_mismatch": 0.3}
    )
    request = {
        "model": {
            "boundary": {"correction_path": str(tmp_path / "bc.nc")},
            "biosphere": {"path": str(tmp_path / "bio.nc")},
        },
        "likelihood": likelihood,
    }
    choices = Co2ModelConfig.from_params(load_co2_params(params=request)).model_arguments()
    built, _ = build_co2_inference(prepared, choices)
    if cached:
        expected = build_co2_model(
            inv.H,
            retained_prior=prior,
            biosphere_prior=bio,
            fixed_prior_contribution=inv.fixed_prior_contribution,
            observations=inv.mf,
            observation_error=inv.mf_error,
            aggregation_error=resolve_aggregation_error(inv, prepared.aggregation_error_mode),
            boundary_correction=bc,
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs={k: v for k, v in likelihood.items() if k != "kind"},
        )
    else:
        python_choices = {**choices, "boundary_correction": bc, "biosphere_prior": bio}
        expected = build_co2_inference(prepared, python_choices)[0].model
    _parity(built.model, expected)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_config_independent_species_corrections_and_one_calibration(tmp_path, cached):
    prepared = _prepared()
    kwargs = _kwargs(prepared)
    prior = prepared.retained_prior
    mean = prior.mean
    weights = xr.DataArray(
        [[-10, 0, 0, 0, 0], [0, 8, 0, 0, 0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-10.0, 8.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs=weights.attrs
    )
    bio_config = BiospherePriorConfig.from_params({"residual_sd_multiplier": 0.5})
    bio = bio_config.prepare(
        prior,
        biosphere=mean.copy(data=[True, True, False, False, False]),
        weights=weights,
        reference=reference,
    )
    bio_config.save(bio, tmp_path / "bio.nc")
    corrections, channels = {}, {}
    for channel, scalar, sd in (("co2", 400.0, 0.2), ("o2", 209300.0, 0.003)):
        obs = select_gathered_data_array(
            kwargs["observations"],
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        )
        row = str(kwargs[f"{channel}_sensitivity"].dims[0])
        obs = obs.rename(observation=row)
        config = BoundaryCorrectionConfig.from_params(
            {"family": "A", "level_sd": width(sd), "pattern_sd": width(0.3, "1")}
        )
        corrections[channel] = config.prepare(operands(obs, scalar), obs)
        config.save(corrections[channel], tmp_path / f"{channel}.nc")
        channels[channel] = {
            "units": "ppm",
            "independent_error_sd": 0.1,
            "boundary": {"correction_path": str(tmp_path / f"{channel}.nc")},
        }
    request = {"channels": channels, "biosphere": {"path": str(tmp_path / "bio.nc")}}
    choices = Co2O2ModelConfig.from_params(load_co2_o2_params(params=request)).bind(prepared)
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    if cached:
        configured = build_co2_o2_cached_sigma_model(**{**kwargs, **choices}, **ou).model
    else:
        configured = build_rhime_co2_o2(prepared_inputs=prepared, **choices).model
    expected = build_co2_o2_model(
        **kwargs, biosphere_prior=bio, boundary_correction=corrections, **(ou if cached else {})
    )
    _parity(configured, expected)
    assert len([rv for rv in configured.free_RVs if rv.name.startswith("flux_scaling")]) == 1


@pytest.mark.parametrize("cached", [False, True])
def test_shared_configured_graph_matches_python_and_saves_absolute_trace(tmp_path, cached):
    obs, _, _, prepared = shared_case("multiplicative_shrunk")
    inputs = dict(
        observations=obs,
        observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    likelihood = {"kind": "fixed_ou", **ou}
    if not cached:
        likelihood.pop("site_amplitude_prior")
        likelihood["fixed_site_amplitudes"] = 0.75
    actual = build_shared_co2_from_config({"likelihood": likelihood}, prepared=prepared, **inputs)
    model = actual.model if cached else actual
    expected = build_shared_co2_model(
        prepared,
        **inputs,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={k: v for k, v in likelihood.items() if k != "kind"},
    )
    _parity(model, expected)
    from openghg_inversions.models.boundary import restore_absolute_predictives

    with model:
        trace = pm.sample_prior_predictive(2, random_seed=57)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    trace = restore_absolute_predictives(trace)
    trace = annotate_shared_baseline_trace(trace, model)
    save_trace(trace, tmp_path / "trace.nc")
    restored = load_trace(tmp_path / "trace.nc")
    xr.testing.assert_identical(restored.shared_baseline.to_dataset(), prepared)
    xr.testing.assert_identical(restored.prior_predictive.to_dataset(), trace.prior_predictive.to_dataset())


@pytest.mark.parametrize("no_model_error", [False, True])
def test_shared_additive_sigma_keeps_configured_defaults_and_fixed_mismatch(no_model_error):
    from openghg_inversions.sigma import SigmaAlignment

    obs, _, _, prepared = shared_case()
    inputs = dict(
        observations=obs,
        observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )
    actual = build_shared_co2_from_config(
        {
            "likelihood": {
                "kind": "additive_sigma",
                "no_model_error": no_model_error,
                "fixed_model_mismatch": 0.4,
            }
        },
        prepared=prepared,
        **inputs,
    )
    expected = build_shared_co2_model(
        prepared,
        **inputs,
        fixed_model_mismatch=0.4,
        sigma_alignment=None if no_model_error else SigmaAlignment.from_observations(obs),
    )
    _parity(actual, expected)


def test_inference_options_cannot_change_saved_calibration_or_boundary():
    for other in (
        {"enabled": True},
        {"frequency": "8D"},
        {"prior": {"pdf": "normal", "mu": 1, "sigma": 0.1}},
    ):
        with pytest.raises(ValueError, match="owns"):
            Co2ModelConfig.from_params(
                {
                    "model": {"boundary": {"correction_path": "frozen.nc", **other}},
                    "likelihood": {"kind": "additive_sigma"},
                }
            )
    with pytest.raises(ValueError, match="Unknown"):
        load_co2_params(params={"model": {"biosphere": {"path": "frozen.nc", "target_gross_sd": 1}}})
    with pytest.raises(ValueError, match="Unknown"):
        load_co2_o2_params(params={"shared_baseline": {"path": "frozen.nc"}})


def test_toml_component_preparation_is_same_as_python():
    request = tomllib.loads("""
        [boundary]
        family = "A"
        reference = { value = 400, units = "ppm" }
        level_sd = { value = 5, units = "ppm" }
        pattern_sd = { value = 0.7, units = "1" }
        [biosphere]
        family = "lognormal"
        target_gross_sd = { value = 8, units = "kg/year" }
    """)
    assert BoundaryCorrectionConfig.from_params(request["boundary"]).reference == (400.0, "ppm")
    assert BiospherePriorConfig.from_params(request["biosphere"]).policy == "target_gross_sd"


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("stage", ["sample", "prior_predictive"])
def test_saved_workflow_protects_selected_calibration_from_output_alias(tmp_path, monkeypatch, linked, stage):
    from openghg_inversions.rhime.co2 import handoffs, co2_o2_handoffs

    module = co2_o2_handoffs if linked else handoffs
    filename = "posterior.nc" if stage == "sample" else "prior-predictive.nc"
    prior, inputs = bio_case()
    config = BiospherePriorConfig.from_params({"residual_sd_multiplier": 0.5})
    config.save(config.prepare(prior, **inputs), tmp_path / filename)
    if linked:
        model = Co2O2ModelConfig.from_params(
            {
                "channels": {name: {"units": "ppm", "independent_error_sd": 0.1} for name in ("co2", "o2")},
                "biosphere": {"path": str(tmp_path / filename)},
            }
        )
    else:
        model = Co2ModelConfig.from_params(
            {
                "model": {"biosphere": {"path": str(tmp_path / filename)}},
                "likelihood": {"kind": "additive_sigma"},
            }
        )
    record = SimpleNamespace(
        prepared=_prepared(), payload=tmp_path / "prepared.nc", affine_payload=None, generation="g"
    )
    monkeypatch.setattr(module, "load_preparation", lambda *a: record)
    with pytest.raises(ValueError, match="overwrite"):
        getattr(module, stage)(tmp_path / "prepare.json", model=model, output_dir=tmp_path)
    xr.testing.assert_identical(load_biosphere_prior(tmp_path / filename).baseline.mean, prior.mean)


def test_saved_calibration_documented_example(tmp_path, monkeypatch):
    import sys
    from examples.biosphere_prior.saved_configuration import main

    output = tmp_path / "example"
    monkeypatch.setattr(sys, "argv", ["saved_configuration", str(output)])
    main()
    assert {path.name for path in output.iterdir()} == {"biosphere.nc", "boundary.nc", "prior-predictive.nc"}
