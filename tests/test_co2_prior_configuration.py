"""Configured priors retain the explicit Python target; boundary and shared choices bind at construction."""

from types import SimpleNamespace
from dataclasses import replace
import tomllib

import numpy as np
import pymc as pm
import pytest
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AnomalyScalingChoice, RawCenteredSensitivity
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import AggregationError, resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    BiospherePriorConfig,
    Co2ModelConfig,
    Co2O2ModelConfig,
    build_co2_model,
    build_rhime_co2_o2,
    build_shared_co2_from_config,
    build_shared_co2_model,
    annotate_shared_baseline_trace,
    load_biosphere_prior,
    shared_baseline_choice,
)
from openghg_inversions.models.shared_baseline import SharedBaseline, SharedBaselineChoice
from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.execution import load_co2_params
from openghg_inversions.rhime.co2.inference import build_co2_inference
from openghg_inversions.serialization import save_trace, load_trace
from test_biosphere_prior import case as bio_case
from test_co2_bound_frontends import _native, _operands
from test_co2_o2_fixed_ou import _prepared, _kwargs
from test_centered_handoffs import _ordinary
from test_shared_baseline import SCALING, case as shared_case, operands as shared_operands


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def test_public_configuration_annotations_resolve_at_runtime():
    from typing import get_type_hints

    assert get_type_hints(Co2ModelConfig.model_arguments)
    assert get_type_hints(build_shared_co2_from_config)["aggregation_error"] is AggregationError
    assert get_type_hints(build_shared_co2_from_config)["baseline"] is SharedBaseline
    assert get_type_hints(shared_baseline_choice)["return"] is SharedBaselineChoice
    for method in ("from_params", "prepare", "save"):
        assert get_type_hints(getattr(BiospherePriorConfig, method))


def width(value, units="ppm"):
    return {"value": value, "units": units}


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
    bio_config.save(bio, tmp_path / "bio.nc")
    mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": 0.2}
    boundary = AnomalyScalingChoice(mean_shift_prior, 0.3)
    likelihood = (
        {"kind": "fixed_ou", "tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}}
        if cached
        else {"kind": "additive_sigma", "fixed_model_mismatch": 0.3}
    )
    request = {
        "model": {
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
            "biosphere": {"path": str(tmp_path / "bio.nc")},
        },
        "likelihood": likelihood,
    }
    choices = Co2ModelConfig.from_params(load_co2_params(params=request)).model_arguments()
    assert choices["boundary"] == boundary
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
            boundary=boundary.bind(inv.mf, RawCenteredSensitivity.validate(inv)),
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs={k: v for k, v in likelihood.items() if k != "kind"},
        )
    else:
        python_choices = {**choices, "boundary": boundary, "biosphere_prior": bio}
        expected = build_co2_inference(prepared, python_choices)[0].model
    _parity(built.model, expected)


class _Built(Exception):
    pass


@pytest.mark.parametrize("cached", [False, True])
def test_linked_config_independent_species_corrections_and_one_calibration(tmp_path, monkeypatch, cached):
    from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner

    prepared = _prepared()
    prepared = replace(
        prepared, boundary_operands={channel: _operands(_native(prepared, channel)) for channel in ("co2", "o2")},
    )
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
    expected_boundary, channels = {}, {}
    for channel, sd in (("co2", 0.2), ("o2", 0.003)):
        mean_shift_prior = {"pdf": "normal", "mu": 0.0, "sigma": sd}
        native_dim = str(kwargs[f"{channel}_sensitivity"].dims[0])
        expected_boundary[channel] = AnomalyScalingChoice(mean_shift_prior, 0.3).bind(
            _native(prepared, channel).rename(observation=native_dim),
            prepared.boundary_operands[channel],
        )
        channels[channel] = {
            "units": "ppm",
            "independent_error_sd": 0.1,
            "boundary": {"type": "anomaly_scaling", "scale_sd": 0.3, "mean_shift_prior": mean_shift_prior},
        }
    request = {"channels": channels, "biosphere": {"path": str(tmp_path / "bio.nc")}}
    choices = Co2O2ModelConfig.from_params(load_co2_o2_params(params=request)).bind(prepared)
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    if cached:
        built = []
        builder = co2_o2_cached_sigma_runner._build_co2_o2_cached_sigma_model

        def capture(**model_kwargs):
            built.append(builder(**model_kwargs).model)
            raise _Built

        monkeypatch.setattr(co2_o2_cached_sigma_runner, "_build_co2_o2_cached_sigma_model", capture)
        with pytest.raises(_Built):
            co2_o2_cached_sigma_runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared, **choices, **ou,
            )
        configured = built[0]
    else:
        configured = build_rhime_co2_o2(prepared_inputs=prepared, **choices).model
    expected = build_co2_o2_model(
        **kwargs, biosphere_prior=bio, boundary=expected_boundary, **(ou if cached else {})
    )
    _parity(configured, expected)
    assert len([rv for rv in configured.free_RVs if rv.name.startswith("flux_scaling")]) == 1


def _shared_inputs(obs):
    return dict(
        observations=obs,
        observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )


@pytest.mark.parametrize("cached", [False, True])
def test_shared_configured_graph_matches_python_and_saves_absolute_trace(tmp_path, cached):
    obs, h, prior, expected_baseline = shared_case("multiplicative_shrunk")
    choice = shared_baseline_choice({
        "mode": "multiplicative_shrunk",
        "delta_sd": 5.0,
        "boundary": {"type": "anomaly_scaling", "scale_sd": 0.7, "frequency": "2D", "anchor_time": "2021-01-01"},
    })
    assert choice == SharedBaselineChoice(SCALING, 5.0, "multiplicative_shrunk")
    weights = xr.DataArray(
        [[0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
        dims=("process", "region"),
        coords={"process": ["GPP", "TER"], **dict(prior.mean.coords)},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-2.0, 3.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    sampled = prior.mean.copy(data=[False, True, True, False])
    baseline = choice.bind(
        obs,
        flux_sensitivity=h,
        retained_prior=prior,
        fixed_prior_contribution=xr.full_like(obs, 0.2),
        boundary_sensitivity=shared_operands(obs),
        outer=prior.mean.copy(data=[False, True, True, True]),
        sampled_outer=sampled,
        calibrated_prior=calibrate_biosphere_prior(
            prior, biosphere=sampled, weights=weights, reference=reference, residual_sd_multiplier=0.5
        ),
    )
    xr.testing.assert_identical(baseline.data, expected_baseline.data)
    inputs = _shared_inputs(obs)
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    likelihood = {"kind": "fixed_ou", **ou}
    if not cached:
        likelihood.pop("site_amplitude_prior")
        likelihood["fixed_site_amplitudes"] = 0.75
    actual = build_shared_co2_from_config({"likelihood": likelihood}, baseline=baseline, **inputs)
    model = actual.model if cached else actual
    expected = build_shared_co2_model(
        expected_baseline,
        **inputs,
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={k: v for k, v in likelihood.items() if k != "kind"},
    )
    _parity(model, expected)
    from openghg_inversions.postprocessing.boundary_outputs import restore_absolute_predictives

    with model:
        trace = pm.sample_prior_predictive(2, random_seed=57)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    trace = restore_absolute_predictives(trace)
    trace = annotate_shared_baseline_trace(trace, baseline)
    save_trace(trace, tmp_path / "trace.nc")
    restored = load_trace(tmp_path / "trace.nc")
    xr.testing.assert_identical(restored.shared_baseline.to_dataset(), baseline.data.compute())
    xr.testing.assert_identical(restored.prior_predictive.to_dataset(), trace.prior_predictive.to_dataset())


@pytest.mark.parametrize("no_model_error", [False, True])
def test_shared_additive_sigma_keeps_configured_defaults_and_fixed_mismatch(no_model_error):
    from openghg_inversions.sigma import SigmaAlignment

    obs, _, _, baseline = shared_case()
    inputs = _shared_inputs(obs)
    actual = build_shared_co2_from_config(
        {
            "likelihood": {
                "kind": "additive_sigma",
                "no_model_error": no_model_error,
                "fixed_model_mismatch": 0.4,
            }
        },
        baseline=baseline,
        **inputs,
    )
    expected = build_shared_co2_model(
        baseline,
        **inputs,
        fixed_model_mismatch=0.4,
        sigma_alignment=None if no_model_error else SigmaAlignment.from_observations(obs),
    )
    _parity(actual, expected)


def test_inference_options_cannot_change_saved_calibration():
    with pytest.raises(ValueError, match="Unknown"):
        load_co2_params(params={"model": {"biosphere": {"path": "frozen.nc", "target_gross_sd": 1}}})
    with pytest.raises(ValueError, match="Unknown"):
        load_co2_o2_params(params={"shared_baseline": {"path": "frozen.nc"}})


def test_toml_component_choices_are_same_as_python():
    request = tomllib.loads("""
        [shared_baseline]
        mode = "multiplicative"
        delta_sd = 5.0
        [shared_baseline.boundary]
        type = "anomaly_scaling"
        scale_sd = 0.7
        frequency = "2D"
        anchor_time = "2021-01-01"
        [biosphere]
        family = "lognormal"
        target_gross_sd = { value = 8, units = "kg/year" }
    """)
    assert shared_baseline_choice(request["shared_baseline"]) == SharedBaselineChoice(SCALING, 5.0, "multiplicative")
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
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        getattr(module, stage)(tmp_path / "prepare.json", model=model, output_dir=tmp_path)
    xr.testing.assert_identical(load_biosphere_prior(tmp_path / filename).baseline.mean, prior.mean)


def test_saved_calibration_documented_example(tmp_path, monkeypatch):
    import sys
    from examples.biosphere_prior.saved_configuration import main

    output = tmp_path / "example"
    monkeypatch.setattr(sys, "argv", ["saved_configuration", str(output)])
    main()
    assert {path.name for path in output.iterdir()} == {"biosphere.nc", "prior-predictive.nc"}


@pytest.mark.parametrize("request_kind", ["params_file", "config_file"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_full_run_protects_consumed_request_before_model_resolution(tmp_path, monkeypatch, request_kind, alias):
    import json
    from openghg_inversions.rhime.co2 import execution

    prepared = _ordinary()
    source = tmp_path / "request.nc"
    destination = source if alias == "direct" else tmp_path / "trace.nc"
    facts = {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"}
    request = {
        "prepared_inputs": facts,
        "likelihood": {"kind": "additive_sigma"},
        "outputs": {"output_format": "basic", "save_trace": str(destination)},
    }
    if request_kind == "params_file":
        source.write_text(json.dumps(request), encoding="utf-8")
    else:
        source.write_text(
            "[prepared_inputs]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in facts.items())
            + '\n[likelihood]\nkind = "additive_sigma"\n[outputs]\noutput_format = "basic"\nsave_trace = '
            + json.dumps(str(destination)) + "\n", encoding="utf-8",
        )
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        destination.hardlink_to(source)
    before = source.read_bytes()
    monkeypatch.setattr(Co2ModelConfig, "model_arguments", lambda *a: pytest.fail("model resolution reached"))
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        execution.run_co2(**{request_kind: source}, prepared=prepared)
    assert source.read_bytes() == before
    assert destination.read_bytes() == before


def test_full_run_mapping_request_adds_no_file_to_protected_inputs(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2 import execution

    request = {
        "prepared_inputs": {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"},
        "likelihood": {"kind": "additive_sigma"},
        "outputs": {"output_format": "basic", "save_trace": str(tmp_path / "trace.nc")},
    }
    def reached_model(*args):
        raise RuntimeError("request passed preflight")
    monkeypatch.setattr(Co2ModelConfig, "model_arguments", reached_model)
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(RuntimeError, match="request passed preflight"):
        execution.run_co2(params=request, prepared=_ordinary())
    assert not (tmp_path / "trace.nc").exists()


def _save_biosphere_prior(prepared, artifact):
    inv = prepared.inv_inputs
    prior = CorrelatedLognormalPrior(inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov")
    weights = xr.DataArray([[-1., 0.], [0., 1.]], dims=("process", "region"),
                           coords={"process": ["GPP", "TER"], **prior.mean.coords}, attrs={"units": "kg/year"})
    config = BiospherePriorConfig.from_params({"residual_sd_multiplier": 1.})
    bio = config.prepare(prior, biosphere=prior.mean.copy(data=[True, True]), weights=weights,
                         reference=xr.dot(weights, prior.mean, dim="region").assign_attrs(units="kg/year"))
    config.save(bio, artifact)
    return prior


@pytest.mark.parametrize("target", ["trace", "basic"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_full_run_protects_selected_prior_before_sampling(tmp_path, monkeypatch, target, alias):
    from openghg_inversions.rhime.co2 import execution

    prepared = _ordinary()
    artifact = tmp_path / "selected-prior.nc"
    prior = _save_biosphere_prior(prepared, artifact)
    model = {"biosphere": {"path": str(artifact)}}
    destination = tmp_path / ("trace.nc" if target == "trace" else "rhime_basic_co2_EUROPE_2020-01-01.nc")
    if alias == "direct":
        artifact.rename(destination)
        artifact = destination
        model["biosphere"]["path"] = str(artifact)
    elif alias == "symlink":
        destination.symlink_to(artifact)
    else:
        destination.hardlink_to(artifact)
    before = artifact.read_bytes()
    request = {
        "prepared_inputs": {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"},
        "model": model, "likelihood": {"kind": "additive_sigma"},
        "outputs": {"output_format": "basic", "output_path": str(tmp_path),
                    "save_trace": str(destination) if target == "trace" else False},
    }
    # Resolve the real matching scientific artifact, rather than bypass loading.
    resolved = Co2ModelConfig.from_params(request).model_arguments()
    xr.testing.assert_identical(resolved["biosphere_prior"].baseline.mean, prior.mean)
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        execution.run_co2(params=request, prepared=prepared)
    assert artifact.read_bytes() == before
    assert destination.read_bytes() == before


@pytest.mark.parametrize("enclosing_trace", [False, True])
def test_full_run_protects_prior_from_overlapping_directory_targets(tmp_path, monkeypatch, enclosing_trace):
    from openghg_inversions.rhime.co2 import execution

    prepared = _ordinary()
    trace = tmp_path / "trace.zarr"
    artifact = trace / "prior.nc" if enclosing_trace else tmp_path / "prior.zarr"
    if not enclosing_trace:
        trace = artifact / "trace.nc"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    _save_biosphere_prior(prepared, artifact)
    def contents():
        files = sorted(artifact.rglob("*")) if artifact.is_dir() else [artifact]
        return {str(path): path.read_bytes() for path in files if path.is_file()}
    before = contents()
    request = {
        "prepared_inputs": {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"},
        "model": {"biosphere": {"path": str(artifact)}},
        "likelihood": {"kind": "additive_sigma"},
        "outputs": {"save_trace": str(trace)},
    }
    monkeypatch.setattr(execution, "sample_co2_inference", lambda *a, **kw: pytest.fail("sampling reached"))
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        execution.run_co2(params=request, prepared=prepared)
    assert contents() == before


def test_ordinary_linked_configuration_binds_declared_fields():
    prepared = _prepared()
    request = {"channels": {name: {"units": "ppm", "independent_error_sd": .1}
                             for name in ("co2", "o2")}}
    bound = Co2O2ModelConfig.from_params(request).bind(prepared)
    np.testing.assert_array_equal(bound["independent_error_sd"], .1)
    assert "site_amplitude_prior" not in bound
