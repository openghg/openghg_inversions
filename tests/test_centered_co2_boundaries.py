"""Centered CO2 keeps the absolute statistical model and durable replay semantics."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import align_boundary_operands
from openghg_inversions.models import StateActivity, get_coord_registry, restore_inferencedata_coords
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.rhime.co2 import build_co2_model, Co2PreparedInputs, Co2O2PreparedInputs
from openghg_inversions.rhime.co2.centered_boundary import restore_absolute_predictives
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2, annotate_co2_o2_trace
from openghg_inversions.rhime.co2.co2_runner import (
    materialize_co2_inputs,
    build_co2_from_materialized,
    annotate_co2_trace,
)
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from test_co2_rhime import _golden_inputs
from test_co2_affine_output import _prepared
from test_rhime_co2_o2_baselines import _prepared as _linked


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _operands(observations, *, constant=False):
    dim = str(observations.dims[0])
    n = observations.size
    coords = {name: coord for name, coord in observations.coords.items() if coord.dims == (dim,)}
    values = xr.Dataset(
        {
            "H_bc_anomaly": (
                ("bc_region", dim),
                np.zeros((2, n)) if constant else np.arange(2 * n).reshape(2, n) * 0.03 - 0.04,
            ),
            "H_bc_mass": (("bc_region", dim), np.linspace(0.2, 1.5, 2 * n).reshape(2, n)),
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        coords={dim: observations[dim], "bc_region": ["north", "south"], **coords},
        attrs={"boundary_representation": "centered-v1"},
    )
    if "time" not in values.coords:
        values = values.assign_coords(time=(dim, np.arange(n).astype("datetime64[D]")))
    values.H_bc_anomaly.attrs["units"] = "ppm"
    values.H_bc_mass.attrs["units"] = "1"
    if dim != "nmeasure":
        values = values.rename({dim: "nmeasure"})
    return values


def _single_models(constant, fixed, frequency=None, default_prior=False):
    inputs = _golden_inputs().assign_coords(time=("nmeasure", np.arange(2).astype("datetime64[D]")))
    operands = _operands(inputs.mf, constant=constant)
    aligned = align_boundary_operands(operands, frequency=frequency)
    absolute = aligned.H_bc_anomaly + aligned.bc_reference * aligned.H_bc_mass
    options = dict(
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=inputs.mf,
        observation_error=xr.full_like(inputs.mf, 0.3),
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        bc_prior=None if default_prior else {"pdf": "normal", "mu": 1.0, "sigma": 0.3},
        bc_state_activity=StateActivity(
            active=[True, False] if fixed else True, fixed_value=[1.0, 1.7] if fixed else 1.0
        ),
    )
    return (
        build_co2_model(inputs.H, boundary_sensitivity=absolute, **options),
        build_co2_model(inputs.H, boundary_operands=operands, bc_freq=frequency, **options),
    )


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_absolute_centered_normalized_logp_and_gradients(constant, fixed):
    absolute, centered = _single_models(constant, fixed)
    assert [v.name for v in absolute.free_RVs] == [v.name for v in centered.free_RVs]
    point = absolute.initial_point()
    absolute_logp, centered_logp = absolute.compile_logp(), centered.compile_logp()
    absolute_gradient, centered_gradient = absolute.compile_dlogp(), centered.compile_dlogp()
    for value in (-0.3, 0.65, 1.35):
        bc_name = "bc_active" if fixed else "bc"
        point[bc_name] = np.full_like(point[bc_name], value)
        np.testing.assert_allclose(absolute_logp(point), centered_logp(point), rtol=2e-12)
        np.testing.assert_allclose(absolute_gradient(point), centered_gradient(point), rtol=2e-12, atol=1e-8)


def test_trace_restoration_idempotent_without_model(tmp_path):
    _, model = _single_models(True, True)
    with model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=12)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    centered_y = trace.prior_predictive.y.copy()
    restore_absolute_predictives(trace)
    xr.testing.assert_allclose(
        trace.prior_predictive.y, centered_y + trace.constant_data.boundary_reference_shift
    )
    once = trace.copy(deep=True)
    restore_absolute_predictives(trace)
    xr.testing.assert_identical(trace, once)
    from openghg_inversions.serialization import save_trace, load_trace

    path = tmp_path / "trace.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    restore_absolute_predictives(reopened)
    xr.testing.assert_allclose(reopened.prior_predictive.y, trace.prior_predictive.y)


def _centered_prepared():
    prepared, _ = _prepared()
    operands = _operands(prepared.inv_inputs.mf)
    inputs = (
        prepared.inv_inputs.assign({name: value.variable for name, value in operands.data_vars.items()})
        .assign_coords(bc_region=operands.bc_region)
        .assign_attrs(boundary_representation="centered-v1")
    )
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))


@pytest.mark.parametrize("predictions", [False, True])
def test_prepared_single_save_reopen_and_tiny_runner(tmp_path, predictions):
    prepared = _centered_prepared()
    path = tmp_path / "co2.nc"
    prepared.save(path)
    restored = Co2PreparedInputs.load(path)
    xr.testing.assert_allclose(restored.inv_inputs.H_bc_mass, prepared.inv_inputs.H_bc_mass)
    options = dict(use_bc=True, no_model_error=True, bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.01})
    inputs, priors = materialize_co2_inputs(prepared_inputs=restored, **options)
    built = build_co2_from_materialized(inputs, prepared_inputs=restored, priors=priors, **options)
    sampler = RhimeSampler(
        draws=2,
        tune=2,
        chains=1,
        burn=0,
        nuts_sampler="pymc",
        sample_prior_predictive=True,
        sample_posterior_predictive=True,
        posterior_predictive_kwargs={"predictions": predictions},
        sample_kwargs={"compute_convergence_checks": False, "random_seed": 3},
    )
    trace = sample_rhime_model(built, sampler)
    group = "predictions" if predictions else "posterior_predictive"
    centered_y = trace[group].y.copy(deep=True)
    if predictions:
        standalone = xr.DataTree.from_dict({
            name: trace[name].to_dataset()
            for name in ("predictions", "predictions_constant_data")
        })
        restore_absolute_predictives(standalone)
        xr.testing.assert_allclose(
            standalone.predictions.y,
            centered_y + standalone.predictions_constant_data.boundary_reference_shift,
        )
    annotate_co2_trace(trace, built, concentration_units="ppm")
    xr.testing.assert_allclose(trace.observed_data.y, prepared.inv_inputs.mf)
    xr.testing.assert_allclose(trace[group].y, centered_y + trace.constant_data.boundary_reference_shift)
    assert trace[group].y.attrs["concentration_representation"] == "absolute-v1"
    assert "boundary_reference_shift" in trace.constant_data
    from openghg_inversions.serialization import save_trace, load_trace

    trace_path = tmp_path / "trace.nc"
    save_trace(trace, trace_path)
    reopened = load_trace(trace_path)
    annotate_co2_trace(reopened, built, concentration_units="ppm")
    xr.testing.assert_identical(reopened[group].y, trace[group].y)


def test_centered_prepared_with_absolute_diagnostic_save_reopen_build(tmp_path):
    prepared = _centered_prepared()
    inputs = prepared.inv_inputs
    diagnostic = (inputs.H_bc_anomaly + inputs.bc_reference * inputs.H_bc_mass).assign_attrs(
        boundary_sensitivity_preparation="unexpanded", units="ppm"
    )
    prepared = replace(
        prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs.assign(H_bc=diagnostic))
    )
    path = tmp_path / "diagnostic.nc"
    prepared.save(path)
    restored = Co2PreparedInputs.load(path)
    options = dict(use_bc=True, no_model_error=True)
    materialized, priors = materialize_co2_inputs(prepared_inputs=restored, **options)
    assert "H_bc" in materialized
    built = build_co2_from_materialized(materialized, prepared_inputs=restored, priors=priors, **options)
    assert "H_bc_anomaly" in built.model.named_vars


@pytest.mark.parametrize("predictions", [False, True])
def test_linked_independent_axes_codec_and_prior_prediction(tmp_path, predictions):
    prepared = _linked()
    from openghg_inversions.array_ops import select_gathered_data_array

    operands = {}
    for channel in ("co2", "o2"):
        native = select_gathered_data_array(
            prepared.observations,
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        )
        operands[channel] = _operands(native)
    prepared = replace(prepared, boundary_operands=operands)
    path = tmp_path / "linked.nc"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    assert restored.boundary_operands["co2"].sizes["nmeasure"] == 2
    assert restored.boundary_operands["o2"].sizes["nmeasure"] == 3
    built = build_rhime_co2_o2(
        prepared_inputs=restored,
        independent_error_sd=xr.full_like(prepared.observations, 0.3),
        bc_state_activity={channel: StateActivity(active=False, fixed_value=1.3) for channel in operands},
    )
    if predictions:
        trace = sample_rhime_model(
            built,
            RhimeSampler(
                draws=2, tune=2, chains=1, burn=0, nuts_sampler="pymc",
                sample_prior_predictive=True, sample_posterior_predictive=True,
                posterior_predictive_kwargs={"predictions": True},
                sample_kwargs={"compute_convergence_checks": False, "random_seed": 3},
            ),
        )
        centered_y = trace.predictions.y.copy(deep=True)
    else:
        with built.model:
            trace = pm.sample_prior_predictive(draws=2, random_seed=3)
        trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    annotate_co2_o2_trace(trace, built=built)
    xr.testing.assert_allclose(trace.observed_data.y, prepared.observations)
    assert trace.prior.co2_bc.shape[-1] == 2
    if predictions:
        xr.testing.assert_allclose(trace.predictions.y, centered_y + trace.constant_data.boundary_reference_shift)
        assert trace.predictions.y.attrs["concentration_representation"] == "absolute-v1"
        from openghg_inversions.serialization import save_trace, load_trace

        trace_path = tmp_path / "linked_trace.nc"
        save_trace(trace, trace_path)
        reopened = load_trace(trace_path)
        annotate_co2_o2_trace(reopened, built=built)
        xr.testing.assert_identical(reopened.predictions.y, trace.predictions.y)
    np.testing.assert_allclose(
        trace.prior.modelled_concentration,
        trace.prior.co2_o2_flux_contribution
        + trace.constant_data.fixed_prior_contribution
        + trace.prior.boundary_concentration,
    )


@pytest.mark.parametrize("standalone", [False, True])
def test_prediction_shift_selection_uses_labels_and_prediction_constants(tmp_path, standalone):
    from openghg_inversions.serialization import save_trace, load_trace

    shift = xr.DataArray([10.0, 20.0, 30.0], dims="observation", coords={"observation": ["a", "b", "c"]})
    y = xr.DataArray([[1.0, 2.0]], dims=("draw", "observation"), coords={"observation": ["c", "a"]})
    trace = xr.DataTree.from_dict({
        "predictions": xr.Dataset({"y": y}),
        "predictions_constant_data": xr.Dataset({"boundary_reference_shift": shift}),
    })
    if not standalone:
        trace["constant_data"] = xr.Dataset({"boundary_reference_shift": shift * 100})
    restore_absolute_predictives(trace)
    np.testing.assert_array_equal(trace.predictions.y, [[31.0, 12.0]])
    assert trace.predictions.y.attrs["concentration_representation"] == "absolute-v1"
    path = tmp_path / "predictions.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    restore_absolute_predictives(reopened)
    xr.testing.assert_identical(reopened, trace)


def test_unsupported_likelihood_fails_before_materialization():
    prepared = _centered_prepared()
    with pytest.raises(ValueError, match="custom or fixed-OU"):
        materialize_co2_inputs(
            prepared_inputs=prepared, likelihood_builder=lambda **kwargs: None, use_bc=True
        )


def test_linked_logp_gradients_independent_channel_labels():
    prepared = _linked()
    from openghg_inversions.array_ops import select_gathered_data_array

    operands, boundaries = {}, {}
    for channel in ("co2", "o2"):
        native = select_gathered_data_array(
            prepared.observations,
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        )
        operands[channel] = _operands(native, constant=channel == "o2")
        aligned = align_boundary_operands(operands[channel])
        sensitivity = prepared.co2_sensitivity if channel == "co2" else prepared.o2_sensitivity
        boundaries[channel] = (
            (aligned.H_bc_anomaly + aligned.bc_reference * aligned.H_bc_mass)
            .rename(nmeasure=sensitivity.dims[0])
            .transpose(sensitivity.dims[0], "bc_region")
        )
    options = dict(
        independent_error_sd=xr.full_like(prepared.observations, 0.3),
        bc_prior={channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.2} for channel in operands},
        bc_state_activity={
            channel: StateActivity(active=[True, False], fixed_value=[1.0, 1.7 if channel == "co2" else 0.7])
            for channel in operands
        },
    )
    absolute = build_rhime_co2_o2(
        prepared_inputs=replace(prepared, boundary_sensitivity=boundaries), **options
    ).model
    centered = build_rhime_co2_o2(
        prepared_inputs=replace(prepared, boundary_operands=operands), **options
    ).model
    point = absolute.initial_point()
    point["co2_bc_active"][:] = 1.25
    point["o2_bc_active"][:] = 0.75
    np.testing.assert_allclose(absolute.compile_logp()(point), centered.compile_logp()(point), rtol=1e-12)
    np.testing.assert_allclose(
        absolute.compile_dlogp()(point), centered.compile_dlogp()(point), rtol=1e-12, atol=1e-8
    )


def test_public_linked_preparation_retains_raw_operands():
    from test_rhime_co2_o2 import _inputs
    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs

    options = _inputs()
    options["o2_units"] = "ppm"
    operands = {channel: _operands(options[f"{channel}_observations"]) for channel in ("co2", "o2")}
    prepared = prepare_co2_o2_inputs(**options, boundary_operands=operands)
    assert prepared.boundary_operands["co2"] is operands["co2"]
    assert prepared.to_datatree().attrs["schema_version"] == 2
    with pytest.raises(ValueError, match="not both"):
        prepare_co2_o2_inputs(
            **options,
            boundary_operands=operands,
            boundary_sensitivity={"co2": operands["co2"].H_bc_anomaly.T},
        )


def test_explicit_float32_request_is_rejected_without_global_change():
    with pytensor.config.change_flags(floatX="float32"):
        with pytest.raises(ValueError, match="floatX=float64"):
            _single_models(False, False)
        assert pytensor.config.floatX == "float32"


def test_model_owned_period_expansion_keeps_absolute_logp():
    absolute, centered = _single_models(True, False, frequency="1D")
    point = absolute.initial_point()
    assert point["bc"].size == 4
    point["bc"][:] = [0.7, 1.4, 1.2, 0.9]
    np.testing.assert_allclose(absolute.compile_logp()(point), centered.compile_logp()(point), rtol=1e-12)
    np.testing.assert_allclose(
        absolute.compile_dlogp()(point), centered.compile_dlogp()(point), rtol=1e-12, atol=1e-8
    )


def test_related_centered_arrays_materialize_one_shared_graph():
    import dask.array as da
    from dask import delayed

    executions = []

    @delayed
    def payload():
        executions.append("computed")
        return np.array([401.0, 0.1, -0.04, 0.05, 0.2, 1.5, 400.0])

    values = da.from_delayed(payload(), shape=(7,), dtype=float)
    prepared = _centered_prepared()
    inputs = prepared.inv_inputs.assign(
        mf=prepared.inv_inputs.mf.copy(data=values[:1]),
        mf_error=prepared.inv_inputs.mf_error.copy(data=values[1:2]),
        H_bc_anomaly=prepared.inv_inputs.H_bc_anomaly.copy(data=values[2:4].reshape(2, 1)),
        H_bc_mass=prepared.inv_inputs.H_bc_mass.copy(data=values[4:6].reshape(2, 1)),
        bc_reference=prepared.inv_inputs.bc_reference.copy(data=values[6]),
    )
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    assert not executions
    options = dict(use_bc=True, no_model_error=True)
    materialized, priors = materialize_co2_inputs(prepared_inputs=prepared, **options)
    assert executions == ["computed"]
    build_co2_from_materialized(materialized, prepared_inputs=prepared, priors=priors, **options)
    assert executions == ["computed"]
    assert isinstance(prepared.inv_inputs.H_bc_mass.data, da.Array)


def test_centered_cached_and_fixed_ou_rejected_before_materialization(monkeypatch):
    from openghg_inversions.rhime.co2 import co2_o2_runner, co2_cached_sigma_runner

    def unexpected(*args, **kwargs):
        pytest.fail("materialized unsupported centered request")

    monkeypatch.setattr(co2_o2_runner, "_materialize_co2_o2_replay_inputs", unexpected)
    prepared = _linked()
    operands = _operands(prepared.co2_sensitivity.isel({prepared.retained_prior.state_dim: 0}))
    prepared = replace(prepared, boundary_operands={"co2": operands})
    with pytest.raises(ValueError, match="fixed-OU"):
        build_rhime_co2_o2(prepared_inputs=prepared, tau_hours=2.0)
    with pytest.raises(ValueError, match="cached"):
        co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
            prepared_inputs=_centered_prepared(),
            tau_hours=2.0,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.1},
            use_bc=True,
        )


def test_linked_codec_rejects_missing_centered_marker():
    prepared = _linked()
    operands = _operands(prepared.co2_sensitivity.isel({prepared.retained_prior.state_dim: 0}))
    tree = replace(prepared, boundary_operands={"co2": operands}).to_datatree()
    tree["co2_boundary_operands"].attrs.pop("boundary_representation")
    with pytest.raises(ValueError, match="centered-v1"):
        Co2O2PreparedInputs.from_datatree(tree)


def test_default_boundary_prior_support_and_error_arrays_are_unchanged():
    absolute, centered = _single_models(False, False, default_prior=True)
    for values in (np.array([0.95, 1.05]), np.array([-0.1, 1.0])):
        prior_absolute = pm.logp(absolute["bc"], values).eval()
        prior_centered = pm.logp(centered["bc"], values).eval()
        np.testing.assert_array_equal(prior_absolute, prior_centered)
        if values[0] < 0:
            assert np.isneginf(prior_centered[0])
    for name in ("error", "aggregation_error_covariance"):
        np.testing.assert_array_equal(absolute[name].get_value(), centered[name].get_value())


def test_linked_period_override_rejects_active_absolute_channel_at_direct_binding():
    prepared = _linked()
    co2 = prepared.co2_sensitivity.isel({str(prepared.co2_sensitivity.dims[-1]): 0}, drop=True)
    absolute = prepared.o2_sensitivity.rename({str(prepared.o2_sensitivity.dims[-1]): "bc_region"})
    prepared = replace(prepared, boundary_operands={"co2": _operands(co2)}, boundary_sensitivity={"o2": absolute})
    with pytest.raises(ValueError, match="every active boundary channel"):
        build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=xr.full_like(prepared.observations, 0.3), bc_freq="1D"
        )


@pytest.mark.parametrize("linked", [False, True])
def test_direct_centered_binding_rejects_nat_anchor(linked):
    if linked:
        from test_centered_handoffs import _joint
        with pytest.raises(ValueError, match="valid datetime"):
            build_rhime_co2_o2(prepared_inputs=_joint(), bc_freq="1D", bc_anchor_time="NaT")
    else:
        prepared = _centered_prepared()
        options = dict(use_bc=True, no_model_error=True, bc_freq="1D", bc_anchor_time="NaT")
        inputs, priors = materialize_co2_inputs(prepared_inputs=prepared, **options)
        with pytest.raises(ValueError, match="valid datetime"):
            build_co2_from_materialized(inputs, prepared_inputs=prepared, priors=priors, **options)
