"""Pre-bound scientific records enter every frontend without raw BC arrays."""

from dataclasses import replace
from typing import Any
import json

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models import StateActivity, get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.boundary import (
    BoundBoundary,
    boundary_output_metadata,
    boundary_output_roles,
    boundary_output_units,
    boundary_payload,
    prepare_boundary,
    replace_boundary_payload,
)
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.rhime.co2 import (
    co2_cached_sigma_runner,
    co2_o2_cached_sigma_runner,
    co2_o2_runner,
    co2_runner,
    inference,
)
from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations
from openghg_inversions.rhime.sampling import RhimeSampler
from test_centered_co2_boundaries import _operands
from test_co2_affine_output import _prepared as _single
from test_rhime_co2_o2_baselines import _prepared as _linked
from test_rhime_co2_o2 import _independent_error


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _forbid_binding(*args: Any, **kwargs: Any) -> None:
    pytest.fail("A supplied boundary must never rebind raw inputs.")


def _draws(built: Any, sampler: Any = None) -> xr.DataTree:
    with built.model:
        trace = pm.sample_prior_predictive(draws=3, random_seed=471)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    trace["posterior"] = trace.prior.to_dataset().isel(draw=slice(0, 2))
    return trace


def _bound(observations: xr.DataArray, additive: bool) -> BoundBoundary:
    operands = _operands(observations)
    choices = (
        dict(bc_mean_shift_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.3},
             bc_anomaly_scale=0.17,
             bc_centering_weights=xr.DataArray([2., 3.], dims="bc_curtain",
                                               coords={"bc_curtain": ["north", "south"]},
                                               attrs={"units": "1"}))
        if additive else dict(bc_prior={"pdf": "normal", "mu": 1.1, "sigma": 0.2},
                              bc_state_activity=StateActivity(active=[True, False], fixed_value=[1., 1.3]))
    )
    bound = prepare_boundary(observations, boundary_operands=operands, **choices)
    assert bound is not None
    return bound


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("via_inference", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_single_bound_frontends_compute_shared_payload_once_and_replay(
    monkeypatch, cached: bool, via_inference: bool, additive: bool,
) -> None:
    prepared, _ = _single()
    bound = _bound(prepared.inv_inputs.mf, additive)
    executions = []

    @delayed
    def shared_value():
        executions.append("shared")
        return np.asarray(1.0)

    shared = xr.DataArray(da.from_delayed(shared_value(), shape=(), dtype=float))
    inputs = prepared.inv_inputs.assign(H=(prepared.inv_inputs.H * shared).assign_attrs(prepared.inv_inputs.H.attrs))
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    payload = boundary_payload(bound)
    payload["data"] = payload["data"].assign(H_bc_anomaly=(payload["data"].H_bc_anomaly * shared).assign_attrs(payload["data"].H_bc_anomaly.attrs))
    payload["prior"] = dict(payload["prior"], mu=payload["prior"]["mu"] * shared)
    bound = replace_boundary_payload(bound, payload)
    assert not any(name.startswith("H_bc") or name == "bc_reference" for name in inputs)
    monkeypatch.setattr(co2_runner, "prepare_boundary", _forbid_binding)
    module = co2_cached_sigma_runner if cached else co2_runner
    name = "build_co2_cached_sigma_model" if cached else "build_co2_model"
    builder = getattr(module, name)
    forwarded = []

    def capture(*args: Any, **kwargs: Any):
        forwarded.append(kwargs["boundary"])
        return builder(*args, **kwargs)

    monkeypatch.setattr(module, name, capture)
    choices = dict(boundary=bound)
    if cached:
        choices.update(likelihood_builder=add_fixed_ou_gaussian_likelihood,
                       likelihood_kwargs={"tau_hours": 6., "site_amplitude_prior":
                                          {"pdf": "halfnormal", "sigma": 0.5}})
    else:
        choices["no_model_error"] = True
    graphs = []
    with Callback(start=lambda graph: graphs.append(graph)):
        if via_inference:
            built, _ = inference.build_co2_inference(prepared, choices)
        elif cached:
            cached_model = co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
                prepared_inputs=prepared, boundary=bound, **choices["likelihood_kwargs"],
            )
            built = co2_cached_sigma_runner.co2_cached_sigma_build_result(cached_model, prepared)
        else:
            model_inputs, priors, eager = co2_runner.materialize_co2_inputs(
                prepared_inputs=prepared, **choices,
            )
            built = co2_runner.build_co2_from_materialized(
                model_inputs, prepared_inputs=prepared, priors=priors, boundary=eager,
                no_model_error=True,
            )
    assert executions == ["shared"]
    assert len(graphs) == 1
    eager = forwarded[0]
    assert boundary_payload(eager)["prior"]["mu"].item() == (0.2 if additive else 1.1)
    assert not isinstance(eager.data.H_bc_anomaly.data, da.Array)
    assert isinstance(bound.data.H_bc_anomaly.data, da.Array)
    xr.testing.assert_identical(eager.data.bc_region, bound.data.bc_region)
    if additive:
        assert eager.anomaly_scale == bound.anomaly_scale == 0.17
    else:
        np.testing.assert_array_equal(eager.activity.active, [True, False])
        np.testing.assert_allclose(eager.activity.fixed_value, [1., 1.3])
    assert dict(built.output_contract.variable_roles).items() >= boundary_output_roles(bound).items()
    assert built.output_contract.metadata["boundary_units"] == boundary_output_units(bound)
    for key, value in boundary_output_metadata(bound).items():
        assert built.output_contract.metadata[key] == value
    trace = co2_runner.annotate_co2_trace(_draws(built), built, concentration_units="ppm")
    original = trace.copy(deep=True)
    components = reconstruct_co2_concentrations(trace, prepared)
    for group in ("prior", "posterior"):
        expected = components[f"pollution_{group}"] + components[f"boundary_{group}"]
        xr.testing.assert_allclose(components[f"modelled_{group}"], expected)
        assert components[f"boundary_{group}"].attrs["units"] == "ppm"
        state = trace[group].to_dataset()
        if additive:
            correction = state.bc_correction
            absolute = (eager.data.H_bc_anomaly + eager.data.bc_reference * eager.data.H_bc_mass).sum("bc_region")
            expected_boundary = absolute + xr.dot(eager.data.H_bc_mass, correction, dim="bc_region")
        else:
            absolute = eager.data.H_bc_anomaly + eager.data.bc_reference * eager.data.H_bc_mass
            expected_boundary = xr.dot(absolute, state.bc, dim="bc_region")
        xr.testing.assert_allclose(state.mu_bc, expected_boundary.transpose(*state.mu_bc.dims))
    xr.testing.assert_identical(trace, original)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_linked_bound_frontends_retain_channel_records_and_scientific_closure(
    monkeypatch, cached: bool, additive: bool,
) -> None:
    prepared = _linked()
    boundaries = {}
    for channel in ("co2", "o2"):
        dim = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        native = select_gathered_data_array(
            prepared.observations, key=channel, key_dim="species",
            ragged_dim="channel_observation", stack_dim="observation",
        ).rename({"observation": dim}).assign_attrs(units="ppm")
        boundaries[channel] = _bound(native, additive)
    executions = []

    @delayed
    def shared_value():
        executions.append("shared")
        return np.asarray(1.0)

    shared = xr.DataArray(da.from_delayed(shared_value(), shape=(), dtype=float))
    prepared = replace(prepared, co2_sensitivity=(prepared.co2_sensitivity * shared).assign_attrs(
        prepared.co2_sensitivity.attrs,
    ))
    for channel, bound in boundaries.items():
        payload = boundary_payload(bound)
        anomaly = payload["data"].H_bc_anomaly
        payload["data"] = payload["data"].assign(H_bc_anomaly=(anomaly * shared).assign_attrs(anomaly.attrs))
        payload["prior"] = dict(payload["prior"], mu=payload["prior"]["mu"] * shared)
        boundaries[channel] = replace_boundary_payload(bound, payload)
    assert not prepared.boundary_sensitivity and not prepared.boundary_operands
    monkeypatch.setattr(co2_o2_runner, "prepare_boundary", _forbid_binding)
    module = co2_o2_cached_sigma_runner if cached else co2_o2_runner
    monkeypatch.setattr(module, "sample_rhime_model", _draws)
    if cached:
        monkeypatch.setattr(module, "_sampler_for_cached_graph", lambda sampler, **_: sampler)
        runner = module.run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        options = dict(tau_hours=6., site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5})
    else:
        runner = module.run_rhime_co2_o2_from_prepared_inputs
        options = {}
    name = "build_co2_o2_cached_sigma_model" if cached else "build_co2_o2_model"
    builder = getattr(module, name)
    forwarded = []

    def capture(**kwargs: Any):
        forwarded.append(kwargs["boundary"])
        return builder(**kwargs)

    monkeypatch.setattr(module, name, capture)
    graphs = []
    with Callback(start=lambda graph: graphs.append(graph)):
        trace = runner(prepared_inputs=prepared, boundary=boundaries,
                       independent_error_sd=_independent_error(prepared),
                       sampler=RhimeSampler(nuts_sampler="pymc"), **options)
    assert executions == ["shared"]
    assert len(graphs) == 1
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    for channel, bound in boundaries.items():
        assert metadata["boundary_metadata"][channel] == boundary_output_metadata(bound)
        assert roles.items() >= boundary_output_roles(bound, prefix=f"{channel}_").items()
        eager = forwarded[0][channel]
        assert boundary_payload(eager)["prior"]["mu"].item() == (0.2 if additive else 1.1)
        assert isinstance(bound.data.H_bc_anomaly.data, da.Array)
        xr.testing.assert_identical(eager.data.bc_region, bound.data.bc_region)
        if additive:
            assert eager.anomaly_scale == bound.anomaly_scale == 0.17
        else:
            np.testing.assert_array_equal(eager.activity.active, [True, False])
            np.testing.assert_allclose(eager.activity.fixed_value, [1., 1.3])
        native_dim = next(dim for dim in eager.data.H_bc_mass.dims if dim != "bc_region")
        mass = eager.data.H_bc_mass.transpose(native_dim, "bc_region").values
        absolute = (eager.data.H_bc_anomaly + eager.data.bc_reference * eager.data.H_bc_mass).transpose(
            native_dim, "bc_region",
        ).values
        for group in ("prior", "posterior"):
            draws = trace[group].to_dataset()
            state = draws[f"{channel}_bc_correction" if additive else f"{channel}_bc"]
            state_dim = next(dim for dim in state.dims if dim not in {"chain", "draw"})
            coefficients = state.transpose("chain", "draw", state_dim).values
            native_mean = (absolute.sum(axis=1) + coefficients @ mass.T if additive
                           else coefficients @ absolute.T)
            expected = np.zeros((1, draws.sizes["draw"], prepared.observations.size))
            expected[..., prepared.observations.species.values == channel] = native_mean
            np.testing.assert_allclose(draws[f"{channel}_mu_bc"].transpose("chain", "draw", "observation"), expected)
            for name, units in boundary_output_units(bound, prefix=f"{channel}_").items():
                if name in draws:
                    assert draws[name].attrs["units"] == units
            np.testing.assert_allclose(
                draws[f"{channel}_mu_bc"].where(prepared.observations.species != channel, drop=True), 0,
            )
    for group in ("prior", "posterior"):
        draws = trace[group].to_dataset()
        xr.testing.assert_allclose(draws.boundary_concentration, draws.co2_mu_bc + draws.o2_mu_bc)
        xr.testing.assert_allclose(draws.modelled_concentration,
                                  draws.co2_o2_flux_contribution + draws.boundary_concentration
                                  + trace.constant_data.fixed_prior_contribution)


def _enter_frontend(kind: str, prepared: Any, boundary: Any, **choices: Any) -> Any:
    if kind == "single":
        return co2_runner.materialize_co2_inputs(prepared_inputs=prepared, boundary=boundary, **choices)
    if kind == "single_cached":
        return co2_cached_sigma_runner.build_rhime_co2_cached_sigma(
            prepared_inputs=prepared, boundary=boundary, tau_hours=6.,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5}, **choices,
        )
    if kind == "linked":
        return co2_o2_runner.build_rhime_co2_o2(prepared_inputs=prepared, boundary=boundary, **choices)
    return co2_o2_cached_sigma_runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
        prepared_inputs=prepared, boundary=boundary, tau_hours=6.,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5}, **choices,
    )


@pytest.mark.parametrize("kind", ["single", "single_cached", "linked", "linked_cached"])
@pytest.mark.parametrize("choice", ["bc_freq", "bc_anomaly_scale", "bc_prior"])
def test_bound_frontends_reject_competing_raw_choices(kind: str, choice: str) -> None:
    prepared, _ = _single()
    bound = _bound(prepared.inv_inputs.mf, True)
    if kind.startswith("linked"):
        prepared = _linked()
        bound = {"co2": bound}
    with pytest.raises(ValueError, match=rf"boundary replaces raw boundary choices; omit \['{choice}'\]"):
        _enter_frontend(kind, prepared, bound, **{choice: "explicit conflicting choice"})


@pytest.mark.parametrize("kind", ["single", "single_cached", "linked", "linked_cached"])
def test_bound_precision_rejected_before_delayed_payload_compute(kind: str) -> None:
    prepared, _ = _single()
    bound = _bound(prepared.inv_inputs.mf, True)

    @delayed
    def unread_payload():
        pytest.fail("Precision rejection must precede payload computation.")

    anomaly = bound.data.H_bc_anomaly
    bound = replace(bound, data=bound.data.assign(H_bc_anomaly=anomaly.copy(
        data=da.from_delayed(unread_payload(), shape=anomaly.shape, dtype=float),
    )))
    if kind.startswith("linked"):
        prepared = _linked()
        bound = {"co2": bound}
    with pytensor.config.change_flags(floatX="float32"):
        with pytest.raises(ValueError, match="float64"):
            _enter_frontend(kind, prepared, bound)


@pytest.mark.parametrize("cached", [False, True])
def test_empty_linked_bound_mapping_disables_available_raw_boundaries(monkeypatch, cached: bool) -> None:
    prepared = _linked(("co2", "o2"))
    monkeypatch.setattr(co2_o2_runner, "prepare_boundary", _forbid_binding)
    module = co2_o2_cached_sigma_runner if cached else co2_o2_runner
    monkeypatch.setattr(module, "sample_rhime_model", _draws)
    options = {}
    if cached:
        monkeypatch.setattr(module, "_sampler_for_cached_graph", lambda sampler, **_: sampler)
        options.update(tau_hours=6., site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5})
    runner = (module.run_rhime_co2_o2_cached_sigma_from_prepared_inputs if cached
              else module.run_rhime_co2_o2_from_prepared_inputs)
    trace = runner(prepared_inputs=prepared, boundary={},
                   independent_error_sd=_independent_error(prepared),
                   sampler=RhimeSampler(nuts_sampler="pymc"), **options)
    assert not any("boundary" in role for role in json.loads(trace.attrs["rhime_variable_roles"]))
    for group in ("prior", "posterior"):
        draws = trace[group].to_dataset()
        xr.testing.assert_allclose(draws.modelled_concentration,
                                  draws.co2_o2_flux_contribution + trace.constant_data.fixed_prior_contribution)


@pytest.mark.parametrize("cached", [False, True])
def test_raw_cli_keeps_requested_calendar_metadata_while_bound_uses_fitted_facts(cached: bool) -> None:
    prepared, _ = _single()
    operands = _operands(prepared.inv_inputs.mf)
    bound = prepare_boundary(prepared.inv_inputs.mf, boundary_operands=operands,
                             bc_prior={"pdf": "normal", "mu": 1., "sigma": 0.2},
                             bc_frequency="2D")
    assert bound is not None
    facts = boundary_output_metadata(bound)
    assert facts["bc_anchor_time"] is not None
    likelihood = (dict(likelihood_builder=add_fixed_ou_gaussian_likelihood,
                       likelihood_kwargs={"tau_hours": 6., "site_amplitude_prior":
                                          {"pdf": "halfnormal", "sigma": 0.5}})
                  if cached else dict(no_model_error=True))
    supplied, _ = inference.build_co2_inference(prepared, {"boundary": bound, **likelihood})
    for key, value in facts.items():
        assert supplied.output_contract.metadata[key] == value
    raw_inputs = prepared.inv_inputs.assign(operands)
    raw_inputs.attrs["boundary_representation"] = "centered-v1"
    raw_prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=raw_inputs))
    configured, _ = inference.build_co2_inference(
        raw_prepared, dict(use_bc=True, bc_freq="2D", bc_anchor_time=None,
                           bc_prior={"pdf": "normal", "mu": 1., "sigma": 0.2}, **likelihood),
    )
    assert configured.output_contract.metadata["bc_frequency"] == "2D"
    assert configured.output_contract.metadata["bc_anchor_time"] is None
