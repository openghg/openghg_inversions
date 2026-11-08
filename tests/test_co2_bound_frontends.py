"""Boundary choices bind once in every frontend and materialize with the model inputs."""

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
    AdditiveChoice,
    BoundaryChoice,
    MultiplicativeChoice,
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
from test_co2_affine_output import _prepared as _single
from test_rhime_co2_o2_baselines import _prepared as _linked
from test_rhime_co2_o2 import _independent_error
from openghg_inversions.rhime.co2.co2_o2_preparation import _raw_centered_channels
from boundary_test_utils import raw_transport


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _draws(built: Any, sampler: Any = None, *, recompute_deterministics: bool = False) -> xr.DataTree:
    assert recompute_deterministics is (sampler is not None)
    with built.model:
        trace = pm.sample_prior_predictive(draws=3, random_seed=471)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    trace["posterior"] = trace.prior.to_dataset().isel(draw=slice(0, 2))
    return trace


def _operands(observations: xr.DataArray) -> xr.Dataset:
    """Raw two-curtain centered operands on the native observation rows."""
    dim = str(observations.dims[0])
    n = observations.size
    coords = {name: coord for name, coord in observations.coords.items() if coord.dims == (dim,)}
    values = xr.Dataset(
        {
            "H_bc_anomaly": (("bc_region", dim), np.arange(2 * n).reshape(2, n) * 0.03 - 0.04),
            "H_bc_mass": (("bc_region", dim), np.linspace(0.2, 1.5, 2 * n).reshape(2, n)),
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        coords={dim: observations[dim], "bc_region": ["north", "south"], **coords},
        attrs={"boundary_representation": "centered-v1"},
    )
    values.H_bc_anomaly.attrs["units"] = "ppm"
    values.H_bc_mass.attrs["units"] = "1"
    return values if dim == "nmeasure" else values.rename({dim: "nmeasure"})


def _weights(observations: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        [[2.0], [3.0]], dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": ["north", "south"], "bc_period": [observations.time.values.min().astype("datetime64[ns]")]},
        attrs={"units": "1"},
    )


def _choice(additive: bool, shared: Any = 1.0) -> BoundaryChoice:
    if additive:
        return AdditiveChoice({"pdf": "normal", "mu": 0.2 * shared, "sigma": 0.3}, 0.17)
    return MultiplicativeChoice(
        {"pdf": "normal", "mu": 1.1 * shared, "sigma": 0.2},
        StateActivity(active=[True, False], fixed_value=[1., 1.3]),
    )


def _lazy(array: xr.DataArray, factor: Any) -> xr.DataArray:
    return (array * factor).assign_attrs(array.attrs)


def _single_with_operands(additive: bool, anomaly_factor: Any = 1.0) -> Any:
    prepared, _ = _single()
    operands = _operands(prepared.inv_inputs.mf)
    operands["H_bc_anomaly"] = _lazy(operands.H_bc_anomaly, anomaly_factor)
    inputs = prepared.inv_inputs.assign(operands)
    if additive:
        inputs["bc_centering_weights"] = _weights(prepared.inv_inputs.mf)
    inputs.attrs["boundary_representation"] = "centered-v1"
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))


def _native(prepared: Any, channel: str) -> xr.DataArray:
    return select_gathered_data_array(
        prepared.observations, key=channel, key_dim="species",
        ragged_dim="channel_observation", stack_dim="observation",
    )


def _linked_with_operands(additive: bool, anomaly_factor: Any = 1.0, channels=("co2", "o2")) -> Any:
    prepared = _linked()
    operands, weights = {}, {}
    for channel in channels:
        native = _native(prepared, channel)
        raw = _operands(native)
        operands[channel] = raw.assign(H_bc_anomaly=_lazy(raw.H_bc_anomaly, anomaly_factor))
        if additive:
            weights[channel] = _weights(native)
    return replace(prepared, boundary_operands=_raw_centered_channels(operands, weights))


def _record_binding(monkeypatch: Any) -> list[BoundaryChoice]:
    calls: list[BoundaryChoice] = []
    for family in (MultiplicativeChoice, AdditiveChoice):
        def record(self, *args, _bind=family.bind, **kwargs):
            calls.append(self)
            return _bind(self, *args, **kwargs)

        monkeypatch.setattr(family, "bind", record)
    return calls


def _shared_value(executions: list[str]) -> xr.DataArray:
    @delayed
    def shared_value():
        executions.append("shared")
        return np.asarray(1.0)

    return xr.DataArray(da.from_delayed(shared_value(), shape=(), dtype=float))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("via_inference", [False, True])
@pytest.mark.parametrize("additive", [False, True])
def test_single_choice_frontends_compute_shared_payload_once_and_replay(
    monkeypatch, cached: bool, via_inference: bool, additive: bool,
) -> None:
    executions: list[str] = []
    shared = _shared_value(executions)
    prepared = _single_with_operands(additive, shared)
    inputs = prepared.inv_inputs.assign(H=_lazy(prepared.inv_inputs.H, shared))
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    choice = _choice(additive, shared)
    calls = _record_binding(monkeypatch)
    module = co2_cached_sigma_runner if cached else co2_runner
    name = "build_co2_cached_sigma_model" if cached else "build_co2_model"
    builder = getattr(module, name)
    forwarded = []

    def capture(*args: Any, **kwargs: Any):
        forwarded.append(kwargs["boundary"])
        return builder(*args, **kwargs)

    monkeypatch.setattr(module, name, capture)
    choices: dict[str, Any] = dict(boundary=choice)
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
                prepared_inputs=prepared, boundary=choice, **choices["likelihood_kwargs"],
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
    assert calls == [choice]
    eager = forwarded[0]
    prior = eager.choice.mean_prior if additive else eager.choice.prior
    assert prior["mu"].item() == (0.2 if additive else 1.1)
    assert not isinstance(eager.data.H_bc_anomaly.data, da.Array)
    assert isinstance(prepared.inv_inputs.H_bc_anomaly.data, da.Array)
    lazy = choice.bind(prepared.inv_inputs.mf, raw_transport(prepared.inv_inputs))
    xr.testing.assert_identical(eager.data.bc_region, lazy.data.bc_region)
    if additive:
        assert eager.choice.anomaly_scale == choice.anomaly_scale == 0.17
    else:
        np.testing.assert_array_equal(eager.choice.activity.active, [True, False])
        np.testing.assert_allclose(eager.choice.activity.fixed_value, [1., 1.3])
    outputs = eager.outputs()
    assert dict(built.output_contract.variable_roles).items() >= outputs.roles.items()
    assert built.output_contract.metadata["boundary_units"] == outputs.units
    for key, value in outputs.metadata.items():
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
def test_linked_choice_frontends_retain_channel_records_and_scientific_closure(
    monkeypatch, cached: bool, additive: bool,
) -> None:
    executions: list[str] = []
    shared = _shared_value(executions)
    prepared = _linked_with_operands(additive, shared)
    prepared = replace(prepared, co2_sensitivity=_lazy(prepared.co2_sensitivity, shared))
    choices = {channel: _choice(additive, shared) for channel in ("co2", "o2")}
    calls = _record_binding(monkeypatch)
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
        trace = runner(prepared_inputs=prepared, boundary=choices,
                       independent_error_sd=_independent_error(prepared),
                       sampler=RhimeSampler(nuts_sampler="pymc"), **options)
    assert executions == ["shared"]
    assert len(graphs) == 1
    assert sorted(map(id, calls)) == sorted(map(id, choices.values()))
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    for channel, choice in choices.items():
        eager = forwarded[0][channel]
        outputs = eager.outputs(prefix=f"{channel}_")
        assert metadata["boundary_metadata"][channel] == outputs.metadata
        assert roles.items() >= outputs.roles.items()
        prior = eager.choice.mean_prior if additive else eager.choice.prior
        assert prior["mu"].item() == (0.2 if additive else 1.1)
        assert isinstance(prepared.boundary_operands[channel].H_bc_anomaly.data, da.Array)
        native_dim = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        lazy = choice.bind(
            _native(prepared, channel).rename({"observation": native_dim}),
            prepared.boundary_operands[channel],
        )
        xr.testing.assert_identical(eager.data.bc_region, lazy.data.bc_region)
        if additive:
            assert eager.choice.anomaly_scale == 0.17
        else:
            np.testing.assert_array_equal(eager.choice.activity.active, [True, False])
            np.testing.assert_allclose(eager.choice.activity.fixed_value, [1., 1.3])
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
            for variable, units in outputs.units.items():
                if variable in draws:
                    assert draws[variable].attrs["units"] == units
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
def test_centered_precision_rejected_before_delayed_payload_compute(kind: str) -> None:
    @delayed
    def unread_payload():
        pytest.fail("Precision rejection must precede payload computation.")

    unread = xr.DataArray(da.from_delayed(unread_payload(), shape=(), dtype=float))
    if kind.startswith("linked"):
        prepared = _linked_with_operands(True, unread, channels=("co2",))
        boundary: Any = {"co2": _choice(True)}
    else:
        prepared = _single_with_operands(True, unread)
        boundary = _choice(True)
    with pytensor.config.change_flags(floatX="float32"):
        with pytest.raises(ValueError, match="float64"):
            _enter_frontend(kind, prepared, boundary)


@pytest.mark.parametrize("cached", [False, True])
def test_empty_linked_boundary_mapping_disables_available_boundaries(monkeypatch, cached: bool) -> None:
    prepared = _linked(("co2", "o2"))
    calls = _record_binding(monkeypatch)
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
    assert calls == []
    assert not any("boundary" in role for role in json.loads(trace.attrs["rhime_variable_roles"]))
    for group in ("prior", "posterior"):
        draws = trace[group].to_dataset()
        xr.testing.assert_allclose(draws.modelled_concentration,
                                  draws.co2_o2_flux_contribution + trace.constant_data.fixed_prior_contribution)


@pytest.mark.parametrize("cached", [False, True])
def test_choice_records_requested_calendar_metadata(cached: bool) -> None:
    prepared = _single_with_operands(False)
    choice = MultiplicativeChoice({"pdf": "normal", "mu": 1., "sigma": 0.2}, frequency="2D")
    likelihood = (dict(likelihood_builder=add_fixed_ou_gaussian_likelihood,
                       likelihood_kwargs={"tau_hours": 6., "site_amplitude_prior":
                                          {"pdf": "halfnormal", "sigma": 0.5}})
                  if cached else dict(no_model_error=True))
    configured, _ = inference.build_co2_inference(prepared, {"boundary": choice, **likelihood})
    metadata = configured.output_contract.metadata
    assert metadata["bc_frequency"] == "2D"
    assert metadata["bc_anchor_time"] is None
    assert metadata["boundary_representation"] == "centered-v1"
