"""Linked cached runners retain absolute products and model-local BC periods."""

from dataclasses import replace
import json

import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner as runner
from openghg_inversions.rhime.co2.centered_boundary import restore_absolute_predictives
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import load_trace, save_trace

from test_centered_co2_boundaries import _operands
from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error


def _prepared_boundaries(mixed):
    prepared = _prepared()
    operands, absolute = {}, {}
    for channel in ("co2", "o2"):
        native = select_gathered_data_array(
            prepared.observations, key=channel, key_dim="species",
            ragged_dim="channel_observation", stack_dim="observation",
        )
        raw = _operands(native)
        positions = [0, 1] if channel == "co2" else [0, 1, 0]
        raw = raw.isel(bc_region=positions).assign_coords(
            bc_region=[f"{channel}-curtain-{i}" for i in range(len(positions))],
        )
        raw["bc_reference"] = xr.DataArray(
            400.0 if channel == "co2" else 209500.0, attrs={"units": "ppm"},
        )
        operands[channel] = raw
        native_dim = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        absolute[channel] = (
            raw.H_bc_anomaly + raw.bc_reference * raw.H_bc_mass
        ).rename(nmeasure=native_dim).transpose(native_dim, "bc_region").assign_attrs(
            units="ppm", boundary_sensitivity_preparation="raw-spatial",
        )
    return replace(
        prepared,
        boundary_operands={"co2": operands["co2"]} if mixed else operands,
        boundary_sensitivity={"o2": absolute["o2"]} if mixed else {},
        independent_error_sd=_independent_error(prepared),
    ), operands


@pytest.mark.parametrize("route", ["centered", "mixed", "disabled"])
def test_linked_cached_runner_restores_absolute_predictions_and_replay(tmp_path, monkeypatch, route):
    prepared, raw_operands = _prepared_boundaries(mixed=route != "centered")
    aggregation_covariance = prepared.aggregation_error.covariance.copy(deep=True)
    captured = {}

    def prior_prediction(built, sampler):
        with built.model:
            trace = pm.sample_prior_predictive(draws=3, random_seed=73)
        trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
        captured["centered_y"] = trace.prior_predictive.y.copy(deep=True)
        return trace

    monkeypatch.setattr(runner, "sample_rhime_model", prior_prediction)
    anchor = "2020-12-31T23:00:00"
    with pytensor.config.change_flags(floatX="float64"):
        trace = runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
            prepared_inputs=prepared, tau_hours=3.0,
            site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.1},
            bc_freq="3h", bc_anchor_time=anchor,
            bc_prior={channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.01}
                      for channel in ("co2", "o2") if channel != "co2" or route != "disabled"},
            use_bc={"co2": route != "disabled", "o2": True},
            sampler=RhimeSampler(draws=3, tune=2, chains=1, nuts_sampler="pymc"),
        )
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    assert roles["observation"] == "y"
    xr.testing.assert_allclose(trace.observed_data.y, prepared.observations)
    xr.testing.assert_identical(prepared.aggregation_error.covariance, aggregation_covariance)
    expected_total = trace.prior.co2_o2_flux_contribution.values + trace.constant_data.fixed_prior_contribution.values
    for channel in ("co2", "o2"):
        if channel == "co2" and route == "disabled":
            assert "co2_bc" not in trace.prior
            assert "co2_boundary_scale" not in roles
            continue
        scale = trace.prior[f"{channel}_bc"]
        curtains = scale[f"{channel}_bc_curtain"].values
        periods = scale[f"{channel}_bc_period"].values
        raw = raw_operands[channel]
        raw = raw.sel(bc_region=xr.DataArray(curtains, dims="state"))
        matrix = (raw.H_bc_anomaly + raw.bc_reference * raw.H_bc_mass).transpose("nmeasure", "state").values
        times = raw.time.values.astype("datetime64[ns]")
        row_periods = ((times - np.datetime64(anchor)) // np.timedelta64(3, "h")).astype(int)
        row_periods -= row_periods.min()
        matrix *= row_periods[:, None] == periods[None, :]
        expected = np.zeros_like(expected_total)
        selected = prepared.observations.species.values == channel
        expected[..., selected] = scale.values @ matrix.T
        np.testing.assert_allclose(trace.prior[f"{channel}_mu_bc"], expected, atol=1e-9)
        expected_total += expected
        if channel in prepared.boundary_operands:
            assert roles[f"{channel}_boundary_anomaly"] == f"{channel}_H_bc_anomaly"
            assert roles[f"{channel}_boundary_mass"] == f"{channel}_H_bc_mass"
            assert f"{channel}_boundary_sensitivity" not in roles
        else:
            assert roles[f"{channel}_boundary_sensitivity"] == f"{channel}_hbc"
    np.testing.assert_allclose(trace.prior.modelled_concentration, expected_total, atol=1e-9)
    if route != "disabled":
        metadata = json.loads(trace.attrs["rhime_model_metadata"])
        assert metadata["bc_frequency"] == "3h"
        assert metadata["bc_anchor_time"] == anchor
        xr.testing.assert_allclose(
            trace.prior_predictive.y,
            captured["centered_y"] + trace.constant_data.boundary_reference_shift,
        )
        assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    else:
        assert "boundary_reference_shift" not in trace.constant_data
        xr.testing.assert_allclose(trace.prior_predictive.y, captured["centered_y"])
    np.testing.assert_allclose(trace.prior_predictive.y, expected_total, atol=10, rtol=0)
    path = tmp_path / "linked-native-trace.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    restore_absolute_predictives(reopened)
    xr.testing.assert_identical(reopened.observed_data.y, trace.observed_data.y)
    xr.testing.assert_identical(reopened.prior_predictive.y, trace.prior_predictive.y)


@pytest.mark.parametrize("enabled", [True, False])
def test_linked_cached_precision_guard_runs_after_selection_before_materialization(monkeypatch, enabled):
    prepared, _ = _prepared_boundaries(mixed=True)

    class ReachedMaterialization(Exception):
        pass

    def materialize(prepared, error, boundaries, operands):
        assert not operands
        raise ReachedMaterialization

    monkeypatch.setattr(runner, "_materialize_co2_o2_replay_inputs", materialize)
    with pytensor.config.change_flags(floatX="float32"):
        expected = ValueError if enabled else ReachedMaterialization
        with pytest.raises(expected, match="floatX=float64" if enabled else None):
            runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared, tau_hours=3.0,
                site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.1},
                use_bc={"co2": enabled, "o2": True},
            )
        assert pytensor.config.floatX == "float32"
