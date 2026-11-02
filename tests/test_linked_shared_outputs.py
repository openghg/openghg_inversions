"""Saved linked shared products preserve atmospheric and physical accounting."""

from dataclasses import replace
import json

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.rhime.co2.co2_o2_model import _gather_co2_o2_sensitivity
from openghg_inversions.rhime.co2.co2_o2_outputs import (
    Co2O2OutputOptions, calculate_co2_o2_products, make_co2_o2_output,
    validate_co2_o2_product_request, validate_co2_o2_trace,
)
from openghg_inversions.rhime.co2.linked_shared_baseline_model import reconstruct_linked_shared_baseline
from openghg_inversions.rhime.co2.shared_preparation import prepare_linked_shared_inputs
from test_linked_shared_preparation import native, options, precision  # noqa: F401


@pytest.fixture
def saved(native, monkeypatch):  # noqa: F811
    prepared = prepare_linked_shared_inputs(native, **options(native))
    selected = prepared.shared_baseline["co2"].retained_mean
    state = selected.expand_dims(chain=[0], draw=[0, 1]) * xr.DataArray([1., 1.2], dims="draw")
    state.attrs["units"] = "dimensionless flux scale"
    full = state.reindex(state=prepared.retained_prior.mean.state).fillna(prepared.retained_prior.mean)
    full = full.assign_coords(prepared.retained_prior.mean.coords)
    full = full.rename(state="physical_state", source="physical_source", tracer_scope="physical_tracer_scope")
    draws = xr.Dataset({"flux_scaling": state, "full_physical_flux_scaling": full})
    for channel in ("co2", "o2"):
        frozen = prepared.shared_baseline[channel]
        draws[f"{channel}_bc_level"] = xr.DataArray([[[0.], [0.4 if channel == "co2" else -.2]]],
                                                    dims=("chain", "draw", f"{channel}_bc_level_term"))
        draws[f"{channel}_bc_pattern"] = xr.DataArray(
            np.full((1, 2, frozen.sizes["bc_pattern_term"]), .25),
            dims=("chain", "draw", f"{channel}_bc_pattern_term"),
            coords={f"{channel}_bc_pattern_term": frozen.bc_pattern_term.values},
        )
    native_components = reconstruct_linked_shared_baseline(prepared.shared_baseline, draws)
    modelled = _gather_co2_o2_sensitivity(*(native_components[c].modelled_concentration for c in ("co2", "o2")))
    modelled = modelled.assign_coords(observation_units=prepared.observations.observation_units)
    draws["modelled_concentration"] = modelled.transpose("chain", "draw", "observation")
    draws["epsilon"] = xr.ones_like(draws.modelled_concentration)
    roles = {"flux_scale": "full_physical_flux_scaling", "modelled_concentration": "modelled_concentration",
             "total_marginal_error": "epsilon", "concentration": "y"}
    tree = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"rhime_recipe": "co2_o2_shared_baseline", "rhime_variable_roles": json.dumps(roles)}),
        "posterior": draws, "prior": draws.isel(draw=[0]),
        "posterior_predictive": xr.Dataset({"y": draws.modelled_concentration + .1}),
        **{f"shared_baseline_{c}": p for c, p in prepared.shared_baseline.items()},
    })
    monkeypatch.setattr(pm.Model, "__init__", lambda *a, **kw: pytest.fail("products constructed a model"))
    monkeypatch.setattr(pm, "sample", lambda *a, **kw: pytest.fail("products sampled"))
    return prepared, tree


def test_graph_free_products_close_with_separate_removed_means(saved):
    prepared, trace = saved
    result = make_co2_o2_output(prepared=prepared, trace=trace, start_date="2020-01-01",
                               end_date="2020-01-02", domain="EUROPE")
    calculate_co2_o2_products(result=result, output=Co2O2OutputOptions())
    for channel, size in (("co2", 2), ("o2", 3)):
        values = result.outputs[f"{channel}_concentration_components"]
        row = prepared.shared_baseline[channel].attrs["native_observation_dim"]
        assert values.sizes[row] == size
        for group in ("prior", "posterior"):
            xr.testing.assert_allclose(
                values[f"modelled_concentration_{group}"],
                values[f"physical_atmospheric_total_{group}"] - values[f"removed_mean_correction_{group}"],
            )
            xr.testing.assert_allclose(values[f"residual_{group}"],
                                       values.observed - values[f"modelled_concentration_{group}"])
        np.testing.assert_allclose(values.bc_raw_correction_posterior - values.bc_centered_correction_posterior,
                                   values.bc_removed_mean_posterior)
        assert "physical_ff_posterior" in values
        assert "predictive_posterior" in values
        assert "total_error" in values
    scaling = result.outputs["state_scaling"].flux_scale_posterior
    xr.testing.assert_identical(scaling.state, prepared.retained_prior.mean.state)
    assert scaling.sizes["state"] == prepared.retained_prior.mean.size
    assert trace.posterior.flux_scaling.sizes["state"] < scaling.sizes["state"]


@pytest.mark.parametrize("fault", ["missing", "changed", "old_preparation"])
def test_reject_saved_shared_identity_mismatch(saved, fault):
    prepared, trace = saved
    if fault == "missing":
        del trace["shared_baseline_o2"]
    elif fault == "changed":
        trace["shared_baseline_o2"].attrs["mode"] = "fixed"
    else:
        prepared = replace(prepared, shared_baseline={})
    with pytest.raises(ValueError, match="shared|differs|recipe"):
        validate_co2_o2_trace(trace, prepared)


def test_shared_paris_fails_before_destination_access(saved, tmp_path):
    prepared, _ = saved
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="removed-mean"):
        validate_co2_o2_product_request(Co2O2OutputOptions(output_format="paris", output_path=destination), prepared)
    assert not destination.exists()


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_saved_trace_replay_preserves_shared_components(saved, tmp_path, extension):
    from openghg_inversions.serialization import load_trace, save_trace

    prepared, trace = saved
    path = tmp_path / f"trace.{extension}"
    save_trace(trace, path)
    restored = load_trace(path)
    result = make_co2_o2_output(prepared=prepared, trace=restored, start_date="2020-01-01",
                               end_date="2020-01-02", domain="EUROPE")
    calculate_co2_o2_products(result=result, output=Co2O2OutputOptions())
    assert result.outputs["o2_concentration_components"].sizes["o2_measure"] == 3
    xr.testing.assert_allclose(result.outputs["o2_concentration_components"].baseline_delta_posterior,
                               trace.posterior.o2_bc_level.isel(o2_bc_level_term=0, drop=True))
