"""Native surface template assembly preserves scientific labels and lazy graphs."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2.surface_templates import prepare_linked_surface_templates

_spec = spec_from_file_location("linked_workflow", Path(__file__).parents[1] / "examples/shared_baseline/linked_workflow.py")
_example = module_from_spec(_spec)
_spec.loader.exec_module(_example)


@pytest.fixture(scope="module")
def inputs():
    channels = _example.synthetic_surface_inputs()
    prepared = _example.synthetic_linked_inputs()
    return channels, prepared


def assemble(channels, prepared, **kwargs):
    options = dict(retained_prior=prepared.retained_prior, source_state=_example.SOURCE_STATE,
                   aggregation_error=prepared.aggregation_error, provenance={"test": True})
    options.update(kwargs)
    return prepare_linked_surface_templates(*channels, **options)


def test_native_support_signed_responses_and_zero_intercept(inputs):
    channels, prepared = inputs
    result = assemble(channels, prepared)
    assert result.provenance["start_date"] == "2020-01-01"
    assert result.provenance["end_date"] == "2020-01-02"
    assert result.provenance["domain"] == "SYNTHETIC"
    assert result.co2_sensitivity.sizes["co2_observation"] == 6
    assert result.o2_sensitivity.sizes["o2_observation"] == 4
    assert result.observations.indexes["observation"].tolist() == [
        *(("co2", i) for i in range(6)), *(("o2", i) for i in range(4)),
    ]
    for array in (result.observations, result.independent_error_sd, result.co2_sensitivity,
                  result.o2_sensitivity, result.boundary_operands["o2"].H_bc_anomaly):
        assert isinstance(array.data, da.Array)
    np.testing.assert_array_equal(result.fixed_prior_contribution.compute(), 0.)
    np.testing.assert_array_equal(result.o2_sensitivity.sel(state="co2_ocean").compute(), 0.)
    np.testing.assert_array_equal(result.co2_sensitivity.sel(state="o2_ocean").compute(), 0.)
    native = channels[1].site_data["SYNTH"]
    expected = native.fp_x_flux_sectoral.sel(source="gpp").sum(("lat", "lon"))
    np.testing.assert_allclose(result.o2_sensitivity.sel(state="gpp").compute(), expected.compute())
    assert result.o2_co2_flux_ratio is None
    assert result.o2_co2_flux_ratio_unavailable_reason
    record = result.provenance["surface_template_recipe"]
    assert record["co2"]["sites"]["SYNTH"]["selected_surface_inputs"]["sources"].keys() == _example.SOURCE_STATE["co2"].keys()
    assert record["co2"]["sites"]["SYNTH"]["bc_reference"] == 400.
    boundary = result.boundary_operands["o2"]
    assert boundary.bc_region.values.tolist() == list("nesw")
    np.testing.assert_allclose(boundary.H_bc_mass.compute(), .25)
    np.testing.assert_allclose(boundary.H_bc_anomaly.compute(), 0.)
    assert boundary.bc_reference.item() == 209000.


def test_physical_unit_conversion_once_preserves_original_reference(inputs):
    channels, prepared = inputs
    original = channels[0].site_data["SYNTH"]
    changed = original.copy(deep=False)
    concentration_fields = ["mf", "mf_error", "fp_x_flux_sectoral", "bc_reference",
                            *(f"bc_anomaly_{direction}" for direction in "nesw")]
    for name in concentration_fields:
        changed[name] = (changed[name] * 1000).assign_attrs(units="ppb")
    converted = replace(channels[0], site_data={"SYNTH": changed})
    result = assemble((converted, channels[1]), prepared)
    np.testing.assert_allclose(result.observations.compute(), prepared.observations.compute())
    np.testing.assert_allclose(result.co2_sensitivity.compute(), prepared.co2_sensitivity.compute())
    assert result.boundary_operands["co2"].bc_reference.item() == 400.
    assert result.provenance["surface_template_recipe"]["co2"]["sites"]["SYNTH"]["bc_reference"] == 400000.
    assert original.bc_reference.item() == 400.


@pytest.mark.parametrize("failure", ["missing_source", "extra_source", "unknown_state", "missing_state", "opposite_ocean"])
def test_source_state_rejects_incomplete_or_unavailable_mapping(inputs, failure):
    channels, prepared = inputs
    mapping = {channel: dict(values) for channel, values in _example.SOURCE_STATE.items()}
    if failure == "missing_source":
        del mapping["co2"]["gpp"]
    elif failure == "extra_source":
        mapping["co2"]["extra"] = "gpp"
    elif failure == "unknown_state":
        mapping["co2"]["gpp"] = "not-a-state"
    elif failure == "missing_state":
        mapping["co2"]["gpp"] = "ter"
    else:
        mapping["co2"]["co2_ocean"] = "o2_ocean"
    with pytest.raises(ValueError, match="source_state"):
        assemble(channels, prepared, source_state=mapping)


@pytest.mark.parametrize("failure,match", [
    ("error", "mf_error"), ("provenance", "provenance"), ("fact", "acquisition facts"),
    ("inlet", "site/inlet"), ("endpoint", "bc_mass_w"), ("reference", "scalar bc_reference"),
    ("units", "convert observation units"), ("cov_units", "ppm units"), ("cov_labels", "exact gathered"),
])
def test_rejects_missing_evidence_and_covariance_mismatches(inputs, failure, match):
    channels, prepared = inputs
    co2, o2 = channels
    site = co2.site_data["SYNTH"].copy(deep=False)
    if failure == "error":
        site = site.drop_vars("mf_error")
    elif failure == "provenance":
        site.attrs = {**site.attrs, "surface_policy_revision": "unknown"}
    elif failure == "fact":
        co2 = replace(co2, acquisition=replace(co2.acquisition, domain=None))
    elif failure == "inlet":
        co2 = replace(co2, site_options=replace(co2.site_options, inlet=(None,)))
    elif failure == "endpoint":
        site = site.drop_vars("bc_mass_w")
    elif failure == "reference":
        site["bc_reference"] = xr.DataArray([400.], dims="extra", attrs={"units": "ppm"})
    elif failure == "units":
        site["mf"] = site.mf.assign_attrs(units="kg")
    elif failure == "cov_units":
        covariance = prepared.aggregation_error.covariance.assign_coords(
            observation_units=("observation", ["ppb"] * 10))
        prepared = replace(prepared, aggregation_error=replace(prepared.aggregation_error, covariance=covariance))
    else:
        covariance = prepared.aggregation_error.covariance.isel(observation=slice(None, None, -1))
        prepared = replace(prepared, aggregation_error=replace(prepared.aggregation_error, covariance=covariance))
    co2 = replace(co2, site_data={"SYNTH": site})
    with pytest.raises((ValueError, KeyError), match=match):
        assemble((co2, o2), prepared)


def test_explicit_multiple_sources_sum_into_one_coefficient(inputs):
    import json

    channels, prepared = inputs
    co2, o2 = channels
    original = co2.site_data["SYNTH"]
    site = original.drop_vars(["fp_x_flux_sectoral", "mf_mod_sectoral"]).drop_dims("source")
    responses = original.fp_x_flux_sectoral
    extra = responses.sel(source=["gpp"]).assign_coords(source=["extra_gpp"])
    site["fp_x_flux_sectoral"] = xr.concat((responses, extra), dim="source")
    records = json.loads(site.attrs["surface_transport_provenance"])
    records["sources"]["extra_gpp"] = records["sources"]["gpp"]
    site.attrs = {**site.attrs, "surface_transport_provenance": json.dumps(records)}
    co2 = replace(co2, site_data={"SYNTH": site}, flux_data={**co2.flux_data, "extra_gpp": co2.flux_data["gpp"]})
    mapping = {channel: dict(values) for channel, values in _example.SOURCE_STATE.items()}
    mapping["co2"]["extra_gpp"] = "gpp"
    result = assemble((co2, o2), prepared, source_state=mapping)
    np.testing.assert_allclose(result.co2_sensitivity.sel(state="gpp").compute(),
                               2 * prepared.co2_sensitivity.sel(state="gpp").compute())


def test_rejects_disagreeing_native_site_references(inputs):
    from openghg_inversions.inversion_data import SiteOptions

    channels, prepared = inputs
    co2, o2 = channels
    second = co2.site_data["SYNTH"].copy(deep=False)
    second["bc_reference"] = xr.DataArray(401., attrs={"units": "ppm"})
    second.attrs = {**second.attrs, "site": "OTHER"}
    co2 = replace(co2, site_data={**co2.site_data, "OTHER": second}, site_options=SiteOptions.from_inputs(
        sites=["SYNTH", "OTHER"], inlet="10m", fp_height="10m", averaging_period="1h", platform="surface",
    ))
    with pytest.raises(ValueError, match="one agreed scalar bc_reference"):
        assemble((co2, o2), prepared)
