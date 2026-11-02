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


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_shared_control_targets_use_frozen_atmospheric_recipe(mode):
    import pytensor
    from openghg_inversions.rhime.co2 import reconstruct_linked_shared_baseline
    from openghg_inversions.array_ops import select_gathered_data_array

    with pytensor.config.change_flags(floatX="float64"):
        zero = _example.prepare_control(mode=mode)
        injected = _example.prepare_control(mode=mode, injected_baseline=True)
    truth = xr.DataArray(_example.TRUTH, dims="state", coords={"state": _example.STATE_LABELS})
    coefficients = xr.Dataset({"flux_scaling": truth.sel(state=zero.shared_baseline["co2"].state)})
    for channel in ("co2", "o2"):
        coefficients[f"{channel}_bc_level"] = xr.DataArray([0.], dims=f"{channel}_bc_level_term")
    expected = reconstruct_linked_shared_baseline(zero.shared_baseline, coefficients)
    for channel, delta in (("co2", .35), ("o2", -.45)):
        def native(array):
            return select_gathered_data_array(array, key=channel, key_dim="species",
                ragged_dim="channel_observation", stack_dim="observation").rename(
                    observation=zero.shared_baseline[channel].attrs["native_observation_dim"])
        xr.testing.assert_allclose(native(zero.observations), expected[channel].modelled_concentration,
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(native(injected.observations) - native(zero.observations), delta,
                                   rtol=0, atol=1e-10)


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_declared_boundary_policy_survives_saved_product_replay(inputs, tmp_path, monkeypatch, extension):
    import json
    import pymc as pm
    import pytensor
    from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2ModelConfig
    from openghg_inversions.rhime.co2.co2_o2_outputs import (
        Co2O2OutputOptions, calculate_co2_o2_products, make_co2_o2_output, save_co2_o2_products,
    )
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
    from openghg_inversions.rhime.co2.shared_workflow import (
        annotate_linked_shared_workflow, build_linked_shared_workflow,
    )
    from openghg_inversions.serialization import load_trace, save_trace

    channels, _ = inputs
    declared = {}
    copied = []
    original_attrs = [dict(channel.site_data["SYNTH"].attrs) for channel in channels]
    for name, channel in zip(("co2", "o2"), channels, strict=True):
        method = "nearest" if name == "o2" else ("linear" if extension == "nc" else "interval")
        attrs = {"boundary_temporal_policy": method,
                 "boundary_input_semantics": "interval_mean" if method == "interval" else "point",
                 "boundary_retrieval_padding": "0 days 06:00:00",
                 "boundary_temporal_policy_revision": "native-fields-v1"}
        if method == "interval":
            attrs["boundary_temporal_bounds"] = "native_time_bounds"
        else:
            attrs["boundary_temporal_max_gap"] = "0 days 06:00:00"
        if method == "nearest":
            attrs["boundary_temporal_nearest_tie"] = "later"
        declared[name] = attrs
        site = channel.site_data["SYNTH"].assign_attrs({**channel.site_data["SYNTH"].attrs, **attrs})
        copied.append(replace(channel, site_data={"SYNTH": site}))
    monkeypatch.setattr(_example, "synthetic_surface_inputs", lambda **kwargs: tuple(copied))
    with pytensor.config.change_flags(floatX="float64"):
        prepared = _example.prepare_control()
        config = Co2O2ModelConfig.from_params({
            "recipe": "co2_o2_shared_baseline", "channels": {
                name: {"units": "ppm", "independent_error_sd": .05} for name in ("co2", "o2")},
        })
        built, _ = build_linked_shared_workflow(prepared, **config.bind(prepared))
        with built.model:
            trace = pm.sample_prior_predictive(2, random_seed=12)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    # Synthetic replay draws test metadata, not posterior scientific acceptance.
    trace["posterior"] = xr.DataTree(trace.prior.to_dataset())
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
    expected = prepared.provenance["surface_template_recipe"]
    for name in ("co2", "o2"):
        record = expected[name]["sites"]["SYNTH"]
        assert record["boundary_temporal_provenance"] == declared[name]
        assert record["selected_surface_inputs"]["sources"].keys() == _example.SOURCE_STATE[name].keys()
    empty = [prepared.shared_baseline[channel][name] for channel in ("co2", "o2")
             for name in ("bc_pattern_raw", "bc_pattern_mean", "bc_pattern_centered")]
    borrowed = [(value.data, value.data.__dask_graph__(), value.chunks, dict(value.encoding)) for value in empty]
    path = tmp_path / f"prepared.{extension}"
    prepared.save(path)
    for value, (data, graph, chunks, encoding) in zip(empty, borrowed, strict=True):
        assert value.data is data and data.__dask_graph__() is graph
        assert value.chunks == chunks and value.encoding == encoding
    restored = Co2O2PreparedInputs.load(path)
    assert restored.provenance == prepared.provenance
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored.shared_baseline[channel], prepared.shared_baseline[channel].compute())
    trace_path = tmp_path / f"trace.{extension}"
    save_trace(trace, trace_path)
    reopened = load_trace(trace_path)
    assert json.loads(reopened.attrs["rhime_model_metadata"])["provenance"] == prepared.provenance
    monkeypatch.setattr(pm.Model, "__init__", lambda *a, **kw: pytest.fail("replay constructed model"))
    monkeypatch.setattr(pm, "sample", lambda *a, **kw: pytest.fail("replay sampled"))
    result = make_co2_o2_output(prepared=restored, trace=reopened, start_date="2020-01-01",
                               end_date="2020-01-02", domain="SYNTHETIC")
    output = Co2O2OutputOptions(output_path=tmp_path / "products", save_concentrations=True)
    calculate_co2_o2_products(result=result, output=output)
    save_co2_o2_products(result, output=output)
    for name, product in result.outputs.items():
        assert json.loads(product.attrs["linked_preparation_provenance"]) == prepared.provenance
        with xr.open_dataset(result.output_metadata[f"{name}_path"]) as saved:
            assert json.loads(saved.attrs["linked_preparation_provenance"]) == prepared.provenance
    legacy_channels = []
    for channel in channels:
        site = channel.site_data["SYNTH"].copy(deep=False)
        site.attrs = {name: value for name, value in site.attrs.items() if name not in {
            "boundary_temporal_policy", "boundary_temporal_policy_revision",
            "boundary_temporal_nearest_tie", "boundary_input_semantics",
            "boundary_temporal_max_gap", "boundary_temporal_bounds", "boundary_retrieval_padding",
        }}
        legacy_channels.append(replace(channel, site_data={"SYNTH": site}))
    legacy = assemble(legacy_channels, inputs[1]).provenance["surface_template_recipe"]
    for channel, attrs in zip(channels, original_attrs, strict=True):
        assert channel.site_data["SYNTH"].attrs == attrs
        assert "boundary_temporal_provenance" not in legacy[channel.acquisition.species]["sites"]["SYNTH"]
