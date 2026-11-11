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
    ("inlet", "site/inlet"), ("endpoint", "directional A/M operands"), ("reference", "scalar bc_reference"),
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
    coefficients = xr.Dataset({"flux_scaling": truth.sel(state=zero.shared_baseline["co2"].data.state)})
    for channel in ("co2", "o2"):
        coefficients[f"{channel}_bc_mean_shift"] = xr.DataArray(0.)
    expected = reconstruct_linked_shared_baseline({channel: bound.data for channel, bound in zero.shared_baseline.items()}, coefficients)
    for channel, delta in (("co2", .35), ("o2", -.45)):
        def native(array):
            return select_gathered_data_array(array, key=channel, key_dim="species",
                ragged_dim="channel_observation", stack_dim="observation").rename(
                    observation=zero.shared_baseline[channel].data.attrs["native_observation_dim"])
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
        # Current level-only patterns are eager; this fixture tests borrowing lazy empty arrays.
        prepared = replace(prepared, shared_baseline={
            channel: replace(bound, data=bound.data.assign(
                bc_pattern_raw=bound.data.bc_pattern_raw.chunk(),
                bc_pattern_mean=bound.data.bc_pattern_mean.chunk(),
            )) for channel, bound in prepared.shared_baseline.items()
        })
        config = Co2O2ModelConfig.from_params({
            "recipe": "co2_o2_shared_baseline", "channels": {
                name: {"units": "ppm", "independent_error_sd": .05} for name in ("co2", "o2")},
        })
        built, _, materialized_prepared = build_linked_shared_workflow(prepared, **config.bind(prepared))
        with built.model:
            trace = pm.sample_prior_predictive(2, random_seed=12)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    # Synthetic replay draws test metadata, not posterior scientific acceptance.
    trace["posterior"] = xr.DataTree(trace.prior.to_dataset())
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=materialized_prepared)
    expected = prepared.provenance["surface_template_recipe"]
    for name in ("co2", "o2"):
        record = expected[name]["sites"]["SYNTH"]
        assert record["boundary_temporal_provenance"] == declared[name]
        assert record["selected_surface_inputs"]["sources"].keys() == _example.SOURCE_STATE[name].keys()
    empty = [value for bound in prepared.shared_baseline.values()
             for value in (bound.data.bc_pattern_raw, bound.data.bc_pattern_mean,
                           bound.data.bc_pattern_raw - bound.data.bc_pattern_mean)]
    borrowed = [(value.data, value.data.__dask_graph__(), value.chunks, dict(value.encoding)) for value in empty]
    path = tmp_path / f"prepared.{extension}"
    prepared.save(path)
    for value, (data, graph, chunks, encoding) in zip(empty, borrowed, strict=True):
        assert value.data is data and data.__dask_graph__() is graph
        assert value.chunks == chunks and value.encoding == encoding
    with pytensor.config.change_flags(floatX="float64"):
        restored = Co2O2PreparedInputs.load(path)
    assert restored.provenance == prepared.provenance
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored.shared_baseline[channel].data, prepared.shared_baseline[channel].data.compute())
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


def test_surface_example_frozen_codec_accepts_dask_reduction_roundoff(tmp_path):
    import pytensor
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs

    with pytensor.config.change_flags(floatX="float64"):
        prepared = _example.prepare_control()
    path = tmp_path / "surface-prepared.nc"
    prepared.save(path)
    with pytensor.config.change_flags(floatX="float64"):
        restored = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored.shared_baseline[channel].data, prepared.shared_baseline[channel].data.compute())


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_numpy_integer_state_labels_preserve_science_and_saved_provenance(inputs, tmp_path, extension):
    import json
    from openghg_inversions.correlated_state import CorrelatedLognormalPrior
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs

    channels, prepared = inputs
    original_prior = prepared.retained_prior
    labels = np.arange(10, 10 + original_prior.mean.size, dtype=np.int64)
    mean = original_prior.mean.assign_coords(state=labels)
    prior = CorrelatedLognormalPrior(mean, original_prior.arithmetic_covariance.data)
    by_source = dict(zip(_example.STATE_LABELS, prior.mean.state.values, strict=True))
    mapping = {channel: {source: by_source[source] for source in sources}
               for channel, sources in _example.SOURCE_STATE.items()}
    assert all(isinstance(label, np.integer) for values in mapping.values() for label in values.values())
    caller_provenance = {"caller": {"label": "unchanged"}}
    result = assemble(channels, prepared, retained_prior=prior, source_state=mapping,
                      provenance=caller_provenance)
    np.testing.assert_array_equal(result.retained_prior.mean.state, labels)
    for channel in ("co2", "o2"):
        response = getattr(result, f"{channel}_sensitivity")
        np.testing.assert_array_equal(response.state, labels)
        np.testing.assert_allclose(response.compute(), getattr(prepared, f"{channel}_sensitivity").compute())
    expected_mapping = {channel: {source: int(label) for source, label in values.items()}
                        for channel, values in mapping.items()}
    assert result.provenance["surface_template_recipe"]["source_state"] == expected_mapping
    assert result.provenance["caller"] == caller_provenance["caller"]
    assert json.loads(json.dumps(result.provenance)) == result.provenance
    path = tmp_path / f"integer-states.{extension}"
    result.save(path)
    restored = Co2O2PreparedInputs.load(path)
    xr.testing.assert_identical(restored.retained_prior.mean, result.retained_prior.mean)
    assert restored.provenance == result.provenance
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(getattr(restored, f"{channel}_sensitivity"),
                                   getattr(result, f"{channel}_sensitivity").compute())
    assert all(isinstance(label, np.integer) for values in mapping.values() for label in values.values())
    assert caller_provenance == {"caller": {"label": "unchanged"}}


def test_mixed_compatible_boundary_units_project_to_common_ppm(inputs):
    channels, prepared = inputs
    original = channels[0].site_data["SYNTH"]
    original_units = original.bc_reference.attrs["units"]
    changed = original.copy(deep=False)
    expected = []
    for index, (direction, units, factor) in enumerate(zip(
        "nesw", ("ppb", "ppm", "mol/mol", "mmol/mol"), (1000., 1., 1e-6, .001), strict=True,
    )):
        name = f"bc_anomaly_{direction}"
        physical = xr.ones_like(original[name]) * (.01 * (index + 1))
        expected.append(physical.sum([dim for dim in physical.dims if dim != "time"], skipna=False))
        changed[name] = (physical * factor).assign_attrs(units=units)
    changed["bc_reference"] = (original.bc_reference * 1000).assign_attrs(units="ppb")
    converted = replace(channels[0], site_data={"SYNTH": changed})
    result = assemble((converted, channels[1]), prepared)
    boundary = result.boundary_operands["co2"]
    expected = xr.concat(expected, dim=xr.IndexVariable("bc_region", list("nesw")))
    np.testing.assert_allclose(boundary.H_bc_anomaly.compute(), expected.compute())
    assert boundary.H_bc_anomaly.attrs["units"] == "ppm"
    assert boundary.H_bc_mass.attrs["units"] == "1"
    assert boundary.bc_reference.item() == 400.
    assert original.bc_reference.item() == 400.
    assert original.bc_reference.attrs["units"] == original_units


def test_native_projection_borrows_lazy_scientific_payloads(inputs):
    from dask.callbacks import Callback
    from openghg_inversions.rhime.co2.surface_templates import _channel_templates

    channels, prepared = inputs
    original = channels[0].site_data["SYNTH"]
    payloads = {name: original[name].data for name in original.data_vars}
    with Callback(pretask=lambda *a: pytest.fail("template projection computed a borrowed payload")):
        observations, errors, sensitivity, boundary, _ = _channel_templates(
            channels[0], "co2", _example.SOURCE_STATE["co2"], prepared.retained_prior,
            {"SYNTH": original.bc_reference},
        )
    for value in (observations, errors, sensitivity, boundary.H_bc_anomaly, boundary.H_bc_mass):
        assert isinstance(value.data, da.Array)
    assert all(original[name].data is payload for name, payload in payloads.items())


def test_scalar_reference_boundary_shares_work_without_transport(inputs):
    from dask import delayed
    from dask.callbacks import Callback
    from openghg_inversions.inversion_data import SiteOptions
    from openghg_inversions.rhime.co2.surface_templates import (
        _channel_templates, _materialize_surface_references,
    )

    channels, prepared = inputs
    calls = []

    @delayed
    def references():
        calls.append("references")
        return np.array([400., 209000.])

    shared = da.from_delayed(references(), shape=(2,), dtype=float)
    copied = {}
    for index, (name, channel) in enumerate(zip(("co2", "o2"), channels, strict=True)):
        site = channel.site_data["SYNTH"].assign(
            bc_reference=xr.DataArray(shared[index], attrs={"units": "ppm"}),
        )
        second = site.assign_attrs({**site.attrs, "site": "OTHER"})
        copied[name] = replace(channel, site_data={"SYNTH": site, "OTHER": second},
                               site_options=SiteOptions.from_inputs(
                                   sites=["OTHER", "SYNTH"], inlet="10m", fp_height="10m",
                                   averaging_period="1h", platform="surface"))
    resolved = _materialize_surface_references(copied)
    assert calls == ["references"]
    with Callback(pretask=lambda *a: pytest.fail("channel assembly computed transport or metadata again")):
        for name, channel in copied.items():
            _, _, sensitivity, boundary, record = _channel_templates(
                channel, name, _example.SOURCE_STATE[name], prepared.retained_prior, resolved[name],
            )
            assert isinstance(sensitivity.data, da.Array)
            assert isinstance(boundary.H_bc_anomaly.data, da.Array)
            assert list(record["sites"]) == ["OTHER", "SYNTH"]
            assert isinstance(channel.site_data["SYNTH"].bc_reference.data, da.Array)
    assert calls == ["references"]
    invalid = copied["o2"].site_data["SYNTH"].assign(
        bc_reference=xr.DataArray(shared, dims="invalid", attrs={"units": "ppm"}),
    )
    copied["o2"] = replace(copied["o2"], site_data={**copied["o2"].site_data, "SYNTH": invalid})
    with Callback(pretask=lambda *a: pytest.fail("scalar preflight executed before rejecting shape")):
        with pytest.raises(ValueError, match="scalar bc_reference"):
            _materialize_surface_references(copied)


@pytest.mark.parametrize("units,factor", [("ppm", 1.), ("ppb", 1000.), ("mol/mol", 1e-6)])
def test_all_template_concentrations_convert_to_ppm(inputs, units, factor):
    channels, prepared = inputs
    converted = []
    for channel in channels:
        site = channel.site_data["SYNTH"].copy(deep=False)
        for name in ("mf", "mf_error", "fp_x_flux_sectoral", "bc_reference",
                     *(f"bc_anomaly_{direction}" for direction in "nesw")):
            site[name] = (site[name] * factor).assign_attrs(units=units)
        converted.append(replace(channel, site_data={"SYNTH": site}))
    result = assemble(converted, prepared)
    for name in ("observations", "independent_error_sd", "co2_sensitivity", "o2_sensitivity"):
        np.testing.assert_allclose(getattr(result, name).compute(), getattr(prepared, name).compute())
    for channel, reference in (("co2", 400.), ("o2", 209000.)):
        assert result.boundary_operands[channel].bc_reference.item() == pytest.approx(reference)


def test_native_site_and_time_order_are_preserved(inputs):
    from openghg_inversions.inversion_data import SiteOptions
    from openghg_inversions.rhime.co2.surface_templates import _channel_templates

    channels, prepared = inputs
    channel = channels[0]
    first = channel.site_data["SYNTH"].isel(time=[4, 1, 3])
    second = channel.site_data["SYNTH"].isel(time=[5, 0]).assign_attrs(
        {**first.attrs, "site": "OTHER", "inlet": "20m"},
    )
    channel = replace(channel, site_data={"SYNTH": first, "OTHER": second},
                      site_options=SiteOptions.from_inputs(
                          sites=["OTHER", "SYNTH"], inlet=["20m", "10m"], fp_height="10m",
                          averaging_period="1h", platform="surface"))
    observations, _, h, _, record = _channel_templates(
        channel, "co2", _example.SOURCE_STATE["co2"], prepared.retained_prior,
        {"SYNTH": first.bc_reference, "OTHER": second.bc_reference},
    )
    np.testing.assert_array_equal(observations.co2_observation, np.arange(5))
    np.testing.assert_array_equal(observations.site, ["OTHER", "OTHER", "SYNTH", "SYNTH", "SYNTH"])
    np.testing.assert_array_equal(observations.time, np.concatenate([second.time.data, first.time.data]))
    np.testing.assert_allclose(h.compute(), prepared.co2_sensitivity.isel(co2_observation=[5, 0, 4, 1, 3]).compute())
    assert record["sites"]["OTHER"]["inlet"] == "20m"


@pytest.mark.parametrize("failure", ["duplicate", "outside_window", "domain", "window"])
def test_native_time_and_acquisition_contract(inputs, failure):
    channels, prepared = inputs
    co2, o2 = channels
    site = co2.site_data["SYNTH"]
    if failure == "duplicate":
        site = site.isel(time=[0, 0, 2, 3, 4, 5])
        match = "unique native"
    elif failure == "outside_window":
        co2 = replace(co2, acquisition=replace(co2.acquisition, end_date="2020-01-01T03:00:00"))
        match = "outside the declared"
    else:
        o2 = replace(o2, acquisition=replace(o2.acquisition, **(
            {"domain": "OTHER"} if failure == "domain" else {"end_date": "2020-01-03"})))
        match = "same declared acquisition"
    co2 = replace(co2, site_data={"SYNTH": site})
    with pytest.raises(ValueError, match=match):
        assemble((co2, o2), prepared)


def test_joint_dense_covariance_keeps_cross_channel_terms(inputs):
    channels, prepared = inputs
    values = np.eye(10) * .02 + np.ones((10, 10)) * .001
    covariance = prepared.aggregation_error.covariance.copy(data=values)
    result = assemble(channels, prepared, aggregation_error=replace(prepared.aggregation_error, covariance=covariance))
    xr.testing.assert_identical(result.aggregation_error.covariance, covariance)


def test_covariance_column_level_names_are_required(inputs):
    channels, prepared = inputs
    covariance = prepared.aggregation_error.covariance.rename(species_cov="wrong_species")
    with pytest.raises(ValueError, match="exact gathered"):
        assemble(channels, prepared, aggregation_error=replace(prepared.aggregation_error, covariance=covariance))


@pytest.mark.parametrize("source,co2_response,ratio", [
    ("gpp", [-.2, -.6, -1., -.9, -.4, -.1], -1.1),
    ("ter", [.8, .7, .5, .4, .6, .9], -1.2),
    ("ff", [.2, .7, .3, .8, .4, .6], -1.4),
])
def test_source_resolved_o2_sign_and_multiplier_once(inputs, source, co2_response, ratio):
    _, prepared = inputs
    np.testing.assert_allclose(prepared.co2_sensitivity.sel(state=source).compute(), co2_response)
    expected = np.array(co2_response)[[1, 2, 4, 5]] * ratio
    np.testing.assert_allclose(prepared.o2_sensitivity.sel(state=source).compute(), expected)
    assert (expected > 0).all() if source == "gpp" else (expected < 0).all()


def test_lazy_covariance_axis_units_at_explicit_metadata_boundary(inputs):
    channels, prepared = inputs
    covariance = prepared.aggregation_error.covariance.assign_coords(
        observation_units=("observation", da.from_array(np.full(10, "ppm"), chunks=5)),
        observation_units_cov=("observation_cov", da.from_array(np.full(10, "ppm"), chunks=5)),
    )
    result = assemble(channels, prepared, aggregation_error=replace(prepared.aggregation_error, covariance=covariance))
    np.testing.assert_array_equal(result.aggregation_error.covariance, 0.)
    assert isinstance(covariance.observation_units.data, da.Array)
    assert isinstance(covariance.observation_units_cov.data, da.Array)


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_nonunit_native_templates_bind_snapshot_and_replay_independent_physical_sum(inputs, tmp_path, monkeypatch, extension):
    import pymc as pm
    import pytensor
    from openghg_inversions.models.shared_baseline import SharedBaselineChoice
    from openghg_inversions.rhime.co2 import prepare_linked_shared_inputs, reconstruct_linked_shared_baseline
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs

    channels, original = inputs
    altered, masses = [], {}
    for channel, name in zip(channels, ("co2", "o2"), strict=True):
        site = channel.site_data["SYNTH"]
        mass = np.resize([.5, 2., 1.5, .75, 1.25, 1.], site.sizes["time"])
        masses[name] = mass
        scale = xr.DataArray(mass, dims="time", coords={"time": site.time})
        fields = {}
        for i, direction in enumerate("nesw"):
            fields[f"bc_mass_{direction}"] = (site[f"bc_mass_{direction}"] * scale).assign_attrs(units="1")
            fields[f"bc_anomaly_{direction}"] = (xr.ones_like(site[f"bc_anomaly_{direction}"]) * scale * (.01 * (i + 1))).assign_attrs(units="ppm")
        altered.append(replace(channel, site_data={"SYNTH": site.assign(fields)}))
    linked = assemble(altered, original, provenance={"overlay": "independent nonunit-mass control"})
    mean = linked.retained_prior.mean
    outer = mean.copy(data=[True, True, False, True, True])
    sampled = mean.copy(data=[True, True, False, False, False])
    with pytensor.config.change_flags(floatX="float64"):
        prepared = prepare_linked_shared_inputs(
            linked, choices={name: SharedBaselineChoice(None, 1., "multiplicative") for name in ("co2", "o2")},
            outer=outer, sampled_outer=sampled,
        )
        path = tmp_path / f"nonunit.{extension}"
        prepared.save(path)
        monkeypatch.setattr(SharedBaselineChoice, "bind", lambda *a, **kw: pytest.fail("snapshot rebound support"))
        restored = Co2O2PreparedInputs.load(path)
    monkeypatch.setattr(pm.Model, "__init__", lambda *a, **kw: pytest.fail("replay constructed model"))
    monkeypatch.setattr(pm, "sample", lambda *a, **kw: pytest.fail("replay sampled"))
    coefficients = xr.Dataset({"flux_scaling": restored.shared_baseline["co2"].data.retained_mean,
                               "co2_bc_mean_shift": xr.DataArray(0.), "o2_bc_mean_shift": xr.DataArray(0.)})
    replay = reconstruct_linked_shared_baseline({name: bound.data for name, bound in restored.shared_baseline.items()}, coefficients)
    for name, reference in (("co2", 400.), ("o2", 209000.)):
        raw = restored.boundary_operands[name]
        mass = masses[name]
        np.testing.assert_allclose(raw.H_bc_mass, np.vstack([.25 * mass] * 4))
        np.testing.assert_allclose(raw.H_bc_anomaly, np.vstack([.01 * (i + 1) * mass for i in range(4)]))
        assert raw.bc_reference.item() == reference
        scientific_raw = raw.copy(deep=False)
        scientific_raw.attrs = {key: value for key, value in raw.attrs.items()
                                if key not in {"array_names_json", "multiindex_dims_json"}}
        xr.testing.assert_identical(scientific_raw, linked.boundary_operands[name].compute())
        bound = restored.shared_baseline[name]
        np.testing.assert_allclose(bound.data.bc_reference_shift, reference * mass)
        np.testing.assert_allclose(bound.data.physical_bc_prior, (.1 + reference) * mass)
        sensitivity = getattr(linked, f"{name}_sensitivity").compute()
        expected = sensitivity.data @ linked.retained_prior.mean.data + (.1 + reference) * mass
        np.testing.assert_array_equal(linked.fixed_prior_contribution, 0.)
        np.testing.assert_allclose(replay[name].modelled_concentration, expected, rtol=0, atol=1e-10)
        row = bound.data.attrs["native_observation_dim"]
        subset = replace(bound, data=bound.data.isel({row: [1]}))
        xr.testing.assert_identical(subset.data.outer_column_mean, bound.data.outer_column_mean)
        xr.testing.assert_identical(subset.data.bc_pattern_mean, bound.data.bc_pattern_mean)
        subset_replay = reconstruct_linked_shared_baseline({name: subset.data}, coefficients)[name]
        np.testing.assert_allclose(subset_replay.modelled_concentration, expected[[1]], rtol=0, atol=1e-10)
        xr.testing.assert_identical(restored.shared_baseline[name].data, prepared.shared_baseline[name].data.compute())
    assert restored.provenance == linked.provenance
    assert restored.provenance["overlay"] == "independent nonunit-mass control"
