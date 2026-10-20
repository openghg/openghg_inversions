"""Role-driven linked products and graph-free export."""

from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs
from openghg_inversions.rhime.co2.co2_o2_outputs import (
    Co2O2OutputOptions,
    calculate_co2_o2_products,
    make_co2_o2_output,
    save_co2_o2_products,
    validate_co2_o2_product_request,
    validate_co2_o2_trace,
)
from openghg_inversions.serialization import load_trace
from test_linked_paris_outputs import _joint_fixture


def _result(prepared, trace):
    return make_co2_o2_output(
        prepared=prepared, trace=trace, start_date="2024-01-01",
        end_date="2024-02-01", domain="EUROPE", output_metadata={"sampler": {"draws": 3}},
    )


def test_posterior_only_basic_preserves_channels_states_and_exact_predictions(tmp_path):
    prepared, trace = _joint_fixture()
    del trace["prior"]
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles["concentration"] = "actual_replicate"
    trace.attrs["rhime_variable_roles"] = json.dumps(roles)
    prediction = trace["posterior"].prediction.isel(chain=[0], draw=[0]).to_dataset(name="actual_replicate")
    trace["posterior_predictive"] = prediction
    before = trace.copy(deep=True)
    result = _result(prepared, trace)
    output = Co2O2OutputOptions(output_path=tmp_path, save_trace=True, save_concentrations=True)
    calculate_co2_o2_products(result=result, output=output)
    assert result.idata is trace
    assert result.prepared is prepared
    assert result.sampler is None
    for species, size in (("co2", 2), ("o2", 3)):
        values = result.outputs[f"{species}_concentration_components"]
        assert values.sizes["observation"] == size
        assert values.sizes["draw"] == 3
        assert values.sizes["posterior_predictive_draw"] == 1
        assert "modelled_prior" not in values
        np.testing.assert_allclose(values.observed - values.modelled_posterior, values.residual_posterior)
        np.testing.assert_allclose(
            values.pollution_posterior + values.boundary_posterior + values.offset_posterior,
            values.modelled_posterior.transpose(*values.pollution_posterior.dims),
        )
        assert "predictive_posterior_mean" in result.outputs[f"{species}_basic"]
        assert values.attrs["linked_ratio_provenance"] == prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"]
    xr.testing.assert_identical(result.outputs["state_scaling"].flux_scale_posterior, trace["posterior"].scales.rename("flux_scale_posterior"))
    save_co2_o2_products(result, output=output)
    restored = load_trace(result.output_metadata["trace_path"])
    xr.testing.assert_identical(restored["posterior"], before["posterior"])
    for group in before.children:
        xr.testing.assert_identical(trace[group], before[group])
    assert result.output_metadata["sampler"] == {"draws": 3}


def test_basic_and_paris_preserve_independent_prior_axes_and_prior_requirement():
    prepared, trace = _joint_fixture()
    result = _result(prepared, trace)
    calculate_co2_o2_products(result=result, output=Co2O2OutputOptions(output_format="paris"))
    assert result.outputs["co2_concentration_components"].sizes["prior_draw"] == 4
    assert result.outputs["co2_concentration_components"].sizes["draw"] == 3
    assert result.outputs["co2_paris_concentration"].sizes["index"] == 2
    del trace["prior"]
    with pytest.raises(ValueError, match="flux_scale.*prior"):
        calculate_co2_o2_products(result=result, output=Co2O2OutputOptions(output_format="paris"))


def test_role_validation_rejects_reordered_states_and_missing_declared_terms():
    prepared, trace = _joint_fixture()
    validate_co2_o2_trace(trace, prepared)
    bad = trace.copy()
    bad["posterior"] = bad["posterior"].to_dataset().isel(state=slice(None, None, -1))
    with pytest.raises(ValueError, match="align|index|indexes"):
        validate_co2_o2_trace(bad, prepared)
    del trace["posterior"]["bc"]
    with pytest.raises(ValueError, match="boundary_concentration.*posterior"):
        validate_co2_o2_trace(trace, prepared)


def test_result_boundary_rejects_static_mean_and_incompatible_predictive_units():
    prepared, trace = _joint_fixture(baseline=False)
    bad = trace.copy()
    bad["posterior"]["prediction"] = bad["posterior"].prediction.mean(("chain", "draw"))
    with pytest.raises(ValueError, match="modelled_concentration.*chain/draw"):
        _result(prepared, bad)
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles["concentration"] = "prediction"
    trace.attrs["rhime_variable_roles"] = json.dumps(roles)
    trace["posterior_predictive"] = trace["posterior"].to_dataset()[["prediction"]].assign_coords(
        observation_units=("observation", ["ppb"] * 5),
    )
    with pytest.raises(ValueError, match="posterior_predictive.*observation_units"):
        _result(prepared, trace)


def test_output_requests_reject_unsupported_units_and_options_before_writing(tmp_path):
    prepared, _ = _joint_fixture(o2_units="per meg")
    destination = tmp_path / "unopened"
    output = Co2O2OutputOptions(output_path=destination)
    with pytest.raises(ValueError, match="OPE-86"):
        validate_co2_o2_product_request(output, prepared)
    assert not destination.exists()
    with pytest.raises(ValueError, match="Unsupported linked output"):
        Co2O2OutputOptions.from_params({"output_format": "inv_out", "invented": 1})
    with pytest.raises(ValueError, match="native_flux_bases"):
        Co2O2OutputOptions(output_format="paris", native_flux_bases={"ch4": "ignored.nc"})


def test_native_reporting_loads_explicit_saved_basis(tmp_path):
    from openghg_inversions.basis.basis_functions import BasisFunctions

    prepared, _ = _joint_fixture(baseline=False, numeric_state=True)
    flat = xr.DataArray(np.arange(1, 6)[None, :], dims=("lat", "lon"),
                        coords={"lat": [50.0], "lon": np.arange(5.0)})
    flux = xr.where(flat == 5, 0.0, 1e-6).assign_attrs(units="mol / m^2 / s")
    basis = BasisFunctions.from_flat_basis(flat, flux)
    path = tmp_path / "co2-basis.nc"
    basis.save(path)
    output = Co2O2OutputOptions(output_format="paris", native_flux_bases={"co2": path})
    loaded = validate_co2_o2_product_request(output, prepared)
    xr.testing.assert_identical(loaded["co2"].flux, basis.flux.rename("flux"))


@pytest.mark.parametrize("output_format", ["basic", "none"])
@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_trace_export_cannot_overwrite_referenced_artifact(tmp_path, output_format, alias):
    prepared, trace = _joint_fixture()
    protected = tmp_path / "posterior.nc"
    protected.write_text("saved posterior")
    destination = tmp_path / "products"
    result = _result(prepared, trace)
    target = protected
    if alias != "same":
        target = tmp_path / "alias.nc"
        if alias == "symlink":
            target.symlink_to(protected)
        else:
            target.hardlink_to(protected)
    output = Co2O2OutputOptions(output_format=output_format, output_path=destination, save_trace=target)
    calculate_co2_o2_products(result=result, output=output)
    with pytest.raises(ValueError, match="overwrite"):
        save_co2_o2_products(result, output=output, referenced=(protected,))
    assert protected.read_text() == "saved posterior"
    assert not destination.exists()


@pytest.mark.parametrize("layout", ["checkpoints", "products", "protected_store"])
def test_zarr_trace_rejects_containment_overlaps_before_writing(tmp_path, layout):
    prepared, trace = _joint_fixture()
    store = tmp_path / "run.zarr"
    checkpoint_root = store if layout == "checkpoints" else tmp_path / "checkpoints"
    prep = handoffs.save_preparation(prepared, output_dir=checkpoint_root / "prepare",
                                    start_date="2024-01-01", end_date="2024-02-01", domain="EUROPE")
    sample = handoffs.save_samples(
        prep, trace, actual_inference={}, preparation_generation=handoffs.load_preparation(prep).generation,
        output_dir=checkpoint_root / "sample",
    )
    before = {path: path.read_bytes() for path in checkpoint_root.rglob("*") if path.is_file()}
    destination = store if layout == "products" else store / "products"
    target = checkpoint_root / "trace.zarr" if layout == "protected_store" else store
    output = Co2O2OutputOptions(output_path=destination, save_trace=target)
    with pytest.raises(ValueError, match="overlap|overwrite"):
        if layout == "protected_store":
            result = _result(prepared, trace)
            calculate_co2_o2_products(result=result, output=output)
            save_co2_o2_products(result, output=output, referenced=(checkpoint_root,))
        else:
            handoffs.postprocess(sample, output=output)
    assert {path: path.read_bytes() for path in checkpoint_root.rglob("*") if path.is_file()} == before
    assert not destination.exists()


@pytest.mark.parametrize("trace_setting", ["netcdf", "zarr", "default"])
def test_none_still_exports_requested_trace(tmp_path, trace_setting):
    prepared, trace = _joint_fixture()
    result = _result(prepared, trace)
    target = tmp_path / ("trace.zarr" if trace_setting == "zarr" else "trace.nc")
    destination = tmp_path / "products"
    output = Co2O2OutputOptions(output_format="none", output_path=destination if trace_setting == "default" else None,
                               save_trace=True if trace_setting == "default" else target)
    calculate_co2_o2_products(result=result, output=output)
    save_co2_o2_products(result, output=output)
    assert result.outputs == {}
    restored = load_trace(result.output_metadata["trace_path"])
    xr.testing.assert_identical(restored, trace)
    if trace_setting == "default":
        assert list(destination.iterdir()) == [destination / "rhime2024-01-01_trace.nc"]
    else:
        assert not destination.exists()


def test_none_trace_default_requires_output_directory():
    with pytest.raises(ValueError, match="output_path"):
        Co2O2OutputOptions(output_format="none", save_trace=True)


def test_basic_summaries_share_sensitivities_within_independent_sample_groups():
    import dask.array as da
    from dask import delayed

    prepared, trace = _joint_fixture()
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles["concentration"] = "prediction"
    trace.attrs["rhime_variable_roles"] = json.dumps(roles)
    for group, draws in (("prior", 2), ("posterior", 1)):
        trace[f"{group}_predictive"] = trace.posterior.to_dataset()[["prediction"]].isel(chain=[0], draw=slice(draws))
    expected = _result(prepared, trace)
    calculate_co2_o2_products(result=expected, output=Co2O2OutputOptions())
    executions = {"co2": 0, "o2": 0}

    @delayed
    def load_sensitivity(species, values):
        executions[species] += 1
        return values

    lazy = {}
    for species in executions:
        sensitivity = getattr(prepared, f"{species}_sensitivity")
        data = da.from_delayed(load_sensitivity(species, sensitivity.to_numpy()),
                               shape=sensitivity.shape, dtype=sensitivity.dtype)
        lazy[f"{species}_sensitivity"] = sensitivity.copy(data=data)
    borrowed = replace(prepared, **lazy)
    result = _result(borrowed, trace)
    assert executions == {"co2": 0, "o2": 0}
    calculate_co2_o2_products(result=result, output=Co2O2OutputOptions())
    # One computation per channel and prior/posterior sample group.
    assert executions == {"co2": 2, "o2": 2}
    for species in executions:
        xr.testing.assert_identical(result.outputs[f"{species}_basic"], expected.outputs[f"{species}_basic"])
        assert isinstance(getattr(borrowed, f"{species}_sensitivity").data, da.Array)


def test_none_suppresses_products_and_writing(tmp_path):
    prepared, trace = _joint_fixture()
    result = _result(prepared, trace)
    output = Co2O2OutputOptions(output_format="none", output_path=tmp_path / "unused")
    calculate_co2_o2_products(result=result, output=output)
    save_co2_o2_products(result, output=output)
    assert result.outputs == {}
    assert not (tmp_path / "unused").exists()


def test_importing_linked_paris_before_public_output_types_succeeds():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "import openghg_inversions.postprocessing.linked_paris_outputs; "
         "from openghg_inversions.rhime.co2 import Co2O2Output, Co2O2OutputOptions"],
        check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize("resource", ["basis", "country"])
def test_trace_export_protects_reporting_input_paths(tmp_path, resource):
    prepared, trace = _joint_fixture()
    basis_path, country_path = tmp_path / "basis.nc", tmp_path / "countries.nc"
    protected = basis_path if resource == "basis" else country_path
    protected.write_text("reporting input")
    destination = tmp_path / "products"
    output = Co2O2OutputOptions(
        output_format="paris", output_path=destination, save_trace=protected,
        native_flux_bases={"co2": basis_path}, country_file=country_path,
    )
    with pytest.raises(ValueError, match="overwrite"):
        save_co2_o2_products(_result(prepared, trace), output=output)
    assert protected.read_text() == "reporting input"
    assert not destination.exists()


def test_export_rejects_path_components_in_requested_domain_before_writing(tmp_path):
    prepared, trace = _joint_fixture()
    result = _result(prepared, trace)
    result.domain = "../redirected"
    output = Co2O2OutputOptions(output_path=tmp_path / "unopened")
    calculate_co2_o2_products(result=result, output=output)
    with pytest.raises(ValueError, match="domain.*filename component"):
        save_co2_o2_products(result, output=output)
    assert not (tmp_path / "unopened").exists()
