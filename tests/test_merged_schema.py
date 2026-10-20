"""The acquired-data boundary checks metadata without executing payloads."""

from dataclasses import replace

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import MergedData, RhimePreparedInputs, SiteOptions
from openghg_inversions.rhime import preparation
from openghg_inversions.rhime.multisector import prepare_multisector
from openghg_inversions.rhime.standard import prepare_standard


def _merged(*, sectoral=False, timed_flux=False):
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    grid = {"lat": [50., 51.], "lon": [-2., -1.]}
    site = xr.Dataset({
        "mf": ("time", [1., 2.], {"units": "ppm"}),
        "mf_error": ("time", [.1, .2]),
        "fp_x_flux": (("lat", "lon", "time"), np.ones((2, 2, 2))),
        "custom_diagnostic": ("time", [4., 5.], {"meaning": "extension"}),
    }, coords={**grid, "time": time}, attrs={"custom_attr": "retained"}).chunk()
    flux = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords=grid, attrs={"units": "mol m-2 s-1"})
    sources = {"inventory": flux.to_dataset(name="flux").chunk()}
    if sectoral:
        site["fp_x_flux_sectoral"] = xr.concat(
            [site.fp_x_flux, 2 * site.fp_x_flux], dim=xr.IndexVariable("source", ["inventory", "other"]),
        )
        sources["other"] = flux.to_dataset(name="flux").chunk()
    if timed_flux:
        sources = {name: ds.expand_dims(time=pd.date_range("2019-01-01", periods=1)) for name, ds in sources.items()}
    return MergedData(
        site_data={"TAC": site}, flux_data=sources,
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"), split_by_sectors=sectoral,
    )


def _validate(merged, *, require_footprints=False, use_bc=False, grid_dims=("lat", "lon")):
    merged.validate_scientific_inputs(
        split_by_sectors=merged.split_by_sectors, require_footprints=require_footprints,
        use_bc=use_bc, flux_sources=tuple(merged.flux_data), grid_dims=grid_dims,
    )


@pytest.mark.parametrize("sectoral", [False, True])
@pytest.mark.parametrize("timed_flux", [False, True])
def test_metadata_validation_preserves_arrays_graphs_chunks_and_extensions(sectoral, timed_flux):
    merged = _merged(sectoral=sectoral, timed_flux=timed_flux)
    site = merged.site_data["TAC"]

    @delayed
    def forbidden():
        raise AssertionError("Validation executed a scientific payload.")

    site["custom_auxiliary"] = ("time", da.from_delayed(forbidden(), shape=(2,), dtype=float))
    site = site.set_coords("custom_auxiliary")
    merged.site_data["TAC"] = site
    payloads = {name: value.data for name, value in site.variables.items()}
    attrs = {name: dict(value.attrs) for name, value in site.variables.items()}
    executions = []
    with Callback(pretask=lambda *args: executions.append(args)):
        _validate(merged)
        schema = merged.site_schema(split_by_sectors=sectoral)
        schema.validate(site)
    assert not executions
    assert merged.site_data["TAC"] is site
    assert all(site[name].data is payload for name, payload in payloads.items() if name not in site.indexes)
    assert {name: value.attrs for name, value in site.variables.items()} == attrs
    assert site.custom_diagnostic.attrs["meaning"] == "extension"
    assert "units" not in site.mf_error.attrs
    assert site.fp_x_flux.chunks == ((2,), (2,), (2,))


@pytest.mark.parametrize("components", [("mf_error",), ("mf_repeatability",), ("mf_variability",),
                                         ("mf_repeatability", "mf_variability")])
def test_uncertainty_alternatives_are_valid_before_error_derivation(components):
    merged = _merged()
    site = merged.site_data["TAC"].drop_vars("mf_error")
    for name in components:
        site[name] = xr.full_like(site.mf, .1)
    merged.site_data["TAC"] = site
    _validate(merged)


def test_missing_uncertainty_reports_site_before_any_payload_execution():
    merged = _merged()
    merged.site_data["TAC"] = merged.site_data["TAC"].drop_vars("mf_error")
    with Callback(pretask=lambda *args: pytest.fail("executed payload")):
        with pytest.raises(ValueError, match=r"site_data\['TAC'\].*mf_error"):
            _validate(merged)


@pytest.mark.parametrize("field,values", [
    ("mf", np.array(["a", "b"])), ("mf_error", np.ones((2, 2))),
    ("fp_x_flux", np.ones((2, 2))),
])
def test_wrong_dtype_or_rank_names_the_site_and_variable(field, values):
    merged = _merged()
    site = merged.site_data["TAC"].drop_vars(field)
    dims = ("time",) if values.ndim == 1 else ("lat", "time")
    site[field] = xr.DataArray(values, dims=dims, attrs={"units": "ppm"})
    merged.site_data["TAC"] = site
    with Callback(pretask=lambda *args: pytest.fail("executed payload")):
        with pytest.raises(ValueError, match=rf"(?s)site_data\['TAC'\].*data_vars.{field}"):
            _validate(merged)


def test_permuted_dimensions_and_named_grid_are_accepted_without_transposing_input():
    merged = _merged(sectoral=True)
    site = merged.site_data["TAC"].transpose("time", "lon", "lat", "source").rename(lat="y", lon="x")
    merged.site_data["TAC"] = site
    merged.flux_data = {name: ds.rename(lat="y", lon="x") for name, ds in merged.flux_data.items()}
    _validate(merged, grid_dims=("y", "x"))
    assert site.fp_x_flux_sectoral.dims == ("time", "x", "y", "source")


def test_sites_keep_independent_times_and_empty_sites_and_missing_values():
    merged = _merged()
    site = merged.site_data["TAC"]
    missing = site.assign(mf=xr.full_like(site.mf, np.nan))
    merged = replace(
        merged, site_data={"TAC": site, "MHD": missing.assign_coords(time=site.time + pd.Timedelta(days=1)),
                           "EMPTY": site.isel(time=slice(0, 0))},
        site_options=SiteOptions.from_inputs(sites=["TAC", "MHD", "EMPTY"], averaging_period=["1h", "4h", "1h"]),
    )
    _validate(merged)


@pytest.mark.parametrize("time", [["2020-01-01", "2020-01-01"], ["2020-01-01", "NaT"]])
def test_invalid_index_labels_fail_without_scanning_payloads(time):
    merged = _merged()
    merged.site_data["TAC"] = merged.site_data["TAC"].assign_coords(time=pd.to_datetime(time))
    with pytest.raises(ValueError, match="coords.time.*unique"):
        _validate(merged)


def test_footprints_are_required_only_for_generation():
    merged = _merged()
    _validate(merged)
    with pytest.raises(ValueError, match="fp"):
        _validate(merged, require_footprints=True)
    merged.site_data["TAC"]["fp"] = merged.site_data["TAC"].fp_x_flux
    _validate(merged, require_footprints=True)


def test_directional_bc_inputs_are_conditional_and_precede_H_bc():
    merged = _merged()
    with pytest.raises(ValueError, match="bc_"):
        _validate(merged, use_bc=True)
    site = merged.site_data["TAC"].assign_coords(height=[500.])
    for direction in "nesw":
        edge = "lon" if direction in "ns" else "lat"
        site[f"bc_{direction}"] = xr.DataArray(np.ones((2, 1, 2)), dims=(edge, "height", "time"))
    merged.site_data["TAC"] = site
    assert merged.boundary_data is None
    assert "H_bc" not in site
    _validate(merged, use_bc=True)


@pytest.mark.parametrize("name", ["mf", "mf_error", "fp_x_flux"])
@pytest.mark.parametrize("units", ["ppb", "kg", "", None, 1])
def test_explicit_units_must_share_concentration_scale(name, units):
    merged = _merged()
    # mf must also agree with a different site, beyond its own inherited fields.
    if name == "mf":
        merged = replace(merged, site_data={"REFERENCE": merged.site_data["TAC"].copy(deep=False),
                                           "TAC": merged.site_data["TAC"].copy(deep=False)},
                         site_options=SiteOptions.from_inputs(sites=["REFERENCE", "TAC"], averaging_period="1h"))
    merged.site_data["TAC"][name].attrs = {"units": units}
    with pytest.raises(ValueError, match=rf"(?s)site_data\['TAC'\].*{name}.*units"):
        _validate(merged)


def test_equal_scale_pint_expressions_and_inherited_units_are_accepted():
    merged = _merged()
    merged.site_data["TAC"].mf_error.attrs = {"units": "umol/mol"}
    merged.site_data["TAC"].fp_x_flux.attrs = {"units": "1e-6"}
    _validate(merged)


def test_lazy_scalar_unit_attribute_is_rejected_without_coercion():
    merged = _merged()

    @delayed
    def forbidden():
        pytest.fail("Executed a malformed metadata attribute.")

    merged.site_data["TAC"].mf_error.attrs["units"] = da.from_delayed(forbidden(), shape=(), dtype=float)
    with Callback(pretask=lambda *args: pytest.fail("executed a payload")):
        with pytest.raises(ValueError, match="mf_error.attrs.units.*string"):
            _validate(merged)


@pytest.mark.parametrize("flag", ["split_by_sectors", "use_bc", "require_footprints"])
def test_lazy_schema_flags_are_rejected_without_truth_testing(flag):
    @delayed
    def forbidden():
        pytest.fail("Executed a malformed schema flag.")

    choices = {flag: da.from_delayed(forbidden(), shape=(), dtype=bool)}
    with Callback(pretask=lambda *args: pytest.fail("executed graph")):
        with pytest.raises(ValueError, match=f"{flag}.*bool"):
            MergedData.site_schema(**choices)


def test_lazy_stored_layout_is_rejected_at_both_owning_boundaries():
    @delayed
    def forbidden():
        pytest.fail("Executed a malformed stored layout flag.")

    merged = replace(_merged(), split_by_sectors=da.from_delayed(forbidden(), shape=(), dtype=bool))
    with Callback(pretask=lambda *args: pytest.fail("executed graph")):
        with pytest.raises(ValueError, match="split_by_sectors.*bool"):
            _validate(merged)
        with pytest.raises(ValueError, match="split_by_sectors.*bool"):
            merged.validate_for_preparation(species="ch4", domain="EUROPE", start_date="2020-01-01",
                                            end_date="2020-01-02", split_by_sectors=False)


def test_native_flux_units_have_the_canonical_numeric_scale():
    merged = _merged()
    merged.flux_data["inventory"].flux.attrs["units"] = "mmol m-2 s-1"
    with pytest.raises(ValueError, match=r"flux_data\['inventory'\].*flux.attrs.units.*scale"):
        _validate(merged)


def test_sector_sources_must_be_complete_but_may_arrive_in_different_order():
    merged = _merged(sectoral=True)
    merged.site_data["TAC"] = merged.site_data["TAC"].sel(source=["other", "inventory"])
    _validate(merged)
    merged.site_data["TAC"] = merged.site_data["TAC"].sel(source=["inventory"])
    with pytest.raises(ValueError, match="coords.source.*exactly"):
        _validate(merged)


@pytest.mark.parametrize("sectoral", [False, True])
@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_public_preparation_and_disk_roundtrip_keep_values_and_borrowed_inputs(tmp_path, sectoral, suffix):
    merged = _merged(sectoral=sectoral)
    site = merged.site_data["TAC"]
    borrowed = site.copy(deep=True)
    labels = xr.zeros_like(merged.flux_data["inventory"].flux, dtype=int)
    prepare = prepare_multisector if sectoral else prepare_standard
    prepared = prepare(
        merged, domain="EUROPE", start_date="2020-01-01", flux_sources=tuple(merged.flux_data),
        averaging_error=True, filters=None, basis_algorithm=None, nbasis=1, fix_basis_outer_regions=False,
        use_bc=False, bc_freq=None, min_error=0., min_error_options={"by_site": False},
        basis_inputs={"basis_partition": labels},
    )
    assert "fp" not in site
    assert "H" not in site
    assert "units" not in site.mf_error.attrs
    xr.testing.assert_identical(site, borrowed)
    np.testing.assert_array_equal(prepared.inv_inputs.mf_error, [.1, .2])
    np.testing.assert_array_equal(prepared.inv_inputs.custom_diagnostic, [4., 5.])
    if sectoral:
        np.testing.assert_array_equal(np.sort(np.asarray(prepared.inv_inputs.H).reshape(-1)), [4., 4., 8., 8.])
    else:
        np.testing.assert_array_equal(prepared.inv_inputs.H, [[4., 4.]])
    path = tmp_path / f"prepared{suffix}"
    prepared.save(path)
    reopened = RhimePreparedInputs.load(path)
    xr.testing.assert_identical(reopened.inv_inputs, prepared.inv_inputs)


@pytest.mark.parametrize("prepare", [prepare_standard, prepare_multisector])
def test_public_prepare_rejects_bad_inputs_before_error_derivation(prepare, monkeypatch):
    merged = _merged(sectoral=prepare is prepare_multisector)
    response = "fp_x_flux_sectoral" if merged.split_by_sectors else "fp_x_flux"
    merged.site_data["TAC"] = merged.site_data["TAC"].drop_vars(response)
    module = __import__(prepare.__module__, fromlist=["prepare_observation_errors"])
    monkeypatch.setattr(module, "prepare_observation_errors", lambda *a, **k: pytest.fail("derived errors"))
    with pytest.raises(ValueError, match=response):
        prepare(merged, domain="EUROPE", start_date="2020-01-01", flux_sources=tuple(merged.flux_data),
                averaging_error=True, filters=None, basis_algorithm=None, nbasis=1, fix_basis_outer_regions=False,
                use_bc=False, bc_freq=None, min_error=0., min_error_options={}, basis_inputs={"basis_path": "unused"})


def test_filter_valueerror_is_propagated_once_without_computing_all_sites(monkeypatch):
    merged = _merged()
    calls = []
    error = ValueError("Custom filter rejected its scientific configuration.")

    def reject(*args, **kwargs):
        calls.append("called")
        raise error

    monkeypatch.setattr(preparation, "filtering", reject)
    with Callback(pretask=lambda *args: pytest.fail("Computed borrowed sites after a filter error.")):
        with pytest.raises(ValueError) as caught:
            preparation.filter_observations(merged.site_data, merged.site_options, filters="custom")
    assert caught.value is error
    assert calls == ["called"]


def test_valid_daytime_filter_keeps_scientific_payloads_lazy_and_input_unchanged():
    merged = _merged()
    site = merged.site_data["TAC"].assign_coords(time=pd.to_datetime(["2020-01-01T10:00", "2020-01-01T12:00"]))
    site["release_lon"] = ("time", [0., 0.])
    merged.site_data["TAC"] = site
    original_time = site.time.copy()
    with Callback(pretask=lambda *args: pytest.fail("A time-index filter computed scientific payloads.")):
        filtered, selectors = preparation.filter_observations(merged.site_data, merged.site_options, filters="daytime")
    assert selectors.sites == ("TAC",)
    assert filtered["TAC"].sizes["time"] == 1
    assert isinstance(filtered["TAC"].mf.data, da.Array)
    xr.testing.assert_identical(site.time, original_time)
    np.testing.assert_array_equal(filtered["TAC"].mf.compute(), [2.])
