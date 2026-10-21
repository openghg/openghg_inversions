"""Scientific support and alignment contracts at the nested prepare boundary."""

from dataclasses import replace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.rhime.nested import (
    _NestedDomainData,
    _align_domain_sites,
    _inner_measurement_positions,
    align_inner_merged_to_outer_observations,
    mask_outer_merged_for_inner_domain,
)


def _data(lat, lon, *, times=None):
    times = pd.to_datetime(["2019-01-01"] if times is None else times)
    values = da.ones((len(times), len(lat), len(lon)), chunks="auto")
    dataset = xr.Dataset(
        {"fp": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    return _NestedDomainData(
        site_data={"A": dataset, "B": dataset.copy(deep=False)},
        flux_data={"inventory": dataset.rename({"fp": "flux"})},
        site_options=SiteOptions.from_inputs(sites=["A", "B"], averaging_period=["1h", "1h"]),
    )


@pytest.mark.parametrize("domain", ["outer", "inner"])
def test_nested_rejects_different_site_grids(domain):
    outer = _data([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    inner = _data([0.5, 1.5], [0.5, 1.5])
    chosen = outer if domain == "outer" else inner
    datasets = dict(chosen.site_data)
    datasets["B"] = datasets["B"].assign_coords(lat=datasets["B"].lat + 0.25)
    chosen = replace(chosen, site_data=datasets)
    with pytest.raises(ValueError, match=f"{domain.capitalize()}.*same indexed lat/lon grid"):
        mask_outer_merged_for_inner_domain(chosen if domain == "outer" else outer,
                                          chosen if domain == "inner" else inner)


def test_shared_inner_extent_masks_each_actual_target_grid_without_computation():
    outer = _data([2.0, 1.0, 0.0], [0.0, 1.0, 2.0])
    inner = _data([0.5, 1.0, 1.5], [0.5, 1.0, 1.5])
    flux = xr.DataArray(
        da.ones((4, 4), chunks=(2, 2)), dims=("lat", "lon"),
        coords={"lat": [0.0, 0.5, 1.5, 2.0], "lon": [0.0, 0.5, 1.5, 2.0]},
    )
    outer = replace(outer, flux_data={"inventory": flux.rename("flux").to_dataset()})
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        masked = mask_outer_merged_for_inner_domain(outer, inner)
    assert not tasks
    for site in outer.sites:
        np.testing.assert_array_equal(masked.site_data[site].fp.compute(),
                                      [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        assert set(outer.site_data[site].fp.data.dask).issubset(masked.site_data[site].fp.data.dask)
        np.testing.assert_array_equal(outer.site_data[site].fp.compute(), np.ones((1, 3, 3)))
    np.testing.assert_array_equal(masked.flux_data["inventory"].flux.compute(),
                                  [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])


def test_unsorted_site_local_alignment_is_reapplied_after_assembly_drops_rows():
    outer = _data([0.0], [0.0], times=["2019-01-01T02:00", "2019-01-01T00:00"])
    inner = _data([0.0], [0.0], times=["2019-01-01T01:10", "2019-01-01T00:10", "2019-01-01T02:10"])
    inner = replace(inner, site_data={site: dataset.assign(
        fp=dataset.fp * xr.DataArray([10, 20, 30], dims="time"))
        for site, dataset in inner.site_data.items()})
    aligned = align_inner_merged_to_outer_observations(outer, inner, time_tolerance="15min")
    for site in outer.sites:
        np.testing.assert_array_equal(aligned.site_data[site].fp.compute().values[:, 0, 0], [30, 20])
    # Assembly can remove different rows at each site. The prepared boundary
    # selects the surviving rows again rather than reusing acquisition positions.
    outer_index = pd.MultiIndex.from_tuples(
        [("B", pd.Timestamp("2019-01-01T00:00")), ("A", pd.Timestamp("2019-01-01T02:00"))],
        names=["site", "time"],
    )
    inner_index = pd.MultiIndex.from_tuples(
        [(site, time) for site in ["A", "B"] for time in aligned.site_data[site].time.values],
        names=["site", "time"],
    )
    assert _inner_measurement_positions(outer_index, inner_index, time_tolerance=None) == [3, 0]


def test_unknown_named_domain_uses_first_site_without_computing_or_mutating():
    data = _data([50.0, 51.0], [-2.0, -1.0])
    shifted = data.site_data["B"].assign_coords(lat=data.site_data["B"].lat + 1e-7)
    shifted.lat.attrs["units"] = "degrees_north"
    data = replace(data, site_data={"A": data.site_data["A"], "B": shifted})
    tasks = []
    with pytest.warns(UserWarning, match="CUSTOM.*retained site 'A'") as notices:
        with Callback(pretask=lambda key, *_: tasks.append(key)):
            aligned = _align_domain_sites(data, domain="CUSTOM", label="Inner")
    assert len(notices) == 1
    assert not tasks
    np.testing.assert_array_equal(aligned.site_data["B"].lat, data.site_data["A"].lat)
    assert aligned.site_data["B"].lat.attrs == {"units": "degrees_north"}
    assert aligned.site_data["B"].fp.data is shifted.fp.data
    assert not shifted.lat.equals(aligned.site_data["B"].lat)


def test_known_domain_uses_public_normalization_on_actual_datasets(monkeypatch):
    from openghg_inversions.rhime import nested

    data = _data([50.000001, 51.000001], [-2.0, -1.0])
    data.site_data["A"].lat.attrs["units"] = "degrees_north"
    reference = xr.Dataset(coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]})
    calls = []

    def normalize(dataset, domain):
        calls.append((dataset, domain))
        return dataset.assign_coords(lat=reference.lat, lon=reference.lon)

    monkeypatch.setattr(nested, "get_domain_info", lambda: {"KNOWN": {}})
    monkeypatch.setattr(nested, "align_lat_lon", normalize)
    aligned = _align_domain_sites(data, domain="KNOWN", label="Inner")
    assert [domain for _, domain in calls] == ["KNOWN", "KNOWN", "KNOWN"]
    for site in data.sites:
        np.testing.assert_array_equal(aligned.site_data[site].lat, reference.lat)
        assert aligned.site_data[site].fp.data is data.site_data[site].fp.data
    assert aligned.site_data["A"].lat.attrs == {"units": "degrees_north"}
    assert data.site_data["A"].lat[0] == 50.000001


def test_public_openghg_domain_normalization_preserves_lazy_payload():
    from openghg.util import find_domain

    lat, lon, *_ = find_domain("EUROPE")
    data = _data(lat, lon)
    shifted = data.site_data["B"].assign_coords(lat=data.site_data["B"].lat + 1e-7)
    data = replace(data, site_data={"A": data.site_data["A"], "B": shifted})
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        aligned = _align_domain_sites(data, domain="EUROPE", label="Outer")
    assert not tasks
    np.testing.assert_array_equal(aligned.site_data["B"].lat, lat)
    assert aligned.site_data["B"].fp.data is shifted.fp.data
    malformed = replace(data, site_data={"A": data.site_data["A"].isel(lat=slice(1, None)),
                                       "B": shifted})
    with pytest.raises(ValueError, match="same indexed lat/lon grid"):
        _align_domain_sites(malformed, domain="EUROPE", label="Outer")


@pytest.mark.parametrize("footprint_shift", [1e-7, .01])
def test_shifted_europe_footprint_and_flux_generate_reconstructable_quadtree(footprint_shift):
    from openghg.util import find_domain
    from openghg_inversions.rhime.nested import _prepare_one_domain

    lat, lon, *_ = find_domain("EUROPE")
    data = _data(lat + 1e-7, lon + 1e-7)
    data = replace(data, site_data={
        site: dataset.assign_coords(lat=lat + footprint_shift, lon=lon + footprint_shift)
        for site, dataset in data.site_data.items()
    })
    data = replace(data, site_data={
        site: dataset.assign(fp_x_flux=dataset.fp, mf=("time", [10.]), mf_error=("time", [1.]),
                             mf_repeatability=("time", [1.]), mf_variability=("time", [0.]))
        for site, dataset in data.site_data.items()
    })
    data.flux_data["inventory"].lat.attrs["units"] = "degrees_north"
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        aligned = _align_domain_sites(data, domain="EUROPE", label="Inner")
    assert not tasks
    assert aligned.flux_data["inventory"].flux.data is data.flux_data["inventory"].flux.data
    assert aligned.flux_data["inventory"].lat.attrs == {"units": "degrees_north"}
    np.testing.assert_array_equal(data.flux_data["inventory"].lat, lat + 1e-7)
    prepared = _prepare_one_domain(
        aligned, domain="EUROPE", start_date="2019-01-01", flux_sources=("inventory",),
        basis_algorithm="quadtree", nbasis=4, fix_basis_outer_regions=False,
        use_bc=False, bc_freq=None, min_error=0., min_error_options={"by_site": False},
        basis_inputs={},
    )
    basis = prepared.basis_functions
    state = xr.ones_like(prepared.inv_inputs.H.isel(nmeasure=0, drop=True))
    reconstructed = basis.state_to_flux(state)
    np.testing.assert_array_equal(reconstructed.lat, lat)
    np.testing.assert_array_equal(reconstructed.lon, lon)
    np.testing.assert_array_equal(reconstructed.compute(), np.ones(reconstructed.shape))
    assert isinstance(basis.flux.data, da.Array)


@pytest.mark.parametrize("domain", [None, "CUSTOM"])
def test_corresponding_flux_uses_site_reference_lazily_with_metadata(domain):
    data = _data([50., 51.], [-2., -1.])
    flux = data.flux_data["inventory"].assign_coords(lat=data.site_data["A"].lat + 1e-7)
    flux.lat.attrs["units"] = "degrees_north"
    flux.attrs["inventory"] = "borrowed"
    data = replace(data, flux_data={"inventory": flux})
    tasks = []
    from contextlib import nullcontext
    notice = pytest.warns(UserWarning, match="CUSTOM.*retained site 'A'") if domain else nullcontext()
    with notice, Callback(pretask=lambda key, *_: tasks.append(key)):
        aligned = _align_domain_sites(data, domain=domain, label="Outer")
    result = aligned.flux_data["inventory"]
    assert not tasks
    assert result.flux.data is flux.flux.data
    assert result.flux.chunks == flux.flux.chunks
    assert result.attrs == flux.attrs
    assert result.lat.attrs == flux.lat.attrs == {"units": "degrees_north"}
    np.testing.assert_array_equal(result.lat, data.site_data["A"].lat)
    np.testing.assert_array_equal(flux.lat, data.site_data["A"].lat + 1e-7)


@pytest.mark.parametrize("domain, shift", [("EUROPE", 1e-7), ("EUROPE", .01), ("CUSTOM", 1e-7)])
@pytest.mark.parametrize("plural", [False, True])
def test_supplied_partition_reconstructs_on_normalized_runtime_grid(domain, shift, plural, tmp_path):
    from contextlib import nullcontext
    from openghg.util import find_domain
    from openghg_inversions.rhime.nested import _prepare_one_domain
    from openghg_inversions.rhime.params import DataConfig
    from openghg_inversions.rhime.preparation import resolve_basis_inputs

    lat, lon = find_domain("EUROPE")[:2] if domain == "EUROPE" else ([50., 51.], [-2., -1.])
    lat, lon = np.asarray(lat), np.asarray(lon)
    data = _data(lat + shift, lon + shift)
    data = replace(data, site_data={
        site: dataset.assign(fp_x_flux=dataset.fp, mf=("time", [10.]), mf_error=("time", [1.]),
                             mf_repeatability=("time", [1.]), mf_variability=("time", [0.]))
        for site, dataset in data.site_data.items()
    })
    partition = xr.ones_like(data.site_data["A"].fp.isel(time=0, drop=True), dtype=int).compute()
    if domain == "CUSTOM":
        partition = partition.assign_coords(lat=partition.lat + shift, lon=partition.lon + shift)
    partition.lat.attrs["units"] = "degrees_north"
    path = tmp_path / "partition.nc"
    partition.expand_dims(time=[pd.Timestamp("2019-01-01")]).rename("basis").to_netcdf(path)
    config = DataConfig(domain=domain, start_date="2019-01-01", end_date="2019-01-02", species="ch4",
                        site_options=data.site_options, flux_sources=("inventory",),
                        basis_partition_path=None if plural else path,
                        basis_partition_paths=[path] if plural else None)
    basis_inputs = resolve_basis_inputs(config)
    borrowed = basis_inputs["basis_partition"].copy(deep=True)
    notice = pytest.warns(UserWarning, match="CUSTOM") if domain == "CUSTOM" else nullcontext()
    with notice:
        aligned = _align_domain_sites(data, domain=domain, label="Inner")
    prepared = _prepare_one_domain(
        aligned, domain=domain, start_date="2019-01-01", flux_sources=("inventory",),
        basis_algorithm=None, nbasis=1, fix_basis_outer_regions=False,
        use_bc=False, bc_freq=None, min_error=0., min_error_options={"by_site": False},
        basis_inputs=basis_inputs,
    )
    basis = prepared.basis_functions
    state = xr.ones_like(prepared.inv_inputs.H.isel(nmeasure=0, drop=True))
    reconstructed = basis.state_to_flux(state)
    xr.testing.assert_equal(reconstructed.lat, aligned.flux_data["inventory"].lat)
    xr.testing.assert_equal(reconstructed.lon, aligned.flux_data["inventory"].lon)
    np.testing.assert_array_equal(reconstructed.compute(), np.ones(reconstructed.shape))
    xr.testing.assert_identical(basis_inputs["basis_partition"], borrowed)
    np.testing.assert_array_equal(data.flux_data["inventory"].lat, lat + shift)
    assert isinstance(basis.flux.data, da.Array)


def test_supplied_partition_on_distinct_inventory_grid_reconstructs_without_domain_relabeling(monkeypatch):
    from openghg.util import find_domain
    from openghg_inversions.rhime import nested

    lat, lon, *_ = find_domain("EUROPE")
    data = _data(lat, lon)
    flux = data.flux_data["inventory"].assign_coords(lat=lat + 5., lon=lon + 5.)
    data = replace(data, flux_data={"inventory": flux})
    partition = xr.ones_like(flux.flux.isel(time=0, drop=True), dtype=int).compute()
    partition = partition.assign_coords(lat=partition.lat + 1e-7, lon=partition.lon + 1e-7)
    # Observation projection is independent of native prior-flux reconstruction.
    monkeypatch.setattr(nested, "build_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(nested, "assemble_rhime_inputs", lambda sites, basis, **kwargs: basis)
    basis = nested._prepare_one_domain(
        data, domain="EUROPE", start_date="2019-01-01", flux_sources=("inventory",),
        basis_algorithm=None, nbasis=1, fix_basis_outer_regions=False,
        use_bc=False, bc_freq=None, min_error=0., min_error_options={"by_site": False},
        basis_inputs={"basis_partition": partition},
    )
    reconstructed = basis.state_to_flux(xr.DataArray([1.], dims="region", coords={"region": [0]}))
    xr.testing.assert_equal(reconstructed.lat, flux.lat)
    xr.testing.assert_equal(reconstructed.lon, flux.lon)
    np.testing.assert_array_equal(reconstructed.compute(), np.ones(reconstructed.shape))


@pytest.mark.parametrize("domain", [None, "EUROPE"])
@pytest.mark.parametrize("grid_change", ["extent", "resolution"])
def test_genuinely_different_flux_grid_is_retained(domain, grid_change):
    from openghg.util import find_domain

    lat, lon, *_ = find_domain("EUROPE")
    data = _data(lat, lon)
    flux = data.flux_data["inventory"]
    flux = (flux.assign_coords(lat=flux.lat + 5.) if grid_change == "extent"
            else flux.isel(lat=slice(None, None, 2)))
    data = replace(data, flux_data={"inventory": flux})
    with Callback(pretask=lambda *_: pytest.fail("normalization computed borrowed flux")):
        aligned = _align_domain_sites(data, domain=domain, label="Outer")
    assert aligned.flux_data["inventory"] is flux


def test_normalized_corresponding_flux_and_footprint_share_inclusive_mask_boundary():
    from openghg.util import find_domain

    lat, lon, *_ = find_domain("EUROPE")
    outer = _data(lat + 1e-7, lon + 1e-7)
    inner = _data(lat[10:12], lon[10:12])
    with Callback(pretask=lambda *_: pytest.fail("masking computed borrowed payloads")):
        masked = mask_outer_merged_for_inner_domain(outer, inner, outer_domain="EUROPE")
    footprint = masked.site_data["A"].fp
    flux = masked.flux_data["inventory"].flux
    xr.testing.assert_equal(footprint.rename("flux"), flux)
    np.testing.assert_array_equal(flux.isel(lat=slice(10, 12), lon=slice(10, 12)).compute(), 0.)
    assert flux.isel(lat=12, lon=12).compute().item() == 1.
    assert set(outer.flux_data["inventory"].flux.data.dask).issubset(flux.data.dask)
    assert outer.flux_data["inventory"].flux.isel(lat=11, lon=11).compute().item() == 1.
