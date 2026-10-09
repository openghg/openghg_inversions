"""Dataset basis orchestration preserves scientific weights and retained flux."""

from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis import (
    basis_functions_from_flat_basis,
    basis_weights_from_data,
    bucket_basis_from_weights,
    make_basis_functions,
    quadtree_basis_from_weights,
    region_constrained_basis_from_weights,
)
from openghg_inversions.basis._functions import _mean_fp_times_mean_flux
from openghg_inversions.basis.basis_functions import flux_from_data


def _inputs():
    coords = {"lat": [50., 51., 52., 53.], "lon": [0., 1., 2., 3.],
              "time": pd.date_range("2020-01-01", periods=2)}
    flux = xr.DataArray(da.ones((4, 4, 2), chunks=(2, 2, 1)),
                        dims=("lat", "lon", "time"), coords=coords, name="flux")
    sites = {"A": (flux * 2).rename("fp").to_dataset(),
             "B": (flux * 4).rename("fp").to_dataset()}
    sources = {"a": flux.to_dataset(), "b": (flux * 7).to_dataset()}
    return sites, sources


@pytest.mark.parametrize("split_by_sectors", [False, True])
def test_dataset_weights_select_first_requested_source_and_retain_all_flux(split_by_sectors):
    sites, sources = _inputs()
    weights = basis_weights_from_data(sites, sources, ["b", "a"])
    xr.testing.assert_equal(weights, _mean_fp_times_mean_flux(
        sources["b"].flux, [value.fp for value in sites.values()]).compute())
    np.testing.assert_array_equal(weights, np.full((4, 4), 21.))
    assert not isinstance(weights.data, da.Array)
    assert isinstance(sources["a"].flux.data, da.Array)
    labels = xr.ones_like(weights, dtype=int)
    retained = basis_functions_from_flat_basis(
        flux_data=sources, split_by_sectors=split_by_sectors, basis_flat=labels)
    assert isinstance(retained.flux.data, da.Array)
    if split_by_sectors:
        assert list(retained.flux.source.values) == ["a", "b"]
        np.testing.assert_array_equal(retained.flux.sel(source="b"), np.full((4, 4, 2), 7.))
    else:
        np.testing.assert_array_equal(retained.flux, np.full((4, 4, 2), 8.))


@pytest.mark.parametrize("algorithm", ["quadtree", "weighted", "region_constrained"])
def test_dataset_generated_basis_matches_array_kernel(algorithm, tmp_path):
    sites, sources = _inputs()
    weights = basis_weights_from_data(sites, sources, ["b"])
    classes = xr.ones_like(weights, dtype=int)
    xr.Dataset({"country": classes}).to_netcdf(tmp_path / "country-land-sea_TEST.nc")
    kwargs = {"nbasis": 4}
    if algorithm == "quadtree":
        expected = quadtree_basis_from_weights(weights, "2020-01-01", "TEST", seed=1, **kwargs)
    elif algorithm == "weighted":
        expected = bucket_basis_from_weights(weights, "2020-01-01", "TEST",
                                             country_directory=str(tmp_path), **kwargs)
    else:
        expected = region_constrained_basis_from_weights(weights, "2020-01-01", "TEST",
                                                         region_classes=classes, **kwargs)
    actual = make_basis_functions(
        site_data=sites, flux_data=sources, split_by_sectors=True,
        species="ch4", domain="TEST", start_date="2020-01-01", emissions_name=["b"],
        nbasis=4, basis_algorithm=algorithm, country_directory=str(tmp_path), region_classes=classes)
    xr.testing.assert_equal(actual.flat_basis(), expected.squeeze("time", drop=True))
    assert list(actual.flux.source.values) == ["a", "b"]
    assert isinstance(actual.flux.data, da.Array)


@pytest.mark.parametrize("output_format", ["legacy", "datatree"])
def test_dataset_saved_basis_uses_runtime_flux_and_source_order(output_format, tmp_path):
    sites, sources = _inputs()
    classes = xr.ones_like(sources["a"].flux.isel(time=0, drop=True), dtype=int).compute()
    generated = make_basis_functions(
        site_data=sites, flux_data=sources, split_by_sectors=True,
        species="ch4", domain="TEST", start_date="2020-01-01", emissions_name=["b"],
        nbasis=2, basis_algorithm="region_constrained", region_classes=classes,
        output_path=str(tmp_path), basis_output_format=output_format)
    runtime = {"b": sources["b"] * 3, "a": sources["a"] * 5}
    loaded = make_basis_functions(
        site_data=sites, flux_data=runtime, split_by_sectors=True,
        species="ch4", domain="TEST", start_date="2020-01-01", emissions_name=["a"],
        nbasis=2, fp_basis_case="region_constrained_ch4", basis_directory=str(tmp_path))
    xr.testing.assert_equal(loaded.flat_basis(), generated.flat_basis())
    xr.testing.assert_equal(loaded.flux, flux_from_data(runtime, split_by_sectors=True))
    assert list(loaded.flux.source.values) == ["b", "a"]
    assert isinstance(loaded.flux.data, da.Array)


@pytest.mark.parametrize("empty_inner", [False, True])
def test_dataset_fixed_outer_matches_label_composition(empty_inner, tmp_path):
    sites, sources = _inputs()
    outer = xr.DataArray([[0, 0, 2, 2], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]],
                         dims=("lat", "lon"), coords={"lat": sources["a"].lat, "lon": sources["a"].lon},
                         name="region", attrs={"inner_region_label": 1})
    if empty_inner:
        sites = {key: dataset.assign(fp=dataset.fp.where(outer != 1, 0.))
                 for key, dataset in sites.items()}
    outer_path = tmp_path / "outer.nc"
    outer.to_dataset().to_netcdf(outer_path)
    actual = make_basis_functions(
        site_data=sites, flux_data=sources, split_by_sectors=True,
        species="ch4", domain="TEST", start_date="2020-01-01", emissions_name=["b"],
        nbasis=1, basis_algorithm="region_constrained", region_classes=xr.ones_like(outer),
        fix_outer_regions=True, outer_regions_path=outer_path, allow_empty_inner_region=empty_inner)
    expected = outer + 1
    if not empty_inner:
        expected = expected.where(outer != 1, 4)
    xr.testing.assert_equal(actual.flat_basis(), expected.rename("basis"))
    assert isinstance(actual.flux.data, da.Array)


def test_rhime_basis_passes_borrowed_datasets_without_legacy_adapter(monkeypatch):
    from openghg_inversions.rhime import preparation
    sites, sources = _inputs()
    merged = SimpleNamespace(site_data=sites, flux_data=sources, split_by_sectors=True)
    expected = object()

    def build(**kwargs):
        assert kwargs["site_data"] is sites
        assert kwargs["flux_data"] is sources
        assert kwargs["split_by_sectors"] is True
        assert kwargs["emissions_name"] == ["b", "a"]
        return expected

    monkeypatch.setattr(preparation, "make_basis_functions", build)
    assert preparation.build_rhime_basis(
        merged, species="ch4", domain="TEST", start_date="2020-01-01",
        flux_sources=["b", "a"], output_name="test") is expected


def test_dataset_region_constrained_requires_classes():
    sites, sources = _inputs()
    with pytest.raises(ValueError, match="region_classes must be supplied"):
        make_basis_functions(
            site_data=sites, flux_data=sources, species="ch4", domain="TEST",
            start_date="2020-01-01", emissions_name=["a"], nbasis=2,
            basis_algorithm="region_constrained")
