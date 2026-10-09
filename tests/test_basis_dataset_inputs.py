"""Dataset basis orchestration preserves scientific weights and retained flux."""


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
        species="ch4", domain="TEST", start_date="2020-01-01", flux_sources=["b"],
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
        species="ch4", domain="TEST", start_date="2020-01-01", flux_sources=["b"],
        nbasis=2, basis_algorithm="region_constrained", region_classes=classes,
        output_path=str(tmp_path), basis_output_format=output_format)
    runtime = {"b": sources["b"] * 3, "a": sources["a"] * 5}
    loaded = make_basis_functions(
        site_data=sites, flux_data=runtime, split_by_sectors=True,
        species="ch4", domain="TEST", start_date="2020-01-01", flux_sources=["a"],
        nbasis=2, fp_basis_case="region_constrained_ch4", basis_directory=str(tmp_path),
        basis_algorithm="unused_invalid_algorithm")
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
        species="ch4", domain="TEST", start_date="2020-01-01", flux_sources=["b"],
        nbasis=1, basis_algorithm="region_constrained", region_classes=xr.ones_like(outer),
        fix_outer_regions=True, outer_regions_path=outer_path, allow_empty_inner_region=empty_inner)
    expected = outer + 1
    if not empty_inner:
        expected = expected.where(outer != 1, 4)
    xr.testing.assert_equal(actual.flat_basis(), expected.rename("basis"))
    assert isinstance(actual.flux.data, da.Array)



def test_dataset_region_constrained_requires_classes():
    sites, sources = _inputs()
    with pytest.raises(ValueError, match="region_classes must be supplied"):
        make_basis_functions(
            site_data=sites, flux_data=sources, species="ch4", domain="TEST",
            start_date="2020-01-01", flux_sources=["a"], nbasis=2,
            basis_algorithm="region_constrained")


@pytest.mark.parametrize("entry_point", ["make", "make_fixed", "fixed"])
def test_unknown_algorithm_does_not_execute_borrowed_dask_graph(entry_point):
    from dask import delayed
    from openghg_inversions.basis import fixed_outer_regions_basis_from_data

    @delayed
    def must_remain_lazy():
        raise AssertionError("Invalid algorithms must not compute weights")

    sites, sources = _inputs()
    sources["a"] = sources["a"].assign(
        flux=sources["a"].flux.copy(data=da.from_delayed(
            must_remain_lazy(), shape=(4, 4, 2), dtype=float)))
    with pytest.raises(ValueError, match="Basis algorithm not recognised"):
        if entry_point == "fixed":
            fixed_outer_regions_basis_from_data(
                sites, sources, "2020-01-01", "invalid", "TEST", ["a"])
        else:
            make_basis_functions(
                site_data=sites, flux_data=sources, species="ch4", domain="TEST",
                start_date="2020-01-01", flux_sources=["a"], nbasis=1,
                basis_algorithm="invalid", fix_outer_regions=entry_point == "make_fixed")


@pytest.mark.parametrize("fixed_outer", [False, True])
def test_generated_kernel_key_error_is_not_relabelled(monkeypatch, tmp_path, fixed_outer):
    import openghg_inversions.basis._functions as functions

    sites, sources = _inputs()
    outer = xr.DataArray([[0, 0, 2, 2], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]],
                         dims=("lat", "lon"), coords={"lat": sources["a"].lat, "lon": sources["a"].lon},
                         name="region", attrs={"inner_region_label": 1})
    outer_path = tmp_path / "outer.nc"
    outer.to_dataset().to_netcdf(outer_path)
    expected_error = KeyError("missing kernel input")

    def fail_kernel(*args, **kwargs):
        raise expected_error

    monkeypatch.setattr(functions, "quadtree_basis_from_weights", fail_kernel)
    with pytest.raises(KeyError) as error:
        make_basis_functions(
            site_data=sites, flux_data=sources, species="ch4", domain="TEST",
            start_date="2020-01-01", flux_sources=["a"], nbasis=1,
            basis_algorithm="quadtree", fix_outer_regions=fixed_outer, outer_regions_path=outer_path)
    assert error.value is expected_error


@pytest.mark.parametrize("empty_inner", [False, True])
def test_fixed_outer_executes_shared_weight_graph_once(empty_inner, tmp_path):
    from dask import delayed
    from openghg_inversions.basis import fixed_outer_regions_basis_from_data

    sites, sources = _inputs()
    outer = xr.DataArray([[0, 0, 2, 2], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]],
                         dims=("lat", "lon"), coords={"lat": sources["a"].lat, "lon": sources["a"].lon},
                         name="region", attrs={"inner_region_label": 1})
    executions = []

    @delayed
    def shared_payload():
        executions.append(True)
        return np.ones((4, 4, 2))

    flux = sources["a"].flux.copy(data=da.from_delayed(
        shared_payload(), shape=(4, 4, 2), dtype=float))
    sources = {"a": flux.to_dataset()}
    footprint = flux * 2
    if empty_inner:
        footprint = footprint.where(outer != 1, 0.)
    sites = {"A": footprint.rename("fp").to_dataset()}
    outer_path = tmp_path / "outer.nc"
    outer.to_dataset().to_netcdf(outer_path)
    actual = fixed_outer_regions_basis_from_data(
        sites, sources, "2020-01-01", "region_constrained", "TEST", ["a"], nbasis=1,
        region_classes=xr.ones_like(outer), outer_regions_path=outer_path, allow_empty_inner_region=True)
    expected = outer + 1
    if not empty_inner:
        expected = expected.where(outer != 1, 4)
    xr.testing.assert_equal(actual.squeeze("time", drop=True), expected.rename("basis"))
    assert len(executions) == 1
    assert isinstance(sources["a"].flux.data, da.Array)


@pytest.mark.parametrize("sources", [None, ["b", "a"]])
def test_shipped_basis_source_keyword_warns_and_preserves_science(sources):
    """Deprecated source selection produces identical labels and retained flux."""
    sites, flux = _inputs()
    classes = xr.ones_like(flux["a"].flux.isel(time=0, drop=True), dtype=int).compute()
    options = dict(site_data=sites, flux_data=flux, split_by_sectors=True,
                   species="ch4", domain="TEST", start_date="2020-01-01", nbasis=2,
                   basis_algorithm="region_constrained", region_classes=classes)
    canonical = make_basis_functions(**options, flux_sources=sources)
    with pytest.warns(DeprecationWarning, match=r"removed in 0\.9"):
        deprecated = make_basis_functions(**options, emissions_name=sources)
    xr.testing.assert_equal(canonical.flat_basis(), deprecated.flat_basis())
    # Independent preparations stamp their sanitation history separately.
    canonical_history = canonical.flux.attrs["history"]
    deprecated_history = deprecated.flux.attrs["history"]
    assert canonical_history.split(" OpenGHG Inversions:", 1)[1] == deprecated_history.split(
        " OpenGHG Inversions:", 1
    )[1]
    xr.testing.assert_identical(
        canonical.flux.assign_attrs(history=deprecated_history), deprecated.flux
    )
    assert list(canonical.flux.source.values) == ["a", "b"]
    assert isinstance(flux["a"].flux.data, da.Array)


@pytest.mark.parametrize("sources", [None, ["b", "a"]])
def test_basis_source_keyword_conflicts_fail_before_reading_inputs(sources):
    with pytest.raises(ValueError, match="cannot be supplied together"):
        make_basis_functions(site_data={}, flux_data={}, species="ch4", domain="TEST",
                             start_date="2020-01-01", nbasis=2,
                             flux_sources=sources, emissions_name=sources)
