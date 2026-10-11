"""Native prior resolution and retained-operator reuse have distinct owners."""

from contextlib import contextmanager

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr
from dask import delayed
from openghg.analyse._utils import match_dataset_dims, stack_datasets

from openghg_inversions.basis import BasisFunctions, basis_functions_from_flat_basis
from openghg_inversions.basis.operators import BasisMeta, MultiSourceBucketBasisOperator
from openghg_inversions.native_flux import resolve_runtime_flux


def _flux(value=1.0):
    return xr.DataArray(
        np.full((2, 2, 2), value),
        dims=("lat", "lon", "time"),
        coords={"lat": [50.0, 51.0], "lon": [0.0, 1.0], "time": pd.date_range("2020-01-01", periods=2)},
        name="flux",
    )


@pytest.mark.parametrize("source_specific", [False, True])
@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_operator_reuse_binds_current_prior_and_full_load_restores_saved_prior(
    tmp_path, source_specific, suffix
):
    old = resolve_runtime_flux(
        {"z": _flux(2).to_dataset(), "a": _flux(3).to_dataset()}, split_by_sectors=True
    )
    labels = xr.ones_like(old.sel(source="z").isel(time=0, drop=True), dtype=int)
    partitions = {"z": labels, "a": labels * 2} if source_specific else labels
    saved = basis_functions_from_flat_basis(flux=old, basis_flat=partitions)
    path = tmp_path / f"basis{suffix}"
    saved.save(path)
    sources = (
        {"a": _flux(-7).chunk().to_dataset()}
        if source_specific
        else {"new-source": _flux(-7).chunk().to_dataset()}
    )
    current = resolve_runtime_flux(sources, split_by_sectors=True)
    current_payload = current.data
    reused = BasisFunctions.reuse_operator(path, flux=current)
    assert reused.source_labels == tuple(sources)
    assert reused.operator.meta.state_dim == "region"
    assert current.data is current_payload
    assert isinstance(reused.flux.data, da.Array)
    state = xr.ones_like(reused.operator.basis_matrix.isel(lat=0, lon=0, drop=True))
    expected = current.rename(source="native_source") if source_specific else current
    reconstructed = reused.state_to_flux(state)
    xr.testing.assert_equal(reconstructed, expected.transpose(*reconstructed.dims))
    response = current * 5
    sensitivity = reused.sensitivity(response)
    np.testing.assert_allclose(sensitivity, -140.0)
    restored = BasisFunctions.load(path)
    xr.testing.assert_equal(restored.flux, old)
    old_state = xr.ones_like(restored.operator.basis_matrix.isel(lat=0, lon=0, drop=True))
    expected_old = old.rename(source="native_source") if source_specific else old
    restored_flux = restored.state_to_flux(old_state)
    xr.testing.assert_equal(restored_flux, expected_old.transpose(*restored_flux.dims))


def test_reuse_preserves_source_subset_order_and_operator_dimensions(tmp_path):
    flux = resolve_runtime_flux(
        {source: _flux(i + 1).to_dataset() for i, source in enumerate(["z", "b", "a"])}, split_by_sectors=True
    )
    labels = xr.ones_like(flux.isel(source=0, time=0, drop=True), dtype=int).rename(lat="y", lon="x")
    operator = MultiSourceBucketBasisOperator(
        {source: labels for source in ["z", "b", "a"]},
        meta=BasisMeta(grid_dims=("y", "x"), state_dim="custom_state"),
        source_dim="inventory",
        region_in_source_dim="local_region",
    )
    basis = BasisFunctions(operator=operator, flux=flux.rename(lat="y", lon="x"))
    path = tmp_path / "basis.nc"
    basis.save(path)
    reused = BasisFunctions.reuse_operator(path, flux=basis.flux.sel(source=["a", "z"]))
    assert reused.source_labels == ("a", "z")
    assert reused.operator.meta == operator.meta
    assert reused.operator.source_dim == "inventory"
    assert reused.operator.region_in_source_dim == "local_region"
    with pytest.raises(ValueError, match="missing sources"):
        BasisFunctions.reuse_operator(
            path, flux=basis.flux.sel(source=["a"]).assign_coords(source=["missing"])
        )


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_operator_reuse_does_not_execute_saved_flux_payload(monkeypatch, tmp_path, suffix):
    basis = basis_functions_from_flat_basis(
        flux=_flux(), basis_flat=xr.ones_like(_flux().isel(time=0, drop=True), dtype=int)
    )
    path = tmp_path / f"basis{suffix}"
    basis.save(path)
    open_tree = xr.open_datatree
    executions = []

    @delayed
    def forbidden_payload():
        executions.append(True)
        raise AssertionError("saved flux payload must not be read during operator reuse")

    @contextmanager
    def trapped_tree(*args, **kwargs):
        with open_tree(*args, **kwargs) as tree:
            tree["flux"]["flux"] = tree["flux"]["flux"].copy(
                data=da.from_delayed(forbidden_payload(), shape=(2, 2, 2), dtype=float)
            )
            yield tree

    monkeypatch.setattr(xr, "open_datatree", trapped_tree)
    reused = BasisFunctions.reuse_operator(path, flux=_flux(-4))
    xr.testing.assert_equal(reused.flux, _flux(-4))
    assert executions == []
    with pytest.raises(AssertionError, match="saved flux payload"):
        BasisFunctions.load(path)
    assert executions == [True]


def test_native_nonsectoral_resolution_matches_openghg_time_forward_fill():
    first = _flux(2)
    second = _flux(-3).assign_coords(time=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    sources = {"z": first.to_dataset(), "a": second.to_dataset()}
    expected = stack_datasets(
        match_dataset_dims(list(sources.values()), dims=["lat", "lon"]), dim="time", method="ffill"
    ).flux
    actual = resolve_runtime_flux(sources)
    xr.testing.assert_equal(actual, expected.fillna(0))
    with pytest.raises(ValueError, match="join.*exact"):
        resolve_runtime_flux(sources, split_by_sectors=True)


@pytest.mark.parametrize("split", [False, True])
def test_runtime_flux_borrows_sparse_dask_sources_and_preserves_shared_graph(split):
    executions = []

    @delayed
    def payload():
        executions.append(True)
        return sparse.COO.from_numpy(np.full((2, 2, 2), -2.0))

    source = _flux().copy(
        data=da.from_delayed(
            payload(), shape=(2, 2, 2), dtype=float, meta=sparse.COO.from_numpy(np.empty((0, 0, 0)))
        )
    )
    borrowed = source.data
    runtime = resolve_runtime_flux(
        {"z": source.to_dataset(), "a": (source * 3).to_dataset()}, split_by_sectors=split
    )
    assert source.data is borrowed
    assert isinstance(runtime.data, da.Array)
    assert isinstance(runtime.data._meta, sparse.COO)
    assert executions == []
    runtime.compute()
    assert executions == [True]


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("schema", "Unexpected schema"),
        ("version", "Unexpected schema_version"),
        ("missing_flux", "Missing variable 'flux'"),
    ],
)
def test_operator_reuse_validates_artifact_structure_before_loading(monkeypatch, tmp_path, mutation, error):
    basis = basis_functions_from_flat_basis(
        flux=_flux(), basis_flat=xr.ones_like(_flux().isel(time=0, drop=True), dtype=int)
    )
    tree = basis.to_datatree()
    if mutation == "schema":
        tree.attrs["schema"] = "unsupported"
    elif mutation == "version":
        tree.attrs["schema_version"] = 99
    else:
        tree["flux"] = xr.DataTree(xr.Dataset())
    path = tmp_path / "invalid.nc"
    tree.to_netcdf(path)
    monkeypatch.setattr(xr.DataTree, "load", lambda self, **kwargs: pytest.fail("invalid artifact executed"))
    with pytest.raises(ValueError, match=error):
        BasisFunctions.reuse_operator(path, flux=_flux())


def test_operator_reuse_rejects_incompatible_current_grid_without_mutation(tmp_path):
    basis = basis_functions_from_flat_basis(
        flux=_flux(), basis_flat=xr.ones_like(_flux().isel(time=0, drop=True), dtype=int)
    )
    path = tmp_path / "basis.nc"
    basis.save(path)
    current = _flux(-2).assign_coords(lat=[60.0, 61.0]).chunk()
    before = current.copy(deep=True)
    with pytest.raises(ValueError, match="Coordinate mismatch"):
        BasisFunctions.reuse_operator(path, flux=current)
    xr.testing.assert_identical(current, before)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.isel(time=[0]), "join.*exact"),
        (lambda value: value.isel(lon=0, drop=True), "do not exist"),
        (lambda value: value.expand_dims(level=[0]), "permuted list"),
        (lambda value: value.drop_vars("time"), "explicit one-dimensional time coordinate"),
        (lambda value: value.assign_coords(time=[value.time.values[0]] * 2), "duplicate time coordinates"),
        (lambda value: value.isel(time=0, drop=True), "timed and timeless"),
        (lambda value: value.assign_coords(lat=[60.0, 61.0]), "Coordinate mismatch"),
    ],
)
def test_native_sectoral_resolution_rejects_incompatible_sources(mutate, message):
    sources = {"z": _flux().to_dataset(), "a": mutate(_flux()).to_dataset()}
    with pytest.raises(ValueError, match=message):
        resolve_runtime_flux(sources, split_by_sectors=True)



def test_retained_native_prior_and_observation_response_have_independent_times():
    native = resolve_runtime_flux({"inventory": _flux(-2).chunk().to_dataset()})
    basis = basis_functions_from_flat_basis(
        flux=native, basis_flat=xr.ones_like(native.isel(time=0, drop=True), dtype=int),
    )
    observation_times = pd.date_range("2020-02-01", periods=3, freq="h")
    response = xr.DataArray(
        np.arange(1., 4.)[None, None, :] * np.ones((2, 2, 1)),
        dims=("lat", "lon", "time"),
        coords={"lat": native.lat, "lon": native.lon, "time": observation_times},
    )
    sensitivity = basis.sensitivity(response)
    np.testing.assert_allclose(sensitivity.squeeze("region"), [4., 8., 12.])
    np.testing.assert_array_equal(sensitivity.time, observation_times)
    xr.testing.assert_equal(basis.flux.time, native.time)
