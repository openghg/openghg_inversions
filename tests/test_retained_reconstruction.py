"""Directional linear reconstruction contracts for retained basis states."""

import numpy as np
import pytest
import xarray as xr
from dask.array import Array as DaskArray
from sparse import COO

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing._basis_products import reconstruct_flux_stats
from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs


def _ragged_basis() -> BasisFunctions:
    grid = {"lat": [50.0], "lon": [-2.0, -1.0]}
    bases = {
        "zeta": xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=grid),
        "alpha": xr.DataArray([[1, 1]], dims=("lat", "lon"), coords=grid),
    }
    flux = xr.DataArray(
        [[[-2.0, 3.0]], [[5.0, -7.0]]],
        dims=("source", "lat", "lon"),
        coords={"source": ["zeta", "alpha"], **grid},
        attrs={"units": "mol/m2/s"},
    )
    return BasisFunctions.from_multi_source_flat_basis(bases, flux)


def test_ragged_reconstruction_preserves_source_order_and_samples() -> None:
    basis = _ragged_basis()
    state = xr.DataArray(
        np.array([1.0, 2.0, 3.0])[:, None, None] * np.ones((1, 2, 3)),
        dims=("state", "chain", "draw"),
        coords={"state": basis.operator.basis_matrix.state, "chain": [0, 1], "draw": [0, 1, 2]},
    )
    native = basis.state_to_native(state)
    flux = basis.state_to_flux(state)
    assert native.native_source.values.tolist() == ["zeta", "alpha"]
    assert native.dims == ("native_source", "lat", "lon", "chain", "draw")
    np.testing.assert_array_equal(native.isel(chain=0, draw=0).values[:, 0, :], [[1, 2], [3, 3]])
    np.testing.assert_array_equal(flux.isel(chain=0, draw=0).values[:, 0, :], [[-2, 6], [15, -21]])
    with pytest.warns(DeprecationWarning):
        legacy = basis.interpolate(state, flux=True)
    xr.testing.assert_allclose(flux.sum("native_source").transpose(*legacy.dims), legacy)


def test_shared_basis_broadcasts_signed_source_flux() -> None:
    grid = {"lat": [50.0], "lon": [-2.0, -1.0]}
    flat = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=grid)
    flux = xr.DataArray(
        [[[-2.0, 3.0]], [[5.0, -7.0]]],
        dims=("source", "lat", "lon"), coords={"source": ["zeta", "alpha"], **grid},
    )
    basis = BasisFunctions.from_flat_basis(flat, flux)
    state = xr.DataArray([2.0, 4.0], dims="state", coords={"state": basis.operator.basis_matrix.state})
    assert "source" not in basis.state_to_native(state).dims
    reconstructed = basis.state_to_flux(state)
    assert reconstructed.source.values.tolist() == ["zeta", "alpha"]
    np.testing.assert_array_equal(
        reconstructed.transpose("source", "lat", "lon").values[:, 0, :],
        [[-4, 12], [10, -28]],
    )


def test_reconstruction_rejects_reordered_labels_and_remains_lazy() -> None:
    basis = _ragged_basis()
    state = xr.DataArray(
        [1.0, 2.0, 3.0], dims="state", coords={"state": basis.operator.basis_matrix.state},
    ).chunk()
    assert isinstance(basis.operator.basis_matrix.data._meta, COO)
    assert isinstance(basis.state_to_native(state).data, DaskArray)
    lazy_flux = basis.with_flux(basis.flux.chunk({"source": 1}))
    assert isinstance(lazy_flux.state_to_flux(state).data, DaskArray)
    assert isinstance(basis.flux.data, np.ndarray)
    with pytest.raises(ValueError, match="exactly match"):
        basis.state_to_native(state.isel(state=[1, 0, 2]))
    with pytest.raises(ValueError, match="exactly match"):
        basis.with_flux(basis.flux.isel(lat=[0], lon=[1, 0])).state_to_flux(state)


def test_sample_source_axis_is_distinct_from_flux_source() -> None:
    grid = {"lat": [50.0], "lon": [-2.0]}
    flat = xr.DataArray([[1]], dims=("lat", "lon"), coords=grid)
    flux = xr.DataArray(
        [[[-2.0]], [[3.0]]], dims=("source", "lat", "lon"),
        coords={"source": ["zeta", "alpha"], **grid},
    )
    basis = BasisFunctions.from_flat_basis(flat, flux)
    state = xr.DataArray(
        [[2.0, 4.0]], dims=("state", "source"),
        coords={"state": basis.operator.basis_matrix.state, "source": ["draw-a", "draw-b"]},
    )
    output = basis.state_to_flux(state)
    assert output.source.values.tolist() == ["zeta", "alpha"]
    assert output.state_source.values.tolist() == ["draw-a", "draw-b"]
    np.testing.assert_array_equal(
        output.transpose("source", "state_source", "lat", "lon").values[:, :, 0, 0],
        [[-4, -8], [6, 12]],
    )


def test_source_selection_and_flux_mask_preserve_requested_output() -> None:
    basis = _ragged_basis().select_sources(["alpha"])
    state = xr.DataArray(
        [3.0], dims="state", coords={"state": basis.operator.basis_matrix.state},
    )
    masked = basis.with_flux(basis.flux.where(basis.flux > 0, 0)).state_to_flux(state)
    assert masked.native_source.values.tolist() == ["alpha"]
    np.testing.assert_array_equal(masked.values[0, 0, :], [15.0, 0.0])
    assert basis.flux.values[0, 0, 1] == -7.0


def test_native_source_sample_axis_is_renamed() -> None:
    basis = _ragged_basis()
    state = xr.DataArray(
        np.ones((3, 2)), dims=("state", "native_source"),
        coords={"state": basis.operator.basis_matrix.state, "native_source": ["draw-a", "draw-b"]},
    )
    result = basis.state_to_native(state)
    assert result.native_source.values.tolist() == ["zeta", "alpha"]
    assert result.state_native_source.values.tolist() == ["draw-a", "draw-b"]


def test_sparse_completed_product_writes_netcdf(tmp_path) -> None:
    dense = np.array([[1.0, 0.0], [0.0, -2.0]])
    product = xr.Dataset(
        {"flux": (("lat", "lon"), COO.from_numpy(dense))},
        coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]},
    )
    path = tmp_path / "sparse_flux.nc"
    write_netcdf_preserving_bounds_attrs(product, path)
    with xr.open_dataset(path) as saved:
        np.testing.assert_array_equal(saved.flux.values, dense)


def test_postprocessing_uses_retained_flux_and_keeps_flux_time_labels() -> None:
    grid = {"lat": [50.0], "lon": [-2.0, -1.0]}
    flat = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=grid)
    times = np.array(["2019-01-01", "2019-02-01"], dtype="datetime64[ns]")
    retained = xr.DataArray(
        [[[2.0, -3.0]], [[4.0, -5.0]]],
        dims=("time", "lat", "lon"), coords={"time": times, **grid},
        attrs={"units": "mol/m2/s", "time_period": "monthly"},
    )
    basis = BasisFunctions.from_flat_basis(flat, retained)
    stats = xr.Dataset({"x_posterior_mean": ("state", [10.0, 20.0])},
                       coords={"state": basis.operator.basis_matrix.state})
    presentation = (retained * 100).rename(time="flux_time")
    result = reconstruct_flux_stats(basis, presentation, stats, report_flux_on_inversion_grid=False)
    assert "flux_time" in result.dims and "time" not in result.dims
    np.testing.assert_array_equal(result.flux_time.values, times)
    np.testing.assert_array_equal(result.x_posterior_mean.values[:, 0, :], [[20, -60], [40, -100]])
    assert result.attrs.get("time_period") == "monthly"


def test_state_sample_time_does_not_align_with_flux_time() -> None:
    flat = xr.DataArray([[1]], dims=("lat", "lon"), coords={"lat": [50.0], "lon": [-2.0]})
    flux = xr.DataArray(
        [[[2.0]], [[3.0]]], dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [50.0], "lon": [-2.0]},
    )
    basis = BasisFunctions.from_flat_basis(flat, flux)
    state = xr.DataArray(
        [[4.0, 5.0]], dims=("state", "time"),
        coords={"state": basis.operator.basis_matrix.state, "time": [10, 11]},
    )
    output = basis.state_to_flux(state)
    assert output.time.values.tolist() == [0, 1]
    assert output.state_time.values.tolist() == [10, 11]
    np.testing.assert_array_equal(
        output.transpose("time", "state_time", "lat", "lon").values[:, :, 0, 0],
        [[8, 10], [12, 15]],
    )
