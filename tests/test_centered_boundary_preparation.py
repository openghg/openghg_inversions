"""Centered boundary operands survive preparation without changing their identity."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.filters import pblh_min
from openghg_inversions.inversion_data import PreparedInputs, prepare_rhime_inputs_from_xarray
from openghg_inversions.inversion_inputs import make_inv_inputs
from tests.test_xarray_input_adapter import _basis_functions, _site_dataset


def _site():
    ds = _site_dataset("AAA", ["2020-01-01", "2020-01-01T03:00", "2020-01-01T12:00"])
    ds["H"] = xr.DataArray(np.ones((2, 3)), dims=("region", "time"),
                           coords={"region": [0, 1], "time": ds.time}, attrs={"units": "ppm"})
    ds["H_bc_anomaly"] = xr.DataArray(
        [[1., 3., 2.], [-2., 0., 4.]], dims=("bc_region", "time"),
        coords={"bc_region": ["north", "south"], "time": ds.time}, attrs={"units": "ppm"})
    ds["H_bc_mass"] = xr.ones_like(ds.H_bc_anomaly) * .25
    ds.H_bc_mass.attrs["units"] = "1"
    ds["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
    ds.attrs.update(boundary_representation="centered-v1", boundary_normalization="all-curtains")
    return ds.drop_vars(["release_lat", "release_lon"])


def test_arbitrary_basis_projects_both_operands_lazily():
    coords = {"lat": [0], "lon": [0], "height": [0], "time": pd.date_range("2020", periods=2)}
    ds = xr.Dataset(attrs={"boundary_representation": "centered-v1"})
    basis = xr.Dataset()
    for i, direction in enumerate("nesw"):
        for operand, units in [("anomaly", "ppm"), ("mass", "1")]:
            ds[f"bc_{operand}_{direction}"] = xr.DataArray(
                da.full((1, 1, 1, 2), i + 1., chunks=(1, 1, 1, 1)),
                dims=("lat", "lon", "height", "time"), coords=coords, attrs={"units": units})
        basis[f"bc_basis_{direction}"] = xr.DataArray(
            [[[[i + 2., -(i + 1.)]]]], dims=("lat", "lon", "height", "region"),
            coords={"lat": [0], "lon": [0], "height": [0], "region": ["a", "b"]})
    ds["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
    original = ds.copy(deep=True)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args)):
        result = bc_sensitivity({"AAA": ds}, basis)["AAA"]
    assert not tasks
    for name in ("H_bc_anomaly", "H_bc_mass"):
        np.testing.assert_allclose(result[name].transpose("bc_region", "time"), [[40., 40.], [-30., -30.]])
    xr.testing.assert_identical(result.bc_reference, ds.bc_reference)
    xr.testing.assert_identical(ds, original)








def test_gather_keeps_raw_regions_and_scalar_reference():
    ds = _site()
    original = ds.copy(deep=True)
    result = make_inv_inputs({"AAA": ds, "CCC": ds})
    assert result.sizes["bc_region"] == 2
    assert result.bc_reference.ndim == 0
    assert result.attrs["boundary_normalization"] == "all-curtains"
    assert not isinstance(result.indexes["bc_region"], pd.MultiIndex)
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("failure", ["mixed", "reference", "partial", "units"])
def test_gather_rejects_inconsistent_sites(failure):
    first, second = _site(), _site()
    if failure == "mixed":
        second = second.drop_vars(["H_bc_anomaly", "H_bc_mass", "bc_reference"])
        second.attrs = {}
    elif failure == "reference":
        second["bc_reference"] = second.bc_reference + 1
        second.bc_reference.attrs["units"] = "ppm"
    elif failure == "partial":
        second = second.drop_vars("H_bc_mass")
    else:
        second.H_bc_mass.attrs["units"] = "ppm"
    with pytest.raises(ValueError):
        make_inv_inputs({"AAA": first, "BBB": second})


def test_core_drops_common_nonfinite_rows():
    ds = _site()
    ds.H_bc_mass[0, 1] = np.inf
    result = make_inv_inputs({"AAA": ds})
    assert result.sizes["nmeasure"] == 2
    ds["bc_reference"] = xr.DataArray(np.nan, attrs={"units": "ppm"})
    with pytest.raises(ValueError, match="finite"):
        make_inv_inputs({"AAA": ds})










def test_core_materializes_shared_operands_once_without_mutating_inputs():
    from dask import delayed

    executions = []

    @delayed
    def shared_value():
        executions.append(True)
        return np.array(1.)

    scalar = da.from_delayed(shared_value(), shape=(), dtype=float)
    ds = _site()
    for name in ("H", "H_bc_anomaly", "H_bc_mass", "bc_reference", "mf", "mf_error"):
        ds[name] = (ds[name] * scalar).assign_attrs(ds[name].attrs)
    original = ds.copy(deep=True)
    gathered = make_inv_inputs({"AAA": ds})
    assert executions == [True]
    assert not hasattr(gathered.H_bc_anomaly.data, "__dask_graph__")
    assert hasattr(ds.H_bc_anomaly.data, "__dask_graph__")
    xr.testing.assert_identical(ds, original)







def test_gather_preserves_site_specific_transport_and_normalization_provenance():
    import json

    first, second = _site(), _site()
    for number, data in enumerate((first, second)):
        data.attrs.update(surface_transport='{"inventory": "integrated"}', surface_provider_revision="abc123",
                          surface_transport_provenance=f"site-{number}", boundary_mass_tolerance=number * 0.001)
    prepared = prepare_rhime_inputs_from_xarray({"AAA": first, "BBB": second}, basis_functions=_basis_functions())
    restored = PreparedInputs.from_datatree(prepared.to_datatree())
    facts = json.loads(restored.inv_inputs.attrs["surface_preparation_provenance"])
    assert facts["AAA"]["surface_transport_provenance"] == "site-0"
    assert facts["BBB"]["boundary_mass_tolerance"] == 0.001
    assert "surface_transport_provenance" not in restored.inv_inputs.attrs




def test_supported_selection_keeps_scalar_reference_and_shared_operand_rows():
    ds = _site()
    ds["PBLH"] = ("time", [300., 100., 400.])
    original = ds.copy(deep=True)
    selected = pblh_min(ds)
    assert selected.bc_reference.ndim == 0
    xr.testing.assert_identical(selected.bc_reference, ds.bc_reference)
    for name in ("mf", "H_bc_anomaly", "H_bc_mass"):
        xr.testing.assert_equal(selected[name], ds[name].isel(time=[0, 2]))
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("lazy", [False, True])
def test_sparse_centered_operands_select_dense_sensitivity_rows(lazy):
    ds = _site()
    ds.H_bc_mass[0, 1] = np.inf
    expected = make_inv_inputs({"AAA": ds})
    for name in ("H_bc_anomaly", "H_bc_mass"):
        ds[name].data = sparse.COO.from_numpy(ds[name].data)
        if lazy:
            ds[name] = ds[name].chunk({"time": 1})
    original = ds.copy(deep=True)
    result = make_inv_inputs({"AAA": ds})
    assert isinstance(result.H.data, np.ndarray)
    assert result.sizes["nmeasure"] == 2
    for name in ("H_bc_anomaly", "H_bc_mass"):
        assert isinstance(result[name].data, sparse.COO)
        np.testing.assert_allclose(result[name].data.todense(), expected[name].data)
    xr.testing.assert_identical(result.H, expected.H)
    xr.testing.assert_identical(ds, original)


def test_linked_centered_preparation_rejects_serialization_before_losing_operands(tmp_path):
    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from tests.test_rhime_co2_o2_preparation import _inputs

    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    native = inputs["co2_observations"]
    operands = xr.Dataset(
        {
            "H_bc_anomaly": (("bc_region", "nmeasure"), [[1., 2.]], {"units": "ppm"}),
            "H_bc_mass": (("bc_region", "nmeasure"), [[.5, .5]], {"units": "1"}),
            "bc_reference": xr.DataArray(400., attrs={"units": "ppm"}),
        },
        coords={"bc_region": ["north"], "nmeasure": native.indexes[native.dims[0]].to_numpy()},
        attrs={"boundary_representation": "centered-v1"},
    )
    prepared = prepare_co2_o2_inputs(**inputs, boundary_operands={"co2": operands})
    assert "co2" in prepared.boundary_operands
    with pytest.raises(ValueError, match="Centered.*serialization is not yet available"):
        prepared.to_datatree()
    path = tmp_path / "centered.nc"
    with pytest.raises(ValueError, match="Centered.*serialization is not yet available"):
        prepared.save(path)
    assert not path.exists()
