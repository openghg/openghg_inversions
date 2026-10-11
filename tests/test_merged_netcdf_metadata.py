"""Stable-line regressions for merged-data NetCDF metadata."""

import copy

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openghg.dataobjects import BoundaryConditionsData, FluxData

from openghg_inversions.hbmcmc.legacy_data import (
    _save_merged_data,
    datatree_to_fp_all,
    fp_all_to_datatree,
    load_merged_data,
    openghg_data_to_dataset,
)


@pytest.mark.parametrize("flag", [False, True, np.bool_(False), np.bool_(True)])
@pytest.mark.parametrize("engine", ["h5netcdf", "netcdf4"])
def test_merged_netcdf_metadata_roundtrip(tmp_path, monkeypatch, flag, engine):
    """Both backends preserve flags and omit absent units without input mutation."""
    scenario = xr.Dataset(
        {"mf": ("time", [1.0])},
        coords={"height": ("height", [500.0], {"units": None, "long_name": "height"})},
    )
    fp_all = {"TAC": scenario, ".split_by_sectors": flag}
    original_writer = xr.DataTree.to_netcdf
    monkeypatch.setattr(
        xr.DataTree,
        "to_netcdf",
        lambda self, *args, **kwargs: original_writer(self, *args, engine=engine, **kwargs),
    )

    _save_merged_data(fp_all, tmp_path, merged_data_name="prepared.nc")
    restored = load_merged_data(tmp_path, merged_data_name="prepared.nc")

    assert restored[".split_by_sectors"] is bool(flag)
    expected = scenario.copy()
    expected["height"].attrs.pop("units")
    xr.testing.assert_identical(restored["TAC"], expected)
    assert scenario["height"].attrs["units"] is None
    assert fp_all[".split_by_sectors"] is flag
    assert fp_all_to_datatree(fp_all).attrs[".split_by_sectors"] is flag


@pytest.mark.parametrize("flag", [False, True])
def test_existing_dotted_marker_remains_supported(flag):
    tree = xr.DataTree.from_dict({"scenarios": xr.DataTree.from_dict({"TAC": xr.Dataset()})})
    tree.attrs[".split_by_sectors"] = flag

    assert datatree_to_fp_all(tree)[".split_by_sectors"] is flag


def test_merged_saves_preserve_borrowed_flux_and_boundary_metadata(tmp_path):
    """NetCDF then Zarr saving leaves borrowed data usable for direct serialization."""
    time = np.array(["2019-01-01", "2019-01-02"], dtype="datetime64[ns]")
    flux_payload = da.ones((2, 1, 1), chunks=(1, 1, 1))
    boundary_payload = da.ones((2, 1), chunks=(1, 1))
    flux = FluxData(
        data=xr.Dataset(
            {"flux": (("time", "lat", "lon"), flux_payload, {"units": "mol m-2 s-1"})},
            coords={"time": time, "lat": [52.0], "lon": [1.0]},
            attrs={"title": "borrowed flux"},
        ),
        metadata={"data_type": "flux", "source": "inventory"},
    )
    boundary = BoundaryConditionsData(
        data=xr.Dataset(
            {f"vmr_{side}": (("time", "height"), boundary_payload, {"units": "ppb"}) for side in "nesw"},
            coords={"time": time, "height": [500.0]},
            attrs={"title": "borrowed boundary conditions"},
        ),
        metadata={"data_type": "boundary_conditions", "species": "ch4"},
    )
    scenario = xr.Dataset({"mf": ("time", [1.0, 2.0])}, coords={"time": time})
    fp_all = {"TAC": scenario, ".flux": {"inventory": flux}, ".bc": boundary}
    snapshots = [(item.data.copy(deep=False), copy.deepcopy(item.metadata)) for item in (flux, boundary)]

    for filename in ("merged.nc", "merged.zarr"):
        _save_merged_data(fp_all, tmp_path, merged_data_name=filename)
        restored = load_merged_data(tmp_path, merged_data_name=filename)
        for original, recovered, (dataset, metadata) in zip(
            (flux, boundary), (restored[".flux"]["inventory"], restored[".bc"]), snapshots, strict=True
        ):
            xr.testing.assert_identical(original.data, dataset)
            xr.testing.assert_identical(recovered.data, dataset)
            assert original.metadata == recovered.metadata == metadata
            adapted = openghg_data_to_dataset(original)
            assert adapted is not original.data
            for name in original.data.data_vars:
                assert adapted[name].data is original.data[name].data
                assert original.data[name].data is dataset[name].data

    flux.data.to_netcdf(tmp_path / "direct-flux.nc")
    with xr.open_dataset(tmp_path / "direct-flux.nc") as direct:
        xr.testing.assert_identical(direct, flux.data)
