"""Stable-line regressions for merged-data NetCDF metadata."""

from zipfile import ZipFile

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data.serialise import (
    _save_merged_data,
    datatree_to_fp_all,
    fp_all_to_datatree,
    load_merged_data,
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


@pytest.mark.parametrize("filename", ["merged.zarr", "merged.zarr.zip"])
def test_merged_zarr_roundtrip_preserves_lazy_data_and_zip_integrity(tmp_path, filename):
    """Directory and ZIP stores retain lazy payloads through format 2 writes."""
    scenario = xr.Dataset({"mf": ("time", da.arange(4, chunks=2))})

    _save_merged_data({"TAC": scenario}, tmp_path, merged_data_name=filename)

    if filename.endswith(".zip"):
        with ZipFile(tmp_path / filename) as archive:
            assert archive.testzip() is None
            assert ".zgroup" in archive.namelist()
    restored = load_merged_data(tmp_path, merged_data_name=filename)
    xr.testing.assert_equal(restored["TAC"].compute(), scenario.compute())
    assert isinstance(scenario.mf.data, da.Array)
