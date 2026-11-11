"""Tests for eager region-map reading and the shared NetCDF backend fallback."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions import utils
from openghg_inversions.basis.support import load_basis_support_map
from openghg_inversions.region_maps import load_country_dataset, load_region_map

_REAL_OPEN_DATASET = xr.open_dataset


def _write_map(path: Path, *variables: str, names: bool = True) -> None:
    """Write a small integer region map with attributes on the file and each variable."""
    coords = {"lat": [50.0, 51.0], "lon": [0.0, 1.0, 2.0]}
    codes = np.array([[0, 1, 2], [2, 1, 0]], dtype="int32")
    data_vars = {
        name: xr.DataArray(codes + offset, dims=("lat", "lon"), attrs={"inner_region_label": 1, "long_name": name})
        for offset, name in enumerate(variables)
    }
    dataset = xr.Dataset(data_vars, coords=coords, attrs={"title": "test map", "inner_region_label": 9})
    if names:
        dataset["name"] = ("ncountries", ["OCEAN", "LAND", "OTHER"])
    dataset.to_netcdf(path)


def _fail_engines(monkeypatch: pytest.MonkeyPatch, *failing: str | None) -> list[str | None]:
    """Make selected xarray engines raise an HDF-style error, recording every attempt."""
    calls: list[str | None] = []

    def open_dataset(path, *args, **kwargs):
        engine = kwargs.get("engine")
        calls.append(engine)
        if engine in failing:
            raise RuntimeError("Unspecified error in H5DSget_num_scales")
        return _REAL_OPEN_DATASET(path, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_dataset)
    return calls


@pytest.fixture
def region_file(tmp_path: Path) -> Path:
    path = tmp_path / "outer_regions.nc"
    _write_map(path, "region")
    return path


def test_open_ds_falls_back_to_h5netcdf_with_warning(monkeypatch, region_file):
    calls = _fail_engines(monkeypatch, None)
    with pytest.warns(RuntimeWarning, match="Falling back to xarray engine 'h5netcdf'.*H5DSget_num_scales"):
        dataset = utils.open_ds(region_file)
    assert calls == [None, "h5netcdf"]
    assert "region" in dataset


def test_open_ds_reports_every_failed_engine(monkeypatch, region_file):
    _fail_engines(monkeypatch, None, "h5netcdf")
    with pytest.raises(OSError, match="default: RuntimeError.*h5netcdf: RuntimeError") as error:
        utils.open_ds(region_file)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_open_ds_missing_file_does_not_try_other_engines(monkeypatch, tmp_path):
    calls = _fail_engines(monkeypatch)
    with pytest.raises(FileNotFoundError):
        utils.open_ds(tmp_path / "missing.nc")
    assert calls == [None]


def test_region_map_variable_inference(tmp_path):
    both = tmp_path / "both.nc"
    _write_map(both, "country", "region")
    assert load_region_map(both).name == "country"
    assert load_region_map(both, "region").name == "region"

    region_only = tmp_path / "region.nc"
    _write_map(region_only, "region")
    assert load_region_map(region_only).name == "region"

    neither = tmp_path / "neither.nc"
    _write_map(neither, "basis")
    with pytest.raises(KeyError, match="country.*region"):
        load_region_map(neither)
    with pytest.raises(KeyError, match="'region'"):
        load_basis_support_map(neither, "region")


def test_region_map_merges_attrs_variable_first(region_file):
    regions = load_region_map(region_file, "region")
    assert regions.attrs["title"] == "test map"
    assert regions.attrs["inner_region_label"] == 1


@pytest.mark.parametrize(
    ("failing", "warning"),
    [
        ((), None),
        ((None,), "xarray engine 'h5netcdf'"),
        ((None, "h5netcdf"), "direct HDF5 reader 'h5py'"),
    ],
    ids=["default", "h5netcdf", "h5py"],
)
def test_region_map_is_the_same_from_every_reader(monkeypatch, region_file, failing, warning):
    expected = load_region_map(region_file, "region")
    _fail_engines(monkeypatch, *failing)
    if warning is None:
        regions = load_region_map(region_file, "region")
    else:
        with pytest.warns(RuntimeWarning, match=warning):
            regions = load_region_map(region_file, "region")

    xr.testing.assert_equal(regions, expected)
    assert regions.name == "region"
    assert regions.attrs["inner_region_label"] == 1
    assert regions.attrs["title"] == "test map"


@pytest.mark.parametrize("failing", [(), (None, "h5netcdf")], ids=["default", "h5py"])
def test_country_dataset_names_region_codes_country_on_every_reader(monkeypatch, region_file, failing):
    _fail_engines(monkeypatch, *failing)
    with pytest.warns(RuntimeWarning) if failing else nullcontext():
        dataset = load_country_dataset(region_file)
    assert "country" in dataset and "region" not in dataset
    assert list(dataset["name"].values) == ["OCEAN", "LAND", "OTHER"]
    assert dataset["country"].dims == ("lat", "lon")


def test_region_map_unreadable_by_every_reader(monkeypatch, tmp_path):
    path = tmp_path / "not_netcdf.nc"
    path.write_text("not a netcdf file")
    with pytest.raises(OSError, match="h5py"):
        load_region_map(path)


def test_region_map_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_region_map(tmp_path / "missing.nc")


def test_get_country_reads_region_only_file_and_rejects_missing_codes(tmp_path):
    region_only = tmp_path / "country_TEST.nc"
    _write_map(region_only, "region")
    country = utils.get_country("TEST", country_file=region_only)
    np.testing.assert_array_equal(country.country, load_region_map(region_only).values)
    assert list(country.name) == ["OCEAN", "LAND", "OTHER"]

    neither = tmp_path / "country_NONE.nc"
    _write_map(neither, "basis")
    with pytest.raises(KeyError):
        utils.get_country("NONE", country_file=neither)

