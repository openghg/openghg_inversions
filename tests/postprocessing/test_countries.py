import h5py
import numpy as np
import pytest
import xarray as xr

import openghg_inversions._country_file as country_file_loader
from openghg_inversions.postprocessing.countries import Countries, CountryRegions, paris_regions_dict
from openghg_inversions.postprocessing._country_codes import CountryInfoList


def _write_minimal_country_file(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("lat", data=np.array([1.0, 2.0], dtype="float32"))
        h5.create_dataset("lon", data=np.array([3.0, 4.0], dtype="float32"))
        h5.create_dataset("country", data=np.array([[0.0, 1.0], [1.0, 0.0]]))
        h5.create_dataset("name", data=np.array([b"OCEAN", b"FRANCE"]))
        h5.create_dataset("country_code", data=np.array([b"OCEAN", b"FRA"]))


def test_countries_from_file_falls_back_to_h5netcdf(monkeypatch, europe_country_file):
    """Check country loading falls back when xarray's default backend hits an HDF error."""
    real_open_dataset = xr.open_dataset
    open_calls = []

    def open_dataset_with_default_failure(path, *args, **kwargs):
        engine = kwargs.get("engine", "default")
        open_calls.append(engine)

        if "engine" not in kwargs:
            raise OSError("[Errno -101] NetCDF: HDF error")

        return real_open_dataset(path, *args, **kwargs)

    monkeypatch.setattr(country_file_loader.xr, "open_dataset", open_dataset_with_default_failure)

    with pytest.warns(RuntimeWarning, match="Falling back to xarray engine 'h5netcdf'"):
        dataset = country_file_loader.load_country_dataset(europe_country_file)
    assert open_calls == ["default", "h5netcdf"]
    assert dataset.attrs[country_file_loader.COUNTRY_FILE_SELECTED_ENGINE_ATTR] == "h5netcdf"

    open_calls.clear()
    with pytest.warns(RuntimeWarning, match="Falling back to xarray engine 'h5netcdf'"):
        countries = Countries.from_file(domain="EUROPE", country_file=europe_country_file)

    assert open_calls == ["default", "h5netcdf"]
    assert len(countries.country_selections) == len(dataset.name)


def test_countries_from_file_falls_back_when_xarray_engines_fail(monkeypatch, tmp_path):
    """Check country files can be read directly when xarray/HDF5 scale decoding fails."""
    country_file = tmp_path / "country_TEST.nc"
    _write_minimal_country_file(country_file)

    def fail_open_dataset(*args, **kwargs):
        raise RuntimeError("Unspecified error in H5DSget_num_scales")

    monkeypatch.setattr(country_file_loader.xr, "open_dataset", fail_open_dataset)

    with pytest.warns(RuntimeWarning, match="direct HDF5 reader 'h5py'"):
        dataset = country_file_loader.load_country_dataset(country_file)

    assert dataset.attrs[country_file_loader.COUNTRY_FILE_SELECTED_ENGINE_ATTR] == "h5py"

    countries = Countries(dataset)

    assert list(countries.country_labels) == ["OCEAN", "FRANCE"]
    assert dict(countries.matrix.sizes) == {"lat": 2, "lon": 2, "country": 2}


def test_country_regions_missing_check():
    paris_regions_countries = CountryInfoList(
        [
            "AUT",
            "BEL",
            "CHE",
            "CZE",
            "DEU",
            "DNK",
            "ESP",
            "FRA",
            "GBR",
            "HRV",
            "HUN",
            "IRL",
            "ITALY",
            "LUX",
            "NLD",
            "POLAND",
            "PRT",
            "SVK",
            "SVN",
        ]
    )

    paris_regions = CountryRegions(paris_regions_dict["europe"])

    # check 1: "ITALY" vs "ITA" and "POLAND" vs "POL" doesn't affect check
    missing = paris_regions.region_countries_missing_from(paris_regions_countries)

    assert "CW_EU" not in missing

    # check 2: omitting countries required by CW_EU definition flags missing countries
    missing = paris_regions.region_countries_missing_from(paris_regions_countries[3:])

    assert "CW_EU" in missing


def test_country_regions_align(country_ds):
    """Check that aligning country regions defined with alpha3 codes results in definitions with input names
    for EUROPE domain."""
    paris_regions = CountryRegions(paris_regions_dict["europe"])
    countries_list = CountryInfoList(country_ds.name.values)

    assert list(paris_regions.align(countries_list).to_dict()["BELUX"]) == ["BELGIUM", "LUXEMBOURG"]


def test_asia_paris_regions_include_operational_aggregations():
    assert paris_regions_dict["eastasia"] == {
        "EASTERN_ASIA": ["EChi1", "PRK", "KOR", "JPN"],
        "WMC": ["EChi2", "NChina", "WChina"],
        "WESTERN_JPN": ["WJP", "CJP"],
        "EASTERN_JPN": ["CJP", "NJP"],
        "CHN_EC": ["CHN_E", "CHN_C"],
        "CHN": ["CHN_E", "CHN_C", "CHN_W", "CHN_N"],
        "JPN_WC": ["JPN_W", "JPN_C"],
        "JPN": ["JPN_W", "JPN_C", "JPN_N"],
        "NEA": ["KOR", "PRK", "JPN_W", "JPN_C", "CHN_E", "CHN_C"],
        "NEA_C": ["KOR", "PRK", "JPN_W", "CHN_E"],
    }
    assert paris_regions_dict["centralasia"] == {
        "INDIA": ["INDIA-SOUTH", "INDIA-NORTH", "INDIA-EAST", "INDIA-WEST", "INDIA-JK", "INDIA-ANDAMAN"],
        "INDIA-noJK": ["INDIA-SOUTH", "INDIA-NORTH", "INDIA-EAST", "INDIA-WEST", "INDIA-ANDAMAN"],
        "INDIA-NS": ["INDIA-NORTH", "INDIA-SOUTH"],
        "INDIA-NSE": ["INDIA-NORTH", "INDIA-SOUTH", "INDIA-EAST"],
        "INDIA-NSW": ["INDIA-NORTH", "INDIA-SOUTH", "INDIA-WEST"],
        "INDIA-NSEW": ["INDIA-NORTH", "INDIA-SOUTH", "INDIA-EAST", "INDIA-WEST"],
    }


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_with_regions(country_code, country_ds, europe_country_file):
    """Check that country regions combine with countries correctly in EUROPE domain."""
    countries = Countries.from_file(
        domain="EUROPE",
        country_regions=paris_regions_dict["europe"],
        country_code=country_code,
        country_file=europe_country_file,
    )

    assert len(countries.country_selections) == len(country_ds.name) + len(paris_regions_dict["europe"])


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_with_regions_EASTASIA(country_code, country_ds_eastasia, eastasia_country_file):
    """Check that country regions combine with countries correctly in EASTASIA domain."""
    country_regions = {"EASTERN_ASIA": paris_regions_dict["eastasia"]["EASTERN_ASIA"]}
    countries = Countries.from_file(
        domain="EASTASIA",
        country_regions=country_regions,
        country_code=country_code,
        country_file=eastasia_country_file,
    )

    assert len(countries.country_selections) == len(country_ds_eastasia.name) + len(country_regions)


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_skips_regions_with_missing_country_codes(country_code, eastasia_country_file):
    """Check that regions with missing country names/codes are skipped instead of raising."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    countries = Countries.from_file(
        domain="EASTASIA",
        country_regions=country_regions,
        country_code=country_code,
        country_file=eastasia_country_file,
        drop_missing_regions=True,
    )

    assert "EASTERN_ASIA" not in countries.country_selections


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_warns_when_regions_dropped(country_code, eastasia_country_file):
    """Check that dropping regions due to missing country names/codes emits a warning."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    with pytest.warns(UserWarning, match="Dropping country regions with unmatched countries"):
        countries = Countries.from_file(
            domain="EASTASIA",
            country_regions=country_regions,
            country_code=country_code,
            country_file=eastasia_country_file,
            drop_missing_regions=True,
        )

    assert "EASTERN_ASIA" not in countries.country_selections


@pytest.mark.parametrize("country_code", ["alpha2", "alpha3", None])
def test_countries_matrix_raises_on_missing_regions_by_default(country_code, eastasia_country_file):
    """Check that missing countries in region definitions raise by default."""
    country_regions = {
        "EASTERN_ASIA": ["NOT_A_COUNTRY", "PRK", "KOR", "JPN"],
    }

    with pytest.raises(ValueError, match="Could not find the following countries needed for regions"):
        Countries.from_file(
            domain="EASTASIA",
            country_regions=country_regions,
            country_code=country_code,
            country_file=eastasia_country_file,
        )
