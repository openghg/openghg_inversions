from collections import Counter, namedtuple
from pathlib import Path
import shutil
import tempfile
from importlib.metadata import version
from typing import Iterator
from unittest.mock import patch

import pytest
from openghg.retrieve import search
from openghg.standardise import standardise_surface, standardise_bc, standardise_flux, standardise_footprint
from openghg.types import ObjectStoreError
import xarray as xr


_raw_data_path = Path(".").resolve() / "tests/data/"


@pytest.fixture(scope="session")
def openghg_version():
    try:
        return tuple(map(int, version("openghg").split(".")))
    except ValueError:
        return (1000, 0, 0)


@pytest.fixture(scope="session")
def raw_data_path():
    return _raw_data_path


@pytest.fixture(scope="session")
def merged_data_dir():
    return Path(tempfile.gettempdir(), "openghg_inversions_testing_merged_data_dir")


@pytest.fixture(scope="session", autouse=True)
def using_zarr_store():
    try:
        current_version = tuple(int(x) for x in version("openghg").split("."))
    except ValueError:
        # assume tests are being run on devel
        return True
    return current_version >= (0, 8)


@pytest.fixture(scope="session")
def merged_data_file_name(using_zarr_store, openghg_version):
    if openghg_version >= (0, 13):
        return "merged_data_test_tac_combined_scenario_v13"
    elif using_zarr_store:
        return "merged_data_test_tac_combined_scenario_v8"
    else:
        return "merged_data_test_tac_combined_scenario"


@pytest.fixture(scope="session", autouse=True)
def add_frozen_merged_data(merged_data_dir, merged_data_file_name, using_zarr_store):
    """Copy merged data from tests/data to temporary merged_data_dir.

    Data created/frozen around 15 Apr, 2024.

    If the zarr backend is being used, we load the merged data with xarray then write to (zipped) zarr.
    Otherwise, if netCDF is being used, we copy the merged data directly.
    """
    merged_data_dir.mkdir(exist_ok=True)

    if using_zarr_store and not (merged_data_dir / (merged_data_file_name + ".zarr.zip")).exists():
        import zarr

        ds = xr.open_dataset(_raw_data_path / (merged_data_file_name + ".nc"))

        with zarr.ZipStore(merged_data_dir / (merged_data_file_name + ".zarr.zip"), mode="w") as store:
            ds.to_zarr(store)

        ds.to_zarr(merged_data_dir / (merged_data_file_name + "no_zip" + ".zarr"))

    elif not (merged_data_dir / (merged_data_file_name + ".nc")).exists():
        shutil.copy(_raw_data_path / (merged_data_file_name + ".nc"), merged_data_dir)


bc_basis_function_path = Path(".").resolve() / "bc_basis_functions"
countries_path = Path(".").resolve() / "countries"


@pytest.fixture
def europe_country_file(raw_data_path):
    """Provides path to the EUROPE countryfile"""
    return raw_data_path / "country_EUROPE.nc"


@pytest.fixture
def eastasia_country_file(raw_data_path):
    """Provides path to the EASTASIA countryfile"""
    return raw_data_path / "country_EASTASIA.nc"


@pytest.fixture
def country_ds(raw_data_path):
    """Provides EUROPE countryfile dataset"""
    ds = xr.load_dataset(raw_data_path / "country_EUROPE.nc")
    yield ds


@pytest.fixture
def country_ds_eastasia(raw_data_path):
    """Provides EUROPE countryfile dataset"""
    ds = xr.load_dataset(raw_data_path / "country_EASTASIA.nc")
    yield ds


@pytest.fixture(scope="session", autouse=True)
def session_config_mocker(using_zarr_store) -> Iterator[None]:
    if using_zarr_store:
        inversions_test_store_path = Path(tempfile.gettempdir(), "openghg_inversions_zarr_testing_store")
    else:
        inversions_test_store_path = Path(tempfile.gettempdir(), "openghg_inversions_testing_store")

    mock_config = {
        "object_store": {
            "inversions_tests": {"path": str(inversions_test_store_path), "permissions": "rw"},
        },
        "user_id": "test-id-123",
        "config_version": "2",
    }

    with patch("openghg.objectstore._local_store.read_local_config", return_value=mock_config):
        yield


# TEST DATA
TestData = namedtuple("TestData", ["func", "metadata", "path", "data_type"])
test_data_list = []

## Obs data
tac_obs_metadata = {
    "source_format": "openghg",
    "network": "decc",
    "site": "tac",
    "inlet": "185m",
    "instrument": "picarro",
}
tac_obs_data_path = _raw_data_path / "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc"
test_data_list.append(TestData(standardise_surface, tac_obs_metadata, tac_obs_data_path, "surface"))

mhd_obs_metadata = {
    "source_format": "openghg",
    "network": "agage",
    "site": "mhd",
    "inlet": "10m",
    "instrument": "gcmd",
    "calibration_scale": "WMO-x2004a",
}
mhd_obs_data_path = _raw_data_path / "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(TestData(standardise_surface, mhd_obs_metadata, mhd_obs_data_path, "surface"))

## BC data
bc_metadata = {"species": "ch4", "bc_input": "cams", "domain": "europe", "store": "inversions_tests"}
bc_data_path = _raw_data_path / "bc_ch4_europe_cams_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_bc, bc_metadata, bc_data_path, "boundary_conditions"))

## Footprint data
tac_footprints_metadata = {
    "site": "tac",
    "domain": "europe",
    "model": "name",
    "inlet": "185m",
    # "metmodel": "ukv",
}
tac_footprints_data_path = _raw_data_path / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(TestData(standardise_footprint, tac_footprints_metadata, tac_footprints_data_path, "footprints"))

mhd_footprints_metadata = {
    "site": "mhd",
    "domain": "europe",
    "model": "name",
    "inlet": "10m",
    "source_format": "paris",
    # "metmodel": "ukv",
}
mhd_footprints_data_path = _raw_data_path / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(TestData(standardise_footprint, mhd_footprints_metadata, mhd_footprints_data_path, "footprints"))

## Flux data
flux_metadata = {"species": "ch4", "source": "total-ukghg-edgar7", "domain": "europe"}
flux_data_path = _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_flux, flux_metadata, flux_data_path, "flux"))

flux_dim_shuffle_metadata = {"species": "ch4", "source": "total-ukghg-edgar7-shuffled", "domain": "europe"}
flux_dim_shuffled_data_path = (
    _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data_dim_shuffled.nc"
)
test_data_list.append(TestData(standardise_flux, flux_dim_shuffle_metadata, flux_dim_shuffled_data_path, "flux"))


@pytest.fixture(scope="session", autouse=True)
def session_object_store(session_config_mocker) -> None:
    """Add data to test object.

    Check first if there is enough data. A more specific
    check for the data necessary for testing is carried out
    in "test_conftest.py".

    This fixture depends on `sesson_config_mocker` to make sure
    that `session_config_mocker` runs first.
    """
    add_data = False  # flag, True if data needs to be added

    try:
        results = search(store="inversions_tests")
    except ObjectStoreError:
        add_data = True
    else:
        try:
            found_dtypes = results.results["data_type"].to_list()
        except KeyError:
            add_data = True
        else:
            add_data = Counter([x.data_type for x in test_data_list]) != Counter(found_dtypes)


    # check if there are four pieces of data in the object store
    # if not, add the missing data
    if add_data:
        for test_data in test_data_list:
            standardise_fn = test_data.func
            file_path = test_data.path
            metadata = test_data.metadata
            metadata["store"] = "inversions_tests"
            standardise_fn(filepath=file_path, **metadata)


@pytest.fixture(scope="session", autouse=True)
def session_ancilliary_files() -> None:
    # Add bc basis function file
    if not bc_basis_function_path.exists():
        bc_basis_function_path.mkdir()
    if not (bc_basis_function_path / "EUROPE").exists():
        (bc_basis_function_path / "EUROPE").mkdir()

    # copy basis file into default location if there isn't a file with the same name there
    if not (bc_basis_function_path / "EUROPE" / "NESW_EUROPE_2019.nc").exists():
        shutil.copy(
            (_raw_data_path / "bc_basis_NESW_EUROPE_2019.nc"),
            (bc_basis_function_path / "EUROPE" / "NESW_EUROPE_2019.nc"),
        )

    # Add country file
    if not countries_path.exists():
        countries_path.mkdir()

    # copy country file into default location if there isn't a file with the same name there
    if not (countries_path / "country_EUROPE.nc").exists():
        shutil.copy((_raw_data_path / "country_EUROPE.nc"), (countries_path / "country_EUROPE.nc"))


@pytest.fixture(scope="module")
def tac_ch4_data_args():
    data_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        # "met_model": "ukv",
        "averaging_period": ["1h"],
    }
    return data_args


@pytest.fixture(scope="module")
def mhd_and_tac_ch4_data_args():
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        # "met_model": "ukv",
        "averaging_period": ["1h", "1h"],
    }
    return data_args
