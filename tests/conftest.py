from collections import Counter
from collections.abc import Iterator
import getpass
from pathlib import Path
import shutil
import tempfile
from importlib.metadata import version
from unittest.mock import patch

import pytest
from openghg.retrieve import search
from openghg.types import ObjectStoreError
import xarray as xr
import zarr

from openghg_inversions.tutorial.data_helpers import add_test_data, test_data_list


_raw_data_path = Path(".").resolve() / "tests/data/"

@pytest.fixture(scope="session", autouse=True)
def user_data_path():
    _user_data_path = Path(tempfile.gettempdir()) / f"{getpass.getuser()}_openghg_inversions_test_data"
    _user_data_path.mkdir(exist_ok=True)
    return _user_data_path

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
def merged_data_dir(user_data_path):
    return user_data_path / "openghg_inversions_testing_merged_data_dir"


@pytest.fixture(scope="session")
def merged_data_file_name():
    return "merged_data_test_tac_combined_scenario_v14"


@pytest.fixture(scope="session", autouse=True)
def add_frozen_merged_data(merged_data_dir, merged_data_file_name):
    """Copy merged data from tests/data to temporary merged_data_dir.

    Data created/frozen around 15 Apr, 2024.

    If the zarr backend is being used, we load the merged data with xarray then write to (zipped) zarr.
    Otherwise, if netCDF is being used, we copy the merged data directly.
    """
    merged_data_dir.mkdir(exist_ok=True)

    if not (merged_data_dir / (merged_data_file_name + ".zarr.zip")).exists():

        ds = xr.open_dataset(_raw_data_path / (merged_data_file_name + ".nc"))

        with zarr.ZipStore(merged_data_dir / (merged_data_file_name + ".zarr.zip"), mode="w") as store:
            ds.to_zarr(store)

        ds.to_zarr(merged_data_dir / (merged_data_file_name + "no_zip" + ".zarr"))


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

@pytest.fixture
def southamerica_country_file(raw_data_path):
    """Provides path to the SOUTHAMERICA countryfile"""
    return raw_data_path /"satellite"/"country"/ "country_SOUTHAMERICA.nc"

@pytest.fixture(scope="session", autouse=True)
def session_config_mocker(user_data_path) -> Iterator[None]:
    inversions_test_store_path = user_data_path / "openghg_inversions_testing_store"

    mock_config = {
        "object_store": {
            "inversions_tests": {"path": str(inversions_test_store_path), "permissions": "rw"},
        },
        "user_id": "test-id-123",
        "config_version": "2",
    }

    with patch("openghg.objectstore._local_store.read_local_config", return_value=mock_config):
        yield


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
        add_test_data(store="inversions_tests")


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
def satellite_ch4_data_args():
    data_args = {
        "species" : "ch4",
        "sites": ['GOSAT-BRAZIL'], 
        "averaging_period": ["1H"],
        "start_date": "2016-01-01",
        "end_date": "2016-02-01",
        "platform": ["satellite"],
        "max_level": 17,
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["column"],
        "instrument": [None],
        "domain": "SOUTHAMERICA",
        "fp_height": ["column"],
        "fp_species": "ch4",
        "fp_model": None,
        "emissions_name": ["SWAMPS"],
        # "met_model": "ukv",
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
