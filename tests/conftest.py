import json
from collections import namedtuple
from contextlib import contextmanager
import getpass
import os
from pathlib import Path
import shutil
import tempfile
import time
from importlib.metadata import version
from typing import Iterator
from types import MappingProxyType
from unittest.mock import patch

import pytest
from openghg.standardise import (
    standardise_surface,
    standardise_bc,
    standardise_flux,
    standardise_footprint,
    standardise_column,
)
import xarray as xr
import zarr

_raw_data_path = Path(".").resolve() / "tests/data/"
_TEST_STORE_DIR_NAME = "openghg_inversions_testing_store"
_TEST_STORE_READY_MARKER_NAME = "openghg_inversions_testing_store.ready.json"
_TEST_STORE_LOCK_NAME = "openghg_inversions_testing_store.lock"


@pytest.fixture(scope="session")
def user_data_path():
    """Shared OpenGHG test-data root.

    The object store under this path is expensive to populate and is treated as
    read-only by tests after setup.
    """
    _user_data_path = Path(tempfile.gettempdir()) / f"{getpass.getuser()}_openghg_inversions_test_data"
    _user_data_path.mkdir(exist_ok=True)
    return _user_data_path


@pytest.fixture(scope="session")
def writable_data_path(tmp_path_factory):
    """Per-run root for test artifacts that are written during pytest."""
    return tmp_path_factory.mktemp("openghg_inversions_test_data")


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
def merged_data_dir(writable_data_path, merged_data_file_name):
    merged_data_dir = writable_data_path / "openghg_inversions_testing_merged_data_dir"
    merged_data_dir.mkdir(exist_ok=True)

    if not (merged_data_dir / (merged_data_file_name + ".zarr.zip")).exists():
        with xr.open_dataset(_raw_data_path / (merged_data_file_name + ".nc")) as ds:
            with zarr.ZipStore(merged_data_dir / (merged_data_file_name + ".zarr.zip"), mode="w") as store:
                ds.to_zarr(store)

            ds.to_zarr(merged_data_dir / (merged_data_file_name + "no_zip" + ".zarr"))

    return merged_data_dir


@pytest.fixture(scope="session")
def merged_data_file_name():
    return "merged_data_test_tac_combined_scenario_v14"


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
    return raw_data_path / "satellite" / "country" / "country_SOUTHAMERICA.nc"


@pytest.fixture(scope="session")
def session_config_mocker(user_data_path) -> Iterator[dict]:
    inversions_test_store_path = user_data_path / _TEST_STORE_DIR_NAME

    mock_config = {
        "object_store": {
            "inversions_tests": {"path": str(inversions_test_store_path), "permissions": "r"},
        },
        "user_id": "test-id-123",
        "config_version": "2",
    }

    with patch("openghg.objectstore._local_store.read_local_config", return_value=mock_config):
        yield mock_config


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

## Satellite Column data
satellite_gosat_obs_metadata = {
    "source_format": "openghg",
    "satellite": "gosat",
    "network": "gosat",
    "domain": "southamerica",
    "instrument": "tanso-fts",
    "species": "ch4",
}
satellite_gosat_obs_data_path = (
    _raw_data_path / "satellite" / "column" / "gosat-fts_gosat_20160101_ch4-column.nc"
)
test_data_list.append(
    TestData(standardise_column, satellite_gosat_obs_metadata, satellite_gosat_obs_data_path, "column")
)

## BC data
bc_metadata = {"species": "ch4", "bc_input": "cams", "domain": "europe", "store": "inversions_tests"}
bc_data_path = _raw_data_path / "bc_ch4_europe_cams_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_bc, bc_metadata, bc_data_path, "boundary_conditions"))

satellite_bc_metadata = {
    "species": "ch4",
    "bc_input": "cams",
    "domain": "southamerica",
    "store": "inversions_tests",
}
satellite_bc_data_path = _raw_data_path / "satellite" / "bc" / "ch4_SOUTHAMERICA_201601_CAMS-inversion.nc"
test_data_list.append(
    TestData(standardise_bc, satellite_bc_metadata, satellite_bc_data_path, "boundary_conditions")
)

## Footprint data
tac_footprints_metadata = {
    "site": "tac",
    "domain": "europe",
    "model": "name",
    "inlet": "185m",
    # "metmodel": "ukv",
}
tac_footprints_data_path = _raw_data_path / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(
    TestData(standardise_footprint, tac_footprints_metadata, tac_footprints_data_path, "footprints")
)

mhd_footprints_metadata = {
    "site": "mhd",
    "domain": "europe",
    "model": "name",
    "inlet": "10m",
    "source_format": "paris",
    # "metmodel": "ukv",
}
mhd_footprints_data_path = _raw_data_path / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
test_data_list.append(
    TestData(standardise_footprint, mhd_footprints_metadata, mhd_footprints_data_path, "footprints")
)

footprints_satellite_metadata = {
    "satellite": "GOSAT",
    "domain": "southamerica",
    "model": "NAME",
    "inlet": "column",
    "source_format": "acrg_org",
    "obs_region": "brazil",
    "species": "ch4",
}
footprints_satellite_data = (
    _raw_data_path / "satellite" / "footprints" / "GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc"
)
test_data_list.append(
    TestData(standardise_footprint, footprints_satellite_metadata, footprints_satellite_data, "footprints")
)

## Flux data
flux_metadata = {"species": "ch4", "source": "total-ukghg-edgar7", "domain": "europe"}
flux_data_path = _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
test_data_list.append(TestData(standardise_flux, flux_metadata, flux_data_path, "flux"))

flux_dim_shuffle_metadata = {"species": "ch4", "source": "total-ukghg-edgar7-shuffled", "domain": "europe"}
flux_dim_shuffled_data_path = (
    _raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data_dim_shuffled.nc"
)
test_data_list.append(
    TestData(standardise_flux, flux_dim_shuffle_metadata, flux_dim_shuffled_data_path, "flux")
)

flux_satellite_metadata = {"species": "ch4", "source": "SWAMPS", "domain": "southamerica"}
flux_satellite_datapath = (
    _raw_data_path / "satellite" / "flux" / "ch4_SOUTHAMERICA_2016_SWAMPS-v32-5_Saunois-Annual-Mean.nc"
)
test_data_list.append(TestData(standardise_flux, flux_satellite_metadata, flux_satellite_datapath, "flux"))


def _test_store_path(user_data_path: Path) -> Path:
    """Return the shared OpenGHG object-store path used by tests."""
    return user_data_path / _TEST_STORE_DIR_NAME


def _test_store_ready_marker_path(user_data_path: Path) -> Path:
    """Return the marker path used to identify a populated shared test store."""
    return user_data_path / _TEST_STORE_READY_MARKER_NAME


def _test_store_signature() -> str:
    """Return a stable signature for the source data used to populate the store."""
    entries = []
    for test_data in test_data_list:
        file_stat = test_data.path.stat()
        entries.append(
            {
                "data_type": test_data.data_type,
                "standardise_function": test_data.func.__name__,
                "metadata": test_data.metadata,
                "path": str(test_data.path),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
            }
        )

    return json.dumps(entries, indent=2, sort_keys=True)


def _test_store_is_marked_ready(user_data_path: Path) -> bool:
    """Return True when the ready marker matches the current fixture manifest."""
    if not _test_store_path(user_data_path).exists():
        return False

    marker_path = _test_store_ready_marker_path(user_data_path)
    try:
        return marker_path.read_text() == _test_store_signature()
    except OSError:
        return False


def _mark_test_store_ready(user_data_path: Path) -> None:
    """Write the ready marker after the shared test store has been populated."""
    _test_store_ready_marker_path(user_data_path).write_text(_test_store_signature())


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 120.0) -> Iterator[None]:
    """Use an atomic lock file to protect first-time shared fixture population."""
    start_time = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for fixture lock {lock_path}")
            time.sleep(0.1)

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def openghg_test_store(session_config_mocker, user_data_path) -> None:
    """Add data to test object.

    The shared test store is expensive to populate, so first-time setup is
    guarded by a cross-worker lock and an explicit ready marker. A more specific
    check for the data necessary for testing is carried out in
    "test_conftest.py".

    This fixture depends on `session_config_mocker` to make sure
    that `session_config_mocker` runs first.
    """
    if _test_store_is_marked_ready(user_data_path):
        return

    with _file_lock(user_data_path / _TEST_STORE_LOCK_NAME):
        if _test_store_is_marked_ready(user_data_path):
            return

        shutil.rmtree(_test_store_path(user_data_path), ignore_errors=True)
        _test_store_ready_marker_path(user_data_path).unlink(missing_ok=True)

        session_config_mocker["object_store"]["inversions_tests"]["permissions"] = "rw"
        try:
            for test_data in test_data_list:
                standardise_fn = test_data.func
                file_path = test_data.path
                metadata = dict(test_data.metadata)
                metadata["store"] = "inversions_tests"
                standardise_fn(filepath=file_path, **metadata)
        finally:
            session_config_mocker["object_store"]["inversions_tests"]["permissions"] = "r"

        _mark_test_store_ready(user_data_path)


@pytest.fixture(scope="session")
def default_bc_basis_directory(writable_data_path, raw_data_path) -> Path:
    """Create a worker-local default BC basis directory for real-data tests."""
    bc_basis_function_path = writable_data_path / "bc_basis_functions"
    domain_dir = bc_basis_function_path / "EUROPE"
    domain_dir.mkdir(parents=True, exist_ok=True)

    # copy basis file into default location if there isn't a file with the same name there
    if not (domain_dir / "NESW_EUROPE_2019.nc").exists():
        shutil.copy(
            (raw_data_path / "bc_basis_NESW_EUROPE_2019.nc"),
            (domain_dir / "NESW_EUROPE_2019.nc"),
        )

    return bc_basis_function_path


@pytest.fixture(scope="module")
def tac_ch4_data_args(openghg_test_store):
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
def satellite_ch4_data_args(openghg_test_store):
    data_args = {
        "species": "ch4",
        "sites": ["GOSAT-BRAZIL"],
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
def mhd_and_tac_ch4_data_args(openghg_test_store):
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


@pytest.fixture(scope="module")
def mhd_and_tac_fp_data(mhd_and_tac_ch4_data_args, default_bc_basis_directory):
    from openghg_inversions.basis import basis_functions_wrapper
    from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer

    fp_all, *_ = data_processing_surface_notracer(**mhd_and_tac_ch4_data_args)

    basis_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 100,
        "use_bc": True,
        "basis_algorithm": "weighted",
        "bc_basis_case": "NESW",
        "bc_basis_directory": default_bc_basis_directory,
    }

    fp_data = basis_functions_wrapper(fp_all, **basis_args)

    return MappingProxyType(fp_data)  # read-only
