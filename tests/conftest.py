from pathlib import Path
import tempfile
from typing import Iterator
from unittest.mock import patch
from openghg.types import ObjectStoreError

import pytest

from openghg.standardise import standardise_surface, standardise_bc, standardise_flux, standardise_footprint
from openghg.retrieve import search

raw_data_path = Path().resolve() / "tests/data/"
inversions_test_store_path = Path(tempfile.gettempdir(), "openghg_inversions_testing_store")


@pytest.fixture(scope="session", autouse=True)
def session_config_mocker() -> Iterator[None]:
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
obs_metadata = {
    "source_format": "openghg",
    "network": "decc",
    "site": "tac",
    "inlet": "185m",
    "instrument": "picarro",
}
bc_metadata = {"species": "ch4", "bc_input": "cams", "domain": "europe", "store": "inversions_tests"}
footprints_metadata = {
    "site": "tac",
    "domain": "europe",
    "model": "name",
    "inlet": "185m",
    "metmodel": "ukv",
}
flux_metadata = {"species": "ch4", "source": "total-ukghg-edgar7", "domain": "europe"}

obs_data_path = raw_data_path / "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc"
bc_data_path = raw_data_path / "bc_ch4_europe_cams_2019-01-01_2019-12-31_data.nc"
footprints_data_path = raw_data_path / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
flux_data_path = raw_data_path / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"

data_info = {
    "surface": [standardise_surface, obs_metadata, obs_data_path],
    "boundary_conditions": [standardise_bc, bc_metadata, bc_data_path],
    "emissions": [standardise_flux, flux_metadata, flux_data_path],
    "footprints": [standardise_footprint, footprints_metadata, footprints_data_path],
}


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
        found_dtypes = []
    else:
        add_data = results.results.shape[0] != 4
        try:
            found_dtypes = results.results["data_type"].to_list()
        except KeyError:
            found_dtypes = []

    # check if there are four pieces of data in the object store
    # if not, add the missing data
    if add_data:
        to_add = set(["surface", "boundary_conditions", "emissions", "footprints"]) - set(found_dtypes)

        for dtype in to_add:
            standardise_fn = data_info[dtype][0]
            file_path = data_info[dtype][2]
            metadata = data_info[dtype][1]
            metadata["store"] = "inversions_tests"
            standardise_fn(file_path, **metadata)
