import sys

import pytest

from openghg.retrieve import search

pytestmark = pytest.mark.usefixtures("openghg_test_store")


def _test_fixture_module():
    """Return pytest's loaded test fixture module."""
    for module in sys.modules.values():
        if getattr(module, "_TEST_STORE_DIR_NAME", None) == "openghg_inversions_testing_store":
            return module
    raise AssertionError("Could not find loaded tests/conftest.py module")


def test_openghg_test_store_rebuilds_unmarked_store(monkeypatch, tmp_path):
    """The shared test store uses an explicit marker instead of OpenGHG search."""
    test_fixtures = _test_fixture_module()
    source_path = tmp_path / "source.nc"
    source_path.write_text("fake data")
    store_path = test_fixtures._test_store_path(tmp_path)
    store_path.mkdir()
    stale_file = store_path / "stale"
    stale_file.write_text("left over from partial setup")
    standardise_calls = []

    def standardise_fake(filepath, **metadata):
        standardise_calls.append((filepath, metadata))
        store_path.mkdir(exist_ok=True)

    fake_data = test_fixtures.TestData(
        standardise_fake,
        {"species": "ch4"},
        source_path,
        "surface",
    )
    mock_config = {"object_store": {"inversions_tests": {"permissions": "r"}}}
    monkeypatch.setattr(test_fixtures, "test_data_list", [fake_data])

    test_fixtures.openghg_test_store.__wrapped__(mock_config, tmp_path)

    assert standardise_calls == [(source_path, {"species": "ch4", "store": "inversions_tests"})]
    assert not stale_file.exists()
    assert mock_config["object_store"]["inversions_tests"]["permissions"] == "r"
    assert test_fixtures._test_store_is_marked_ready(tmp_path)

    test_fixtures.openghg_test_store.__wrapped__(mock_config, tmp_path)

    assert len(standardise_calls) == 1


def test_default_session_fixture():
    """The default session fixture should mock `read_local_config`
    so that the object store path is set to:

    <temp dir>/openghg_inversions_testing_store
    """
    from openghg.objectstore._local_store import read_local_config

    conf = read_local_config()

    assert conf
    assert "inversions_tests" in conf["object_store"]

    assert "openghg_inversions_testing_store" in conf["object_store"]["inversions_tests"]["path"]
    assert conf["object_store"]["inversions_tests"]["permissions"] == "r"


def test_obs_in_test_store():
    results = search(site="tac", species="ch4", data_type="surface", store="inversions_tests")
    assert results


def test_footprints_in_test_store():
    results = search(site="tac", data_type="footprints")
    assert results


def test_bc_in_test_store():
    results = search(species="ch4", data_type="boundary_conditions")
    assert results


def test_flux_in_test_store():
    results = search(species="ch4", data_type="flux")
    assert results
