from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

from filelock import FileLock, Timeout
import numpy as np
import pytest
import xarray as xr

from openghg.retrieve import search


_SATELLITE_TIMES = np.array(
    ["2016-01-01T14:59:12.500", "2016-01-01T16:40:42", "2016-01-01T16:43:46"],
    dtype="datetime64[ns]",
)


def _test_fixture_module():
    """Return pytest's loaded test fixture module."""
    for module in sys.modules.values():
        if getattr(module, "_TEST_STORE_DIR_NAME", None) == "openghg_inversions_testing_store":
            return module
    raise AssertionError("Could not find loaded tests/conftest.py module")


def test_openghg_test_store_rebuilds_unmarked_store(tmp_path):
    """The shared test store uses an explicit marker instead of OpenGHG search."""
    test_fixtures = _test_fixture_module()
    source_path = tmp_path / "source.nc"
    source_path.write_text("fake data")
    store_path = tmp_path / "ground"
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
    store_config = {"path": str(store_path), "permissions": "r"}

    test_fixtures._populate_test_store("inversions_tests", store_config, [fake_data])

    assert standardise_calls == [(source_path, {"species": "ch4", "store": "inversions_tests"})]
    assert not stale_file.exists()
    assert store_config["permissions"] == "r"
    assert store_path.with_suffix(".ready").exists()

    test_fixtures._populate_test_store("inversions_tests", store_config, [fake_data])

    assert len(standardise_calls) == 1


def test_input_store_root_is_shared_with_workers_and_isolated_between_runs(tmp_path):
    test_fixtures = _test_fixture_module()
    run_path = tmp_path / "run"
    roots = []
    for worker in (None, "gw0", "gw1"):
        base = run_path if worker is None else run_path / f"popen-{worker}"
        base.mkdir(parents=True, exist_ok=True)
        config = SimpleNamespace() if worker is None else SimpleNamespace(workerinput={"workerid": worker})
        factory = SimpleNamespace(getbasetemp=lambda: base)
        roots.append(test_fixtures.user_data_path.__wrapped__(factory, SimpleNamespace(config=config)))

    assert len(set(roots)) == 1
    other_run = tmp_path / "other"
    other_run.mkdir()
    other_factory = SimpleNamespace(getbasetemp=lambda: other_run)
    other_root = test_fixtures.user_data_path.__wrapped__(
        other_factory, SimpleNamespace(config=SimpleNamespace())
    )
    assert other_root != roots[0]


def test_failed_input_store_setup_retries_without_touching_other_family(tmp_path):
    test_fixtures = _test_fixture_module()
    other_store = tmp_path / "satellite"
    other_store.mkdir()
    (other_store / "input").write_text("preserve")
    other_ready = other_store.with_suffix(".ready")
    other_ready.touch()
    store_path = tmp_path / "ground"
    store_config = {"path": str(store_path), "permissions": "r"}
    calls = []

    def standardise_fake(filepath, **metadata):
        assert not store_path.exists()
        store_path.mkdir()
        (store_path / "input").write_text("new input")
        calls.append(filepath)
        if len(calls) == 1:
            raise RuntimeError("standardisation failed")

    manifest = [test_fixtures.TestData(standardise_fake, {}, tmp_path / "source.nc", "surface")]
    with pytest.raises(RuntimeError, match="standardisation failed"):
        test_fixtures._populate_test_store("inversions_tests", store_config, manifest)

    assert store_config["permissions"] == "r"
    assert not store_path.with_suffix(".ready").exists()
    test_fixtures._populate_test_store("inversions_tests", store_config, manifest)
    assert len(calls) == 2
    assert store_config["permissions"] == "r"
    assert store_path.with_suffix(".ready").exists()
    assert (other_store / "input").read_text() == "preserve"
    assert other_ready.exists()


def test_concurrent_input_store_setup_runs_standardisation_once(tmp_path):
    test_fixtures = _test_fixture_module()
    store_path = tmp_path / "ground"
    started = Event()
    release = Event()
    calls = []

    def standardise_fake(filepath, **metadata):
        calls.append(filepath)
        store_path.mkdir()
        started.set()
        assert release.wait(timeout=5)

    manifest = [test_fixtures.TestData(standardise_fake, {}, tmp_path / "source.nc", "surface")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            test_fixtures._populate_test_store,
            "inversions_tests",
            {"path": str(store_path), "permissions": "r"},
            manifest,
        )
        try:
            assert started.wait(timeout=5)
            second = pool.submit(
                test_fixtures._populate_test_store,
                "inversions_tests",
                {"path": str(store_path), "permissions": "r"},
                manifest,
            )
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls == [tmp_path / "source.nc"]


def test_dead_input_store_owner_releases_lock_and_partial_store_is_rebuilt(tmp_path):
    test_fixtures = _test_fixture_module()
    store_path = tmp_path / "ground"
    script = """
from pathlib import Path
import sys
from filelock import FileLock
store = Path(sys.argv[1])
with FileLock(store.with_suffix('.lock')):
    store.mkdir()
    (store / 'partial').touch()
    print('locked', flush=True)
    sys.stdin.read()
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        with subprocess.Popen(
            [sys.executable, "-c", script, str(store_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                assert pool.submit(process.stdout.readline).result(timeout=10) == "locked\n"
                with pytest.raises(Timeout):
                    with FileLock(store_path.with_suffix(".lock"), timeout=0.05):
                        pass
            finally:
                process.kill()
                process.communicate(timeout=5)

    def standardise_fake(filepath, **metadata):
        assert not store_path.exists()
        store_path.mkdir()

    manifest = [test_fixtures.TestData(standardise_fake, {}, tmp_path / "source.nc", "surface")]
    test_fixtures._populate_test_store(
        "inversions_tests", {"path": str(store_path), "permissions": "r"}, manifest
    )
    assert store_path.with_suffix(".ready").exists()
    assert not (store_path / "partial").exists()


def test_default_session_fixture(session_config_mocker):
    """The default session fixture should mock `read_local_config`
    so that the object store path is set to:

    <pytest run>/openghg_inversions_test_data/openghg_inversions_testing_store
    """
    from openghg.objectstore._local_store import read_local_config

    conf = read_local_config()

    assert conf
    assert "inversions_tests" in conf["object_store"]

    assert "openghg_inversions_testing_store" in conf["object_store"]["inversions_tests"]["path"]
    assert conf["object_store"]["inversions_tests"]["permissions"] == "r"
    assert conf["object_store"]["inversions_satellite_tests"]["permissions"] == "r"
    assert (
        conf["object_store"]["inversions_satellite_tests"]["path"]
        != conf["object_store"]["inversions_tests"]["path"]
    )


def test_obs_in_test_store(openghg_test_store):
    results = search(site="tac", species="ch4", data_type="surface", store="inversions_tests")
    assert results


def test_footprints_in_test_store(openghg_test_store):
    results = search(site="tac", data_type="footprints", store="inversions_tests")
    assert results


def _assert_january_bc_coverage(domain, year, store):
    results = search(species="ch4", data_type="boundary_conditions", domain=domain, store=store)
    assert results
    boundary_conditions = results.retrieve_all()
    expected_coverage = {
        "start_date": f"{year}-01-01 00:00:00+00:00",
        "end_date": f"{year}-01-31 23:59:59+00:00",
        "time_period": "1 month",
    }
    for key, value in expected_coverage.items():
        assert boundary_conditions.metadata[key] == value
        assert boundary_conditions.data.attrs[key] == value


def test_bc_in_test_store(openghg_test_store):
    """The single-time ground BC fixture retains its monthly source coverage."""
    _assert_january_bc_coverage("europe", 2019, "inversions_tests")


def test_satellite_bc_in_test_store(openghg_satellite_test_store):
    """The satellite BC fixture retains its distinct January coverage."""
    _assert_january_bc_coverage("southamerica", 2016, "inversions_satellite_tests")


def test_satellite_footprint_fixture_retains_matched_soundings(raw_data_path):
    """A small time subset keeps the original footprint and particle-boundary layout."""
    satellite = raw_data_path / "satellite"
    with (
        xr.open_dataset(satellite / "footprints" / "GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc") as footprint,
        xr.open_dataset(satellite / "column" / "gosat-fts_gosat_20160101_ch4-column.nc") as column,
    ):
        assert dict(footprint.sizes) == {"lat": 357, "lon": 190, "time": 3, "height": 20, "lev": 1}
        dimensions = {"fp": ("lat", "lon", "time")}
        dimensions.update({
            name: ("time",)
            for name in ("temperature", "pressure", "wind_speed", "wind_direction", "PBLH", "release_lon", "release_lat")
        })
        dimensions.update({
            f"particle_locations_{side}": ("height", axis, "time")
            for side, axis in (("n", "lon"), ("e", "lat"), ("s", "lon"), ("w", "lat"))
        })
        assert set(footprint.data_vars) == set(dimensions)
        for name, dims in dimensions.items():
            assert footprint[name].dims == dims
            assert footprint[name].dtype == np.dtype("float32")
        np.testing.assert_array_equal(footprint.time, _SATELLITE_TIMES)
        np.testing.assert_array_equal(footprint.height, np.arange(500, 20_000, 1_000))
        np.testing.assert_allclose(footprint.lat.values[[0, -1]], [-60.981, 22.323])
        np.testing.assert_allclose(footprint.lon.values[[0, -1]], [-91.329, -24.801])
        assert footprint.fp.attrs["units"] == "(mol/mol)/(mol/m2/s)"
        matched_column = column.sel(time=footprint.time)
        # Original NAME release coordinates were quantized to four decimal places.
        np.testing.assert_allclose(footprint.release_lat, matched_column.lat, rtol=0, atol=1e-4)
        np.testing.assert_allclose(footprint.release_lon, matched_column.lon, rtol=0, atol=1e-4)


def test_satellite_inputs_in_test_store(openghg_satellite_test_store, raw_data_path):
    """Bundled column, footprint and flux files remain compatible with OpenGHG ingestion."""
    products = {}
    for data_type in ("column", "footprints", "flux"):
        results = search(data_type=data_type, domain="southamerica", store="inversions_satellite_tests")
        assert results
        products[data_type] = results.retrieve_all().data
        assert products[data_type].sizes["time"] > 0
    footprint = products["footprints"]
    np.testing.assert_array_equal(footprint.time, _SATELLITE_TIMES)
    np.testing.assert_array_equal(products["column"].sel(time=footprint.time).time, _SATELLITE_TIMES)
    with xr.open_dataset(
        raw_data_path / "satellite" / "footprints" / "GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc"
    ) as source:
        for name in ("fp", *(f"particle_locations_{side}" for side in "nesw")):
            xr.testing.assert_allclose(footprint[name].transpose(*source[name].dims), source[name])


def test_flux_in_test_store(openghg_test_store):
    results = search(species="ch4", data_type="flux", store="inversions_tests")
    assert results
