"""Runtime attribution observes pytest without changing graph or lock behavior."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

from filelock import BaseFileLock, FileLock, Timeout
import pytest

from tests import ci_runtime as runtime


@pytest.fixture(autouse=True)
def exclude_selftests_from_active_monitor(monkeypatch, request):
    # Synthetic lock delays and fake configs must not affect an enclosing report.
    monkeypatch.setattr(FileLock, "acquire", BaseFileLock.acquire)
    if (state := request.config.stash.get(runtime._STATE, None)) is not None:
        monkeypatch.setitem(state, "call_settings", {})


@pytest.fixture
def lock_stats(monkeypatch, tmp_path):
    compiledir = tmp_path / "compiled"
    compiledir.mkdir()
    monkeypatch.setitem(sys.modules, "pytensor.configdefaults", SimpleNamespace(
        config=SimpleNamespace(compiledir=compiledir),
    ))
    return compiledir / ".lock", {
        "count": 0, "total_acquire_elapsed_s": 0.0, "max_acquire_elapsed_s": 0.0, "timeout_count": 0,
    }


def test_plugin_installs_before_conftest_without_scientific_imports(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        "import sys\nfrom filelock import BaseFileLock, FileLock\n"
        "assert FileLock.acquire is not BaseFileLock.acquire\n"
        "assert not {'numpy', 'pytensor', 'pymc', 'dask', 'threadpoolctl'} & sys.modules.keys()\n"
    )
    (tmp_path / "test_probe.py").write_text("def test_probe():\n    pass\n")
    destination = tmp_path / "runtime.json"
    environment = {**os.environ, runtime._ENV: str(destination), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                   "PYTHONPATH": str(Path(__file__).parents[1])}
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.ci_runtime", "-c", str(tmp_path / "pytest.ini"),
         "--confcutdir", str(tmp_path), "-q", str(tmp_path / "test_probe.py")],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(destination.read_text())
    assert report["workers"] == {}
    assert report["controller"]["pytensor"] is None and report["controller"]["dask"] is None
    assert report["controller"]["pytensor_c_lock"]["count"] == 0
    assert "upper bound on waiting" in result.stdout


def test_lock_monitor_delegates_once_and_preserves_returns_and_errors(monkeypatch, lock_stats):
    path, stats = lock_stats
    proxy = object()
    timeout = Timeout(str(path))
    failure = ValueError("invalid acquisition")
    outcomes = iter([proxy, timeout, failure])
    calls = []

    def original(lock, *args, **kwargs):
        calls.append((lock, args, kwargs))
        outcome = next(outcomes) if lock.lock_file == str(path) else proxy
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(FileLock, "acquire", original)
    clock = iter([1.0, 1.1, 2.0, 2.4, 3.0, 3.2])
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(clock))
    runtime._monitor_locks(monkeypatch, stats)
    unrelated = FileLock(path.parent / "numba" / ".lock")
    assert unrelated.acquire(timeout=0) is proxy
    lock = FileLock(path)
    assert lock.acquire(7, poll_interval=0.02, blocking=False) is proxy
    with pytest.raises(Timeout) as caught:
        lock.acquire(timeout=0)
    assert caught.value is timeout
    with pytest.raises(ValueError) as caught:
        lock.acquire(timeout=-1)
    assert caught.value is failure
    assert calls == [(unrelated, (), {"timeout": 0}),
                     (lock, (7,), {"poll_interval": 0.02, "blocking": False}),
                     (lock, (), {"timeout": 0}), (lock, (), {"timeout": -1})]
    assert stats == {"count": 3, "total_acquire_elapsed_s": pytest.approx(0.7),
                     "max_acquire_elapsed_s": pytest.approx(0.4), "timeout_count": 1}


def test_real_held_lock_delay_is_observed_without_changing_exclusion(monkeypatch, lock_stats):
    path, stats = lock_stats
    started, acquired = Event(), Event()

    def contender():
        started.set()
        with FileLock(path).acquire(timeout=2, poll_interval=0.005):
            acquired.set()

    owner = FileLock(path)
    with owner:
        runtime._monitor_locks(monkeypatch, stats)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(contender)
            try:
                assert started.wait(1)
                assert not acquired.wait(0.05)
            finally:
                owner.release()
            pending.result(timeout=2)
    assert acquired.is_set() and stats["count"] == 1
    assert stats["total_acquire_elapsed_s"] >= 0.01


def test_lock_monitor_does_not_initialize_a_lazy_pytensor_cache(monkeypatch, lock_stats):
    path, stats = lock_stats

    class Config:
        def __init__(self):
            self._config_var_dict = {"compiledir": SimpleNamespace()}

        @property
        def compiledir(self):
            pytest.fail("Diagnostics initialized PyTensor's unused C cache")

    monkeypatch.setitem(sys.modules, "pytensor.configdefaults", SimpleNamespace(config=Config()))
    runtime._monitor_locks(monkeypatch, stats)
    with FileLock(path):
        pass
    assert stats["count"] == 0


def test_call_settings_capture_fixture_precision_and_deduplicate(monkeypatch):
    config = SimpleNamespace(stash=pytest.Stash())
    config.stash[runtime._STATE] = {"call_settings": {}}
    settings = SimpleNamespace(floatX="float64", warn_float64="raise", mode="FAST_RUN", cxx="c++")
    monkeypatch.setitem(sys.modules, "pytensor.configdefaults", SimpleNamespace(config=settings))
    for nodeid, precision in (("first", "float64"), ("second", "float32"), ("third", "float64")):
        settings.floatX = precision
        runtime.pytest_runtest_call(SimpleNamespace(config=config, nodeid=nodeid))
    records = list(config.stash[runtime._STATE]["call_settings"].values())
    assert [(record["settings"]["floatX"], record["test_count"], record["example_nodeid"])
            for record in records] == [("float64", 2, "first"), ("float32", 1, "second")]


def test_snapshot_uses_loaded_runtime_and_existing_c_cache(monkeypatch, lock_stats):
    path, stats = lock_stats
    pools = [{"internal_api": "openblas", "num_threads": 2}]
    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(threadpool_info=lambda: pools))
    monkeypatch.setitem(sys.modules, "pytensor.configdefaults", SimpleNamespace(config=SimpleNamespace(
        compiledir=path.parent, base_compiledir=path.parent.parent, floatX="float64", cxx="/usr/bin/c++",
        mode="FAST_RUN", cmodule__debug=False,
    )))
    monkeypatch.setitem(sys.modules, "pytensor.link.c.cmodule", SimpleNamespace(
        _module_cache=SimpleNamespace(stats=[3, 4, 5]),
    ))
    monkeypatch.setitem(sys.modules, "dask.config", SimpleNamespace(get=lambda name, default: 2))
    monkeypatch.setitem(sys.modules, "dask.system", SimpleNamespace(CPU_COUNT=8))
    report = runtime._snapshot(SimpleNamespace(workerinput={"workerid": "gw0"}), stats)
    assert report["role"] == "worker" and report["worker_id"] == "gw0"
    assert report["pytensor"]["floatX"] == "float64"
    assert report["pytensor"]["compiledir"] == path.parent
    assert report["pytensor_c_cache"] == {"hits": 3, "loads": 4, "compiles": 5}
    assert report["threadpools"] is pools
    assert report["dask"] == {"num_workers": 2, "cpu_count": 8}


def test_controller_aggregates_worker_reports_and_restores_acquire(monkeypatch, tmp_path):
    destination = tmp_path / "reports" / "runtime.json"
    monkeypatch.setenv(runtime._ENV, str(destination))
    cleanups = []
    controller = SimpleNamespace(stash=pytest.Stash(), add_cleanup=cleanups.append)
    worker = SimpleNamespace(stash=pytest.Stash(), add_cleanup=cleanups.append,
                             workerinput={"workerid": "gw0"}, workeroutput={})
    original = FileLock.acquire
    runtime.pytest_load_initial_conftests(controller)
    runtime.pytest_load_initial_conftests(worker)
    monkeypatch.setattr(runtime, "_snapshot", lambda config, locks: {
        "worker_id": getattr(config, "workerinput", {}).get("workerid"), "snapshot": "actual process",
    })
    runtime.pytest_sessionfinish(SimpleNamespace(config=worker))
    assert not destination.exists()
    runtime.pytest_testnodedown(SimpleNamespace(
        config=controller, workeroutput=worker.workeroutput, gateway=SimpleNamespace(id="gw0"),
    ), None)
    runtime.pytest_testnodedown(SimpleNamespace(config=controller, gateway=SimpleNamespace(id="gw1")), "crashed")
    runtime.pytest_sessionfinish(SimpleNamespace(config=controller))
    report = json.loads(destination.read_text())
    assert report["controller"]["worker_id"] is None
    assert report["workers"]["gw0"] == worker.workeroutput["ogi_runtime"]
    assert report["workers"]["gw1"]["missing_report"] == "crashed"
    assert FileLock.acquire is original
    for cleanup in reversed(cleanups):
        cleanup()
