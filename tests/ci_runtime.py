"""Opt-in pytest process diagnostics: -p tests.ci_runtime and OGI_TEST_RUNTIME_REPORT=file.json."""

from functools import wraps
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

from filelock import FileLock, Timeout
import pytest


_STATE = pytest.StashKey[dict]()
_ENV = "OGI_TEST_RUNTIME_REPORT"
_THREAD_ENV = (
    "DASK_NUM_WORKERS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "PYTENSOR_FLAGS",
)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_limits():
    membership = _read("/proc/self/cgroup") or ""
    group = next((line.split(":", 2)[2] for line in membership.splitlines() if line.startswith("0::")), "/")
    root = Path("/sys/fs/cgroup")
    current = root / group.lstrip("/")
    if not current.is_dir():
        current = root
    limits = {}
    while current.is_relative_to(root):
        value = _read(current / "cpu.max")
        if value is not None:
            limits[str(current / "cpu.max")] = value
        if current == root:
            break
        current = current.parent
    return {"membership": membership, "cpu_max": limits}


def _monitor_locks(patch, stats):
    original = FileLock.acquire

    @wraps(original)
    def acquire(lock, *args, **kwargs):
        config = getattr(sys.modules.get("pytensor.configdefaults"), "config", None)
        # An unevaluated ConfigParam must not create a C cache just for diagnostics.
        parameter = getattr(config, "_config_var_dict", {}).get("compiledir")
        compiledir = getattr(parameter, "val", None) if parameter is not None else getattr(config, "compiledir", None)
        if (Path(lock.lock_file).name != ".lock" or compiledir is None
                or os.path.abspath(lock.lock_file) != os.path.abspath(os.path.join(str(compiledir), ".lock"))):
            return original(lock, *args, **kwargs)
        stats["count"] += 1
        started = perf_counter()
        try:
            return original(lock, *args, **kwargs)
        except Timeout:
            stats["timeout_count"] += 1
            raise
        finally:
            elapsed = perf_counter() - started
            stats["total_acquire_elapsed_s"] += elapsed
            stats["max_acquire_elapsed_s"] = max(stats["max_acquire_elapsed_s"], elapsed)

    patch.setattr(FileLock, "acquire", acquire)


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config):
    if not (destination := os.environ.get(_ENV)):
        return
    patch = pytest.MonkeyPatch()
    stats = {"count": 0, "total_acquire_elapsed_s": 0.0, "max_acquire_elapsed_s": 0.0, "timeout_count": 0}
    early_config.stash[_STATE] = {
        "path": Path(destination), "patch": patch, "locks": stats, "workers": {}, "call_settings": {},
    }
    early_config.add_cleanup(patch.undo)
    _monitor_locks(patch, stats)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    state = item.config.stash.get(_STATE, None)
    config = getattr(sys.modules.get("pytensor.configdefaults"), "config", None)
    if state is None or config is None:
        return
    settings = {name: getattr(config, name, None) for name in ("floatX", "warn_float64", "mode", "cxx")}
    record = state["call_settings"].setdefault(tuple(settings.values()), {
        "settings": settings, "test_count": 0, "example_nodeid": item.nodeid,
    })
    record["test_count"] += 1


def _snapshot(config, locks):
    from threadpoolctl import threadpool_info

    worker_id = getattr(config, "workerinput", {}).get("workerid")
    cpuinfo = _read("/proc/cpuinfo") or ""
    model = next((line.partition(":")[2].strip() for line in cpuinfo.splitlines()
                  if line.startswith(("model name", "Hardware"))), None)
    pytensor = getattr(sys.modules.get("pytensor.configdefaults"), "config", None)
    dask = sys.modules.get("dask.config")
    cache = getattr(sys.modules.get("pytensor.link.c.cmodule"), "_module_cache", None)
    return {
        "role": "worker" if worker_id else "controller", "worker_id": worker_id, "pid": os.getpid(),
        "python": {"version": sys.version, "executable": sys.executable,
                   "build": platform.python_build(), "compiler": platform.python_compiler()},
        "cpu": {"model": model, "logical_count": os.cpu_count(),
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "cgroup": _cpu_limits()},
        "environment": {name: os.environ.get(name) for name in _THREAD_ENV},
        "threadpools": threadpool_info(),
        "pytensor": None if pytensor is None else {name: getattr(pytensor, name, None) for name in (
            "floatX", "warn_float64", "base_compiledir", "compiledir", "mode", "cxx",
            "cmodule__debug", "cmodule__remove_gxx_opt", "cmodule__compilation_warning",
            "cmodule__preload_cache", "cmodule__warn_no_version",
        )},
        "pytensor_c_cache": None if cache is None else dict(zip(("hits", "loads", "compiles"), cache.stats)),
        "dask": None if dask is None else {"num_workers": dask.get("num_workers", None),
                                           "cpu_count": getattr(sys.modules.get("dask.system"), "CPU_COUNT", None)},
        "pytensor_c_lock": dict(locks),
        "scope": "Session-end process state; excludes sampler/CLI child processes and temporary sampling limits.",
        "lock_scope": "Conftest/collection/test FileLock acquire elapsed; upper bound on waiting, not compile time.",
    }


def pytest_sessionfinish(session):
    config = session.config
    if (state := config.stash.get(_STATE, None)) is None:
        return
    state["patch"].undo()
    report = _snapshot(config, state["locks"])
    report["pytensor_call_settings"] = list(state["call_settings"].values())
    report["call_settings_scope"] = "Test call entry after fixtures; excludes changes inside tests and child processes."
    if hasattr(config, "workeroutput"):
        config.workeroutput["ogi_runtime"] = report
    else:
        state["report"] = {"controller": report, "workers": state["workers"]}
        state["path"].parent.mkdir(parents=True, exist_ok=True)
        state["path"].write_text(json.dumps(state["report"], indent=2, default=str) + "\n")


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    if (state := node.config.stash.get(_STATE, None)) is not None:
        state["workers"][node.gateway.id] = getattr(node, "workeroutput", {}).get(
            "ogi_runtime", {"role": "worker", "worker_id": node.gateway.id, "missing_report": str(error)},
        )


def pytest_terminal_summary(terminalreporter, config):
    if (state := config.stash.get(_STATE, None)) is None or "report" not in state:
        return
    terminalreporter.write_line(f"Runtime process diagnostics: {state['path']}")
    for report in (state["report"]["controller"], *state["workers"].values()):
        name = report["worker_id"] or "controller"
        if "missing_report" in report:
            terminalreporter.write_line(f"  {name}: missing runtime report ({report['missing_report']})")
            continue
        locks = report["pytensor_c_lock"]
        affinity = report["cpu"]["affinity"]
        terminalreporter.write_line(
            f"  {name}: CPU={report['cpu']['model']}, affinity_cpus={len(affinity) if affinity is not None else '?'}; "
            f"C-lock acquires={locks['count']}, acquire_elapsed_s={locks['total_acquire_elapsed_s']:.3f} "
            "(upper bound on waiting)"
        )
