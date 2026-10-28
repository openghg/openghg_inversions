"""Isolated PR986 memory experiment; every invocation owns one fresh process.

Correctness compilation runs only in `check`, never after measured sampling.
The four variants use the same collapsed research recipe, not production APIs.
"""

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import pymc as pm
import pytensor
import xarray as xr
from pytensor.graph.traversal import graph_inputs

import precision_probe as probe
from openghg_inversions.models import cached_sigma as numerical
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import load_trace, save_trace
from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner as runner


VARIANTS = ("float64", "active_b32", "blocked64", "combined")


def blocked_init(self, *, prepared, observations, fixed_contribution, design):
    """Original owned float64 inputs/validation; bounded construction temporary."""
    self.prepared = prepared
    self.observations = numerical._vector(observations, "observations")
    self.fixed_contribution = numerical._vector(fixed_contribution, "fixed_contribution")
    self.design = numerical._matrix(design, "design")
    expected = (prepared.n_observation,)
    if self.observations.shape != expected:
        raise ValueError(f"observations has shape {self.observations.shape}, expected {expected}.")
    if self.fixed_contribution.shape != expected:
        raise ValueError(
            f"fixed_contribution has shape {self.fixed_contribution.shape}, expected {expected}."
        )
    if self.design.shape[0] != prepared.n_observation:
        raise ValueError("design must have one row per observation.")
    self._residual_at_zero = numerical._vector(
        self.observations - self.fixed_contribution, "zero-state residual"
    )
    self._transformed_rhs = np.empty((prepared.n_observation, self.design.shape[1] + 1), dtype="float64")
    self._transformed_rhs[:, 0] = prepared.mode_transform @ self._residual_at_zero
    for start in range(0, self.design.shape[1], 32):
        stop = min(start + 32, self.design.shape[1])
        self._transformed_rhs[:, start + 1 : stop + 1] = prepared.mode_transform @ self.design[:, start:stop]


@contextmanager
def recipe(variant, capture):
    original = numerical.FixedOuCachedSigmaTarget.__init__
    if variant in ("blocked64", "combined"):
        numerical.FixedOuCachedSigmaTarget.__init__ = blocked_init
    try:
        with probe.research_recipe(
            "float32" if variant in ("active_b32", "combined") else "float64", capture
        ):
            yield
    finally:
        numerical.FixedOuCachedSigmaTarget.__init__ = original


def process_tree():
    """PSS apportions shared pages; never sum RSS from parent and children."""
    result, pending, seen = [], [os.getpid()], set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            root = Path(f"/proc/{pid}")
            stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
            identity = (pid, stat[19])  # Linux stat field 22: starttime.
            rollup = (root / "smaps_rollup").read_text()
            pss = int(next(line.split()[1] for line in rollup.splitlines() if line.startswith("Pss:"))) * 1024
            pending.extend(int(item) for item in (root / f"task/{pid}/children").read_text().split())
            if (root / "stat").read_text().rsplit(")", 1)[1].split()[19] == identity[1]:
                result.append(
                    {
                        "pid": pid,
                        "starttime": identity[1],
                        "pss_bytes": pss,
                        "cmdline": (root / "cmdline")
                        .read_bytes()
                        .replace(b"\0", b" ")
                        .decode(errors="replace"),
                    }
                )
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    return result


def cgroup_memory():
    """Report kernel accounting separately; cgroup peak is cumulative, not a phase peak."""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, name = line.split(":", 2)
        if not controllers:
            directory = Path("/sys/fs/cgroup") / name.lstrip("/")
            names = {"current_bytes": "memory.current", "cumulative_peak_bytes": "memory.peak"}
        elif "memory" in controllers.split(","):
            directory = Path("/sys/fs/cgroup/memory") / name.lstrip("/")
            names = {
                "current_bytes": "memory.usage_in_bytes",
                "cumulative_peak_bytes": "memory.max_usage_in_bytes",
            }
        else:
            continue
        if directory.exists():
            values = {"path": str(directory)}
            for label, filename in names.items():
                try:
                    values[label] = int((directory / filename).read_text())
                except (FileNotFoundError, PermissionError, ValueError):
                    pass
            return values
    return {}


class MemoryMonitor:
    def __init__(self, path):
        self.phase = "imports_complete"
        self.started = time.perf_counter()
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.output = path.open("w")
        self.summary = {}
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        with self.lock:
            processes = process_tree()
            sample = {
                "seconds": time.perf_counter() - self.started,
                "phase": self.phase,
                "tree_pss_bytes": sum(p["pss_bytes"] for p in processes),
                "processes": processes,
                "cgroup": cgroup_memory(),
            }
            self.output.write(json.dumps(sample) + "\n")
            self.output.flush()
            phase = self.summary.setdefault(
                self.phase,
                {
                    "sampled_peak_tree_pss_bytes": 0,
                    "last_tree_pss_bytes": 0,
                    "max_process_count": 0,
                    "samples": 0,
                },
            )
            phase["sampled_peak_tree_pss_bytes"] = max(
                phase["sampled_peak_tree_pss_bytes"], sample["tree_pss_bytes"]
            )
            phase["last_tree_pss_bytes"] = sample["tree_pss_bytes"]
            phase["max_process_count"] = max(phase["max_process_count"], len(processes))
            phase["samples"] += 1

    def loop(self):
        try:
            while not self.stopped.wait(0.1):
                self.sample()
        except Exception as error:
            self.error = error

    def start(self):
        self.sample()
        self.thread.start()

    def mark(self, phase):
        with self.lock:
            self.sample()
            self.phase = phase
            self.sample()

    def stop(self):
        self.stopped.set()
        self.thread.join()
        try:
            if self.error is not None:
                raise RuntimeError("Memory sampling failed") from self.error
            self.sample()
            return self.summary
        finally:
            self.output.close()


def inventory(roots):
    """Enumerate actual arrays and deduplicate their retained NumPy backing storage."""
    arrays, owners, visited = [], {}, set()

    def walk(value, path, depth=0):
        if id(value) in visited or depth > 12:
            return
        visited.add(id(value))
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            owner = str(id(base))
            owners[owner] = {"bytes": int(base.nbytes), "dtype": str(base.dtype), "shape": list(base.shape)}
            arrays.append(
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "logical_bytes": int(value.nbytes),
                    "owner": owner,
                }
            )
        elif isinstance(value, xr.DataArray):
            walk(value.data, path + ".data", depth + 1)
            for name, coordinate in value.coords.items():
                walk(coordinate.data, path + ".coords." + name, depth + 1)
        elif isinstance(value, xr.Dataset):
            for name, variable in value.data_vars.items():
                walk(variable, path + "." + name, depth + 1)
        elif hasattr(value, "get_value"):
            walk(value.get_value(borrow=True), path + ".shared", depth + 1)
        elif hasattr(value, "maker") and hasattr(value, "vm") and hasattr(value.maker, "fgraph"):
            for variable in graph_inputs(value.maker.fgraph.outputs):
                if hasattr(variable, "data"):
                    walk(variable.data, path + ".graph_constant." + str(variable.name), depth + 1)
                elif hasattr(variable, "get_value"):
                    walk(variable, path + ".graph_shared." + str(variable.name), depth + 1)
            for variable, storage in value.vm.storage_map.items():
                walk(storage[0], path + ".compiled_storage." + str(variable.name), depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + "." + str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth + 1)
        elif type(value).__module__.startswith(
            ("openghg_inversions", "pymc", "pytensor", "scipy.sparse")
        ) and hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                if key not in ("model", "_model", "fgraph", "maker"):
                    walk(item, path + "." + key, depth + 1)

    for name, root in roots.items():
        walk(root, name)
    return {
        "arrays": arrays,
        "backing_arrays": owners,
        "retained_unique_backing_bytes": sum(owner["bytes"] for owner in owners.values()),
        "deduplication": "Ultimate ndarray.base identity; distinct wrappers sharing an external buffer can remain double-counted",
        "limit": "Reachable NumPy backing arrays, shared graph inputs, compiled storage and sampler state; walk limited to 12 object levels; allocator/BLAS/Python memory measured by PSS.",
    }


def check(config, variant):
    prepared = probe.prepare(config)
    capture = {}
    with recipe(variant, capture):
        cached = probe.build(prepared, config)
    result = {"consistency": probe.consistency(cached), "input_sha256": probe.input_fingerprint(prepared)}
    if variant == "float64":
        result["original_production_integrity"] = probe.baseline_integrity(
            probe.build(prepared, config), cached
        )
    if variant in ("blocked64", "combined"):
        baseline_capture = {}
        with recipe("active_b32" if variant == "combined" else "float64", baseline_capture):
            baseline = probe.build(prepared, config)
        np.testing.assert_array_equal(cached.target.design, baseline.target.design)
        np.testing.assert_allclose(
            cached.target._transformed_rhs, baseline.target._transformed_rhs, rtol=2e-13, atol=2e-9
        )
        for sigma in (0.1, 0.5, 1.0):
            expected, actual = baseline.target.refresh(sigma), cached.target.refresh(sigma)
            for name in ("constant", "linear", "precision"):
                np.testing.assert_allclose(
                    getattr(actual, name), getattr(expected, name), rtol=2e-10, atol=2e-7
                )
        result["blocked_transform_and_refresh"] = "pass at amplitudes 0.1, 0.5, 1.0"
    return result


def measure(config, variant, chains, out):
    monitor = MemoryMonitor(out / "memory_samples.jsonl")
    monitor.start()
    timings, snapshots, capture = {}, {}, {}
    original_build = runner.build_co2_o2_cached_sigma_model
    original_sampler = runner._sampler_for_cached_graph
    original_sample = runner.sample_rhime_model
    compiled_step = None

    def snapshot(name):
        cached = capture["cached"]
        snapshots[name] = inventory(
            {
                "prepared_inputs": prepared,
                "target": cached.target,
                "initial_cache": cached.initial_cache,
                "shared_cache": cached.shared_cache,
                "model_shared": {n: v for n, v in cached.model.named_vars.items() if hasattr(v, "get_value")},
                "sampler": compiled_step,
            }
        )

    def build(*args, **kwargs):
        monitor.mark("construction")
        start = time.perf_counter()
        built = original_build(*args, **kwargs)
        timings["construction_seconds"] = time.perf_counter() - start
        capture["cached"] = built
        snapshot("constructed")
        return built

    def sampler(*args, **kwargs):
        nonlocal compiled_step
        monitor.mark("compile")
        start = time.perf_counter()
        result = original_sampler(*args, **kwargs)
        timings["compile_seconds"] = time.perf_counter() - start
        compiled_step = result.sample_kwargs["step"]
        snapshot("compiled")
        monitor.mark("refresh")
        start = time.perf_counter()
        for _ in range(5):
            cache = capture["cached"].target.refresh(0.5)
            capture["cached"].shared_cache.update(cache)
        timings["five_refreshes_seconds"] = time.perf_counter() - start
        # Restore the sampler's original initial-amplitude cache before sampling.
        capture["cached"].shared_cache.update(capture["cached"].initial_cache)
        snapshot("refreshed")
        return result

    def sample(*args, **kwargs):
        monitor.mark("sampling")
        start = time.perf_counter()
        result = original_sample(*args, **kwargs)
        timings["sampling_seconds"] = time.perf_counter() - start
        snapshot("sampled")
        return result

    try:
        monitor.mark("prepare")
        start = time.perf_counter()
        prepared = probe.prepare(config)
        input_sha = probe.input_fingerprint(prepared)
        timings["prepare_seconds"] = time.perf_counter() - start
        runner.build_co2_o2_cached_sigma_model = build
        runner._sampler_for_cached_graph = sampler
        runner.sample_rhime_model = sample
        with recipe(variant, capture):
            trace = runner.run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                prepared_inputs=prepared,
                tau_hours=config["tau_hours"],
                bc_prior=config["bc_prior"],
                site_amplitude_prior=config["site_amplitude_prior"],
                sampler=RhimeSampler(
                    draws=config["sampling"]["draws"],
                    tune=config["sampling"]["tune"],
                    chains=chains,
                    sample_prior_predictive=False,
                    sample_posterior_predictive=False,
                    sample_kwargs={
                        "cores": chains,
                        "mp_ctx": "spawn",
                        "random_seed": config["sampling"]["seed"],
                        "compute_convergence_checks": False,
                        "progressbar": False,
                    },
                ),
            )
        actual_chains = int(trace.posterior.dataset.sizes["chain"])
        assert actual_chains == chains
        monitor.mark("save")
        start = time.perf_counter()
        save_trace(trace, out / "posterior.nc")
        reopened = load_trace(out / "posterior.nc")
        xr.testing.assert_allclose(trace.posterior.dataset, reopened.posterior.dataset)
        del reopened
        timings["save_reopen_seconds"] = time.perf_counter() - start
        monitor.mark("complete")
        gc.collect()
        return {
            "timings": timings,
            "array_inventory": snapshots,
            "input_sha256": input_sha,
            "memory": monitor.stop(),
            "rows": capture["cached"].target.n_obs,
            "cache_states": capture["cached"].target.n_state,
            "chains": chains,
            "actual_posterior_chains": actual_chains,
            "cores": chains,
            "mp_ctx": "spawn",
            "sampling_scope": "250 tune + 250 draws per chain; timing only, not scientific acceptance",
        }
    finally:
        runner.build_co2_o2_cached_sigma_model = original_build
        runner._sampler_for_cached_graph = original_sampler
        runner.sample_rhime_model = original_sample
        if monitor.thread.is_alive():
            monitor.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "measure"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--chains", type=int, choices=(1, 4), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert pytensor.config.floatX == "float64", "Global arithmetic remains float64"
    assert Path(numerical.__file__).resolve().is_relative_to(Path.cwd().resolve()), (
        "Set PYTHONPATH=. and run from the benchmark worktree root"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "result.json").exists(), "Use a fresh output directory"
    config = json.loads(args.config.read_text())
    assert config["scope"] == "controlled_technical_benchmark_scientific_acceptance_pending"
    result = (
        check(config, args.variant)
        if args.stage == "check"
        else measure(config, args.variant, args.chains, args.out)
    )
    result.update(
        scope=config["scope"],
        stage=args.stage,
        variant=args.variant,
        config=config,
        config_sha256=probe.sha(args.config),
        script_sha256={p.name: probe.sha(p) for p in Path(__file__).parent.glob("*.py")},
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pymc": pm.__version__,
            "pytensor": pytensor.__version__,
        },
        runtime={
            "source_package_path": str(Path(numerical.__file__).resolve()),
            "pytensor_mode": str(pytensor.config.mode),
            "pytensor_optimizer": str(pytensor.config.optimizer),
            "pytensor_compiledir": str(pytensor.config.compiledir),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "DASK_NUM_WORKERS",
                    "PYTENSOR_FLAGS",
                    "SLURM_JOB_ID",
                    "SLURM_ARRAY_TASK_ID",
                    "SLURM_CPUS_PER_TASK",
                    "SLURM_MEM_PER_NODE",
                )
            },
        },
        memory_method="100 ms sampled process-tree PSS; cgroup current and cumulative peak separately; no summed RSS",
    )
    probe.write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
