"""Profile linked-result return phases from reproducible states, without fitting.

Run each case in a fresh process; RSS includes imports and the scientific model.
The synthetic low-rank fixed-OU fixture exercises the production linked graph,
not the 334-state/rank-256 real-data sampler or its unidentified OOM allocation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pymc as pm
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.stats.convergence import run_convergence_checks
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inference.diagnostics import posterior_summary
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.serialization import load_trace, save_trace


def fixture(rows: int, offsets: bool, cached: bool) -> pm.Model:
    """Build a labelled rank-two, eight-state, four-station linked fixture."""
    rng = np.random.default_rng(641)
    state = xr.DataArray(np.ones(8), dims="state", coords={"state": np.arange(8)})
    native = np.arange(rows)
    species = np.repeat(["co2", "o2"], rows)
    index = pd.MultiIndex.from_arrays(
        [species, np.tile(native, 2)], names=["species", "channel_observation"],
    )
    coords = xr.Coordinates.from_pandas_multiindex(index, "observation")
    observations = xr.DataArray(
        np.repeat([410.0, 209300.0], rows), dims="observation", coords=coords, name="observations",
    ).assign_coords(
        site=("observation", np.tile(np.array(["A", "B", "C", "D"])[native % 4], 2)),
        time=("observation", np.tile(np.datetime64("2021-07-01T00") + native.astype("timedelta64[h]"), 2)),
        observation_units=("observation", np.repeat("ppm", 2 * rows)),
    )
    observations.attrs["units"] = "ppm"
    fixed = observations.copy().rename("fixed_prior_contribution")
    error = xr.ones_like(observations).rename("independent_error_sd") * 0.1
    factor = xr.DataArray(
        rng.normal(0, 0.02, (2 * rows, 2)), dims=("observation", "agg_rank"),
        coords={"observation": observations.observation, "agg_rank": [0, 1]},
    )
    diagonal = xr.full_like(observations, 0.02)
    kwargs = dict(
        observations=observations, fixed_prior_contribution=fixed,
        retained_prior=CorrelatedLognormalPrior(state, np.eye(8) * 0.01),
        independent_error_sd=error,
        aggregation_error=AggregationError(
            "low_rank", np.sum(factor.values**2, axis=1) + diagonal.values,
            factor=factor, diagonal_variance=diagonal,
        ),
        tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        boundary_sensitivity={},
        bc_prior={channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.01} for channel in ("co2", "o2")},
    )
    for channel in ("co2", "o2"):
        dim = f"{channel}_measure"
        kwargs[f"{channel}_sensitivity"] = xr.DataArray(
            rng.normal(0, 0.1, (rows, 8)), dims=(dim, "state"), coords={dim: native, "state": state.state},
        )
        kwargs["boundary_sensitivity"][channel] = xr.DataArray(
            rng.uniform(0.1, 0.3, (rows, 4)), dims=(dim, "bc_region"),
            coords={dim: native, "bc_region": ["N", "E", "S", "W"]},
        )
    if offsets:
        kwargs.update(
            offset_prior={"o2": {"pdf": "normal", "mu": 0.0, "sigma": 0.5}},
            offset_args={"o2": {"per_site": True, "drop_first": True}},
        )
    built = (build_co2_o2_cached_sigma_model if cached else build_co2_o2_model)(**kwargs)
    return built.model if cached else built


def rss_mib() -> float:
    return int(Path("/proc/self/statm").read_text().split()[1]) * resource.getpagesize() / 2**20


@contextmanager
def phase(results: dict, name: str):
    """Sample process RSS during one bounded phase, including its endpoints."""
    start = time.perf_counter()
    initial = peak = rss_mib()
    stop = threading.Event()

    def observe():
        nonlocal peak
        while not stop.wait(0.005):
            peak = max(peak, rss_mib())

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        yield
    finally:
        stop.set()
        observer.join()
        final = rss_mib()
        results[name] = {
            "seconds": round(time.perf_counter() - start, 4),
            "start_rss_mib": round(initial, 3), "end_rss_mib": round(final, 3),
            "sampled_peak_rss_mib": round(max(peak, final), 3),
            "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        }
        print(name, results[name], file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=("full", "compact"))
    parser.add_argument("--rows", type=int, default=256, help="Rows per species")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--tune", type=int, default=64)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--no-offsets", action="store_true")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert min(args.rows, args.draws, args.chains) > 0 and args.tune >= 0
    result = {
        "options": vars(args) | {"output": str(args.output)}, "pymc": pm.__version__,
        "python": sys.version, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phases": {},
    }
    phases = result["phases"]
    with phase(phases, "model_construction"):
        model = fixture(args.rows, not args.no_offsets, args.cached)
    with phase(phases, "backend_compile"):
        variables = model.replace_rvs_by_values(model.free_RVs) if args.strategy == "compact" else None
        template = NDArray(model=model, vars=variables)
        point = model.initial_point()
        random = np.random.default_rng(2718)
    with phase(phases, "backend_recording"):
        backends = []
        for chain in range(args.chains):
            backend = NDArray(
                model=model, vars=template.vars, fn=template.fn,
                var_shapes=template.var_shapes, var_dtypes=template.var_dtypes,
            )
            backend.setup(args.tune + args.draws, chain)
            for draw in range(args.tune + args.draws):
                state = {key: value + random.normal(0, 0.015, value.shape) for key, value in point.items()}
                backend.record(state, in_warmup=draw < args.tune)
            backend.close()
            backends.append(backend)
        result["backend_bytes"] = sum(v.nbytes for b in backends for v in b.samples.values())
    with phase(phases, "retained_slice"):
        retained = MultiTrace(backends)[args.tune:]
        retained.report._n_tune = args.tune
        retained.report._n_draws = args.draws
        retained.report._t_sampling = 0
        result["slice_borrows_backend"] = all(
            np.shares_memory(retained._straces[c].samples[name], b.samples[name])
            for c, b in enumerate(backends) for name in b.varnames
        )
    with phase(phases, "xarray_conversion"):
        trace = pm.to_inference_data(retained, model=model, log_likelihood=False)
    # Full mode matches pm.sample: likelihood and diagnostics coexist with backends.
    if args.strategy == "compact":
        with phase(phases, "backend_release"):
            del retained, backends, backend, template
            gc.collect()
    with phase(phases, "log_likelihood"):
        pm.compute_log_likelihood(trace, model=model, progressbar=False)
        assert trace.log_likelihood.y.dims == ("chain", "draw")
        result["log_likelihood_sha256"] = hashlib.sha256(trace.log_likelihood.y.values.data).hexdigest()
    with phase(phases, "pymc_convergence"):
        run_convergence_checks(trace, model)
    if args.strategy == "full":
        with phase(phases, "backend_release"):
            del retained, backends, backend, template
            gc.collect()
    else:
        from openghg_inversions.inference.sampling import _restore_deterministics

        with phase(phases, "deterministic_reconstruction"):
            trace["posterior"] = _restore_deterministics(trace.posterior.to_dataset(), model)
    with phase(phases, "coordinate_restoration"):
        trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    posterior = trace.posterior.to_dataset()
    result["posterior_bytes"] = sum(var.nbytes for var in posterior.data_vars.values())
    result["posterior_variables"] = {
        name: {"shape": list(var.shape), "dtype": str(var.dtype),
               "sha256": hashlib.sha256(np.ascontiguousarray(var.values).data).hexdigest()}
        for name, var in sorted(posterior.data_vars.items())
    }
    with tempfile.TemporaryDirectory(prefix="linked-return-") as directory:
        path = Path(directory) / "trace.nc"
        with phase(phases, "serialization"):
            save_trace(trace, path)
        result["serialized_bytes"] = path.stat().st_size
        with phase(phases, "serialization_roundtrip"):
            reopened = load_trace(path)
            xr.testing.assert_equal(trace.posterior.to_dataset(), reopened.posterior.to_dataset())
            xr.testing.assert_equal(trace.log_likelihood.to_dataset(), reopened.log_likelihood.to_dataset())
            del reopened
    with phase(phases, "posterior_diagnostics"):
        summary = posterior_summary(posterior)
        result["diagnostic_variables"] = len(summary.data_vars)
    result["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("options", "backend_bytes", "posterior_bytes", "process_peak_rss_mib")}))


if __name__ == "__main__":
    main()
