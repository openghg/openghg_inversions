"""Bounded interleaved check of the frozen rank-256 production candidate."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_ope152_fixed_ou as reference  # noqa: E402
from openghg_inversions.rhime.co2 import Co2PreparedInputs, build_rhime_co2_cached_sigma  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--expected-sha256", required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
revision = reference.git("rev-parse", "HEAD").strip()
baseline_revision = "c61ef039929b5bd9f9058de7a6728a486df04001"
# Freeze numerical sources while allowing later documentation-only commits.
assert reference.sha256(Path(reference.cached_sigma.__file__)) == "827f703a0d4fe7d31370cd5fa296501f4a4b707f2a03a306b13e0ce0a0585acd"
assert reference.sha256(Path(reference.fixed_ou.__file__)) == "e11e4de7c4f394198d15ecaa8bcf8e75c589dcb76c1813b16db7a41cc8948bc2"
# This measured runtime crashes when increasing an initial one-thread BLAS pool.
if any(pool["num_threads"] < 4 for pool in threadpool_info() if pool["user_api"] == "blas"):
    raise RuntimeError("Start Python with OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4")
assert reference.sha256(args.prepared) == args.expected_sha256
baseline_refresh, baseline_source = reference.baseline_method(
    baseline_revision, reference.cached_sigma, "FixedOuCachedSigmaTarget", "refresh",
)
current_refresh = reference.FixedOuCachedSigmaTarget.refresh
methods = {"baseline": baseline_refresh, "current": current_refresh}
manifest = {
    "status": "running", "started": datetime.now(timezone.utc).isoformat(),
    "revision": revision, "baseline_revision": baseline_revision,
    "prepared": str(args.prepared.resolve()), "prepared_sha256": args.expected_sha256,
    "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "environment": {key: os.environ.get(key) for key in (
        "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST",
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTENSOR_FLAGS",
    )},
    "amplitude_ppm": 0.75, "repeats_per_method_per_thread_count": 12,
    "note": "Both BLAS pools use the recorded limit. Parity checks and final tracing are outside timers.",
    "blocks": [],
}
for name, source in (
    ("baseline_cached_sigma.py", baseline_source),
    ("current_cached_sigma.py", Path(reference.cached_sigma.__file__).read_text()),
    ("current_fixed_ou.py", Path(reference.fixed_ou.__file__).read_text()),
    ("profile_ope152_fixed_ou.py", Path(reference.__file__).read_text()),
    (Path(__file__).name, Path(__file__).read_text()),
):
    destination = args.out / name
    destination.write_text(source)
    manifest[name + "_sha256"] = reference.sha256(destination)
manifest["cpu_topology"] = {}
for cpu in manifest["cpu_affinity"]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    manifest["cpu_topology"][str(cpu)] = {
        key: (topology / key).read_text().strip()
        for key in ("physical_package_id", "core_id", "thread_siblings_list")
    }
started = perf_counter()


def save():
    manifest["elapsed_seconds"] = perf_counter() - started
    manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = args.out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out / "manifest.json")


try:
    prepared = Co2PreparedInputs.load(args.prepared)
    assert prepared.aggregation_error_mode == "low_rank"
    cached = build_rhime_co2_cached_sigma(
        prepared_inputs=prepared, tau_hours=5.0,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        use_bc=True, bc_prior=reference.BC_PRIOR,
    )
    target = cached.target
    manifest["shapes"] = {"observations": target.n_obs, "rank": target.prepared.rank,
                          "state": target.n_state, "sites": target.n_group}
    assert 0 < target.prepared.rank < target.n_obs
    sigma = np.full(target.n_group, 0.75)
    ones = np.ones(target.n_state)
    perturbation = np.random.default_rng(152).normal(scale=0.01, size=target.n_state)
    states = [ones, ones + perturbation, ones - perturbation]
    original_solve = reference.fixed_ou.FixedOuLowRank._solve_transformed
    calls = [0]

    def counted_solve(self, *values, **kwargs):
        calls[0] += 1
        return original_solve(self, *values, **kwargs)

    for threads in (4, 1):
        with threadpool_limits(limits=threads, user_api="blas"):
            block = {"blas_threads": threads, "threadpools": threadpool_info(),
                     "calls": [], "diagnostics": {}}
            manifest["blocks"].append(block)
            assert all(pool["num_threads"] == threads for pool in block["threadpools"]
                       if pool["user_api"] == "blas")
            direct = []
            for state in states:
                residual = target.observations - target.fixed_contribution - target.design @ state
                evaluated = target.evaluate_from_residual(residual, sigma)
                direct.append((evaluated.log_likelihood, -target.design.T @ evaluated.gradient_residual))
            expected = baseline_refresh(target, sigma)
            with patch.object(reference.fixed_ou.FixedOuLowRank, "_solve_transformed", counted_solve):
                for method in methods.values():
                    method(target, sigma)
                for repeat in range(12):
                    order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                    for position, name in enumerate(order):
                        before = calls[0]
                        cpu_start = process_time()
                        wall_start = perf_counter()
                        result = methods[name](target, sigma)
                        wall = perf_counter() - wall_start
                        cpu = process_time() - cpu_start
                        count = calls[0] - before
                        record = {"repeat": repeat, "position": position, "algorithm": name,
                                  "wall_seconds": wall, "cpu_seconds": cpu,
                                  "solve_transformed_calls": count}
                        block["calls"].append(record)
                        assert count == (1 if name == "baseline" else 0)
                        errors = {}
                        for key in ("constant", "linear", "precision"):
                            actual_value, expected_value = getattr(result, key), getattr(expected, key)
                            np.testing.assert_allclose(actual_value, expected_value, rtol=1e-8, atol=1e-6)
                            errors[key] = float(np.max(np.abs(actual_value - expected_value)))
                        record["coefficient_max_absolute_errors"] = errors
                        density_errors, gradient_errors = [], []
                        for state, (density, gradient) in zip(states, direct, strict=True):
                            actual_density, actual_gradient = result.log_likelihood(state), result.gradient(state)
                            np.testing.assert_allclose(actual_density, density, rtol=1e-8, atol=1e-6)
                            np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-7, atol=1e-6)
                            density_errors.append(abs(actual_density - density))
                            gradient_errors.append(float(np.max(np.abs(actual_gradient - gradient))))
                        record["density_max_absolute_error"] = max(density_errors)
                        record["gradient_max_absolute_error"] = max(gradient_errors)
                        save()
            for name, method in methods.items():
                _, diagnostic = reference.line_profile(lambda: method(target, sigma), method)
                block["diagnostics"][name] = diagnostic
            np.testing.assert_allclose(
                block["diagnostics"]["current"]["logdet"],
                block["diagnostics"]["baseline"]["logdet"], rtol=1e-10, atol=1e-7,
            )
            assert block["diagnostics"]["current"]["low_rank_route_accepted"]
            block["summary"] = {
                name: {key: float(np.median([record[key] for record in block["calls"]
                                            if record["algorithm"] == name]))
                       for key in ("wall_seconds", "cpu_seconds")}
                for name in methods
            }
            print(json.dumps({"threads": threads, "medians": block["summary"]}), flush=True)
            save()
    assert reference.fixed_ou.FixedOuLowRank._solve_transformed is original_solve
    assert reference.sha256(args.prepared) == args.expected_sha256
    assert reference.sha256(Path(reference.cached_sigma.__file__)) == manifest["current_cached_sigma.py_sha256"]
    assert reference.sha256(Path(reference.fixed_ou.__file__)) == manifest["current_fixed_ou.py_sha256"]
    assert reference.git("rev-parse", "HEAD").strip() == revision
    manifest["status"] = "passed"
except Exception as error:
    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    save()
