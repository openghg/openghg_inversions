"""Compare cached fixed-OU solves on one immutable prepared-input artifact.

Run large inputs on Slurm. ``--self-check`` uses only tiny synthetic arrays.
Historical Woodbury is a benchmark reference, not a supported runtime option.
``--aggregation-error-rank`` prepares a shared low-rank-plus-diagonal target
from the source artifact before comparing algorithms; the source stays intact.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import linecache
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_info

from openghg_inversions.models import cached_sigma, fixed_ou
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget


BASELINE_REVISION = "1fe38ad91a5f5c14f7d0c4736ddcaf66487776a6"
HISTORICAL_REVISION = "bbdfd60d151eef854c3e0be6ff119f59f33d9ab2^"
PREPARED_SHA256 = "7ab7cf733eba7b12dd1fea7a5c8812236eda1d579486b03e5e2d27feab8866bf"
SOURCE_PATH = "openghg_inversions/models/fixed_ou.py"
BC_PRIOR = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.01, "lower": 0.0}
ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline_method(revision, module, class_name, method_name):
    """Extract the original method unchanged, using current compatible types."""
    source_path = Path(module.__file__).resolve().relative_to(ROOT)
    source = git("show", f"{revision}:{source_path}")
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == method_name)
    filename = f"<{module.__name__}:{revision}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
    return namespace[method_name], source


def historical_woodbury(self, rhs, site_amplitude):
    """Pre-bbdfd60 algebra, retaining current result type and mode validation."""
    weights, logdet = self._mode_weights_and_logdet(np.asarray(site_amplitude))
    transformed_rhs = self.mode_transform @ rhs
    rhs_2d = transformed_rhs[:, None] if transformed_rhs.ndim == 1 else transformed_rhs
    solved = weights[:, None] * rhs_2d
    if self.rank:
        weighted_factor = self.transformed_factor * np.sqrt(weights)[:, None]
        core = np.eye(self.rank) + weighted_factor.T @ weighted_factor
        core = (core + core.T) * 0.5
        cholesky = np.linalg.cholesky(core)
        factor_solution = cho_solve(
            (cholesky, True), self.transformed_factor.T @ solved, check_finite=False,
        )
        solved -= weights[:, None] * self.transformed_factor @ factor_solution
        logdet += float(2.0 * np.log(np.diag(cholesky)).sum())
    solution = self.mode_transform.T @ (solved[:, 0] if transformed_rhs.ndim == 1 else solved)
    return fixed_ou.FixedOuCovarianceSolve(np.asarray(solution), logdet)


def derive_low_rank_inputs(prepared, rank, source_sha256):
    """Use the CO2 preparation helper, preserving the discarded diagonal tail."""
    from openghg_inversions.inversion_data import RhimePreparedInputs
    from openghg_inversions.observation_error import prepare_low_rank_aggregation_error
    from openghg_inversions.rhime.co2 import Co2PreparedInputs

    source = prepared.inv_inputs
    covariance = source["aggregation_error_covariance"]
    approximation = prepare_low_rank_aggregation_error(
        covariance, rank=min(rank, source.sizes["nmeasure"]),
    )
    selected = approximation.aggregation_error
    assert selected.factor is not None and selected.diagonal_variance is not None
    units = source["mf"].attrs["units"]
    derived = source.drop_vars("aggregation_error_covariance").assign(
        low_rank_factor=selected.factor.reset_index("nmeasure", drop=True).assign_attrs(units=units),
        diagonal_residual_variance=selected.diagonal_variance.reset_index("nmeasure", drop=True).assign_attrs(
            units=covariance.attrs.get("units", f"({units})^2"),
        ),
    )
    np.testing.assert_allclose(
        selected.marginal_variance, np.diag(covariance.values), rtol=1e-12, atol=1e-12,
    )
    return Co2PreparedInputs(
        RhimePreparedInputs(
            inv_inputs=derived, basis_functions=prepared.basis_functions,
            site_metadata=prepared.site_metadata,
        ),
        aggregation_error_mode="low_rank",
        provenance={
            **prepared.provenance, "aggregation_error_rank": rank,
            "aggregation_error": dict(approximation.diagnostics),
            "benchmark_source_sha256": source_sha256,
        },
    )


def line_profile(call, *methods):
    """Time source lines in the selected kernel and its fixed-OU helpers.

    Times include callees, so nested function rows must not be added together.
    This separate diagnostic call is excluded from uninstrumented repeat times.
    """
    filenames = {method.__code__.co_filename for method in methods}
    filenames.update((fixed_ou.__file__, cached_sigma.__file__))
    elapsed = defaultdict(float)
    active = {}
    diagnostics = {}

    def trace(frame, event, arg):
        if frame.f_code.co_filename not in filenames:
            return None
        now = perf_counter()
        key = id(frame)
        if key in active:
            previous_line, start = active[key]
            elapsed[(frame.f_code.co_filename, frame.f_code.co_name, previous_line)] += now - start
        if event == "return":
            active.pop(key, None)
            if isinstance(arg, cached_sigma.MarginalQuadraticCache) and "logdet" in frame.f_locals:
                diagnostics["logdet"] = float(frame.f_locals["logdet"])
            if isinstance(arg, fixed_ou.FixedOuCovarianceSolve):
                diagnostics["logdet"] = float(arg.logdet)
                factor = frame.f_locals.get("whitened_factor", frame.f_locals.get("weighted_factor"))
                if factor is not None:
                    diagnostics["whitened_factor_squared_frobenius_norm"] = float(
                        np.einsum("ij,ij->", factor, factor)
                    )
            if "reciprocal_condition" in frame.f_locals:
                diagnostics["reciprocal_condition"] = float(frame.f_locals["reciprocal_condition"])
                diagnostics["condition_estimator_info"] = int(frame.f_locals["info"])
                route = "low_rank_route_accepted" if frame.f_code.co_name == "_sum_squares_gram" else "dense_route_accepted"
                diagnostics[route] = arg is not None
            if "gram_roundoff_bound" in frame.f_locals:
                diagnostics["residual_gram_roundoff_bound"] = float(frame.f_locals["gram_roundoff_bound"])
        else:
            active[key] = (frame.f_lineno, now)
        return trace

    previous_trace = sys.gettrace()
    started = perf_counter()
    try:
        sys.settrace(trace)
        result = call()
    finally:
        sys.settrace(previous_trace)
    diagnostics["seconds"] = perf_counter() - started
    diagnostics["inclusive_line_timings"] = [
        {"file": filename, "function": function, "line": line,
         "source": linecache.getline(filename, line).strip(), "seconds": seconds}
        for (filename, function, line), seconds in sorted(elapsed.items())
    ]
    return result, diagnostics


def self_check(methods):
    """Exercise method substitution, line profiling, and independent dense parity."""
    original = fixed_ou.FixedOuLowRank.solve
    original_refresh = FixedOuCachedSigmaTarget.refresh
    rng = np.random.default_rng(152)
    cases = []
    for rank in (0, 3, 8, 12):
        factor = rng.normal(size=(8, rank)) * 0.2
        diagonal = np.linspace(0.2, 0.5, 8)
        times = np.arange(8, dtype=float)
        sites = np.repeat([0, 1], 4)
        amplitude = np.asarray([0.3, 0.75])
        prepared = fixed_ou.prepare_fixed_ou_low_rank(factor, diagonal, times, sites, 5.0)
        target = FixedOuCachedSigmaTarget(
            prepared=prepared, observations=rng.normal(size=8),
            fixed_contribution=np.zeros(8), design=rng.normal(size=(8, 3)),
        )
        covariance = factor @ factor.T + np.diag(diagonal)
        for block in prepared.site_blocks:
            covariance[np.ix_(block.observation_indices, block.observation_indices)] += (
                amplitude[block.site] ** 2 * block.correlation_cholesky @ block.correlation_cholesky.T
            )
        rhs = np.column_stack((target.observations, target.design))
        expected = np.linalg.solve(covariance, rhs)
        expected_logdet = np.linalg.slogdet(covariance)[1]
        for name, (solve_method, refresh_method) in methods.items():
            start = perf_counter()
            with (
                patch.object(fixed_ou.FixedOuLowRank, "solve", solve_method),
                patch.object(FixedOuCachedSigmaTarget, "refresh", refresh_method),
            ):
                actual = prepared.solve(rhs, amplitude)
                vector = prepared.solve(rhs[:, 0], amplitude)
                cache, profile = line_profile(lambda: target.refresh(amplitude), solve_method, refresh_method)
            assert fixed_ou.FixedOuLowRank.solve is original
            assert FixedOuCachedSigmaTarget.refresh is original_refresh
            np.testing.assert_allclose(actual.solution, expected, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(vector.solution, expected[:, 0], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(actual.logdet, expected_logdet, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(profile["logdet"], actual.logdet)
            np.testing.assert_allclose(cache.precision, target.design.T @ expected[:, 1:])
            np.testing.assert_allclose(cache.linear, target.design.T @ expected[:, 0])
            np.testing.assert_allclose(cache.constant, -0.5 * (
                8 * np.log(2 * np.pi) + expected_logdet + target.observations @ expected[:, 0]
            ))
            cases.append({"rank": rank, "algorithm": name, "seconds": perf_counter() - start})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-sha256", default=PREPARED_SHA256)
    parser.add_argument("--aggregation-error-rank", type=int)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--algorithms", nargs="+", choices=("baseline", "current", "historical"),
                        default=["baseline", "current", "historical"])
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.3, 0.75, 1.2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if Path(fixed_ou.__file__).resolve() != ROOT / SOURCE_PATH:
        parser.error("Set PYTHONPATH to this checkout so the current kernel is profiled")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.aggregation_error_rank is not None and args.aggregation_error_rank < 1:
        parser.error("--aggregation-error-rank must be positive")
    baseline, baseline_source = baseline_method(
        args.baseline_revision, fixed_ou, "FixedOuLowRank", "solve",
    )
    baseline_refresh, baseline_cache_source = baseline_method(
        args.baseline_revision, cached_sigma, "FixedOuCachedSigmaTarget", "refresh",
    )
    methods = {
        "baseline": (baseline, baseline_refresh),
        "current": (fixed_ou.FixedOuLowRank.solve, FixedOuCachedSigmaTarget.refresh),
        "historical": (historical_woodbury, baseline_refresh),
    }
    if args.self_check:
        print(json.dumps({"status": "self-check-passed", "cases": self_check(methods)}, indent=2))
        return
    if args.prepared is None or args.out is None:
        parser.error("--prepared and --out are required unless --self-check is selected")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest = {
        "issue": "OPE-152", "status": "started", "started_at": datetime.now(UTC).isoformat(),
        "scope": "same-input computational profile; no sampling or model selection",
        "prepared": str(args.prepared.resolve()), "script_sha256": sha256(Path(__file__)),
        "revision": git("rev-parse", "HEAD").strip(),
        "baseline_revision": git("rev-parse", args.baseline_revision).strip(),
        "historical_revision": git("rev-parse", HISTORICAL_REVISION).strip(),
        "historical_note": "benchmark-only pre-stability algebra; removed timing result fields omitted",
        "refresh_comparison": "Baseline/historical use pinned original refresh; current uses live refresh.",
        "versions": {name: version(name) for name in
                     ("openghg_inversions", "numpy", "scipy", "pymc", "pytensor")},
        "python": sys.version, "threadpools": threadpool_info(),
        "environment": {name: os.environ.get(name) for name in
                        ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "DASK_NUM_WORKERS", "PYTENSOR_FLAGS")},
        "science": {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                    "bc_prior": BC_PRIOR, "offset": None, "reference_state": "all coefficients = 1"},
        "repeats": args.repeats, "phase_timings": [], "evaluations": [],
        "profile_note": "Line timings are inclusive and overlap across callers/helpers; repeats are uninstrumented.",
    }
    diff = git("diff", "HEAD", "--binary")
    (args.out / "working-tree.patch").write_text(diff)
    manifest["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    (args.out / "baseline_fixed_ou.py").write_text(baseline_source)
    (args.out / "baseline_cached_sigma.py").write_text(baseline_cache_source)
    (args.out / "historical_fixed_ou.py").write_text(git("show", f"{HISTORICAL_REVISION}:{SOURCE_PATH}"))
    (args.out / "current_fixed_ou.py").write_text(Path(fixed_ou.__file__).read_text())
    (args.out / "current_cached_sigma.py").write_text(Path(cached_sigma.__file__).read_text())
    (args.out / Path(__file__).name).write_text(Path(__file__).read_text())
    manifest["current_source_sha256"] = sha256(args.out / "current_fixed_ou.py")
    manifest["current_cache_source_sha256"] = sha256(args.out / "current_cached_sigma.py")

    def save():
        manifest["elapsed_seconds"] = perf_counter() - started
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.out / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "manifest.json")

    @contextmanager
    def phase(name):
        manifest["active_phase"] = name
        save()
        begin = perf_counter()
        try:
            yield
        finally:
            manifest["phase_timings"].append({"phase": name, "seconds": perf_counter() - begin})
            save()

    try:
        from openghg_inversions.rhime.co2 import Co2PreparedInputs, build_rhime_co2_cached_sigma

        with phase("load_prepared"):
            manifest["prepared_sha256"] = sha256(args.prepared)
            if manifest["prepared_sha256"] != args.expected_sha256:
                raise ValueError("Prepared artifact SHA256 does not match the requested input")
            prepared = Co2PreparedInputs.load(args.prepared)
            if prepared.aggregation_error_mode != "dense":
                raise ValueError("Expected the unchanged dense aggregation covariance")
            np.testing.assert_allclose(prepared.inv_inputs.alpha_prior_mean, 1.0, rtol=0, atol=1e-12)
        if args.aggregation_error_rank is not None:
            with phase("derive_low_rank_aggregation_error"):
                prepared = derive_low_rank_inputs(
                    prepared, args.aggregation_error_rank, manifest["prepared_sha256"],
                )
                manifest["derived_aggregation_error"] = {
                    "mode": prepared.aggregation_error_mode,
                    "requested_rank": args.aggregation_error_rank,
                    "actual_rank": prepared.inv_inputs.sizes["agg_rank"],
                    "source_sha256": manifest["prepared_sha256"],
                    "diagnostics": dict(prepared.provenance["aggregation_error"]),
                }
            with phase("save_derived_prepared"):
                derived_path = args.out / f"prepared-inputs-rank{args.aggregation_error_rank}.nc"
                prepared.save(derived_path)
                manifest["derived_aggregation_error"].update(
                    prepared=str(derived_path.resolve()), prepared_sha256=sha256(derived_path),
                )
        with phase("build_current_including_initial_refresh"):
            cached = build_rhime_co2_cached_sigma(
                prepared_inputs=prepared, tau_hours=5.0,
                site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
                use_bc=True, bc_prior=BC_PRIOR,
            )
        target = cached.target
        manifest["shapes"] = {"observations": target.n_obs, "rank": target.prepared.rank,
                              "state": target.n_state, "sites": target.n_group,
                              "refresh_rhs": [target.n_obs, target.n_state + 1]}
        coefficients = np.ones(target.n_state)
        mean = target.fixed_contribution + target.design @ coefficients
        inputs = prepared.inv_inputs
        expected = (inputs.fixed_prior_contribution + (inputs.H * inputs.alpha_prior_mean).sum("region")
                    + inputs.H_bc.sum("bc_region")).transpose("nmeasure")
        np.testing.assert_allclose(mean, expected, rtol=1e-12, atol=1e-8)
        residual = target.observations - mean
        for amplitude in args.amplitudes:
            sigma = np.full(target.n_group, amplitude)
            record = {"site_amplitude_ppm": amplitude, "algorithms": {}}
            manifest["evaluations"].append(record)
            with phase(f"evaluate_{amplitude}"):
                record["evaluate_seconds"] = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    evaluation = target.evaluate_from_residual(residual, sigma)
                    record["evaluate_seconds"].append(perf_counter() - start)
                _, record["evaluate_profile"] = line_profile(
                    lambda: target.evaluate_from_residual(residual, sigma), fixed_ou.FixedOuLowRank.evaluate,
                )
                record["direct_log_likelihood"] = float(evaluation.log_likelihood)
                record["amplitude_gradient"] = evaluation.gradient_site_amplitude.tolist()
                record["residual_gradient_norm"] = float(np.linalg.norm(evaluation.gradient_residual))
                assert np.isfinite(evaluation.log_likelihood)
                assert np.isfinite(evaluation.gradient_residual).all()
                assert np.isfinite(evaluation.gradient_site_amplitude).all()
                expected_gradient = -target.design.T @ evaluation.gradient_residual
            logdets = []
            for name in args.algorithms:
                result = {"refresh_seconds": []}
                record["algorithms"][name] = result
                solve_method, refresh_method = methods[name]
                with (
                    phase(f"refresh_{name}_{amplitude}"),
                    patch.object(fixed_ou.FixedOuLowRank, "solve", solve_method),
                    patch.object(FixedOuCachedSigmaTarget, "refresh", refresh_method),
                ):
                    for _ in range(args.repeats):
                        start = perf_counter()
                        quadratic = target.refresh(sigma)
                        result["refresh_seconds"].append(perf_counter() - start)
                    _, result["refresh_profile"] = line_profile(
                        lambda: target.refresh(sigma), solve_method, refresh_method,
                    )
                    value = quadratic.log_likelihood(coefficients)
                    gradient = quadratic.gradient(coefficients)
                    result["cached_log_likelihood"] = value
                    result["density_absolute_error"] = abs(value - evaluation.log_likelihood)
                    result["state_gradient_max_absolute_error"] = float(np.max(np.abs(gradient - expected_gradient)))
                    np.testing.assert_allclose(value, evaluation.log_likelihood, rtol=1e-8, atol=1e-6)
                    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-7, atol=1e-6)
                    logdets.append(result["refresh_profile"]["logdet"])
                    result["parity"] = "passed"
                    print(json.dumps({"amplitude": amplitude, "algorithm": name,
                                      "median_refresh_seconds": float(np.median(result["refresh_seconds"])),
                                      "density_absolute_error": result["density_absolute_error"]}), flush=True)
            np.testing.assert_allclose(logdets, logdets[0], rtol=1e-10, atol=1e-7)
        if sha256(args.prepared) != manifest["prepared_sha256"]:
            raise ValueError("Prepared artifact changed during profiling")
        manifest.update(status="computational-profile-passed", active_phase=None)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
