"""Experimental cached sum-of-squares benchmark; run production inputs on Slurm.

The condition-screened variant is an experiment, not a stability guarantee.
No production methods are replaced. Use --self-check for tiny dense oracles.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dpocon

from scripts import profile_ope152_fixed_ou as baseline
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget, MarginalQuadraticCache
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank


def sum_squares_gram(factor, rhs, *, guarded=False, diagnostics=None):
    """Return ``B.T (I + W W.T)^-1 B`` and its covariance logdet update.

    Inputs are already whitened. A guarded rejection returns ``(None, None)``
    for the caller to fall back. Raw Cholesky failures propagate. Supplying a
    diagnostics dictionary adds phase timings, norms, and a raw-route condition
    estimate; omit it for uninstrumented timings. Conditioning alone does not
    bound cancellation in the computed conditional residual.
    """
    if diagnostics is not None:
        diagnostics.update(factor_squared_norm=float(np.einsum("ij,ij->", factor, factor)),
                           rhs_squared_norm=float(np.einsum("ij,ij->", rhs, rhs)))
    start = perf_counter() if diagnostics is not None else 0.0
    if not factor.shape[1]:
        gram = rhs.T @ rhs
        if diagnostics is not None:
            diagnostics.update(route="zero_rank", gram_seconds=perf_counter() - start,
                               reciprocal_condition=1.0)
        return gram, 0.0
    core = factor.T @ factor
    core.flat[:: core.shape[0] + 1] += 1.0
    core = (core + core.T) * 0.5
    if diagnostics is not None:
        diagnostics["core_seconds"] = perf_counter() - start
        start = perf_counter()
    try:
        cholesky = np.linalg.cholesky(core)
    except np.linalg.LinAlgError:
        if diagnostics is not None:
            diagnostics.update(route="cholesky_failure", cholesky_seconds=perf_counter() - start)
        if guarded:
            return None, None
        raise
    if diagnostics is not None:
        diagnostics["cholesky_seconds"] = perf_counter() - start
        start = perf_counter()
    if guarded or diagnostics is not None:
        reciprocal_condition, info = dpocon(cholesky, np.linalg.norm(core, 1), uplo="L")
        if diagnostics is not None:
            diagnostics.update(
                reciprocal_condition=float(reciprocal_condition), condition_estimator_info=int(info),
                condition_seconds=perf_counter() - start,
                condition_is_timed_work=guarded,
            )
        if guarded and (info or not reciprocal_condition > np.sqrt(np.finfo(np.float64).eps)):
            if diagnostics is not None:
                diagnostics["route"] = "condition_rejection"
            return None, None
    start = perf_counter() if diagnostics is not None else 0.0
    projected = factor.T @ rhs
    if diagnostics is not None:
        diagnostics["rhs_projection_seconds"] = perf_counter() - start
        start = perf_counter()
    latent = cho_solve((cholesky, True), projected, check_finite=False)
    if diagnostics is not None:
        diagnostics["rhs_solve_seconds"] = perf_counter() - start
        start = perf_counter()
    conditional = rhs - factor @ latent
    if diagnostics is not None:
        diagnostics["conditional_residual_seconds"] = perf_counter() - start
        start = perf_counter()
    gram = conditional.T @ conditional + latent.T @ latent
    logdet = float(2.0 * np.log(np.diag(cholesky)).sum())
    if diagnostics is not None:
        diagnostics.update(
            route="sum_squares", gram_and_logdet_seconds=perf_counter() - start,
            conditional_squared_norm=float(np.einsum("ij,ij->", conditional, conditional)),
            latent_squared_norm=float(np.einsum("ij,ij->", latent, latent)),
        )
    return gram, logdet


def sum_squares_refresh(target, sigma, *, guarded=False, diagnostics=None):
    """Return the experimental cache and route; fallback includes all wasted work."""
    start = perf_counter() if diagnostics is not None else 0.0
    amplitude = np.asarray(sigma, dtype=np.float64)
    if amplitude.ndim == 0:
        amplitude = np.full(target.n_group, amplitude.item())
    if amplitude.shape != (target.n_group,) or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError("Expected one finite non-negative amplitude per site")
    weights, logdet = target.prepared._mode_weights_and_logdet(amplitude)
    scale = np.sqrt(weights)[:, None]
    factor = scale * target.prepared.transformed_factor
    rhs = scale * target._transformed_rhs
    if diagnostics is not None:
        diagnostics["weights_and_whitening_seconds"] = perf_counter() - start
    gram, logdet_update = sum_squares_gram(factor, rhs, guarded=guarded, diagnostics=diagnostics)
    if gram is None:
        if diagnostics is None:
            return target.refresh(amplitude), "fallback_current"
        fallback, profile = baseline.line_profile(lambda: target.refresh(amplitude), target.refresh)
        diagnostics.update(fallback_profile=profile, logdet=profile["logdet"])
        return fallback, "fallback_current"
    logdet += logdet_update
    if diagnostics is not None:
        diagnostics["logdet"] = logdet
    return MarginalQuadraticCache(
        constant=-0.5 * (target.n_obs * np.log(2.0 * np.pi) + logdet + float(gram[0, 0])),
        linear=gram[1:, 0].copy(), precision=(gram[1:, 1:] + gram[1:, 1:].T) * 0.5,
        sigma=amplitude.copy(),
    ), "zero_rank" if not factor.shape[1] else "sum_squares"


def cache_gram(cache, logdet, nobs):
    """Recover every symmetric Gram coefficient represented by a cache."""
    gram = np.empty((cache.linear.size + 1, cache.linear.size + 1))
    gram[0, 0] = -2.0 * cache.constant - nobs * np.log(2.0 * np.pi) - logdet
    gram[1:, 0] = gram[0, 1:] = cache.linear
    gram[1:, 1:] = cache.precision
    return gram


def comparison(actual, expected, *, rtol=1e-8, atol=1e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return {
        "passed": finite and bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_absolute_error": float(np.max(np.abs(actual - expected))) if finite else None,
        "finite": finite, "rtol": rtol, "atol": atol,
    }


def require_finite_cache(cache):
    """Reject failed numerical outputs outside the timed refresh interval."""
    if not all(np.isfinite(value).all() for value in (cache.constant, cache.linear, cache.precision)):
        raise FloatingPointError("Refresh produced nonfinite cache coefficients")


def failed_scalar_diagnostics(diagnostics):
    """Keep failed experimental scalar diagnostics valid in strict JSON."""
    return {key: failed_scalar_diagnostics(value) if isinstance(value, dict) else
            None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in diagnostics.items()}


def self_check():
    rng = np.random.default_rng(152)
    cases = []
    for rank in (0, 3, 8, 12):
        factor = rng.normal(size=(8, rank)) * 0.2
        diagonal = np.linspace(0.2, 0.5, 8)
        prepared = prepare_fixed_ou_low_rank(factor, diagonal, np.arange(8.), np.repeat([0, 1], 4), 5.0)
        target = FixedOuCachedSigmaTarget(
            prepared=prepared, observations=rng.normal(size=8),
            fixed_contribution=rng.normal(size=8), design=rng.normal(size=(8, 3)),
        )
        amplitude = np.asarray([0.3, 0.75])
        covariance = factor @ factor.T + np.diag(diagonal)
        for block in prepared.site_blocks:
            covariance[np.ix_(block.observation_indices, block.observation_indices)] += (
                amplitude[block.site] ** 2 * block.correlation_cholesky @ block.correlation_cholesky.T
            )
        rhs = np.column_stack((target.observations - target.fixed_contribution, target.design))
        expected_gram = rhs.T @ np.linalg.solve(covariance, rhs)
        expected_logdet = np.linalg.slogdet(covariance)[1]
        for guarded in (False, True):
            diagnostics = {}
            cache, route = sum_squares_refresh(target, amplitude, guarded=guarded, diagnostics=diagnostics)
            np.testing.assert_allclose(cache_gram(cache, diagnostics["logdet"], 8), expected_gram, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(diagnostics["logdet"], expected_logdet, rtol=1e-10, atol=1e-10)
            current = target.refresh(amplitude)
            for state in (np.zeros(3), np.ones(3), rng.normal(size=3)):
                np.testing.assert_allclose(cache.log_likelihood(state), current.log_likelihood(state), rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(cache.gradient(state), current.gradient(state), rtol=1e-10, atol=1e-10)
            timed_cache, timed_route = sum_squares_refresh(target, amplitude, guarded=guarded)
            np.testing.assert_array_equal(timed_cache.precision, cache.precision)
            assert route == timed_route
            cases.append({"rank": rank, "guarded": guarded, "route": route})
    diagnostics = {}
    gram, logdet = sum_squares_gram(np.diag([1e5, 1.]), np.eye(2), guarded=True, diagnostics=diagnostics)
    assert gram is None and logdet is None and diagnostics["route"] == "condition_rejection"
    prepared = prepare_fixed_ou_low_rank(np.diag([1e5, 1.]), np.ones(2), np.arange(2.), np.arange(2), 5.0)
    target = FixedOuCachedSigmaTarget(prepared=prepared, observations=[1., 2.], fixed_contribution=[0., 0.], design=np.eye(2))
    cache, route = sum_squares_refresh(target, [0.3, 0.75], guarded=True)
    assert route == "fallback_current"
    np.testing.assert_array_equal(cache.precision, target.refresh([0.3, 0.75]).precision)
    diagnostics = {}
    with np.errstate(over="ignore", invalid="ignore"):
        _, logdet = sum_squares_gram(np.full((2, 1), 1e200), np.ones((2, 1)), diagnostics=diagnostics)
    assert not np.isfinite(logdet)
    invalid = MarginalQuadraticCache(-0.5 * logdet, np.zeros(1), np.eye(1), np.ones(1))
    try:
        require_finite_cache(invalid)
    except FloatingPointError:
        diagnostics["logdet"] = logdet
        diagnostics["fallback_profile"] = {"logdet": logdet}
        saved = failed_scalar_diagnostics(diagnostics)
        assert saved["logdet"] is None
        assert saved["fallback_profile"]["logdet"] is None
        json.dumps(saved, allow_nan=False)
    else:
        raise AssertionError("Finite-input overflow must be recorded as a failed cache")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--aggregation-error-rank", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--expected-sha256", default=baseline.PREPARED_SHA256)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps({"status": "self-check-passed", "cases": self_check()}, indent=2))
        return
    if args.prepared is None or args.out is None:
        parser.error("--prepared and --out are required")
    repeats = args.repeats if args.repeats is not None else (7 if args.aggregation_error_rank else 3)
    if repeats < 1 or (args.aggregation_error_rank is not None and args.aggregation_error_rank < 1):
        parser.error("Rank and repeat count must be positive")
    if Path(baseline.fixed_ou.__file__).resolve() != baseline.ROOT / baseline.SOURCE_PATH:
        parser.error("Select this checkout on PYTHONPATH")
    args.out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest = {
        "issue": "OPE-152", "status": "started", "started_at": datetime.now(UTC).isoformat(),
        "scope": "experimental sum-of-squares; no production change or posterior sampling",
        "guard_caveat": "DPOCON > sqrt(float64 epsilon) screens the core; it does not guarantee residual accuracy.",
        "revision": baseline.git("rev-parse", "HEAD").strip(), "python": sys.version,
        "versions": {name: baseline.version(name) for name in ("openghg_inversions", "numpy", "scipy", "pymc", "pytensor")},
        "threadpools": baseline.threadpool_info(),
        "environment": {name: baseline.os.environ.get(name) for name in
                        ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTENSOR_FLAGS")},
        "prepared": str(args.prepared.resolve()), "repeats": repeats,
        "science": {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
                    "bc_prior": baseline.BC_PRIOR, "offset": None},
        "phase_timings": [], "evaluations": [],
        "timing_note": "One warmup per method/amplitude; rotated-order repeats exclude diagnostics and validation. Fallback repeats include attempted Gram/guard and current refresh. RSS is process-wide cumulative, not per method.",
    }
    (args.out / "working-tree.patch").write_text(baseline.git("diff", "HEAD", "--binary"))
    manifest["diff_sha256"] = baseline.sha256(args.out / "working-tree.patch")
    manifest["sources"] = {}
    for source in (Path(__file__), Path(baseline.__file__), Path(baseline.fixed_ou.__file__), Path(baseline.cached_sigma.__file__)):
        (args.out / source.name).write_text(source.read_text())
        manifest["sources"][source.name] = {"path": str(source.resolve()), "sha256": baseline.sha256(source)}

    def save():
        manifest.update(elapsed_seconds=perf_counter() - started,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = args.out / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "manifest.json")

    @contextmanager
    def phase(name):
        manifest["active_phase"] = name
        save()
        start = perf_counter()
        try:
            yield
        finally:
            manifest["phase_timings"].append({"phase": name, "seconds": perf_counter() - start})
            save()

    try:
        from openghg_inversions.rhime.co2 import Co2PreparedInputs, build_rhime_co2_cached_sigma

        with phase("load_prepared"):
            manifest["prepared_sha256"] = baseline.sha256(args.prepared)
            if manifest["prepared_sha256"] != args.expected_sha256:
                raise ValueError("Unexpected source prepared-input hash")
            prepared = Co2PreparedInputs.load(args.prepared)
            if prepared.aggregation_error_mode != "dense":
                raise ValueError("Expected the original dense source artifact")
        if args.aggregation_error_rank is not None:
            with phase("derive_low_rank_aggregation_error"):
                prepared = baseline.derive_low_rank_inputs(prepared, args.aggregation_error_rank, manifest["prepared_sha256"])
                manifest["derived_aggregation_error"] = dict(prepared.provenance)
            with phase("save_derived_prepared"):
                path = args.out / f"prepared-inputs-rank{args.aggregation_error_rank}.nc"
                prepared.save(path)
                manifest["derived_prepared"] = {"path": str(path.resolve()), "sha256": baseline.sha256(path)}
        with phase("build_current_including_initial_refresh"):
            cached = build_rhime_co2_cached_sigma(
                prepared_inputs=prepared, tau_hours=5.0,
                site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
                use_bc=True, bc_prior=baseline.BC_PRIOR,
            )
        target = cached.target
        manifest["shapes"] = {"observations": target.n_obs, "rank": target.prepared.rank,
                              "state": target.n_state, "sites": target.n_group,
                              "refresh_rhs": list(target._transformed_rhs.shape)}
        inputs = prepared.inv_inputs
        np.testing.assert_allclose(inputs.alpha_prior_mean, 1.0, rtol=0, atol=1e-12)
        expected_mean = (inputs.fixed_prior_contribution + (inputs.H * inputs.alpha_prior_mean).sum("region")
                         + inputs.H_bc.sum("bc_region")).transpose("nmeasure")
        np.testing.assert_allclose(target.fixed_contribution + target.design @ np.ones(target.n_state), expected_mean, rtol=1e-12, atol=1e-8)
        perturbation = np.random.default_rng(152).normal(0.0, 0.01, target.n_state)
        states = {"ones": np.ones(target.n_state), "plus": 1.0 + perturbation, "minus": 1.0 - perturbation}
        manifest["state_points"] = {name: state.tolist() for name, state in states.items()}
        all_passed = True
        for amplitude in (0.3, 0.75, 1.2):
            sigma = np.full(target.n_group, amplitude)
            methods = {
                "current": lambda: (target.refresh(sigma), "current"),
                "raw": lambda: sum_squares_refresh(target, sigma),
                "guarded": lambda: sum_squares_refresh(target, sigma, guarded=True),
            }
            record = {"site_amplitude_ppm": amplitude, "warmups": {}, "repeat_order": [],
                      "algorithms": {name: {"repeats": []} for name in methods}}
            manifest["evaluations"].append(record)
            with phase(f"warmup_{amplitude}"):
                for name, call in methods.items():
                    start = perf_counter()
                    try:
                        cache, route = call()
                        seconds = perf_counter() - start
                        require_finite_cache(cache)
                        record["warmups"][name] = {"seconds": seconds, "route": route}
                    except (np.linalg.LinAlgError, FloatingPointError) as error:
                        record["warmups"][name] = {"seconds": perf_counter() - start, "error": str(error)}
            with phase(f"repeats_{amplitude}"):
                names = list(methods)
                for repeat in range(repeats):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    record["repeat_order"].append(order)
                    for name in order:
                        start = perf_counter()
                        try:
                            cache, route = methods[name]()
                            seconds = perf_counter() - start
                            require_finite_cache(cache)
                            result = {"seconds": seconds, "route": route, "success": True}
                        except (np.linalg.LinAlgError, FloatingPointError) as error:
                            result = {"seconds": perf_counter() - start, "error": str(error), "success": False}
                            all_passed = False
                        record["algorithms"][name]["repeats"].append(result)
            with phase(f"diagnostics_and_validation_{amplitude}"):
                reference, profile = baseline.line_profile(lambda: target.refresh(sigma), target.refresh)
                reference_logdet = profile["logdet"]
                reference_gram = cache_gram(reference, reference_logdet, target.n_obs)
                direct = {}
                for name, state in states.items():
                    residual = target.observations - target.fixed_contribution - target.design @ state
                    start = perf_counter()
                    value = target.evaluate_from_residual(residual, sigma)
                    seconds = perf_counter() - start
                    direct[name] = (value.log_likelihood, -target.design.T @ value.gradient_residual)
                    record.setdefault("direct_evaluations", {})[name] = {
                        "seconds": seconds, "log_likelihood": value.log_likelihood,
                        "state_gradient": direct[name][1].tolist(),
                        "amplitude_gradient": value.gradient_site_amplitude.tolist(),
                    }
                coefficients = {"current_gram": reference_gram, "current_logdet": reference_logdet}
                for name in methods:
                    result = record["algorithms"][name]
                    diagnostics = profile if name == "current" else {}
                    try:
                        cache, route = ((reference, "current") if name == "current" else
                                        sum_squares_refresh(target, sigma, guarded=name == "guarded", diagnostics=diagnostics))
                        require_finite_cache(cache)
                        logdet = diagnostics["logdet"]
                        if not np.isfinite(logdet):
                            raise FloatingPointError("Refresh produced a nonfinite log determinant")
                        gram = cache_gram(cache, logdet, target.n_obs)
                        checks = {
                            "gram": comparison(gram, reference_gram),
                            "constant": comparison(cache.constant, reference.constant),
                            "linear": comparison(cache.linear, reference.linear),
                            "precision": comparison(cache.precision, reference.precision),
                            "logdet": comparison(logdet, reference_logdet, rtol=1e-10, atol=1e-7),
                        }
                        for point, state in {"zero": np.zeros(target.n_state), **states}.items():
                            checks[f"{point}_current_value"] = comparison(cache.log_likelihood(state), reference.log_likelihood(state))
                            checks[f"{point}_current_gradient"] = comparison(cache.gradient(state), reference.gradient(state))
                            if point in direct:
                                checks[f"{point}_direct_value"] = comparison(cache.log_likelihood(state), direct[point][0])
                                checks[f"{point}_direct_gradient"] = comparison(cache.gradient(state), direct[point][1])
                        result.update(diagnostics=diagnostics, diagnostic_route=route, checks=checks,
                                      validation_passed=all(check["passed"] for check in checks.values()))
                        all_passed &= result["validation_passed"]
                        coefficients[f"{name}_gram"] = gram
                        coefficients[f"{name}_logdet"] = logdet
                    except (np.linalg.LinAlgError, FloatingPointError) as error:
                        result.update(diagnostics=failed_scalar_diagnostics(diagnostics),
                                      validation_passed=False, error=str(error))
                        all_passed = False
                    successes = [item["seconds"] for item in result["repeats"] if item["success"]]
                    result["median_success_seconds"] = float(np.median(successes)) if successes else None
                    result["median_seconds_by_route"] = {
                        route: float(np.median([item["seconds"] for item in result["repeats"]
                                               if item.get("route") == route]))
                        for route in {item["route"] for item in result["repeats"] if item["success"]}
                    }
                    print(json.dumps({"amplitude": amplitude, "algorithm": name,
                                      "median_success_seconds": result["median_success_seconds"],
                                      "validation_passed": result["validation_passed"]}), flush=True)
                path = args.out / f"coefficients-{amplitude}.npz"
                np.savez_compressed(path, **coefficients)
                record["coefficient_artifact"] = {"path": str(path.resolve()), "sha256": baseline.sha256(path)}
        if baseline.sha256(args.prepared) != manifest["prepared_sha256"]:
            raise ValueError("Source prepared-input artifact changed")
        manifest.update(status="computational-profile-passed" if all_passed else "completed-with-validation-failures", active_phase=None)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
