"""Synthetic selective-storage prototype; no production precision policy changes.

Run from the repository root with ``python -m scripts.probe_centered_storage``.
Each invocation measures one storage dtype in a fresh process. Assertions check
internal target consistency; reported differences are NOT science acceptance.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr
from scipy.linalg import cho_factor, cho_solve

from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.rhime.co2.co2_cached_sigma_model import _centered_cached_projection


def probe(dtype: str, reference: float, activity: str) -> dict:
    """Check a compiled mean/gradient and real cache against a dense Gaussian."""
    rng = np.random.default_rng(979)
    n, p = 48, 4
    anomaly = rng.normal(size=(n, p)) * 0.1
    # Deliberately nonpartition mass: do not assume M @ 1 == 1.
    mass = rng.uniform(0.1, 0.4, size=(n, p))
    anomaly[:, -1] = -reference * mass[:, -1]  # Zero complete design, nonzero anomaly.
    operands = xr.Dataset(
        {
            "H_bc_anomaly": (("nmeasure", "bc_region"), anomaly),
            "H_bc_mass": (("nmeasure", "bc_region"), mass),
            "bc_reference": reference,
        },
        coords={"nmeasure": np.arange(n), "bc_region": ["N", "E", "S", "W"]},
    )
    complete = operands.H_bc_anomaly + reference * operands.H_bc_mass
    sensitivity = prepare_linear_sensitivity(complete)
    policy = StateActivity(
        active={"active": True, "mixed": [True, False, True, True], "fixed": False}[activity],
        fixed_value=[1.0002, 0.9998, 1.0001, 1.1],
    )
    resolved = resolve_state_activity(sensitivity.removed, policy)
    design64, fixed = _centered_cached_projection(operands, resolved, output_dim="nmeasure")
    active = resolved.active.values
    assert not active[-1]

    # The only quantization boundary. Shared and numerical targets consume these
    # same values; states, fixed intercept, shifts, covariance and caches stay64.
    stored = design64.astype(dtype)
    shared = pytensor.shared(stored, name="active_boundary_design", borrow=True)
    physical_bc = pt.dvector("physical_bc")
    precision = pt.dmatrix("precision")
    logdet = pt.dscalar("logdet")
    observations = fixed + rng.normal(scale=0.2, size=n)
    mean = fixed + pt.dot(pt.cast(shared, "float64"), physical_bc - 1.0)
    residual = observations - mean
    logp = -0.5 * (n * np.log(2 * np.pi) + logdet + residual @ precision @ residual)
    evaluate = pytensor.function(
        [physical_bc, precision, logdet], [mean, logp, pt.grad(logp, physical_bc)]
    )
    assert shared.dtype == dtype and physical_bc.dtype == "float64"

    factor = rng.normal(scale=0.03, size=(n, 3))
    errors = np.linspace(0.15, 0.3, n)
    sites = np.repeat(np.arange(2), n // 2)
    times = np.tile(np.arange(n // 2), 2)
    prepared = prepare_fixed_ou_low_rank(factor, errors**2, times, sites, 5.0)
    target = FixedOuCachedSigmaTarget(
        prepared=prepared, observations=observations, fixed_contribution=fixed, design=stored
    )
    np.testing.assert_array_equal(target.design, shared.get_value())
    assert target.design.dtype == target._transformed_rhs.dtype == np.dtype("float64")
    shift = reference * mass.sum(axis=1)
    rows = []
    for sigma in (0.2, 0.5):
        amplitudes = np.full(2, sigma)
        # Independent dense covariance, including cross-block low-rank terms.
        correlation = np.exp(-np.abs(times[:, None] - times[None, :]) / 5.0)
        correlation *= sites[:, None] == sites[None, :]
        covariance = factor @ factor.T + np.diag(errors**2) + sigma**2 * correlation
        chol = cho_factor(covariance, lower=True)
        inverse = cho_solve(chol, np.eye(n))
        determinant = 2 * np.log(np.diag(chol[0])).sum()
        cache = target.refresh(amplitudes)
        assert cache.precision.dtype == np.dtype("float64")
        for deviation in (0.0, 1e-8, 1e-4, 1e-2):
            b = np.ones(resolved.n_active) + deviation * np.arange(1, resolved.n_active + 1)
            delta = b - 1.0
            actual_mean, graph_logp, graph_grad = evaluate(b, inverse, determinant)
            full_b = resolved.fixed_value.values.copy()
            full_b[active] = b
            # Independent full-operand centered formula, before column pruning.
            oracle = anomaly @ full_b + reference * (mass @ (full_b - 1.0))
            baseline = fixed + design64 @ delta
            np.testing.assert_allclose(baseline, oracle, rtol=2e-13, atol=2e-10)
            candidate = fixed + target.design @ delta
            np.testing.assert_allclose(actual_mean, candidate, rtol=2e-13, atol=2e-10)
            r = observations - candidate
            solved = cho_solve(chol, r)
            dense_logp = -0.5 * (n * np.log(2 * np.pi) + determinant + r @ solved)
            dense_grad = target.design.T @ solved
            # These tolerances compare the SAME rounded target, not float32
            # against the original model, and are not scientific error budgets.
            np.testing.assert_allclose(
                [graph_logp, cache.log_likelihood(delta), target.log_likelihood(delta, amplitudes)],
                dense_logp, rtol=2e-10, atol=2e-8,
            )
            np.testing.assert_allclose([graph_grad, cache.gradient(delta)],
                                       np.broadcast_to(dense_grad, (2, dense_grad.size)),
                                       rtol=2e-10, atol=2e-6)
            e = candidate - baseline
            r64 = observations - baseline
            solved_error = cho_solve(chol, e)
            d = float(e @ solved_error)
            change = float(r64 @ solved_error - d / 2)
            baseline_logp = -0.5 * (
                n * np.log(2 * np.pi) + determinant + r64 @ cho_solve(chol, r64)
            )
            # Subtracting large log likelihoods loses digits; bound that
            # subtraction roundoff separately from any approximation budget.
            subtraction_bound = 100 * np.finfo(float).eps * max(1.0, abs(baseline_logp))
            np.testing.assert_allclose(dense_logp - baseline_logp, change,
                                       rtol=1e-7, atol=subtraction_bound)
            # A common shift is a change of coordinates, not a residual error.
            np.testing.assert_allclose((observations + shift) - (candidate + shift), r,
                                       rtol=2e-12, atol=1e-9)
            if activity == "fixed" or deviation == 0:
                np.testing.assert_array_equal(e, np.zeros(n))
            rows.append({
                "sigma": sigma, "deviation_grid_value": deviation,
                "mean_error_ppm": float(np.max(np.abs(e))), "D": d,
                "delta_log_likelihood": change,
                "baseline_log_likelihood": float(baseline_logp),
                "baseline_max_residual_ppm": float(np.max(np.abs(r64))),
                "cache_vs_dense_log_likelihood": float(cache.log_likelihood(delta) - dense_logp),
            })
    arrays = {
        "shared_design": shared.get_value(borrow=True), "target_design": target.design,
        "transformed_rhs": target._transformed_rhs, "fixed": fixed, "shift": shift,
        "cache_precision": cache.precision,
    }
    return {
        "reference_ppm": reference, "activity": activity, "points": rows,
        "arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape), "bytes": a.nbytes}
                   for name, a in arrays.items()},
        "compiled_operations": [str(node.op) for node in evaluate.maker.fgraph.toposort()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    args = parser.parse_args()
    if pytensor.config.floatX != "float64":
        parser.error("Keep global floatX=float64; only selected design storage is varied.")
    start = time.perf_counter()
    cases = [probe(args.dtype, reference, activity)
             for reference in (420.0, 209300.0) for activity in ("active", "mixed", "fixed")]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "scope": "synthetic kernel/graph screen, not posterior or production-memory acceptance",
        "dtype": args.dtype, "numpy": np.__version__, "pytensor": pytensor.__version__,
        "mode": str(pytensor.config.mode), "floatX": pytensor.config.floatX,
        "seconds": time.perf_counter() - start,
        "process_peak_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
