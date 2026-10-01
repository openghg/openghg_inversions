#!/usr/bin/env python3
"""Compare cached CO2 inference with a direct observed-CustomDist replacement.

This GH769 experiment leaves production builders and runners unchanged. Run
``uv run python scripts/prototype_cached_sigma_customdist.py`` to compare both
graphs in separate processes. Measurements are synthetic, not production
performance or posterior convergence evidence.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.rhime.co2.co2_cached_sigma_model import (
    Co2CachedSigmaModel,
    build_co2_cached_sigma_model,
)


def replace_cached_potential(cached: Co2CachedSigmaModel) -> None:
    """Replace the likelihood in a fresh graph, before compiling or sampling.

    Mutates ``cached.model``: removes its sole cached likelihood and adds an
    observed, normalized joint Gaussian ``y``. Its density and random callback
    consume explicit mean/amplitude parameters; neither reads the shared cache.
    The original compound sampler remains usable but its state trajectory now
    evaluates the covariance rather than the cached quadratic.
    """
    model = cached.model
    potential = model["cached_fixed_ou_likelihood"]
    model.potentials.remove(potential)
    del model.named_vars["cached_fixed_ou_likelihood"]
    with model:
        pm.CustomDist(
            "y",
            cached.modelled_mean,
            cached.amplitude,
            logp=cached.target.prepared.logp,
            random=cached.target.prepared.random,
            signature="(n),(s)->(n)",
            observed=model["Y"],
            dims=model.named_vars_to_dims["modelled_concentration"],
        )


def build_fixture(
    *, observed: bool, n_observations: int = 8, n_states: int = 2, rank: int = 2
) -> Co2CachedSigmaModel:
    """Build the real CO2 recipe with two sites and reproducible ppm inputs."""
    rng = np.random.default_rng(769)
    rows = np.arange(n_observations)
    states = np.arange(n_states)
    design = rng.uniform(0.05, 0.4, (n_observations, n_states))
    factor = rng.normal(0, 0.1, (n_observations, rank))
    fixed = np.full(n_observations, 0.2)
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), design),
            "mf": ("nmeasure", fixed + design @ np.ones(n_states) + rng.normal(0, 0.1, n_observations)),
            "mf_error": ("nmeasure", np.full(n_observations, 0.15)),
            "fixed_prior_contribution": ("nmeasure", fixed),
            "low_rank_factor": (("nmeasure", "agg_rank"), factor),
            "diagonal_residual_variance": ("nmeasure", np.full(n_observations, 0.01)),
            "alpha_prior_mean": ("region", np.ones(n_states)),
            "alpha_prior_covariance": (("region", "region_cov"), np.eye(n_states) * 0.08),
        },
        coords={
            "nmeasure": rows,
            "region": states,
            "region_cov": states,
            "agg_rank": np.arange(rank),
            "site": ("nmeasure", np.where(rows % 2, "BBB", "AAA")),
            "time": ("nmeasure", np.datetime64("2021-01-01T00") + (rows // 2).astype("timedelta64[h]")),
        },
    )
    for name in ("mf", "mf_error", "fixed_prior_contribution", "H", "low_rank_factor"):
        inputs[name].attrs["units"] = "ppm"
    inputs["diagonal_residual_variance"].attrs["units"] = "ppm^2"
    cached = build_co2_cached_sigma_model(
        inputs.H,
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=inputs.mf,
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "low_rank"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes=0.4,
    )
    if observed:
        replace_cached_potential(cached)
    return cached


def measure(args: argparse.Namespace) -> dict:
    """Measure one graph, with compilation excluded from timed evaluations."""
    cached = build_fixture(
        observed=args.mode == "observed",
        n_observations=args.observations,
        n_states=args.states,
        rank=args.rank,
    )
    model = cached.model
    point = model.initial_point()
    state_fn = model.compile_fn(
        [model.logp(), model.dlogp(vars=list(cached.states))],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    state_fn(point)  # Compile/warm up before timing.
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(args.evaluations):
            state_fn(point)
        durations.append((time.perf_counter() - start) / args.evaluations)
    with model:
        step = make_cached_sigma_compound_step(
            model=model,
            sigma=cached.amplitude,
            states=cached.states,
            modelled_mean=cached.modelled_mean,
            target=cached.target,
            shared_cache=cached.shared_cache,
            initial_cache=cached.initial_cache,
            prior_scale=cached.site_amplitude_prior_scale,
            rng=np.random.default_rng(args.seed),
        )
        start = time.perf_counter()
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=2,
            cores=1,
            step=step,
            random_seed=args.seed,
            progressbar=False,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": False},
        )
        sampling_seconds = time.perf_counter() - start

    # Adaptation has stopped. Time warmed compound sweeps without compilation,
    # tuning, output conversion, or counting wrappers in the measured interval.
    transition_point, _ = step.step(point)
    transition_durations = []
    for _ in range(args.transitions):
        start = time.perf_counter()
        transition_point, _ = step.step(transition_point)
        transition_durations.append(time.perf_counter() - start)

    # Count separately: instrumentation must not distort the timed measurements.
    cached.shared_cache.update(cached.initial_cache)
    prepared_type = type(cached.target.prepared)
    with (
        patch.object(prepared_type, "evaluate", autospec=True, side_effect=prepared_type.evaluate) as calls,
        patch("numpy.linalg.cholesky", wraps=np.linalg.cholesky) as factorizations,
    ):
        state_fn(point)
        evaluation_calls = calls.call_count
        evaluation_factorizations = factorizations.call_count
        calls.reset_mock()
        factorizations.reset_mock()
        _, stats = step.methods[1].step(point)
        state_step_calls = calls.call_count
        state_step_factorizations = factorizations.call_count

    result = {
        "mode": args.mode,
        "observations": args.observations,
        "states": args.states,
        "rank": args.rank,
        "seed": args.seed,
        "chains": 2,
        "draws": args.draws,
        "tune": args.tune,
        "state_logp_gradient_median_seconds": float(np.median(durations)),
        "state_logp_gradient_covariance_evaluations": evaluation_calls,
        "state_logp_gradient_cholesky_calls": evaluation_factorizations,
        "state_transition_covariance_evaluations": state_step_calls,
        "state_transition_cholesky_calls": state_step_factorizations,
        "state_transition_tree_steps": stats[0]["tree_size"],
        "compound_transition_median_seconds": float(np.median(transition_durations)),
        "timed_compound_transitions": args.transitions,
        "sampling_seconds_including_setup_and_tuning": sampling_seconds,
        "retained_cache_refreshes": int(trace.sample_stats.cache_refreshes.sum()),
        "inference_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "posterior_flux_mean": trace.posterior.flux_scaling.mean(("chain", "draw")).values.tolist(),
        "posterior_amplitude_mean": trace.posterior.ou_site_amplitude.mean(("chain", "draw")).values.tolist(),
    }
    if args.mode == "observed":
        with model:
            prior = pm.sample_prior_predictive(4, random_seed=args.seed)
            predictive = pm.sample_posterior_predictive(
                trace, var_names=["y"], random_seed=args.seed, progressbar=False
            )
        result["prior_predictive_shape"] = list(prior.prior_predictive.y.shape)
        result["posterior_predictive_shape"] = list(predictive.posterior_predictive.y.shape)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cached", "observed"))
    parser.add_argument("--observations", type=int, default=32)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--tune", type=int, default=30)
    parser.add_argument("--evaluations", type=int, default=100)
    parser.add_argument("--transitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=769)
    args = parser.parse_args()
    if args.observations < 2:
        parser.error("the two-site fixture requires at least two observations")
    if min(args.observations, args.states, args.draws, args.evaluations, args.transitions) < 1:
        parser.error("observations, states, draws, evaluations and transitions must be positive")
    if min(args.rank, args.tune) < 0:
        parser.error("rank and tune must be non-negative")
    if args.mode:
        print(json.dumps(measure(args)))
        return
    for mode in ("cached", "observed"):
        proc = subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--mode", mode],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        print(proc.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
