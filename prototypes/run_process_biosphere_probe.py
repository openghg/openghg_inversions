"""Sample process-prior research models from labelled, campaign-owned inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from scipy.optimize import brentq

from openghg_inversions.rhime import RhimeSampler
from prototypes.process_biosphere_prior import (
    BASE_LOG_SD,
    build_process_biosphere_model,
    process_prior_covariance,
)


ARMS = ("independent", "linked", "linked-net-matched", "hierarchical")


def prior_settings(data: xr.Dataset, arm: str) -> tuple[str, float, np.ndarray]:
    """Resolve the fixed whole-domain net-variance control before seeing data."""
    weights = data.reference_flux.values
    mode = "linked" if arm == "linked-net-matched" else arm
    log_sd = BASE_LOG_SD
    if arm == "linked-net-matched":
        baseline = process_prior_covariance(data.H, mode="independent")
        target = float(weights @ baseline @ weights)

        def difference(width: float) -> float:
            covariance = process_prior_covariance(data.H, mode="linked", log_sd=width)
            return float(weights @ covariance @ weights) - target

        log_sd = brentq(difference, BASE_LOG_SD, 1.0)
    covariance = process_prior_covariance(data.H, mode=mode, log_sd=log_sd)
    return mode, log_sd, covariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--support", default="all_hours")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    with xr.open_dataset(args.input) as opened:
        data = opened.load()
    selected = data if args.support == "all_hours" else data.isel(
        nmeasure=np.flatnonzero(data[args.support].values)
    )
    mode, width, covariance = prior_settings(data, args.arm)
    observations = selected.observations.sel(scenario=args.scenario, drop=True)
    start = time.perf_counter()
    model, activity = build_process_biosphere_model(
        selected.H,
        observations=observations,
        fixed_prior_contribution=selected.fixed_prior_contribution,
        mode=mode,
        log_sd=width,
        observation_sd=float(data.attrs["observation_sd_ppm"]),
    )
    initial_logp = float(model.compile_logp()(model.initial_point()))
    if not np.isfinite(initial_logp):
        raise ValueError("Non-finite initial log probability.")
    setup_seconds = time.perf_counter() - start
    if args.build_only:
        print(json.dumps({"arm": args.arm, "log_sd": width, "initial_logp": initial_logp}))
        return
    out = args.output / args.scenario / args.support / args.arm
    out.mkdir(parents=True, exist_ok=False)
    activity.to_netcdf(out / "state_activity.nc")
    start = time.perf_counter()
    trace = RhimeSampler(
        draws=args.draws, tune=args.tune, chains=4, nuts_sampler="numpyro",
        sample_prior_predictive=False, sample_posterior_predictive=False,
        sample_kwargs={
            "random_seed": args.seed, "target_accept": .95, "cores": 4,
            "idata_kwargs": {"log_likelihood": False},
            "nuts_sampler_kwargs": {"jitter": False},
        },
    ).sample(model)
    elapsed = time.perf_counter() - start
    trace.to_netcdf(out / "posterior.nc")
    state = trace.posterior.flux_scaling
    truth = data.truth_scaling.sel(scenario=args.scenario, drop=True)
    weights = data.reference_flux
    functionals = {}
    truths = {}
    for name, mask in {
        "GPP": data.source == "GPP", "Ra": data.source == "Ra",
        "Rh": data.source == "Rh", "TER": data.source != "GPP",
        "net_biosphere": xr.ones_like(data.source, dtype=bool),
    }.items():
        vector = weights.where(mask, 0)
        functionals[name] = (state * vector).sum("flux_state")
        truths[name] = float((truth * vector).sum())
    functionals = xr.Dataset(functionals)
    functionals.attrs.update(units=weights.attrs["units"], interpretation=data.attrs["truth_definition"])
    functionals.to_netcdf(out / "functionals.nc")
    summaries = {}
    for name, variable in functionals.data_vars.items():
        values = variable.values.reshape(-1)
        summaries[name] = {
            "truth": truths[name], "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)), "error": float(values.mean() - truths[name]),
            "interval_95": np.quantile(values, [.025, .975]).tolist(),
            "rhat": float(az.rhat(functionals[[name]])[name]),
            "ess_bulk": float(az.ess(functionals[[name]], method="bulk")[name]),
        }
    variable_names = ["flux_scaling"] + (["shared_log_sd"] if mode == "hierarchical" else [])
    max_rhat = float(az.rhat(trace, var_names=variable_names).to_array().max())
    min_ess = float(az.ess(trace, var_names=variable_names, method="bulk").to_array().min())
    min_tail_ess = float(az.ess(trace, var_names=variable_names, method="tail").to_array().min())
    max_rhat = max(max_rhat, max(value["rhat"] for value in summaries.values()))
    min_ess = min(min_ess, min(value["ess_bulk"] for value in summaries.values()))
    divergences = int(trace.sample_stats.diverging.sum())
    depth_hits = int((trace.sample_stats.tree_depth >= 10).sum())
    estimate = state.mean(("chain", "draw")).values
    prediction = selected.H.values @ estimate + selected.fixed_prior_contribution.values
    generative_mean = selected.H.values @ truth.values + selected.fixed_prior_contribution.values
    net_weights = weights.values
    summary = {
        "arm": args.arm, "scenario": args.scenario, "support": args.support,
        "n_observations": selected.sizes["nmeasure"], "n_states": data.sizes["flux_state"],
        "chains": 4, "draws": args.draws, "tune": args.tune, "seed": args.seed,
        "initial_logp": initial_logp, "setup_seconds": setup_seconds,
        "sampling_compile_seconds": elapsed, "fixed_log_sd_or_hyper_reference": width,
        "prior_scaling_sd": np.sqrt(np.diag(covariance)).tolist(),
        "prior_net_flux_sd": float(np.sqrt(net_weights @ covariance @ net_weights)),
        "prior_covariance": covariance.tolist(),
        "max_rhat": max_rhat, "min_ess_bulk": min_ess, "min_ess_tail": min_tail_ess,
        "min_ess_bulk_per_total_second": min_ess / (setup_seconds + elapsed),
        "target_accept": .95, "max_tree_depth": 10,
        "divergences": divergences, "depth_10_hits": depth_hits,
        "convergence_gate": max_rhat <= 1.01 and min_ess >= 100 and divergences == 0 and depth_hits == 0,
        "posterior_scaling_mean": estimate.tolist(),
        "scaling_truth": truth.values.tolist(),
        "concentration_rmse_ppm": float(np.sqrt(np.mean((prediction - observations.values)**2))),
        "generative_mean_rmse_ppm": float(np.sqrt(np.mean((prediction - generative_mean)**2))),
        "functionals": summaries, "functional_units": weights.attrs["units"],
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "ogi_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "component_sha256": hashlib.sha256(Path(__file__).with_name("process_biosphere_prior.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scientific_scope": {key: value.item() if isinstance(value, np.generic) else value for key, value in data.attrs.items()},
    }
    summary["prediction_scores"] = {}
    for label, design in (("fitted", selected), ("all_hours", data)):
        expected = design.H.values @ truth.values + design.fixed_prior_contribution.values
        fitted = design.H.values @ estimate + design.fixed_prior_contribution.values
        prior = design.H.values.sum(axis=1) + design.fixed_prior_contribution.values
        observed = design.observations.sel(scenario=args.scenario).values
        summary["prediction_scores"][label] = {
            "n": design.sizes["nmeasure"],
            "posterior_vs_generative_rmse_ppm": float(np.sqrt(np.mean((fitted - expected)**2))),
            "prior_vs_generative_rmse_ppm": float(np.sqrt(np.mean((prior - expected)**2))),
            "posterior_vs_observed_rmse_ppm": float(np.sqrt(np.mean((fitted - observed)**2))),
            "prior_vs_observed_rmse_ppm": float(np.sqrt(np.mean((prior - observed)**2))),
        }
    if mode == "hierarchical":
        samples = trace.posterior.shared_log_sd.values.reshape(-1)
        summary["shared_log_sd"] = {"mean": float(samples.mean()), "interval_95": np.quantile(samples, [.025, .975]).tolist()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
