"""Sample an explicit biosphere prototype from a frozen labelled input file.

Campaign-specific inputs and scientific identities belong in that file, not
in the model component. The information diagnostic is a linear-Gaussian
reference with the supplied arithmetic moments, not an exact lognormal result.
"""

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

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime import RhimeSampler
from prototypes.biosphere_controls import (
    PARAMETERIZATIONS,
    build_biosphere_model,
    likelihood_rotation,
    linear_gaussian_posterior,
)


def select_support(data: xr.Dataset, support: str) -> xr.Dataset:
    if support == "all_hours":
        return data
    return data.isel(nmeasure=np.flatnonzero(data[support].values))


def information(data: xr.Dataset) -> dict:
    """Compare observation support under a common linear-Gaussian reference."""
    result = {}
    prior_covariance = data.prior_covariance.values
    prior_mean = data.prior_mean.values
    net_weights = data.reference_flux.values.copy()
    net_weights[2] = 0
    for support in ("wur_available", "all_hours", "day_equal_count", "night_equal_count"):
        selected = select_support(data, support)
        h = selected.H.values
        error_covariance = np.diag(selected.observation_error.values**2)
        _, covariance = linear_gaussian_posterior(
            h, h @ prior_mean, error_covariance, prior_mean, prior_covariance
        )
        singular = np.linalg.svd(
            (h / selected.observation_error.values[:, None]) @ np.linalg.cholesky(prior_covariance),
            compute_uv=False,
        )
        result[support] = {
            "n_observations": h.shape[0],
            "prior_error_whitened_singular_values": singular.tolist(),
            "gpp_ter_column_cosine": float(h[:, 0] @ h[:, 1] / np.linalg.norm(h[:, 0]) / np.linalg.norm(h[:, 1])),
            "gaussian_reference_scaling_sd": np.sqrt(np.diag(covariance)).tolist(),
            "gaussian_reference_net_biosphere_sd": float(np.sqrt(net_weights @ covariance @ net_weights)),
            "gaussian_reference_gpp_ter_scale_correlation": float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", type=int, choices=range(16))
    parser.add_argument("--information-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    args = parser.parse_args()
    with xr.open_dataset(args.input) as source:
        data = source.load()
    if args.information_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "information.json").write_text(json.dumps(information(data), indent=2) + "\n")
        return
    if args.index is None:
        parser.error("--index is required for build or sample.")
    mode = PARAMETERIZATIONS[args.index % 4]
    support = ("wur_available", "all_hours")[(args.index // 4) % 2]
    scenario = ("base", "ff10")[args.index // 8]
    selected = select_support(data, support)
    prior = CorrelatedLognormalPrior(data.prior_mean, data.prior_covariance, covariance_dim="flux_state_cov")
    observations = selected.observations.sel(scenario=scenario, drop=True)
    errors = selected.observation_error
    rotation = None
    start = time.perf_counter()
    if mode == "rotated-lognormal":
        rotation = likelihood_rotation(selected.H, prior, np.diag(errors.values**2))
    model = build_biosphere_model(
        selected.H,
        retained_prior=prior,
        fixed_prior_contribution=selected.fixed_prior_contribution,
        observations=observations,
        observation_error=errors,
        aggregation_error=resolve_aggregation_error(
            xr.Dataset({"mf": observations, "mf_error": errors}), "none"
        ),
        parameterization=mode,
        rotation=rotation,
        gross_flux_weights=(float(-data.reference_flux.values[0]), float(data.reference_flux.values[1])),
    )
    build_seconds = time.perf_counter() - start
    initial_logp = float(model.compile_logp()(model.initial_point()))
    if not np.isfinite(initial_logp):
        raise ValueError("Prototype initial density is not finite.")
    setup_seconds = time.perf_counter() - start
    if args.build_only:
        print(json.dumps({"mode": mode, "support": support, "initial_logp": initial_logp,
                          "free_variables": [rv.name for rv in model.free_RVs]}))
        return
    out = args.output / scenario / support / mode
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    trace = RhimeSampler(
        draws=args.draws, tune=args.tune, chains=4, nuts_sampler="numpyro",
        sample_prior_predictive=False, sample_posterior_predictive=False,
        sample_kwargs={"random_seed": 20260928, "target_accept": 0.95, "cores": 4,
                       "idata_kwargs": {"log_likelihood": False},
                       "nuts_sampler_kwargs": {"jitter": False}},
    ).sample(model)
    elapsed = time.perf_counter() - started
    trace.to_netcdf(out / "posterior.nc")
    state = trace.posterior.flux_scaling
    contributions = state * data.reference_flux
    functionals = xr.Dataset({
        "GPP_template": contributions.sel(flux_state="GPP", drop=True),
        "TER_template": contributions.sel(flux_state="TER", drop=True),
        "FF": contributions.sel(flux_state="FF", drop=True),
        "net_biosphere": contributions.sel(flux_state=["GPP", "TER"]).sum("flux_state"),
    })
    functionals.attrs["interpretation"] = (
        "Full-domain July-weighted template functionals; Gaussian biosphere template weights are signed, "
        "not independently inferred physical gross fluxes."
    )
    functionals.to_netcdf(out / "functionals.nc")
    truth = data.truth_scaling.sel(scenario=scenario).values * data.reference_flux.values
    truths = dict(zip(["GPP_template", "TER_template", "FF", "net_biosphere"], [*truth, truth[:2].sum()]))
    summaries = {}
    for name, variable in functionals.data_vars.items():
        values = variable.values.reshape(-1)
        summaries[name] = {
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)), "truth": float(truths[name]),
            "error": float(values.mean() - truths[name]), "interval_95": np.quantile(values, [.025, .975]).tolist(),
            "mcse_mean": float(az.mcse(functionals[[name]], method="mean")[name]),
            "ess_bulk": float(az.ess(functionals[[name]], method="bulk")[name]),
            "rhat": float(az.rhat(functionals[[name]])[name]),
        }
    stats = trace.sample_stats
    max_rhat = float(az.rhat(trace, var_names=["flux_scaling"]).to_array().max())
    min_ess = float(az.ess(trace, var_names=["flux_scaling"], method="bulk").to_array().min())
    divergences = int(stats.diverging.sum())
    depth_hits = int((stats.tree_depth >= 10).sum())
    state_values = state.transpose("chain", "draw", "flux_state").values.reshape(-1, 3)
    posterior_prediction = selected.H.values @ state_values.mean(axis=0) + selected.fixed_prior_contribution.values
    true_prediction = selected.H.values @ data.truth_scaling.sel(scenario=scenario).values + selected.fixed_prior_contribution.values
    max_functional_rhat = max(item["rhat"] for item in summaries.values())
    min_functional_ess = min(item["ess_bulk"] for item in summaries.values())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except subprocess.CalledProcessError:
        revision = "unavailable"
    summary = {
        "parameterization": mode, "support": support, "scenario": scenario,
        "n_observations": selected.sizes["nmeasure"], "chains": 4, "draws": args.draws, "tune": args.tune,
        "seed": 20260928, "target_accept": .95, "build_seconds": build_seconds,
        "setup_seconds": setup_seconds, "sample_compile_seconds": elapsed, "initial_logp": initial_logp,
        "max_state_rhat": max_rhat, "min_state_ess_bulk": min_ess,
        "divergences": divergences, "depth_10_hits": depth_hits,
        "mean_n_steps": float(stats.n_steps.mean()), "min_state_ess_per_second": min_ess / elapsed,
        "min_state_ess_per_total_second": min_ess / (elapsed + setup_seconds),
        "max_functional_rhat": max_functional_rhat, "min_functional_ess_bulk": min_functional_ess,
        "convergence_gate": divergences == 0 and depth_hits == 0 and max(max_rhat, max_functional_rhat) <= 1.01 and min(min_ess, min_functional_ess) >= 100,
        "posterior_scaling_mean": state_values.mean(axis=0).tolist(),
        "gpp_ter_scaling_correlation": float(np.corrcoef(state_values[:, 0], state_values[:, 1])[0, 1]),
        "concentration_rmse_ppm": float(np.sqrt(np.mean((posterior_prediction - observations.values)**2))),
        "generative_mean_rmse_ppm": float(np.sqrt(np.mean((posterior_prediction - true_prediction)**2))),
        "functional_units": data.reference_flux.attrs["units"], "functionals": summaries,
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "ogi_revision": revision, "component_sha256": hashlib.sha256(Path(__file__).with_name("biosphere_controls.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scientific_scope": {key: value.item() if isinstance(value, np.generic) else value for key, value in data.attrs.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
