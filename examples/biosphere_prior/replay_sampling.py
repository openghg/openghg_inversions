"""Bounded implementation replay of an archived enhancement-only preparation.

The adapter uses current public builders, checks the saved preparation identity,
selects exact full-prior marginal moments and leaves its H, affine term and own
aggregation covariance unchanged. It does not import any historical runner.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version
import subprocess
import time

import numpy as np
import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.inference.diagnostics import posterior_summary
from openghg_inversions.models.biosphere import annotate_biosphere_prior
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.rhime.co2 import Co2PreparedInputs
from openghg_inversions.rhime.co2.inference import build_co2_inference, sample_co2_inference
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import save_trace, reset_serialisation_multiindexes
from examples.biosphere_prior.replay_calibration import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prior-arrays", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--preparation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--family",
        choices=["lognormal_original", "lognormal_shrunk", "gaussian_original", "gaussian_shrunk"],
        required=True,
    )
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.prepared.read_bytes()).hexdigest()
    receipt = json.loads(args.preparation_receipt.read_text())
    if digest != receipt["prepared_sha256"]:
        raise ValueError("Prepared input does not match its frozen receipt.")
    full, _ = replay(args.prior_arrays, units="Tg / year", expected_sha256=args.prior_sha256)
    prepared = Co2PreparedInputs.load(args.prepared)
    inv = prepared.inv_inputs
    labels = list(inv.get_index("region"))
    positions = full.data.get_index("region").get_indexer(labels)
    if (positions < 0).any():
        raise ValueError("Enhancement state labels absent from full prior.")
    factor = 1.0 if args.family.endswith("original") else 0.5
    if factor == 1:
        full = calibrate_biosphere_prior(
            full.baseline,
            biosphere=full.data.biosphere,
            weights=full.data.weights,
            reference=full.data.reference,
            residual_sd_multiplier=1,
        )
    selected = full.lognormal_marginal(labels)
    np.testing.assert_array_equal(inv.alpha_prior_mean, selected.mean)
    np.testing.assert_array_equal(
        inv.alpha_prior_covariance, full.data.baseline_covariance.to_numpy()[np.ix_(positions, positions)]
    )
    # The authoritative prior for this run is the exact selected full covariance.
    changed = inv.assign(
        alpha_prior_covariance=inv.alpha_prior_covariance.copy(
            data=selected.arithmetic_covariance.to_numpy()
        ).variable
    )
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=changed))
    model_options = {
        "use_bc": False,
        "likelihood_builder": add_fixed_ou_gaussian_likelihood,
        "likelihood_kwargs": {"tau_hours": 5.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75}},
    }
    if args.family.startswith("gaussian"):
        # The fixture's excluded other states have zero biosphere covariance.
        # Verify that fact before using the same conditional diagnostic on the marginal.
        bio = full.data.biosphere.to_numpy()
        dropped = np.setdiff1d(np.arange(full.baseline.mean.size), positions)
        np.testing.assert_array_equal(
            full.data.baseline_covariance.to_numpy()[np.ix_(np.flatnonzero(bio), dropped)], 0
        )
        from openghg_inversions.correlated_state import CorrelatedLognormalPrior

        local_base = CorrelatedLognormalPrior(
            inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
        )
        local = calibrate_biosphere_prior(
            local_base,
            biosphere=full.data.biosphere.isel(region=positions),
            weights=full.data.weights.isel(region=positions),
            reference=full.data.reference,
            residual_sd_multiplier=factor,
            family="gaussian",
        )
        np.testing.assert_allclose(local.data.covariance, selected.arithmetic_covariance, atol=1e-13)
        # The builder checks calibration against its unchanged baseline.
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inv))
        model_options["biosphere_prior"] = local
    if args.check_only:
        sites = list(dict.fromkeys(inv.site.to_numpy().tolist()))
        rows = np.sort(np.concatenate([np.flatnonzero(inv.site.to_numpy() == site)[:3] for site in sites]))
        prepared = replace(
            prepared,
            rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=prepared.inv_inputs.isel(nmeasure=rows)),
        )
    started = time.monotonic()
    report = {
        "family": args.family,
        "prepared_sha256": digest,
        "prior_sha256": args.prior_sha256,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("pymc", "pytensor", "numpy", "scipy", "openghg")},
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "rows": inv.sizes["nmeasure"],
        "states": len(positions),
        "seed": 20260924,
        "scope": "bounded enhancement implementation replay; fixed paired enhancement aggregation; no endpoint BC",
        "selection": "exact marginal moments; no Schur complement",
        "state_target_accept": 0.99,
        "sigma_target_accept": 0.9,
    }
    built, _cached = build_co2_inference(prepared, model_options)
    point = built.model.initial_point()
    report["initial_logp"] = float(built.model.compile_logp()(point))
    if not np.isfinite(report["initial_logp"]):
        raise ValueError("Nonfinite graph preflight.")
    if not args.check_only:
        sampler = RhimeSampler(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            nuts_sampler="pymc",
            sample_kwargs={
                "cores": args.chains,
                "blas_cores": 1,
                "random_seed": 20260924,
                "compute_convergence_checks": False,
            },
            sample_prior_predictive=False,
            sample_posterior_predictive=False,
        )
        trace, actual = sample_co2_inference(
            prepared, model_options, sampler, sigma_target_accept=0.9, state_target_accept=0.99
        )
        # Persist full-domain calibration and the labelled marginal selection even
        # when the graph consumes only the exact enhancement marginal.
        if "biosphere_prior" not in trace.children:
            annotate_biosphere_prior(trace, full)
        trace["full_biosphere_calibration"] = xr.DataTree(full.to_dataset())
        trace["biosphere_marginal_selection"] = xr.DataTree(
            xr.Dataset(
                {"parent_position": ("region", positions)},
                coords={"region": selected.mean.region},
                attrs={"selection": "exact marginal from full_biosphere_calibration"},
            )
        )
        post = trace["posterior"].to_dataset()
        x = post.flux_scaling
        with np.load(args.prior_arrays, allow_pickle=False) as arrays:
            weights = xr.DataArray(
                arrays["functional_weights"][:, positions],
                dims=("functional", "region"),
                coords={"functional": arrays["functional_names"], "region": selected.mean.region},
            )
            ref = xr.DataArray(
                arrays["functional_prior_totals"],
                dims="functional",
                coords={"functional": arrays["functional_names"]},
            )
        functionals = ref + xr.dot(weights, x - selected.mean, dim="region")
        functionals = functionals.rename("flux_functional")
        functionals.attrs["units"] = "Tg CO2 / year; annualized July mean"
        trace["biosphere_functionals"] = xr.DataTree(functionals.to_dataset())
        diagnostic = posterior_summary(functionals.to_dataset())
        reset_serialisation_multiindexes(diagnostic).to_netcdf(args.output / "functional_diagnostics.nc")
        state_diagnostic = posterior_summary(post[["flux_scaling", "ou_site_amplitude"]])
        reset_serialisation_multiindexes(state_diagnostic).to_netcdf(args.output / "state_diagnostics.nc")
        diagnostic.to_dataframe().to_csv(args.output / "functional_diagnostics.csv")
        stats = trace["sample_stats"].to_dataset()
        report["functionals"] = {
            str(name): {
                "mean": float(functionals.sel(functional=name).mean()),
                "sd": float(functionals.sel(functional=name).std()),
                "mcse_mean": float(diagnostic.flux_functional.sel(functional=name, metric="mcse_mean")),
            }
            for name in functionals.functional.to_numpy()
        }
        report["events"] = {
            name: int(value.sum())
            for name, value in stats.data_vars.items()
            if name.endswith(("diverging", "reached_max_treedepth"))
        }
        report["max_rhat"] = max(
            float(v.sel(metric="r_hat").max()) for v in state_diagnostic.data_vars.values()
        )
        report["min_bulk_ess"] = min(
            float(v.sel(metric="ess_bulk").min()) for v in state_diagnostic.data_vars.values()
        )
        report["negative_biosphere_coefficient_fraction"] = float(
            (x.isel(region=np.flatnonzero(full.data.biosphere.to_numpy()[positions])) < 0).mean()
        )
        report["actual_inference"] = actual
        save_trace(trace, args.output / "posterior.nc")
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
