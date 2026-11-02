"""Replay the accepted shared-baseline r3 comparison using current builders.

The archive is an explicit read-only input, never an imported Python module.
All four arms consume its frozen operators, weights and prior widths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.shared_baseline import prepare_shared_baseline
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
    save_trace,
)

ARMS = ("fixed_outer", "additive_net", "multiplicative", "multiplicative_shrunk")
ACCEPTED_RECEIPT_SHA256 = "bbfcb1d9a941adc3c36d1fc4d411f5d1df7cc48b0fcc66261b89d45baf8321d6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archive(archive):
    """Verify every accepted input, then materialize at this replay boundary."""
    if sha256(archive / "preparation.json") != ACCEPTED_RECEIPT_SHA256:
        raise ValueError("Expected the immutable accepted r3 preparation receipt.")
    receipt = json.loads((archive / "preparation.json").read_text())
    if receipt["rows"] != 5314 or receipt["sites"] != 25:
        raise ValueError("Expected the accepted 5,314-row, 25-site r3 comparison.")
    for path, expected in receipt["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input hash differs: {path}")
    expected_designs = next(digest for path, digest in receipt["source_hashes"].items() if path.endswith("/designs.npz"))
    if sha256(archive / "designs.npz") != expected_designs:
        raise ValueError("Loaded design arrays differ from accepted r3 arrays.")
    prepared_path = next(Path(p) for p in receipt["source_hashes"] if p.endswith("/prepared.nc"))
    boundary_path = next(Path(p) for p in receipt["source_hashes"] if p.endswith("midpoint_ffill__delta_A_8day.nc"))
    with xr.open_dataset(prepared_path, group="rhime_inputs/inv_inputs") as opened:
        inv = opened.load()
    inv = decode_cf_multiindexes(inv, tuple(k for k, v in inv.coords.items() if "compress" in v.attrs))
    with xr.open_dataset(boundary_path) as opened:
        boundary = restore_serialisation_multiindexes(opened.load(), strict=True)
    with np.load(archive / "designs.npz", allow_pickle=False) as arrays:
        designs = {name: arrays[name] for name in arrays.files}
    xr.testing.assert_equal(inv.nmeasure, boundary.nmeasure)
    return receipt, inv, boundary, designs


def prepare_arm(arm, receipt, inv, boundary, designs):
    """Keep the retained transport and aggregation error; replace only states."""
    observations = inv.mf.assign_attrs(units="ppm")
    operands = boundary[["H_bc_anomaly", "H_bc_mass"]].assign(
        bc_reference=xr.DataArray(receipt["reference_ppm"], attrs={"units": "ppm"})
    )
    operands.H_bc_anomaly.attrs["units"] = "ppm"
    operands.H_bc_mass.attrs["units"] = "1"
    operands.attrs["boundary_representation"] = "centered-v1"
    correction = prepare_boundary_correction(
        operands,
        observations,
        method="A",
        level_sd=5.0,
        pattern_sd=float(boundary.attrs["pattern_coefficient_sd"]),
        reference=receipt["reference_ppm"],
        frequency="8D",
        anchor_time="2021-07-01",
    )
    np.testing.assert_allclose(
        correction.bc_pattern_operator * correction.bc_pattern_sd,
        designs["bc_raw"], rtol=0, atol=1e-11,
    )
    prior = CorrelatedLognormalPrior(inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov")
    outer = inv.basis_group == "outer"
    sampled = xr.zeros_like(outer, dtype=bool)
    sampled.data[designs["active_outer"]] = True
    additive = xr.DataArray(
        designs["additive_raw"], dims=("nmeasure", "outer_pattern"),
        coords={**dict(observations.coords), "outer_pattern": [f"{sector}_{r}" for sector in ("bio", "FF") for r in range(6)],
                "outer_source": ("outer_pattern", ["net_bio"] * 6 + ["FF"] * 6)},
        attrs={"units": "ppm"},
    )
    calibrated = None
    if arm == "multiplicative_shrunk":
        weights = xr.DataArray(
            designs["outer_functionals"][:2], dims=("process", "region"),
            coords={"process": ["GPP", "TER"], "region": inv.region}, attrs={"units": "Tg / year"},
        )
        reference = xr.DataArray(designs["outer_prior_totals"][:2], dims="process",
                                 coords={"process": ["GPP", "TER"]}, attrs={"units": "Tg / year"})
        bio = outer & xr.DataArray(
            [str(source).endswith(("_GPP", "_TER")) for source in inv.source.values],
            dims="region", coords={"region": inv.region},
        )
        calibrated = calibrate_biosphere_prior(
            prior, biosphere=bio, weights=weights, reference=reference, residual_sd_multiplier=0.5,
        )
        np.testing.assert_allclose(calibrated.data.covariance, designs["covariance_shrunk"], rtol=1e-11, atol=1e-12)
    density_sd = xr.DataArray(designs["additive_density_sd"].reshape(12), dims="outer_pattern",
                              coords={"outer_pattern": additive.outer_pattern}, attrs={"units": "micromol / m^2 / s"})
    result = prepare_shared_baseline(
        inv.H, prior, inv.fixed_prior_contribution, correction,
        outer=outer, sampled_outer=sampled,
        mode={"fixed_outer": "fixed", "additive_net": "additive"}.get(arm, arm),
        delta_sd=5.0, additive_design=additive if arm == "additive_net" else None,
        additive_density_sd=density_sd if arm == "additive_net" else None,
        calibrated_prior=calibrated,
    )
    np.testing.assert_allclose(result.centering_weights, designs["weights"], rtol=0, atol=1e-16)
    np.testing.assert_allclose(result.bc_pattern_centered, designs["bc_centered"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.physical_bc_prior, designs["physical_bc"], rtol=0, atol=1e-10)
    expected_baseline = inv.fixed_prior_contribution.values + designs["physical_bc"] + inv.H.values @ inv.alpha_prior_mean.values
    np.testing.assert_allclose(result.complete_prior_concentration, expected_baseline, rtol=0, atol=1e-10)
    expected_states = 258 if arm.startswith("multiplicative") else 240
    assert result.sizes["region"] == expected_states
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    receipt, inv, boundary, designs = load_archive(args.archive)
    prepared = prepare_arm(args.arm, receipt, inv, boundary, designs)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "prepared.nc"
    encode_multiindexes_for_storage(prepared).to_netcdf(path)
    with xr.open_dataset(path) as opened:
        reopened = restore_serialisation_multiindexes(opened.load(), strict=True)
    xr.testing.assert_equal(prepared, reopened)
    report = {
        "arm": args.arm, "status": "prepared_and_reopened", "rows": inv.sizes["nmeasure"],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "accepted_preparation_sha256": sha256(args.archive / "preparation.json"),
        "prepared_sha256": sha256(path), "reference_ppm": receipt["reference_ppm"],
        "replay_script_sha256": sha256(__file__),
        "source_sha256": {name: sha256(name) for name in (
            "openghg_inversions/models/shared_baseline.py",
            "openghg_inversions/rhime/co2/shared_baseline_model.py",
        )},
    }
    try:
        run_graphs(args, reopened, inv, designs, report)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


def run_graphs(args, prepared, inv, designs, report):
    """Compare ordinary/cached targets, then sample the cached recipe."""
    import pymc as pm
    import pytensor.tensor as pt

    from openghg_inversions.inference.diagnostics import posterior_summary
    from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
    from openghg_inversions.observation_error import resolve_aggregation_error
    from openghg_inversions.rhime.co2.shared_baseline_model import (
        build_shared_co2_model, build_shared_co2_cached_sigma_model,
    )
    from openghg_inversions.rhime.co2.co2_cached_sigma_runner import _sampler_for_cached_graph
    from openghg_inversions.rhime.sampling import RhimeSampler

    start = time.monotonic()
    if args.check_only:
        rows = np.sort(np.concatenate([
            np.flatnonzero((inv.site.values == s) & (designs["period"] == p))[:2]
            for s in np.unique(inv.site) for p in np.unique(designs["period"])
        ]))
        prepared = prepared.isel(nmeasure=rows)
        inv = inv.isel(nmeasure=rows)
    inputs = dict(observations=inv.mf.assign_attrs(units="ppm"), observation_error=inv.mf_error,
                  aggregation_error=resolve_aggregation_error(inv, "low_rank"))
    noise = dict(tau_hours=5., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75})
    ordinary = build_shared_co2_model(prepared, **inputs,
                                     likelihood_builder=add_fixed_ou_gaussian_likelihood,
                                     likelihood_kwargs=noise)
    cached = build_shared_co2_cached_sigma_model(prepared, **inputs, **noise)
    point = pm.initial_point.make_initial_point_fn(
        model=cached.model, overrides=cached.initial_values, return_transformed=True,
    )(seed=0)
    for name, value in point.items():
        if name != str(cached.model.rvs_to_values[cached.amplitude].name):
            point[name] = np.linspace(-.15, .17, value.size).reshape(value.shape)
    ordinary_point = {name: point[name] for name in ordinary.initial_point()}
    ordinary_logp = ordinary.compile_logp()(ordinary_point)
    cached_logp = cached.model.compile_logp()(point)
    np.testing.assert_allclose(ordinary_logp, cached_logp, atol=1e-6, rtol=1e-10)
    if [v.name for v in ordinary.value_vars] != [v.name for v in cached.model.value_vars]:
        raise ValueError("Ordinary/cached parameter order differs.")
    gradient = ordinary.compile_dlogp()(ordinary_point)
    cached_gradient = cached.model.compile_dlogp()(point)
    assert np.isfinite(gradient).all() and np.isfinite(cached_gradient).all()
    np.testing.assert_allclose(gradient, cached_gradient, atol=1e-5, rtol=1e-8)
    np.testing.assert_allclose(
        float(cached.model.compile_fn(cached.model.replace_rvs_by_values([cached.cached_state_logp]),
              inputs=cached.model.value_vars, on_unused_input="ignore")(point)[0]),
        cached.model.compile_logp(vars=cached.model.observed_RVs)(point), atol=1e-6, rtol=1e-10,
    )
    state_values = [cached.model.rvs_to_values[rv] for rv in cached.states]
    likelihood = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    cached_state_gradient = pt.concatenate([pt.grad(likelihood, value).ravel() for value in state_values])
    evaluated = cached.model.compile_fn([cached_state_gradient], inputs=cached.model.value_vars,
                                       on_unused_input="ignore")(point)[0]
    direct_likelihood = cached.model.logp(vars=cached.model.observed_RVs)
    direct_state_gradient = pt.concatenate([pt.grad(direct_likelihood, value).ravel() for value in state_values])
    expected = cached.model.compile_fn([direct_state_gradient], inputs=cached.model.value_vars,
                                      on_unused_input="ignore")(point)[0]
    np.testing.assert_allclose(evaluated, expected, atol=1e-5, rtol=1e-8)
    report.update(status="nonzero_graph_checks_passed", graph_rows=inv.sizes["nmeasure"],
                  logp_difference=float(cached_logp - ordinary_logp),
                  gradient_max_difference=float(abs(gradient - cached_gradient).max()))
    report["cached_likelihood_gradient_max_difference"] = float(abs(evaluated - expected).max())
    if not args.check_only:
        sampler = RhimeSampler(draws=args.draws, tune=args.tune, chains=args.chains, nuts_sampler="pymc",
            sample_kwargs={"cores": args.chains, "blas_cores": 1, "random_seed": 20260924,
                           "compute_convergence_checks": False, "idata_kwargs": {"log_likelihood": False}},
            sample_prior_predictive=False, sample_posterior_predictive=False)
        active = _sampler_for_cached_graph(sampler, cached_model=cached, sigma_target_accept=.9, state_target_accept=.99)
        trace = active.sample(cached.model)
        trace["shared_baseline"] = xr.DataTree(prepared)
        post = trace["posterior"].to_dataset()
        names = [n for n in post if n.startswith("flux_scaling") or n in ("bc_level", "bc_pattern", "ou_site_amplitude")]
        diagnostics = posterior_summary(post[names])
        encode_multiindexes_for_storage(diagnostics).to_netcdf(args.output / "diagnostics.nc")
        save_trace(trace, args.output / "posterior.nc")
        score_replay(args, prepared, inv, post, report)
        report.update(status="sampling_complete", chains=args.chains, draws=args.draws, tune=args.tune,
                      max_rhat=max(float(v.sel(metric="r_hat").max()) for v in diagnostics.data_vars.values()),
                      min_bulk_ess=min(float(v.sel(metric="ess_bulk").min()) for v in diagnostics.data_vars.values()))
        stats = trace["sample_stats"].to_dataset()
        report["sampler_events"] = {name: int(value.sum()) for name, value in stats.data_vars.items()
                                    if name.endswith(("diverging", "reached_max_treedepth"))}
        report["convergence_gate_passed"] = bool(report["max_rhat"] <= 1.01 and report["min_bulk_ess"] >= 100
                                                 and report["sampler_events"] and not any(report["sampler_events"].values()))
    report["elapsed_seconds"] = time.monotonic() - start


def score_replay(args, prepared, inv, post, report):
    """Compare physical inner flux means with the accepted chain-based MCSEs."""
    from openghg_inversions.inference.diagnostics import posterior_summary

    receipt = json.loads((args.archive / "preparation.json").read_text())
    coherent = next(Path(p) for p in receipt["source_hashes"] if p.endswith("/coherent_full.nc"))
    with xr.open_dataset(coherent) as data:
        source_labels = list(data.source.values.astype(str))
        keys = [(str(s), int(r)) for s in data.source.values for r in data.region.values]
        full = data.regional_inner_flux_conditional_design.fillna(0).sum("region")
        full = full.transpose("source", "source_cov", "region_cov").values.reshape(4, -1)
        totals = data.signed_inner_region_prior_flux_total.sum("region").values
    conversion = 365 * 86400 * 44.0095 / 1e12
    full, totals = full * conversion, totals * conversion

    def functionals(posterior):
        state_keys = [(str(s), int(r)) for s, r in zip(posterior.source.values, posterior.region_in_source.values)]
        if len(state_keys) != len(set(state_keys)):
            raise ValueError("Duplicate retained physical state labels.")
        positions = [keys.index(k) for k in state_keys]
        original_keys = [(str(s), int(r)) for s, r in zip(inv.source.values, inv.region_in_source.values)]
        prior_positions = [original_keys.index(k) for k in state_keys]
        x = posterior.flux_scaling.transpose("chain", "draw", "region").values
        flux = totals + np.einsum("...j,kj->...k", x - inv.alpha_prior_mean.values[prior_positions], full[:, positions])
        flux = np.concatenate([flux, (flux[..., 0] + flux[..., 1])[..., None]], axis=-1)
        return xr.DataArray(flux, dims=("chain", "draw", "functional"),
                            coords={"functional": source_labels + ["NET_BIOSPHERE"]}, name="inner_flux")

    current = functionals(post)
    with xr.open_dataset(args.archive / "fits" / args.arm / "posterior.nc", group="posterior") as opened:
        accepted_post = opened[["flux_scaling", "modelled_concentration"]].load()
    accepted = functionals(accepted_post)
    current_summary = posterior_summary(current.to_dataset()).inner_flux
    accepted_summary = posterior_summary(accepted.to_dataset()).inner_flux
    comparisons = {}
    for name in current.functional.values:
        new_mean = float(current.sel(functional=name).mean())
        old_mean = float(accepted.sel(functional=name).mean())
        new_mcse = float(current_summary.sel(functional=name, metric="mcse_mean"))
        old_mcse = float(accepted_summary.sel(functional=name, metric="mcse_mean"))
        comparisons[str(name)] = dict(mean=new_mean, accepted_mean=old_mean,
            sd=float(current.sel(functional=name).std(ddof=1)), accepted_sd=float(accepted.sel(functional=name).std(ddof=1)),
            difference=new_mean-old_mean, mcse_mean=new_mcse, accepted_mcse_mean=old_mcse,
            difference_over_summed_mcse=(new_mean-old_mean)/(new_mcse+old_mcse))
    encode_multiindexes_for_storage(current_summary.to_dataset()).to_netcdf(args.output / "functional_diagnostics.nc")
    report["inner_flux_comparison_tg_co2_year"] = comparisons
    report["accepted_posterior_sha256"] = sha256(args.archive / "fits" / args.arm / "posterior.nc")
    enhancement_path = next(Path(p) for p in receipt["source_hashes"] if p.endswith("enhancement.nc"))
    with xr.open_dataset(enhancement_path, group="rhime_inputs/inv_inputs") as opened:
        enhancement = opened.load()
    enhancement = decode_cf_multiindexes(enhancement, tuple(k for k, v in enhancement.coords.items() if "compress" in v.attrs))
    fixed_inner = enhancement.sel(nmeasure=inv.indexes["nmeasure"]).fixed_prior_contribution.values
    with np.load(args.archive / "designs.npz", allow_pickle=False) as designs:
        background = designs["baseline_wur"]
    report["atmospheric_score_comparison"] = atmospheric_scores(inv, post, accepted_post, fixed_inner, background)
    # Reopen multiple saved samples and reconstruct the absolute concentration.
    with xr.open_dataset(args.output / "posterior.nc", group="posterior") as reopened:
        samples = reopened.isel(chain=[0, -1], draw=[0, -1]).load()
    state_keys = [(str(s), int(r)) for s, r in zip(samples.source.values, samples.region_in_source.values)]
    original_keys = [(str(s), int(r)) for s, r in zip(inv.source.values, inv.region_in_source.values)]
    outer = inv.basis_group.values == "outer"
    inner = ~outer
    x = samples.flux_scaling.transpose("chain", "draw", "region").values
    inner_positions = [state_keys.index(original_keys[i]) for i in np.flatnonzero(inner)]
    absolute = inv.fixed_prior_contribution.values + inv.H.values[:, outer] @ inv.alpha_prior_mean.values[outer]
    absolute = absolute + prepared.physical_bc_prior.values
    absolute = absolute + np.einsum("ij,...j->...i", inv.H.values[:, inner], x[..., inner_positions])
    absolute = absolute + samples.baseline_delta.values[..., None]
    absolute = absolute + samples.bc_centered_correction.values + samples.outer_centered_correction.values
    np.testing.assert_allclose(samples.modelled_concentration, absolute, atol=1e-9, rtol=0)
    report["saved_absolute_closure_max_ppm"] = float(abs(samples.modelled_concentration.values - absolute).max())


def atmospheric_scores(inv, current, accepted, fixed_inner, background):
    """Delta-method MCSE for scores of posterior-mean absolute predictions."""
    from openghg_inversions.inference.diagnostics import posterior_summary

    original_keys = [(str(s), int(r)) for s, r in zip(inv.source.values, inv.region_in_source.values)]
    inner = np.flatnonzero(inv.basis_group.values != "outer")
    h = inv.H.values[:, inner]

    def scores(post):
        keys = [(str(s), int(r)) for s, r in zip(post.source.values, post.region_in_source.values)]
        positions = [keys.index(original_keys[i]) for i in inner]
        x = post.flux_scaling.transpose("chain", "draw", "region").values[..., positions]
        prediction = post.modelled_concentration.transpose("chain", "draw", "nmeasure").values
        inner_mean = fixed_inner + h @ x.mean((0, 1))
        results = {}
        for target, truth, prediction_mean in (
            ("baseline", background, prediction.mean((0, 1)) - inner_mean),
            ("total", inv.mf.values, prediction.mean((0, 1))),
        ):
            residual = prediction_mean - truth
            rmse = float(np.sqrt(np.mean(residual**2)))
            for metric, value, direction in (
                ("bias_ppm", float(residual.mean()), np.full(len(residual), 1 / len(residual))),
                ("rmse_ppm", rmse, residual / (len(residual) * rmse)),
            ):
                influence = np.einsum("...i,i->...", prediction, direction)
                if target == "baseline":
                    influence -= np.einsum("...j,j->...", x, h.T @ direction)
                diagnostic = posterior_summary(xr.DataArray(influence, dims=("chain", "draw"), name="score").to_dataset()).score
                results[target + "_" + metric] = dict(value=value, mcse=float(diagnostic.sel(metric="mcse_mean")))
        return results

    new, old = scores(current), scores(accepted)
    return {name: dict(value=values["value"], accepted_value=old[name]["value"], mcse=values["mcse"],
                      accepted_mcse=old[name]["mcse"], mcse_method="delta method on chain-based influence samples",
                      difference_over_summed_mcse=(values["value"]-old[name]["value"])/(values["mcse"]+old[name]["mcse"]))
            for name, values in new.items()}


if __name__ == "__main__":
    main()
