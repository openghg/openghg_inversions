"""Replay the frozen July BC designs using physical correction coefficients.

Run --prepare once in a NEW --root, then --check-only or --sample --arm NAME.
Sampling belongs on Slurm. Frozen scripts, manifests and results are read only.
This is matched historical implementation replay, not repaired physical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inference.diagnostics import posterior_summary
from openghg_inversions.models.boundary_corrections import (
    correction_design,
    materialize_boundary_correction,
    prepare_boundary_correction,
    _boundary_correction_roles,
)
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_cached_sigma_runner import (
    _sampler_for_cached_graph,
    _annotate_cached_co2_trace,
)
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
    save_trace,
)

WORK = Path("/group/chem/acrg/verification_games_round_2/verification-games-work")
INPUT_HASH = "056c8ad508ebbb02fdaf234026dea2a789cd7df63a061945192638f4408e9ad6"
ARMS = ("delta_only", "delta_A", "delta_M_Helmert", "global_delta_8day_A", "period_delta_8day_A")
REPO = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def provenance():
    return {
        "revision": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD"])
        ).hexdigest(),
        "adapter_sha256": digest(__file__),
        "uv_lock_sha256": digest(REPO / "uv.lock"),
        "python": sys.version,
        "executable": sys.executable,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("pymc", "pytensor", "numpy", "scipy", "xarray", "openghg")
        },
    }


def read_inputs(path):
    with xr.open_dataset(path, group="rhime_inputs/inv_inputs") as opened:
        ds = opened.load()
    return decode_cf_multiindexes(ds, tuple(k for k, v in ds.coords.items() if "compress" in v.attrs))


def read_design(path):
    with xr.open_dataset(path) as opened:
        return restore_serialisation_multiindexes(opened.load(), strict=True)


def frozen_root(work, arm):
    name = "frequency" if "8day" in arm else "patterns"
    return work / f"ope225-wur-bc-{name}-20261030-r1"


def widths(prepared):
    return np.r_[
        np.full(prepared.sizes["bc_level_term"], float(prepared.bc_level_sd)),
        np.full(prepared.sizes["bc_pattern_term"], float(prepared.bc_pattern_sd)),
    ]


def prior_diagnostics(prepared, obs):
    matrix = correction_design(prepared) * widths(prepared)
    nlevel = prepared.sizes["bc_level_term"]
    index = pd.MultiIndex.from_arrays([obs.site.values, obs.time.values])
    future = index.get_indexer(
        pd.MultiIndex.from_arrays([obs.site.values, pd.to_datetime(obs.time.values) + pd.Timedelta(hours=24)])
    )
    rows = np.flatnonzero(future >= 0)
    result = {}
    for name, f in (("level", matrix[:, :nlevel]), ("pattern", matrix[:, nlevel:]), ("total", matrix)):
        differences = np.linalg.norm(f[future[rows]] - f[rows], axis=1)
        result[name] = {
            "median_pointwise_sd": float(np.median(np.linalg.norm(f, axis=1))),
            "whole_window_row_mean_sd": float(np.linalg.norm(f.mean(axis=0))),
            "median_within_site_24h_difference_sd": float(np.median(differences)),
            "p95_within_site_24h_difference_sd": float(np.quantile(differences, 0.95)),
            "max_within_site_24h_difference_sd": float(differences.max()),
            "within_site_24h_pairs": len(rows),
        }
    return result


def prepare(root, work):
    root.mkdir(parents=True, exist_ok=False)
    (root / "designs").mkdir()
    source = work / "ope225-wur-base-bc-offset-20261029-r2/prepared.nc"
    assert digest(source) == INPUT_HASH
    inv = read_inputs(source)
    raw = inv[["H_bc_anomaly", "H_bc_mass", "bc_reference"]].assign_attrs(
        boundary_representation="centered-v1"
    )
    report = {"source": str(source), "prepared_sha256": INPUT_HASH, "provenance": provenance(), "arms": {}}
    for arm in ARMS:
        frozen = frozen_root(work, arm)
        receipt = json.loads((frozen / "preparation.json").read_text())
        design_path = frozen / "designs" / f"{arm}.nc"
        assert digest(design_path) == receipt["design_sha256"][arm]
        design = read_design(design_path).sel(nmeasure=inv.nmeasure)
        method = "level" if arm == "delta_only" else "curtain_contrasts" if arm == "delta_M_Helmert" else "A"
        kw = dict(
            method=method,
            level_sd=float(design.attrs["level_sd_ppm"]),
            reference=float(design.attrs["reference_ppm"]),
        )
        if method != "level":
            kw["pattern_sd"] = float(design.attrs["pattern_coefficient_sd"])
        if "8day" in arm:
            kw.update(frequency="8D", anchor_time="2021-07-01", level_by_period=arm.startswith("period"))
        prepared = materialize_boundary_correction(prepare_boundary_correction(raw, inv.mf, **kw), inv.mf)
        factor = correction_design(prepared) * widths(prepared)
        np.testing.assert_allclose(factor, design.correction_design, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            prepared.bc_prescribed_centered, design.centered_physical_bc, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(prepared.bc_reference_shift, design.reference_shift, atol=1e-12, rtol=0)
        auto = prepare_boundary_correction(raw, inv.mf, **{**kw, "reference": "whole_window_transport_mean"})
        np.testing.assert_allclose(auto.bc_reference, prepared.bc_reference, atol=1e-12, rtol=0)
        destination = root / "designs" / f"{arm}.nc"
        encode_multiindexes_for_storage(prepared).to_netcdf(destination)
        xr.testing.assert_identical(read_design(destination), prepared)
        # Compare exact covariance on a small row subset, avoiding an N by N allocation.
        selected = np.linspace(0, inv.sizes["nmeasure"] - 1, 20, dtype=int)
        old = design.correction_design.values[selected]
        np.testing.assert_allclose(factor[selected] @ factor[selected].T, old @ old.T, atol=1e-12, rtol=0)
        report["arms"][arm] = {
            "reference": float(prepared.bc_reference),
            "columns": factor.shape[1],
            "design_max_abs_error": float(abs(factor - design.correction_design.values).max()),
            "frozen_root": str(frozen),
            "frozen_receipt_sha256": digest(frozen / "preparation.json"),
            "frozen_design_sha256": digest(design_path),
            "new_design_sha256": digest(destination),
            "prior_diagnostics_ppm": prior_diagnostics(prepared, inv.mf),
        }
    write_json(root / "preparation.json", report)
    print(json.dumps(report, indent=2))


def load_arm(root, arm):
    receipt = json.loads((root / "preparation.json").read_text())
    source = Path(receipt["source"])
    assert digest(source) == receipt["prepared_sha256"] == INPUT_HASH
    inv = read_inputs(source)
    entry = receipt["arms"][arm]
    assert digest(root / "designs" / f"{arm}.nc") == entry["new_design_sha256"]
    if "8day" in arm:
        path = Path(entry["frozen_root"]) / "prior_arrays.npz"
        frozen_receipt = json.loads((Path(entry["frozen_root"]) / "preparation.json").read_text())
        assert digest(path) == frozen_receipt["prior_arrays_sha256"]
        with np.load(path) as arrays:
            np.testing.assert_array_equal(inv.source.values.astype(str), arrays["state_source"].astype(str))
            np.testing.assert_array_equal(inv.region_in_source, arrays["state_region"])
            inv["alpha_prior_covariance"] = inv.alpha_prior_covariance.copy(data=arrays["shrunk_covariance"])
    return inv, read_design(root / "designs" / f"{arm}.nc"), entry


def build(inv, correction):
    return build_co2_cached_sigma_model(
        inv.H,
        retained_prior=CorrelatedLognormalPrior(
            inv.alpha_prior_mean, inv.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inv.fixed_prior_contribution,
        observations=inv.mf,
        observation_error=inv.mf_error,
        aggregation_error=resolve_aggregation_error(inv, "low_rank"),
        tau_hours=5.0,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
        boundary_correction=correction,
    )


def check(root, arm):
    inv, correction, entry = load_arm(root, arm)
    # Exercise all sites and all coefficient periods with three rows per block.
    groups = pd.DataFrame({"site": inv.site.values, "period": correction.bc_period_index.values})
    rows = np.sort(groups.groupby(["site", "period"]).head(3).index.values)
    inv, correction = inv.isel(nmeasure=rows), correction.isel(nmeasure=rows)
    cached = build(inv, correction)
    model = cached.model
    point = pm.initial_point.make_initial_point_fn(
        model=model, overrides=cached.initial_values, return_transformed=True
    )(seed=0)
    for kind in ("level", "pattern"):
        if (name := f"bc_{kind}") in point:
            point[name] = np.linspace(-0.15, 0.2, point[name].size)
    variables = [
        cached.modelled_mean,
        model["modelled_concentration"],
        model["flux_scaling"],
        cached.cached_state_logp,
    ]
    centered, absolute, x, quadratic = model.compile_fn(
        model.replace_rvs_by_values(variables), inputs=model.value_vars, on_unused_input="ignore"
    )(point)
    theta = np.r_[point.get("bc_level", []), point.get("bc_pattern", [])]
    expected = (
        inv.H.values @ x
        + inv.fixed_prior_contribution.values
        + correction.bc_prescribed_centered.values
        + correction_design(correction) @ theta
    )
    np.testing.assert_allclose(centered, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(absolute, expected + correction.bc_reference_shift, atol=1e-10, rtol=0)
    ordinary = model.compile_logp(vars=model.observed_RVs)(point)
    np.testing.assert_allclose(ordinary, quadratic, atol=1e-7, rtol=1e-10)
    frozen_path = Path(entry["frozen_root"]) / "designs" / f"{arm}.nc"
    assert digest(frozen_path) == entry["frozen_design_sha256"]
    frozen = read_design(frozen_path).sel(nmeasure=inv.nmeasure)
    frozen_mean = (
        inv.H.values @ x
        + inv.fixed_prior_contribution.values
        + frozen.centered_physical_bc.values
        + frozen.correction_design.values @ (theta / widths(correction))
    )
    frozen_residual = inv.mf.values - frozen.reference_shift.values - frozen_mean
    residual = inv.mf.values - correction.bc_reference_shift.values - centered
    np.testing.assert_allclose(frozen_residual, residual, atol=1e-10, rtol=0)
    frozen_logp = cached.target.prepared.evaluate(
        frozen_residual,
        np.exp(point["ou_site_amplitude_log__"]),
    ).log_likelihood
    np.testing.assert_allclose(ordinary, frozen_logp, atol=1e-7, rtol=1e-10)
    result = {
        "arm": arm,
        "rows": len(rows),
        "provenance": provenance(),
        "mean_max_abs_error": float(np.max(abs(centered - expected))),
        "ordinary_minus_cached_logp": float(ordinary - quadratic),
        "frozen_mean_max_abs_error": float(np.max(abs(centered - frozen_mean))),
        "ordinary_minus_frozen_logp": float(ordinary - frozen_logp),
        "source_design": entry,
    }
    write_json(root / f"check-{arm}.json", result)
    print(json.dumps(result, indent=2))


def sample(root, arm):
    if not __import__("os").environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full replay sampling must run on Slurm.")
    inv, correction, entry = load_arm(root, arm)
    destination = root / "fits" / arm
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "request.json", {"arm": arm, "provenance": provenance(), "source_design": entry})
    cached = build(inv, correction)
    sampler = RhimeSampler(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="pymc",
        sample_kwargs={
            "cores": 4,
            "blas_cores": 4,
            "random_seed": 20260924,
            "compute_convergence_checks": False,
            "idata_kwargs": {"log_likelihood": False},
        },
        sample_prior_predictive=False,
        sample_posterior_predictive=False,
    )
    active = _sampler_for_cached_graph(
        sampler, cached_model=cached, sigma_target_accept=0.9, state_target_accept=0.95
    )
    trace = active.sample(cached.model)
    trace = restore_inferencedata_coords(trace, get_coord_registry(cached.model))
    built = RhimeModelBuildResult(
        model=cached.model,
        output_contract=OutputContract(
            variable_roles={
                "observation": "Y",
                "concentration": "y",
                "model_mean": "modelled_concentration",
                "flux_scale": "flux_scaling",
                "pollution_concentration": "co2_flux_contribution",
                **_boundary_correction_roles(cached.model),
            },
            metadata={"recipe": "co2", "boundary_formulation": "independent_correction"},
        ),
    )
    trace = _annotate_cached_co2_trace(trace, built, concentration_units="ppm")
    save_trace(trace, destination / "posterior.nc")
    post = trace.posterior.to_dataset()
    variables = [
        name
        for name in post
        if str(name).startswith(("flux_scaling", "bc_level", "bc_pattern", "ou_site_amplitude"))
    ]
    diagnostic = posterior_summary(post[variables])
    encode_multiindexes_for_storage(diagnostic).to_netcdf(destination / "diagnostics.nc")
    stats = trace.sample_stats.to_dataset()
    health = {
        "max_rhat": max(float(v.sel(metric="r_hat").max()) for v in diagnostic.values()),
        "min_bulk_ess": min(float(v.sel(metric="ess_bulk").min()) for v in diagnostic.values()),
        "events": {
            str(k): int(v.sum())
            for k, v in stats.items()
            if str(k).endswith(("diverging", "reached_max_treedepth"))
        },
    }
    write_json(destination / "sampling-health.json", health)
    compare(root, arm)


def compare(root, arm):
    """Compare saved new/reference functionals with independent-run MC uncertainty."""
    inv, correction, entry = load_arm(root, arm)
    sys.path.append(str(WORK.parent / "verification-games/src"))
    from verification_games.wur_production import build_wur_inner_flux_functionals

    state = pd.DataFrame(
        {
            "source": inv.source.values.astype(str),
            "region": inv.region_in_source.values.astype(str),
            "prior_mean": inv.alpha_prior_mean.values,
        }
    )
    coherent = (
        WORK.parent
        / "verification-games/.codex-data-work/wur25-july-coherent-auke-v1/prior_wide/coherent_full.nc"
    )
    functionals = build_wur_inner_flux_functionals(input_path=coherent, state=state)
    with xr.open_dataset(root / "fits" / arm / "posterior.nc", group="posterior") as opened:
        post = opened[
            ["flux_scaling", "bc_level"] + (["bc_pattern"] if correction.sizes["bc_pattern_term"] else [])
        ].load()
    frozen = Path(entry["frozen_root"])
    with xr.open_dataset(frozen / "scores" / arm / "derived_draws.nc") as opened:
        old = opened.load()
    x = post.flux_scaling.values
    flux = functionals.prior_totals_tg_co2_yr + (x - inv.alpha_prior_mean.values) @ functionals.weights.T
    variables = {str(name): (("chain", "draw"), flux[..., i]) for i, name in enumerate(functionals.names)}
    old_variables = {str(name): old.flux_total.sel(functional=name) for name in functionals.names}
    for component, operator, coefficient in (
        ("level", correction.bc_level_operator, post.bc_level),
        ("pattern", correction.bc_pattern_operator, post.get("bc_pattern")),
    ):
        values = (
            np.zeros(x.shape[:2])
            if coefficient is None
            else coefficient.values @ operator.values.mean(axis=0)
        )
        variables[f"{component}_rowmean_ppm"] = (("chain", "draw"), values)
    if "8day" not in arm:
        old_variables.update(level_rowmean_ppm=old.delta, pattern_rowmean_ppm=old.pattern_mean)
        # Keep exterior flux and the two fixed affine terms separate from physical BC.
        full_receipt = json.loads(
            (
                Path(entry["frozen_root"]).parent / "ope225-wur-base-bc-offset-20261029-r2/preparation.json"
            ).read_text()
        )
        enhancement_path = Path(full_receipt["enhancement_source"])
        assert digest(enhancement_path) == full_receipt["enhancement_sha256"]
        enhancement = read_inputs(enhancement_path).sel(nmeasure=inv.nmeasure)
        outer = inv.basis_group.values.astype(str) == "outer"
        fixed_outer = float((inv.fixed_prior_contribution - enhancement.fixed_prior_contribution).mean())
        outer_mean = fixed_outer + x[..., outer] @ inv.H.values[:, outer].mean(axis=0)
        physical_mean = float((correction.bc_prescribed_centered + correction.bc_reference_shift).mean())
        new_level = variables["level_rowmean_ppm"][1]
        new_pattern = variables["pattern_rowmean_ppm"][1]
        variables["outer_rowmean_ppm"] = (("chain", "draw"), outer_mean)
        variables["complete_baseline_rowmean_ppm"] = (
            ("chain", "draw"),
            physical_mean + new_level + new_pattern + outer_mean,
        )
        old_variables.update(
            outer_rowmean_ppm=old.outer_mean,
            complete_baseline_rowmean_ppm=physical_mean + old.delta + old.pattern_mean + old.outer_mean,
        )
    rows = []
    for label, new in xr.Dataset(variables).items():
        if label not in old_variables:
            continue
        reference = old_variables[label]
        new_sd = float(posterior_summary(new.to_dataset(name="value")).value.sel(metric="mcse_mean"))
        old_sd = float(posterior_summary(reference.to_dataset(name="value")).value.sel(metric="mcse_mean"))
        difference = float(new.mean() - reference.mean())
        rows.append(
            {
                "functional": str(label),
                "new_mean": float(new.mean()),
                "reference_mean": float(reference.mean()),
                "new_mcse": new_sd,
                "reference_mcse": old_sd,
                "difference": difference,
                "difference_over_combined_mcse": difference / np.hypot(new_sd, old_sd),
            }
        )
    pd.DataFrame(rows).to_csv(root / "fits" / arm / "reference-comparison.csv", index=False)
    write_json(
        root / "fits" / arm / "comparison-provenance.json",
        {
            "reference_draws": str(frozen / "scores" / arm / "derived_draws.nc"),
            "reference_draws_sha256": digest(frozen / "scores" / arm / "derived_draws.nc"),
            "coherent_flux_functional_source": str(coherent),
            "coherent_sha256": digest(coherent),
            "verification_games_revision": subprocess.check_output(
                ["git", "-C", str(WORK.parent / "verification-games"), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "new_provenance": provenance(),
            "meaning": "Matched historical posterior target; independent-run Monte Carlo comparison; endpoint defects retained",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--arm", choices=ARMS)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--sample", action="store_true")
    action.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    for arm in ARMS:
        if args.root.resolve().is_relative_to(frozen_root(args.work, arm).resolve()):
            parser.error("--root must be a new replay directory outside frozen experiment roots.")
    if args.prepare:
        prepare(args.root, args.work)
    elif args.check_only and args.arm is None:
        for arm in ARMS:
            check(args.root, arm)
    elif args.arm is None:
        parser.error("--arm is required for sampling/comparison")
    elif args.check_only:
        check(args.root, args.arm)
    elif args.sample:
        sample(args.root, args.arm)
    else:
        compare(args.root, args.arm)


if __name__ == "__main__":
    main()
