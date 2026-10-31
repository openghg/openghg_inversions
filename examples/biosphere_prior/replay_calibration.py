"""Replay a saved experiment's NPZ calibration through the public implementation.

This adapter reads frozen evidence without importing or patching its runners.
Paths, units and expected file digest must be supplied explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.serialization import load_trace, save_trace


def replay(path, *, units, expected_sha256):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Fixture digest mismatch: {digest}")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {name: opened[name] for name in opened.files}
    labels = pd.MultiIndex.from_arrays(
        [arrays["state_source"], arrays["state_region"]], names=["source", "region_in_source"]
    )
    mean = xr.DataArray(
        arrays["mean"], dims="region", coords=xr.Coordinates.from_pandas_multiindex(labels, "region")
    )
    baseline = CorrelatedLognormalPrior(mean, arrays["original_covariance"], covariance_dim="region_cov")
    biosphere = mean.copy(data=np.isin(np.arange(mean.size), arrays["bio_indices"]))
    process_positions = [list(arrays["functional_names"]).index(name) for name in ("GPP", "TER")]
    weights = xr.DataArray(
        arrays["functional_weights"][process_positions],
        dims=("process", "region"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": units},
    )
    reference = xr.DataArray(
        arrays["functional_prior_totals"][process_positions],
        dims="process",
        coords={"process": ["GPP", "TER"]},
        attrs={"units": units},
    )
    result = calibrate_biosphere_prior(
        baseline, biosphere=biosphere, weights=weights, reference=reference, residual_sd_multiplier=0.5
    )
    np.testing.assert_allclose(result.data.covariance, arrays["shrunk_covariance"], atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(
        result.data.residual_weights, arrays["conditional_gross_weights"], atol=1e-8, rtol=1e-10
    )
    sd = result.data.functional_sd
    targets = dict(
        net=(8313.353249, 8313.353249),
        residual_gross=(8212.589028, 4106.294514),
        gross=(8313.353249, 4304.286043),
        GPP=(6318.215261, 5222.413533),
        TER=(5402.961980, 4067.647623),
    )
    for name, values in targets.items():
        np.testing.assert_allclose(sd.sel(functional=name), values, atol=6e-7, rtol=0)
    identity = calibrate_biosphere_prior(
        baseline, biosphere=biosphere, weights=weights, reference=reference, residual_sd_multiplier=1
    )
    np.testing.assert_array_equal(identity.data.covariance, arrays["original_covariance"])
    assert identity.positive_prior is baseline
    factor_target = xr.DataArray(
        float(sd.sel(prior="requested", functional="residual_gross")), attrs={"units": units}
    )
    absolute = calibrate_biosphere_prior(
        baseline, biosphere=biosphere, weights=weights, reference=reference, target_residual_sd=factor_target
    )
    np.testing.assert_allclose(absolute.data.covariance, result.data.covariance, atol=1e-13)
    # Independent prior draws are secondary evidence for the analytic audit.
    positive = result.positive_prior
    rng = np.random.default_rng(20260924)
    draws = np.exp(
        positive.latent_mean.to_numpy()
        + rng.normal(size=(20000, baseline.mean.size)) @ positive.latent_cholesky.to_numpy().T
    )
    functional_weights = np.vstack(
        [
            weights.to_numpy(),
            arrays["net_weights"],
            arrays["gross_weights"],
            result.data.residual_weights.to_numpy(),
        ]
    )
    sample_sd = ((draws - baseline.mean.to_numpy()) @ functional_weights.T).std(axis=0, ddof=1)
    analytic_sd = result.data.functional_sd.sel(prior="requested").to_numpy()
    np.testing.assert_allclose(sample_sd, analytic_sd, rtol=0.04)
    draw_report = {
        "draws": len(draws),
        "seed": 20260924,
        "negative_coefficient_fraction": float((draws < 0).mean()),
        "sample_sd": dict(zip(result.data.functional.to_numpy().tolist(), sample_sd.tolist())),
        "max_relative_sd_error": float(np.max(abs(sample_sd / analytic_sd - 1))),
    }
    return result, {
        "fixture_sha256": digest,
        "prior_draw_check": draw_report,
        "report": json.loads(result.data.attrs["calibration_report"]),
        "max_covariance_replay_error": float(
            np.max(abs(result.data.covariance.to_numpy() - arrays["shrunk_covariance"]))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--units", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result, report = replay(args.input, units=args.units, expected_sha256=args.sha256)
    save_trace(
        xr.DataTree.from_dict({"biosphere_prior": result.to_dataset()}), args.output / "calibration.nc"
    )
    loaded = BiospherePrior.from_dataset(
        load_trace(args.output / "calibration.nc")["biosphere_prior"].to_dataset()
    )
    xr.testing.assert_allclose(loaded.data, result.data)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
