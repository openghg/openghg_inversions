"""Calibrate a labelled retained prior and write a replayable analytic report.

Run ``python -m examples.biosphere_prior.calibrate INPUT.nc OUTPUT.nc``.
The input contains baseline_mean, baseline_covariance, biosphere, weights and
reference, with the dimension and unit contracts of calibrate_biosphere_prior.
"""

import argparse
from pathlib import Path

import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.serialization import load_trace, save_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--multiplier", type=float)
    target.add_argument("--residual-sd", type=float)
    parser.add_argument("--sd-units")
    parser.add_argument("--family", choices=("lognormal", "gaussian"), default="lognormal")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new destination.")
    data = load_trace(args.input)["biosphere_prior"].dataset
    covariance = data.baseline_covariance
    baseline = CorrelatedLognormalPrior(data.baseline_mean, covariance, covariance_dim=covariance.dims[-1])
    target_sd = None
    if args.residual_sd is not None:
        if args.sd_units is None:
            parser.error("--residual-sd requires --sd-units")
        target_sd = xr.DataArray(args.residual_sd, attrs={"units": args.sd_units})
    result = calibrate_biosphere_prior(
        baseline,
        biosphere=data.biosphere,
        weights=data.weights,
        reference=data.reference,
        residual_sd_multiplier=args.multiplier,
        target_residual_sd=target_sd,
        family=args.family,
    )
    save_trace(xr.DataTree.from_dict({"biosphere_prior": result.to_dataset()}), args.output)
    print(result.data.attrs["calibration_report"])


if __name__ == "__main__":
    main()
