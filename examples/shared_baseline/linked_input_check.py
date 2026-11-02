"""Bounded saved UOB input checks; no sampling or invented observations.

Run from the repository with PYTHONPATH=. and an existing scientific environment.
The frozen observations are enhancements, so this checks the new frozen baseline
operators separately and deliberately does not fit an absolute-baseline model.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytensor
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.linked_shared_baseline import prepare_linked_shared_baseline
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
)

ROOT = Path("/group/chem/acrg/verification_games_round_2")
FROZEN = (
    ROOT
    / "forward_model_intermediates/rhime_o2_spatial_shared_land_split_ocean_four_site_connected_floor6_20260814/prepared/uob-base/july/joint_co2_o2_coherent.nc"
)
ENDPOINTS = ROOT / "forward_model_intermediates/name_endpoint_repair_20261031_ope239_r2"


def check(frozen, endpoints):
    report = {
        "frozen_input": str(frozen),
        "endpoint_root": str(endpoints),
        "scope": "UOB synthetic NAME inner-only; separate native absolute BC operator check",
        "observations_modified": False,
        "sampling_performed": False,
        "references_ppm": {"co2": 400.0, "o2": 209300.0},
        "outer_mode": "fixed",
        "outer_states": 0,
        "sampled_outer_states": 0,
        "checks": {},
    }
    # Select both covariance axes before loading any observation payload.
    with xr.open_dataset(frozen) as parent:
        sites, tracers, times = parent.site.values, parent.tracer.values, parent.time.values
        rows = []
        for channel, count in [("co2", 8), ("o2", 6)]:
            for site in ["MHD", "TAC"]:
                eligible = np.flatnonzero(
                    (sites == site)
                    & (tracers == channel)
                    & (times >= np.datetime64("2021-07-12"))
                    & (times < np.datetime64("2021-07-19"))
                )
                if eligible.size < count:
                    raise ValueError(f"Insufficient saved rows for {channel}:{site}")
                rows.extend(eligible[:count].tolist())
        rows = np.sort(rows)
        selected = parent.isel(nmeasure=rows, nmeasure_cov=rows)
        # Only retained 365x365 moments, 28 H rows and 28x28 covariance are read.
        names = [
            "H",
            "x_prior_mean",
            "alpha_prior_covariance",
            "state_is_active",
            "prior_forward_mf",
            "mf_clean",
            "forward_baseline",
            "aggregation_error_covariance",
        ]
        selected = selected[names].load()
        sector = parent.sector.values.copy()
        scope = parent.state_tracer_scope.values.copy()
        regions = parent.region.values.copy()
        identity = parent.attrs.get("verification_games:coherent_reduction_id")
        generation = parent.attrs.get("verification_games:observation_generation")
    report.update(
        parent_rows=rows.tolist(), coherent_reduction_id=identity, observation_generation=generation
    )
    active = selected.state_is_active.values.astype(bool).ravel()
    means = selected.x_prior_mean.values.ravel()
    covariance = selected.alpha_prior_covariance.values.reshape(means.size, means.size)
    index = pd.MultiIndex.from_arrays(
        [
            np.repeat(sector, len(regions))[active],
            np.repeat(scope, len(regions))[active],
            np.tile(regions, len(sector))[active],
        ],
        names=["source", "tracer_scope", "region_in_source"],
    )
    coords = xr.Coordinates.from_pandas_multiindex(index, "state")
    mean = xr.DataArray(means[active], dims="state", coords=coords)
    try:
        prior = CorrelatedLognormalPrior(mean, covariance[np.ix_(active, active)])
    except ValueError as error:
        report.update(status="prior_rejected", rejection=str(error))
        return report
    report["checks"]["exact_active_moments_accepted"] = int(active.sum())
    np.testing.assert_array_equal(prior.mean.values, means[active])
    np.testing.assert_array_equal(prior.arithmetic_covariance.values, covariance[np.ix_(active, active)])
    all_h = selected.H.transpose("nmeasure", "source", "region").values.reshape(len(rows), -1)
    # Preserve the coherent affine remainder, not just H times the retained mean.
    affine = selected.prior_forward_mf.values - all_h[:, active] @ means[active]
    report["checks"]["inactive_state_mean_max_abs"] = float(np.max(np.abs(means[~active])))
    report["checks"]["prior_affine_max_abs_ppm"] = float(np.max(np.abs(affine)))
    np.testing.assert_allclose(
        all_h[:, active] @ means[active] + affine, selected.prior_forward_mf.values, rtol=0, atol=1e-12
    )
    tracer = selected.tracer.values
    cross = selected.aggregation_error_covariance.values[np.ix_(tracer == "co2", tracer == "o2")]
    assert np.any(cross != 0)
    report["checks"]["cross_covariance_max_abs_ppm2"] = float(np.max(np.abs(cross)))
    assert np.all(selected.forward_baseline.values == 0)
    report["checks"]["saved_observation_quantity"] = "inner-only enhancement; saved baseline zero"

    h, fixed, corrections = {}, {}, {}
    for channel, reference in report["references_ppm"].items():
        positions = np.flatnonzero(tracer == channel)
        row = f"{channel}_measure"
        observation = xr.DataArray(
            selected.mf_clean.values[positions],
            dims=row,
            coords={
                row: rows[positions],
                "site": (row, selected.site.values[positions]),
                "time": (row, selected.time.values[positions]),
            },
            attrs={"units": "ppm"},
        )
        h[channel] = xr.DataArray(
            all_h[positions][:, active],
            dims=(row, "state"),
            coords={**dict(observation.coords), **dict(prior.mean.coords)},
            attrs={"units": "ppm"},
        )
        fixed[channel] = xr.DataArray(
            affine[positions], dims=row, coords=observation.coords, attrs={"units": "ppm"}
        )
        anomalies, masses, backgrounds = [], [], []
        for site, time in zip(observation.site.values, observation.time.values, strict=True):
            with xr.open_dataset(endpoints / "sites" / f"{site}.nc") as data:
                one = data.sel(species=channel, time=time)
                anomalies.append(one.H_bc_anomaly.values)
                masses.append(one.H_bc_mass.values)
                backgrounds.append(one.background.item())
                assert data.attrs["boundary_time_policy"].startswith(
                    "Archived CORSO midpoint-label forward fill"
                )
        anomaly, mass = np.stack(anomalies), np.stack(masses)
        assert anomaly.dtype == mass.dtype == np.dtype("float64")
        np.testing.assert_allclose(mass.sum(axis=1), 1.0, rtol=0, atol=2e-15)
        reconstructed = reference + anomaly.sum(axis=1)
        np.testing.assert_allclose(reconstructed, backgrounds, rtol=0, atol=3e-11)
        operands = xr.Dataset(
            {
                "H_bc_anomaly": xr.DataArray(
                    anomaly,
                    dims=(row, "bc_region"),
                    coords={**dict(observation.coords), "bc_region": ["n", "e", "s", "w"]},
                    attrs={"units": "ppm"},
                ),
                "H_bc_mass": xr.DataArray(
                    mass,
                    dims=(row, "bc_region"),
                    coords={**dict(observation.coords), "bc_region": ["n", "e", "s", "w"]},
                    attrs={"units": "1"},
                ),
                "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
            },
            attrs={"boundary_representation": "centered-v1"},
        )
        # Widths are explicit numerical-check choices, not campaign calibration.
        corrections[channel] = prepare_boundary_correction(
            operands,
            observation,
            method="A",
            level_sd=1.0,
            pattern_sd=0.1,
        )
        report["checks"][channel] = {
            "rows": len(positions),
            "sites": sorted(set(observation.site.values.tolist())),
            "baseline_reconstruction_max_abs_ppm": float(np.max(np.abs(reconstructed - backgrounds))),
            "native_operand_dtype": str(anomaly.dtype),
        }
    mask = xr.zeros_like(prior.mean, dtype=bool)
    prepared = prepare_linked_shared_baseline(
        co2_sensitivity=h["co2"],
        o2_sensitivity=h["o2"],
        retained_prior=prior,
        fixed_prior_contribution=fixed,
        boundary_correction=corrections,
        outer=mask,
        sampled_outer=mask,
        mode="fixed",
        delta_sd={"co2": 1.0, "o2": 1.0},
    )
    with tempfile.TemporaryDirectory(prefix="linked-input-check-") as temporary:
        joint = selected[["mf_clean", "aggregation_error_covariance"]]
        np.testing.assert_array_equal(joint.nmeasure.values, joint.nmeasure_cov.values)
        joint_path = Path(temporary) / "joint-observation-error.nc"
        joint.to_netcdf(joint_path)
        with xr.open_dataset(joint_path) as saved:
            xr.testing.assert_identical(joint, saved.load())
        report["checks"]["labelled_joint_observation_covariance_roundtrip"] = "passed"
        for channel, value in prepared.items():
            row = value.attrs["native_observation_dim"]
            # Frozen full-support centering survives an unequal saved subset.
            path = Path(temporary) / f"{channel}.nc"
            encode_multiindexes_for_storage(value).to_netcdf(path)
            with xr.open_dataset(path) as saved:
                restored = restore_serialisation_multiindexes(saved.load())
            xr.testing.assert_identical(value, restored)
            subset = restored.isel({row: [0, 2]})
            xr.testing.assert_identical(subset.bc_pattern_mean, restored.bc_pattern_mean)
            np.testing.assert_allclose(
                value.centering_weights @ value.bc_pattern_centered, 0, rtol=0, atol=1e-11
            )
            physical = fixed[channel] + h[channel] @ prior.mean
            np.testing.assert_allclose(
                value.complete_prior_concentration,
                physical
                + corrections[channel].bc_prescribed_centered
                + corrections[channel].bc_reference_shift,
                rtol=0,
                atol=3e-11,
            )
    report["checks"]["native_preparation_serialization_and_subset"] = "passed"
    report.update(
        status="bounded_operator_checks_passed_likelihood_not_applicable",
        likelihood_checked=False,
        limitation="Frozen observations are inner-only enhancements with zero baseline. "
        "Corrected endpoint baselines are absolute concentrations. No observations were "
        "altered, so no direct/cached likelihood or gradient was evaluated for their combination. "
        "There are no outer states: this does not validate real-data outer modes.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--endpoints", type=Path, default=ENDPOINTS)
    parser.add_argument(
        "--output", type=Path, default=Path("inversion_outputs/linked-validation/real-input-check.json")
    )
    args = parser.parse_args()
    with pytensor.config.change_flags(floatX="float64"):
        result = check(args.frozen, args.endpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
