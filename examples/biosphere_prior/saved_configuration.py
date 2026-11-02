"""Offline configured calibration, frozen artifacts and absolute trace replay.

Run ``python -m examples.biosphere_prior.saved_configuration NEW_DIRECTORY``
with PyTensor floatX=float64. Illustrative widths are not production defaults.
"""

import argparse
from pathlib import Path
import tomllib

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.biosphere import annotate_biosphere_prior
from openghg_inversions.models.boundary import restore_absolute_predictives
from openghg_inversions.models.boundary_corrections import annotate_boundary_correction_trace
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    BoundaryCorrectionConfig,
    BiospherePriorConfig,
    Co2ModelConfig,
    build_co2_model,
    load_biosphere_prior,
)
from openghg_inversions.serialization import load_trace, save_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # BEGIN EXAMPLE
    choices = tomllib.loads("""
        [boundary]
        family = "A"
        reference = "whole_window_transport_mean"
        level_sd = { value = 0.2, units = "ppm" }
        pattern_sd = { value = 0.3, units = "1" }
        [biosphere]
        family = "lognormal"
        target_gross_sd = { value = 0.8, units = "kg/year" }
    """)
    mean = xr.DataArray([1.0, 1.0], dims="state", coords={"state": ["uptake", "respiration"]})
    baseline = CorrelatedLognormalPrior(mean, [[0.08, 0.01], [0.01, 0.06]])
    reporting_weights = xr.DataArray(
        [[-2.0, 0.0], [0.0, 3.0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-2.0, 3.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    bio_config = BiospherePriorConfig.from_params(choices["biosphere"])
    calibrated = bio_config.prepare(
        baseline,
        biosphere=mean.copy(data=[True, True]),
        weights=reporting_weights,
        reference=reference,
    )
    bio_config.save(calibrated, output / "biosphere.nc")

    observations = xr.DataArray(
        [400.8, 401.1, 400.9, 401.2],
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "A", "B", "B"]),
            "time": ("nmeasure", np.datetime64("2021-01-01") + np.arange(4).astype("timedelta64[h]")),
        },
        attrs={"units": "ppm"},
    )
    sensitivity = xr.DataArray(
        [[-0.2, 0.8], [-0.1, 0.9], [-0.3, 0.7], [-0.2, 1.0]],
        dims=("nmeasure", "state"),
        coords={**observations.coords, **mean.coords},
        attrs={"units": "ppm"},
    )
    mass = xr.DataArray(
        np.full((4, 2), 0.5),
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": observations.nmeasure, "bc_region": ["N", "S"]},
        attrs={"units": "1"},
    )
    operands = xr.Dataset(
        {
            "H_bc_mass": mass,
            "H_bc_anomaly": mass.copy(
                data=np.array([[-0.2, 0.1], [0.1, 0.2], [-0.1, 0.1], [0.2, -0.1]])
            ).assign_attrs(units="ppm"),
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        attrs={"boundary_representation": "centered-v1"},
    )
    bc_config = BoundaryCorrectionConfig.from_params(choices["boundary"])
    bc_config.save(bc_config.prepare(operands, observations), output / "boundary.nc")

    # Inference loads resolved artifacts. Widths, reference and calibration cannot
    # be overridden here. File paths resolve from the invocation directory.
    request = {
        "model": {
            "boundary": {"correction_path": str(output / "boundary.nc")},
            "biosphere": {"path": str(output / "biosphere.nc")},
        },
        "likelihood": {"kind": "fixed_ou", "tau_hours": 5.0, "fixed_site_amplitudes": 0.3},
    }
    model = build_co2_model(
        sensitivity,
        retained_prior=baseline,
        fixed_prior_contribution=xr.full_like(observations, 0.2),
        observations=observations,
        observation_error=xr.full_like(observations, 0.1),
        aggregation_error=resolve_aggregation_error(observations.to_dataset(name="mf"), "none"),
        **Co2ModelConfig.from_params(request).model_arguments(),
    )
    with model:
        trace = pm.sample_prior_predictive(5, random_seed=42)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    trace = restore_absolute_predictives(trace)
    trace = annotate_boundary_correction_trace(trace)
    trace = annotate_biosphere_prior(trace, model.biosphere_prior)
    save_trace(trace, output / "prior-predictive.nc")
    reopened = load_trace(output / "prior-predictive.nc")
    xr.testing.assert_identical(reopened.prior_predictive.to_dataset(), trace.prior_predictive.to_dataset())
    xr.testing.assert_identical(load_biosphere_prior(output / "biosphere.nc").data, calibrated.data)
    # END EXAMPLE
    print(calibrated.data.attrs["calibration_report"])
    print(f"Saved calibration, boundary and absolute prior predictions in {output}")


if __name__ == "__main__":
    main()
