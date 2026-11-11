"""Offline configured boundary choice, saved calibration and absolute trace replay.

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
from openghg_inversions.models.boundary import RawCenteredSensitivity
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.co2.co2_runner import annotate_co2_trace
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    BiospherePriorConfig,
    Co2ModelConfig,
    build_co2_model,
    load_biosphere_prior,
)
from openghg_inversions.serialization import load_trace, save_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # BEGIN EXAMPLE
    choices = tomllib.loads("""
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
    # The boundary is a choice bound to this fit's operands; only the calibration
    # is a saved artifact. File paths resolve from the invocation directory.
    request = tomllib.loads(f"""
        [model.boundary]
        type = "anomaly_scaling"
        scale_sd = 0.3
        reference = "whole_window_transport_mean"
        mean_shift_prior = {{ pdf = "normal", mu = 0.0, sigma = 0.2 }}
        [model.biosphere]
        path = "{output / 'biosphere.nc'}"
        [likelihood]
        kind = "fixed_ou"
        tau_hours = 5.0
        fixed_site_amplitudes = 0.3
    """)
    model_options = Co2ModelConfig.from_params(request).model_arguments()
    boundary = model_options.pop("boundary").bind(observations, RawCenteredSensitivity.validate(operands))
    model = build_co2_model(
        sensitivity,
        retained_prior=baseline,
        fixed_prior_contribution=xr.full_like(observations, 0.2).rename("fixed_prior_contribution"),
        observations=observations,
        observation_error=xr.full_like(observations, 0.1),
        aggregation_error=resolve_aggregation_error(observations.to_dataset(name="mf"), "none"),
        boundary=boundary,
        **model_options,
    )
    with model:
        trace = pm.sample_prior_predictive(5, random_seed=42)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    outputs = boundary.outputs()
    built = RhimeModelBuildResult(model, OutputContract(
        variable_roles={"model_mean": "modelled_concentration", "flux_scale": "flux_scaling",
                        "pollution_concentration": "co2_flux_contribution", **outputs.roles},
        metadata={**outputs.metadata, "boundary_units": outputs.units},
    ))
    trace = annotate_co2_trace(
        trace, built, concentration_units="ppm", biosphere_prior=model_options["biosphere_prior"],
        likelihood_builder=model_options["likelihood_builder"],
        likelihood_kwargs=model_options["likelihood_kwargs"],
    )
    save_trace(trace, output / "prior-predictive.nc")
    reopened = load_trace(output / "prior-predictive.nc")
    xr.testing.assert_identical(reopened.prior_predictive.to_dataset(), trace.prior_predictive.to_dataset())
    xr.testing.assert_identical(load_biosphere_prior(output / "biosphere.nc").data, calibrated.data)
    # END EXAMPLE
    print(calibrated.data.attrs["calibration_report"])
    print(f"Saved calibration and absolute prior predictions in {output}")


if __name__ == "__main__":
    main()
