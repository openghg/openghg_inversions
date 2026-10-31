"""Offline physical boundary correction, period support and labelled replay."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.models.boundary_corrections import (
    add_boundary_correction,
    annotate_boundary_correction_trace,
    calibrate_pattern_width,
    materialize_boundary_correction,
    prepare_boundary_correction,
)
from openghg_inversions.models.boundary import restore_absolute_predictives
from openghg_inversions.models.components import likelihood_representation
from openghg_inversions.models.coords import (
    registered_model,
    get_coord_registry,
    restore_inferencedata_coords,
)
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
)


def main():
    # BEGIN EXAMPLE
    observations = xr.DataArray(
        [411.0, 410.0, 409.0, 412.0],
        dims="nmeasure",
        coords={
            "nmeasure": range(4),
            "site": ("nmeasure", ["A"] * 4),
            "time": (
                "nmeasure",
                np.array(["2021-07-01", "2021-07-09", "2021-07-17", "2021-07-25"], dtype="datetime64[ns]"),
            ),
        },
        attrs={"units": "ppm"},
    )
    mass = xr.DataArray(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]],
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": observations.nmeasure, "bc_region": ["N", "S"]},
        attrs={"units": "1"},
    )
    operands = xr.Dataset(
        {
            "H_bc_mass": mass,
            "H_bc_anomaly": mass.copy(data=[[8.0, 2.0], [3.0, 6.0], [1.0, 7.0], [6.0, 5.0]]).assign_attrs(
                units="ppm"
            ),
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        attrs={"boundary_representation": "centered-v1"},
    )
    choices = dict(
        method="A",
        level_sd=5.0,
        reference="whole_window_transport_mean",
        frequency="8D",
        anchor_time="2021-07-01",
        level_by_period=False,
    )
    # Explicit illustrative calibration, never an automatic model default.
    initial = prepare_boundary_correction(operands, observations, pattern_sd=1.0, **choices)
    width = calibrate_pattern_width(initial.bc_pattern_operator, 1.0, coefficient_dim="bc_pattern_term")
    correction = prepare_boundary_correction(operands, observations, pattern_sd=width, **choices)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "boundary-correction.nc"
        encode_multiindexes_for_storage(correction).to_netcdf(path)
        with xr.open_dataset(path) as opened:
            replay = restore_serialisation_multiindexes(opened.load())
    eager = materialize_boundary_correction(replay, observations)
    with registered_model() as model:
        boundary = add_boundary_correction(eager)
        mean, y, shift = likelihood_representation([boundary.term], observations, boundary.output)
        from openghg_inversions.models.coords import add_model_data

        add_model_data(shift.rename("boundary_reference_shift"))
        pm.Normal("y", mu=mean, sigma=0.2, observed=y.values, dims="nmeasure")
        trace = pm.sample_prior_predictive(draws=4, random_seed=237)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    annotate_boundary_correction_trace(trace)
    restore_absolute_predictives(trace)
    np.testing.assert_allclose(
        trace.prior.mu_bc, trace.prior.mu_bc_prescribed + trace.prior.mu_bc_level + trace.prior.mu_bc_pattern
    )
    # END EXAMPLE
    print(f"Scalar reference: {float(eager.bc_reference):.6f} ppm; A coefficient SD: {width:.6f}")
    print(f"Physical coefficient variables: {[v.name for v in model.free_RVs]}")


if __name__ == "__main__":
    main()
