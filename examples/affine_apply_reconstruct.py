"""Compare component application and physical reconstruction on tiny arrays.

Run ``python -m examples.affine_apply_reconstruct``. No files, stores, sampling
or network are used. The coherent example constructs mutually consistent
synthetic operands directly; real preparations use ``coherent_forward_map``
with their already bound reconstruction.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.rhime.co2.centered_boundary import CenteredBoundaryMap
from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap, CoherentForwardMap


def boundary_example() -> None:
    """Apply a small centered signal and reconstruct absolute concentration."""
    anomaly = np.array([[0.01, -0.02], [0.03, 0.004]])
    mass = np.array([[0.2, 0.1], [0.4, 0.25]])
    reference = 209300.0
    with pytensor.config.change_flags(floatX="float64"):
        state = pt.dvector("b")
        boundary_map = CenteredBoundaryMap(
            pt.constant(anomaly),
            pt.constant(mass),
            pt.constant(reference),
            pt.constant(reference * mass.sum(axis=1)),
        )
        evaluate = pytensor.function(
            [state],
            [boundary_map.apply(state), boundary_map.reconstruct(state)],
        )
        centered, absolute = evaluate(np.array([1.0 + 1e-8, 1.0 - 2e-8]))
    print("Boundary apply, centered concentration (ppm):", centered)
    print("Boundary reconstruct, absolute concentration (ppm):", absolute)


def coherent_example() -> None:
    """Evaluate coherent concentration and signed native conditional flux."""
    mean = np.array([0.7, 1.3, 1.8])
    covariance = np.diag([0.04, 0.09, 0.16])
    restriction = np.array([0.5, 0.3, 0.2])
    native_sensitivity = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    retained_variance = restriction @ covariance @ restriction
    prolongation = covariance @ restriction / retained_variance
    response = native_sensitivity @ prolongation
    reference = restriction @ mean
    state_coords = {"region": ["retained"]}
    native_coords = {"cell": ["west", "middle", "east"]}
    flux_map = AffineFluxMap(
        xr.DataArray(mean, dims="cell", coords=native_coords, attrs={"units": "1"}),
        xr.DataArray([2.0, -1.0, 0.5], dims="cell", coords=native_coords, attrs={"units": "mol m-2 s-1"}),
        xr.DataArray(
            prolongation[:, None],
            dims=("cell", "region"),
            coords={**native_coords, **state_coords},
            attrs={"units": "1"},
        ),
        state_dim="region",
    )
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(flux_map, "synthetic-example", {}, {}),
        xr.DataArray([reference], dims="region", coords=state_coords, attrs={"units": "1"}),
    )
    observation_coords = {"nmeasure": ["first", "second"]}
    coherent_map = CoherentForwardMap(
        bound,
        xr.DataArray(
            response[:, None],
            dims=("nmeasure", "region"),
            coords={**observation_coords, **state_coords},
            attrs={"units": "ppm"},
        ),
        xr.DataArray(
            native_sensitivity @ mean - response * reference,
            dims="nmeasure",
            coords=observation_coords,
            attrs={"units": "ppm"},
        ),
    )
    state = xr.DataArray(
        [[0.8], [reference], [1.5]],
        dims=("draw", "region"),
        coords={"draw": [0, 1, 2], **state_coords},
        attrs={"units": "1"},
    )
    concentration = coherent_map.apply(state)
    flux = coherent_map.reconstruct(state)
    native_mean = bound.state_to_native(state).transpose("cell", "draw")
    np.testing.assert_allclose(concentration, native_sensitivity @ native_mean.data)
    print("Coherent apply, concentration contribution (ppm):", concentration.data)
    print("Coherent reconstruct, conditional flux (mol m-2 s-1):", flux.data)


if __name__ == "__main__":
    boundary_example()
    coherent_example()
