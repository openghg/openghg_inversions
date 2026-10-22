"""Run a small, offline surface CO2 preparation and absolute predictive replay.

Run from the repository with ``python -m examples.surface_centered_pilot /tmp/surface-pilot``
using the normal locked OpenGHG release. No store or network is
used. Two signed flux sources share one dimensionless scaling field; two native
cells reduce coherently to one retained bucket. Tiny prior draws demonstrate
arithmetic and saved handoffs, not inference or convergence.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import xarray as xr

from openghg_inversions.basis import BasisFunctions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import SiteOptions, prepare_rhime_inputs_from_xarray, prepare_surface_site
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.rhime.co2.centered_boundary import restore_absolute_predictives
from openghg_inversions.rhime.co2.co2_runner import (
    annotate_co2_trace,
    build_co2_from_materialized,
    materialize_co2_inputs,
)
from openghg_inversions.rhime.preparation import filter_observations
from openghg_inversions.serialization import load_trace, save_trace


def surface_inputs():
    """Return six three-hour releases, two cells, and monthly boundary curtains."""
    times = pd.date_range("2020-01-16", periods=6, freq="3h")
    grid = {"lat": [50.0], "lon": [-2.0, -1.0]}
    obs = xr.Dataset(
        {
            "co2": ("time", [402.0, 403.0, 401.0, 404.0, 402.5, 403.5], {"units": "ppm"}),
            "co2_repeatability": ("time", [.3, .8, .6, .4, .3, .2], {"units": "ppm"}),
            "co2_variability": ("time", [.4, .6, .8, .3, .4, .1], {"units": "ppm"}),
        },
        coords={"time": times},
        attrs={"platform": "surface", "sampling_period": 10800, "scale": "WMO"},
    )
    fp = xr.Dataset(
        {"fp": (("lat", "lon", "time"), [[[1., 2., 1., 3., 2., 1.], [2., 1., 3., 1., 1., 2.]]],
                {"units": "s m2 / mol"})},
        coords={**grid, "time": times, "height": [10.]},
        attrs={"model": "NAME", "met_model": "toy"},
    )
    fluxes = {}
    for source, values in {"fossil": [1., 2.], "biosphere": [-.2, .4]}.items():
        fluxes[source] = xr.Dataset(
            {"flux": (("lat", "lon", "time"), np.tile(np.array(values)[None, :, None] * 1e-6, (1, 1, 6)),
                      {"units": "mol / m2 / s"})}, coords={**grid, "time": times},
        )
    bc = xr.Dataset(coords={**grid, "height": [10.], "time": [pd.Timestamp("2020-01-01")]},
                    attrs={"time_period": "1 month"})
    for direction, concentration in zip("nesw", [400.1, 399.8, 400.3, 400.0], strict=True):
        side = "lon" if direction in "ns" else "lat"
        shape = (1, fp.sizes[side], 6)
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            np.full(shape, .25 / fp.sizes[side]), dims=("height", side, "time"), attrs={"units": "1"}
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.full((1, fp.sizes[side], 1), concentration), dims=("height", side, "time"), attrs={"units": "ppm"}
        )
    return obs, fp, fluxes, bc


def acquire_supplied(obs, fp, fluxes, bc):
    """Use OpenGHG convolution and local reference-sensitivity arithmetic."""
    return prepare_surface_site(
        obs, fp, fluxes, species="co2", transport={source: "integrated" for source in fluxes},
        boundary_data=bc, boundary_reference=400., boundary_reference_units="ppm", boundary_mass_tolerance=1e-6,
    )


def prepare_pilot(site, fluxes):
    """Derive errors before means, project curtains, gather, then reduce H/B/m."""
    errors = prepare_observation_errors({"TOY": site})
    filtered, options = filter_observations(
        errors, SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"), filters=["six_hr_mean"],
    )
    retained = filtered["TOY"]
    # Two physical boundary states: north/east and south/west. Apply one spatial
    # map to anomaly and mass, preserving the original multiplicative factors.
    curtain_basis = xr.Dataset()
    for direction in "nesw":
        template = retained[f"bc_mass_{direction}"].isel(time=0, drop=True)
        curtain_basis[f"bc_basis_{direction}"] = template.copy(
            data=np.ones(template.shape)
        ).expand_dims(region=["north_east" if direction in "ne" else "south_west"]).reindex(
            region=["north_east", "south_west"], fill_value=0.
        )
    projected = bc_sensitivity(filtered, curtain_basis)["TOY"]
    total_flux = sum(value.flux for value in fluxes.values()).mean("time").assign_attrs(units="mol / m2 / s")
    labels = xr.ones_like(total_flux, dtype=int)
    basis = BasisFunctions.from_flat_basis(labels, total_flux, operator_kwargs={"state_dim": "region"})
    # This recipe deliberately gives both sources the same cell scaling. Retain
    # source totals for reporting, and feed the summed response to that state.
    canonical = prepare_rhime_inputs_from_xarray(
        {"TOY": projected.drop_vars("fp_x_flux_sectoral")}, basis_functions=basis,
        averaging_period=options.averaging_period, start_date="2020-01-16",
    )
    observations = canonical.inv_inputs.indexes["nmeasure"]
    native_h = retained.fp_x_flux.rename(time="observation").assign_coords(
        xr.Coordinates.from_pandas_multiindex(observations, "observation")
    ).transpose("observation", "lat", "lon")
    covariance = SeparableExponentialCovariance(native_h.lat, native_h.lon, sigma=.2, correlation_length=1.)
    native_mean = total_flux.copy(data=[[.9, 1.1]]).assign_attrs(units="1")
    reduction = reduce_native_gaussian(
        covariance=covariance,
        basis_prolongation=basis.operator.basis_matrix,
        state_dim="region", native_mean=native_mean, native_sensitivity=native_h, observation_dim="observation",
    )
    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=None)
    return retained, canonical, reduction, prepared


def prior_prediction(prepared):
    """Bind saved centered operands and return absolute concentration draws."""
    options = dict(use_bc=True, no_model_error=True, bc_prior={"pdf": "normal", "mu": 1., "sigma": .01})
    inputs, priors = materialize_co2_inputs(prepared_inputs=prepared, **options)
    built = build_co2_from_materialized(inputs, prepared_inputs=prepared, priors=priors, **options)
    with built.model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=957)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    annotate_co2_trace(trace, built, concentration_units="ppm")
    return trace


def run_example(destination):
    """Save/reopen prepared inputs and predictive data, checking absolute replay."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with pytensor.config.change_flags(floatX="float64"):
        obs, fp, fluxes, bc = surface_inputs()
        site = acquire_supplied(obs, fp, fluxes, bc)
        retained, canonical, reduction, prepared = prepare_pilot(site, fluxes)
        prepared.save(destination / "prepared.nc")
        reopened = Co2PreparedInputs.load(destination / "prepared.nc")
        trace = prior_prediction(reopened)
        xr.testing.assert_allclose(trace.observed_data.y, canonical.inv_inputs.mf)
        absolute = trace.prior_predictive.y.copy(deep=True)
        restore_absolute_predictives(trace)
        xr.testing.assert_identical(trace.prior_predictive.y, absolute)
        save_trace(trace, destination / "prior.nc")
        replay = load_trace(destination / "prior.nc")
        restore_absolute_predictives(replay)
        xr.testing.assert_allclose(replay.prior_predictive.y, absolute)
        print("Source totals (ppm), biosphere then fossil:", retained.mf_mod_sectoral.values)
        print("Absolute boundary at unit factors (ppm):", retained.bc_mod.values)
        print("Absolute sampled boundary (ppm):", replay.prior.mu_bc.values)
        print("Absolute prior predictive (ppm):", replay.prior_predictive.y.values)
    return retained, canonical, reduction, reopened, replay


if __name__ == "__main__":
    run_example(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/surface-centered-pilot"))
