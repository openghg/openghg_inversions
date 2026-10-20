"""Compare inferred fixed-OU CO2 full execution with saved scientific handoffs.

Run ``python -m examples.co2_handoffs /tmp/co2-handoffs``. The synthetic
observations, coherent reduction, signed native fluxes, sampling and products
are real. Small draw counts exercise the workflow, not convergence.
"""

from pathlib import Path
from copy import deepcopy
import sys

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions._labelled_matrices import renamed_column_coordinates
from openghg_inversions.basis import BasisFunctions
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import RhimePreparedInputs, SiteOptions
from openghg_inversions.rhime.co2 import prepare_co2_inputs
from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map
from openghg_inversions.rhime.preparation import filter_observations


START_DATE = "2020-01-01"
END_DATE = "2020-01-06"


def synthetic_inputs(destination: Path) -> tuple:
    """Construct a coherent case after filtering observations at both edges."""
    destination.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime([
        "2020-01-01T00:00", "2020-01-02T12:00", "2020-01-03T12:00",
        "2020-01-04T12:00", "2020-01-05T00:00",
    ])
    original = xr.Dataset(
        {"mf": ("time", [400., 401., 402., 403., 404.]),
         "mf_error": ("time", [.1, .1, .2, .3, .1]),
         "release_lon": ("time", np.zeros(5))},
        coords={"time": dates},
    )
    original.mf.attrs["units"] = "ppm"
    original.mf_error.attrs["units"] = "ppm"
    filtered, _ = filter_observations(
        {"MHD": original}, SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
        filters=["daytime"],
    )
    retained = filtered["MHD"]
    assert retained.sizes["time"] == 3
    assert retained.time.min() > original.time.min()
    assert retained.time.max() < original.time.max()
    observations = pd.MultiIndex.from_arrays(
        [["MHD"] * 3, retained.time.values], names=("site", "time"),
    )
    coords = xr.Coordinates.from_pandas_multiindex(observations, "nmeasure")
    grid = {"lat": [50., 51.], "lon": [-2., -1.]}
    labels = xr.DataArray([[1, 2], [1, 2]], dims=("lat", "lon"), coords=grid)
    flux = labels.copy(data=[[-2., 3.], [-1., 4.]]).assign_attrs(units="mol m-2 s-1")
    basis = BasisFunctions.from_flat_basis(labels, flux, operator_kwargs={"state_dim": "region"})
    state = basis.operator.basis_matrix.region.values
    inputs = xr.Dataset(
        {"H": (("nmeasure", "region"), np.zeros((3, 2))),
         "H_bc": (("nmeasure", "bc_region"), [[.1], [.2], [.15]]),
         "mf": ("nmeasure", retained.mf.values),
         "mf_error": ("nmeasure", retained.mf_error.values)},
        coords={**coords, "region": state, "bc_region": ["background"]},
    )
    for name in ("H", "H_bc", "mf", "mf_error"):
        inputs[name].attrs["units"] = "ppm"
    canonical = RhimePreparedInputs(
        inputs, basis,
        xr.Dataset({"averaging_period": ("site", ["1h"])}, coords={"site": ["MHD"]}),
    )
    observation_coords = xr.Coordinates.from_pandas_multiindex(observations, "observation")
    mean = xr.DataArray(retained.mf.values, dims="observation", coords=observation_coords,
                        attrs={"units": "ppm"})
    covariance = xr.DataArray(
        [[.5, .12, .04], [.12, .4, .08], [.04, .08, .3]],
        dims=("observation", "observation_cov"),
        coords={**observation_coords, **renamed_column_coordinates(
            mean, row_dim="observation", column_dim="observation_cov")},
        attrs={"units": "ppm**2"},
    )
    reduction = CoherentGaussianReduction(
        retained_mean=xr.DataArray([1.1, .9], dims="state", coords={"state": state}, attrs={"units": "1"}),
        retained_covariance=xr.DataArray(
            [[.2, .03], [.03, .15]], dims=("state", "state_cov"),
            coords={"state": state, "state_cov": state}, attrs={"units": "1"}),
        effective_observation_operator=xr.DataArray(
            [[.7, .1], [.3, .4], [.2, .6]], dims=("observation", "state"),
            coords={**observation_coords, "state": state}, attrs={"units": "ppm"}),
        native_observation_mean=mean,
        observation_intercept=(mean - xr.DataArray([.8, .7, .9], dims="observation")).assign_attrs(units="ppm"),
        unresolved_observation_covariance=covariance,
        projection_strategy="synthetic-coherent-restriction",
    )
    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=1)
    native_mean = flux.copy(data=[[.8, 1.4], [.8, 1.4]]).assign_attrs(units="1")
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="unsaved")
    countries = destination / "countries.nc"
    xr.Dataset(
        {"country": (("lat", "lon"), [[0, 1], [1, 0]]),
         "name": ("ncountries", ["FRANCE", "GERMANY"])}, coords=grid,
    ).to_netcdf(countries)
    return canonical, reduction, prepared, affine, countries


def configuration(destination: Path, countries: Path) -> dict:
    """Select explicit inference, requested-window and output choices."""
    return {
        "format_version": 1, "recipe": "co2",
        "prepared_inputs": {"start_date": START_DATE, "end_date": END_DATE, "domain": "EUROPE"},
        "likelihood": {"kind": "additive_sigma", "no_model_error": True, "fixed_model_mismatch": 0.0},
        "model": {
            "boundary": {"enabled": True, "prior": {"pdf": "normal", "mu": 1., "sigma": .1}},
            "offset": {"prior": {"pdf": "normal", "mu": 0., "sigma": .1}, "per_site": True,
                       "drop_first": False},
        },
        "sampling": {"draws": 12, "tune": 12, "chains": 1,
                     "progressbar": False, "random_seed": 126, "sample_prior_predictive": 8,
                     "sample_posterior_predictive": True},
        "outputs": {"output_format": "basic", "output_name": "co2",
                    "output_path": str(destination), "country_file": str(countries)},
    }


def run_example(destination: Path) -> None:
    """Save a new prepared/affine pair and compare replay with full execution."""
    from openghg_inversions.rhime.co2 import run_co2
    from openghg_inversions.rhime.co2.execution import resolve_co2_output
    from openghg_inversions.rhime.co2 import handoffs

    canonical, reduction, prepared, affine, countries = synthetic_inputs(destination)
    params = configuration(destination / "full", countries)
    params["likelihood"] = {
        "kind": "fixed_ou", "tau_hours": 24.0,
        "site_amplitude_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
    }
    manifest = handoffs.save_preparation(
        prepared, output_dir=destination / "prepare", start_date=START_DATE,
        end_date=END_DATE, domain="EUROPE", affine_artifact=affine,
    )
    preparation_bytes = manifest.read_bytes()
    model = {name: params[name] for name in ("model", "likelihood")}
    samples = handoffs.sample(
        manifest, model=model, sampling=params["sampling"], output_dir=destination / "sample",
    )
    replay_params = {"outputs": {**params["outputs"], "output_path": str(destination / "replay")}}
    replay = handoffs.postprocess(samples, **resolve_co2_output(replay_params))
    full = run_co2(
        params=params, canonical_inputs=canonical, reduction=reduction,
        aggregation_error_rank=1, affine_artifact=affine,
    )
    for name in ("basic", "concentration_components", "native_flux", "country_flux"):
        xr.testing.assert_allclose(full.outputs[name], replay.outputs[name])
    assert replay.start_date == START_DATE
    assert replay.end_date == END_DATE
    # Independent BC/offset priors and mismatch choices do not redefine reduction.
    alternative = deepcopy(params)
    alternative["model"]["boundary"]["prior"]["sigma"] = .2
    alternative["model"]["offset"]["prior"]["mu"] = .05
    alternative["likelihood"]["tau_hours"] = 18.0
    changed_model = {name: alternative[name] for name in ("model", "likelihood")}
    other_samples = handoffs.sample(
        manifest, model=changed_model, sampling=params["sampling"], output_dir=destination / "alternative-sample",
    )
    other = handoffs.postprocess(other_samples, **resolve_co2_output(
        {"outputs": {**params["outputs"], "output_path": str(destination / "alternative-products")}},
    ))
    assert other.outputs["native_flux"].attrs["uncertainty_scope"] == "retained_state_conditional"
    assert manifest.read_bytes() == preparation_bytes


if __name__ == "__main__":
    run_example(Path(sys.argv[1]))
