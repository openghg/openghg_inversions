"""Tests for labelled per-site mismatch output provenance."""

import json

import arviz as az
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.rhime.outputs import annotate_likelihood_trace


@pytest.mark.parametrize("mode", ["fixed", "inferred"])
def test_site_sigma_annotation_round_trips_labels_units_and_provenance(tmp_path, mode: str) -> None:
    """NetCDF preserves site labels, units, model mode, equation, and VG source."""
    site_labels = ["MHD", "TAC"]
    posterior_variables = {
        "sigma_observation": (
            ("chain", "draw", "nmeasure"),
            np.array([[[0.5, 0.7, 0.5]]]),
        ),
        "epsilon": (
            ("chain", "draw", "nmeasure"),
            np.ones((1, 1, 3)),
        ),
    }
    if mode == "inferred":
        posterior_variables["sigma_site"] = (
            ("chain", "draw", "sigma_site_dim"),
            np.array([[[0.5, 0.7]]]),
        )
        constant_data = xr.Dataset(
            {"sigma_site_index": ("nmeasure", [0, 1, 0])},
            coords={"nmeasure": [0, 1, 2]},
        )
        options = {
            "fixed_site_amplitudes": None,
            "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
        }
    else:
        constant_data = xr.Dataset(
            {
                "sigma_site": ("sigma_site_dim", [0.5, 0.7]),
                "sigma_site_index": ("nmeasure", [0, 1, 0]),
            },
            coords={"sigma_site_dim": site_labels, "nmeasure": [0, 1, 2]},
        )
        options = {
            "fixed_site_amplitudes": {"MHD": 0.5, "TAC": 0.7},
            "site_amplitude_prior": None,
        }
    posterior = xr.Dataset(
        posterior_variables,
        coords={
            "chain": [0],
            "draw": [0],
            "nmeasure": [0, 1, 2],
            "sigma_site_dim": site_labels,
        },
    )
    idata = az.InferenceData(
        posterior=posterior,
        constant_data=constant_data,
        observed_data=xr.Dataset({"y": ("nmeasure", [1.1, 0.9, 1.2])}),
    )
    identity = {
        "module": "openghg_inversions.models.site_sigma",
        "qualname": "add_site_sigma_gaussian_likelihood",
    }

    annotate_likelihood_trace(
        idata,
        builder_identity=identity,
        likelihood_kwargs=options,
        concentration_units="ppm",
        component_metadata=add_site_sigma_gaussian_likelihood.rhime_metadata,
    )
    path = tmp_path / f"site-sigma-{mode}.nc"
    idata.to_netcdf(path)
    loaded = az.from_netcdf(path)

    assert json.loads(loaded.attrs["rhime_likelihood_builder"]) == identity
    assert json.loads(loaded.attrs["rhime_likelihood_kwargs"]) == options
    assert loaded.attrs["rhime_mismatch_component"] == "iid_site_sigma"
    assert loaded.attrs["rhime_residual_covariance"] == ("A + D_obs + diag(sigma_site[site(i)]^2)")
    assert loaded.attrs["rhime_verification_games_source"] == (
        "src/verification_games/rhime_calibration/site_sigma.py@41d061aea153ddc56130694bfa18b7e801fcd9df"
    )
    assert loaded.attrs["rhime_verification_games_original_model_commit"] == (
        "88f8d4cb21c7eb84b601c26fa51e806ff0bb3ed7"
    )
    assert loaded.attrs["rhime_site_order"] == "stable_first_occurrence"
    sigma_site_group = loaded.posterior if mode == "inferred" else loaded.constant_data
    assert tuple(sigma_site_group.sigma_site_dim.values) == ("MHD", "TAC")
    assert sigma_site_group["sigma_site"].attrs == {
        "rhime_scientific_role": "site_iid_mismatch_standard_deviation",
        "units": "ppm",
    }
    assert loaded.constant_data["sigma_site_index"].attrs == {
        "rhime_scientific_role": "observation_to_site_sigma_index",
        "units": "1",
    }
    assert loaded.posterior["sigma_observation"].attrs == {
        "rhime_scientific_role": ("observation_aligned_site_iid_mismatch_standard_deviation"),
        "units": "ppm",
    }
    assert loaded.posterior["epsilon"].attrs == {
        "rhime_scientific_role": "total_marginal_observation_standard_deviation",
        "units": "ppm",
    }
    assert loaded.observed_data["y"].attrs == {
        "rhime_scientific_role": "observed_concentration",
        "units": "ppm",
    }
