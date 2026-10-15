"""Adapt concise array fixtures to the standard eager builder contract."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import xarray as xr

from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.specs import DEFAULT_X_PRIOR, RhimeModelSpec, SectorSpec
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs


def build_standard_graph(
    flux_sensitivity, *, observations, observation_error, aggregation_error,
    minimum_error=None, boundary_sensitivity=None, likelihood_settings=None,
    likelihood_builder=None, likelihood_kwargs=None, sigma_alignment=None, **options,
):
    """Materialize fixture arrays and build the ordinary production graph."""
    inputs = xr.Dataset({"H": flux_sensitivity, "mf": observations, "mf_error": observation_error})
    for name, value in (("min_error", minimum_error), ("H_bc", boundary_sensitivity),
                        ("aggregation_error_covariance", aggregation_error.covariance),
                        ("low_rank_factor", aggregation_error.factor),
                        ("diagonal_residual_variance", aggregation_error.diagonal_variance)):
        if value is not None:
            inputs[name] = value
    if aggregation_error.mode == "diagonal":
        inputs["aggregation_error_sd"] = xr.DataArray(
            np.sqrt(aggregation_error.marginal_variance), dims=observations.dims, coords=observations.coords,
        )
    prior = options.pop("x_prior", None) or DEFAULT_X_PRIOR
    spec = RhimeModelSpec(
        species="ch4", domain="EUROPE", sectors=(SectorSpec("total", "inventory", prior, "total"),),
        likelihood=likelihood_settings, aggregation_error_mode=aggregation_error.mode, **options,
    )
    prepared = cast(RhimePreparedInputs, SimpleNamespace(inv_inputs=inputs))
    model_inputs, priors = materialize_standard_inputs(prepared, spec)
    return build_standard_model(
        model_inputs, spec, priors=priors, likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs, sigma_alignment=sigma_alignment,
    ).model
