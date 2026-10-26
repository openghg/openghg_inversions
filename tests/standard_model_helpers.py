"""Adapt concise array fixtures to the standard eager builder contract."""

from dask import compute

import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.rhime.specs import DEFAULT_X_PRIOR, RhimeModelSpec, SectorSpec
from openghg_inversions.rhime.standard import (
    _standard_input_names,
    _standard_prior_parameters,
    build_standard_model,
)


def build_standard_graph(
    flux_sensitivity, *, observations, observation_error,
    minimum_error=None, boundary_sensitivity=None, likelihood_settings=None,
    likelihood_builder=None, likelihood_kwargs=None, sigma_alignment=None, **options,
):
    """Materialize fixture arrays and build the ordinary production graph."""
    inputs = xr.Dataset({"H": flux_sensitivity, "mf": observations, "mf_error": observation_error})
    for name, value in (("min_error", minimum_error), ("H_bc", boundary_sensitivity)):
        if value is not None:
            inputs[name] = value
    prior = options.pop("x_prior", None) or DEFAULT_X_PRIOR
    spec = RhimeModelSpec(
        species="ch4", domain="EUROPE", sectors=(SectorSpec("total", "inventory", prior, "total"),),
        likelihood=likelihood_settings, **options,
    )
    selected_priors = {
        name: {key: to_dense(value) if isinstance(value, xr.DataArray) else value
               for key, value in params.items()}
        for name, params in _standard_prior_parameters(spec).items()
    }
    model_inputs, priors = compute(
        inputs[list(_standard_input_names(inputs, spec, materialized=True))].map(to_dense), selected_priors,
    )
    return build_standard_model(
        model_inputs, spec, priors=priors, likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs, sigma_alignment=sigma_alignment,
    ).model
