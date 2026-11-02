"""Materialize scientific fixtures for the canonical multisector builder."""

from dask import compute
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.rhime.multisector import (
    _multisector_input_names,
    _multisector_prior_parameters,
    build_multisector_model,
)


def build_multisector_graph(
    model_inputs,
    model_spec,
    *,
    likelihood_builder=None,
    likelihood_kwargs=None,
    sigma_alignment=None,
):
    """Materialize fixture arrays and priors together, then build the graph."""
    selected_priors = {
        name: {
            key: to_dense(value) if isinstance(value, xr.DataArray) else value
            for key, value in params.items()
        }
        for name, params in _multisector_prior_parameters(model_spec).items()
    }
    inputs, priors = compute(
        model_inputs[list(_multisector_input_names(model_inputs, model_spec, materialized=True))].map(to_dense),
        selected_priors,
    )
    return build_multisector_model(
        inputs,
        model_spec,
        priors=priors,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
        sigma_alignment=sigma_alignment,
    ).model
