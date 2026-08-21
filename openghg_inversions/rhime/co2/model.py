"""Concrete PyMC graph for the CO2 coherent-reduction recipe.

The recipe is deliberately procedural. It samples one labelled retained
state, applies the reduced observation operator, adds the fixed affine prior
contribution, and finally constructs the observation likelihood.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import add_additive_sigma_gaussian_likelihood
from openghg_inversions.models.components import (
    add_correlated_lognormal_state_with_activity,
    add_model_data,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.state_activity import (
    ResolvedStateActivity,
    StateActivity,
    detect_zero_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.specs import DEFAULT_SIGMA_PRIOR
from openghg_inversions.sigma import SigmaAlignment


def _resolve_co2_state_activity(
    flux_sensitivity: xr.DataArray,
    state_activity: StateActivity | None,
) -> ResolvedStateActivity:
    """Resolve activity once in the retained-state coordinate order."""
    return resolve_state_activity(
        detect_zero_sensitivity(flux_sensitivity),
        state_activity,
    )


def co2_prior_forward_mean(
    flux_sensitivity: xr.DataArray,
    *,
    prior_mean: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    state_activity: StateActivity | None = None,
) -> xr.DataArray:
    """Return the deterministic prior-mean concentration for closure checks.

    The affine closure equation is ``fixed_prior_contribution + H @ x_prior``.
    Active states use ``prior_mean`` and inactive states use their exact fixed
    values.
    """
    sensitivity = flux_sensitivity.transpose("nmeasure", ...)
    state_dim = next(str(dim) for dim in sensitivity.dims if dim != "nmeasure")
    activity = _resolve_co2_state_activity(sensitivity, state_activity)
    mean = prior_mean.transpose(state_dim)
    sensitivity, mean = xr.align(sensitivity, mean, join="exact", copy=False)
    mean = xr.where(activity.active, mean, activity.fixed_value)
    pollution_mean = xr.dot(sensitivity, mean, dim=state_dim)
    fixed_prior, pollution_mean = xr.align(
        fixed_prior_contribution.transpose("nmeasure"),
        pollution_mean,
        join="exact",
        copy=False,
    )
    return (fixed_prior + pollution_mean).rename("prior_forward_mean")


def _add_co2_retained_state(
    flux_sensitivity: xr.DataArray,
    *,
    prior_mean: xr.DataArray,
    prior_covariance: xr.DataArray,
    state_activity: StateActivity | None,
) -> Any:
    """Add the correlated active state and restore fixed states in full order."""
    design = flux_sensitivity.transpose("nmeasure", ...)
    state_dim = next(str(dim) for dim in design.dims if dim != "nmeasure")
    activity = _resolve_co2_state_activity(design, state_activity)
    covariance_dims = [str(dim) for dim in prior_covariance.dims if dim != state_dim]
    if len(covariance_dims) != 1:
        raise ValueError(
            "CO2 prior covariance must have the retained state dimension and "
            f"one covariance-column dimension; got {prior_covariance.dims!r}."
        )
    prior = CorrelatedLognormalPrior(
        prior_mean,
        prior_covariance,
        covariance_dim=covariance_dims[0],
    )
    return add_correlated_lognormal_state_with_activity(
        activity,
        prior,
        var_name="x",
    ).state


def _fixed_mismatch_array(
    observations: xr.DataArray,
    fixed_model_mismatch: float | xr.DataArray | None,
) -> xr.DataArray | None:
    """Return a labelled fixed mismatch in the observations' concentration units."""
    if fixed_model_mismatch is None:
        return None
    if isinstance(fixed_model_mismatch, xr.DataArray):
        if fixed_model_mismatch.dims != ("nmeasure",):
            raise ValueError(
                "`fixed_model_mismatch` must be a scalar or a DataArray with "
                f"dims ('nmeasure',); got {fixed_model_mismatch.dims!r}."
            )
        mismatch, _ = xr.align(fixed_model_mismatch, observations, join="exact")
        return mismatch.transpose("nmeasure").rename("fixed_model_mismatch")
    return xr.full_like(
        observations,
        fixed_model_mismatch,
        dtype=np.float64,
    ).rename("fixed_model_mismatch")


def build_co2_rhime_model(
    flux_sensitivity: xr.DataArray,
    *,
    prior_mean: xr.DataArray,
    prior_covariance: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: Mapping[str, Any] | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    no_model_error: bool = False,
) -> pm.Model:
    """Build the CO2 coherent-reduction model from explicit scientific arrays.

    ``flux_sensitivity`` is the reduced operator ``H_alpha`` and
    ``fixed_prior_contribution`` is the affine term
    ``H m - H_alpha (Pi m)``. The latter is a fixed prior contribution, not an
    atmospheric boundary condition. ``prior_covariance`` is the labelled
    arithmetic covariance of the retained positive state.

    Known fixed states remain in ``x`` and in the forward calculation, but are
    omitted from the sampled correlated state. ``fixed_model_mismatch`` is an
    optional known concentration standard deviation. OGI leaves this policy
    unset by default; the Verification Games fixed likelihood passes 1 ppm
    explicitly.
    """
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    fixed_mismatch = _fixed_mismatch_array(observations, fixed_model_mismatch)

    with registered_model() as model:
        h = add_model_data(flux_sensitivity.transpose("nmeasure", ...), "hx")
        x = _add_co2_retained_state(
            flux_sensitivity,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            state_activity=state_activity,
        )
        pollution_mean = pm.Deterministic("mu_pollution", pt.dot(h, x), dims="nmeasure")
        fixed_prior = add_model_data(
            fixed_prior_contribution.transpose("nmeasure"),
            "fixed_prior_contribution",
        )
        modelled_mean = pm.Deterministic(
            "mu",
            fixed_prior + pollution_mean,
            dims="nmeasure",
        )
        add_additive_sigma_gaussian_likelihood(
            observations=observations,
            observation_error=observation_error,
            minimum_error=minimum_error,
            aggregation_error=aggregation_error,
            fixed_model_mismatch=fixed_mismatch,
            mean=modelled_mean,
            sigma_alignment=sigma_alignment,
            sigma_prior=sigma_prior,
            no_model_error=no_model_error,
            output_dim="nmeasure",
        )
    return model


__all__ = ["build_co2_rhime_model", "co2_prior_forward_mean"]
