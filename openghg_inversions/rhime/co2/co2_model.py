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
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import add_additive_sigma_gaussian_likelihood
from openghg_inversions.models.components import (
    add_coherent_affine_component,
    add_correlated_lognormal_state_with_activity,
    add_linear_component,
    add_model_data,
    apply_linear_sensitivity,
)
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR, DEFAULT_SIGMA_PRIOR
from openghg_inversions.sigma import SigmaAlignment

from .outer_regions import (
    OuterRegionTreatment,
    add_outer_state_component,
    add_outer_observation_covariance,
)


def _resolve_co2_state_activity(
    prepared_sensitivity: PreparedLinearSensitivity,
    state_activity: StateActivity | None,
) -> ResolvedStateActivity:
    """Resolve activity once from the prepared full-state mapping."""
    return resolve_state_activity(prepared_sensitivity.removed, state_activity)


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
    prepared = prepare_linear_sensitivity(sensitivity)
    state_dim = prepared.state_dim
    activity = _resolve_co2_state_activity(prepared, state_activity)
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
    prepared_sensitivity: PreparedLinearSensitivity,
    *,
    prior_mean: xr.DataArray,
    prior_covariance: xr.DataArray,
    state_activity: StateActivity | None,
) -> Any:
    """Add the correlated active state and restore fixed states in full order."""
    state_dim = prepared_sensitivity.state_dim
    activity = _resolve_co2_state_activity(prepared_sensitivity, state_activity)
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
        var_name="flux_scaling",
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


def _reject_outer_state_double_counting(
    prepared_sensitivity: PreparedLinearSensitivity,
    outer_treatment: OuterRegionTreatment | None,
) -> None:
    """Require explicit disjoint inner/outer basis-group partitions."""
    if outer_treatment is None:
        return
    state_dim = prepared_sensitivity.state_dim
    full_state = prepared_sensitivity.removed
    if "basis_group" not in full_state.coords:
        raise ValueError(
            "CO2 flux_sensitivity requires state-aligned basis_group metadata when an "
            "outer_treatment is supplied, to prove the partitions are disjoint."
        )
    inner_groups = full_state["basis_group"]
    if inner_groups.dims != (state_dim,) or bool(inner_groups.isnull().any().compute().item()):
        raise ValueError("CO2 flux_sensitivity basis_group metadata must be complete and state-aligned.")
    if bool((inner_groups == "outer").any().compute().item()):
        raise ValueError(
            "CO2 flux_sensitivity contains basis_group='outer' states while an outer_treatment "
            "is supplied; remove them to avoid double counting."
        )

    outer_metadata = outer_treatment.state_metadata
    if "basis_group" not in outer_metadata:
        raise ValueError("outer_treatment requires basis_group metadata proving every state is outer.")
    outer_groups = outer_metadata["basis_group"]
    if outer_groups.ndim != 1 or bool(outer_groups.isnull().any().compute().item()):
        raise ValueError("outer_treatment basis_group metadata must be complete and one-dimensional.")
    if not bool((outer_groups == "outer").all().compute().item()):
        raise ValueError("outer_treatment basis_group metadata must label every state 'outer'.")


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
    outer_treatment: OuterRegionTreatment | None = None,
    boundary_sensitivity: xr.DataArray | None = None,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    no_model_error: bool = False,
) -> pm.Model:
    """Build the CO2 coherent-reduction model from explicit scientific arrays.

    ``flux_sensitivity`` is prepared once as the reduced operator ``H_alpha``:
    exact-zero columns are omitted from the backend ``co2_sensitivity`` while
    ``flux_scaling`` retains the complete labelled scientific state.
    ``fixed_prior_contribution`` is then added with the shared coherent-affine
    component to produce ``modelled_concentration``.

    ``fixed_prior_contribution`` is the affine term
    ``H m - H_alpha (Pi m)``. The latter is a fixed prior contribution, not an
    atmospheric boundary condition. ``prior_covariance`` is the labelled
    arithmetic covariance of the retained positive state.

    Known fixed states remain in the public state and forward calculation but
    are omitted from the sampled correlated state. ``fixed_model_mismatch`` is
    an optional known concentration standard deviation. OGI leaves this policy
    unset by default; the Verification Games fixed likelihood passes 1 ppm
    explicitly. ``outer_treatment`` and optional sampled
    ``boundary_sensitivity @ bc`` remain separate linear components named
    ``outer_flux_contribution`` and ``mu_bc``. Reporting code may group them
    when presenting a baseline.
    """
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    fixed_mismatch = _fixed_mismatch_array(observations, fixed_model_mismatch)
    prepared_flux = prepare_linear_sensitivity(flux_sensitivity, output_dim="nmeasure")
    _reject_outer_state_double_counting(prepared_flux, outer_treatment)
    aggregation_error = (
        aggregation_error
        if outer_treatment is None
        else add_outer_observation_covariance(aggregation_error, outer_treatment)
    )

    with registered_model() as model:
        flux_scaling = _add_co2_retained_state(
            prepared_flux,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            state_activity=state_activity,
        )
        linear_signal = apply_linear_sensitivity(
            prepared_flux,
            flux_scaling,
            data_name="co2_sensitivity",
            output_name="co2_flux_contribution",
        )
        if boundary_sensitivity is not None:
            boundary_mean = add_linear_component(
                prepare_linear_sensitivity(boundary_sensitivity),
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
            ).output
            linear_signal = linear_signal + boundary_mean

        if outer_treatment is not None:
            if outer_treatment.mode == "marginalized":
                assert outer_treatment.mean_contribution is not None
                outer_mean_data = add_model_data(
                    outer_treatment.mean_contribution.transpose("nmeasure"),
                    "outer_mean_contribution",
                )
                outer_mean = pm.Deterministic(
                    "outer_flux_contribution",
                    outer_mean_data,
                    dims="nmeasure",
                )
            else:
                outer_mean = add_outer_state_component(outer_treatment)
            linear_signal = linear_signal + outer_mean
        modelled_mean = add_coherent_affine_component(
            fixed_prior_contribution.transpose("nmeasure").rename("fixed_prior_contribution"),
            linear_signal,
            output_name="modelled_concentration",
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
