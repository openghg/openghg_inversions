"""Concrete PyMC graph for the CO2 coherent-reduction recipe.

The recipe is deliberately procedural. It samples one labelled retained
state, applies the reduced observation operator, adds the fixed affine prior
contribution, and finally constructs the observation likelihood.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_gaussian_likelihood,
)
from openghg_inversions.models.components import (
    add_coherent_affine_component,
    add_correlated_lognormal_state_with_activity,
    add_linear_component,
    add_offset_component,
    apply_linear_sensitivity,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from openghg_inversions.sigma import SigmaAlignment


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


def _retained_group_indices(
    prepared_flux: PreparedLinearSensitivity,
    *,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate one group's retained sensitivity columns and full-state entries.

    Sensitivity preparation removes exact-zero columns while preserving the
    complete scientific state in ``prepared_flux.removed``. A reporting
    projection therefore needs both the compact column positions used by the
    registered sensitivity and the corresponding positions in the complete
    flux state.

    Args:
        prepared_flux: Prepared sensitivity with the retained-to-full-state
            mapping and state-aligned ``basis_group`` metadata.
        group: ``basis_group`` label to select.

    Returns:
        A pair containing the selected column positions in the retained
        sensitivity and their corresponding positions in the complete state.
        Both arrays are empty when ``basis_group`` is absent or the requested
        group has no retained sensitivity columns.

    Raises:
        ValueError: If ``basis_group`` is not indexed solely by the complete
            flux-state dimension.

    Notes:
        This function only resolves labelled positions. It neither constructs
        PyMC variables nor creates a separate state for the selected group.
    """
    state_dim = prepared_flux.state_dim
    if "basis_group" not in prepared_flux.removed.coords:
        empty = np.empty(0, dtype=int)
        return empty, empty
    basis_group = prepared_flux.removed["basis_group"]
    if basis_group.dims != (state_dim,):
        raise ValueError("`basis_group` must be indexed only by the flux state dimension.")

    group_values = np.asarray(basis_group.compute().values).astype(str)
    retained_indices = prepared_flux.retained_indices
    group_columns = np.flatnonzero(group_values[retained_indices] == group)
    return group_columns, retained_indices[group_columns]


def build_co2_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: xr.DataArray | None = None,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    site_indicator: xr.DataArray | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: dict | None = None,
) -> pm.Model:
    """Build the CO2 coherent-reduction model from explicit scientific arrays.

    ``flux_sensitivity`` is prepared once as the reduced operator ``H_alpha``:
    exact-zero columns are omitted from the backend ``co2_sensitivity`` while
    ``flux_scaling`` retains the complete labelled scientific state.
    ``fixed_prior_contribution`` is then added with the shared coherent-affine
    component to produce ``modelled_concentration``.

    ``fixed_prior_contribution`` is the affine term
    ``H m - H_alpha (Pi m)``. The latter is a fixed prior contribution, not an
    atmospheric boundary condition. ``retained_prior`` contains the complete
    labelled arithmetic moments for the positive flux state.

    Known fixed states remain in the public state and forward calculation but
    are omitted from the sampled correlated state. ``fixed_model_mismatch`` is
    an optional known concentration standard deviation. ``openghg_inversions``
    leaves this policy unset by default; the Verification Games fixed likelihood
    passes 1 ppm explicitly. Inner and outer same-grid states remain in this one
    flux state and are distinguished by ``basis_group`` metadata. When that
    metadata includes ``"outer"``, ``outer_flux_contribution`` is exposed as a
    reporting projection of ``co2_sensitivity @ flux_scaling``; it is never
    added to the mean a second time. Optional boundary and offset terms remain
    scientifically distinct components named ``mu_bc`` and ``offset``.

    Direct custom callers are responsible for supplying scientifically
    coherent arrays from one preparation and for their positional semantics
    when labels are absent.

    Args:
        flux_sensitivity: Reduced CO2 sensitivity with observation dimension
            ``nmeasure`` and one labelled retained-state dimension.
        retained_prior: Complete labelled arithmetic-moment prior for the
            retained positive state.
        fixed_prior_contribution: Fixed coherent-reduction affine intercept
            named ``fixed_prior_contribution`` on ``nmeasure``.
        observations: Observed CO2 concentrations on ``nmeasure``.
        observation_error: Reported observation standard deviation.
        minimum_error: Minimum independent model-data mismatch standard
            deviation.
        aggregation_error: Prepared fixed aggregation-error representation.
        sigma_alignment: Optional grouping policy for inferred additive model
            error.
        sigma_prior: Optional prior arguments for inferred additive model
            error.
        fixed_model_mismatch: Optional known scalar or labelled concentration
            standard deviation.
        state_activity: Optional labelled activity policy for retained flux
            states.
        boundary_sensitivity: Optional atmospheric boundary-condition
            sensitivity.
        bc_prior: Optional prior arguments for boundary-condition scaling.
        bc_state_activity: Optional labelled activity policy for boundary
            states.
        site_indicator: Optional observation-to-site index used by an offset.
        offset_prior: Optional prior for an offset component. When omitted, no
            offset is added.
        offset_args: Extra keyword arguments for the offset component.

    Returns:
        A registered PyMC model containing the complete affine concentration
        and Gaussian likelihood.

    Raises:
        ValueError: If shared preparation, prior construction, or registered
            coordinate alignment fails.
    """
    if sigma_alignment is not None:
        sigma_prior = dict(DEFAULT_ADDITIVE_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    else:
        sigma_prior = None
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    if offset_prior is not None:
        offset_prior = dict(offset_prior)
    fixed_mismatch = _fixed_mismatch_array(observations, fixed_model_mismatch)
    prepared_flux = prepare_linear_sensitivity(flux_sensitivity, output_dim="nmeasure")
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    outer_columns, outer_states = _retained_group_indices(prepared_flux, group="outer")

    with registered_model() as model:
        flux_scaling = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state
        co2_flux_contribution = apply_linear_sensitivity(
            prepared_flux,
            flux_scaling,
            data_name="co2_sensitivity",
            output_name="co2_flux_contribution",
        )
        if outer_columns.size:
            outer_flux_contribution = pt.dot(
                model["co2_sensitivity"][:, outer_columns],
                flux_scaling[outer_states],
            )
            pm.Deterministic(
                "outer_flux_contribution",
                outer_flux_contribution,
                dims=prepared_flux.output_dim,
            )
        modelled_linear_signal = co2_flux_contribution
        if boundary_sensitivity is not None:
            boundary_contribution = add_linear_component(
                prepare_linear_sensitivity(boundary_sensitivity),
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
            ).output
            modelled_linear_signal = modelled_linear_signal + boundary_contribution

        if offset_prior is not None:
            if site_indicator is None:
                raise ValueError("CO2 offset component requires `site_indicator`.")
            offset = add_offset_component(
                site_indicator,
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )
            modelled_linear_signal = modelled_linear_signal + offset
        modelled_mean = add_coherent_affine_component(
            fixed_prior_contribution,
            modelled_linear_signal,
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
            output_dim="nmeasure",
        )
    return model


__all__ = ["build_co2_model"]
