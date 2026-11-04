"""Concrete PyMC graph for the CO2 coherent-reduction recipe.

The recipe is deliberately procedural. It samples one labelled retained
state, constructs the complete coherent flux contribution, adds optional
boundary and offset terms, and finally constructs the observation likelihood.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr
from dask import compute as dask_compute

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.models.biosphere import add_biosphere_state
from openghg_inversions.array_ops import to_dense_tree
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_likelihood,
)
from openghg_inversions.models.boundary import Boundary
from openghg_inversions.models.components import (
    ConcentrationTerm,
    add_coherent_affine_component,
    add_correlated_lognormal_state_with_activity,
    add_offset_component,
    apply_linear_sensitivity,
    likelihood_representation,
    warn_constant_offset_confounding,
)
from openghg_inversions.models.coords import add_model_data, registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime._model_building import _call_custom_likelihood
from openghg_inversions.rhime.builders import RhimeLikelihoodBuilder
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


def _normalise_offset_args(
    offset_args: Mapping[str, Any] | None,
) -> tuple[str | None, bool, bool]:
    """Validate the small offset option set used by the CO2 runners."""
    options = dict(offset_args or {})
    supported = {"offset_freq", "drop_first", "per_site"}
    unknown = sorted(options.keys() - supported)
    if unknown:
        raise ValueError(f"Unsupported offset_args option(s): {unknown!r}.")

    frequency = options.get("offset_freq")
    if frequency is not None and not isinstance(frequency, str):
        raise TypeError("offset_freq must be a string or None.")

    drop_first = options.get("drop_first", False)
    per_site = options.get("per_site", True)
    if not isinstance(drop_first, bool) or not isinstance(per_site, bool):
        raise TypeError("drop_first and per_site must be booleans.")
    return frequency, drop_first, per_site


def build_co2_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    biosphere_prior: BiospherePrior | None = None,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    boundary: Boundary | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> pm.Model:
    """Build the CO2 coherent-reduction model from explicit scientific arrays.

    ``flux_sensitivity`` is prepared once as the reduced operator ``H_alpha``:
    exact-zero columns are omitted from the backend ``co2_sensitivity`` while
    ``flux_scaling`` retains the complete labelled scientific state.
    ``fixed_prior_contribution`` is then added with the shared coherent-affine
    component to produce the complete ``co2_flux_contribution``. Optional
    boundary and offset terms are added afterwards to produce
    ``modelled_concentration``.

    ``fixed_prior_contribution`` is the affine term
    ``H m - H_alpha (Pi m)``. The latter is a fixed prior contribution, not an
    atmospheric boundary condition. ``retained_prior`` contains the complete
    labelled arithmetic moments for the positive flux state.

    Known fixed states remain in the public state and forward calculation but
    are omitted from the sampled correlated state. ``fixed_model_mismatch`` is
    an optional known concentration standard deviation. ``openghg_inversions``
    leaves this policy unset by default; the Verification Games fixed likelihood
    passes 1 ppm explicitly. Inner and outer same-grid states remain in this one
    flux state and are distinguished by ``basis_group`` metadata retained for
    output-side selection. The model builds only their complete shared flux
    contribution. Optional boundary and offset terms remain scientifically
    distinct components named ``mu_bc`` and ``offset``.

    Direct custom callers are responsible for supplying scientifically
    coherent arrays from one preparation and for their positional semantics
    when labels are absent. Centered observations and model means are forwarded
    explicitly to the selected likelihood, including custom callables.

    With a centered ``boundary``, raw PyMC observed and predictive
    ``y`` values are centered; BC and total concentration deterministics remain
    absolute. Custom likelihood components receive the centered observations
    and mean with unchanged observation-error and aggregation-error operands.
    A custom density depending on absolute concentration must handle the known
    fixed translation and any captured physical reference context explicitly.
    After restoring sample coordinates, direct callers must restore absolute
    ``y`` with
    :func:`~openghg_inversions.rhime.co2.boundary_output.restore_absolute_predictives`
    before serialization or postprocessing. Ordinary recipe runners do this
    through
    :func:`~openghg_inversions.rhime.co2.co2_runner.annotate_co2_trace`.
    See :ref:`surface-centered-boundary-contract`.

    Args:
        flux_sensitivity: Reduced CO2 sensitivity with observation dimension
            ``nmeasure`` and one labelled retained-state dimension.
        biosphere_prior: Optional explicit retained-state calibration; omission preserves the existing prior.
        retained_prior: Complete labelled arithmetic-moment prior for the
            retained positive state.
        fixed_prior_contribution: Fixed coherent-reduction affine intercept
            named ``fixed_prior_contribution`` on ``nmeasure``.
        observations: Observed CO2 concentrations on ``nmeasure``.
        observation_error: Reported observation standard deviation.
        aggregation_error: Prepared fixed aggregation-error representation.
        likelihood_builder: Optional ordinary likelihood component. When
            supplied, it replaces the default additive-sigma likelihood.
        likelihood_kwargs: Options passed only to the selected likelihood.
        sigma_alignment: Optional grouping policy for inferred additive model
            error.
        sigma_prior: Optional prior arguments for inferred additive model
            error.
        fixed_model_mismatch: Optional known scalar or labelled concentration
            standard deviation.
        state_activity: Optional labelled activity policy for retained flux
            states.
        boundary: Resolved boundary component bound to this fit.
        offset_prior: Optional prior for an offset component. When omitted, no
            offset is added. Site codes are derived from the ``site`` coordinate
            on ``observations``.
        offset_args: Optional offset settings: ``offset_freq``, ``drop_first``,
            and ``per_site``.

    Returns:
        A registered PyMC model containing the complete affine concentration
        and Gaussian likelihood.

    Raises:
        ValueError: If shared preparation, prior construction, or registered
            coordinate alignment fails, or if ``sigma_prior`` is supplied
            without ``sigma_alignment``.
    """
    if likelihood_builder is None:
        if likelihood_kwargs:
            raise ValueError("likelihood_kwargs require likelihood_builder.")
        if sigma_alignment is None and sigma_prior is not None:
            raise ValueError("`sigma_prior` requires `sigma_alignment`.")
        if sigma_alignment is not None:
            sigma_prior = dict(DEFAULT_ADDITIVE_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    elif any(value is not None for value in (sigma_alignment, sigma_prior, fixed_model_mismatch)):
        raise ValueError(
            "A selected likelihood_builder cannot be combined with default "
            "additive-sigma or fixed-mismatch options; pass its options in "
            "likelihood_kwargs."
        )
    (flux_sensitivity, fixed_prior_contribution, observations, observation_error), boundary, priors = dask_compute(
        *to_dense_tree((
            (flux_sensitivity, fixed_prior_contribution, observations, observation_error), boundary,
            {"sigma": sigma_prior, "offset": offset_prior, "fixed_mismatch": fixed_model_mismatch},
        ))
    )
    sigma_prior, offset_prior = priors["sigma"], priors["offset"]
    fixed_model_mismatch = priors["fixed_mismatch"]
    if offset_prior is not None:
        offset_prior = dict(offset_prior)
    offset_freq, offset_drop_first, offset_per_site = _normalise_offset_args(offset_args)
    fixed_mismatch = _fixed_mismatch_array(observations, fixed_model_mismatch)
    prepared_flux = prepare_linear_sensitivity(flux_sensitivity, output_dim="nmeasure")
    activity = resolve_state_activity(prepared_flux.removed, state_activity)

    with registered_model() as model:
        flux_scaling = (
            add_correlated_lognormal_state_with_activity(
                activity, retained_prior, var_name="flux_scaling",
            )
            if biosphere_prior is None
            else add_biosphere_state(activity, retained_prior, biosphere_prior)
        ).state
        scaled_flux_contribution = apply_linear_sensitivity(
            prepared_flux,
            flux_scaling,
            data_name="co2_sensitivity",
            output_name="co2_flux_contribution",
            compute_deterministic=False,
        )
        co2_flux_contribution = add_coherent_affine_component(
            fixed_prior_contribution,
            scaled_flux_contribution,
            output_name="co2_flux_contribution",
        )
        terms: list[ConcentrationTerm] = [ConcentrationTerm.absolute(co2_flux_contribution)]
        if boundary is not None:
            terms.append(boundary.add())
        if offset_prior is not None:
            offset = add_offset_component(
                observations,
                prior_args=offset_prior,
                offset_freq=offset_freq,
                output_name="offset",
                output_dim="nmeasure",
                drop_first=offset_drop_first,
                per_site=offset_per_site,
            )
            warn_constant_offset_confounding(terms, drop_first=offset_drop_first)
            terms.append(ConcentrationTerm.absolute(offset))

        modelled_mean = pm.Deterministic(
            "modelled_concentration",
            sum((term.output for term in terms[1:]), start=terms[0].output),
            dims="nmeasure",
        )
        likelihood_mean, likelihood_observations, shift = likelihood_representation(
            terms,
            observations,
            modelled_mean,
        )
        if shift is not None:
            add_model_data(shift.rename("boundary_reference_shift"), "boundary_reference_shift")
        if likelihood_builder is None:
            add_additive_sigma_likelihood(
                observations=likelihood_observations,
                observation_error=observation_error,
                aggregation_error=aggregation_error,
                fixed_model_mismatch=fixed_mismatch,
                mean=likelihood_mean,
                additive_sigma_alignment=sigma_alignment,
                additive_sigma_prior=sigma_prior,
                output_dim="nmeasure",
            )
        else:
            _call_custom_likelihood(
                likelihood_builder,
                observations=likelihood_observations,
                observation_error=observation_error,
                aggregation_error=aggregation_error,
                mean=likelihood_mean,
                output_dim="nmeasure",
                likelihood_kwargs=likelihood_kwargs,
            )
    return model


__all__ = ["build_co2_model"]
