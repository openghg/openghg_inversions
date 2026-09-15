"""CO2 graph for cached fixed-OU site-amplitude sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import expand_mapping
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.cached_sigma import (
    FixedOuCachedSigmaTarget,
    MarginalQuadraticCache,
)
from openghg_inversions.models.components import (
    _add_offset_component_result,
    _add_prepared_correlated_lognormal_state_with_activity,
    add_model_data,
    add_state_vector,
    add_coherent_affine_component,
    apply_linear_sensitivity,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import add_coords, registered_model
from openghg_inversions.models.fixed_ou import FixedOuLowRank, prepare_fixed_ou_low_rank
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_as_low_rank,
    validate_observation_error_arrays,
)
from openghg_inversions.rhime.cached_sigma import PytensorMarginalQuadraticCache
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from openghg_inversions.sigma import SigmaAlignment


OU_SITE_DIM = "ou_site"
OU_SITE_INDEX = "ou_site_index"
OU_SITE_AMPLITUDE = "ou_site_amplitude"


@dataclass(frozen=True)
class Co2CachedSigmaModel:
    """Concrete graph plus the numerical objects required by its sampler."""

    model: pm.Model
    target: FixedOuCachedSigmaTarget
    shared_cache: PytensorMarginalQuadraticCache
    initial_cache: MarginalQuadraticCache
    amplitude: TensorVariable
    states: tuple[TensorVariable, ...]
    modelled_mean: TensorVariable
    site_amplitude_prior_scale: float


@dataclass(frozen=True)
class _CachedAffineTerm:
    """One scientific affine term and its private cached representation."""

    fixed_contribution: NDArray[np.float64]
    active_design: NDArray[np.float64]
    coefficients: TensorVariable | None
    sampled_rvs: tuple[TensorVariable, ...]
    output: TensorVariable


@dataclass(frozen=True)
class _CachedLikelihood:
    """Private likelihood products needed by the matched sampler."""

    target: FixedOuCachedSigmaTarget
    shared_cache: PytensorMarginalQuadraticCache
    initial_cache: MarginalQuadraticCache
    amplitude: TensorVariable
    states: tuple[TensorVariable, ...]


def _materialize_cached_linear_projection(
    prepared: PreparedLinearSensitivity,
    activity: ResolvedStateActivity,
    *,
    output_dim: str,
) -> tuple[
    PreparedLinearSensitivity,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Materialize one cached linear design and split its affine projection."""
    retained_indices = prepared.retained_indices
    sensitivity = prepared.sensitivity.compute()
    retained_active = np.asarray(activity.active.values, dtype=bool)[retained_indices]
    retained_dim = next(
        str(dim) for dim in prepared.sensitivity.dims if dim != output_dim
    )
    design = np.asarray(
        sensitivity.transpose(output_dim, retained_dim).values,
        dtype=np.float64,
    )
    fixed_values = np.asarray(
        activity.fixed_value.values,
        dtype=np.float64,
    )[retained_indices]
    return (
        replace(prepared, sensitivity=sensitivity),
        design[:, retained_active],
        design[:, ~retained_active] @ fixed_values[~retained_active],
    )


def _fixed_ou_site_alignment(
    observations: xr.DataArray,
    *,
    output_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    time = observations.coords.get("time")
    if time is None or time.dims != (output_dim,):
        raise ValueError(
            "The cached fixed-OU CO2 recipe requires observation-aligned "
            "'time' coordinates."
        )
    alignment = SigmaAlignment.from_observations(observations)
    site_index = alignment.site_index.rename(OU_SITE_INDEX)
    site_coord = alignment.site_labels.rename({"nsigma_site": OU_SITE_DIM}).rename(
        OU_SITE_DIM
    )
    return site_coord, site_index


def _site_values(
    value: float | Mapping[str, float] | None,
    *,
    site_coord: xr.DataArray,
    default: float,
) -> np.ndarray:
    if value is None:
        values = np.full(site_coord.size, default, dtype=np.float64)
    elif isinstance(value, Mapping):
        values = np.asarray(
            expand_mapping(value, site_coord, name=OU_SITE_AMPLITUDE).values,
            dtype=np.float64,
        )
    else:
        values = np.full(site_coord.size, value, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Initial site amplitudes must be finite and strictly positive.")
    return values


def _add_cached_likelihood(
    terms: tuple[_CachedAffineTerm, ...],
    *,
    covariance: FixedOuLowRank,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    site_coord: xr.DataArray,
    site_index: xr.DataArray,
    initial_amplitudes: np.ndarray,
    site_amplitude_prior_scale: float,
    output_dim: str,
) -> _CachedLikelihood:
    """Lower ordered scientific terms once at the cached-likelihood boundary."""
    active_terms = tuple(term for term in terms if term.coefficients is not None)
    if not active_terms:
        raise ValueError(
            "The cached-sigma recipe requires at least one active flux, "
            "boundary, or offset coefficient."
        )
    target = FixedOuCachedSigmaTarget(
        prepared=covariance,
        observations=np.asarray(
            observations.transpose(output_dim).compute().values,
            dtype=np.float64,
        ),
        fixed_contribution=np.sum(
            [term.fixed_contribution for term in terms],
            axis=0,
        ),
        design=np.column_stack([term.active_design for term in active_terms]),
    )
    initial_cache = target.refresh(initial_amplitudes)
    shared_cache = PytensorMarginalQuadraticCache(initial_cache)
    add_coords({OU_SITE_DIM: site_coord})
    add_model_data(observations.transpose(output_dim), "Y")
    add_model_data(observation_error.transpose(output_dim), "error")
    add_model_data(site_index)
    add_model_data(
        xr.DataArray(
            covariance.tau_hours_by_site,
            dims=(OU_SITE_DIM,),
            coords={OU_SITE_DIM: site_coord},
            name="ou_tau_hours",
        )
    )
    amplitude = pm.HalfNormal(
        OU_SITE_AMPLITUDE,
        sigma=float(site_amplitude_prior_scale),
        initval=pm.floatX(initial_amplitudes),
        dims=OU_SITE_DIM,
    )
    pm.Deterministic(
        "epsilon",
        pt.sqrt(covariance.marginal_variance(amplitude)),
        dims=output_dim,
    )
    coefficients = pt.concatenate(
        [cast(TensorVariable, term.coefficients) for term in active_terms]
    )
    pm.Potential(
        "cached_fixed_ou_likelihood",
        shared_cache.log_likelihood(coefficients),
    )
    return _CachedLikelihood(
        target=target,
        shared_cache=shared_cache,
        initial_cache=initial_cache,
        amplitude=amplitude,
        states=tuple(rv for term in terms for rv in term.sampled_rvs),
    )


def build_co2_cached_sigma_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior_scale: float,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: xr.DataArray | None = None,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    offset_prior: PriorArgs | None = None,
    offset_freq: str | None = None,
    offset_drop_first: bool = False,
    offset_per_site: bool = True,
    output_dim: str = "nmeasure",
) -> Co2CachedSigmaModel:
    """Build the named CO2 cached-sigma graph in scientific order.

    The complete ``co2_flux_contribution`` is the coherent affine sum of the
    fixed prior contribution and flux sensitivity product. Boundary and offset
    terms are then added to form ``modelled_concentration``. The observation
    likelihood is a normalized cached ``Potential`` over a private
    concatenation of the active flux, boundary, and offset coefficients. These
    states remain separate scientific variables and sampler inputs. The
    returned target owns exact joint likelihood and predictive evaluation
    after sampling; no independent pointwise likelihood is fabricated in the
    graph.
    """
    if not np.isfinite(site_amplitude_prior_scale) or site_amplitude_prior_scale <= 0.0:
        raise ValueError("`site_amplitude_prior_scale` must be finite and strictly positive.")
    validate_observation_error_arrays(
        observations,
        observation_error,
        None,
        owner="Cached fixed-OU CO2 likelihood",
        output_dim=output_dim,
    )
    site_coord, site_index = _fixed_ou_site_alignment(
        observations,
        output_dim=output_dim,
    )
    site_labels = tuple(str(label) for label in site_coord.values)
    initial_amplitudes = _site_values(
        initial_site_amplitudes,
        site_coord=site_coord,
        default=float(site_amplitude_prior_scale),
    )
    prepared_flux = prepare_linear_sensitivity(
        flux_sensitivity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(prepared_flux.removed, state_activity)
    (
        prepared_flux,
        flux_active_design,
        flux_fixed_contribution,
    ) = _materialize_cached_linear_projection(
        prepared_flux,
        activity,
        output_dim=output_dim,
    )
    active_state_prior = prepare_active_correlated_lognormal_prior(
        activity,
        retained_prior,
        var_name="flux_scaling",
    )

    fixed_prior_data = fixed_prior_contribution.transpose(output_dim).compute()
    fixed_prior = np.asarray(fixed_prior_data.values, dtype=np.float64)
    prepared_boundary = None
    boundary_activity = None
    boundary_active_design = None
    boundary_fixed_contribution = None
    if boundary_sensitivity is not None:
        prepared_boundary = prepare_linear_sensitivity(
            boundary_sensitivity,
            output_dim=output_dim,
        )
        boundary_activity = resolve_state_activity(
            prepared_boundary.removed,
            bc_state_activity,
        )
        (
            prepared_boundary,
            boundary_active_design,
            boundary_fixed_contribution,
        ) = _materialize_cached_linear_projection(
            prepared_boundary,
            boundary_activity,
            output_dim=output_dim,
        )
    factor, aggregation_diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor,
        np.square(
            np.asarray(
                observation_error.transpose(output_dim).compute().values,
                dtype=np.float64,
            )
        )
        + aggregation_diagonal,
        np.asarray(observations.coords["time"].compute().values),
        np.asarray(site_index.values, dtype=np.int64),
        tau_hours,
        site_labels=site_labels,
    )
    with registered_model() as model:
        state_result = _add_prepared_correlated_lognormal_state_with_activity(
            activity,
            active_state_prior,
            var_name="flux_scaling",
        )
        contribution = apply_linear_sensitivity(
            prepared_flux,
            state_result.state,
            data_name="co2_sensitivity",
            output_name="co2_flux_contribution",
            compute_deterministic=False,
        )
        co2_flux_mean = add_coherent_affine_component(
            fixed_prior_data,
            contribution,
            output_name="co2_flux_contribution",
        )
        terms = [
            _CachedAffineTerm(
                fixed_contribution=fixed_prior + flux_fixed_contribution,
                active_design=flux_active_design,
                coefficients=(
                    state_result.state[activity.active_indices]
                    if activity.n_active
                    else None
                ),
                sampled_rvs=(
                    (state_result.latent,)
                    if state_result.latent is not None
                    else ()
                ),
                output=co2_flux_mean,
            )
        ]
        if prepared_boundary is not None:
            assert boundary_activity is not None
            assert boundary_active_design is not None
            assert boundary_fixed_contribution is not None
            boundary_result = add_state_vector(
                boundary_activity,
                prior_args=dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior),
                var_name="bc",
            )
            boundary_output = apply_linear_sensitivity(
                prepared_boundary,
                boundary_result.state,
                data_name="hbc",
                output_name="mu_bc",
                compute_deterministic=True,
            )
            boundary_active = boundary_result.activity.active_indices
            terms.append(
                _CachedAffineTerm(
                    fixed_contribution=boundary_fixed_contribution,
                    active_design=boundary_active_design,
                    coefficients=(
                        boundary_result.state[boundary_active]
                        if boundary_active.size
                        else None
                    ),
                    sampled_rvs=(
                        (boundary_result.latent,)
                        if boundary_result.latent is not None
                        else ()
                    ),
                    output=boundary_output,
                )
            )
        if offset_prior is not None:
            offset_result = _add_offset_component_result(
                observations,
                prior_args=dict(offset_prior),
                offset_freq=offset_freq,
                output_name="offset",
                output_dim=output_dim,
                drop_first=offset_drop_first,
                per_site=offset_per_site,
            )
            terms.append(
                _CachedAffineTerm(
                    fixed_contribution=np.zeros(observations.sizes[output_dim]),
                    active_design=np.asarray(
                        offset_result.design.transpose(output_dim, "offset_term").values,
                        dtype=np.float64,
                    ),
                    coefficients=offset_result.coefficients,
                    sampled_rvs=(offset_result.latent,),
                    output=offset_result.output,
                )
            )
        mean_expression = terms[0].output
        for term in terms[1:]:
            mean_expression = mean_expression + term.output
        modelled_mean = pm.Deterministic(
            "modelled_concentration",
            mean_expression,
            dims=output_dim,
        )
        cached = _add_cached_likelihood(
            tuple(terms),
            covariance=covariance,
            observations=observations,
            observation_error=observation_error,
            site_coord=site_coord,
            site_index=site_index,
            initial_amplitudes=initial_amplitudes,
            site_amplitude_prior_scale=site_amplitude_prior_scale,
            output_dim=output_dim,
        )
    return Co2CachedSigmaModel(
        model=model,
        target=cached.target,
        shared_cache=cached.shared_cache,
        initial_cache=cached.initial_cache,
        amplitude=cached.amplitude,
        states=cached.states,
        modelled_mean=modelled_mean,
        site_amplitude_prior_scale=float(site_amplitude_prior_scale),
    )


__all__ = [
    "OU_SITE_AMPLITUDE",
    "OU_SITE_DIM",
    "OU_SITE_INDEX",
    "Co2CachedSigmaModel",
    "build_co2_cached_sigma_model",
]
