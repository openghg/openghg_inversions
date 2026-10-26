"""CO2 graph for cached fixed-OU site-amplitude sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray
import pymc as pm
from pymc.pytensorf import floatX
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor.variable import TensorVariable

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
from openghg_inversions.models.fixed_ou import (
    FixedOuLowRank,
    prepare_fixed_ou_low_rank,
    _site_values as fixed_ou_site_values,
)
from openghg_inversions.models.priors import PriorArgs, parse_prior
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
from openghg_inversions.inference.cached_sigma import PytensorMarginalQuadraticCache
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from openghg_inversions.sigma import SigmaAlignment


OU_SITE_DIM = "ou_site"
OU_SITE_INDEX = "ou_site_index"
OU_SITE_AMPLITUDE = "ou_site_amplitude"


@dataclass(frozen=True)
class Co2CachedSigmaModel:
    """Observed graph, cached state expression, and scientific starting values.

    ``initial_values`` contains untransformed sampling overrides or PyMC
    initialization strategies, lifted from graph metadata. The matched runner
    resolves them separately for construction and each sampling chain.
    """

    model: pm.Model
    target: FixedOuCachedSigmaTarget
    shared_cache: PytensorMarginalQuadraticCache
    initial_cache: MarginalQuadraticCache
    amplitude: TensorVariable
    states: tuple[TensorVariable, ...]
    modelled_mean: TensorVariable
    cached_state_logp: TensorVariable
    initial_values: dict[str, Any]


@dataclass(frozen=True)
class _CachedAffineTerm:
    """One scientific affine term and its private cached representation."""

    fixed_contribution: NDArray[np.float64]
    active_design: NDArray[np.float64]
    coefficients: TensorVariable | None
    sampled_rvs: tuple[TensorVariable, ...]
    output: TensorVariable


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
) -> np.ndarray | None:
    if value is None:
        return None
    return fixed_ou_site_values(
        value, site_coord.size, "Initial site amplitudes", positive=True, coordinate=site_coord,
    )


def _add_cached_likelihood(
    terms: tuple[_CachedAffineTerm, ...],
    *,
    covariance: FixedOuLowRank,
    modelled_mean: TensorVariable,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    site_coord: xr.DataArray,
    site_index: xr.DataArray,
    initial_amplitudes: np.ndarray | None,
    site_amplitude_prior: PriorArgs,
    output_dim: str,
) -> Co2CachedSigmaModel:
    """Lower ordered scientific terms once at the cached-likelihood boundary."""
    active_terms = tuple(term for term in terms if term.coefficients is not None)
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
        design=(
            np.column_stack([term.active_design for term in active_terms])
            if active_terms else np.empty((observations.size, 0))
        ),
    )
    add_coords({OU_SITE_DIM: site_coord})
    amplitude_prior = dict(site_amplitude_prior)
    amplitude_prior["pdf"] = str(amplitude_prior.get("pdf", "")).casefold().replace("-", "")
    if amplitude_prior["pdf"] == "lognormal":
        if amplitude_prior.get("reparameterise", False):
            warn(
                "Cached amplitude LogNormal priors use the log transform; "
                "reparameterise=True is normalized to False without changing the physical prior.",
                UserWarning,
                stacklevel=2,
            )
        amplitude_prior["reparameterise"] = False
    amplitude = parse_prior(OU_SITE_AMPLITUDE, amplitude_prior, positive=True, dims=OU_SITE_DIM)
    model = pm.modelcontext(None)
    amplitude_value = model.rvs_to_values[amplitude]
    physical_amplitude = model.replace_rvs_by_values([amplitude])[0]
    amplitude_logp = model.logp(vars=[amplitude], jacobian=True)
    dependencies = set(graph_inputs([physical_amplitude, amplitude_logp]))
    if any(value in dependencies for value in model.value_vars if value is not amplitude_value):
        raise ValueError("Cached amplitudes require independent priors without other free variables.")
    overrides = {} if initial_amplitudes is None else {amplitude: floatX(initial_amplitudes)}
    point = pm.initial_point.make_initial_point_fn(
        model=model, overrides=overrides, return_transformed=True,
    )(seed=0)
    eta = point[str(amplitude_value.name)]
    initial_amplitudes, prior_density = model.compile_fn(
        [physical_amplitude, amplitude_logp], inputs=[amplitude_value],
    )({str(amplitude_value.name): eta})
    if (not np.isfinite(eta).all() or not np.isfinite(prior_density)
            or not np.isfinite(initial_amplitudes).all() or np.any(initial_amplitudes <= 0.0)):
        raise ValueError("Initial site amplitudes must be strictly positive with finite transformed point and prior density.")
    initial_cache = target.refresh(initial_amplitudes)
    shared_cache = PytensorMarginalQuadraticCache(initial_cache)
    observed = add_model_data(observations.transpose(output_dim), "Y")
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
    pm.Deterministic(
        "epsilon",
        pt.sqrt(covariance.marginal_variance(amplitude)),
        dims=output_dim,
    )
    coefficients = (
        pt.concatenate([cast(TensorVariable, term.coefficients) for term in active_terms])
        if active_terms else pt.as_tensor_variable(np.empty(0, dtype=amplitude.dtype))
    )
    cached_state_logp = shared_cache.log_likelihood(coefficients)
    pm.CustomDist(
        "y",
        modelled_mean,
        amplitude,
        logp=covariance.logp,
        random=covariance.random,
        signature="(n),(s)->(n)",
        observed=observed,
        dims=output_dim,
    )
    model = pm.modelcontext(None)
    initial_values = {
        str(rv.name): value
        for rv, value in model.rvs_to_initial_values.items()
        if value is not None
    }
    initial_values[OU_SITE_AMPLITUDE] = floatX(initial_amplitudes).copy()
    # PyMC's generic log-likelihood transformation requires default graph
    # initvals. Preserve scientific starts separately for the matched runner.
    for rv in model.free_RVs:
        model.set_initval(rv, None)
    return Co2CachedSigmaModel(
        model=model,
        target=target,
        shared_cache=shared_cache,
        initial_cache=initial_cache,
        amplitude=amplitude,
        states=tuple(rv for term in terms for rv in term.sampled_rvs),
        modelled_mean=modelled_mean,
        cached_state_logp=cached_state_logp,
        initial_values=initial_values,
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
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: xr.DataArray | None = None,
    boundary_operands: xr.Dataset | None = None,
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
    likelihood is one normalized observed joint Gaussian with exact density
    and random callbacks. A separate cached quadratic over the active flux,
    boundary, and offset coefficients supplies efficient state trajectories.
    These states remain separate scientific variables and sampler inputs.
    The matched runner supplies the returned scientific initial values through
    sampling inputs. Observations, designs, activity masks, and covariance are
    fixed after construction; rebuild the recipe for different inputs.

    Args:
        flux_sensitivity: Reduced sensitivity with the output dimension and
            one labelled retained-state dimension, in concentration units per
            unit dimensionless flux scaling.
        retained_prior: Complete labelled arithmetic-moment prior for the
            positive, dimensionless retained flux-scaling state.
        fixed_prior_contribution: Fixed coherent-reduction affine contribution
            on the output dimension, in the observations' concentration units.
        observations: Observed CO2 concentrations on the output dimension,
            with observation-aligned ``site`` and ``time`` coordinates.
        observation_error: Reported observation standard deviation, in the
            observations' concentration units.
        aggregation_error: Prepared fixed aggregation-error covariance, in
            squared concentration units.
        tau_hours: Fixed OU decorrelation time in hours. A scalar applies to
            every site; a mapping must cover every observed site label.
        site_amplitude_prior: Independent positive prior specification for site
            amplitudes, in the observations' concentration units.
        initial_site_amplitudes: Optional positive initial amplitude in the
            same units, supplied as one scalar or a mapping covering every
            observed site. Defaults to the prior support point.
        state_activity: Optional labelled active/fixed policy for retained flux
            states. Fixed values are dimensionless flux scalings.
        boundary_sensitivity: Optional atmospheric boundary-condition
            sensitivity, in concentration units per unit dimensionless
            boundary scaling.
        bc_prior: Optional prior for dimensionless boundary-scaling
            coefficients. Requires ``boundary_sensitivity``; when omitted for
            a selected boundary component, the default boundary prior is used.
        bc_state_activity: Optional labelled active/fixed policy for boundary
            scalings. Requires ``boundary_sensitivity``.
        offset_prior: Optional prior for additive offsets. Its location and
            scale parameters use the observations' concentration units. When
            omitted, no offset is added.
        offset_freq: Optional frequency used to derive offset periods from the
            observation-aligned ``time`` coordinate. Requires ``offset_prior``.
        offset_drop_first: Whether to omit the first site's offset terms.
            ``True`` requires ``offset_prior`` and a per-site offset.
        offset_per_site: Whether offsets vary by site. ``False`` creates one
            global offset and does not accept ``offset_freq`` or
            ``offset_drop_first=True``. The non-default value requires
            ``offset_prior``.
        output_dim: Name of the shared observation and model-output dimension.

    Returns:
        The registered model, fixed-OU target and caches, ordered sampler
        states, and completed concentration mean.

    Raises:
        ValueError: If scientific arrays, coordinates, covariance inputs,
            priors, or activity policies are invalid; if boundary options are
            supplied without ``boundary_sensitivity``; or if non-default
            offset options are supplied without ``offset_prior``. Also raised
            for incompatible global-offset options.
    """
    if boundary_operands is not None:
        raise ValueError("Centered boundaries do not support cached likelihoods.")
    if boundary_sensitivity is None and (
        bc_prior is not None or bc_state_activity is not None
    ):
        raise ValueError("bc_prior and bc_state_activity require boundary_sensitivity.")
    if offset_prior is None and (
        offset_freq is not None or offset_drop_first or not offset_per_site
    ):
        raise ValueError("Non-default offset options require offset_prior.")
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
        fixed_ou_site_values(
            tau_hours, site_coord.size, "tau_hours", positive=True, coordinate=site_coord,
        ),
        site_labels=site_labels,
    )
    with registered_model():
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
                coordinate_owner="bc",
            )
            boundary_output = apply_linear_sensitivity(
                prepared_boundary,
                boundary_result.state,
                data_name="hbc",
                output_name="mu_bc",
                coordinate_owner="bc",
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
        return _add_cached_likelihood(
            tuple(terms),
            covariance=covariance,
            modelled_mean=modelled_mean,
            observations=observations,
            observation_error=observation_error,
            site_coord=site_coord,
            site_index=site_index,
            initial_amplitudes=initial_amplitudes,
            site_amplitude_prior=site_amplitude_prior,
            output_dim=output_dim,
        )


__all__ = [
    "OU_SITE_AMPLITUDE",
    "OU_SITE_DIM",
    "OU_SITE_INDEX",
    "Co2CachedSigmaModel",
    "build_co2_cached_sigma_model",
]
