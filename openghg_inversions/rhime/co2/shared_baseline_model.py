"""Direct CO2 recipes with a shared atmospheric level and centred physical corrections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import compute

from openghg_inversions.array_ops import to_dense, to_dense_tree
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import DEFAULT_ADDITIVE_SIGMA_PRIOR, add_additive_sigma_likelihood
from openghg_inversions.models.boundary import require_centered_precision
from openghg_inversions.models.components import (
    AffineBlock, AffineConcentrationTerm, ConcentrationTerm,
    _add_prepared_correlated_lognormal_state_with_activity,
    add_coherent_affine_component, add_correlated_lognormal_state_with_activity,
    add_linear_component, apply_linear_sensitivity, likelihood_representation,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import add_model_data, registered_model
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank, _site_values as fixed_ou_site_values
from openghg_inversions.models.shared_baseline import SharedBaseline
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import prepare_linear_sensitivity, resolve_state_activity
from openghg_inversions.observation_error import AggregationError, aggregation_error_as_low_rank, validate_observation_error_arrays
from openghg_inversions.rhime._model_building import _call_custom_likelihood
from openghg_inversions.rhime.co2.boundary_output import BOUNDARY_FORMAT_VERSION
from openghg_inversions.rhime.builders import RhimeLikelihoodBuilder
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.rhime.co2.co2_model import _fixed_mismatch_array
from openghg_inversions.rhime.co2.co2_cached_sigma_model import (
    Co2CachedSigmaModel,
    _add_cached_likelihood, _fixed_ou_site_alignment,
    _materialize_cached_linear_projection, _site_values,
)


def shared_model_payload(prepared: xr.Dataset, *, linked_identity: bool = False) -> xr.Dataset:
    """Select lazy numerical graph operands, leaving bound diagnostics borrowed.

    Args:
        prepared: Bound shared baseline data.
        linked_identity: Include original moments and calibration facts needed
            to compare two channels before constructing their common prior.

    Returns:
        A lazy dataset in concentration units, with dimensionless retained
        moments and standardized pattern coordinates. No payload is computed.
    """
    if "bc_pattern_centered" not in prepared:
        prepared = prepared.assign(bc_pattern_centered=(prepared.bc_pattern_raw - prepared.bc_pattern_mean).assign_attrs(
            units=prepared.attrs["units"],
        ))
    names = {
        "retained_mean", "retained_covariance", "flux_sensitivity",
        "fixed_prior_contribution", "bc_prescribed_centered", "bc_reference_shift", "delta_sd",
        "bc_pattern_centered", "additive_centered",
    }
    if linked_identity:
        names.update({"original_prior_mean", "original_prior_covariance", "sampled_outer", "original_outer_mean"})
        names.update(name for name in prepared if isinstance(name, str) and name.startswith("calibration_"))
    payload = prepared[[name for name in prepared if name in names]]
    row = prepared.attrs.get("native_observation_dim", "nmeasure")
    unused_coordinates = [
        name for name, coord in payload.coords.items()
        if row in coord.dims and name not in {row, "site", "time"} and name not in payload.indexes
    ]
    return payload.drop_vars(unused_coordinates)


def _add_atmospheric_level(
    response: xr.DataArray, delta_sd: float, *, prefix: str = "", output_dim: str = "nmeasure",
) -> tuple[pt.TensorVariable, AffineBlock, tuple]:
    """One unattributed atmospheric level responding with one on its own rows.

    ``response`` is one on the channel's rows and zero on any other channel's.
    """
    delta = pm.Normal(f"{prefix}bc_mean_shift", mu=0.0, sigma=delta_sd)
    level = pm.Deterministic(
        f"{prefix}mu_bc_mean_shift", pt.as_tensor_variable(response.values) * delta, dims=output_dim,
    )
    design = response.expand_dims(bc_mean=[0], axis=1).assign_attrs(units="1")
    return level, AffineBlock(design, pt.atleast_1d(delta)), (delta,)


def _validate_model_inputs(
    prepared: xr.Dataset, observations: xr.DataArray,
) -> xr.Dataset:
    """Validate materialized bound arrays at their owning recipe boundary."""
    prepared, observations = xr.align(prepared, observations, join="exact", copy=False)
    if observations.attrs.get("units") != prepared.attrs["units"]:
        raise ValueError("Shared baseline and observations must use the same declared concentration units.")
    if any(
        not np.isfinite(v.values).all()
        for v in prepared.data_vars.values()
        if np.issubdtype(v.dtype, np.number)
    ):
        raise ValueError("Shared baseline numerical operands must be finite.")
    return prepared


def _model_inputs(
    prepared: xr.Dataset, observations: xr.DataArray,
) -> tuple[xr.Dataset, CorrelatedLognormalPrior]:
    """Validate bound operands and construct their selected physical prior."""
    prepared = _validate_model_inputs(prepared, observations)
    prior = CorrelatedLognormalPrior(
        prepared.retained_mean,
        prepared.retained_covariance,
        covariance_dim=prepared.attrs["covariance_dim"],
    )
    return prepared, prior


def _materialize_shared_inputs(
    prepared: xr.Dataset,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    *,
    priors: dict[str, Any],
) -> tuple[xr.Dataset, CorrelatedLognormalPrior, xr.DataArray, xr.DataArray, AggregationError, dict[str, Any]]:
    """Compute selected transport, error and prior operands together for PyMC."""
    payload = shared_model_payload(prepared).map(to_dense, keep_attrs=True)
    # Only observation identity and noise-policy coordinates enter the graph.
    observations, observation_error = (
        array.drop_vars([
            name for name in array.coords
            if name not in array.indexes and name not in {"site", "time"}
        ])
        for array in (observations, observation_error)
    )
    aggregation = {
        name: to_dense_tree(getattr(aggregation_error, name))
        for name in ("covariance", "factor", "diagonal_variance")
    }
    prepared, observations, observation_error, aggregation, priors = compute(
        payload, to_dense(observations), to_dense(observation_error), aggregation, to_dense_tree(priors),
    )
    prepared, prior = _model_inputs(prepared, observations)
    observation_error, _ = xr.align(observation_error, observations, join="exact", copy=False)
    return prepared, prior, observations, observation_error, replace(aggregation_error, **aggregation), priors


def build_shared_co2_model(
    baseline: SharedBaseline,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
) -> pm.Model:
    """Build ordinary CO2 with a bound shared baseline and existing noise policy.

    The scalar reference centres only the likelihood; named predictions retain
    absolute concentration. The physical source of delta remains unidentified.
    Linked CO2/O2 channels are not supported by this separate recipe.

    Args:
        baseline: Shared baseline bound to these nmeasure rows.
        observations: Labelled absolute concentrations in the prepared units.
        observation_error: Labelled measurement concentration SDs.
        aggregation_error: Prepared fixed aggregation covariance.
        likelihood_builder (collections.abc.Callable, optional): Ordinary likelihood callable.
        likelihood_kwargs: Resolved options for that likelihood.
        sigma_alignment: Optional observation/site-period mismatch alignment.
        sigma_prior: Optional positive mismatch SD prior.
        fixed_model_mismatch: Optional fixed mismatch concentration SD.

    Returns:
        Concrete PyMC model with distinct atmospheric and physical states.
        Float64 is required before eager joint transport materialization; bound
        centering means and calibrated moments are used as bound.
    """
    require_centered_precision()
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
            "additive-sigma or fixed-mismatch options; pass its options in likelihood_kwargs."
        )
    prepared, prior, observations, observation_error, aggregation_error, priors = _materialize_shared_inputs(
        baseline.data, observations, observation_error, aggregation_error,
        priors={"sigma": sigma_prior, "fixed_mismatch": fixed_model_mismatch, "likelihood": likelihood_kwargs},
    )
    sigma_prior, likelihood_kwargs = priors["sigma"], priors["likelihood"]
    fixed_mismatch = _fixed_mismatch_array(observations, priors["fixed_mismatch"])
    prepared_flux = prepare_linear_sensitivity(prepared.flux_sensitivity, prune_zero_columns=False)
    activity = resolve_state_activity(prepared_flux.removed)

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(activity, prior, var_name="flux_scaling").state
        scaled_flux = apply_linear_sensitivity(
            prepared_flux, state, data_name="co2_sensitivity", output_name="co2_flux_contribution",
            compute_deterministic=False,
        )
        flux = add_coherent_affine_component(
            prepared.fixed_prior_contribution, scaled_flux, output_name="co2_flux_contribution",
        )
        terms = [ConcentrationTerm.absolute(flux)]
        prescribed = add_model_data(prepared.bc_prescribed_centered, "bc_prescribed_centered")
        shift = prepared.bc_reference_shift
        terms.append(ConcentrationTerm(prescribed + shift.data, prescribed, shift))
        level, _, _ = _add_atmospheric_level(
            xr.ones_like(prepared.bc_prescribed_centered), float(prepared.delta_sd),
        )
        terms.append(ConcentrationTerm.absolute(level))
        for source, variable, dimension, coefficient in (
            ("bc", "bc_pattern_centered", "bc_pattern_term", "bc_pattern_standardized"),
            ("outer", "additive_centered", "outer_pattern", "outer_pattern_coefficient"),
        ):
            if variable not in prepared or not prepared.sizes[dimension]:
                continue
            design = prepared[variable].assign_coords({dimension: prepared[dimension].copy(deep=False)})
            design[dimension].attrs = {"units": "1", "prior": "independent standard normal"}
            pattern = add_linear_component(
                prepare_linear_sensitivity(design, prune_zero_columns=False),
                data_name=f"{source}_pattern_operator", prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                var_name=coefficient, output_name=f"{source}_pattern_contribution",
            )
            terms.append(ConcentrationTerm.absolute(pattern.output))
        absolute = pm.Deterministic(
            "modelled_concentration", sum((term.output for term in terms[1:]), start=terms[0].output), dims="nmeasure",
        )
        mean, observed = likelihood_representation(terms, observations, absolute)
        if likelihood_builder is None:
            add_additive_sigma_likelihood(
                observations=observed, observation_error=observation_error, aggregation_error=aggregation_error,
                fixed_model_mismatch=fixed_mismatch, mean=mean,
                additive_sigma_alignment=sigma_alignment, additive_sigma_prior=sigma_prior, output_dim="nmeasure",
            )
        else:
            _call_custom_likelihood(
                likelihood_builder, observations=observed, observation_error=observation_error,
                aggregation_error=aggregation_error, mean=mean, output_dim="nmeasure", likelihood_kwargs=likelihood_kwargs,
            )
    return model


def build_shared_co2_cached_sigma_model(
    baseline: SharedBaseline,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
) -> Co2CachedSigmaModel:
    """Build the same shared CO2 mean with the existing cached fixed-OU target.

    Args:
        baseline: Shared baseline bound to these nmeasure rows.
        observations: Labelled absolute concentrations in the prepared units.
        observation_error: Labelled measurement concentration SDs.
        aggregation_error: Prepared fixed aggregation covariance.
        tau_hours: Fixed OU correlation times by site or channel/site.
        site_amplitude_prior: Positive OU amplitude prior in concentration units.
        initial_site_amplitudes: Optional labelled initial OU amplitudes.

    Returns:
        Cached sampling target with its model and ordered state/cache updates.
        Float64 is required before eager joint transport materialization; bound
        centering means and calibrated moments are used as bound.
    """
    require_centered_precision()
    prepared, prior, observations, observation_error, aggregation_error, priors = _materialize_shared_inputs(
        baseline.data, observations, observation_error, aggregation_error,
        priors={"amplitude": site_amplitude_prior, "initial": initial_site_amplitudes, "tau": tau_hours},
    )
    validate_observation_error_arrays(
        observations, observation_error, None, owner="Shared cached fixed-OU CO2 likelihood", output_dim="nmeasure",
    )
    site_coord, site_index = _fixed_ou_site_alignment(observations, output_dim="nmeasure")
    initial_amplitudes = _site_values(priors["initial"], site_coord=site_coord)
    prepared_flux = prepare_linear_sensitivity(prepared.flux_sensitivity, prune_zero_columns=False)
    activity = resolve_state_activity(prepared_flux.removed)
    prepared_flux, active_design, fixed_flux = _materialize_cached_linear_projection(
        prepared_flux, activity, output_dim="nmeasure",
    )
    active_prior = prepare_active_correlated_lognormal_prior(activity, prior, var_name="flux_scaling")
    factor, diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor, np.square(observation_error.data) + diagonal,
        observations.time.data, np.asarray(site_index.data, dtype=np.int64),
        fixed_ou_site_values(priors["tau"], site_coord.size, "tau_hours", positive=True, coordinate=site_coord),
        site_labels=tuple(str(value) for value in site_coord.data),
    )
    with registered_model():
        state = _add_prepared_correlated_lognormal_state_with_activity(activity, active_prior, var_name="flux_scaling")
        scaled_flux = apply_linear_sensitivity(
            prepared_flux, state.state, data_name="co2_sensitivity", output_name="co2_flux_contribution",
            compute_deterministic=False,
        )
        flux = add_coherent_affine_component(
            prepared.fixed_prior_contribution, scaled_flux, output_name="co2_flux_contribution",
        )
        terms = [AffineConcentrationTerm.absolute(
            flux, fixed_contribution=prepared.fixed_prior_contribution + fixed_flux,
            blocks=(AffineBlock(active_design, state.state[activity.active_indices]),) if activity.n_active else (),
            sampled_rvs=(state.latent,) if state.latent is not None else (),
        )]
        prescribed = add_model_data(prepared.bc_prescribed_centered, "bc_prescribed_centered")
        shift = prepared.bc_reference_shift
        terms.append(AffineConcentrationTerm(
            prescribed + shift.data, prescribed, shift,
            fixed_contribution=prepared.bc_prescribed_centered, blocks=(), sampled_rvs=(),
        ))
        zero = xr.zeros_like(prepared.bc_prescribed_centered)
        level, level_block, level_rvs = _add_atmospheric_level(
            xr.ones_like(prepared.bc_prescribed_centered), float(prepared.delta_sd),
        )
        terms.append(AffineConcentrationTerm.absolute(
            level, fixed_contribution=zero, blocks=(level_block,), sampled_rvs=level_rvs,
        ))
        for source, variable, dimension, coefficient in (
            ("bc", "bc_pattern_centered", "bc_pattern_term", "bc_pattern_standardized"),
            ("outer", "additive_centered", "outer_pattern", "outer_pattern_coefficient"),
        ):
            if variable not in prepared or not prepared.sizes[dimension]:
                continue
            design = prepared[variable].assign_coords({dimension: prepared[dimension].copy(deep=False)})
            design[dimension].attrs = {"units": "1", "prior": "independent standard normal"}
            pattern = add_linear_component(
                prepare_linear_sensitivity(design, prune_zero_columns=False),
                data_name=f"{source}_pattern_operator", prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                var_name=coefficient, output_name=f"{source}_pattern_contribution",
            )
            terms.append(AffineConcentrationTerm.absolute(
                pattern.output, fixed_contribution=zero, blocks=(AffineBlock(design, pattern.state),),
                sampled_rvs=(pattern.latent,) if pattern.latent is not None else (),
            ))
        absolute = pm.Deterministic(
            "modelled_concentration", sum((term.output for term in terms[1:]), start=terms[0].output),
            dims="nmeasure",
        )
        mean, observed = likelihood_representation(tuple(terms), observations, absolute)
        cached = _add_cached_likelihood(
            tuple(terms), covariance=covariance, modelled_mean=mean, observations=observed,
            observation_error=observation_error, site_coord=site_coord, site_index=site_index,
            initial_amplitudes=initial_amplitudes, site_amplitude_prior=priors["amplitude"], output_dim="nmeasure",
        )
    return replace(cached, boundary_outputs=baseline.outputs())


def annotate_shared_baseline_trace(trace: xr.DataTree, baseline: SharedBaseline) -> xr.DataTree:
    """Record the bound baseline the model used beside sampled output.

    Call after the ordinary coordinate restoration and absolute-predictive
    restoration used by the CO2 recipe. The ``shared_baseline`` group holds
    the centring, density widths, retained prior identity and any resolved
    calibration, so replay needs no preparation.

    Args:
        trace: DataTree of coordinate-restored samples and absolute predictive draws.
        baseline: Bound baseline supplied to the shared model builder.

    Returns:
        The trace with its shared-baseline group and coefficient units.
    """
    prepared = baseline.data
    trace["shared_baseline"] = xr.DataTree(prepared)
    trace.attrs["rhime_boundary_format_version"] = BOUNDARY_FORMAT_VERSION
    outputs = baseline.outputs()
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    roles.update(outputs.roles)
    trace.attrs["rhime_variable_roles"] = json.dumps(roles, sort_keys=True)
    metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
    metadata.update(outputs.metadata)
    trace.attrs["rhime_model_metadata"] = json.dumps(metadata, sort_keys=True)
    for group in trace.children.values():
        if "bc_mean_shift" in group:
            group["bc_mean_shift"].attrs.update(units=prepared.attrs["units"],
                interpretation="unattributed atmospheric concentration level")
        if "mu_bc_mean_shift" in group:
            group["mu_bc_mean_shift"].attrs["units"] = prepared.attrs["units"]
        for name in ("bc_pattern_standardized", "outer_pattern_coefficient"):
            if name in group:
                group[name].attrs.update(units="1", prior="independent standard normal")
    return trace


def reconstruct_shared_baseline(
    prepared: xr.Dataset, posterior: xr.Dataset, *, prefix: str = "",
) -> xr.Dataset:
    """Reconstruct atmospheric and physical terms from bound inputs and draws.

    Args:
        prepared: Bound baseline data (or the trace's ``shared_baseline``
            group), optionally with selected rows.
        posterior: Labelled flux_scaling, bc_mean_shift, and active pattern
            coefficient draws, with restored model coordinates.
        prefix: Channel prefix for linked atmospheric and boundary coefficients.

    Returns:
        Dataset of absolute modelled concentration, atmospheric delta, raw and
        centered BC/outer corrections, removed means and physical contributions.
        Computation remains lazy; stored centering means are never recomputed.
    """
    p = prepared
    dim = p.attrs["state_dim"]
    state = posterior.flux_scaling.sel({dim: p[dim]})
    delta = posterior[f"{prefix}bc_mean_shift"]

    def pattern(variable: str, operator: xr.DataArray, dimension: str) -> xr.DataArray:
        coefficient = posterior[variable]
        sample_dim = str(coefficient.dims[-1])
        if not coefficient.get_index(sample_dim).equals(p.get_index(dimension)):
            raise ValueError("Posterior pattern labels must match the bound baseline.")
        coefficient = coefficient.rename({sample_dim: dimension}).assign_coords({dimension: p[dimension]})
        return xr.dot(operator, coefficient, dim=dimension)

    if p.sizes["bc_pattern_term"]:
        bc = pattern(f"{prefix}bc_pattern_standardized", p.bc_pattern_raw - p.bc_pattern_mean, "bc_pattern_term")
        raw_bc = pattern(f"{prefix}bc_pattern_standardized", p.bc_pattern_raw, "bc_pattern_term")
    else:
        bc = xr.zeros_like(p.physical_bc_prior)
        raw_bc = bc
    if "additive_centered" in p:
        outer = pattern(f"{prefix}outer_pattern_coefficient", p.additive_centered, "outer_pattern")
        raw = pattern(f"{prefix}outer_pattern_coefficient", p.additive_raw, "outer_pattern")
    else:
        departure = state - p.retained_mean
        if p.attrs["mode"].startswith("multiplicative"):
            centered = p.flux_sensitivity.where(p.sampled_outer, 0)
            positions = p.get_index("original_state").get_indexer(p.get_index(dim))
            means = p.outer_column_mean.isel(original_state=positions).data
            raw_operator = (centered + xr.DataArray(means, dims=dim, coords={dim: p[dim]})).where(p.sampled_outer, 0)
            outer = xr.dot(centered, departure, dim=dim)
            raw = xr.dot(raw_operator, departure, dim=dim)
        else:
            outer = xr.zeros_like(p.physical_outer_prior)
            raw = outer
    result = xr.Dataset({
        "baseline_delta": delta, "bc_raw_correction": raw_bc,
        "bc_centered_correction": bc, "bc_removed_mean": raw_bc - bc,
        "physical_bc_contribution": p.physical_bc_prior + raw_bc,
        "outer_raw_correction": raw, "outer_centered_correction": outer,
        "outer_removed_mean": raw - outer,
        "outer_physical_contribution": p.physical_outer_prior + raw,
        "shared_baseline_contribution": p.physical_prior_baseline + delta + bc + outer,
        "modelled_concentration": p.fixed_prior_contribution + p.physical_bc_prior + delta + bc
        + xr.dot(p.flux_sensitivity, state, dim=dim) + (outer if "additive_centered" in p else 0),
    }, attrs={"units": p.attrs["units"]})
    for value in result.data_vars.values():
        value.attrs["units"] = p.attrs["units"]
    if "additive_density_sd" in p:
        coefficient = posterior[f"{prefix}outer_pattern_coefficient"]
        coefficient = coefficient.rename({coefficient.dims[-1]: "outer_pattern"})
        result["outer_density_correction"] = coefficient * p.additive_density_sd
        result.outer_density_correction.attrs["units"] = p.additive_density_sd.attrs["units"]
    return result
