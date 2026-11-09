"""Direct linked recipes sharing physical flux and separate atmospheric levels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr
from dask import compute

from openghg_inversions.array_ops import select_gathered_data_array, to_dense, to_dense_tree
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.models.boundary import Outputs, require_centered_precision
from openghg_inversions.postprocessing.boundary_outputs import BOUNDARY_FORMAT_VERSION
from openghg_inversions.models.components import (
    AffineBlock, AffineConcentrationTerm, ConcentrationTerm,
    _add_prepared_correlated_lognormal_state_with_activity,
    add_coherent_affine_component, add_correlated_lognormal_state_with_activity,
    add_linear_component, apply_linear_sensitivity, likelihood_representation,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import add_model_data, registered_model
from openghg_inversions.models.fixed_ou import (
    _site_values as fixed_ou_site_values,
    add_fixed_ou_gaussian_likelihood,
    prepare_fixed_ou_low_rank,
)
from openghg_inversions.models.shared_baseline import SharedBaseline
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import prepare_linear_sensitivity, resolve_state_activity
from openghg_inversions.observation_error import (
    AggregationError, aggregation_error_as_low_rank, validate_observation_error_arrays,
)
from openghg_inversions.rhime.co2.co2_cached_sigma_model import (
    Co2CachedSigmaModel,
    _add_cached_likelihood, _materialize_cached_linear_projection, _site_values,
)

from .co2_o2_model import _gather_co2_o2_sensitivity, _pad_channel_design
from .co2_o2_fixed_ou import linked_fixed_ou_alignment
from .shared_baseline_model import (
    _add_atmospheric_level, _validate_model_inputs, reconstruct_shared_baseline, shared_model_payload,
)


def _linked_outputs(baseline: Mapping[str, SharedBaseline]) -> Outputs:
    """Channel-prefixed level roles and units, with per-channel metadata."""
    roles: dict[str, str] = {}
    units: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "boundary_formulation": "shared_atmospheric", "boundary_representation": "centered-v1",
        "boundary_metadata": {},
    }
    for channel, bound in baseline.items():
        outputs = bound.outputs(prefix=f"{channel}_")
        roles.update(outputs.roles)
        units.update(outputs.units)
        metadata["boundary_metadata"][channel] = outputs.metadata
    return Outputs(roles, units, metadata)


def _linked_inputs(
    baseline: Mapping[str, SharedBaseline], observations: xr.DataArray,
    independent_error_sd: xr.DataArray, *, aggregation_error: AggregationError,
    priors: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, xr.Dataset], CorrelatedLognormalPrior, xr.DataArray,
    xr.DataArray, xr.DataArray, AggregationError, dict[str, Any],
]:
    """Validate native identities jointly before constructing one selected prior."""
    require_centered_precision()
    if set(baseline) != {"co2", "o2"}:
        raise ValueError("Expected co2 and o2 bound shared baselines.")
    prepared = {channel: bound.data for channel, bound in baseline.items()}
    if len({p.attrs["units"] for p in prepared.values()}) != 1:
        raise ValueError("Linked shared baselines require common channel units; prepare with matching reference units.")
    observations = observations.drop_vars([
        name for name in observations.coords if name not in observations.indexes
        and name not in {"site", "time", "observation_units"}
    ])
    independent_error_sd = independent_error_sd.drop_vars([
        name for name in independent_error_sd.coords if name not in independent_error_sd.indexes
    ])
    aggregation = {
        name: to_dense_tree(getattr(aggregation_error, name))
        for name in ("covariance", "factor", "diagonal_variance")
    }
    prepared, observations, independent_error_sd, aggregation, resolved_priors = compute(
        {channel: shared_model_payload(value, linked_identity=True).map(to_dense, keep_attrs=True)
         for channel, value in prepared.items()},
        to_dense(observations), to_dense(independent_error_sd), aggregation, to_dense_tree(dict(priors or {})),
    )
    if "observation_units" not in observations.coords:
        raise ValueError("Linked observations require a per-row observation_units coordinate.")
    bound = {}
    for channel in ("co2", "o2"):
        p = prepared[channel]
        if "bc_prescribed_centered" not in p:
            raise ValueError("Linked shared baseline requires native centered BC operands.")
        if p.attrs["mode"] == "additive":
            raise ValueError("Linked additive net mode requires an explicit GPP/TER oxidative allocation.")
        row = p.attrs["native_observation_dim"]
        native = select_gathered_data_array(
            observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim=str(observations.dims[0]),
        ).rename({observations.dims[0]: "nmeasure"}).assign_attrs(units=p.attrs["units"])
        units = observations.observation_units.sel(
            {observations.dims[0]: observations[observations.dims[0]][observations.species == channel]}
        )
        if any(value != p.attrs["units"] for value in units.values):
            raise ValueError("Observation row units must match each bound baseline.")
        rename = {row: "nmeasure"} if row != "nmeasure" else {}
        b = _validate_model_inputs(p.rename(rename), native)
        undo = {"nmeasure": row} if row != "nmeasure" else {}
        bound[channel] = b.rename(undo)
    for name in ("original_prior_mean", "original_prior_covariance", "sampled_outer", "original_outer_mean"):
        if not bound["co2"][name].identical(bound["o2"][name]):
            raise ValueError(f"Both channels must retain identical {name}.")
    for name in ("retained_mean", "retained_covariance"):
        if not bound["co2"][name].identical(bound["o2"][name]):
            raise ValueError("Both channels must retain identical physical prior moments.")
    calibration = {
        name for name in bound["co2"] if isinstance(name, str) and name.startswith("calibration_")
    }
    if calibration != {
        name for name in bound["o2"] if isinstance(name, str) and name.startswith("calibration_")
    }:
        raise ValueError("Both channels must retain the same resolved calibration identity.")
    for calibration_name in calibration:
        if not bound["co2"][calibration_name].identical(bound["o2"][calibration_name]):
            raise ValueError(f"Both channels must retain identical {calibration_name}.")
    if bound["co2"].attrs.get("calibration_attrs") != bound["o2"].attrs.get("calibration_attrs"):
        raise ValueError("Both channels must retain the same resolved calibration identity.")
    if bound["co2"].attrs["mode"] != bound["o2"].attrs["mode"]:
        raise ValueError("Both channels must use the same resolved outer mode.")
    fixed = _gather_co2_o2_sensitivity(
        bound["co2"].fixed_prior_contribution, bound["o2"].fixed_prior_contribution,
        output_dim=str(observations.dims[0]),
    ).sel({observations.dims[0]: observations[observations.dims[0]]})
    prior = CorrelatedLognormalPrior(
        bound["co2"].retained_mean, bound["co2"].retained_covariance,
        covariance_dim=bound["co2"].attrs["covariance_dim"],
    )
    independent_error_sd, _ = xr.align(independent_error_sd, observations, join="exact", copy=False)
    aggregation_error = replace(aggregation_error, **aggregation)
    return bound, prior, fixed, observations, independent_error_sd, aggregation_error, resolved_priors


def build_shared_co2_o2_model(
    baseline: Mapping[str, SharedBaseline], *, observations: xr.DataArray,
    independent_error_sd: xr.DataArray, aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: PriorArgs | None = None,
) -> pm.Model:
    """Build one linked flux state with independent physical atmospheric deltas.

    Prepared sensitivities embed oxidative ratios. The complete joint retained
    prior and aggregation covariance are retained in the joint likelihood.
    Boundary and total outputs are absolute; the likelihood carries the small
    native centered BC expression and scalar reference, retaining
    precision without constructing and subtracting an O2 background.

    Start Python with ``PYTENSOR_FLAGS=floatX=float64`` before imports,
    preserving any other configured flags. Float64 inputs alone are insufficient;
    see :ref:`linked-shared-baseline-precision`.

    Args:
        baseline: Bound shared baselines keyed by co2/o2.
        observations: Labelled absolute concentrations in the prepared units.
        independent_error_sd: Joint labelled independent concentration SDs.
        aggregation_error: Prepared fixed aggregation covariance.
        tau_hours: Fixed OU correlation times by site or channel/site.
        site_amplitude_prior: Positive OU amplitude prior in concentration units.
        fixed_site_amplitudes: Explicit fixed OU amplitudes instead of an inferred prior.

    Returns:
        Concrete PyMC model with distinct atmospheric and physical states.
        Float64 is required before eager joint transport materialization; bound
        centering means and calibrated moments are used as bound.
    """
    bound, prior, fixed, observations, independent_error_sd, aggregation_error, priors = _linked_inputs(
        baseline, observations, independent_error_sd, aggregation_error=aggregation_error,
        priors={"amplitude": site_amplitude_prior, "fixed": fixed_site_amplitudes, "tau": tau_hours},
    )
    output_dim = str(observations.dims[0])
    joint = _gather_co2_o2_sensitivity(
        bound["co2"].flux_sensitivity, bound["o2"].flux_sensitivity, output_dim=output_dim,
    ).sel({output_dim: observations[output_dim]})
    prepared_flux = prepare_linear_sensitivity(joint, output_dim=output_dim, prune_zero_columns=False)
    activity = resolve_state_activity(prepared_flux.removed)
    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(activity, prior, var_name="flux_scaling").state
        scaled_flux = apply_linear_sensitivity(
            prepared_flux, state, data_name="co2_o2_sensitivity", output_name="co2_o2_flux_contribution",
            compute_deterministic=False,
        )
        flux = add_coherent_affine_component(
            fixed, scaled_flux, output_name="co2_o2_flux_contribution",
        )
        terms = [ConcentrationTerm.absolute(flux)]
        for channel in ("co2", "o2"):
            frozen = bound[channel]
            native = frozen.bc_prescribed_centered
            prescribed_array = _pad_channel_design(native, observations, channel)
            prescribed = add_model_data(prescribed_array, f"{channel}_bc_prescribed_centered")
            channel_shift = _pad_channel_design(frozen.bc_reference_shift, observations, channel)
            terms.append(ConcentrationTerm(prescribed + channel_shift.data, prescribed, channel_shift))
            one = _pad_channel_design(xr.ones_like(native), observations, channel)
            level, _, _ = _add_atmospheric_level(
                one, float(frozen.delta_sd), prefix=f"{channel}_", output_dim=output_dim,
            )
            terms.append(ConcentrationTerm.absolute(level))
            if frozen.sizes["bc_pattern_term"]:
                dimension = "bc_pattern_term"
                design = _pad_channel_design(frozen.bc_pattern_centered, observations, channel)
                # Padding keeps the pattern labels; copy so the attrs stay local to this design.
                design = design.copy(deep=False)
                design[dimension].attrs = {"units": "1", "prior": "independent standard normal"}
                pattern = add_linear_component(
                    prepare_linear_sensitivity(design, output_dim=output_dim, prune_zero_columns=False),
                    data_name=f"{channel}_bc_pattern_operator", prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                    var_name=f"{channel}_bc_pattern_standardized", output_name=f"{channel}_bc_pattern_contribution",
                    output_dim=output_dim, coordinate_owner=channel,
                )
                terms.append(ConcentrationTerm.absolute(pattern.output))
        absolute = pm.Deterministic(
            "modelled_concentration", sum((term.output for term in terms[1:]), start=terms[0].output), dims=output_dim,
        )
        mean, observed = likelihood_representation(terms, observations, absolute)
        if priors["tau"] is None:
            if priors["fixed"] is not None or priors["amplitude"] is not None:
                raise ValueError("Fixed-OU amplitudes require tau_hours.")
            add_additive_sigma_likelihood(
                observations=observed, observation_error=independent_error_sd, mean=mean,
                aggregation_error=aggregation_error, output_dim=output_dim,
                observation_error_name="fixed_independent_error_sd",
            )
        else:
            add_fixed_ou_gaussian_likelihood(
                observations=observed, observation_error=independent_error_sd, mean=mean,
                aggregation_error=aggregation_error, output_dim=output_dim,
                tau_hours=priors["tau"], fixed_site_amplitudes=priors["fixed"],
                site_amplitude_prior=priors["amplitude"], sigma_alignment=linked_fixed_ou_alignment(observations),
            )
    return model


def build_shared_co2_o2_cached_sigma_model(
    baseline: Mapping[str, SharedBaseline], *, observations: xr.DataArray,
    independent_error_sd: xr.DataArray, aggregation_error: AggregationError,
    tau_hours: float | Mapping[str, float], site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
) -> Co2CachedSigmaModel:
    """Build the same linked mean with the existing matched OU cache target.

    Start Python with ``PYTENSOR_FLAGS=floatX=float64`` before imports,
    preserving any other configured flags. Float64 inputs alone are insufficient;
    see :ref:`linked-shared-baseline-precision`.

    Args:
        baseline: Bound shared baselines keyed by co2/o2.
        observations: Labelled absolute concentrations in the prepared units.
        independent_error_sd: Joint labelled independent concentration SDs.
        aggregation_error: Prepared fixed aggregation covariance.
        tau_hours: Fixed OU correlation times by site or channel/site.
        site_amplitude_prior: Positive OU amplitude prior in concentration units.
        initial_site_amplitudes: Optional labelled initial OU amplitudes.

    Returns:
        Cached sampling target with its model and ordered state/cache updates.
        Float64 is required before eager joint transport materialization; bound
        centering means and calibrated moments are used as bound.
    """
    bound, prior, fixed, observations, independent_error_sd, aggregation_error, priors = _linked_inputs(
        baseline, observations, independent_error_sd, aggregation_error=aggregation_error,
        priors={"amplitude": site_amplitude_prior, "initial": initial_site_amplitudes, "tau": tau_hours},
    )
    output_dim = str(observations.dims[0])
    validate_observation_error_arrays(
        observations, independent_error_sd, None, owner="Linked shared cached fixed-OU", output_dim=output_dim,
    )
    alignment = linked_fixed_ou_alignment(observations)
    site_coord = xr.DataArray(alignment.site_labels).rename({"nsigma_site": "ou_site"}).rename("ou_site")
    site_index = alignment.site_index.rename("ou_site_index")
    initial_amplitudes = _site_values(priors["initial"], site_coord=site_coord)
    joint = _gather_co2_o2_sensitivity(
        bound["co2"].flux_sensitivity, bound["o2"].flux_sensitivity, output_dim=output_dim,
    ).sel({output_dim: observations[output_dim]})
    prepared_flux = prepare_linear_sensitivity(joint, output_dim=output_dim, prune_zero_columns=False)
    activity = resolve_state_activity(prepared_flux.removed)
    prepared_flux, active_design, fixed_flux = _materialize_cached_linear_projection(
        prepared_flux, activity, output_dim=output_dim,
    )
    active_prior = prepare_active_correlated_lognormal_prior(activity, prior, var_name="flux_scaling")
    factor, diagonal = aggregation_error_as_low_rank(aggregation_error)
    covariance = prepare_fixed_ou_low_rank(
        factor, np.square(independent_error_sd.data) + diagonal,
        observations.time.data, np.asarray(site_index.data, dtype=np.int64),
        fixed_ou_site_values(priors["tau"], site_coord.size, "tau_hours", positive=True, coordinate=site_coord),
        site_labels=tuple(str(value) for value in site_coord.data),
    )
    with registered_model():
        state = _add_prepared_correlated_lognormal_state_with_activity(activity, active_prior, var_name="flux_scaling")
        scaled_flux = apply_linear_sensitivity(
            prepared_flux, state.state, data_name="co2_o2_sensitivity", output_name="co2_o2_flux_contribution",
            compute_deterministic=False,
        )
        flux = add_coherent_affine_component(
            fixed, scaled_flux, output_name="co2_o2_flux_contribution",
        )
        terms = [AffineConcentrationTerm.absolute(
            flux, fixed_contribution=fixed + fixed_flux,
            blocks=(AffineBlock(active_design, state.state[activity.active_indices]),) if activity.n_active else (),
            sampled_rvs=(state.latent,) if state.latent is not None else (),
        )]
        for channel in ("co2", "o2"):
            frozen = bound[channel]
            native = frozen.bc_prescribed_centered
            prescribed_array = _pad_channel_design(native, observations, channel)
            prescribed = add_model_data(prescribed_array, f"{channel}_bc_prescribed_centered")
            channel_shift = _pad_channel_design(frozen.bc_reference_shift, observations, channel)
            terms.append(AffineConcentrationTerm(
                prescribed + channel_shift.data, prescribed, channel_shift,
                fixed_contribution=prescribed_array, blocks=(), sampled_rvs=(),
            ))
            one = _pad_channel_design(xr.ones_like(native), observations, channel)
            zero = xr.zeros_like(prescribed_array)
            level, level_block, level_rvs = _add_atmospheric_level(
                one, float(frozen.delta_sd), prefix=f"{channel}_", output_dim=output_dim,
            )
            terms.append(AffineConcentrationTerm.absolute(
                level, fixed_contribution=zero, blocks=(level_block,), sampled_rvs=level_rvs,
            ))
            if frozen.sizes["bc_pattern_term"]:
                dimension = "bc_pattern_term"
                design = _pad_channel_design(frozen.bc_pattern_centered, observations, channel)
                # Padding keeps the pattern labels; copy so the attrs stay local to this design.
                design = design.copy(deep=False)
                design[dimension].attrs = {"units": "1", "prior": "independent standard normal"}
                pattern = add_linear_component(
                    prepare_linear_sensitivity(design, output_dim=output_dim, prune_zero_columns=False),
                    data_name=f"{channel}_bc_pattern_operator", prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                    var_name=f"{channel}_bc_pattern_standardized", output_name=f"{channel}_bc_pattern_contribution",
                    output_dim=output_dim, coordinate_owner=channel,
                )
                terms.append(AffineConcentrationTerm.absolute(
                    pattern.output, fixed_contribution=zero, blocks=(AffineBlock(design, pattern.state),),
                    sampled_rvs=(pattern.latent,) if pattern.latent is not None else (),
                ))
        absolute = pm.Deterministic(
            "modelled_concentration", sum((term.output for term in terms[1:]), start=terms[0].output),
            dims=output_dim,
        )
        mean, observed = likelihood_representation(tuple(terms), observations, absolute)
        cached = _add_cached_likelihood(
            tuple(terms), covariance=covariance, modelled_mean=mean, observations=observed,
            observation_error=independent_error_sd, site_coord=site_coord, site_index=site_index,
            initial_amplitudes=initial_amplitudes, site_amplitude_prior=priors["amplitude"], output_dim=output_dim,
        )
    return replace(cached, boundary_outputs=_linked_outputs(baseline))


def annotate_linked_shared_baseline_trace(
    trace: xr.DataTree, baseline: Mapping[str, SharedBaseline],
) -> xr.DataTree:
    """Record both bound native baselines beside restored samples.

    Args:
        trace: Coordinate-restored samples and absolute predictive draws.
        baseline: Channel-keyed bound baselines supplied to the linked builder.

    Returns:
        The trace with ``shared_baseline_<channel>`` groups and atmospheric
        coefficient units.
    """
    outputs = _linked_outputs(baseline)
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    roles.update(outputs.roles)
    metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
    metadata.update(outputs.metadata)
    for channel, bound in baseline.items():
        trace[f"shared_baseline_{channel}"] = xr.DataTree(bound.data)
        for group in trace.children.values():
            for name, units in (*outputs.units.items(), (f"{channel}_bc_pattern_standardized", "1")):
                if name in group:
                    group[name].attrs["units"] = units
            if f"{channel}_bc_mean_shift" in group:
                group[f"{channel}_bc_mean_shift"].attrs["interpretation"] = "unattributed atmospheric concentration level"
            if f"{channel}_bc_pattern_standardized" in group:
                group[f"{channel}_bc_pattern_standardized"].attrs["prior"] = "independent standard normal"
    trace.attrs.update(rhime_boundary_format_version=BOUNDARY_FORMAT_VERSION,
                       rhime_variable_roles=json.dumps(roles, sort_keys=True),
                       rhime_model_metadata=json.dumps(metadata, sort_keys=True))
    return trace


def reconstruct_linked_shared_baseline(
    prepared: Mapping[str, xr.Dataset], posterior: xr.Dataset,
) -> dict[str, xr.Dataset]:
    """Reconstruct native absolute concentrations from restored labelled samples.

    This pure operation consumes the recorded baselines and posterior state,
    level and pattern coefficients. It builds no model and recalibrates nothing.
    Coordinates must first be restored with the model's normal sample codec.

    Args:
        prepared: Channel-keyed bound baseline data (or the trace's
            ``shared_baseline_<channel>`` groups), optionally with selected native rows.
        posterior: Restored labelled selected flux states, scalar atmospheric deltas,
            and active standard-normal BC pattern draws.

    Returns:
        Channel-keyed native-row Datasets separating physical, centered and removed
        concentration terms. Reconstruction remains lazy and builds no model.
    """
    return {
        channel: reconstruct_shared_baseline(frozen, posterior, prefix=f"{channel}_")
        for channel, frozen in prepared.items()
    }
