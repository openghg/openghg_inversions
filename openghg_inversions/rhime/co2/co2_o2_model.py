"""Explicit PyMC graph for the CO2/O2 recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.array_ops import concat_gather_data_arrays
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    StateActivity,
    add_coherent_affine_component,
    add_correlated_lognormal_state_with_activity,
    apply_linear_sensitivity,
    prepare_linear_sensitivity,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.models.components import (
    LinearComponentResult,
    OffsetComponentResult,
    _add_offset_component_result,
    add_linear_component,
    add_model_data,
)
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from .co2_model import _normalise_offset_args
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood

from .co2_o2_fixed_ou import linked_fixed_ou_alignment


def _gather_co2_o2_sensitivity(
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    *,
    output_dim: str = "observation",
) -> xr.DataArray:
    """Gather unequal CO2/O2 rows on one labelled observation axis.

    The gathered ``(species, channel_observation)`` index retains each native
    channel axis.

    Args:
        co2_sensitivity: CO2 sensitivity with its channel observation dimension
            first and the shared retained-state dimension second.
        o2_sensitivity: O2 sensitivity with its distinct channel observation
            dimension first and the same retained-state dimension second.
        output_dim: Name for the gathered observation dimension.

    Returns:
        The full gathered sensitivity.
    """
    channel_dim = "channel_observation"
    return concat_gather_data_arrays(
        {
            "co2": co2_sensitivity.rename({co2_sensitivity.dims[0]: channel_dim}),
            "o2": o2_sensitivity.rename({o2_sensitivity.dims[0]: channel_dim}),
        },
        key_dim="species",
        ragged_dim=channel_dim,
        stack_dim=output_dim,
        join="exact",
    )


def evaluate_co2_o2_prior_forward_mean(
    *,
    fixed_prior_contribution: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    state_activity: StateActivity | None = None,
    output_dim: str = "observation",
) -> xr.DataArray:
    """Evaluate the reduced model so callers can verify prior-forward preservation.

    Args:
        fixed_prior_contribution: Joint coherent-reduction intercept on
            ``output_dim``.
        co2_sensitivity: CO2 retained-state sensitivity with its own observation
            dimension and the retained-prior state dimension.
        o2_sensitivity: O2 retained-state sensitivity with a distinct observation
            dimension and the same retained-prior state dimension.
        retained_prior: Correlated retained-state arithmetic moments.
        state_activity: Optional labelled policy fixing or activating retained
            states. Structural zero columns are inactive in both channels.
        output_dim: Dimension of the returned joint observation vector.

    Returns:
        The labelled joint prior-forward concentration
        ``fixed_prior_contribution + H_joint @ prior_state``.

    Raises:
        ValueError: If the channel sensitivities cannot form one shared-state
            sensitivity or ``state_activity`` does not align with that state.
    """
    joint_sensitivity = _gather_co2_o2_sensitivity(
        co2_sensitivity,
        o2_sensitivity,
        output_dim=output_dim,
    )
    prepared_sensitivity = prepare_linear_sensitivity(
        joint_sensitivity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(prepared_sensitivity.removed, state_activity)
    # Active states are evaluated at their arithmetic prior means; inactive
    # states retain the exact values declared by the activity policy.
    resolved_prior_state = retained_prior.mean.where(
        activity.active,
        activity.fixed_value,
    )
    joint_contribution = xr.dot(
        joint_sensitivity,
        resolved_prior_state,
        dim=retained_prior.state_dim,
    )
    return (fixed_prior_contribution + joint_contribution).rename("prior_forward_concentration")


def _validate_channel_baseline_options(
    boundary_sensitivity: Mapping[str, xr.DataArray] | None,
    bc_prior: Mapping[str, PriorArgs] | None,
    bc_state_activity: Mapping[str, StateActivity] | None,
    offset_prior: Mapping[str, PriorArgs] | None,
    offset_args: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Reject unused channel options before constructing or sampling a model."""
    for name, values in (
        ("boundary_sensitivity", boundary_sensitivity),
        ("bc_prior", bc_prior),
        ("bc_state_activity", bc_state_activity),
        ("offset_prior", offset_prior),
        ("offset_args", offset_args),
    ):
        if values is not None and (not isinstance(values, Mapping) or set(values) - {"co2", "o2"}):
            raise ValueError(f"{name} must be a mapping keyed only by 'co2' and 'o2'.")
    for channel in ("co2", "o2"):
        if channel not in (boundary_sensitivity or {}) and (
            channel in (bc_prior or {}) or channel in (bc_state_activity or {})
        ):
            raise ValueError(f"{channel} boundary options require boundary_sensitivity.")
        if channel in (offset_args or {}) and channel not in (offset_prior or {}):
            raise ValueError(f"{channel} offset_args require offset_prior.")
        _normalise_offset_args((offset_args or {}).get(channel))


def _pad_channel_design(
    design: xr.DataArray,
    observations: xr.DataArray,
    channel: str,
) -> xr.DataArray:
    """Place a native channel design on joint rows with exact zeros elsewhere."""
    native_dim, state_dim = design.dims
    row_index = observations.indexes[observations.dims[0]]
    channel_index = row_index[row_index.get_level_values("species") == channel]
    design = design.sel({native_dim: channel_index.get_level_values("channel_observation").to_numpy()})
    native = design.drop_vars(
        [name for name, coord in design.coords.items() if native_dim in coord.dims]
    ).rename({native_dim: observations.dims[0]})
    native = native.assign_coords(
        xr.Coordinates.from_pandas_multiindex(channel_index, str(observations.dims[0]))
    )
    # Restore the joint row order after selecting the native labels above.
    return native.reindex_like(observations, fill_value=0).transpose(observations.dims[0], state_dim)


def _add_co2_o2_baseline_components(
    *,
    observations: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[tuple[str, LinearComponentResult | OffsetComponentResult, xr.DataArray]]:
    """Build independent channel terms with padded joint designs and outputs.

    Returned tuples retain the channel, ordinary CO2 component result, and
    full joint design so the fixed-OU recipe can use the same affine terms.
    """
    _validate_channel_baseline_options(
        boundary_sensitivity,
        bc_prior,
        bc_state_activity,
        offset_prior,
        offset_args,
    )
    if not boundary_sensitivity and not offset_prior:
        return []
    if len(np.unique(observations["observation_units"].values)) != 1:
        raise ValueError("Linked boundary and offset components currently require identical channel units.")
    output_dim = str(observations.dims[0])
    components = []
    boundaries = []
    offsets = []
    for channel, sensitivity in (("co2", co2_sensitivity), ("o2", o2_sensitivity)):
        if channel in (boundary_sensitivity or {}):
            native = boundary_sensitivity[channel]
            native_dim = str(sensitivity.dims[0])
            if native.ndim != 2 or native.dims[0] != native_dim:
                raise ValueError(
                    f"{channel} boundary sensitivity must have its native observation axis first."
                )
            native, _ = xr.align(native, sensitivity, join="exact", copy=False)
            state_dim = str(native.dims[1])
            rename = {
                name: f"{channel}_{name}"
                for name, coordinate in native.coords.items()
                if state_dim in coordinate.dims
            }
            if isinstance(native.indexes[state_dim], pd.MultiIndex):
                rename.update({name: f"{channel}_{name}" for name in native.indexes[state_dim].names})
            native = native.rename(rename)
            activity = (bc_state_activity or {}).get(channel)
            if activity is not None:
                activity = replace(
                    activity,
                    **{
                        name: value.rename(
                            {
                                key: val
                                for key, val in rename.items()
                                if key in value.dims or key in value.coords
                            }
                        )
                        for name in ("active", "fixed_value")
                        if isinstance(value := getattr(activity, name), xr.DataArray)
                    },
                    group_coord=rename.get(activity.group_coord, activity.group_coord),
                )
            prior = dict((bc_prior or {}).get(channel, DEFAULT_BC_PRIOR))
            prior = {
                name: value.rename(
                    {key: val for key, val in rename.items() if key in value.dims or key in value.coords}
                )
                if isinstance(value, xr.DataArray)
                else value
                for name, value in prior.items()
            }
            design = _pad_channel_design(native, observations, channel)
            result = add_linear_component(
                prepare_linear_sensitivity(design, output_dim=output_dim),
                data_name=f"{channel}_hbc",
                prior_args=prior,
                var_name=f"{channel}_bc",
                output_name=f"{channel}_mu_bc",
                output_dim=output_dim,
                state_activity=activity,
            )
            components.append((channel, result, design))
            boundaries.append(result.output)
        if channel in (offset_prior or {}):
            native_dim = str(sensitivity.dims[0])
            selected = observations.sel(species=channel).rename({"channel_observation": native_dim})
            selected = selected.drop_vars(
                [name for name in selected.coords if name not in (native_dim, "site", "time")]
            )
            # Native rows retain site/time metadata from channel preparation.
            add_coords({native_dim: selected[native_dim]})
            frequency, drop_first, per_site = _normalise_offset_args((offset_args or {}).get(channel))
            result = _add_offset_component_result(
                selected,
                prior_args=dict(offset_prior[channel]),
                offset_freq=frequency,
                var_name=f"{channel}_offset_latent",
                output_name=f"{channel}_offset_native",
                output_dim=native_dim,
                drop_first=drop_first,
                per_site=per_site,
                namespace=f"{channel}_",
            )
            design = _pad_channel_design(result.design, observations, channel)
            design_data = add_model_data(design, f"{channel}_offset_design")
            result = replace(
                result,
                design=design,
                output=pm.Deterministic(
                    f"{channel}_offset",
                    pt.dot(design_data, result.coefficients),
                    dims=output_dim,
                ),
            )
            components.append((channel, result, design))
            offsets.append(result.output)
    if boundaries:
        pm.Deterministic("boundary_concentration", sum(boundaries), dims=output_dim)
    if offsets:
        pm.Deterministic("offset_concentration", sum(offsets), dims=output_dim)
    pm.Deterministic("baseline_concentration", sum(boundaries + offsets), dims=output_dim)
    return components


def build_co2_o2_model(
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    aggregation_error: AggregationError,
    retained_prior: CorrelatedLognormalPrior,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    output_dim: str = "observation",
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> pm.Model:
    """Build the shared-state CO2/O2 affine model and fixed-error likelihood.

    ``fixed_prior_contribution``, both channel sensitivities, the retained prior,
    and every block of ``aggregation_error`` must be products of the same
    coherent reduction. CO2 and O2 retain separate sensitivity row axes because
    their numeric units are declared independently for every row. The
    Verification Games replay uses ppm for both channels; other scientific O2
    products may, for example, use per meg. Sensitivity rows must already be
    expressed in their declared channel units per dimensionless flux scaling;
    covariance blocks use the corresponding row-by-column unit products,
    including the CO2/O2 cross-covariance.

    The O2 sensitivity unconditionally embeds the fixed, signed O2-per-CO2
    oxidation ratios declared by CO2/O2 preparation. The two channel sensitivities
    are therefore gathered and applied to the raw shared state once. A recipe
    starting from a ratio-free O2 sensitivity would need to apply its labelled
    species/state ratio factor before constructing that joint sensitivity.

    The prepared-input runner validates the external ``independent_error_sd``
    as finite and positive. Direct custom callers own that check, exact label
    alignment, row-unit consistency, coherent covariance provenance, and the
    embedded-ratio contract before calling this lower-level builder.

    Args:
        observations: Joint CO2-then-O2 observation vector on ``output_dim``,
            with row-wise species and unit coordinates.
        fixed_prior_contribution: Joint coherent affine intercept on the same
            observation axis and in the corresponding row units.
        co2_sensitivity: CO2 sensitivity with its channel observation axis first and
            the retained-prior state axis second.
        o2_sensitivity: O2 sensitivity with its distinct channel observation axis
            first and the same state axis; its shared-state columns already
            embed the fixed oxidation ratios.
        aggregation_error: Joint fixed aggregation covariance, including its
            cross-channel block.
        retained_prior: Correlated arithmetic-moment prior for the one state
            vector shared by both channels.
        independent_error_sd: Joint per-row independent standard deviations in
            each observation row's native units.
        state_activity: Optional labelled policy fixing or activating retained
            states.
        boundary_sensitivity: Optional co2/o2 mapping of H_bc on each
            native observation axis and one labelled boundary-state axis.
        bc_prior: Channel-keyed independent boundary-scale priors; the CO2
            default is used for each enabled channel when omitted.
        bc_state_activity: Channel-keyed labelled fixed/active boundary policies.
        offset_prior: Channel-keyed offset priors. Omitted channels have no offset.
        offset_args: Per-channel offset_freq, drop_first, and per_site options,
            with the same meanings as the CO2 offset component. Baseline terms
            require identical channel units and are zero on the other channel.
        output_dim: Joint observation dimension used by the likelihood.
        tau_hours: Optional fixed OU timescale in hours, scalar or mapping keyed
            by ``co2:SITE``/``o2:SITE``. Requires common channel units.
        fixed_site_amplitudes: Fixed additive OU standard deviations in the
            common concentration units; scalar or species/site mapping.
        site_amplitude_prior: Prior for inferred species/site OU amplitudes in
            concentration units, mutually exclusive with fixed amplitudes.

    Returns:
        A registered PyMC model containing the shared state, gathered joint
        linear signal, coherent affine intercept, and joint Gaussian likelihood.

    Raises:
        ValueError: If shared-state sensitivity preparation, activity
            resolution, or registered coordinate alignment fails.
    """
    joint_sensitivity = _gather_co2_o2_sensitivity(
        co2_sensitivity,
        o2_sensitivity,
        output_dim=output_dim,
    )
    joint_sensitivity = joint_sensitivity.sel({output_dim: observations[output_dim]})
    prepared_sensitivity = prepare_linear_sensitivity(
        joint_sensitivity,
        output_dim=output_dim,
    )
    activity = resolve_state_activity(prepared_sensitivity.removed, state_activity)

    with registered_model() as model:
        state = add_correlated_lognormal_state_with_activity(
            activity,
            retained_prior,
            var_name="flux_scaling",
        ).state

        # Preparation declares that this sensitivity already contains the fixed,
        # signed O2:CO2 oxidation ratios; do not multiply them again.
        joint_signal = apply_linear_sensitivity(
            prepared_sensitivity,
            state,
            data_name="co2_o2_sensitivity",
            output_name="co2_o2_flux_contribution",
        )
        baseline_components = _add_co2_o2_baseline_components(
            observations=observations,
            co2_sensitivity=co2_sensitivity,
            o2_sensitivity=o2_sensitivity,
            boundary_sensitivity=boundary_sensitivity,
            bc_prior=bc_prior,
            bc_state_activity=bc_state_activity,
            offset_prior=offset_prior,
            offset_args=offset_args,
        )
        mean_signal = joint_signal + sum(result.output for _, result, _ in baseline_components)
        modelled = add_coherent_affine_component(
            fixed_prior_contribution,
            mean_signal,
            output_name="modelled_concentration",
        )

        if tau_hours is None:
            if fixed_site_amplitudes is not None or site_amplitude_prior is not None:
                raise ValueError("Fixed-OU amplitudes require tau_hours.")
            add_additive_sigma_likelihood(
                observations=observations,
                observation_error=independent_error_sd,
                mean=modelled,
                aggregation_error=aggregation_error,
                output_dim=output_dim,
                observation_error_name="fixed_independent_error_sd",
            )
        else:
            add_fixed_ou_gaussian_likelihood(
                observations=observations,
                observation_error=independent_error_sd,
                mean=modelled,
                aggregation_error=aggregation_error,
                output_dim=output_dim,
                tau_hours=tau_hours,
                fixed_site_amplitudes=fixed_site_amplitudes,
                site_amplitude_prior=site_amplitude_prior,
                sigma_alignment=linked_fixed_ou_alignment(observations),
            )
    return model
