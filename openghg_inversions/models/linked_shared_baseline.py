"""Channel-native shared baselines bound against one retained flux prior."""

from collections.abc import Mapping
from dataclasses import replace

import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import RawCenteredSensitivity

from .shared_baseline import SharedBaseline, SharedBaselineChoice


def bind_linked_shared_baseline(
    choices: Mapping[str, SharedBaselineChoice],
    observations: Mapping[str, xr.DataArray],
    *,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: Mapping[str, xr.DataArray],
    boundary_sensitivity: Mapping[str, RawCenteredSensitivity],
    outer: xr.DataArray,
    sampled_outer: xr.DataArray,
    calibrated_prior: BiospherePrior | None = None,
) -> dict[str, SharedBaseline]:
    """Bind each tracer's baseline against the same joint state.

    Sensitivities already contain the source-specific signed oxidative ratios.
    Selectors use the full joint latent axis, including tracer-specific oceans.
    Each channel converts its compatible sensitivity and fixed affine
    contribution into its boundary reference's concentration units, and its
    atmospheric width uses those units. Both channels must declare one common
    concentration unit when constructing a linked model. All modes fix outer
    ocean; ``sampled_outer`` must exclude ocean states.

    Each returned baseline keeps its native observation axis. Collapsed net
    additive biosphere corrections cannot preserve distinct GPP/TER oxidative
    ratios and are deliberately rejected.

    Args:
        choices: Channel-keyed choices sharing one outer mode.
        observations: Channel-keyed native rows with ``site`` and ``time``.
        co2_sensitivity: Native CO2 observation-by-joint-state operator.
        o2_sensitivity: Native signed O2 operator on the same joint state axis.
        retained_prior: Complete labelled positive joint physical prior.
        fixed_prior_contribution: Channel-keyed original affine row contributions.
        boundary_sensitivity: Channel-keyed prepared centered boundary operands
            on their raw ``nmeasure`` or ``time`` axis.
        outer: Boolean selector on the complete joint state axis.
        sampled_outer: Adjustable outer subset, excluding all outer ocean states.
        calibrated_prior: Optional resolved positive outer calibration.

    Returns:
        Channel-keyed bound baselines whose data stay lazy.
    """
    channels = {"co2", "o2"}
    if any(set(value) != channels for value in (choices, observations, fixed_prior_contribution, boundary_sensitivity)):
        raise ValueError("Supply co2 and o2 choices, observations, fixed contributions and boundary transport.")
    if len({choice.mode for choice in choices.values()}) != 1:
        raise ValueError("Both channels must use the same outer mode.")
    if choices["co2"].mode == "additive":
        raise ValueError("Linked additive net mode requires an explicit GPP/TER oxidative allocation.")
    if "source" not in retained_prior.mean.coords:
        raise ValueError("Linked outer selection requires labelled source coordinates.")
    ocean = retained_prior.mean.source.str.lower().isin(["ocean"])
    if bool((ocean & sampled_outer).any()):
        raise ValueError("Outer ocean must remain fixed.")
    result = {}
    for channel, h in (("co2", co2_sensitivity), ("o2", o2_sensitivity)):
        row = str(h.dims[0])
        rename = {row: "nmeasure"} if row != "nmeasure" else {}
        bound = choices[channel].bind(
            observations[channel].rename(rename),
            flux_sensitivity=h.rename(rename), retained_prior=retained_prior,
            fixed_prior_contribution=fixed_prior_contribution[channel].rename(rename),
            boundary_sensitivity=boundary_sensitivity[channel],
            outer=outer, sampled_outer=sampled_outer, calibrated_prior=calibrated_prior,
        )
        data = bound.data.rename({"nmeasure": row}) if rename else bound.data
        result[channel] = replace(bound, data=data.assign_attrs(linked_channel=channel, native_observation_dim=row))
    return result


__all__ = ["bind_linked_shared_baseline"]
