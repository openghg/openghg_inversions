"""Channel-native frozen baselines sharing one retained flux prior."""

from collections.abc import Mapping

import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from .shared_baseline import prepare_shared_baseline


def prepare_linked_shared_baseline(
    *,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: Mapping[str, xr.DataArray],
    boundary_correction: Mapping[str, xr.Dataset],
    outer: xr.DataArray,
    sampled_outer: xr.DataArray,
    mode: str,
    delta_sd: Mapping[str, float],
    calibrated_prior: BiospherePrior | None = None,
) -> dict[str, xr.Dataset]:
    """Freeze independent tracer baselines against the same joint state.

    Sensitivities already contain the campaign's signed oxidative ratios.
    Selectors use the full joint latent axis, including tracer-specific oceans.
    Each channel's sensitivity, fixed affine contribution, boundary operands,
    scalar reference and explicit physical delta SD must use the same declared
    concentration units. Conversion belongs to the caller before preparation.
    All modes fix outer ocean; sampled_outer must exclude ocean states.

    Each returned dataset retains its native observation axis, full-support
    weights and column means. Save with the MultiIndex codec and subset rows
    directly; rebuilding must never repeat centring or prior calibration.
    Collapsed net additive biosphere corrections cannot preserve distinct
    GPP/TER oxidative ratios and are deliberately rejected.
    """
    if mode == "additive":
        raise ValueError("Linked additive net mode requires an explicit GPP/TER oxidative allocation.")
    channels = {"co2", "o2"}
    if any(set(value) != channels for value in (fixed_prior_contribution, boundary_correction, delta_sd)):
        raise ValueError("Supply co2 and o2 fixed contributions, boundary corrections and physical delta SDs.")
    if "source" not in retained_prior.mean.coords:
        raise ValueError("Linked outer selection requires labelled source coordinates.")
    ocean = retained_prior.mean.source.str.lower().isin(["ocean", "co2_ocean", "o2_ocean"])
    if bool((ocean & sampled_outer).any()):
        raise ValueError("Outer ocean must remain fixed.")
    result = {}
    for channel, h in (("co2", co2_sensitivity), ("o2", o2_sensitivity)):
        row = str(h.dims[0])
        rename = {row: "nmeasure"} if row != "nmeasure" else {}
        prepared = prepare_shared_baseline(
            h.rename(rename), retained_prior,
            fixed_prior_contribution[channel].rename(rename),
            boundary_correction[channel].rename(rename),
            outer=outer, sampled_outer=sampled_outer, mode=mode,
            delta_sd=delta_sd[channel], calibrated_prior=calibrated_prior,
        )
        prepared.attrs.update(linked_channel=channel, native_observation_dim=row)
        result[channel] = prepared.rename({"nmeasure": row}) if row != "nmeasure" else prepared
    return result
