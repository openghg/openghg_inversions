"""Report the boundary and outer-region flux contributions together as one atmospheric baseline.

With the outer-flux option (:mod:`openghg_inversions.models.outer_flux`) the
boundary mean shift absorbs the level of the sampled outer fluxes, so neither
``mu_bc`` nor the outer flux contribution alone is the atmospheric baseline
that observations see. This module combines them from posterior draws without
rebuilding the graph::

    atmospheric_baseline = f + mu_bc + H_O x_O - mu_S @ (x_S - m_S)
                         = modelled_concentration - H_I x_I

for the raw ``H``, fixed term ``f``, outer states ``O = S + F`` and inner
states ``I``. Offsets, when used, are not part of the baseline.
"""

from __future__ import annotations

import xarray as xr


def reconstruct_atmospheric_baseline(
    outer_flux: xr.Dataset,
    *,
    flux_sensitivity: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    prior_mean: xr.DataArray,
    posterior: xr.Dataset,
    prefix: str = "",
) -> xr.Dataset:
    """Split posterior draws into outer-flux and atmospheric-baseline terms.

    Args:
        outer_flux: The bound option's ``data`` (``OuterFlux.data``), or the
            same record read back from a trace.
        flux_sensitivity: The raw prepared ``H`` the model was built from, on
            the bound rows and the full flux state axis.
        fixed_prior_contribution: The raw fixed term ``f``, before the outer
            level was restored.
        prior_mean: The flux prior's baseline mean ``m``.
        posterior: Draws with ``flux_scaling`` on the flux state axis and
            ``mu_bc`` on the rows, plus ``outer_pattern_coefficient`` in
            additive mode, with restored coordinates.
        prefix: Graph-name prefix of a linked channel's boundary and pattern
            variables.

    Returns:
        A lazy dataset with ``outer_removed_level`` (the level the boundary
        mean shift absorbs), ``outer_centred_correction``,
        ``outer_flux_contribution`` (``H_O x_O``) and
        ``atmospheric_baseline``; in additive mode also
        ``outer_pattern_contribution`` and ``outer_density_correction``.
    """
    state = str(outer_flux.outer_column_mean.dims[0])
    outer = outer_flux.sampled_outer | outer_flux.fixed_outer
    flux_scaling = posterior.flux_scaling
    departure = flux_scaling - prior_mean

    # The model samples H'_S x_S; the bracketed level went to the boundary mean shift.
    removed_level = xr.dot(outer_flux.outer_column_mean, departure, dim=state)
    sampled_response = xr.dot(flux_sensitivity.where(outer_flux.sampled_outer, 0), departure, dim=state)
    outer_contribution = xr.dot(flux_sensitivity.where(outer, 0), flux_scaling, dim=state)
    baseline = fixed_prior_contribution + posterior[f"{prefix}mu_bc"] + outer_contribution - removed_level

    result = xr.Dataset({
        "outer_removed_level": removed_level,
        "outer_centred_correction": sampled_response - removed_level,
        "outer_flux_contribution": outer_contribution,
    })
    if "additive_centered" in outer_flux:
        coefficient = posterior[f"{prefix}outer_pattern_coefficient"]
        patterns = xr.dot(outer_flux.additive_centered, coefficient, dim="outer_pattern")
        result["outer_pattern_contribution"] = patterns
        result["outer_density_correction"] = (coefficient * outer_flux.additive_density_sd).assign_attrs(
            units=outer_flux.additive_density_sd.attrs["units"],
        )
        baseline = baseline + patterns
    result["atmospheric_baseline"] = baseline

    units = outer_flux.attrs["units"]
    for name in ("outer_removed_level", "outer_centred_correction", "outer_flux_contribution", "atmospheric_baseline"):
        result[name].attrs["units"] = units
    if "outer_pattern_contribution" in result:
        result["outer_pattern_contribution"].attrs["units"] = units
    return result


__all__ = ["reconstruct_atmospheric_baseline"]
