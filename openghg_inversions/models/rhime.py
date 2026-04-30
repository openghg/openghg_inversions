"""RHIME model builders.

These builders are the modern public model-construction names.  They reuse the
component-based PyMC helpers, while keeping the legacy ``inferpymc`` adapter out
of the RHIME runtime path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.models.components import (
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.priors import PriorArgs

DEFAULT_X_PRIOR: PriorArgs = {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0, "reparameterise": True}
DEFAULT_BC_PRIOR: PriorArgs = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.05, "lower": 0.0}
DEFAULT_SIGMA_PRIOR: PriorArgs = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}


def safe_pymc_name(value: str) -> str:
    """Return a stable PyMC-safe suffix for a user-facing sector/source name.

    Args:
        value: User-facing sector or source name.

    Returns:
        Lowercase snake-case suffix safe to use in PyMC variable names.
    """
    name = re.sub(r"\W+", "_", str(value).strip().lower()).strip("_")
    return name or "sector"


def _prepare_builder_priors(
    *,
    x_prior: dict | None,
    bc_prior: dict | None,
    sigma_prior: dict | None,
    offset_prior: dict | None,
) -> tuple[dict, dict, dict, dict]:
    """Copy builder priors, applying RHIME model defaults when omitted."""
    prepared_x_prior = DEFAULT_X_PRIOR.copy() if x_prior is None else x_prior.copy()
    prepared_bc_prior = DEFAULT_BC_PRIOR.copy() if bc_prior is None else bc_prior.copy()
    prepared_sigma_prior = DEFAULT_SIGMA_PRIOR.copy() if sigma_prior is None else sigma_prior.copy()
    prepared_offset_prior = DEFAULT_OFFSET_PRIOR.copy() if offset_prior is None else offset_prior.copy()
    return prepared_x_prior, prepared_bc_prior, prepared_sigma_prior, prepared_offset_prior


def build_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
    sigma_per_site: bool = True,
    offset_prior: dict | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
) -> pm.Model:
    """Build the standard single-sector RHIME model.

    Args:
        inv_inputs: Canonical inversion-input dataset produced by
            ``make_inv_inputs``.
        x_prior: Prior specification for flux scaling factors.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        sigma_per_site: Whether model-error terms vary by site.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        pollution_events_from_obs: Whether to derive pollution-event scaling
            from observations rather than modelled concentrations.
        no_model_error: Whether to suppress the explicit model-error term.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.

    Returns:
        Built PyMC model.
    """
    x_prior, bc_prior, sigma_prior, offset_prior = _prepare_builder_priors(
        x_prior=x_prior,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        flux_component = add_linear_component(
            inv_inputs["H"],
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
        )

        mu_bc = None
        if use_bc:
            if "H_bc" not in inv_inputs:
                raise ValueError("If `use_bc` is True, `inv_inputs` must contain `H_bc`.")
            bc_component = add_linear_component(
                inv_inputs["H_bc"],
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
            )
            mu_bc = bc_component.output

        offset = None
        if add_offset:
            offset_args = offset_args or {}
            offset = add_offset_component(
                inv_inputs["site_indicator"],
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **offset_args,
            )

        add_inferpymc_likelihood_component(
            inv_inputs,
            mu=flux_component.output,
            mu_bc=mu_bc,
            offset=offset,
            sigprior=sigma_prior,
            power=power,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            sigma_per_site=sigma_per_site,
            output_dim="nmeasure",
        )

    return model


def _resolve_sectors(inv_inputs: xr.Dataset, sectors: Sequence[str] | None) -> list[str]:
    """Resolve requested sector names against the source coordinate."""
    if "source" not in inv_inputs["H"].dims:
        raise ValueError("Multi-sector RHIME requires inv_inputs['H'] to include a 'source' dimension.")

    available = [str(value) for value in inv_inputs["H"].coords["source"].values]
    if sectors is None:
        sectors = available
    sectors = [str(sector) for sector in sectors]

    missing = [sector for sector in sectors if sector not in available]
    if missing:
        raise ValueError(f"Sector(s) {missing!r} are not present in inv_inputs['H'].source.")
    if len(sectors) < 2:
        raise ValueError("Multi-sector RHIME requires at least two sectors.")

    return sectors


def _sector_prior(
    sector: str,
    *,
    sector_priors: Mapping[str, dict] | None,
    x_prior: dict | None,
) -> dict:
    """Resolve the prior for a sector, falling back to the shared x prior."""
    if sector_priors is not None and sector in sector_priors:
        return dict(sector_priors[sector])
    return dict(DEFAULT_X_PRIOR if x_prior is None else x_prior)


def build_rhime_multisector_model(
    inv_inputs: xr.Dataset,
    *,
    sectors: Sequence[str] | None = None,
    sector_priors: Mapping[str, dict] | None = None,
    x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
    sigma_per_site: bool = True,
    offset_prior: dict | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
) -> pm.Model:
    """Build the first shared-basis multi-sector RHIME model.

    Each sector receives its own state vector ``x_<sector>`` and forward-model
    contribution ``mu_<sector>``. The total ``mu`` is the sum of sector
    contributions and is passed to the standard RHIME likelihood.

    Args:
        inv_inputs: Canonical inversion-input dataset with
            ``H(region, nmeasure, source)``.
        sectors: Ordered sector/source names to optimise. Defaults to all
            ``inv_inputs.H.source`` values.
        sector_priors: Optional per-sector flux-scaling priors.
        x_prior: Shared fallback flux-scaling prior.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        sigma_per_site: Whether model-error terms vary by site.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        pollution_events_from_obs: Whether to derive pollution-event scaling
            from observations rather than modelled concentrations.
        no_model_error: Whether to suppress explicit model-error terms.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.

    Returns:
        Built PyMC model.
    """
    sectors = _resolve_sectors(inv_inputs, sectors)
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())

        sector_outputs = []
        used_names: set[str] = set()
        for sector in sectors:
            suffix = safe_pymc_name(sector)
            if suffix in used_names:
                raise ValueError(
                    "Sector names must be unique after PyMC name sanitisation; "
                    f"duplicate sanitized name {suffix!r}."
                )
            used_names.add(suffix)

            h_sector = inv_inputs["H"].sel(source=sector).drop_vars("source", errors="ignore")
            component = add_linear_component(
                h_sector,
                data_name=f"hx_{suffix}",
                prior_args=_sector_prior(sector, sector_priors=sector_priors, x_prior=x_prior),
                var_name=f"x_{suffix}",
                output_name=f"mu_{suffix}",
                output_dim="nmeasure",
                compute_deterministic=True,
            )
            sector_outputs.append(component.output)

        total_mu = pm.Deterministic("mu", pt.stack(sector_outputs, axis=0).sum(axis=0), dims="nmeasure")

        mu_bc = None
        if use_bc:
            if "H_bc" not in inv_inputs:
                raise ValueError("If `use_bc` is True, `inv_inputs` must contain `H_bc`.")
            bc_component = add_linear_component(
                inv_inputs["H_bc"],
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
            )
            mu_bc = bc_component.output

        offset = None
        if add_offset:
            offset_args = offset_args or {}
            offset = add_offset_component(
                inv_inputs["site_indicator"],
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **offset_args,
            )

        add_inferpymc_likelihood_component(
            inv_inputs,
            mu=total_mu,
            mu_bc=mu_bc,
            offset=offset,
            sigprior=sigma_prior,
            power=power,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            sigma_per_site=sigma_per_site,
            output_dim="nmeasure",
        )

    return model
