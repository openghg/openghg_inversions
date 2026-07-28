"""RHIME model builders.

These builders are the modern public model-construction names. They reuse the
component-based PyMC helpers, while keeping the legacy ``inferpymc`` adapter out
of the RHIME runtime path.

The standard builder optimizes one flux scaling component. The multi-sector
builder optimizes one component per sector, where each sector is normally backed
by one OpenGHG flux ``source`` coordinate in ``inv_inputs["H"]``. When sector
labels differ from OpenGHG source values, the builder selects data by source
and names PyMC variables by sector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pymc as pm
import xarray as xr

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models._rhime_compiler import _compile_loop_sum, _FluxPlan
from openghg_inversions.models._rhime_flux import (
    _normalize_multisector_flux_plan,
    _normalize_standard_flux_plan,
    _resolve_sector_bindings,
)
from openghg_inversions.models._rhime_flux import (
    safe_pymc_name as _safe_pymc_name,
)
from openghg_inversions.models.components import (
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.sigma import SigmaAlignment

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
    return _safe_pymc_name(value)


@dataclass(frozen=True)
class SectorSpec:
    """Configuration for one separately optimised flux sector.

    Args:
        name: User-facing sector name.
        flux_source: OpenGHG flux ``source`` used to retrieve this sector.
        x_prior: Prior specification for this sector's flux scaling factors.
        variable_suffix: PyMC-safe suffix used in multi-sector model variable
            names. Standard single-sector RHIME uses plain ``x``/``mu`` names.
    """

    name: str
    flux_source: str
    x_prior: dict[str, Any]
    variable_suffix: str


@dataclass(frozen=True)
class RhimeModelSpec:
    """Model options used to build a RHIME PyMC model.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        domain: Model domain name.
        sectors: Flux sectors included in the model. Each sector is optimized
            separately and is normally backed by one OpenGHG flux ``source``.
        use_bc: Whether boundary-condition scaling is included.
        sigma_per_site: Whether model-error terms vary by site.
        sigma_freq: Frequency used to derive observation-aligned sigma periods.
            ``None`` uses one shared period.
        sigma_freq_anchor: Optional anchor for fixed-duration sigma periods.
        add_offset: Whether model-data offsets are included.
        pollution_events_from_obs: Whether model error scales with observed
            enhancements instead of modelled enhancements.
        no_model_error: Whether explicit model-error terms are disabled.
        power: Exponent or prior specification used in likelihood error scaling.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        offset_args: Extra keyword arguments forwarded to the offset component.
    """

    species: str
    domain: str
    sectors: tuple[SectorSpec, ...]
    use_bc: bool = True
    sigma_per_site: bool = True
    sigma_freq: str | None = None
    sigma_freq_anchor: DatetimeLike | None = None
    add_offset: bool = False
    pollution_events_from_obs: bool = False
    no_model_error: bool = False
    power: dict[str, Any] | float = 1.99
    bc_prior: dict[str, Any] | None = None
    sigma_prior: dict[str, Any] | None = None
    offset_prior: dict[str, Any] | None = None
    offset_args: dict[str, Any] | None = None


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


def _assemble_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    flux_plan: _FluxPlan,
    sigma_alignment: SigmaAlignment,
    bc_prior: dict,
    sigma_prior: dict,
    offset_prior: dict,
    add_offset: bool,
    use_bc: bool,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    offset_args: dict | None,
    power: dict | float,
) -> pm.Model:
    """Assemble shared RHIME components around a normalized flux plan.

    Args:
        inv_inputs: Canonical inversion inputs.
        flux_plan: Validated linear flux plan to compile.
        sigma_alignment: Prepared observation alignment for model error.
        bc_prior: Prepared boundary-condition prior.
        sigma_prior: Prepared model-error prior.
        offset_prior: Prepared optional offset prior.
        add_offset: Whether to add an offset component.
        use_bc: Whether to add a boundary-condition component.
        pollution_events_from_obs: Whether error scaling uses observations.
        no_model_error: Whether to suppress explicit model error.
        offset_args: Extra offset-component arguments.
        power: Likelihood error-scaling exponent or prior.

    Returns:
        Fully assembled PyMC model.
    """
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        compiled_flux = _compile_loop_sum(flux_plan)

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
            offset = add_offset_component(
                inv_inputs["site_indicator"],
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )

        add_inferpymc_likelihood_component(
            inv_inputs,
            mu=compiled_flux.mu,
            mu_bc=mu_bc,
            offset=offset,
            sigprior=sigma_prior,
            sigma_alignment=sigma_alignment,
            power=power,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            output_dim="nmeasure",
        )

    return model


def build_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    sigma_alignment: SigmaAlignment,
    x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
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
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        x_prior: Prior specification for flux scaling factors.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
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
    flux_plan = _normalize_standard_flux_plan(inv_inputs, x_prior)
    return _assemble_rhime_model(
        inv_inputs,
        flux_plan=flux_plan,
        sigma_alignment=sigma_alignment,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        add_offset=add_offset,
        use_bc=use_bc,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        offset_args=offset_args,
        power=power,
    )


def build_rhime_model_from_spec(inv_inputs: xr.Dataset, model_spec: RhimeModelSpec) -> pm.Model:
    """Build the standard single-sector RHIME model from a model spec.

    Args:
        inv_inputs: Canonical inversion-input dataset produced by
            ``make_inv_inputs``.
        model_spec: Normalized RHIME model specification.

    Returns:
        Built PyMC model.

    Raises:
        ValueError: If the model spec does not describe exactly one sector.
    """
    if len(model_spec.sectors) != 1:
        raise ValueError("Standard RHIME model specs must include exactly one sector.")

    sector = model_spec.sectors[0]
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    return build_rhime_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        x_prior=dict(sector.x_prior),
        bc_prior=model_spec.bc_prior,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
    )


def build_rhime_multisector_model(
    inv_inputs: xr.Dataset,
    *,
    sigma_alignment: SigmaAlignment,
    sectors: Sequence[str] | None = None,
    sector_sources: Mapping[str, str] | None = None,
    sector_variable_suffixes: Mapping[str, str] | None = None,
    sector_priors: Mapping[str, dict] | None = None,
    x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
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
        inv_inputs: Canonical inversion-input dataset. Shared-basis inputs use
            ``H(region, nmeasure, source)``. Source-specific bases use
            ``H(state, nmeasure)`` with ``state`` gathered over
            ``(source, region_in_source)``.
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        sectors: Ordered model sector labels to optimize. Defaults to
            ``sector_sources`` keys when supplied, otherwise all
            ``inv_inputs.H.source`` values, where each source becomes one
            separately optimized sector.
        sector_sources: Optional mapping from sector label to OpenGHG
            ``source`` value in ``inv_inputs.H``.
        sector_variable_suffixes: Optional mapping from sector label to
            PyMC-safe suffix used in ``x_<suffix>`` and ``mu_<suffix>`` names.
        sector_priors: Optional per-sector flux-scaling priors.
            When supplied, the mapping must contain exactly one entry for every
            sector.
        x_prior: Shared flux-scaling prior used when ``sector_priors`` is absent.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
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
    sector_bindings = _resolve_sector_bindings(
        inv_inputs,
        sectors,
        sector_sources=sector_sources,
        sector_variable_suffixes=sector_variable_suffixes,
    )
    flux_plan = _normalize_multisector_flux_plan(
        inv_inputs,
        sector_bindings,
        sector_priors=sector_priors,
        x_prior=x_prior,
        default_x_prior=DEFAULT_X_PRIOR,
    )
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)
    return _assemble_rhime_model(
        inv_inputs,
        flux_plan=flux_plan,
        sigma_alignment=sigma_alignment,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        add_offset=add_offset,
        use_bc=use_bc,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        offset_args=offset_args,
        power=power,
    )


def build_rhime_multisector_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
) -> pm.Model:
    """Build the shared-basis multi-sector RHIME model from a model spec.

    Args:
        inv_inputs: Canonical inversion-input dataset using either rectangular
            shared-basis or gathered source-specific sensitivity.
        model_spec: Normalized RHIME model specification.

    Returns:
        Built PyMC model.
    """
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    return build_rhime_multisector_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        sectors=[sector.name for sector in model_spec.sectors],
        sector_sources={sector.name: sector.flux_source for sector in model_spec.sectors},
        sector_variable_suffixes={sector.name: sector.variable_suffix for sector in model_spec.sectors},
        sector_priors={sector.name: dict(sector.x_prior) for sector in model_spec.sectors},
        bc_prior=model_spec.bc_prior,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
    )
