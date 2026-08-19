"""Shared RHIME model specifications and observation-component assembly.

Concrete standard and multisector PyMC graphs live beside their readable
recipes. This module retains only the scientific options and small helpers
shared by those recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models._rhime_flux import safe_pymc_name as _safe_pymc_name
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
    add_state_linear_component,
)
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.rhime_likelihood import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    build_gaussian_rhime_likelihood,
)
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import AggregationErrorMode
from openghg_inversions.sigma import SigmaAlignment

DEFAULT_X_PRIOR: PriorArgs = {
    "pdf": "lognormal",
    "mean": 1.0,
    "stdev": 1.0,
    "reparameterise": True,
}
DEFAULT_BC_PRIOR: PriorArgs = {
    "pdf": "truncatednormal",
    "mu": 1.0,
    "sigma": 0.05,
    "lower": 0.0,
}
DEFAULT_SIGMA_PRIOR: PriorArgs = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}

_LIKELIHOOD_RESULT_ATTR = "_openghg_rhime_likelihood_result"


def get_rhime_likelihood_result(model: pm.Model) -> RhimeLikelihoodResult:
    """Return the explicit likelihood roles attached by a RHIME model builder."""
    try:
        return cast(RhimeLikelihoodResult, getattr(model, _LIKELIHOOD_RESULT_ATTR))
    except AttributeError as exc:
        raise ValueError(
            "The PyMC model has no RHIME likelihood result. Build it with a public RHIME model "
            "builder or return explicit roles from a complete `RhimeModelBuilder`."
        ) from exc


def safe_pymc_name(value: str) -> str:
    """Return a stable PyMC-safe suffix for a user-facing sector/source name."""
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
        state_activity: Optional labelled active/fixed policy for this sector's
            flux-scaling states. ``None`` still applies the default flux policy,
            fixing exactly-zero sensitivity columns to one.
    """

    name: str
    flux_source: str
    x_prior: dict[str, Any]
    variable_suffix: str
    state_activity: StateActivity | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class RhimeModelSpec:
    """Scientific options used by the concrete RHIME model recipes.

    ``sectors`` records the separately optimized flux components and their
    OpenGHG sources. The remaining values select the baseline, offset,
    model-data mismatch, aggregation-error, and likelihood behavior shared by
    the standard and multisector recipes.
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
    aggregation_error_mode: AggregationErrorMode = field(default="auto", kw_only=True)
    power: dict[str, Any] | float = 1.99
    bc_prior: dict[str, Any] | None = None
    sigma_prior: dict[str, Any] | None = None
    offset_prior: dict[str, Any] | None = None
    offset_args: dict[str, Any] | None = None
    bc_state_activity: StateActivity | None = field(default=None, kw_only=True)
    state_activity: StateActivity | None = field(default=None, kw_only=True)
    sector_state_activities: dict[str, StateActivity] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate model options resolved before graph construction."""
        if self.aggregation_error_mode not in ("auto", "none", "dense", "low_rank", "diagonal"):
            raise ValueError(
                "`aggregation_error_mode` must be one of 'auto', 'none', 'dense', "
                f"'low_rank', or 'diagonal'; got {self.aggregation_error_mode!r}."
            )


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


def _add_rhime_observation_components(
    inv_inputs: xr.Dataset,
    *,
    mu: TensorVariable,
    sigma_alignment: SigmaAlignment,
    bc_prior: dict,
    sigma_prior: dict,
    offset_prior: dict,
    add_offset: bool,
    use_bc: bool,
    bc_state_activity: StateActivity | None,
    pollution_events_from_obs: bool,
    no_model_error: bool,
    aggregation_error_mode: AggregationErrorMode,
    offset_args: dict | None,
    power: dict | float,
    likelihood_builder: RhimeLikelihoodBuilder | None,
) -> RhimeLikelihoodResult:
    """Add the shared baseline, offset, error, and likelihood components."""
    mu_bc = None
    if use_bc:
        if "H_bc" not in inv_inputs:
            raise ValueError("If `use_bc` is True, `inv_inputs` must contain `H_bc`.")
        if bc_state_activity is None:
            bc_component = add_linear_component(
                inv_inputs["H_bc"],
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
            )
        else:
            bc_component = add_state_linear_component(
                inv_inputs["H_bc"],
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
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

    context = RhimeLikelihoodContext(
        data=inv_inputs,
        flux_mean=mu,
        boundary_mean=mu_bc,
        offset=offset,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        power=power,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        aggregation_error_mode=aggregation_error_mode,
        output_dim="nmeasure",
    )
    result = build_gaussian_rhime_likelihood(context) if likelihood_builder is None else likelihood_builder(context)
    if not isinstance(result, RhimeLikelihoodResult):
        raise TypeError(
            "A RHIME likelihood builder must return `RhimeLikelihoodResult`; "
            f"got {type(result).__name__}."
        )
    missing_names = sorted(set(result.variable_roles.values()) - set(pm.modelcontext(None).named_vars))
    if missing_names:
        raise ValueError(
            "RHIME likelihood roles refer to variables absent from the active PyMC model: "
            f"{missing_names!r}."
        )
    return result
