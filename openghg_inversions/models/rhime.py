"""RHIME model builders.

These builders are the modern public model-construction names. They reuse the
component-based PyMC helpers, while keeping the legacy ``inferpymc`` adapter out
of the RHIME runtime path. Public builders compose those components directly;
``RhimeModelSpec`` can opt into the private flux-plan compiler for development
and parity testing.

The standard builder optimizes one flux scaling component. The multi-sector
builder optimizes one component per sector, where each sector is normally backed
by one OpenGHG flux ``source`` coordinate in ``inv_inputs["H"]``. When sector
labels differ from OpenGHG source values, the builder selects data by source
and names PyMC variables by sector.

Modern flux builders fix exact-zero sensitivity columns to one by default and
restore them into the full public state vector. Boundary-condition activity is
opt-in: omitting its policy preserves the ordinary fully sampled BC graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models._rhime_compiler import _compile_loop_sum, _FluxPlan
from openghg_inversions.models._rhime_flux import (
    _normalize_multisector_flux_plan,
    _normalize_standard_flux_plan,
    _resolve_multisector_components,
    _resolve_sector_bindings,
)
from openghg_inversions.models._rhime_flux import (
    safe_pymc_name as _safe_pymc_name,
)
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
    add_state_linear_component,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
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

DEFAULT_X_PRIOR: PriorArgs = {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0, "reparameterise": True}
DEFAULT_BC_PRIOR: PriorArgs = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.05, "lower": 0.0}
DEFAULT_SIGMA_PRIOR: PriorArgs = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}
DEFAULT_OFFSET_PRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}

#: Public RHIME model-construction strategy.
#:
#: ``"concrete"`` selects the default, readable reference implementation.
#: ``"compiled"`` selects the opt-in extension and regression-checking path.
#: Compiler plan objects remain private, while these public strategy values and
#: the graph contract of unchanged model components are stable.
RhimeBuilderStrategy: TypeAlias = Literal["concrete", "compiled"]


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
        aggregation_error_mode: Fixed aggregation-error covariance
            representation. ``"auto"`` selects from prepared inputs.
        power: Exponent or prior specification used in likelihood error scaling.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        offset_args: Extra keyword arguments forwarded to the offset component.
        bc_state_activity: Optional active/fixed policy for the boundary-
            condition scaling vector. ``None`` preserves the ordinary fully
            sampled BC graph without zero pruning. Supplying a policy opts into
            active/fixed BC construction; when all states are fixed, ``mu_bc``
            remains without a boundary-condition RV.
        state_activity: Optional labelled active/fixed state policy shared by
            flux sectors. The default retains exact-zero pruning.
        sector_state_activities: Optional activity-policy overrides keyed by
            sector name for multi-sector models. A policy stored directly on a
            ``SectorSpec`` takes precedence over this compatibility mapping.
        builder_strategy: Public model-construction strategy. ``"concrete"``
            directly composes the default, readable reference model.
            ``"compiled"`` opts into the private semantic-plan compiler for
            extension work and regression checking. There is no automatic
            fallback between strategies.
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
    builder_strategy: RhimeBuilderStrategy = field(default="concrete", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the explicitly supported model-construction strategies."""
        if self.builder_strategy not in ("concrete", "compiled"):
            raise ValueError(
                f"`builder_strategy` must be either 'concrete' or 'compiled'; got {self.builder_strategy!r}."
            )
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
    """Add boundary, offset, error, and likelihood components to a RHIME model.

    Args:
        inv_inputs: Canonical inversion inputs.
        mu: Total flux contribution in observation space.
        sigma_alignment: Prepared observation alignment for model error.
        bc_prior: Prepared boundary-condition prior.
        sigma_prior: Prepared model-error prior.
        offset_prior: Prepared optional offset prior.
        add_offset: Whether to add an offset component.
        use_bc: Whether to add a boundary-condition component.
        bc_state_activity: Optional active/fixed policy for boundary-condition
            scaling states.
        pollution_events_from_obs: Whether error scaling uses observations.
        no_model_error: Whether to suppress explicit model error.
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra offset-component arguments.
        power: Likelihood error-scaling exponent or prior.
        likelihood_builder: Complete observation-component builder. ``None``
            uses the built-in Gaussian likelihood and RHIME error model.

    """
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


def _assemble_compiled_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    flux_plan: _FluxPlan,
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
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Compile a normalized flux plan and add the shared RHIME components.

    Args:
        inv_inputs: Canonical inversion inputs.
        flux_plan: Validated linear flux plan to compile.
        sigma_alignment: Prepared observation alignment for model error.
        bc_prior: Prepared boundary-condition prior.
        sigma_prior: Prepared model-error prior.
        offset_prior: Prepared optional offset prior.
        add_offset: Whether to add an offset component.
        use_bc: Whether to add a boundary-condition component.
        bc_state_activity: Optional active/fixed policy for boundary-condition
            scaling states.
        pollution_events_from_obs: Whether error scaling uses observations.
        no_model_error: Whether to suppress explicit model error.
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra offset-component arguments.
        power: Likelihood error-scaling exponent or prior.
        likelihood_builder: Optional complete observation-component builder.

    Returns:
        Fully assembled PyMC model.
    """
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        compiled_flux = _compile_loop_sum(flux_plan)
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=compiled_flux.mu,
            sigma_alignment=sigma_alignment,
            bc_prior=bc_prior,
            sigma_prior=sigma_prior,
            offset_prior=offset_prior,
            add_offset=add_offset,
            use_bc=use_bc,
            bc_state_activity=bc_state_activity,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            offset_args=offset_args,
            power=power,
            likelihood_builder=likelihood_builder,
        )
        setattr(model, _LIKELIHOOD_RESULT_ATTR, likelihood_result)

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
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
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
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.
        state_activity: Optional labelled active/fixed state policy. By
            default, only exactly-zero ``H`` columns are fixed to one; every
            nonzero column remains active.
        bc_state_activity: Optional active/fixed policy for ``H_bc`` scaling
            states. Omit it to preserve the standard fully sampled BC graph
            without automatic zero-column pruning.
        likelihood_builder: Optional complete observation-component builder.
            The callable receives labelled means, inputs, error policies, and
            priors and must return :class:`RhimeLikelihoodResult`.

    Returns:
        Built PyMC model.

    Raises:
        KeyError: If required sensitivity inputs are absent.
        ValueError: If the state layout, activity policy, labels, fixed values,
            or state-valued prior parameters are invalid.
    """
    x_prior, bc_prior, sigma_prior, offset_prior = _prepare_builder_priors(
        x_prior=x_prior,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        flux_component = add_state_linear_component(
            inv_inputs["H"],
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=state_activity,
        )
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=flux_component.output,
            sigma_alignment=sigma_alignment,
            bc_prior=bc_prior,
            sigma_prior=sigma_prior,
            offset_prior=offset_prior,
            add_offset=add_offset,
            use_bc=use_bc,
            bc_state_activity=bc_state_activity,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            offset_args=offset_args,
            power=power,
            likelihood_builder=likelihood_builder,
        )
        setattr(model, _LIKELIHOOD_RESULT_ATTR, likelihood_result)

    return model


def build_nested_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    sigma_alignment: SigmaAlignment,
    outer_x_prior: dict | None = None,
    inner_x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
    offset_prior: dict | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    outer_state_activity: StateActivity | None = None,
    inner_state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build a two-grid nested-domain RHIME model.

    ``H`` and ``H_inner`` retain independent labelled state dimensions.  The
    two forward-model contributions are constructed separately and summed in
    observation space, so the inner grid is never coerced onto the outer grid
    or represented as a same-grid emissions sector.

    Args:
        inv_inputs: Canonical inversion inputs containing ``H`` and
            ``H_inner`` aligned on ``nmeasure``.
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        outer_x_prior: Prior for outer-domain flux scaling factors.
        inner_x_prior: Prior for inner-domain flux scaling factors. Defaults to
            a copy of ``outer_x_prior`` (or the standard RHIME prior).
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms from the outer
            domain.
        pollution_events_from_obs: Whether error scaling uses observations.
        no_model_error: Whether to suppress explicit model-error terms.
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra keyword arguments for the offset component.
        power: Likelihood error-scaling exponent or prior.
        outer_state_activity: Optional outer-domain state policy.
        inner_state_activity: Optional inner-domain state policy.
        bc_state_activity: Optional boundary-condition state policy.
        likelihood_builder: Optional complete observation-component builder.

    Returns:
        Fully assembled PyMC model with ``x_outer``, ``x_inner``, their
        separate contributions, and their summed ``mu``.

    Raises:
        KeyError: If either domain sensitivity is absent.
        ValueError: If the two sensitivities do not share the same labelled
            measurement coordinate or do not have distinct state dimensions.
    """
    if "H" not in inv_inputs or "H_inner" not in inv_inputs:
        raise KeyError("Nested RHIME inputs must contain both `H` and `H_inner`.")

    outer_sensitivity = inv_inputs["H"]
    inner_sensitivity = inv_inputs["H_inner"]
    outer_state_dims = [str(dim) for dim in outer_sensitivity.dims if dim != "nmeasure"]
    inner_state_dims = [str(dim) for dim in inner_sensitivity.dims if dim != "nmeasure"]
    if len(outer_state_dims) != 1 or len(inner_state_dims) != 1:
        raise ValueError(
            "Nested RHIME sensitivities must each have `nmeasure` and exactly one state "
            f"dimension; H state dims={outer_state_dims!r}, H_inner state dims={inner_state_dims!r}."
        )
    if outer_state_dims[0] == inner_state_dims[0]:
        raise ValueError(
            "Nested RHIME outer and inner sensitivities require distinct state-dimension names; "
            f"both use {outer_state_dims[0]!r}."
        )
    if not outer_sensitivity.get_index("nmeasure").equals(inner_sensitivity.get_index("nmeasure")):
        raise ValueError("Nested RHIME H and H_inner must have identical labelled nmeasure indexes.")

    prepared_outer_prior, prepared_bc_prior, prepared_sigma_prior, prepared_offset_prior = (
        _prepare_builder_priors(
            x_prior=outer_x_prior,
            bc_prior=bc_prior,
            sigma_prior=sigma_prior,
            offset_prior=offset_prior,
        )
    )
    prepared_inner_prior = prepared_outer_prior.copy() if inner_x_prior is None else inner_x_prior.copy()

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        outer_component = add_state_linear_component(
            outer_sensitivity,
            data_name="hx_outer",
            prior_args=prepared_outer_prior,
            var_name="x_outer",
            output_name="mu_outer",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=outer_state_activity,
        )
        inner_component = add_state_linear_component(
            inner_sensitivity,
            data_name="hx_inner",
            prior_args=prepared_inner_prior,
            var_name="x_inner",
            output_name="mu_inner",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=inner_state_activity,
        )
        mu = pm.Deterministic(
            "mu",
            outer_component.output + inner_component.output,
            dims="nmeasure",
        )
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=mu,
            sigma_alignment=sigma_alignment,
            bc_prior=prepared_bc_prior,
            sigma_prior=prepared_sigma_prior,
            offset_prior=prepared_offset_prior,
            add_offset=add_offset,
            use_bc=use_bc,
            bc_state_activity=bc_state_activity,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            offset_args=offset_args,
            power=power,
            likelihood_builder=likelihood_builder,
        )
        setattr(model, _LIKELIHOOD_RESULT_ATTR, likelihood_result)

    return model


def _build_compiled_rhime_model(
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
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the standard RHIME model through the opt-in flux compiler.

    This produces the same public model graph as :func:`build_rhime_model`,
    while retaining the private plan/compiler path for further development.

    Args:
        inv_inputs: Canonical inversion-input dataset.
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        x_prior: Prior specification for flux scaling factors.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        pollution_events_from_obs: Whether pollution scaling uses observations.
        no_model_error: Whether to suppress the explicit model-error term.
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.
        state_activity: Optional labelled active/fixed flux-state policy.
        bc_state_activity: Optional labelled active/fixed BC-state policy.
        likelihood_builder: Optional complete observation-component builder.

    Returns:
        Built PyMC model.
    """
    x_prior, bc_prior, sigma_prior, offset_prior = _prepare_builder_priors(
        x_prior=x_prior,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
    )
    flux_plan = _normalize_standard_flux_plan(inv_inputs, x_prior, state_activity=state_activity)
    return _assemble_compiled_rhime_model(
        inv_inputs,
        flux_plan=flux_plan,
        sigma_alignment=sigma_alignment,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        add_offset=add_offset,
        use_bc=use_bc,
        bc_state_activity=bc_state_activity,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        aggregation_error_mode=aggregation_error_mode,
        offset_args=offset_args,
        power=power,
        likelihood_builder=likelihood_builder,
    )


def build_rhime_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the standard single-sector RHIME model from a model spec.

    Args:
        inv_inputs: Canonical inversion-input dataset produced by
            ``make_inv_inputs``.
        model_spec: Normalized RHIME model specification.
        likelihood_builder: Optional direct-Python observation-component
            builder. The callable is not stored on ``model_spec``.

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
    builder = build_rhime_model if model_spec.builder_strategy == "concrete" else _build_compiled_rhime_model
    state_activity = model_spec.state_activity
    if model_spec.sector_state_activities is not None:
        state_activity = model_spec.sector_state_activities.get(sector.name, state_activity)
    if sector.state_activity is not None:
        state_activity = sector.state_activity
    return builder(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        x_prior=dict(sector.x_prior),
        state_activity=state_activity,
        bc_prior=model_spec.bc_prior,
        bc_state_activity=model_spec.bc_state_activity,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        aggregation_error_mode=model_spec.aggregation_error_mode,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
        likelihood_builder=likelihood_builder,
    )


def build_nested_rhime_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    inner_x_prior: dict[str, Any] | None = None,
    inner_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build a two-grid nested-domain model from a standard RHIME spec.

    Nested domains are spatial resolutions of one emissions source, not
    emissions sectors. Consequently the current nested builder requires the
    standard one-sector spec and the concrete construction strategy.
    """
    if len(model_spec.sectors) != 1:
        raise ValueError("Nested RHIME model specs must include exactly one emissions sector.")
    if model_spec.builder_strategy != "concrete":
        raise ValueError("Nested RHIME currently supports only builder_strategy='concrete'.")

    sector = model_spec.sectors[0]
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    outer_state_activity = model_spec.state_activity
    if model_spec.sector_state_activities is not None:
        outer_state_activity = model_spec.sector_state_activities.get(
            sector.name,
            outer_state_activity,
        )
    if sector.state_activity is not None:
        outer_state_activity = sector.state_activity

    return build_nested_rhime_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        outer_x_prior=dict(sector.x_prior),
        inner_x_prior=inner_x_prior,
        outer_state_activity=outer_state_activity,
        inner_state_activity=inner_state_activity,
        bc_prior=model_spec.bc_prior,
        bc_state_activity=model_spec.bc_state_activity,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        aggregation_error_mode=model_spec.aggregation_error_mode,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
        likelihood_builder=likelihood_builder,
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
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    state_activity: StateActivity | None = None,
    sector_state_activities: Mapping[str, StateActivity] | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
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
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.
        state_activity: Policy shared by sectors without an explicit override.
            When omitted, exactly-zero sensitivity columns are fixed to one.
        sector_state_activities: Optional activity-policy overrides keyed by
            sector name. Overrides win over ``state_activity``; missing sectors
            use the shared policy. Unknown keys are rejected. An all-false
            policy freezes a complete sector.
        bc_state_activity: Optional active/fixed policy for ``H_bc`` scaling
            states. Omit it to preserve the standard fully sampled BC graph.
        likelihood_builder: Optional complete observation-component builder.

    Returns:
        Built PyMC model.

    Raises:
        KeyError: If required sensitivity inputs are absent.
        ValueError: If sector/source/prior/activity mappings or state layouts,
            labels, fixed values, or state-valued prior parameters are invalid.
    """
    sector_bindings = _resolve_sector_bindings(
        inv_inputs,
        sectors,
        sector_sources=sector_sources,
        sector_variable_suffixes=sector_variable_suffixes,
    )
    sector_components = _resolve_multisector_components(
        inv_inputs,
        sector_bindings,
        sector_priors=sector_priors,
        x_prior=x_prior,
        default_x_prior=DEFAULT_X_PRIOR,
        state_activity=state_activity,
        sector_state_activities=sector_state_activities,
    )
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        sector_outputs = []
        for component in sector_components:
            linear_component = add_state_linear_component(
                component.design,
                data_name=f"hx_{component.variable_suffix}",
                prior_args=dict(component.prior_args),
                var_name=f"x_{component.variable_suffix}",
                output_name=f"mu_{component.variable_suffix}",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=component.state_activity,
            )
            sector_outputs.append(linear_component.output)

        total_mu = pm.Deterministic(
            "mu",
            cast(Any, pt.stack(sector_outputs, axis=0)).sum(axis=0),
            dims="nmeasure",
        )
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=total_mu,
            sigma_alignment=sigma_alignment,
            bc_prior=bc_prior,
            sigma_prior=sigma_prior,
            offset_prior=offset_prior,
            add_offset=add_offset,
            use_bc=use_bc,
            bc_state_activity=bc_state_activity,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            offset_args=offset_args,
            power=power,
            likelihood_builder=likelihood_builder,
        )
        setattr(model, _LIKELIHOOD_RESULT_ATTR, likelihood_result)

    return model


def _build_compiled_rhime_multisector_model(
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
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    state_activity: StateActivity | None = None,
    sector_state_activities: Mapping[str, StateActivity] | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build multisector RHIME through the opt-in flux compiler.

    Args:
        inv_inputs: Canonical inversion-input dataset.
        sigma_alignment: Backend-neutral site and period alignment for sigma.
        sectors: Ordered model sector labels to optimize.
        sector_sources: Mapping from sector label to prepared flux source.
        sector_variable_suffixes: Mapping from sector label to PyMC suffix.
        sector_priors: Complete optional mapping of per-sector priors.
        x_prior: Shared prior used when ``sector_priors`` is absent.
        bc_prior: Prior specification for boundary-condition scaling factors.
        sigma_prior: Prior specification for model-error terms.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        pollution_events_from_obs: Whether pollution scaling uses observations.
        no_model_error: Whether to suppress explicit model-error terms.
        aggregation_error_mode: Aggregation-error representation to use.
        offset_args: Extra keyword arguments forwarded to the offset component.
        power: Exponent or prior specification used in likelihood error scaling.
        state_activity: Optional activity policy shared by all flux sectors.
        sector_state_activities: Optional activity overrides keyed by sector.
        bc_state_activity: Optional active/fixed policy for BC scaling states.
        likelihood_builder: Optional complete observation-component builder.

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
        state_activity=state_activity,
        sector_state_activities=sector_state_activities,
    )
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)
    return _assemble_compiled_rhime_model(
        inv_inputs,
        flux_plan=flux_plan,
        sigma_alignment=sigma_alignment,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
        add_offset=add_offset,
        use_bc=use_bc,
        bc_state_activity=bc_state_activity,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        aggregation_error_mode=aggregation_error_mode,
        offset_args=offset_args,
        power=power,
        likelihood_builder=likelihood_builder,
    )


def build_rhime_multisector_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the shared-basis multi-sector RHIME model from a model spec.

    Args:
        inv_inputs: Canonical inversion-input dataset using either rectangular
            shared-basis or gathered source-specific sensitivity.
        model_spec: Normalized RHIME model specification.
        likelihood_builder: Optional direct-Python observation-component
            builder. The callable is not stored on ``model_spec``.

    Returns:
        Built PyMC model.
    """
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    builder = (
        build_rhime_multisector_model
        if model_spec.builder_strategy == "concrete"
        else _build_compiled_rhime_multisector_model
    )
    sector_state_activities = dict(model_spec.sector_state_activities or {})
    sector_state_activities.update(
        {
            sector.name: sector.state_activity
            for sector in model_spec.sectors
            if sector.state_activity is not None
        }
    )
    return builder(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        sectors=[sector.name for sector in model_spec.sectors],
        sector_sources={sector.name: sector.flux_source for sector in model_spec.sectors},
        sector_variable_suffixes={sector.name: sector.variable_suffix for sector in model_spec.sectors},
        sector_priors={sector.name: dict(sector.x_prior) for sector in model_spec.sectors},
        bc_prior=model_spec.bc_prior,
        bc_state_activity=model_spec.bc_state_activity,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        aggregation_error_mode=model_spec.aggregation_error_mode,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
        state_activity=model_spec.state_activity,
        sector_state_activities=sector_state_activities or None,
        likelihood_builder=likelihood_builder,
    )
