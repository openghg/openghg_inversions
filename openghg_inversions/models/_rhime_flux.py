"""Normalize RHIME flux declarations for concrete builders and compiler plans.

Semantic sector names remain separate from OpenGHG source labels and backend
variable suffixes. Rectangular shared-basis and gathered source-specific layouts
use the same entry points. Priors follow per-sector, shared, then default
precedence; activity follows per-sector override, then shared policy. Scientific
labels are resolved before backend namespacing and graph mutation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models._rhime_compiler import (
    _FluxPlan,
    _ForwardTermPlan,
    _StatePlan,
)
from openghg_inversions.models.state_activity import (
    StateActivity,
    active_prior_args,
    detect_zero_sensitivity,
    resolve_state_activity,
)


@dataclass(frozen=True)
class _ResolvedSectorBinding:
    """Bind one semantic sector to prepared source data and backend naming."""

    name: str
    flux_source: str
    variable_suffix: str


@dataclass(frozen=True)
class _ResolvedSectorComponent:
    """Describe one selected sector design and its flux-scaling prior."""

    name: str
    flux_source: str
    variable_suffix: str
    design: xr.DataArray
    prior_args: Mapping[str, Any]
    state_activity: StateActivity


def safe_pymc_name(value: str) -> str:
    """Return a stable PyMC-safe suffix for a user-facing sector/source name."""
    name = re.sub(r"\W+", "_", str(value).strip().lower()).strip("_")
    return name or "sector"


def _prepared_sources(design: xr.DataArray, *, observation_dim: str = "nmeasure") -> list[str]:
    """Return ordered source labels from rectangular or gathered sensitivity."""
    source_coord = design.coords.get("source")
    if source_coord is None:
        raise ValueError("Multi-sector RHIME requires inv_inputs['H'] to contain OpenGHG source labels.")

    source_labels = [str(value) for value in source_coord.values]
    if "source" in design.dims:
        duplicate_sources = list(
            dict.fromkeys(source for source in source_labels if source_labels.count(source) > 1)
        )
        if duplicate_sources:
            raise ValueError(
                "Multi-sector RHIME requires unique inv_inputs['H'].source labels; "
                f"duplicate source {duplicate_sources[0]!r}."
            )
        return source_labels

    state_dims = [str(dim) for dim in design.dims if dim != observation_dim]
    if len(state_dims) != 1 or source_coord.dims != (state_dims[0],):
        raise ValueError(
            "Gathered multi-sector H requires one state dimension carrying the 'source' coordinate."
        )
    state_index = design.indexes.get(state_dims[0])
    if not isinstance(state_index, pd.MultiIndex) or "source" not in state_index.names:
        raise ValueError(
            "Gathered multi-sector H requires its state dimension to use a MultiIndex "
            "containing a 'source' level."
        )
    if not state_index.is_unique:
        duplicate = state_index[state_index.duplicated()][0]
        raise ValueError(
            f"Gathered multi-sector H requires unique state labels; duplicate state {duplicate!r}."
        )
    return list(dict.fromkeys(source_labels))


def _resolve_sector_bindings(
    inv_inputs: xr.Dataset,
    sectors: Sequence[str] | None,
    *,
    sector_sources: Mapping[str, str] | None,
    sector_variable_suffixes: Mapping[str, str] | None,
) -> tuple[_ResolvedSectorBinding, ...]:
    """Resolve semantic sectors against prepared source provenance."""
    available = _prepared_sources(inv_inputs["H"])
    if sectors is None:
        sectors = list(sector_sources) if sector_sources is not None else available
    sector_names = [str(sector) for sector in sectors]

    if len(sector_names) < 2:
        raise ValueError("Multi-sector RHIME requires at least two sectors.")
    duplicate_sectors = list(
        dict.fromkeys(sector for sector in sector_names if sector_names.count(sector) > 1)
    )
    if duplicate_sectors:
        raise ValueError(
            f"Multi-sector RHIME requires unique sector names; duplicate sector {duplicate_sectors[0]!r}."
        )
    if any(not sector.strip() for sector in sector_names):
        raise ValueError("Multi-sector RHIME requires non-empty sector names.")

    source_by_sector = (
        {str(sector): str(source) for sector, source in sector_sources.items()}
        if sector_sources is not None
        else {sector: sector for sector in sector_names}
    )
    suffix_by_sector = (
        {str(sector): str(suffix) for sector, suffix in sector_variable_suffixes.items()}
        if sector_variable_suffixes is not None
        else {}
    )

    unused_mappings = [sector for sector in source_by_sector if sector not in sector_names]
    if unused_mappings:
        raise ValueError(f"`sector_sources` contains unused sector key(s): {unused_mappings!r}.")
    missing_mappings = [sector for sector in sector_names if sector not in source_by_sector]
    if missing_mappings:
        raise ValueError(f"Sector(s) {missing_mappings!r} are missing from `sector_sources`.")

    source_sectors: dict[str, list[str]] = {}
    for sector in sector_names:
        source_sectors.setdefault(source_by_sector[sector], []).append(sector)
    duplicate_sources = {
        source: mapped_sectors for source, mapped_sectors in source_sectors.items() if len(mapped_sectors) > 1
    }
    if duplicate_sources:
        details = ", ".join(
            f"source {source!r} is mapped by sectors {mapped_sectors!r}"
            for source, mapped_sectors in duplicate_sources.items()
        )
        raise ValueError(
            "Multi-sector RHIME requires a distinct source for each current sector; " + details + "."
        )

    missing = [
        (sector, source_by_sector[sector])
        for sector in sector_names
        if source_by_sector[sector] not in available
    ]
    if missing:
        details = ", ".join(f"sector {sector!r} -> source {source!r}" for sector, source in missing)
        raise ValueError(
            f"Source data required by {details} is not present in inv_inputs['H'].source; "
            f"available source(s): {available!r}."
        )

    unused_suffixes = [sector for sector in suffix_by_sector if sector not in sector_names]
    if unused_suffixes:
        raise ValueError(f"`sector_variable_suffixes` contains unused sector key(s): {unused_suffixes!r}.")
    bindings = tuple(
        _ResolvedSectorBinding(
            name=sector,
            flux_source=source_by_sector[sector],
            variable_suffix=suffix_by_sector.get(sector, safe_pymc_name(sector)),
        )
        for sector in sector_names
    )
    suffixes = [binding.variable_suffix for binding in bindings]
    if len(suffixes) != len(set(suffixes)):
        duplicate = next(suffix for suffix in suffixes if suffixes.count(suffix) > 1)
        raise ValueError(
            "Sector names must be unique after PyMC name sanitisation; "
            f"duplicate sanitized name {duplicate!r}."
        )
    return bindings


def _validate_unpadded_sector_design(
    design: xr.DataArray,
    *,
    sector: str,
    source: str,
    observation_dim: str = "nmeasure",
) -> None:
    """Reject rectangular source layouts containing declared padding."""
    state_dims = [str(dim) for dim in design.dims if dim != observation_dim]
    if design.ndim != 2 or observation_dim not in design.dims or len(state_dims) != 1:
        return

    state_dim = state_dims[0]
    source_region_count = design.coords.get("source_region_count")
    if source_region_count is None:
        return
    declared_regions = int(source_region_count.item())
    prepared_regions = design.sizes[state_dim]
    if declared_regions != prepared_regions:
        raise ValueError(
            f"Sector {sector!r} -> source {source!r} declares {declared_regions} "
            f"{state_dim} elements but prepared H has {prepared_regions}. Ragged "
            "source-specific state blocks must use a gathered state coordinate."
        )


def _select_sector_design(
    design: xr.DataArray,
    *,
    sector: str,
    source: str,
    variable_suffix: str,
    observation_dim: str = "nmeasure",
    namespace_state_dim: bool = True,
) -> xr.DataArray:
    """Select one source design from rectangular or gathered sensitivity."""
    available_sources = _prepared_sources(design, observation_dim=observation_dim)
    if source not in available_sources:
        raise ValueError(
            f"Sector {sector!r} requires source {source!r}, but prepared H contains {available_sources!r}."
        )

    if "source" in design.dims:
        selected = design.sel(source=source, drop=False)
        _validate_unpadded_sector_design(
            selected,
            sector=sector,
            source=source,
            observation_dim=observation_dim,
        )
        return design.sel(source=source, drop=True)

    state_dims = [str(dim) for dim in design.dims if dim != observation_dim]
    state_dim = state_dims[0]
    index = design.indexes.get(state_dim)
    assert isinstance(index, pd.MultiIndex)
    ragged_levels = [str(name) for name in index.names if name != "source"]
    if len(ragged_levels) != 1:
        raise ValueError(
            f"Gathered H state dimension {state_dim!r} must contain exactly the "
            f"'source' and one ragged-region level; found {list(index.names)!r}."
        )

    selected = select_gathered_data_array(
        design,
        key=source,
        key_dim="source",
        ragged_dim=ragged_levels[0],
        stack_dim=state_dim,
    )
    if namespace_state_dim:
        return selected.rename({state_dim: f"{state_dim}_{variable_suffix}"})
    return selected


def _namespace_sector_state_coords(
    design: xr.DataArray,
    *,
    variable_suffix: str,
    observation_dim: str = "nmeasure",
    namespace_state_dim: bool = False,
) -> xr.DataArray:
    """Namespace sector-local auxiliary coordinates for backend registration.

    Semantic policies are resolved before this transformation so a shared
    policy can continue to refer to scientific coordinate names such as
    ``basis_group``. Non-scalar auxiliary coordinates spanning a sector-local
    state dimension receive the backend ``variable_suffix``, including
    coordinates that also span the shared observation dimension. Gathered
    layouts also namespace the state dimension so sectors with different
    lengths can coexist.

    Args:
        design: Selected two-dimensional sector sensitivity.
        variable_suffix: Validated backend suffix for the sector.
        observation_dim: Name of the shared observation dimension.
        namespace_state_dim: Whether to suffix the state dimension itself.

    Returns:
        Design with sector-local auxiliary coordinate names namespaced.

    Raises:
        ValueError: If a generated coordinate name already exists.
    """
    state_dims = {str(dim) for dim in design.dims if dim != observation_dim}
    rename: dict[str, str] = {}
    if namespace_state_dim:
        rename.update({dim: f"{dim}_{variable_suffix}" for dim in state_dims})
    for name, coord in design.coords.items():
        coord_name = str(name)
        if coord_name in design.dims or not coord.dims:
            continue
        if state_dims.intersection(map(str, coord.dims)):
            namespaced = f"{coord_name}_{variable_suffix}"
            if namespaced in design.coords and namespaced != coord_name:
                raise ValueError(
                    f"Cannot namespace sector coordinate {coord_name!r}: "
                    f"coordinate {namespaced!r} already exists."
                )
            rename[coord_name] = namespaced
    return design.rename(rename)


def _normalize_standard_flux_plan(
    inv_inputs: xr.Dataset,
    x_prior: Mapping[str, Any],
    *,
    state_activity: StateActivity | None = None,
) -> _FluxPlan:
    """Normalize the standard flux component into a one-state linear plan.

    Args:
        inv_inputs: Canonical inputs containing the required ``H`` design.
        x_prior: Flux-scaling prior specification.
        state_activity: Optional activity policy, resolved during compilation.

    Returns:
        A single-state, single-forward-term compiler plan.

    Raises:
        KeyError: If ``inv_inputs`` does not contain ``H``.
    """
    state_id = "flux"
    return _FluxPlan(
        states=(
            _StatePlan(
                state_id=state_id,
                variable_name="x",
                prior_args=x_prior,
                state_activity=state_activity,
            ),
        ),
        terms=(
            _ForwardTermPlan(
                term_id=state_id,
                state_id=state_id,
                design=inv_inputs["H"],
                data_name="hx",
                deterministic_name="mu",
                coefficient=1.0,
            ),
        ),
    )


def _resolve_multisector_components(
    inv_inputs: xr.Dataset,
    sector_bindings: Sequence[_ResolvedSectorBinding],
    *,
    sector_priors: Mapping[str, Mapping[str, Any]] | None,
    x_prior: Mapping[str, Any] | None,
    default_x_prior: Mapping[str, Any],
    state_activity: StateActivity | None = None,
    sector_state_activities: Mapping[str, StateActivity] | None = None,
) -> tuple[_ResolvedSectorComponent, ...]:
    """Resolve multisector designs and priors independently of graph construction.

    Each semantic sector binding selects sensitivity data by its OpenGHG
    ``source`` label. Rectangular canonical inputs use a ``source`` dimension;
    selection removes that dimension, leaving the observation and state
    dimensions. Gathered ragged inputs use one state dimension with a
    MultiIndex containing ``source`` and one region level; the selected state
    dimension is renamed with the sector variable suffix so sectors may retain
    different scientific coordinates.

    Prior precedence is explicit: a supplied ``sector_priors`` mapping wins,
    otherwise ``x_prior`` is shared by every sector, and ``default_x_prior`` is
    used only when neither is supplied. When ``sector_priors`` is present it
    must define exactly the resolved sector set.

    Args:
        inv_inputs: Canonical inversion inputs containing source-resolved
            sensitivity in ``H``.
        sector_bindings: Ordered semantic sector, source, and backend-name
            bindings already validated against the requested sectors.
        sector_priors: Optional complete mapping from sector names to
            flux-scaling prior specifications.
        x_prior: Optional prior shared by every sector when per-sector priors
            are absent.
        default_x_prior: Final shared prior used when neither explicit prior
            option is supplied.
        state_activity: Optional activity policy shared by every sector.
        sector_state_activities: Optional activity-policy overrides keyed by
            semantic sector name.

    Returns:
        Ordered resolved components containing each sector identity, selected
        two-dimensional design, variable suffix, and copied prior metadata.

    Raises:
        KeyError: If ``inv_inputs`` does not contain ``H``.
        ValueError: If per-sector prior keys are missing or unused, source
            selection fails, a rectangular design declares padding, or a
            gathered design has an invalid source/state layout.
    """
    sector_names = [binding.name for binding in sector_bindings]
    if sector_priors is not None:
        missing_priors = [sector for sector in sector_names if sector not in sector_priors]
        unused_priors = [sector for sector in sector_priors if sector not in sector_names]
        if missing_priors or unused_priors:
            raise ValueError(
                "`sector_priors` must define exactly one prior for every sector when supplied; "
                f"missing sector prior(s): {missing_priors!r}; "
                f"unused sector prior key(s): {unused_priors!r}."
            )

    sector_state_activities = dict(sector_state_activities or {})
    unknown_activity_sectors = sorted(set(sector_state_activities) - set(sector_names))
    if unknown_activity_sectors:
        raise ValueError(f"State activity supplied for unknown sector(s) {unknown_activity_sectors!r}.")

    components = []
    for binding in sector_bindings:
        if sector_priors is not None:
            prior = sector_priors[binding.name]
        elif x_prior is not None:
            prior = x_prior
        else:
            prior = default_x_prior
        gathered_layout = "source" not in inv_inputs["H"].dims
        design = _select_sector_design(
            inv_inputs["H"],
            sector=binding.name,
            source=binding.flux_source,
            variable_suffix=binding.variable_suffix,
            namespace_state_dim=False,
        )
        semantic_policy = sector_state_activities.get(binding.name, state_activity)
        resolved_activity = resolve_state_activity(
            detect_zero_sensitivity(design),
            semantic_policy,
        )
        all_active = replace(
            resolved_activity,
            active=xr.ones_like(resolved_activity.active, dtype=bool),
        )
        normalized_prior = active_prior_args(dict(prior), all_active)
        backend_design = _namespace_sector_state_coords(
            design,
            variable_suffix=binding.variable_suffix,
            namespace_state_dim=gathered_layout,
        )
        backend_state_dim = next(str(dim) for dim in backend_design.dims if dim != "nmeasure")
        semantic_state_dim = resolved_activity.state_dim
        rename_state = (
            {semantic_state_dim: backend_state_dim} if semantic_state_dim != backend_state_dim else {}
        )
        resolved_policy = StateActivity(
            active=resolved_activity.active.rename(rename_state),
            fixed_value=resolved_activity.fixed_value.rename(rename_state),
            prune_zero=False,
        )
        components.append(
            _ResolvedSectorComponent(
                name=binding.name,
                flux_source=binding.flux_source,
                variable_suffix=binding.variable_suffix,
                design=backend_design,
                prior_args=normalized_prior,
                state_activity=resolved_policy,
            )
        )
    return tuple(components)


def _normalize_multisector_flux_plan(
    inv_inputs: xr.Dataset,
    sector_bindings: Sequence[_ResolvedSectorBinding],
    *,
    sector_priors: Mapping[str, Mapping[str, Any]] | None,
    x_prior: Mapping[str, Any] | None,
    default_x_prior: Mapping[str, Any],
    state_activity: StateActivity | None = None,
    sector_state_activities: Mapping[str, StateActivity] | None = None,
) -> _FluxPlan:
    """Normalize selected sector designs into separate state and term plans.

    Per-sector priors and activity policies override shared equivalents;
    gathered scientific labels are resolved before backend namespacing.

    Args:
        inv_inputs: Canonical source-resolved inversion inputs.
        sector_bindings: Ordered validated sector/source/backend bindings.
        sector_priors: Optional complete per-sector prior mapping.
        x_prior: Optional prior shared by all sectors.
        default_x_prior: Fallback prior when explicit priors are absent.
        state_activity: Optional policy shared by all sectors.
        sector_state_activities: Optional per-sector policy overrides.

    Returns:
        One compiler state and forward term per sector.

    Raises:
        KeyError: If required sensitivity input ``H`` is absent.
        ValueError: If layouts or sector/prior/activity mappings are invalid.
    """
    states: list[_StatePlan] = []
    terms: list[_ForwardTermPlan] = []
    components = _resolve_multisector_components(
        inv_inputs,
        sector_bindings,
        sector_priors=sector_priors,
        x_prior=x_prior,
        default_x_prior=default_x_prior,
        state_activity=state_activity,
        sector_state_activities=sector_state_activities,
    )
    for component in components:
        states.append(
            _StatePlan(
                state_id=component.name,
                variable_name=f"x_{component.variable_suffix}",
                prior_args=component.prior_args,
                state_activity=component.state_activity,
            )
        )
        terms.append(
            _ForwardTermPlan(
                term_id=component.name,
                state_id=component.name,
                design=component.design,
                data_name=f"hx_{component.variable_suffix}",
                deterministic_name=f"mu_{component.variable_suffix}",
                coefficient=1.0,
            )
        )
    return _FluxPlan(states=tuple(states), terms=tuple(terms))
