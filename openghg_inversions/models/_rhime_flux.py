"""Resolve RHIME flux declarations for concrete builders and compiler plans."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models._rhime_compiler import (
    _FluxPlan,
    _ForwardTermPlan,
    _StatePlan,
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
    return selected.rename({state_dim: f"{state_dim}_{variable_suffix}"})


def _normalize_standard_flux_plan(inv_inputs: xr.Dataset, x_prior: Mapping[str, Any]) -> _FluxPlan:
    """Normalize the standard flux component into a one-state linear plan."""
    state_id = "flux"
    return _FluxPlan(
        states=(
            _StatePlan(
                state_id=state_id,
                variable_name="x",
                prior_args=x_prior,
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
) -> tuple[_ResolvedSectorComponent, ...]:
    """Resolve selected sector designs and priors independently of graph construction."""
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

    components = []
    for binding in sector_bindings:
        if sector_priors is not None:
            prior = sector_priors[binding.name]
        elif x_prior is not None:
            prior = x_prior
        else:
            prior = default_x_prior
        components.append(
            _ResolvedSectorComponent(
                name=binding.name,
                flux_source=binding.flux_source,
                variable_suffix=binding.variable_suffix,
                design=_select_sector_design(
                    inv_inputs["H"],
                    sector=binding.name,
                    source=binding.flux_source,
                    variable_suffix=binding.variable_suffix,
                ),
                prior_args=dict(prior),
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
) -> _FluxPlan:
    """Normalize selected sector designs into separate state and term plans."""
    states: list[_StatePlan] = []
    terms: list[_ForwardTermPlan] = []
    components = _resolve_multisector_components(
        inv_inputs,
        sector_bindings,
        sector_priors=sector_priors,
        x_prior=x_prior,
        default_x_prior=default_x_prior,
    )
    for component in components:
        states.append(
            _StatePlan(
                state_id=component.name,
                variable_name=f"x_{component.variable_suffix}",
                prior_args=component.prior_args,
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
