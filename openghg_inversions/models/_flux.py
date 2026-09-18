"""Select and namespace source-resolved designs for model recipes."""

from __future__ import annotations

import re

import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array


def safe_pymc_name(value: str) -> str:
    """Return a stable PyMC-safe suffix for a sector or source name.

    Args:
        value: User-facing scientific label.

    Returns:
        Lowercase underscore-separated suffix, or ``"sector"`` when the label
        contains no usable characters.
    """
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


def _validate_unpadded_sector_design(
    design: xr.DataArray,
    *,
    sector: str,
    source: str,
    observation_dim: str = "nmeasure",
) -> None:
    """Reject rectangular source layouts containing declared padding.

    Args:
        design: Selected source-resolved sensitivity design.
        sector: Scientific sector name used in diagnostics.
        source: OpenGHG source label used in diagnostics.
        observation_dim: Shared observation dimension name.

    Raises:
        ValueError: If declared and prepared state sizes differ.
    """
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
    """Select one source design from rectangular or gathered sensitivity.

    Args:
        design: Source-resolved sensitivity design.
        sector: Scientific sector name used in diagnostics.
        source: OpenGHG source label to select.
        variable_suffix: Backend-safe suffix for gathered state names.
        observation_dim: Shared observation dimension name.
        namespace_state_dim: Whether to suffix a gathered state dimension.

    Returns:
        Two-dimensional sensitivity design for one source.

    Raises:
        ValueError: If the source, rectangular padding, or gathered layout is
            invalid.
    """
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
