"""Helpers for managing xarray and PyMC coordinate mismatches.

Xarray coordinates can contain rich objects, including MultiIndex coordinates
from stacked or ragged dimensions such as ``nmeasure`` representing stacked
``(site, time)`` observations. PyMC does not reliably accept all such objects as
model coordinates, so model construction should use sanitized, PyMC-safe coords.

For Stage C, the sanitization policy is intentionally simple: convert each known
dimension coordinate to a range index. The original scientific coordinates are
stored separately so they can later be restored onto ArviZ ``InferenceData``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr


def _coords_to_mapping(coords: dict[str, Any] | xr.Coordinates) -> dict[str, Any]:
    if isinstance(coords, xr.Coordinates):
        return {name: coords[name] for name in coords}
    return dict(coords)


def _coord_values(coord: Any) -> Any:
    if isinstance(coord, xr.DataArray):
        return coord.to_index() if coord.ndim == 1 and coord.name in coord.indexes else coord.values
    if isinstance(coord, xr.IndexVariable):
        return coord.to_index()
    return coord


def _coord_length(coord: Any) -> int:
    if isinstance(coord, xr.DataArray):
        return coord.sizes[coord.dims[0]]
    if isinstance(coord, xr.IndexVariable):
        return coord.sizes[coord.dims[0]]
    return len(coord)


def _coords_equal(left: Any, right: Any) -> bool:
    if isinstance(left, pd.Index) or isinstance(right, pd.Index):
        try:
            return pd.Index(left).equals(pd.Index(right))
        except Exception:
            return False

    try:
        return np.array_equal(np.asarray(left), np.asarray(right))
    except Exception:
        return False


def sanitize_coords_for_pymc(
    coords: dict[str, Any] | xr.Coordinates | object,
    *,
    model_dims: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return PyMC-safe coordinates for the provided dimensions."""
    if not isinstance(coords, (dict, xr.Coordinates)):
        return {}

    dims_to_register = set(model_dims) if model_dims is not None else None
    result: dict[str, np.ndarray] = {}
    for name, coord in _coords_to_mapping(coords).items():
        if dims_to_register is not None and name not in dims_to_register:
            continue
        try:
            size = _coord_length(coord)
        except Exception:
            continue
        result[name] = np.arange(size)
    return result


@dataclass
class CoordRegistry:
    """Track scientific and PyMC-safe coordinates for a model."""

    pymc_coords: dict[str, np.ndarray] = field(default_factory=dict)
    original_coords: dict[str, Any] = field(default_factory=dict)
    auxiliary_coords: dict[str, xr.DataArray] = field(default_factory=dict)

    def _store_original_coord(self, name: str, coord: Any) -> None:
        if name in self.original_coords:
            existing = self.original_coords[name]
            if len(existing) != len(coord):
                raise ValueError(f"Conflicting coord registration for {name!r}: length mismatch.")
            if not _coords_equal(existing, coord):
                raise ValueError(f"Conflicting coord registration for {name!r}: values differ.")
            return

        self.original_coords[name] = coord

    def _store_auxiliary_coord(self, name: str, coord: xr.DataArray) -> None:
        if name in self.auxiliary_coords:
            existing = self.auxiliary_coords[name]
            if existing.dims != coord.dims or existing.shape != coord.shape:
                raise ValueError(f"Conflicting coord registration for {name!r}: shape mismatch.")
            if not _coords_equal(existing.values, coord.values):
                raise ValueError(f"Conflicting coord registration for {name!r}: values differ.")
            return

        self.auxiliary_coords[name] = coord

    def add(
        self,
        coords: dict[str, Any] | xr.Coordinates,
        *,
        model_dims: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> None:
        """Register coordinates, storing scientific coords and PyMC-safe coords."""
        mapping = _coords_to_mapping(coords)
        dims_to_register = tuple(model_dims) if model_dims is not None else tuple(mapping)
        dim_set = set(dims_to_register)
        pymc_coords = sanitize_coords_for_pymc(coords, model_dims=dims_to_register)

        for dim, safe_coord in pymc_coords.items():
            if dim in self.pymc_coords and len(self.pymc_coords[dim]) != len(safe_coord):
                raise ValueError(f"Conflicting coord registration for {dim!r}: length mismatch.")
            self.pymc_coords[dim] = safe_coord

        for dim in dims_to_register:
            if dim not in mapping:
                continue
            coord = mapping[dim]
            original = _coord_values(coord)
            self._store_original_coord(dim, original)

            if isinstance(original, pd.MultiIndex):
                for level_name in original.names:
                    if level_name is None:
                        continue
                    level_values = original.get_level_values(level_name)
                    self._store_auxiliary_coord(
                        level_name,
                        xr.DataArray(
                            level_values.to_numpy(),
                            dims=(dim,),
                            coords={dim: np.arange(len(original))},
                            name=level_name,
                        ),
                    )

        for name, coord in mapping.items():
            if name in dim_set or not isinstance(coord, xr.DataArray):
                continue
            if not set(coord.dims).issubset(dim_set):
                continue
            self._store_auxiliary_coord(
                name,
                xr.DataArray(
                    coord.values,
                    dims=coord.dims,
                    coords={dim: np.arange(coord.sizes[dim]) for dim in coord.dims},
                    name=name,
                ),
            )


def attach_coord_registry(model: pm.Model, registry: CoordRegistry) -> None:
    """Attach a coordinate registry to a PyMC model."""
    setattr(model, "_openghg_coord_registry", registry)


def get_coord_registry(model: pm.Model) -> CoordRegistry | None:
    """Return the coordinate registry attached to a PyMC model, if any."""
    return getattr(model, "_openghg_coord_registry", None)


def add_coords(
    coords: dict[str, np.ndarray] | xr.Coordinates,
    *,
    model_dims: tuple[str, ...] | list[str] | set[str] | None = None,
) -> None:
    """Register coords on the active model and store originals when possible."""
    pymc_coords = sanitize_coords_for_pymc(coords, model_dims=model_dims)
    pymc_coords_list = {name: coord.tolist() for name, coord in pymc_coords.items()}

    with pm.modelcontext(None) as model:
        model.add_coords(pymc_coords_list)

        registry = get_coord_registry(model)
        if registry is not None:
            registry.add(coords, model_dims=model_dims)


def restore_inferencedata_coords(
    idata: az.InferenceData,
    coords_or_registry: CoordRegistry | dict[str, Any],
) -> az.InferenceData:
    """Restore saved coordinates onto matching InferenceData groups."""
    original_coords = (
        coords_or_registry.original_coords
        if isinstance(coords_or_registry, CoordRegistry)
        else coords_or_registry
    )

    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if not isinstance(group, xr.Dataset):
            continue

        restored_multiindex_dims: set[str] = set()
        for dim, coord in original_coords.items():
            if dim not in group.dims:
                continue
            if len(coord) != group.sizes[dim]:
                continue
            if isinstance(coord, pd.MultiIndex):
                group = group.assign_coords(xr.Coordinates.from_pandas_multiindex(coord, dim))
                restored_multiindex_dims.add(dim)
            else:
                group = group.assign_coords({dim: coord})

        if isinstance(coords_or_registry, CoordRegistry):
            for name, coord in coords_or_registry.auxiliary_coords.items():
                if not set(coord.dims).issubset(set(group.dims)):
                    continue
                if any(group.sizes[dim] != coord.sizes[dim] for dim in coord.dims):
                    continue
                if any(dim in restored_multiindex_dims for dim in coord.dims):
                    continue
                group = group.assign_coords({name: coord})

        setattr(idata, group_name, group)

    return idata
