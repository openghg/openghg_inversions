"""Helpers for managing xarray and PyMC coordinate mismatches.

Xarray coordinates can contain rich objects, including MultiIndex coordinates
from stacked or ragged dimensions such as ``nmeasure`` representing stacked
``(site, time)`` observations. PyMC does not reliably accept all such objects as
model coordinates, so model construction should use sanitized, PyMC-safe coords.

The current sanitization policy is intentionally simple: convert each known
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
    """Convert supported coordinate containers to a plain mapping."""
    if isinstance(coords, xr.Coordinates):
        return {str(name): coords[name] for name in coords}
    return dict(coords)


def _coord_values(coord: Any) -> Any:
    """Extract comparable coordinate values from xarray coordinate objects."""
    if isinstance(coord, xr.DataArray):
        return coord.to_index() if coord.ndim == 1 and coord.name in coord.indexes else coord.values
    if isinstance(coord, xr.IndexVariable):
        return coord.to_index()
    return coord


def _coord_length(coord: Any) -> int:
    """Return the logical length of a coordinate-like object."""
    if isinstance(coord, xr.DataArray):
        return coord.sizes[coord.dims[0]]
    if isinstance(coord, xr.IndexVariable):
        return coord.sizes[coord.dims[0]]
    return len(coord)


def _coords_equal(left: Any, right: Any) -> bool:
    """Compare coordinate values while tolerating pandas and NumPy types."""
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
    """Convert coordinate metadata into the range-based format to use with PyMC.

    PyMC accepts fewer coordinate types than Xarray, so for simplicity, we convert
    all coordinates to range coordinates, and use the range coordinates with PyMC.

    Args:
        coords: Coordinate mapping or xarray coordinate container.
        model_dims: Optional subset of dimensions to sanitize. When omitted,
            all dimensions found in ``coords`` are considered.

    Returns:
        A mapping from model dimension name to a simple ``np.arange`` index of
        the corresponding length.
    """
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
    """Track scientific and PyMC-safe coordinates for a model.

    Attributes:
        pymc_coords: Sanitized coordinates actually registered with PyMC.
        original_coords: Original scientific coordinates keyed by model
            dimension name.
        auxiliary_coords: Additional non-dimension coordinates attached to
            model dimensions, such as exploded ``time`` or ``site`` coordinates
            derived from a stacked ``nmeasure`` MultiIndex.
    """

    pymc_coords: dict[str, np.ndarray] = field(default_factory=dict)
    original_coords: dict[str, Any] = field(default_factory=dict)
    auxiliary_coords: dict[str, xr.DataArray] = field(default_factory=dict)

    def _store_original_coord(self, name: str, coord: Any) -> None:
        """Store an original coordinate, rejecting conflicting re-registrations."""
        if name in self.original_coords:
            existing = self.original_coords[name]
            if len(existing) != len(coord):
                raise ValueError(f"Conflicting coord registration for {name!r}: length mismatch.")
            if not _coords_equal(existing, coord):
                raise ValueError(f"Conflicting coord registration for {name!r}: values differ.")
            return

        self.original_coords[name] = coord

    def _store_auxiliary_coord(self, name: str, coord: xr.DataArray) -> None:
        """Store an auxiliary coordinate, rejecting conflicting re-registrations."""
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
        """Register model and auxiliary coordinates with consistency checks.

        Args:
            coords: Coordinate mapping or xarray coordinate container to
                register.
            model_dims: Optional subset of model dimensions represented by the
                current data variable. Auxiliary coordinates attached to these
                dimensions are also preserved when possible.

        Raises:
            ValueError: If the same coordinate name is registered more than once
                with conflicting lengths, shapes, or values.
        """
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

            # Preserve exploded MultiIndex levels so users can recover useful
            # scientific coordinates even though PyMC only sees range indices.
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
            # Only keep auxiliary coords that are actually attached to the
            # registered model dims; unrelated coords are not useful to restore.
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
    """Register coordinates on the active model and capture scientific metadata.

    Args:
        coords: Coordinate mapping or xarray coordinate container to register.
        model_dims: Optional subset of model dimensions represented by the
            current data variable. When provided, auxiliary coordinates attached
            to those dimensions are also stored in the registry.

    This helper must be called inside an active ``pm.Model`` context.
    """
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
    """Restore saved scientific coordinates onto matching ``InferenceData`` groups.

    Args:
        idata: Inference data object returned by sampling.
        coords_or_registry: Either a ``CoordRegistry`` or a legacy mapping of
            original coordinates keyed by dimension name.

    Returns:
        The same ``InferenceData`` object with compatible original coordinates
        and auxiliary coordinates restored onto its xarray groups.
    """
    original_coords = (
        coords_or_registry.original_coords
        if isinstance(coords_or_registry, CoordRegistry)
        else coords_or_registry
    )

    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if not isinstance(group, xr.Dataset):
            continue

        restored_multiindex_levels: set[str] = set()
        for dim, coord in original_coords.items():
            if dim not in group.dims:
                continue
            if len(coord) != group.sizes[dim]:
                continue
            # Restore true dimension coords first so downstream auxiliary coords
            # can be validated against the final dimension layout.
            if isinstance(coord, pd.MultiIndex):
                group = group.assign_coords(xr.Coordinates.from_pandas_multiindex(coord, dim))
                restored_multiindex_levels.update(name for name in coord.names if name is not None)
            else:
                group = group.assign_coords({dim: coord})

        if isinstance(coords_or_registry, CoordRegistry):
            for name, coord in coords_or_registry.auxiliary_coords.items():
                if not set(coord.dims).issubset(set(group.dims)):
                    continue
                if any(group.sizes[dim] != coord.sizes[dim] for dim in coord.dims):
                    continue
                # Xarray has already recreated MultiIndex level coordinates,
                # but unrelated auxiliaries on the same dimension still need
                # to be restored.
                if name in restored_multiindex_levels:
                    continue
                # Assign positionally after restoring the scientific dimension
                # coordinate.  Reusing the registry's range coordinate here
                # would trigger xarray label alignment against the restored
                # labels and silently replace ordinary auxiliaries with NaN.
                group = group.assign_coords({name: (coord.dims, np.array(coord.values, copy=True))})

        setattr(idata, group_name, group)

    return idata
