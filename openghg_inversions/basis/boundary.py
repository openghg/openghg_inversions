"""Spatial partitions of the four native NAME boundary curtains.

A :data:`BoundaryBasis` is either a resolved labelled dataset or a builder
called with each site's dataset. Built-in selections are ordinary builders, so
NESW and twelve-region bases follow the same projection path as exact files.
"""

from collections.abc import Callable
from functools import partial
import math
from numbers import Real

import numpy as np
import xarray as xr

BoundaryBasis = xr.Dataset | Callable[[xr.Dataset], xr.Dataset]
"""Resolved ``bc_basis_n/e/s/w`` dataset, or a builder from site coordinates."""

BOUNDARY_BASIS_ALGORITHMS = ("NESW", "twelve-region")


def nesw_basis(curtains: xr.Dataset) -> xr.Dataset:
    """Assign each whole NESW curtain to its own region.

    Args:
        curtains: Site dataset. It is not read; the basis is independent of
            curtain geometry and payloads.

    Returns:
        Dimensionless ``bc_basis_n/e/s/w`` one-hot weights over regions
        ``n,e,s,w``, with no spatial dimensions.
    """
    del curtains
    regions = list("nesw")
    basis = xr.Dataset(coords={"region": regions}, attrs={"bc_basis_algorithm": "NESW"})
    for direction in regions:
        basis[f"bc_basis_{direction}"] = xr.DataArray(
            np.array([region == direction for region in regions], dtype=np.int8),
            dims="region", attrs={"units": "1"},
        )
    return basis


def validate_split_height(split_height: object) -> float:
    """Return a finite native height-coordinate split as ``float``.

    Raises:
        ValueError: If the value is boolean, non-real or non-finite.
    """
    if isinstance(split_height, bool) or not isinstance(split_height, Real) or not math.isfinite(split_height):
        raise ValueError("split_height must be a finite native height-coordinate value.")
    return float(split_height)


def boundary_basis(algorithm: object, *, split_height: object = 3000.0) -> Callable[[xr.Dataset], xr.Dataset]:
    """Select a built-in boundary basis builder without reading data.

    Args:
        algorithm: One of :data:`BOUNDARY_BASIS_ALGORITHMS`.
        split_height: Twelve-region height split; ignored by NESW.

    Raises:
        ValueError: If the algorithm is unknown or, for twelve-region, the
            split height is invalid.
    """
    if algorithm == "NESW":
        return nesw_basis
    if algorithm == "twelve-region":
        return partial(twelve_region_basis, split_height=validate_split_height(split_height))
    raise ValueError(f"Boundary basis algorithm must be one of {BOUNDARY_BASIS_ALGORITHMS!r}; got {algorithm!r}.")


def twelve_region_basis(curtains: xr.Dataset, *, split_height: float = 3000.0) -> xr.Dataset:
    """Split each NESW curtain into two lower halves and one upper region.

    Args:
        curtains: Dataset carrying indexed ``height``, ``lat`` and ``lon``
            coordinates. Only coordinates are read; curtain payloads are borrowed.
        split_height: Native height-coordinate value separating lower cells
            (strictly below) from upper cells (at or above). The initial 3000
            convention follows the prototype, not an exact published InTEM
            threshold. No unit conversion or height-datum assumption is made.

    Returns:
        Dimensionless ``bc_basis_n/e/s/w`` masks, with twelve regions ordered
        by direction ``n,e,s,w``, then ``lower_first,lower_second,upper``.
        North/south split longitude; east/west split latitude. First means
        coordinate strictly below the midpoint of the coordinate extrema;
        equality belongs to second. Each native curtain cell belongs to exactly
        one region. There is no top boundary. Coordinate attributes are retained.

    Raises:
        ValueError: If the split height or spatial coordinates are not finite
            numeric values, or a required indexed coordinate is absent/empty.
    """
    split_height = validate_split_height(split_height)
    coordinates = {}
    for axis in ("height", "lat", "lon"):
        if axis not in curtains.indexes or curtains[axis].dims != (axis,) or not curtains.sizes[axis]:
            raise ValueError(f"twelve-region requires a nonempty indexed {axis!r} coordinate.")
        coordinate = curtains[axis]
        if coordinate.dtype.kind not in "iuf" or not np.isfinite(coordinate.values).all():
            raise ValueError(f"twelve-region requires finite numeric {axis!r} coordinates.")
        coordinates[axis] = coordinate
    labels = [f"{direction}_{part}" for direction in "nesw"
              for part in ("lower_first", "lower_second", "upper")]
    basis = xr.Dataset(coords={**coordinates, "region": labels}, attrs={
        "bc_basis_algorithm": "twelve-region",
        "bc_basis_split_height": split_height,
        "bc_basis_height_units": coordinates["height"].attrs.get("units", "unspecified"),
        "bc_basis_height_datum": coordinates["height"].attrs.get(
            "datum", coordinates["height"].attrs.get("vertical_datum", "unspecified")),
        "bc_basis_lat_units": coordinates["lat"].attrs.get("units", "unspecified"),
        "bc_basis_lon_units": coordinates["lon"].attrs.get("units", "unspecified"),
        "bc_basis_height_rule": "lower: height < split; upper: height >= split; native units and datum",
        "bc_basis_horizontal_rule": "first: coordinate < (min + max) / 2; second: coordinate >= midpoint",
        "bc_basis_geometry": "four lateral curtains; no top boundary",
    })
    lower = coordinates["height"].values[:, None] < split_height
    for index, direction in enumerate("nesw"):
        axis = "lon" if direction in "ns" else "lat"
        horizontal = coordinates[axis].values
        midpoint = float(horizontal.min()) / 2 + float(horizontal.max()) / 2
        basis.attrs[f"bc_basis_midpoint_{direction}"] = midpoint
        first = horizontal[None, :] < midpoint
        mask = np.zeros((len(coordinates["height"]), len(horizontal), 12), dtype=np.int8)
        mask[:, :, 3 * index] = lower & first
        mask[:, :, 3 * index + 1] = lower & ~first
        mask[:, :, 3 * index + 2] = np.broadcast_to(~lower, mask.shape[:2])
        basis[f"bc_basis_{direction}"] = xr.DataArray(
            mask, dims=("height", axis, "region"), attrs={"units": "1", "horizontal_midpoint": midpoint},
        )
    return basis
