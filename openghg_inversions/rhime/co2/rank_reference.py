"""Validate labelled covariance supplied for CO2 aggregation-error rank selection."""

from __future__ import annotations

import re
from typing import Any

import xarray as xr

from openghg_inversions.array_ops import validate_covariance_coordinates
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale


_SQUARED_UNITS = re.compile(r"^\s*\(?\s*(?P<base>[^()]+?)\s*\)?\s*(?:\*\*|\^)\s*2\s*$")


def squared_mole_fraction_scale(units: Any, observation_units: str, *, name: str) -> float:
    """Return the multiplier from squared mole-fraction ``units`` to observation units squared.

    Mole fractions are dimensionless, so a unit registry would also convert an
    unsquared ``ppm`` to ``(ppm)**2``. The units must therefore be written
    explicitly as a squared mole-fraction unit, such as ``(ppb)^2`` or
    ``ppm**2``.

    Raises:
        ValueError: If ``units`` is empty or not a squared mole-fraction unit.
    """
    if not isinstance(units, str) or not units.strip():
        raise ValueError(f"{name} requires non-empty units.")
    match = _SQUARED_UNITS.match(units)
    if match is None:
        raise ValueError(f"{name} units {units!r} must be a squared mole-fraction unit, such as '(ppm)^2'.")
    base = match.group("base")
    try:
        ratio = mole_fraction_unit_scale(base, context=name) / mole_fraction_unit_scale(
            observation_units, context="CO2 observations"
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} units {units!r} must be a squared mole-fraction unit, such as '(ppm)^2'."
        ) from exc
    return float(ratio**2)


def require_reference_labels(
    reference: xr.DataArray,
    *,
    observation_index: Any,
    row_dims: tuple[str, ...],
) -> None:
    """Check a supplied labelled :math:`R_0` against the canonical observation order.

    The square shape and the repeated axis labels are checked by
    :func:`~openghg_inversions.array_ops.validate_covariance_coordinates`, so
    the companion column dimension may carry renamed MultiIndex levels. Once
    the axes match each other, checking the rows against the canonical index
    is sufficient.

    Args:
        reference: Supplied observation-by-observation covariance.
        observation_index: Canonical observation index.
        row_dims: Accepted names for the row dimension.

    Raises:
        ValueError: If the dimensions are not accepted, an axis is not
            indexed, the axes differ, or the rows are not in canonical order.
    """
    if reference.ndim != 2 or reference.dims[0] not in row_dims or reference.dims[1] == reference.dims[0]:
        raise ValueError(
            f"reference_covariance must be observation-by-observation with rows on one of {list(row_dims)!r}."
        )
    row_dim, column_dim = (str(dim) for dim in reference.dims)
    if row_dim not in reference.indexes or column_dim not in reference.indexes:
        raise ValueError("reference_covariance must carry an index on both axes.")
    try:
        validate_covariance_coordinates(reference, dim=row_dim, covariance_dim=column_dim)
    except ValueError as exc:
        raise ValueError(f"reference_covariance: {exc}") from exc
    if not reference.indexes[row_dim].equals(observation_index):
        raise ValueError("reference_covariance rows must match the canonical observation order.")
