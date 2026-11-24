"""Reference mismatch covariance used to choose the CO2 aggregation-error rank."""

from __future__ import annotations

from collections.abc import Mapping
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.observation_error import ReferenceMismatchCovariance, _is_positive_real


@dataclass(frozen=True)
class Co2ReferenceCovariance:
    r"""Settings for the default CO2 reference mismatch covariance :math:`R_0`.

    Preparation builds

    .. math::

        R_0 = \mathrm{diag}(e^2) + S_0 C S_0,

    where :math:`e` is the reported observation error (``mf_error``),
    :math:`S_0` holds one reference amplitude per site and :math:`C` is the
    identity or, with ``tau_hours``, a within-site OU correlation
    :math:`\exp(-|t_i - t_j| / \tau_s)` with one time scale per site. Values
    are resolved by site label, so :math:`R_0` does not depend on row order.
    :math:`R_0` only screens the low-rank aggregation-error rank (see
    :class:`~openghg_inversions.observation_error.WhitenedRankRule`); it is
    not the likelihood. Use floored or prior-median amplitudes, never zero.

    Args:
        reference_sd: Positive reference amplitude: one value for every site
            or a mapping covering every observation site. The default is an
            independent 1 ppm.
        units: Mole-fraction units of ``reference_sd``, converted to the
            observation units.
        tau_hours: Optional positive OU time scale in hours: one value or a
            mapping covering every observation site. ``None`` gives an
            independent (diagonal) reference.

    Raises:
        ValueError: If a value is invalid.
    """

    reference_sd: float | Mapping[str, float] = 1.0
    units: str = "ppm"
    tau_hours: float | Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate the settings at construction."""
        _positive_site_values(self.reference_sd, "reference_sd")
        if self.tau_hours is not None:
            _positive_site_values(self.tau_hours, "tau_hours")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("Co2ReferenceCovariance units must be a non-empty string.")
        mole_fraction_unit_scale(self.units, context="Co2ReferenceCovariance units")

    def build(
        self,
        observation_error_sd: np.ndarray,
        sites: np.ndarray,
        *,
        observation_units: str,
        observation_times: np.ndarray | None = None,
    ) -> ReferenceMismatchCovariance:
        """Build :math:`R_0` in observation units for one set of observations.

        Args:
            observation_error_sd: Per-row reported observation error in
                ``observation_units``, finite and non-negative.
            sites: Per-row site labels.
            observation_units: Mole-fraction units of the observations.
            observation_times: Per-row ``datetime64`` times or hours.
                Required with ``tau_hours``.

        Returns:
            The reference covariance, with a JSON-safe ``description`` of the
            supplied settings and the amplitudes converted to observation
            units.

        Raises:
            ValueError: If shapes disagree, values are invalid, a mapping omits
                a site, or a site block is not positive definite.
        """
        labels = np.asarray(sites).astype(str)
        error = np.asarray(observation_error_sd, dtype=float)
        n = labels.size
        if labels.ndim != 1 or n == 0 or error.shape != (n,):
            raise ValueError("observation_error_sd and sites must be non-empty and one per row.")
        if not np.isfinite(error).all() or np.any(error < 0.0):
            raise ValueError("observation_error_sd must be finite and non-negative.")
        scale = mole_fraction_unit_scale(self.units, context="Co2ReferenceCovariance units") / (
            mole_fraction_unit_scale(observation_units, context="CO2 observations")
        )
        amplitude = _site_rows(self.reference_sd, labels, "reference_sd") * scale
        tau = None if self.tau_hours is None else _site_rows(self.tau_hours, labels, "tau_hours")
        unique_labels = np.unique(labels)
        description: dict[str, Any] = {
            "source": "Co2ReferenceCovariance",
            "reference_sd": _json_site_values(self.reference_sd),
            "units": self.units,
            "tau_hours": None if self.tau_hours is None else _json_site_values(self.tau_hours),
            "correlation": "independent" if tau is None else "within-site OU",
            "observation_units": observation_units,
            "reference_sd_observation_units_by_site": {
                str(label): float(amplitude[np.flatnonzero(labels == label)[0]]) for label in unique_labels
            },
            "observation_error": "reported observation-error variance included",
            "observation_error_sd_min": float(error.min()),
            "observation_error_sd_median": float(np.median(error)),
        }
        variance = np.square(error) + np.square(amplitude)
        if tau is None:
            return ReferenceMismatchCovariance(variance=variance, description=description)

        if observation_times is None:
            raise ValueError("An OU reference covariance requires observation times.")
        hours = _hours(observation_times)
        if hours.shape != (n,) or not np.isfinite(hours).all():
            raise ValueError("Observation times must be finite and one per row.")
        site_rows = []
        blocks = []
        for label in unique_labels:
            rows = np.flatnonzero(labels == label)
            block = np.square(amplitude[rows[0]]) * np.exp(
                -np.abs(hours[rows, None] - hours[None, rows]) / tau[rows[0]]
            )
            block[np.diag_indices(rows.size)] = variance[rows]
            site_rows.append(rows)
            blocks.append(block)
        return ReferenceMismatchCovariance(
            variance=variance, site_rows=tuple(site_rows), blocks=tuple(blocks), description=description
        )


def supplied_reference_covariance(
    values: np.ndarray,
    *,
    units: str,
    observation_units: str,
    unit_scale: float,
    name: str | None,
) -> ReferenceMismatchCovariance:
    """Wrap validated, materialized, observation-ordered supplied :math:`R_0` values.

    The caller has checked both axes against the canonical observation order
    and resolved ``unit_scale`` from ``units`` to observation units squared.
    The covariance is treated as one dense block.

    Args:
        values: Dense covariance values in the supplied units.
        units: Supplied covariance units.
        observation_units: Mole-fraction units of the observations.
        unit_scale: Multiplier converting ``units`` to observation units
            squared.
        name: Supplied array name, recorded if present.

    Returns:
        The reference covariance with a JSON-safe description.
    """
    converted = np.asarray(values, dtype=float) * float(unit_scale)
    n = converted.shape[0]
    return ReferenceMismatchCovariance(
        variance=np.diag(converted).copy(),
        site_rows=(np.arange(n),),
        blocks=(converted,),
        description={
            "source": "supplied labelled covariance",
            "name": name,
            "units": units,
            "observation_units": observation_units,
            "unit_scale_to_observation_units_squared": float(unit_scale),
            "structure": "dense",
        },
    )


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


def _hours(times: np.ndarray) -> np.ndarray:
    """Return times in hours from ``datetime64`` values or numbers."""
    values = np.asarray(times)
    if np.issubdtype(values.dtype, np.datetime64):
        nanoseconds = values.astype("datetime64[ns]")
        return np.asarray((nanoseconds - nanoseconds.min()) / np.timedelta64(1, "h"), dtype=float)
    return values.astype(float)


def _positive_site_values(value: object, name: str) -> None:
    """Validate a positive scalar or a non-empty mapping of positive site values."""
    if isinstance(value, Mapping):
        if not value or not all(
            isinstance(site, str) and _is_positive_real(item) for site, item in value.items()
        ):
            raise ValueError(
                f"Co2ReferenceCovariance {name} mapping must be non-empty with string sites and positive values."
            )
    elif not _is_positive_real(value):
        raise ValueError(f"Co2ReferenceCovariance {name} must be a positive finite number or a site mapping.")


def _json_site_values(value: float | Mapping[str, float]) -> float | dict[str, float]:
    """Return a scalar or site mapping as JSON-safe floats."""
    if isinstance(value, Mapping):
        return {str(site): float(item) for site, item in value.items()}
    return float(value)


def _site_rows(value: float | Mapping[str, float], sites: np.ndarray, name: str) -> np.ndarray:
    """Expand a scalar or site mapping to one value per observation row.

    Raises:
        ValueError: If the mapping omits an observation site.
    """
    labels = np.asarray(sites).astype(str)
    if not isinstance(value, Mapping):
        return np.full(labels.size, float(value))
    missing = sorted(str(label) for label in set(labels) - set(value))
    if missing:
        raise ValueError(f"{name} site mapping is missing observation site(s): {missing!r}.")
    return np.asarray([float(value[label]) for label in labels])


def require_reference_labels(
    reference: xr.DataArray,
    *,
    observation_index: Any,
    row_dims: tuple[str, ...],
) -> str:
    """Check a supplied labelled :math:`R_0` against the canonical observation order.

    Args:
        reference: Supplied observation-by-observation covariance.
        observation_index: Canonical observation index.
        row_dims: Accepted names for the row dimension.

    Returns:
        The column dimension name.

    Raises:
        ValueError: If the dimensions or either axis do not match.
    """
    if reference.ndim != 2 or reference.dims[0] not in row_dims or reference.dims[1] == reference.dims[0]:
        raise ValueError(
            f"reference_covariance must be observation-by-observation with rows on one of {list(row_dims)!r}."
        )
    row_dim, column_dim = (str(dim) for dim in reference.dims)
    for dim, axis in ((row_dim, "rows"), (column_dim, "columns")):
        if dim not in reference.indexes or not reference.indexes[dim].equals(observation_index):
            raise ValueError(f"reference_covariance {axis} must match the canonical observation order.")
    return column_dim


__all__ = ["Co2ReferenceCovariance"]
