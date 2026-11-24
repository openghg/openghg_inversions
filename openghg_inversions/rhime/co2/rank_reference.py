"""Reference mismatch covariance used to choose the CO2 aggregation-error rank."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions._labelled_matrices import matrix_column_dim, square_matrix
from openghg_inversions.observation_error import _is_positive_real


@dataclass(frozen=True)
class Co2ReferenceCovariance:
    r"""Settings for a CO2 reference mismatch covariance :math:`R_0`.

    :meth:`covariance` builds the dense labelled covariance

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

    Keep the white-noise floor the likelihood has. The only white part of
    this :math:`R_0` is the reported error :math:`e`. When that is near zero,
    an OU reference is weak at high frequency: with hourly data and a 5 h
    time scale, adjacent hours have correlation 0.82 and the high-frequency
    directions get only about 10% of the site variance. The stored diagonal
    tail spreads discarded variance evenly across frequencies, so its error
    is magnified in those directions. The spectral dropped-mode KL then
    understates the stored-representation KL (both are recorded) by several
    times. If the likelihood adds a white floor that ``mf_error`` does not
    carry, such as a minimum error or a fixed model-error term, pass
    ``reference_covariance`` as a labelled covariance that includes it.

    Args:
        reference_sd: Positive reference amplitude: one value for every site
            or a mapping covering every observation site. There is no default:
            the rank depends strongly on this choice, so the caller states it,
            for example the run's floored or prior-median site amplitudes.
        units: Mole-fraction units of ``reference_sd``, converted to the
            observation units.
        tau_hours: Optional positive OU time scale in hours: one value or a
            mapping covering every observation site. ``None`` gives an
            independent (diagonal) reference.

    Raises:
        ValueError: If a value is invalid.
    """

    reference_sd: float | Mapping[str, float]
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

    def covariance(self, observations: xr.DataArray, observation_error: xr.DataArray) -> xr.DataArray:
        """Return :math:`R_0` as a dense labelled covariance in observation units squared.

        The result uses the same labelled-covariance contract as a
        caller-supplied ``reference_covariance``: rows on the observation
        dimension of ``observations`` with its coordinates, columns on a
        distinct companion dimension, ``units`` of the observation units
        squared, and a JSON ``description`` attribute recording the supplied
        settings and the amplitudes converted to observation units.

        Args:
            observations: One-dimensional observations with units and a
                ``site`` coordinate, plus a ``time`` coordinate with
                ``tau_hours``.
            observation_error: Reported observation error on the same axis, in
                the observation units, finite and non-negative.

        Returns:
            The labelled ``n x n`` reference covariance.

        Raises:
            ValueError: If coordinates, units or values are missing or
                invalid, or a site mapping omits a site.
        """
        if observations.ndim != 1:
            raise ValueError("Co2ReferenceCovariance needs one-dimensional observations.")
        dim = str(observations.dims[0])
        observation_units = observations.attrs.get("units")
        if not isinstance(observation_units, str) or not observation_units.strip():
            raise ValueError("Co2ReferenceCovariance needs observation units.")
        if observation_error.dims != (dim,) or (
            dim in observations.indexes
            and not observation_error.indexes[dim].equals(observations.indexes[dim])
        ):
            raise ValueError("Co2ReferenceCovariance observation error must share the observation axis.")
        if "site" not in observations.coords or (
            self.tau_hours is not None and "time" not in observations.coords
        ):
            raise ValueError(
                "Co2ReferenceCovariance needs a 'site' coordinate (and, with tau_hours, 'time')."
            )
        labels = np.asarray(observations["site"].values).astype(str)
        error = np.asarray(observation_error.values, dtype=float)
        if not np.isfinite(error).all() or np.any(error < 0.0):
            raise ValueError("Co2ReferenceCovariance observation error must be finite and non-negative.")
        scale = mole_fraction_unit_scale(self.units, context="Co2ReferenceCovariance units") / (
            mole_fraction_unit_scale(observation_units, context="CO2 observations")
        )
        amplitude = _site_rows(self.reference_sd, labels, "reference_sd") * scale
        values = np.diag(np.square(error) + np.square(amplitude))
        if self.tau_hours is not None:
            tau = _site_rows(self.tau_hours, labels, "tau_hours")
            hours = _hours(np.asarray(observations["time"].values))
            if not np.isfinite(hours).all():
                raise ValueError("Co2ReferenceCovariance observation times must be finite.")
            for label in np.unique(labels):
                rows = np.flatnonzero(labels == label)
                block = np.square(amplitude[rows[0]]) * np.exp(
                    -np.abs(hours[rows, None] - hours[None, rows]) / tau[rows[0]]
                )
                block[np.diag_indices(rows.size)] = values[rows, rows]
                values[np.ix_(rows, rows)] = block
        description = {
            "source": "Co2ReferenceCovariance",
            "reference_sd": _json_site_values(self.reference_sd),
            "units": self.units,
            "tau_hours": None if self.tau_hours is None else _json_site_values(self.tau_hours),
            "correlation": "independent" if self.tau_hours is None else "within-site OU",
            "reference_sd_observation_units_by_site": {
                str(label): float(amplitude[np.flatnonzero(labels == label)[0]])
                for label in np.unique(labels)
            },
            "observation_error": "reported observation-error variance included",
            "observation_error_sd_min": float(error.min()),
            "observation_error_sd_median": float(np.median(error)),
        }
        return square_matrix(
            values,
            observations,
            row_dim=dim,
            column_dim=matrix_column_dim(dim, observations.dims),
            name="reference_covariance",
            attrs={"units": f"({observation_units})**2", "description": json.dumps(description)},
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
