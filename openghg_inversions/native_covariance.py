"""Labelled matrix-free covariance actions on a native latitude/longitude grid.

The native scaling perturbation is denoted by ``x`` and its covariance by
``B = Cov(x)``.  A covariance action applies ``B`` to labelled right-hand
sides without constructing the dense native ``N x N`` matrix.  The initial
implementation uses

``B = sigma**2 * (K_lat kron K_lon)``,

where ``K_axis[i, j] = exp(-abs(q_i - q_j) / ell_axis)``.  Grid
arrays use row-major ``(latitude, longitude)`` vectorisation, so the action on
a field ``X`` is ``K_lat @ X @ K_lon.T``.  The default correlation length is
1.5 degrees; it is an introductory configurable value, not a universal
scientific constant.

Both :meth:`SeparableExponentialCovariance.apply` and
:meth:`SeparableExponentialCovariance.solve` preserve all input dimensions,
coordinates, name, and attributes. The action materialises the two
one-dimensional covariance factors and their Cholesky factors, rather than the
dense native covariance. Apply and solve eagerly convert each labelled
right-hand side to a NumPy array; they do not preserve lazy Dask execution.
Distances are coordinate-wise degrees, not geodesic or longitude-wrapped
distances. A missing coordinate ``units`` attr is interpreted as degrees for
compatibility with existing OGI grids.

``SeparableExponentialCovariance`` is an ordinary slotted, identity-based
object. Construction owns eager coordinate copies; coordinate properties
return borrowed arrays whose in-place mutation is unsupported. Scalar
configuration properties are read-only, so changed parameters require explicit
reconstruction.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions.borrowed import BorrowedDataArray, borrow

__all__ = [
    "InvertibleNativeCovarianceAction",
    "NativeCovarianceAction",
    "SeparableExponentialCovariance",
]


class NativeCovarianceAction(Protocol):
    """Structural interface for a labelled native covariance action."""

    @property
    def native_dims(self) -> tuple[str, ...]:
        """Native dimensions consumed by the action."""
        ...

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply ``B`` to labelled RHS arrays containing every native dimension.

        Args:
            rhs: Labelled array containing every native dimension.

        Returns:
            The covariance action with the dimensions, coordinates, name, and
            attributes of ``rhs`` preserved.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
        """
        ...


class InvertibleNativeCovarianceAction(NativeCovarianceAction, Protocol):
    """Native covariance action that can also solve systems in ``B``."""

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Return labelled ``B^-1 rhs`` while preserving the input layout.

        Args:
            rhs: Labelled array containing every native dimension.

        Returns:
            The covariance solve with the dimensions, coordinates, name, and
            attributes of ``rhs`` preserved.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
            numpy.linalg.LinAlgError: If the covariance factorization fails.
        """
        ...


class SeparableExponentialCovariance:
    """Separable exponential covariance over labelled latitude and longitude.

    Args:
        latitude: One-dimensional latitude coordinate in degrees.
        longitude: One-dimensional longitude coordinate in degrees.
        sigma: Strictly positive native-cell marginal standard deviation, in
            native-state units (dimensionless for RHIME scaling perturbations).
        correlation_length: Shared fallback correlation length in degrees.
        latitude_correlation_length: Optional latitude-specific length in degrees.
        longitude_correlation_length: Optional longitude-specific length in degrees.

    Notes:
        Construction copies and owns the two small coordinate vectors because
        cached covariance factors depend on them. Coordinate properties return
        borrowed arrays; callers must not mutate them in place. Configuration
        properties are read-only. Instances are ordinary slotted objects with
        identity-based equality and hashing.

    Raises:
        ValueError: If coordinates, units, or covariance parameters are invalid.
        numpy.linalg.LinAlgError: If a one-dimensional covariance factor is not
            numerically positive definite.
    """

    __slots__ = (
        "_correlation_length",
        "_latitude",
        "_latitude_cholesky",
        "_latitude_correlation_length",
        "_latitude_correlation_length_override",
        "_latitude_factor",
        "_longitude",
        "_longitude_cholesky",
        "_longitude_correlation_length",
        "_longitude_correlation_length_override",
        "_longitude_factor",
        "_sigma",
    )

    def __init__(
        self,
        latitude: xr.DataArray | None = None,
        longitude: xr.DataArray | None = None,
        sigma: float = 1.0,
        correlation_length: float = 1.5,
        latitude_correlation_length: float | None = None,
        longitude_correlation_length: float | None = None,
    ) -> None:
        """Validate constructor inputs and cache factors used by the actions.

        Coordinates are copied once. The resolved covariance parameters,
        separable factors, and Cholesky factors become owned eager state.

        Raises:
            ValueError: If coordinates, units, or covariance parameters are
                invalid.
        """
        latitude = _validate_coordinate(latitude, "latitude")
        longitude = _validate_coordinate(longitude, "longitude")
        if latitude.dims[0] == longitude.dims[0]:
            raise ValueError("latitude and longitude must use distinct dimension names")
        sigma = _positive_finite(sigma, "sigma")
        length = _positive_finite(correlation_length, "correlation_length")
        latitude_override = (
            None
            if latitude_correlation_length is None
            else _positive_finite(latitude_correlation_length, "latitude_correlation_length")
        )
        longitude_override = (
            None
            if longitude_correlation_length is None
            else _positive_finite(longitude_correlation_length, "longitude_correlation_length")
        )
        latitude_length = length if latitude_override is None else latitude_override
        longitude_length = length if longitude_override is None else longitude_override

        latitude_factor = _exponential_factor(latitude.values, latitude_length)
        longitude_factor = _exponential_factor(longitude.values, longitude_length)
        self._latitude = latitude
        self._longitude = longitude
        self._sigma = sigma
        self._correlation_length = length
        self._latitude_correlation_length_override = latitude_override
        self._longitude_correlation_length_override = longitude_override
        self._latitude_correlation_length = latitude_length
        self._longitude_correlation_length = longitude_length
        self._latitude_factor = latitude_factor
        self._longitude_factor = longitude_factor
        self._latitude_cholesky = cho_factor(latitude_factor, lower=True)
        self._longitude_cholesky = cho_factor(longitude_factor, lower=True)

    @property
    def latitude(self) -> BorrowedDataArray:
        """Borrow the owned latitude coordinate; in-place mutation is unsupported."""
        return borrow(self._latitude)

    @property
    def longitude(self) -> BorrowedDataArray:
        """Borrow the owned longitude coordinate; in-place mutation is unsupported."""
        return borrow(self._longitude)

    @property
    def sigma(self) -> float:
        """Read-only native-cell marginal standard deviation."""
        return self._sigma

    @property
    def correlation_length(self) -> float:
        """Read-only shared fallback correlation length in degrees."""
        return self._correlation_length

    @property
    def latitude_correlation_length(self) -> float:
        """Read-only resolved latitude correlation length in degrees."""
        return self._latitude_correlation_length

    @property
    def longitude_correlation_length(self) -> float:
        """Read-only resolved longitude correlation length in degrees."""
        return self._longitude_correlation_length

    @property
    def native_dims(self) -> tuple[str, str]:
        """Latitude and longitude dimension names in vectorisation order."""
        return (str(self._latitude.dims[0]), str(self._longitude.dims[0]))

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply ``B`` while preserving the labelled layout of ``rhs``.

        The right-hand side is eagerly converted to a NumPy array before the
        covariance action is evaluated.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B rhs`` with dimensions and coordinates identical to ``rhs``.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
        """
        matrix, original_dims, rhs_dims = self._validated_matrix(rhs)
        applied = self._apply_separable(matrix)
        return self._restore(applied, rhs, original_dims, rhs_dims)

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve ``B result = rhs`` without constructing the native matrix.

        The separable solve uses one-dimensional Cholesky factors.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B^-1 rhs`` with dimensions and coordinates identical to ``rhs``.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
        """
        matrix, original_dims, rhs_dims = self._validated_matrix(rhs)
        solved = self._solve_separable(matrix)
        return self._restore(solved, rhs, original_dims, rhs_dims)

    def _apply_separable(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the two one-dimensional covariance factors.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Separable covariance-applied values with the same shape.
        """
        left_applied = np.einsum("ij,jkr->ikr", self._latitude_factor, matrix)
        applied = np.einsum("ikr,lk->ilr", left_applied, self._longitude_factor)
        return applied * self.sigma**2

    def _solve_separable(self, matrix: np.ndarray) -> np.ndarray:
        """Solve the separable system using Cholesky factors.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Solved values with the same shape as ``matrix``.
        """
        n_lat, n_lon, n_rhs = matrix.shape
        latitude_solved = cho_solve(
            self._latitude_cholesky,
            matrix.reshape(n_lat, n_lon * n_rhs),
            check_finite=False,
        ).reshape(n_lat, n_lon, n_rhs)
        longitude_solved = (
            cho_solve(
                self._longitude_cholesky,
                latitude_solved.transpose(1, 0, 2).reshape(n_lon, n_lat * n_rhs),
                check_finite=False,
            )
            .reshape(n_lon, n_lat, n_rhs)
            .transpose(1, 0, 2)
        )
        return longitude_solved / self.sigma**2

    def _validated_matrix(self, rhs: xr.DataArray) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
        """Validate and eagerly reshape a labelled right-hand side.

        Args:
            rhs: Candidate right-hand side containing the native dimensions.

        Returns:
            A native-first NumPy matrix, the original dimension order, and the
            ordered non-native right-hand-side dimensions.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
        """
        if not isinstance(rhs, xr.DataArray):
            raise TypeError("rhs must be an xarray.DataArray")
        for dim, expected in zip(self.native_dims, (self._latitude, self._longitude), strict=True):
            if dim not in rhs.dims:
                raise ValueError(f"rhs is missing native dimension {dim!r}")
            if dim not in rhs.coords:
                raise ValueError(f"rhs is missing coordinate labels for native dimension {dim!r}")
            actual_values = np.asarray(rhs.coords[dim].values)
            if actual_values.shape != expected.shape or not np.array_equal(actual_values, expected.values):
                raise ValueError(f"rhs coordinate {dim!r} does not align with the covariance grid")
        original_dims = tuple(str(dim) for dim in rhs.dims)
        rhs_dims = tuple(dim for dim in original_dims if dim not in self.native_dims)
        transposed = rhs.transpose(*self.native_dims, *rhs_dims)
        values = np.asarray(transposed.values)
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("rhs values must be numeric")
        if not np.all(np.isfinite(values)):
            raise ValueError("rhs values must be finite (no NaN or infinity)")
        n_lat = self._latitude.size
        n_lon = self._longitude.size
        n_rhs = int(np.prod([transposed.sizes[dim] for dim in rhs_dims], dtype=np.intp))
        return (
            np.asarray(values, dtype=np.result_type(values.dtype, np.float64)).reshape(n_lat, n_lon, n_rhs),
            original_dims,
            rhs_dims,
        )

    def _restore(
        self,
        values: np.ndarray,
        template: xr.DataArray,
        original_dims: tuple[str, ...],
        rhs_dims: tuple[str, ...],
    ) -> xr.DataArray:
        """Restore eager native-first results to a labelled input layout.

        Args:
            values: Result values shaped as native axes followed by flattened
                right-hand-side axes.
            template: Input array supplying coordinates, name, and attributes.
            original_dims: Original input dimension order.
            rhs_dims: Ordered non-native right-hand-side dimensions.

        Returns:
            A labelled array matching the template layout and metadata.
        """
        shape = tuple(template.sizes[dim] for dim in (*self.native_dims, *rhs_dims))
        result = xr.DataArray(
            values.reshape(shape),
            dims=(*self.native_dims, *rhs_dims),
            coords={name: coordinate for name, coordinate in template.coords.items()},
            name=template.name,
            attrs=template.attrs,
        )
        return result.transpose(*original_dims)


def _validate_coordinate(coordinate: xr.DataArray | None, axis_name: str) -> xr.DataArray:
    """Validate one native coordinate and return an owned copy.

    Args:
        coordinate: Candidate one-dimensional labelled coordinate.
        axis_name: Human-readable axis name used for validation and errors.

    Returns:
        An eager owned copy with an explicit self-coordinate.

    Raises:
        ValueError: If the coordinate is not one-dimensional and non-empty,
            contains non-finite or duplicate values, or has non-degree units.
    """
    if not isinstance(coordinate, xr.DataArray) or coordinate.ndim != 1 or len(coordinate.dims) != 1:
        raise ValueError(f"{axis_name} must be a one-dimensional labelled coordinate")
    if coordinate.size == 0:
        raise ValueError(f"{axis_name} coordinate must contain at least one value")
    # This is the explicit eager ownership boundary for the small coordinate
    # vector. Compute the data and coordinates together, then take an
    # independent copy so cached factors cannot be invalidated through the
    # constructor argument.
    coordinate = coordinate.compute().copy(deep=True)
    dim = coordinate.dims[0]
    values = np.asarray(coordinate.data)
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{axis_name} coordinates must be finite real numeric values")
    if dim in coordinate.coords:
        labels = np.asarray(coordinate.coords[dim].data)
        if labels.shape != values.shape or not np.array_equal(labels, values):
            raise ValueError(f"{axis_name} coordinate data must match its {dim!r} dimension labels")
    else:
        coordinate = coordinate.assign_coords({dim: values})
    if np.unique(values).size != values.size:
        raise ValueError(f"{axis_name} coordinates must be unique; duplicate values were found")
    units = str(coordinate.attrs.get("units", "")).strip().lower()
    allowed_units = {"", "degree", "degrees"}
    if axis_name == "latitude":
        allowed_units.update({"degrees_north", "degree_north"})
    else:
        allowed_units.update({"degrees_east", "degree_east"})
    if units not in allowed_units:
        raise ValueError(f"{axis_name} coordinate units must be degrees, got {units!r}")
    return coordinate


def _positive_finite(value: float, name: str) -> float:
    """Resolve a covariance parameter to a positive finite float.

    Args:
        value: Candidate numeric value.
        name: Parameter name used in validation errors.

    Returns:
        The validated floating-point value.

    Raises:
        TypeError: If ``value`` cannot be converted to a float.
        ValueError: If the resolved value is non-finite or not strictly positive.
    """
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return resolved


def _exponential_factor(coordinates: np.ndarray, length_scale: float) -> np.ndarray:
    """Construct one exponential covariance factor.

    Args:
        coordinates: Validated coordinate values for one native axis.
        length_scale: Positive correlation length in coordinate units.

    Returns:
        The dense one-dimensional exponential covariance factor.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    return np.exp(-np.abs(values[:, np.newaxis] - values[np.newaxis, :]) / length_scale)
