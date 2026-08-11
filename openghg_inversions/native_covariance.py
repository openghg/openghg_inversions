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
coordinates, name, and attributes.  Only the two one-dimensional factors are
materialised.  Optional native-grid class labels replace ``B`` by the blocked
action ``B_class = sum_c M_c B M_c``, where ``M_c`` is the diagonal indicator
for class ``c``.  This prevents covariance between cells in different classes
without constructing either ``M_c`` or the dense native covariance.
Unblocked solves use the separable Cholesky factors; multi-class solves use
matrix-free conjugate gradients. Distances are coordinate-wise degrees, not
geodesic or longitude-wrapped distances. A missing coordinate ``units`` attr
is interpreted as degrees for compatibility with existing OGI grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
import xarray as xr

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

        All non-native dimensions and their labels must be preserved.
        """
        ...


class InvertibleNativeCovarianceAction(NativeCovarianceAction, Protocol):
    """Native covariance action that can also solve systems in ``B``."""

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Return labelled ``B^-1 rhs`` while preserving non-native dimensions."""
        ...


@dataclass(frozen=True, slots=True)
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
        class_labels: Optional native-grid labels defining spatial covariance
            blocks. Cells with different labels have zero cross-covariance.

    Raises:
        ValueError: If coordinates, units, or covariance parameters are invalid.
        numpy.linalg.LinAlgError: If a class-blocked iterative solve does not converge.
    """

    latitude: xr.DataArray
    longitude: xr.DataArray
    sigma: float = 1.0
    correlation_length: float = 1.5
    latitude_correlation_length: float | None = None
    longitude_correlation_length: float | None = None
    class_labels: xr.DataArray | None = None
    _latitude_factor: np.ndarray = field(init=False, repr=False, compare=False)
    _longitude_factor: np.ndarray = field(init=False, repr=False, compare=False)
    _latitude_cholesky: tuple[np.ndarray, bool] = field(init=False, repr=False, compare=False)
    _longitude_cholesky: tuple[np.ndarray, bool] = field(init=False, repr=False, compare=False)
    _class_masks: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)

    schema = "openghg_inversions.separable_exponential_covariance"
    schema_version = 1

    def __post_init__(self) -> None:
        latitude = _validate_coordinate(self.latitude, "latitude")
        longitude = _validate_coordinate(self.longitude, "longitude")
        if latitude.dims[0] == longitude.dims[0]:
            raise ValueError("latitude and longitude must use distinct dimension names")
        sigma = _positive_finite(self.sigma, "sigma")
        length = _positive_finite(self.correlation_length, "correlation_length")
        latitude_length = _positive_finite(
            length if self.latitude_correlation_length is None else self.latitude_correlation_length,
            "latitude_correlation_length",
        )
        longitude_length = _positive_finite(
            length if self.longitude_correlation_length is None else self.longitude_correlation_length,
            "longitude_correlation_length",
        )

        latitude_factor = _exponential_factor(latitude.values, latitude_length)
        longitude_factor = _exponential_factor(longitude.values, longitude_length)
        class_labels, class_masks = _validate_class_labels(
            self.class_labels,
            latitude,
            longitude,
        )
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "correlation_length", length)
        object.__setattr__(self, "latitude_correlation_length", latitude_length)
        object.__setattr__(self, "longitude_correlation_length", longitude_length)
        object.__setattr__(self, "class_labels", class_labels)
        object.__setattr__(self, "_latitude_factor", latitude_factor)
        object.__setattr__(self, "_longitude_factor", longitude_factor)
        object.__setattr__(self, "_latitude_cholesky", cho_factor(latitude_factor, lower=True))
        object.__setattr__(self, "_longitude_cholesky", cho_factor(longitude_factor, lower=True))
        object.__setattr__(self, "_class_masks", class_masks)

    @property
    def native_dims(self) -> tuple[str, str]:
        """Latitude and longitude dimension names in vectorisation order."""
        return (str(self.latitude.dims[0]), str(self.longitude.dims[0]))

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply ``B`` while preserving the labelled layout of ``rhs``.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B rhs`` with dimensions and coordinates identical to ``rhs``.
        """
        matrix, original_dims, rhs_dims = self._validated_matrix(rhs)
        applied = self._apply_matrix(matrix)
        return self._restore(applied, rhs, original_dims, rhs_dims)

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve ``B result = rhs`` without constructing the native matrix.

        The separable case uses one-dimensional Cholesky factors. A covariance
        with multiple classes uses conjugate gradients with a matrix-free
        blocked covariance action.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B^-1 rhs`` with dimensions and coordinates identical to ``rhs``.
        """
        matrix, original_dims, rhs_dims = self._validated_matrix(rhs)
        if len(self._class_masks) <= 1:
            solved = self._solve_separable(matrix)
        else:
            solved = self._solve_class_blocked(matrix)
        return self._restore(solved, rhs, original_dims, rhs_dims)

    def to_dataset(self) -> xr.Dataset:
        """Serialize reproducible coordinates and resolved configuration."""
        dataset = xr.Dataset(
            coords={
                self.native_dims[0]: self.latitude,
                self.native_dims[1]: self.longitude,
            },
            attrs={
                "schema": self.schema,
                "schema_version": self.schema_version,
                "latitude_dim": self.native_dims[0],
                "longitude_dim": self.native_dims[1],
                "sigma": self.sigma,
                "correlation_length_degrees": self.correlation_length,
                "latitude_correlation_length_degrees": self.latitude_correlation_length,
                "longitude_correlation_length_degrees": self.longitude_correlation_length,
                "class_blocked": self.class_labels is not None,
                "class_labels_name": ""
                if self.class_labels is None or self.class_labels.name is None
                else str(self.class_labels.name),
            },
        )
        if self.class_labels is not None:
            dataset["class_labels"] = self.class_labels
        return dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> SeparableExponentialCovariance:
        """Restore a covariance action from :meth:`to_dataset` output.

        Args:
            dataset: Versioned covariance configuration dataset.

        Returns:
            Reconstructed covariance action.
        """
        if dataset.attrs.get("schema") != cls.schema:
            raise ValueError(f"Expected covariance schema {cls.schema!r}")
        if dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported covariance schema version {dataset.attrs.get('schema_version')!r}")
        lat_dim = str(dataset.attrs.get("latitude_dim", ""))
        lon_dim = str(dataset.attrs.get("longitude_dim", ""))
        if lat_dim not in dataset.coords or lon_dim not in dataset.coords:
            raise ValueError("Serialized covariance is missing latitude or longitude coordinates")
        class_labels = dataset.get("class_labels")
        if class_labels is not None:
            serialized_name = str(dataset.attrs.get("class_labels_name", ""))
            class_labels = class_labels.rename(serialized_name or None)
        return cls(
            latitude=dataset.coords[lat_dim],
            longitude=dataset.coords[lon_dim],
            sigma=float(dataset.attrs["sigma"]),
            correlation_length=float(dataset.attrs["correlation_length_degrees"]),
            latitude_correlation_length=float(
                dataset.attrs.get(
                    "latitude_correlation_length_degrees",
                    dataset.attrs["correlation_length_degrees"],
                )
            ),
            longitude_correlation_length=float(
                dataset.attrs.get(
                    "longitude_correlation_length_degrees",
                    dataset.attrs["correlation_length_degrees"],
                )
            ),
            class_labels=class_labels,
        )

    def _apply_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the configured unblocked or class-blocked covariance."""
        if not self._class_masks:
            return self._apply_separable(matrix)
        result = np.zeros_like(matrix, dtype=np.result_type(matrix.dtype, np.float64))
        for mask in self._class_masks:
            masked = matrix * mask[..., np.newaxis]
            result += self._apply_separable(masked) * mask[..., np.newaxis]
        return result

    def _apply_separable(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the two one-dimensional covariance factors."""
        left_applied = np.einsum("ij,jkr->ikr", self._latitude_factor, matrix)
        applied = np.einsum("ikr,lk->ilr", left_applied, self._longitude_factor)
        return applied * self.sigma**2

    def _solve_separable(self, matrix: np.ndarray) -> np.ndarray:
        """Solve the unblocked separable system using Cholesky factors."""
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

    def _solve_class_blocked(self, matrix: np.ndarray) -> np.ndarray:
        """Solve a class-blocked system with matrix-free conjugate gradients."""
        n_lat, n_lon, n_rhs = matrix.shape
        native_size = n_lat * n_lon

        def matvec(vector: np.ndarray) -> np.ndarray:
            """Apply the blocked covariance to one flattened right-hand side."""
            native = vector.reshape(n_lat, n_lon, 1)
            return self._apply_matrix(native).reshape(native_size)

        operator = LinearOperator(
            (native_size, native_size),
            matvec,
        )
        solved = np.empty_like(matrix, dtype=np.result_type(matrix.dtype, np.float64))
        for rhs_index in range(n_rhs):
            solution, info = cg(
                operator,
                matrix[..., rhs_index].reshape(native_size),
                rtol=1e-12,
                atol=0.0,
                maxiter=max(100, 10 * native_size),
            )
            if info != 0:
                reason = "did not converge" if info > 0 else "failed with an illegal input or breakdown"
                raise np.linalg.LinAlgError(f"Class-blocked covariance solve {reason} (CG info={info})")
            solved[..., rhs_index] = solution.reshape(n_lat, n_lon)
        return solved

    def _validated_matrix(self, rhs: xr.DataArray) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
        if not isinstance(rhs, xr.DataArray):
            raise TypeError("rhs must be an xarray.DataArray")
        for dim, expected in zip(self.native_dims, (self.latitude, self.longitude), strict=True):
            if dim not in rhs.dims:
                raise ValueError(f"rhs is missing native dimension {dim!r}")
            if dim not in rhs.coords:
                raise ValueError(f"rhs is missing coordinate labels for native dimension {dim!r}")
            actual_values = np.asarray(rhs.coords[dim].values)
            if actual_values.shape != expected.shape or not np.array_equal(actual_values, expected.values):
                raise ValueError(f"rhs coordinate {dim!r} does not align with the covariance grid")
        values = np.asarray(rhs.values)
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("rhs values must be numeric")
        if not np.all(np.isfinite(values)):
            raise ValueError("rhs values must be finite (no NaN or infinity)")
        original_dims = tuple(str(dim) for dim in rhs.dims)
        rhs_dims = tuple(dim for dim in original_dims if dim not in self.native_dims)
        transposed = rhs.transpose(*self.native_dims, *rhs_dims)
        n_lat = self.latitude.size
        n_lon = self.longitude.size
        return (
            np.asarray(transposed.values, dtype=np.result_type(values.dtype, np.float64)).reshape(
                n_lat, n_lon, -1
            ),
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
        shape = tuple(template.sizes[dim] for dim in (*self.native_dims, *rhs_dims))
        result = xr.DataArray(
            values.reshape(shape),
            dims=(*self.native_dims, *rhs_dims),
            coords={name: coordinate for name, coordinate in template.coords.items()},
            name=template.name,
            attrs=template.attrs,
        )
        return result.transpose(*original_dims)


def _validate_coordinate(coordinate: xr.DataArray, axis_name: str) -> xr.DataArray:
    if not isinstance(coordinate, xr.DataArray) or coordinate.ndim != 1 or len(coordinate.dims) != 1:
        raise ValueError(f"{axis_name} must be a one-dimensional labelled coordinate")
    dim = coordinate.dims[0]
    if dim not in coordinate.coords:
        coordinate = coordinate.assign_coords({dim: coordinate.values})
    values = np.asarray(coordinate.values)
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"{axis_name} coordinates must be finite numeric values")
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
    return coordinate.copy(deep=True)


def _validate_class_labels(
    class_labels: xr.DataArray | None,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> tuple[xr.DataArray | None, tuple[np.ndarray, ...]]:
    """Validate class labels and construct boolean masks in native-grid order.

    Args:
        class_labels: Optional array assigning every native grid cell to a class.
        latitude: Validated latitude coordinate.
        longitude: Validated longitude coordinate.

    Returns:
        A defensive copy of the labels and one native-grid boolean mask per
        class. The empty mask tuple denotes an unblocked covariance.

    Raises:
        ValueError: If dimensions, coordinates, or label values are invalid.
    """
    if class_labels is None:
        return None, ()
    if not isinstance(class_labels, xr.DataArray):
        raise ValueError("class_labels must be an xarray.DataArray")

    native_dims = (latitude.dims[0], longitude.dims[0])
    if set(class_labels.dims) != set(native_dims) or class_labels.ndim != 2:
        raise ValueError(f"class_labels must have exactly the native dimensions {native_dims!r}")
    for dim, expected in zip(native_dims, (latitude, longitude), strict=True):
        if dim not in class_labels.coords:
            raise ValueError(f"class_labels is missing coordinate labels for native dimension {dim!r}")
        actual = np.asarray(class_labels.coords[dim].values)
        if actual.shape != expected.shape or not np.array_equal(actual, expected.values):
            raise ValueError(f"class_labels coordinate {dim!r} does not align with the covariance grid")
    if bool(class_labels.isnull().any().item()):
        raise ValueError("class_labels must assign a non-null class to every native grid cell")

    labels = class_labels.transpose(*native_dims).copy(deep=True)
    values = np.asarray(labels.values)
    try:
        unique_values = np.unique(values)
    except TypeError as error:
        raise ValueError("class_labels values must be mutually comparable scalar labels") from error
    masks = tuple(values == value for value in unique_values)
    return labels, masks


def _positive_finite(value: float, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return resolved


def _exponential_factor(coordinates: np.ndarray, length_scale: float) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    return np.exp(-np.abs(values[:, np.newaxis] - values[np.newaxis, :]) / length_scale)
