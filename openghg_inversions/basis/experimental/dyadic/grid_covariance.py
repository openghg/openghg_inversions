"""Matrix-free separable covariance for a latitude-longitude grid.

The :class:`SeparableGridCovariance` operator builds dense one-dimensional
exponential factors and applies their Kronecker product without materializing
the native ``M x M`` covariance. Native vectors use explicit row-major
``(latitude, longitude)`` ordering: longitude varies fastest when a grid is
flattened. Optional marginal standard deviations produce
``diag(s) @ (K_lat kron K_lon) @ diag(s)``, and numeric class labels can block
covariance between grid locations in different classes.

Coordinate differences are used directly by the exponential kernels. Thus,
if latitude and longitude are supplied in degrees, length scales are also in
degrees. This experimental operator is not yet great-circle- or area-aware.

This implementation is adapted locally from ``verification-games``
``grid_covariance.py`` at commits ``338fa8825212`` and ``28e7bd624948``. The
source informed the separable application and class-blocking algebra; it is
not imported or copied as a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, init=False)
class SeparableGridCovariance:
    """Immutable separable exponential covariance operator.

    The native covariance is
    ``diag(s) @ (K_lat kron K_lon) @ diag(s)``. When ``class_labels`` are
    supplied, entries connecting different classes are zero. Missing or
    non-finite class labels are permitted only at grid locations that are
    inactive in a particular application (all supplied right-hand sides are
    zero there); outputs at those unclassified locations are zero. Class
    blocking performs one separable application per active class and is
    intended for low-cardinality categories such as land/ocean, not one label
    per native grid location.

    Arrays stored by the operator are detached copies marked read-only.

    Attributes:
        latitude: One-dimensional latitude coordinates.
        longitude: One-dimensional longitude coordinates.
        latitude_length_scale: Positive exponential length scale for latitude.
        longitude_length_scale: Positive exponential length scale for longitude.
        latitude_factor: Dense one-dimensional latitude covariance factor.
        longitude_factor: Dense one-dimensional longitude covariance factor.
        marginal_standard_deviations: Row-major marginal standard deviations.
        class_labels: Optional row-major numeric class labels.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    latitude_length_scale: float
    longitude_length_scale: float
    latitude_factor: np.ndarray
    longitude_factor: np.ndarray
    marginal_standard_deviations: np.ndarray
    class_labels: np.ndarray | None

    def __init__(
        self,
        latitude: npt.ArrayLike,
        longitude: npt.ArrayLike,
        *,
        latitude_length_scale: float,
        longitude_length_scale: float,
        marginal_standard_deviations: npt.ArrayLike | None = None,
        class_labels: npt.ArrayLike | None = None,
    ) -> None:
        """Build a separable covariance operator from grid coordinates.

        Args:
            latitude: Finite, non-empty one-dimensional latitude coordinates.
            longitude: Finite, non-empty one-dimensional longitude coordinates.
            latitude_length_scale: Positive finite latitude length scale in the
                same units as ``latitude``.
            longitude_length_scale: Positive finite longitude length scale in
                the same units as ``longitude``.
            marginal_standard_deviations: Optional finite non-negative values
                with shape ``(M,)`` or ``(n_latitude, n_longitude)``. The
                default is one at every grid location.
            class_labels: Optional numeric values with shape ``(M,)`` or
                ``(n_latitude, n_longitude)``. Missing and non-finite labels
                are checked when values are applied.

        Raises:
            ValueError: If coordinates, scales, marginal standard deviations,
                or class-label dimensions are invalid.
        """
        lat = _validated_coordinates(latitude, name="latitude")
        lon = _validated_coordinates(longitude, name="longitude")
        lat_scale = _positive_scale(latitude_length_scale, name="latitude_length_scale")
        lon_scale = _positive_scale(longitude_length_scale, name="longitude_length_scale")
        grid_shape = (lat.size, lon.size)
        n_grid_locations = lat.size * lon.size

        if marginal_standard_deviations is None:
            marginal_sds = np.ones(n_grid_locations, dtype=np.float64)
        else:
            marginal_sds = _validated_grid_values(
                marginal_standard_deviations,
                name="marginal_standard_deviations",
                grid_shape=grid_shape,
            )
            if not np.all(np.isfinite(marginal_sds)):
                raise ValueError("marginal_standard_deviations must be finite.")
            if np.any(marginal_sds < 0.0):
                raise ValueError("marginal_standard_deviations must be non-negative.")

        classes: np.ndarray | None = None
        if class_labels is not None:
            classes = _validated_grid_values(
                class_labels,
                name="class_labels",
                grid_shape=grid_shape,
            )

        object.__setattr__(self, "latitude", _readonly_copy(lat))
        object.__setattr__(self, "longitude", _readonly_copy(lon))
        object.__setattr__(self, "latitude_length_scale", lat_scale)
        object.__setattr__(self, "longitude_length_scale", lon_scale)
        object.__setattr__(
            self,
            "latitude_factor",
            _readonly_copy(_exponential_factor(lat, length_scale=lat_scale)),
        )
        object.__setattr__(
            self,
            "longitude_factor",
            _readonly_copy(_exponential_factor(lon, length_scale=lon_scale)),
        )
        object.__setattr__(self, "marginal_standard_deviations", _readonly_copy(marginal_sds))
        object.__setattr__(self, "class_labels", None if classes is None else _readonly_copy(classes))

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the native grid shape.

        Returns:
            ``(n_latitude, n_longitude)`` in row-major axis order.
        """
        return self.latitude.size, self.longitude.size

    @property
    def size(self) -> int:
        """Return the native state size.

        Returns:
            The number of row-major grid locations.
        """
        return self.latitude.size * self.longitude.size

    def apply(self, values: npt.ArrayLike) -> np.ndarray:
        """Apply the covariance to a vector or batched right-hand sides.

        Args:
            values: Finite array with shape ``(M,)`` or ``(M, n_rhs)``. Rows
                follow row-major ``(latitude, longitude)`` ordering.

        Returns:
            Covariance-applied values with the same shape as the input.

        Raises:
            ValueError: If values have invalid dimensions, contain non-finite
                entries, or use a missing/non-finite class at an active input
                grid location.
        """
        matrix, was_vector = _validated_right_hand_sides(values, size=self.size, name="values")
        active_inputs = np.any(matrix != 0.0, axis=1)
        scaled_values = self.marginal_standard_deviations[:, None] * matrix

        if self.class_labels is None:
            result = self._apply_unscaled(scaled_values)
        else:
            active_classes = self.class_labels[active_inputs]
            if not np.all(np.isfinite(active_classes)):
                raise ValueError("class_labels must be finite at every active input grid location.")
            result = np.zeros_like(scaled_values)
            for class_value in np.unique(active_classes):
                class_mask = self.class_labels == class_value
                class_values = np.where(class_mask[:, None], scaled_values, 0.0)
                result += np.where(class_mask[:, None], self._apply_unscaled(class_values), 0.0)

        result *= self.marginal_standard_deviations[:, None]
        return result[:, 0] if was_vector else result

    def projected_covariance(self, projection: npt.ArrayLike) -> np.ndarray:
        """Project the native covariance with a dense grid-to-target matrix.

        Args:
            projection: Finite matrix ``P`` with shape ``(M, K)``. Its rows use
                row-major ``(latitude, longitude)`` ordering.

        Returns:
            The dense projected covariance ``P.T @ K @ P`` with shape
            ``(K, K)``.

        Raises:
            ValueError: If the projection is non-finite or has incompatible
                dimensions, or class labels are invalid at active rows.
        """
        matrix = _validated_matrix(projection, name="projection", n_rows=self.size)
        return _symmetrize(matrix.T @ self.apply(matrix))

    def observation_cross_covariance(self, observation_operator: npt.ArrayLike) -> np.ndarray:
        """Return native-to-observation signal cross-covariance.

        Args:
            observation_operator: Finite observation design ``H`` with shape
                ``(N, M)``. Native columns use row-major
                ``(latitude, longitude)`` ordering.

        Returns:
            The dense cross-covariance ``K @ H.T`` with shape ``(M, N)``.

        Raises:
            ValueError: If the observation operator is non-finite or has
                incompatible dimensions, or class labels are invalid at active
                native inputs.
        """
        design = _validated_observation_operator(observation_operator, size=self.size)
        return self.apply(design.T)

    def observation_signal_covariance(self, observation_operator: npt.ArrayLike) -> np.ndarray:
        """Return observation-space signal covariance.

        Args:
            observation_operator: Finite observation design ``H`` with shape
                ``(N, M)``. Native columns use row-major
                ``(latitude, longitude)`` ordering.

        Returns:
            The dense signal covariance ``H @ K @ H.T`` with shape ``(N, N)``.

        Raises:
            ValueError: If the observation operator is non-finite or has
                incompatible dimensions, or class labels are invalid at active
                native inputs.
        """
        design = _validated_observation_operator(observation_operator, size=self.size)
        cross_covariance = self.apply(design.T)
        return _symmetrize(design @ cross_covariance)

    def _apply_unscaled(self, values: np.ndarray) -> np.ndarray:
        """Apply unscaled Kronecker factors to validated right-hand sides.

        Args:
            values: Finite matrix with shape ``(M, n_rhs)``.

        Returns:
            Factor-applied matrix with the same shape as ``values``.
        """
        n_latitude, n_longitude = self.grid_shape
        n_rhs = values.shape[1]
        grid_values = values.reshape(n_latitude, n_longitude, n_rhs)
        latitude_applied = np.einsum(
            "ij,jkr->ikr",
            self.latitude_factor,
            grid_values,
            optimize=True,
        )
        fully_applied = np.einsum(
            "ikr,lk->ilr",
            latitude_applied,
            self.longitude_factor,
            optimize=True,
        )
        return fully_applied.reshape(self.size, n_rhs)


def _validated_coordinates(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return validated finite one-dimensional coordinates."""
    coordinates = _real_float_array(values, name=name)
    if coordinates.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if coordinates.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must be finite.")
    return coordinates


def _positive_scale(value: float, *, name: str) -> float:
    """Return a validated positive finite length scale."""
    if isinstance(value, complex):
        raise ValueError(f"{name} must be real, positive, and finite.")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return scale


def _exponential_factor(coordinates: np.ndarray, *, length_scale: float) -> np.ndarray:
    """Build one exponential covariance factor from validated inputs."""
    distances = np.abs(coordinates[:, None] - coordinates[None, :])
    return np.exp(-distances / length_scale)


def _validated_grid_values(
    values: npt.ArrayLike,
    *,
    name: str,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Return row-major grid values after validating accepted shapes.

    Args:
        values: Numeric values shaped as the grid or as one flattened vector.
        name: Input name used in validation messages.
        grid_shape: Expected two-dimensional latitude-longitude shape.

    Returns:
        A one-dimensional float array in row-major order.

    Raises:
        ValueError: If values are complex, non-numeric, or incorrectly shaped.
    """
    array = _real_float_array(values, name=name)
    flattened_shape = (grid_shape[0] * grid_shape[1],)
    if array.shape not in (grid_shape, flattened_shape):
        raise ValueError(f"{name} must have shape {grid_shape} or {flattened_shape}; got {array.shape}.")
    return array.reshape(-1)


def _validated_right_hand_sides(
    values: npt.ArrayLike,
    *,
    size: int,
    name: str,
) -> tuple[np.ndarray, bool]:
    """Validate and normalize covariance right-hand sides.

    Args:
        values: Candidate vector or matrix of right-hand sides.
        size: Required number of native rows.
        name: Input name used in validation messages.

    Returns:
        A finite two-dimensional float array and whether the original input was
        one-dimensional.

    Raises:
        ValueError: If values are complex, non-finite, empty, or dimensionally
            incompatible.
    """
    array = _real_float_array(values, name=name)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (M,) or (M, n_rhs).")
    if array.shape[0] != size:
        raise ValueError(f"{name} must have {size} rows; got {array.shape[0]}.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one right-hand side.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array, was_vector


def _validated_matrix(values: npt.ArrayLike, *, name: str, n_rows: int) -> np.ndarray:
    """Validate a dense projection-like matrix.

    Args:
        values: Candidate dense matrix.
        name: Input name used in validation messages.
        n_rows: Required matrix row count.

    Returns:
        A finite, non-empty two-dimensional float array.

    Raises:
        ValueError: If values are complex, non-finite, empty, or dimensionally
            incompatible.
    """
    matrix = _real_float_array(values, name=name)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if matrix.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows; got {matrix.shape[0]}.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _validated_observation_operator(values: npt.ArrayLike, *, size: int) -> np.ndarray:
    """Return a finite observation operator with the native column count."""
    design = _real_float_array(values, name="observation_operator")
    if design.ndim != 2:
        raise ValueError("observation_operator must be two-dimensional.")
    if design.shape[0] == 0:
        raise ValueError("observation_operator must contain at least one observation.")
    if design.shape[1] != size:
        raise ValueError(f"observation_operator must have {size} columns; got {design.shape[1]}.")
    if not np.all(np.isfinite(design)):
        raise ValueError("observation_operator must be finite.")
    return design


def _real_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert numeric input to float without silently discarding complex parts."""
    candidate = np.asarray(values)
    if np.iscomplexobj(candidate):
        raise ValueError(f"{name} must be real.")
    try:
        return np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    """Return a detached read-only array copy."""
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove insignificant floating-point asymmetry from a covariance."""
    return 0.5 * (matrix + matrix.T)
