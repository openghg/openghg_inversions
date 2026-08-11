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
coordinates, name, and attributes.  An unblocked action materialises the two
one-dimensional covariance factors and their Cholesky factors, rather than the
dense native covariance. Optional native-grid class labels replace ``B`` by
the blocked action ``B_class = sum_c M_c B M_c``, where ``M_c`` is the diagonal
indicator for class ``c``. A blocked action additionally stores one native-grid
boolean mask per class, but does not construct the diagonal ``M_c`` matrices or
the dense native covariance. Apply and solve eagerly convert each labelled
right-hand side to a NumPy array; they do not preserve lazy Dask execution.
Unblocked solves use the separable Cholesky factors; multi-class solves use
matrix-free conjugate gradients. Distances are coordinate-wise degrees, not
geodesic or longitude-wrapped distances. A missing coordinate ``units`` attr
is interpreted as degrees for compatibility with existing OGI grids.

``SeparableExponentialCovariance`` is an ordinary slotted, identity-based
object. Construction owns eager coordinate and class-label copies; array
properties return borrowed values whose in-place mutation is unsupported.
Scalar configuration properties are read-only, so changed parameters require
explicit reconstruction.
"""

from __future__ import annotations

import inspect
import json
from typing import Callable, Protocol, cast

import numpy as np
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator, cg  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions.borrowed import BorrowedDataArray, borrow

_CG_RELATIVE_TOLERANCE = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
_NATIVE_SERIALIZED_VARIABLE_NAMES = {"class_label_encoded"}

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
            numpy.linalg.LinAlgError: If an iterative solve fails.
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
        class_labels: Optional native-grid labels defining spatial covariance
            blocks. Cells with different labels have zero cross-covariance.

    Notes:
        Construction copies and owns the two small coordinate vectors and any
        class labels because cached factors and masks depend on them. Array
        properties return borrowed values; callers must not mutate them in
        place. Configuration properties are read-only. Instances are ordinary
        slotted objects with identity-based equality and hashing.

    Raises:
        ValueError: If coordinates, units, or covariance parameters are invalid.
        numpy.linalg.LinAlgError: If a one-dimensional covariance factor is not
            numerically positive definite.
    """

    __slots__ = (
        "_class_labels",
        "_class_masks",
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

    schema = "openghg_inversions.separable_exponential_covariance"
    schema_version = 1

    def __init__(
        self,
        latitude: xr.DataArray | None = None,
        longitude: xr.DataArray | None = None,
        sigma: float = 1.0,
        correlation_length: float = 1.5,
        latitude_correlation_length: float | None = None,
        longitude_correlation_length: float | None = None,
        class_labels: xr.DataArray | None = None,
    ) -> None:
        """Validate constructor inputs and cache factors used by the actions.

        Coordinates and class labels are copied once. The resolved covariance
        parameters, separable factors, Cholesky factors, and class masks become
        owned eager state.

        Raises:
            ValueError: If coordinates, units, covariance parameters, or class
                labels are invalid.
        """
        latitude = _validate_coordinate(latitude, "latitude")
        longitude = _validate_coordinate(longitude, "longitude")
        if latitude.dims[0] == longitude.dims[0]:
            raise ValueError("latitude and longitude must use distinct dimension names")
        reserved_dims = _NATIVE_SERIALIZED_VARIABLE_NAMES.intersection(
            (str(latitude.dims[0]), str(longitude.dims[0]))
        )
        if reserved_dims:
            reserved = sorted(reserved_dims)[0]
            raise ValueError(f"native dimension {reserved!r} is reserved by the serialized schema")
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
        class_labels, class_masks = _validate_class_labels(class_labels, latitude, longitude)
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
        self._class_labels = class_labels
        self._class_masks = class_masks

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
    def latitude_correlation_length_explicit(self) -> bool:
        """Whether the latitude length was configured independently."""
        return self._latitude_correlation_length_override is not None

    @property
    def longitude_correlation_length_explicit(self) -> bool:
        """Whether the longitude length was configured independently."""
        return self._longitude_correlation_length_override is not None

    @property
    def class_labels(self) -> BorrowedDataArray | None:
        """Borrow the owned class labels, if any; in-place mutation is unsupported."""
        labels = self._class_labels
        return None if labels is None else borrow(labels)

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

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric or non-finite.
            numpy.linalg.LinAlgError: If a class-blocked iterative solve does
                not converge.
        """
        matrix, original_dims, rhs_dims = self._validated_matrix(rhs)
        if len(self._class_masks) <= 1:
            solved = self._solve_separable(matrix)
        else:
            solved = self._solve_class_blocked(matrix)
        return self._restore(solved, rhs, original_dims, rhs_dims)

    def to_dataset(self) -> xr.Dataset:
        """Serialize reproducible coordinates and resolved configuration.

        Returns:
            A versioned dataset containing the native coordinates, resolved
            covariance parameters, and optional class labels. Boolean blocked
            state is represented as a NetCDF-safe ``0`` or ``1`` attribute.
        """
        dataset = xr.Dataset(
            coords={
                self.native_dims[0]: self._latitude.copy(deep=True),
                self.native_dims[1]: self._longitude.copy(deep=True),
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
                "latitude_correlation_length_explicit": int(self.latitude_correlation_length_explicit),
                "longitude_correlation_length_explicit": int(self.longitude_correlation_length_explicit),
                "class_blocked": int(self._class_labels is not None),
                "class_label_encoding": "tagged_json_v1" if self._class_labels is not None else "",
                "class_labels_name": ""
                if self._class_labels is None or self._class_labels.name is None
                else str(self._class_labels.name),
                "class_labels_attrs": "{}"
                if self._class_labels is None
                else json.dumps(self._class_labels.attrs, sort_keys=True, default=_json_default),
            },
        )
        if self._class_labels is not None:
            encoded = np.frompyfunc(_encode_class_label, 1, 1)(self._class_labels.values)
            dataset["class_label_encoded"] = self._class_labels.copy(data=np.asarray(encoded, dtype=str))
        return dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> SeparableExponentialCovariance:
        """Restore a covariance action from :meth:`to_dataset` output.

        Args:
            dataset: Versioned covariance configuration dataset.

        Returns:
            Reconstructed covariance action.

        Raises:
            ValueError: If the schema or version is unsupported, required
                coordinates are missing, or serialized constructor values are
                invalid.
            KeyError: If a required covariance parameter attribute is missing.
        """
        if dataset.attrs.get("schema") != cls.schema:
            raise ValueError(f"Expected covariance schema {cls.schema!r}")
        if dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported covariance schema version {dataset.attrs.get('schema_version')!r}")
        lat_dim = str(dataset.attrs.get("latitude_dim", ""))
        lon_dim = str(dataset.attrs.get("longitude_dim", ""))
        if lat_dim not in dataset.coords or lon_dim not in dataset.coords:
            raise ValueError("Serialized covariance is missing latitude or longitude coordinates")
        try:
            latitude_length_explicit = _serialized_boolean(
                dataset.attrs["latitude_correlation_length_explicit"],
                "latitude_correlation_length_explicit",
            )
            longitude_length_explicit = _serialized_boolean(
                dataset.attrs["longitude_correlation_length_explicit"],
                "longitude_correlation_length_explicit",
            )
        except KeyError as error:
            raise ValueError("Serialized covariance has invalid correlation-length metadata") from error
        try:
            class_blocked = _serialized_boolean(dataset.attrs["class_blocked"], "class_blocked")
        except KeyError as error:
            raise ValueError("Serialized covariance has an invalid class_blocked flag") from error
        correlation_length = _positive_finite(
            float(dataset.attrs["correlation_length_degrees"]),
            "serialized correlation_length_degrees",
        )
        latitude_length = _positive_finite(
            float(dataset.attrs["latitude_correlation_length_degrees"]),
            "serialized latitude_correlation_length_degrees",
        )
        longitude_length = _positive_finite(
            float(dataset.attrs["longitude_correlation_length_degrees"]),
            "serialized longitude_correlation_length_degrees",
        )
        if not latitude_length_explicit and latitude_length != correlation_length:
            raise ValueError("Serialized implicit latitude correlation length contradicts the fallback")
        if not longitude_length_explicit and longitude_length != correlation_length:
            raise ValueError("Serialized implicit longitude correlation length contradicts the fallback")
        has_class_labels = "class_label_encoded" in dataset
        if class_blocked != has_class_labels:
            raise ValueError("Serialized class_blocked flag contradicts class-label data")
        class_labels = None
        if class_blocked:
            if dataset.attrs.get("class_label_encoding") != "tagged_json_v1":
                raise ValueError("Unsupported covariance class-label encoding")
            encoded = dataset["class_label_encoded"]
            decoded = np.frompyfunc(_decode_class_label, 1, 1)(encoded.values)
            serialized_name = str(dataset.attrs.get("class_labels_name", ""))
            try:
                serialized_attrs = json.loads(str(dataset.attrs.get("class_labels_attrs", "{}")))
            except json.JSONDecodeError as error:
                raise ValueError("Serialized covariance has invalid class-label attributes") from error
            class_labels = encoded.copy(data=decoded).rename(serialized_name or None)
            class_labels.attrs = serialized_attrs
        return cls(
            latitude=dataset.coords[lat_dim],
            longitude=dataset.coords[lon_dim],
            sigma=float(dataset.attrs["sigma"]),
            correlation_length=correlation_length,
            latitude_correlation_length=latitude_length if latitude_length_explicit else None,
            longitude_correlation_length=longitude_length if longitude_length_explicit else None,
            class_labels=class_labels,
        )

    def _apply_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the configured covariance to native-first right-hand sides.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Covariance-applied values with the same shape as ``matrix``.
        """
        if not self._class_masks:
            return self._apply_separable(matrix)
        result = np.zeros_like(matrix, dtype=np.result_type(matrix.dtype, np.float64))
        for mask in self._class_masks:
            masked = matrix * mask[..., np.newaxis]
            result += self._apply_separable(masked) * mask[..., np.newaxis]
        return result

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
        """Solve the unblocked separable system using Cholesky factors.

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

    def _solve_class_blocked(self, matrix: np.ndarray) -> np.ndarray:
        """Solve a class-blocked system with matrix-free conjugate gradients.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Solved values with the same shape as ``matrix``.

        Raises:
            numpy.linalg.LinAlgError: If conjugate gradients do not converge or
                fail because of invalid input or numerical breakdown.
        """
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
        cg_function = cast(Callable[..., tuple[np.ndarray, int]], cg)
        for rhs_index in range(n_rhs):
            solution, info = cg_function(
                operator,
                matrix[..., rhs_index].reshape(native_size),
                atol=0.0,
                maxiter=max(100, 10 * native_size),
                **{_CG_RELATIVE_TOLERANCE: 1e-12},
            )
            if info != 0:
                reason = "did not converge" if info > 0 else "failed with an illegal input or breakdown"
                raise np.linalg.LinAlgError(f"Class-blocked covariance solve {reason} (CG info={info})")
            solved[..., rhs_index] = solution.reshape(n_lat, n_lon)
        return solved

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
        One owned eager label array and one native-grid boolean mask per class.
        The empty mask tuple denotes an unblocked covariance.

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
        actual = np.asarray(class_labels.coords[dim].data)
        if actual.shape != expected.shape or not np.array_equal(actual, expected.data):
            raise ValueError(f"class_labels coordinate {dim!r} does not align with the covariance grid")
    labels = class_labels.transpose(*native_dims).compute().copy(deep=True)
    if bool(labels.isnull().any().item()):
        raise ValueError("class_labels must assign a non-null class to every native grid cell")

    values = np.asarray(labels.data)
    unique_values: list[object] = []
    for value in values.reshape(-1):
        matches = [_scalar_label_equal(value, unique) for unique in unique_values]
        if not any(matches):
            unique_values.append(value)
    masks = tuple(
        np.fromiter(
            (_scalar_label_equal(value, unique) for value in values.reshape(-1)),
            dtype=bool,
            count=values.size,
        ).reshape(values.shape)
        for unique in unique_values
    )
    return labels, masks


def _scalar_label_equal(left: object, right: object) -> bool:
    """Compare two class labels without NumPy broadcasting tuple values."""
    try:
        result = left == right
    except (TypeError, ValueError) as error:
        raise ValueError("class_labels values must be mutually comparable scalar labels") from error
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    raise ValueError("class_labels values must be mutually comparable scalar labels")


def _serialized_boolean(value: object, name: str) -> bool:
    """Decode a serialized Boolean represented only by Boolean or integer 0/1."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Serialized {name} must be Boolean or integer 0/1")


def _encode_class_label(value: object) -> str:
    """Encode a supported scalar or nested tuple class label as tagged JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    payload: list[object]
    if isinstance(value, tuple):
        payload = ["tuple", [_encode_class_label(item) for item in value]]
    elif isinstance(value, bool):
        payload = ["bool", value]
    elif isinstance(value, int):
        payload = ["int", value]
    elif isinstance(value, float):
        payload = ["float", value]
    elif isinstance(value, str):
        payload = ["str", value]
    else:
        raise TypeError(f"Unsupported class-label type for serialization: {type(value).__name__}")
    return json.dumps(payload, separators=(",", ":"))


def _decode_class_label(encoded: str) -> object:
    """Decode a class label produced by :func:`_encode_class_label`."""
    kind, value = json.loads(str(encoded))
    if kind == "tuple":
        return tuple(_decode_class_label(item) for item in value)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return str(value)
    raise ValueError(f"Unknown encoded class-label kind {kind!r}")


def _json_default(value: object) -> object:
    """Convert a NumPy scalar class-label attribute to a JSON scalar."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Class-label attr {value!r} is not JSON serializable")


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
