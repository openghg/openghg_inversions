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
matrix-free conjugate gradients. The linear operator reports the flattened
``N x N`` shape required by SciPy but does not materialise that matrix: each
iteration applies the separable kernel once per class through masked native-grid
work arrays. Grouping coordinates by class would expose a block-diagonal
matrix, but arbitrary class blocks are dense principal submatrices and are not
generally separable. The iterative solve therefore preserves the
``O(N + n_lat**2 + n_lon**2)`` storage bound, at the cost of repeated
per-class kernel applications; highly imbalanced classes can consequently have
noticeable computational overhead. Distances are coordinate-wise degrees, not
geodesic or longitude-wrapped distances. A missing coordinate ``units`` attr
is interpreted as degrees for compatibility with existing OGI grids.

``SeparableExponentialCovariance`` is an ordinary slotted, identity-based
object. Construction owns eager coordinate and class-label copies; array
properties return borrowed values whose in-place mutation is unsupported.
Scalar configuration properties are read-only, so changed parameters require
explicit reconstruction.

``ClassMixedExponentialCovariance`` keeps the class blocking but chooses the
kernel per class: classes named in its ``class_kernels`` use a dense,
non-separable :class:`GeodesicExponentialKernel`,
``exp(-d_gc / ell)`` in great-circle distance, and every other class uses the
separable kernel above. Each dense class of ``n`` cells is stored as one
``n x n`` float64 array (``8 n**2`` bytes) holding the kernel and its cached
Cholesky factor, so this action does materialise the dense covariance of the
classes it is asked to make non-separable, and only of those.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Protocol, cast

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular  # type: ignore[import-untyped]
# scipy-stubs 1.15.3 omits the runtime BLAS/LAPACK wrappers used by dense blocks.
from scipy.linalg.blas import dsymm  # type: ignore[attr-defined]
from scipy.linalg.lapack import dpocon, dpotrf  # type: ignore[attr-defined]
from scipy.sparse.linalg import LinearOperator, cg, eigsh  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions.borrowed import BorrowedDataArray, borrow
from openghg_inversions._serialization_codecs import (
    _TAGGED_JSON_VALUE_ENCODING,
    _decode_serialized_bool,
    _decode_tagged_json_value,
    _encode_tagged_json_value,
    _json_value,
)

_CG_RELATIVE_TOLERANCE = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
_NATIVE_SERIALIZED_VARIABLE_NAMES = {"class_label_encoded"}

__all__ = [
    "DEFAULT_EARTH_RADIUS_KM",
    "ClassMixedExponentialCovariance",
    "GeodesicBlockDiagnostics",
    "GeodesicExponentialKernel",
    "InvertibleNativeCovarianceAction",
    "NativeCovarianceAction",
    "SeparableExponentialCovariance",
]


class NativeCovarianceAction(Protocol):
    """Structural interface for a labelled self-adjoint PSD covariance action.

    Implementations represent a real, self-adjoint positive-semidefinite
    operator ``B`` on the labelled native state. They must preserve RHS layout
    and labels and return finite real values for finite real inputs. Consumers
    may trust these semantic guarantees without globally materializing or
    certifying the matrix-free operator.
    """

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
    """Positive-definite native covariance action with a compatible inverse.

    In addition to :class:`NativeCovarianceAction` semantics, ``B`` is positive
    definite and :meth:`solve` implements the inverse compatible with
    :meth:`apply` on the same labelled native state.
    """

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
        matrix, original_dims, rhs_dims = self._aligned_matrix(rhs)
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
        matrix, original_dims, rhs_dims = self._aligned_matrix(rhs)
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
                "class_label_encoding": _TAGGED_JSON_VALUE_ENCODING if self._class_labels is not None else "",
                "class_labels_name": ""
                if self._class_labels is None or self._class_labels.name is None
                else str(self._class_labels.name),
                "class_labels_attrs": "{}"
                if self._class_labels is None
                else json.dumps(
                    _json_value(self._class_labels.attrs),
                    sort_keys=True,
                ),
            },
        )
        if self._class_labels is not None:
            encoded = np.frompyfunc(_encode_tagged_json_value, 1, 1)(self._class_labels.values)
            dataset["class_label_encoded"] = self._class_labels.copy(data=np.asarray(encoded, dtype=str))
            dataset["class_label_encoded"].attrs = {}
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
            latitude_length_explicit = _decode_serialized_bool(
                dataset.attrs["latitude_correlation_length_explicit"],
                "latitude_correlation_length_explicit",
            )
            longitude_length_explicit = _decode_serialized_bool(
                dataset.attrs["longitude_correlation_length_explicit"],
                "longitude_correlation_length_explicit",
            )
        except KeyError as error:
            raise ValueError("Serialized covariance has invalid correlation-length metadata") from error
        try:
            class_blocked = _decode_serialized_bool(dataset.attrs["class_blocked"], "class_blocked")
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
            if dataset.attrs.get("class_label_encoding") != _TAGGED_JSON_VALUE_ENCODING:
                raise ValueError("Unsupported covariance class-label encoding")
            encoded = dataset["class_label_encoded"]
            decoded = np.frompyfunc(_decode_tagged_json_value, 1, 1)(encoded.values)
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
        if len(self._class_masks) <= 1:
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

        # scipy-stubs 1.15.3 omits the public shape/matvec factory overload.
        operator = LinearOperator(  # type: ignore[call-overload]
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

    def _aligned_matrix(self, rhs: xr.DataArray) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
        """Align and eagerly reshape a labelled right-hand side.

        Args:
            rhs: Candidate right-hand side containing the native dimensions.

        Returns:
            A native-first NumPy matrix, the original dimension order, and the
            ordered non-native right-hand-side dimensions.

        Raises:
            ValueError: If native dimensions or coordinates are missing or
                cannot be aligned exactly with the configured grid.
        """
        original_dims = tuple(str(dim) for dim in rhs.dims)
        rhs_dims = tuple(dim for dim in original_dims if dim not in self.native_dims)
        transposed = rhs.transpose(*self.native_dims, *rhs_dims)
        # xr.align treats an unindexed dimension as positional when its size
        # matches, but this public action requires an explicitly labelled grid.
        for dim in self.native_dims:
            if dim not in transposed.indexes:
                raise ValueError(f"rhs is missing indexed native coordinate {dim!r}")
        grid = xr.Dataset(
            coords={
                self.native_dims[0]: self._latitude,
                self.native_dims[1]: self._longitude,
            }
        )
        # The action is positional below this point. Exact alignment prevents
        # applying it to values from a different native grid.
        transposed, _ = xr.align(transposed, grid, join="exact", copy=False)
        values = np.asarray(transposed.values)
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


DEFAULT_EARTH_RADIUS_KM = 111.0 * 180.0 / np.pi
"""Sphere radius in km for which one degree of great-circle arc is 111 km.

Separable kernels in OGI configurations express lengths in degrees and convert
from kilometres with 111 km per degree of latitude; this radius keeps the
geodesic kernel's north-south correlation identical to theirs.
"""

# A dense geodesic block whose 1-norm reciprocal condition estimate falls below
# this value is rejected as numerically singular even when Cholesky succeeds.
_GEODESIC_MIN_RECIPROCAL_CONDITION = 1.0e3 * float(np.finfo(np.float64).eps)
# Columns of a dense geodesic block built per vectorized step (bounds the
# temporary arrays to a few ``n_cells x 512`` float64 arrays).
_GEODESIC_COLUMN_BLOCK = 512
# Blocks up to this size report their minimum eigenvalue from a dense
# eigendecomposition rather than Lanczos iterations on the Cholesky solve.
_GEODESIC_DENSE_EIGEN_LIMIT = 64
_MIXED_SERIALIZED_NAMES = {
    "class_label_encoded",
    "geodesic_class",
    "geodesic_class_encoded",
    "geodesic_class_kernel",
    "geodesic_correlation_length_km",
    "geodesic_earth_radius_km",
}


@dataclass(frozen=True, slots=True)
class GeodesicExponentialKernel:
    """Unit-variance exponential kernel in great-circle distance.

    The correlation between cells ``i`` and ``j`` is
    ``exp(-d_gc(i, j) / correlation_length_km)``, where ``d_gc`` is the
    haversine great-circle distance between cell centres on a sphere of radius
    ``earth_radius_km``. The kernel is positive definite on the sphere for
    every positive length, isotropic in local kilometres, and its east-west
    correlation length in degrees of longitude grows as ``1 / cos(latitude)``.

    Attributes:
        correlation_length_km: E-folding distance in km.
        earth_radius_km: Sphere radius in km. The default makes one degree of
            arc 111 km, consistent with OGI's degree lengths.
    """

    correlation_length_km: float
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM
    kind: ClassVar[str] = "geodesic_exponential"

    def __post_init__(self) -> None:
        """Validate and normalise the kernel parameters to floats.

        Raises:
            ValueError: If a length or radius is not finite and positive.
        """
        length = _positive_finite(self.correlation_length_km, "correlation_length_km")
        object.__setattr__(self, "correlation_length_km", length)
        object.__setattr__(self, "earth_radius_km", _positive_finite(self.earth_radius_km, "earth_radius_km"))

    def correlation(
        self,
        latitude_1: np.ndarray,
        longitude_1: np.ndarray,
        latitude_2: np.ndarray,
        longitude_2: np.ndarray,
    ) -> np.ndarray:
        """Return broadcast kernel values between points given in degrees.

        Args:
            latitude_1: Latitudes of the first points in degrees.
            longitude_1: Longitudes of the first points in degrees.
            latitude_2: Latitudes of the second points in degrees.
            longitude_2: Longitudes of the second points in degrees.

        Returns:
            Correlations broadcast over the four inputs.
        """
        phi_1 = np.deg2rad(np.asarray(latitude_1, dtype=np.float64))
        phi_2 = np.deg2rad(np.asarray(latitude_2, dtype=np.float64))
        delta_longitude = np.deg2rad(
            np.asarray(longitude_1, dtype=np.float64) - np.asarray(longitude_2, dtype=np.float64)
        )
        haversine = (
            np.sin(0.5 * (phi_1 - phi_2)) ** 2
            + np.cos(phi_1) * np.cos(phi_2) * np.sin(0.5 * delta_longitude) ** 2
        )
        distance = 2.0 * self.earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        return np.exp(-distance / self.correlation_length_km)


@dataclass(frozen=True, slots=True)
class GeodesicBlockDiagnostics:
    """Positive-definiteness report for one dense geodesic class block.

    Attributes:
        class_label: Native class label of the block.
        size: Number of native cells in the class.
        correlation_length_km: Kernel e-folding distance in km.
        reciprocal_condition: LAPACK ``dpocon`` estimate of the reciprocal
            1-norm condition number of the unit-variance block.
        one_norm: 1-norm of the unit-variance block.
        nbytes: Bytes held by the block's dense storage.
    """

    class_label: object
    size: int
    correlation_length_km: float
    reciprocal_condition: float
    one_norm: float
    nbytes: int


class _DenseGeodesicBlock:
    """Factorized unit-variance geodesic kernel on one class's cells.

    One Fortran-ordered ``n x n`` array holds both the kernel and its Cholesky
    factor, so the block costs ``8 n**2`` bytes rather than twice that. The
    upper triangle and diagonal hold ``K``; the strict lower triangle holds the
    unit lower-triangular ``L_u`` of ``K = L_u D**2 L_u.T``, whose diagonal
    ``D`` is kept separately. Apply uses ``dsymm`` on the upper triangle and
    solve uses two unit-diagonal triangular solves on the lower triangle.
    """

    __slots__ = ("cell_index", "diagnostics", "kernel", "pivots", "storage")

    def __init__(
        self,
        class_label: object,
        kernel: GeodesicExponentialKernel,
        cell_index: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
    ) -> None:
        """Build, factorize and check the dense unit-variance block.

        Args:
            class_label: Class label, used in diagnostics and errors.
            kernel: Geodesic kernel configuration.
            cell_index: Row-major native-cell indices of the class.
            latitude: Cell-centre latitudes in degrees, one per index.
            longitude: Cell-centre longitudes in degrees, one per index.

        Raises:
            numpy.linalg.LinAlgError: If the block is not numerically positive
                definite.
        """
        size = int(cell_index.size)
        storage = np.empty((size, size), dtype=np.float64, order="F")
        one_norm = 0.0
        for start in range(0, size, _GEODESIC_COLUMN_BLOCK):
            stop = min(size, start + _GEODESIC_COLUMN_BLOCK)
            columns = kernel.correlation(
                latitude[:, np.newaxis],
                longitude[:, np.newaxis],
                latitude[np.newaxis, start:stop],
                longitude[np.newaxis, start:stop],
            )
            storage[:, start:stop] = columns
            one_norm = max(one_norm, float(np.abs(columns).sum(axis=0).max()))
            del columns
        kernel_diagonal = np.diag(storage).copy()

        # Factorize in place: LAPACK overwrites only the lower triangle and
        # the diagonal, leaving K in the strict upper triangle.
        factor, info = dpotrf(storage, lower=1, clean=0, overwrite_a=1)
        if not np.shares_memory(factor, storage):  # pragma: no cover - defensive
            storage = factor
        label = repr(class_label)
        if info > 0:
            raise np.linalg.LinAlgError(
                f"Geodesic covariance block for class {label} "
                f"(correlation_length_km={kernel.correlation_length_km:g}, {size} cells) is not positive "
                f"definite: Cholesky failed at leading minor {info}"
            )
        if info < 0:  # pragma: no cover - LAPACK argument error
            raise np.linalg.LinAlgError(f"LAPACK dpotrf failed with info={info}")
        reciprocal_condition, condition_info = dpocon(storage, one_norm, uplo="L")
        if condition_info != 0:  # pragma: no cover - LAPACK argument error
            raise np.linalg.LinAlgError(f"LAPACK condition estimation failed with info={condition_info}")
        reciprocal_condition = float(reciprocal_condition)
        if not reciprocal_condition >= _GEODESIC_MIN_RECIPROCAL_CONDITION:
            raise np.linalg.LinAlgError(
                f"Geodesic covariance block for class {label} "
                f"(correlation_length_km={kernel.correlation_length_km:g}, {size} cells) is numerically "
                f"singular: reciprocal 1-norm condition estimate {reciprocal_condition:.3g} is below "
                f"{_GEODESIC_MIN_RECIPROCAL_CONDITION:.3g}"
            )
        pivots = np.diag(storage).copy()
        storage[np.diag_indices(size)] = kernel_diagonal
        for column in range(size - 1):
            storage[column + 1 :, column] /= pivots[column]

        self.kernel = kernel
        self.cell_index = cell_index
        self.storage = storage
        self.pivots = pivots
        self.diagnostics = GeodesicBlockDiagnostics(
            class_label=class_label,
            size=size,
            correlation_length_km=kernel.correlation_length_km,
            reciprocal_condition=reciprocal_condition,
            one_norm=one_norm,
            nbytes=int(storage.nbytes + pivots.nbytes),
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Return ``K values`` for an ``(n, rhs)`` array."""
        return np.asarray(dsymm(1.0, self.storage, values, side=0, lower=0))

    def solve(self, values: np.ndarray) -> np.ndarray:
        """Return ``K^-1 values`` for an ``(n, rhs)`` array."""
        forward = solve_triangular(
            self.storage, values, lower=True, unit_diagonal=True, check_finite=False
        )
        forward /= (self.pivots**2)[:, np.newaxis]
        return np.asarray(
            solve_triangular(
                self.storage, forward, lower=True, trans="T", unit_diagonal=True, check_finite=False
            )
        )

    def minimum_eigenvalue(self) -> float:
        """Return the smallest eigenvalue of the unit-variance block."""
        size = self.storage.shape[0]
        if size <= _GEODESIC_DENSE_EIGEN_LIMIT:
            upper = np.triu(self.storage)
            dense = upper + np.triu(self.storage, 1).T
            return float(np.linalg.eigvalsh(dense)[0])

        def precision(vector: np.ndarray) -> np.ndarray:
            return self.solve(vector.reshape(size, 1)).reshape(size)

        # scipy-stubs 1.15.3 omits the public shape/matvec factory overload.
        operator = LinearOperator(  # type: ignore[call-overload]
            (size, size), matvec=precision, dtype=np.float64
        )
        largest = eigsh(operator, k=1, which="LM", return_eigenvectors=False)  # K^-1 is PD
        return float(1.0 / largest[0])


class ClassMixedExponentialCovariance:
    """Class-blocked exponential covariance with a per-class kernel choice.

    Cells in different classes are uncorrelated, as in a class-blocked
    :class:`SeparableExponentialCovariance`. Classes named in
    ``class_kernels`` use a dense non-separable kernel (currently
    :class:`GeodesicExponentialKernel`); every other class uses the separable
    exponential kernel of :class:`SeparableExponentialCovariance` with the
    given degree lengths. With ``K_c`` the unit-variance kernel of class ``c``
    and ``M_c`` its indicator,

    ``B = sigma**2 * sum_c M_c K_c M_c``.

    Args:
        latitude: One-dimensional latitude coordinate in degrees.
        longitude: One-dimensional longitude coordinate in degrees.
        class_labels: Native-grid labels assigning every cell to a class.
        class_kernels: Mapping from class label to the dense kernel of that
            class. It must name at least one class present in
            ``class_labels``.
        sigma: Strictly positive native-cell marginal standard deviation,
            shared by every class.
        correlation_length: Separable fallback correlation length in degrees.
        latitude_correlation_length: Optional separable latitude length.
        longitude_correlation_length: Optional separable longitude length.

    Notes:
        Each dense class of ``n`` cells holds one ``n x n`` float64 array
        (``8 n**2`` bytes, 1.1 GB for 11,832 cells) containing both the kernel
        and its Cholesky factor; it is built and factorized once at
        construction and never copied. Apply costs ``2 n**2`` flops per RHS
        for the dense classes plus the separable classes' factor products.
        Solve uses the cached Cholesky factor for each dense class and the
        matrix-free conjugate-gradient path of the class-blocked separable
        covariance, restricted to the separable classes. Construction fails
        with :class:`numpy.linalg.LinAlgError` if a dense block is not
        numerically positive definite; :attr:`geodesic_diagnostics` reports
        the condition estimates and :meth:`minimum_eigenvalues` the smallest
        eigenvalues. :meth:`with_sigma` rescales without refactorizing.

    Raises:
        ValueError: If coordinates, units, covariance parameters, class labels
            or class kernels are invalid.
        TypeError: If a class kernel is not a supported kernel.
        numpy.linalg.LinAlgError: If a dense class block is not numerically
            positive definite.
    """

    __slots__ = (
        "_blocks",
        "_class_kernels",
        "_separable",
        "_separable_boxes",
        "_separable_index",
        "_sigma",
    )

    schema = "openghg_inversions.class_mixed_exponential_covariance"
    schema_version = 1

    def __init__(
        self,
        latitude: xr.DataArray,
        longitude: xr.DataArray,
        *,
        class_labels: xr.DataArray,
        class_kernels: Mapping[object, GeodesicExponentialKernel],
        sigma: float = 1.0,
        correlation_length: float = 1.5,
        latitude_correlation_length: float | None = None,
        longitude_correlation_length: float | None = None,
    ) -> None:
        """Validate inputs, then build and factorize every dense class block."""
        if class_labels is None:
            raise ValueError("ClassMixedExponentialCovariance requires class_labels")
        separable = SeparableExponentialCovariance(
            latitude,
            longitude,
            sigma=1.0,
            correlation_length=correlation_length,
            latitude_correlation_length=latitude_correlation_length,
            longitude_correlation_length=longitude_correlation_length,
            class_labels=class_labels,
        )
        self._initialize(separable, _positive_finite(sigma, "sigma"), class_kernels)

    def _initialize(
        self,
        separable: SeparableExponentialCovariance,
        sigma: float,
        class_kernels: Mapping[object, GeodesicExponentialKernel],
    ) -> None:
        """Resolve class kernels against the separable class masks.

        Args:
            separable: Unit-sigma class-blocked separable covariance supplying
                coordinates, separable factors and class masks.
            sigma: Validated marginal standard deviation.
            class_kernels: Requested dense kernel per class label.

        Raises:
            ValueError: If the class kernels are empty, name absent classes or
                name one class twice, or a native dimension is reserved.
            TypeError: If a kernel is not a :class:`GeodesicExponentialKernel`.
        """
        reserved = _MIXED_SERIALIZED_NAMES.intersection(separable.native_dims)
        if reserved:
            raise ValueError(f"native dimension {sorted(reserved)[0]!r} is reserved by the serialized schema")
        kernels = dict(class_kernels)
        if not kernels:
            raise ValueError(
                "class_kernels must name at least one class; use SeparableExponentialCovariance otherwise"
            )
        labels = separable._class_labels
        assert labels is not None
        label_values = np.asarray(labels.values)
        masks = separable._class_masks
        mask_labels = [label_values[mask].flat[0] for mask in masks]
        resolved: dict[int, tuple[object, GeodesicExponentialKernel]] = {}
        for key, kernel in kernels.items():
            if not isinstance(kernel, GeodesicExponentialKernel):
                raise TypeError(f"Kernel for class {key!r} must be a GeodesicExponentialKernel")
            matches = [index for index, label in enumerate(mask_labels) if _scalar_label_equal(key, label)]
            if not matches:
                raise ValueError(f"class_kernels names class {key!r}, which is absent from class_labels")
            if matches[0] in resolved:
                raise ValueError(f"class_kernels names class {key!r} more than once")
            resolved[matches[0]] = (key.item() if isinstance(key, np.generic) else key, kernel)

        latitude_values = np.asarray(separable._latitude.values, dtype=np.float64)
        longitude_values = np.asarray(separable._longitude.values, dtype=np.float64)
        n_lon = longitude_values.size
        blocks = []
        for index in sorted(resolved):
            label, kernel = resolved[index]
            cell_index = np.flatnonzero(masks[index].reshape(-1))
            blocks.append(
                _DenseGeodesicBlock(
                    label,
                    kernel,
                    cell_index,
                    latitude_values[cell_index // n_lon],
                    longitude_values[cell_index % n_lon],
                )
            )
        separable_masks = [mask for index, mask in enumerate(masks) if index not in resolved]
        self._separable = separable
        self._sigma = sigma
        self._class_kernels = MappingProxyType({label: kernel for label, kernel in resolved.values()})
        self._blocks = tuple(blocks)
        self._separable_boxes = _mixed_class_bounding_boxes(separable_masks)
        self._separable_index = (
            np.flatnonzero(np.logical_or.reduce(separable_masks).reshape(-1))
            if separable_masks
            else np.empty(0, dtype=np.intp)
        )

    @property
    def latitude(self) -> BorrowedDataArray:
        """Borrow the owned latitude coordinate; in-place mutation is unsupported."""
        return self._separable.latitude

    @property
    def longitude(self) -> BorrowedDataArray:
        """Borrow the owned longitude coordinate; in-place mutation is unsupported."""
        return self._separable.longitude

    @property
    def sigma(self) -> float:
        """Read-only native-cell marginal standard deviation."""
        return self._sigma

    @property
    def correlation_length(self) -> float:
        """Read-only separable fallback correlation length in degrees."""
        return self._separable.correlation_length

    @property
    def latitude_correlation_length(self) -> float:
        """Read-only separable latitude correlation length in degrees."""
        return self._separable.latitude_correlation_length

    @property
    def longitude_correlation_length(self) -> float:
        """Read-only separable longitude correlation length in degrees."""
        return self._separable.longitude_correlation_length

    @property
    def latitude_correlation_length_explicit(self) -> bool:
        """Whether the separable latitude length was configured independently."""
        return self._separable.latitude_correlation_length_explicit

    @property
    def longitude_correlation_length_explicit(self) -> bool:
        """Whether the separable longitude length was configured independently."""
        return self._separable.longitude_correlation_length_explicit

    @property
    def class_labels(self) -> BorrowedDataArray:
        """Borrow the owned class labels; in-place mutation is unsupported."""
        labels = self._separable.class_labels
        assert labels is not None
        return labels

    @property
    def class_kernels(self) -> Mapping[object, GeodesicExponentialKernel]:
        """Read-only mapping from dense class label to its kernel."""
        return self._class_kernels

    @property
    def geodesic_diagnostics(self) -> tuple[GeodesicBlockDiagnostics, ...]:
        """Positive-definiteness and memory report of every dense class block."""
        return tuple(block.diagnostics for block in self._blocks)

    @property
    def native_dims(self) -> tuple[str, str]:
        """Latitude and longitude dimension names in vectorisation order."""
        return self._separable.native_dims

    def minimum_eigenvalues(self) -> dict[object, float]:
        """Return the smallest eigenvalue of each unit-variance dense block.

        Small blocks use a dense eigendecomposition; larger blocks use Lanczos
        iterations on the cached Cholesky solve, without another dense copy.

        Returns:
            Mapping from dense class label to its smallest eigenvalue. Multiply
            by ``sigma**2`` for the covariance block.
        """
        return {block.diagnostics.class_label: block.minimum_eigenvalue() for block in self._blocks}

    def with_sigma(self, sigma: float) -> ClassMixedExponentialCovariance:
        """Return the same kernel with another amplitude, sharing dense factors.

        Args:
            sigma: Strictly positive native-cell marginal standard deviation.

        Returns:
            A covariance whose kernel, class labels and dense blocks are shared
            (read-only) with this one.
        """
        result = object.__new__(type(self))
        result._separable = self._separable
        result._sigma = _positive_finite(sigma, "sigma")
        result._class_kernels = self._class_kernels
        result._blocks = self._blocks
        result._separable_boxes = self._separable_boxes
        result._separable_index = self._separable_index
        return result

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply ``B`` while preserving the labelled layout of ``rhs``.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B rhs`` with dimensions and coordinates identical to ``rhs``.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric.
        """
        matrix, original_dims, rhs_dims = self._separable._aligned_matrix(rhs)
        applied = self._apply_matrix(matrix)
        return self._separable._restore(applied, rhs, original_dims, rhs_dims)

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve ``B result = rhs`` class by class.

        Dense classes use their cached Cholesky factors; separable classes use
        matrix-free conjugate gradients restricted to their cells.

        Args:
            rhs: Array containing both native dimensions and any number of
                additional right-hand-side dimensions.

        Returns:
            ``B^-1 rhs`` with dimensions and coordinates identical to ``rhs``.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If native dimensions or coordinates are missing or
                misaligned, or if values are non-numeric.
            numpy.linalg.LinAlgError: If the separable-class iterative solve
                does not converge.
        """
        matrix, original_dims, rhs_dims = self._separable._aligned_matrix(rhs)
        solved = self._solve_matrix(matrix)
        return self._separable._restore(solved, rhs, original_dims, rhs_dims)

    def to_dataset(self) -> xr.Dataset:
        """Serialize coordinates, separable configuration and class kernels.

        Returns:
            The class-blocked :class:`SeparableExponentialCovariance` dataset
            of the separable configuration under this class's schema, with
            ``sigma`` and the dense class kernels over a ``geodesic_class``
            dimension (tagged-JSON class labels, kernel kind, length and
            radius in km). Two covariances with the same dataset apart from
            the ``sigma`` attribute share an identical unit kernel.
        """
        dataset = self._separable.to_dataset()
        labels = list(self._class_kernels)
        kernels = [self._class_kernels[label] for label in labels]
        dataset.attrs.update(schema=self.schema, schema_version=self.schema_version, sigma=self.sigma)
        encoded = [_encode_tagged_json_value(label) for label in labels]
        dataset["geodesic_class_encoded"] = xr.DataArray(
            np.asarray(encoded, dtype=str), dims="geodesic_class"
        )
        dataset["geodesic_class_kernel"] = xr.DataArray(
            np.asarray([kernel.kind for kernel in kernels], dtype=str), dims="geodesic_class"
        )
        dataset["geodesic_correlation_length_km"] = xr.DataArray(
            [kernel.correlation_length_km for kernel in kernels], dims="geodesic_class"
        )
        dataset["geodesic_earth_radius_km"] = xr.DataArray(
            [kernel.earth_radius_km for kernel in kernels], dims="geodesic_class"
        )
        return dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> ClassMixedExponentialCovariance:
        """Restore a covariance from :meth:`to_dataset` output.

        Args:
            dataset: Versioned class-mixed covariance dataset.

        Returns:
            Reconstructed covariance with its dense blocks refactorized.

        Raises:
            ValueError: If the schema, version, coordinates, class labels or
                class kernels are missing or invalid.
            numpy.linalg.LinAlgError: If a dense block is not numerically
                positive definite.
        """
        if dataset.attrs.get("schema") != cls.schema:
            raise ValueError(f"Expected covariance schema {cls.schema!r}")
        if dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported covariance schema version {dataset.attrs.get('schema_version')!r}")
        kernel_names = (
            "geodesic_class_encoded",
            "geodesic_class_kernel",
            "geodesic_correlation_length_km",
            "geodesic_earth_radius_km",
        )
        missing = [name for name in kernel_names if name not in dataset.data_vars]
        if missing:
            raise ValueError(f"Serialized class-mixed covariance is missing variables {missing}")
        class_kernels = _decode_class_kernels(
            dataset["geodesic_class_encoded"].values,
            dataset["geodesic_class_kernel"].values,
            dataset["geodesic_correlation_length_km"].values,
            dataset["geodesic_earth_radius_km"].values,
        )
        separable_dataset = dataset.drop_vars(list(kernel_names)).assign_attrs(
            schema=SeparableExponentialCovariance.schema,
            schema_version=SeparableExponentialCovariance.schema_version,
            sigma=1.0,
        )
        separable = SeparableExponentialCovariance.from_dataset(separable_dataset)
        if separable.class_labels is None:
            raise ValueError("Serialized class-mixed covariance requires class labels")
        result = object.__new__(cls)
        result._initialize(
            separable, _positive_finite(float(dataset.attrs["sigma"]), "serialized sigma"), class_kernels
        )
        return result

    def _apply_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply ``B`` to native-first right-hand sides.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Covariance-applied values with the same shape as ``matrix``.
        """
        n_lat, n_lon, n_rhs = matrix.shape
        result = self._apply_separable_classes(matrix)
        flat_rhs = matrix.reshape(n_lat * n_lon, n_rhs)
        flat_result = result.reshape(n_lat * n_lon, n_rhs)
        for block in self._blocks:
            flat_result[block.cell_index] = block.apply(flat_rhs[block.cell_index])
        result *= self._sigma**2
        return result

    def _apply_separable_classes(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the unit separable kernel of every separable class.

        Each class is applied on its bounding box. Cells outside the class are
        zeroed before and after, so restricting the factors to the box is
        exact. Dense-class cells are left at zero.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            New array of the same shape.
        """
        latitude_factor = self._separable._latitude_factor
        longitude_factor = self._separable._longitude_factor
        result = np.zeros(matrix.shape, dtype=np.result_type(matrix.dtype, np.float64))
        for lat_slice, lon_slice, local_mask in self._separable_boxes:
            masked = matrix[lat_slice, lon_slice] * local_mask[..., np.newaxis]
            n_lat, n_lon, n_rhs = masked.shape
            lat_applied = latitude_factor[lat_slice, lat_slice] @ masked.reshape(n_lat, n_lon * n_rhs)
            lat_applied = lat_applied.reshape(n_lat, n_lon, n_rhs)
            applied = np.einsum(
                "ikr,lk->ilr", lat_applied, longitude_factor[lon_slice, lon_slice], optimize=True
            )
            result[lat_slice, lon_slice] += applied * local_mask[..., np.newaxis]
        return result

    def _solve_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Solve ``B result = matrix`` for native-first right-hand sides.

        Args:
            matrix: Eager array shaped ``(latitude, longitude, rhs)``.

        Returns:
            Solved values with the same shape as ``matrix``.

        Raises:
            numpy.linalg.LinAlgError: If conjugate gradients do not converge.
        """
        n_lat, n_lon, n_rhs = matrix.shape
        native_size = n_lat * n_lon
        flat_rhs = matrix.reshape(native_size, n_rhs)
        solved = np.zeros((native_size, n_rhs), dtype=np.result_type(matrix.dtype, np.float64))
        for block in self._blocks:
            solved[block.cell_index] = block.solve(flat_rhs[block.cell_index])

        index = self._separable_index
        if index.size:
            work = np.zeros(native_size)

            def matvec(vector: np.ndarray) -> np.ndarray:
                """Apply the separable classes to vectors on separable cells."""
                work[index] = np.ravel(vector)
                applied = self._apply_separable_classes(work.reshape(n_lat, n_lon, 1))
                return applied.reshape(native_size)[index]

            # scipy-stubs 1.15.3 omits the public shape/matvec factory overload.
            operator = LinearOperator(  # type: ignore[call-overload]
                (index.size, index.size),
                matvec,
            )
            cg_function = cast(Callable[..., tuple[np.ndarray, int]], cg)
            for rhs_index in range(n_rhs):
                solution, info = cg_function(
                    operator,
                    flat_rhs[index, rhs_index],
                    atol=0.0,
                    maxiter=max(100, 10 * index.size),
                    **{_CG_RELATIVE_TOLERANCE: 1e-12},
                )
                if info != 0:
                    reason = "did not converge" if info > 0 else "failed with an illegal input or breakdown"
                    raise np.linalg.LinAlgError(f"Separable-class covariance solve {reason} (CG info={info})")
                solved[index, rhs_index] = solution
        solved /= self._sigma**2
        return solved.reshape(n_lat, n_lon, n_rhs)


def _decode_class_kernels(
    encoded_labels: np.ndarray,
    kinds: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
) -> dict[object, GeodesicExponentialKernel]:
    """Decode serialized dense class kernels.

    Args:
        encoded_labels: Tagged-JSON class labels.
        kinds: Kernel kind of each class.
        lengths: Correlation lengths in km.
        radii: Sphere radii in km.

    Returns:
        Mapping from decoded class label to kernel.

    Raises:
        ValueError: If a label, kind or parameter is invalid.
    """
    kernels: dict[object, GeodesicExponentialKernel] = {}
    for encoded, kind, length, radius in zip(encoded_labels, kinds, lengths, radii, strict=True):
        if str(kind) != GeodesicExponentialKernel.kind:
            raise ValueError(f"Unsupported serialized class kernel {str(kind)!r}")
        kernels[_decode_tagged_json_value(str(encoded))] = GeodesicExponentialKernel(
            correlation_length_km=float(length), earth_radius_km=float(radius)
        )
    return kernels


def _mixed_class_bounding_boxes(
    class_masks: list[np.ndarray],
) -> tuple[tuple[slice, slice, np.ndarray], ...]:
    """Return each class mask's latitude/longitude bounding box and local mask.

    Args:
        class_masks: Boolean native-grid masks, one per class.

    Returns:
        One ``(latitude_slice, longitude_slice, local_mask)`` triple per
        non-empty class.
    """
    boxes = []
    for mask in class_masks:
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:  # pragma: no cover - masks come from present labels
            continue
        lat_slice = slice(int(rows[0]), int(rows[-1]) + 1)
        lon_slice = slice(int(columns[0]), int(columns[-1]) + 1)
        boxes.append((lat_slice, lon_slice, mask[lat_slice, lon_slice]))
    return tuple(boxes)
