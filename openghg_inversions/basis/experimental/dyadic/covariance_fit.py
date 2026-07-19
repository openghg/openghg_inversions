"""Reference fits for separable exponential regional covariance models.

These helpers compare an induced regional covariance with conventional
distance constructions. The point-representative reference is

``C[i, j] = sigma[i] * sigma[j] * exp(-|dlat| / ell) * exp(-|dlon| / ell)``.

The projected reference instead constructs this separable covariance on a
native latitude-longitude grid and evaluates ``P.T @ B @ P`` without
materializing dense ``B``. Coordinates are in degrees, so one shared length
scale is not physically isotropic away from the equator. Both forms are useful
as transparent diagnostics for whether a tree prior resembles ordinary
distance-decaying dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize_scalar

from .grid_covariance import SeparableGridCovariance


@dataclass(frozen=True, slots=True)
class ExponentialLengthScaleFit:
    """Diagnostics from one off-diagonal exponential covariance fit.

    Attributes:
        length_scale: Fitted common latitude/longitude scale in degrees.
        rmse: Root-mean-square off-diagonal residual in target units.
        relative_rmse: Residual Euclidean norm divided by target norm.
        target_model_correlation: Pearson correlation between target and fitted
            off-diagonal entries, or ``None`` when either vector is constant.
        pair_count: Number of unique regional pairs used in the fit.
        converged: Whether SciPy reported a successful bounded optimization.
    """

    length_scale: float
    rmse: float
    relative_rmse: float
    target_model_correlation: float | None
    pair_count: int
    converged: bool


def separable_exponential_correlation(
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    length_scale: float,
) -> npt.NDArray[np.float64]:
    """Construct a separable exponential correlation matrix in degrees.

    Args:
        latitude: One representative latitude per region.
        longitude: One representative longitude per region.
        length_scale: Positive shared decay scale for both coordinates.

    Returns:
        Symmetric region-by-region correlation matrix with unit diagonal.

    Raises:
        ValueError: If coordinates or ``length_scale`` are invalid.
    """
    lat, lon = _validated_coordinates(latitude, longitude)
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive.")
    distance = np.abs(lat[:, None] - lat[None, :]) + np.abs(lon[:, None] - lon[None, :])
    return np.exp(-distance / length_scale)


def fit_separable_exponential_length_scale(
    target_covariance: npt.ArrayLike,
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    *,
    standard_deviation: npt.ArrayLike | None = None,
    length_scale_bounds: tuple[float, float] = (0.01, 360.0),
) -> ExponentialLengthScaleFit:
    """Fit one exponential length scale to unique off-diagonal region pairs.

    Args:
        target_covariance: Finite symmetric covariance or correlation matrix.
        latitude: One representative latitude per target region.
        longitude: One representative longitude per target region.
        standard_deviation: Fixed positive marginal standard deviations. When
            omitted, use the square root of the target diagonal. Supply ones to
            fit a target correlation matrix.
        length_scale_bounds: Positive lower and upper bounds in degrees.

    Returns:
        Fitted length scale and residual diagnostics.

    Raises:
        ValueError: If matrix, coordinates, standard deviations, or bounds are
            invalid, or fewer than two regions are supplied.
    """
    target = _validated_covariance(target_covariance)
    lat, lon = _validated_coordinates(latitude, longitude)
    if target.shape[0] != lat.size:
        raise ValueError("target_covariance and coordinates must describe the same regions.")
    if lat.size < 2:
        raise ValueError("At least two regions are required to fit a length scale.")
    lower, upper = (float(value) for value in length_scale_bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("length_scale_bounds must be finite, positive, and increasing.")

    if standard_deviation is None:
        diagonal = np.diag(target)
        if (diagonal <= 0.0).any():
            raise ValueError("target_covariance must have a positive diagonal.")
        sigma = np.sqrt(diagonal)
    else:
        sigma = np.asarray(standard_deviation, dtype=np.float64)
        if sigma.shape != (lat.size,):
            raise ValueError("standard_deviation must contain one value per region.")
        if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
            raise ValueError("standard_deviation must be finite and positive.")

    row, column = np.triu_indices(lat.size, k=1)
    target_pairs = target[row, column]
    sigma_pairs = sigma[row] * sigma[column]
    distance_pairs = np.abs(lat[row] - lat[column]) + np.abs(lon[row] - lon[column])

    def objective(log_length_scale: float) -> float:
        """Return mean squared covariance residual at a log length scale."""
        model_pairs = sigma_pairs * np.exp(-distance_pairs / math.exp(log_length_scale))
        return float(np.mean(np.square(target_pairs - model_pairs)))

    result = cast(
        OptimizeResult,
        minimize_scalar(
            objective,
            method="bounded",
            bounds=(math.log(lower), math.log(upper)),
            options={"xatol": 1.0e-10},
        ),
    )
    length_scale = math.exp(float(result.x))
    model_pairs = sigma_pairs * np.exp(-distance_pairs / length_scale)
    residual = target_pairs - model_pairs
    target_norm = float(np.linalg.norm(target_pairs))
    model_variation = float(np.std(model_pairs))
    target_variation = float(np.std(target_pairs))
    if model_variation > 0.0 and target_variation > 0.0:
        target_model_correlation = float(np.corrcoef(target_pairs, model_pairs)[0, 1])
    else:
        target_model_correlation = None
    return ExponentialLengthScaleFit(
        length_scale=length_scale,
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        relative_rmse=float(np.linalg.norm(residual) / target_norm) if target_norm > 0.0 else 0.0,
        target_model_correlation=target_model_correlation,
        pair_count=int(target_pairs.size),
        converged=bool(result.success),
    )


def projected_exponential_covariance(
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    projection: npt.ArrayLike,
    length_scale: float,
    *,
    class_labels: npt.ArrayLike | None = None,
    regional_standard_deviation: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Project a separable native-grid covariance onto regional variables.

    Args:
        latitude: Native one-dimensional latitude coordinates.
        longitude: Native one-dimensional longitude coordinates.
        projection: Matrix with shape ``(M, K)`` representing ``P.T``, where
            ``P`` restricts native grid variables to ``K`` regional variables.
        length_scale: Positive common latitude/longitude scale in degrees.
        class_labels: Optional native-grid hard classes. Native covariance is
            zero between different classes.
        regional_standard_deviation: Optional positive regional standard
            deviations. When supplied, normalize ``P B P.T`` to correlation
            before imposing these marginal standard deviations.

    Returns:
        Projected covariance with shape ``(K, K)``. Without regional scaling,
        this is exactly ``P B P.T`` for unit native-grid marginal variance.

    Raises:
        ValueError: If the projection or regional standard deviations are
            invalid or inconsistent with the grid.
    """
    lat, lon = _validated_grid_coordinates(latitude, longitude)
    projected_columns = _validated_projection(projection, grid_size=lat.size * lon.size)
    operator = SeparableGridCovariance(
        lat,
        lon,
        latitude_length_scale=length_scale,
        longitude_length_scale=length_scale,
        class_labels=class_labels,
    )
    covariance = operator.projected_covariance(projected_columns)
    if regional_standard_deviation is None:
        return covariance

    sigma = np.asarray(regional_standard_deviation, dtype=np.float64)
    if sigma.shape != (projected_columns.shape[1],):
        raise ValueError("regional_standard_deviation must contain one value per projected region.")
    if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
        raise ValueError("regional_standard_deviation must be finite and positive.")
    correlation = _covariance_to_correlation(covariance)
    return sigma[:, None] * sigma[None, :] * correlation


def fit_projected_exponential_length_scale(
    target_covariance: npt.ArrayLike,
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    projection: npt.ArrayLike,
    *,
    standard_deviation: npt.ArrayLike | None = None,
    class_labels: npt.ArrayLike | None = None,
    length_scale_bounds: tuple[float, float] = (0.01, 360.0),
) -> ExponentialLengthScaleFit:
    """Fit a native-grid scale after exact projection to target regions.

    Args:
        target_covariance: Finite symmetric regional covariance or correlation
            matrix with shape ``(K, K)``.
        latitude: Native one-dimensional latitude coordinates.
        longitude: Native one-dimensional longitude coordinates.
        projection: Matrix with shape ``(M, K)`` representing the transpose of
            the native-to-regional restriction.
        standard_deviation: Fixed positive regional standard deviations. When
            omitted, use the square root of the target diagonal. Supply ones to
            fit a target correlation matrix.
        class_labels: Optional native-grid hard classes.
        length_scale_bounds: Positive lower and upper bounds in degrees.

    Returns:
        Fitted scale and off-diagonal regional residual diagnostics.

    Raises:
        ValueError: If matrices, coordinates, standard deviations, classes, or
            bounds are invalid, or fewer than two regions are supplied.
    """
    target = _validated_covariance(target_covariance)
    lat, lon = _validated_grid_coordinates(latitude, longitude)
    projected_columns = _validated_projection(projection, grid_size=lat.size * lon.size)
    if target.shape[0] != projected_columns.shape[1]:
        raise ValueError("target_covariance and projection must describe the same regions.")
    if target.shape[0] < 2:
        raise ValueError("At least two regions are required to fit a length scale.")
    lower, upper = _validated_length_scale_bounds(length_scale_bounds)
    sigma = _regional_standard_deviation(target, standard_deviation)
    row, column = np.triu_indices(target.shape[0], k=1)
    target_pairs = target[row, column]

    def model_pairs(log_length_scale: float) -> npt.NDArray[np.float64]:
        """Return projected off-diagonal pairs at a log length scale."""
        model = projected_exponential_covariance(
            lat,
            lon,
            projected_columns,
            math.exp(log_length_scale),
            class_labels=class_labels,
            regional_standard_deviation=sigma,
        )
        return model[row, column]

    def objective(log_length_scale: float) -> float:
        """Return mean squared projected covariance residual."""
        return float(np.mean(np.square(target_pairs - model_pairs(log_length_scale))))

    result = cast(
        OptimizeResult,
        minimize_scalar(
            objective,
            method="bounded",
            bounds=(math.log(lower), math.log(upper)),
            options={"xatol": 1.0e-6, "maxiter": 40},
        ),
    )
    length_scale = math.exp(float(result.x))
    fitted_pairs = model_pairs(float(result.x))
    return _fit_diagnostics(
        target_pairs,
        fitted_pairs,
        length_scale=length_scale,
        converged=bool(result.success),
    )


def grouped_exponential_covariance(
    standard_deviation: npt.ArrayLike,
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    group_labels: npt.ArrayLike,
    length_scale: float,
) -> npt.NDArray[np.float64]:
    """Construct block-diagonal distance covariance for hard region classes.

    Args:
        standard_deviation: Positive marginal standard deviation per region.
        latitude: Representative latitude per region.
        longitude: Representative longitude per region.
        group_labels: One hard-class label per region. Regions in different
            classes have zero covariance.
        length_scale: Positive shared decay scale in degrees within each class.

    Returns:
        Group-constrained covariance matrix with the requested diagonal.

    Raises:
        ValueError: If inputs are invalid or have inconsistent lengths.
    """
    lat, lon = _validated_coordinates(latitude, longitude)
    sigma = np.asarray(standard_deviation, dtype=np.float64)
    groups = np.asarray(group_labels)
    if sigma.shape != (lat.size,) or groups.shape != (lat.size,):
        raise ValueError("standard_deviation and group_labels must contain one value per region.")
    if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
        raise ValueError("standard_deviation must be finite and positive.")
    correlation = separable_exponential_correlation(lat, lon, length_scale)
    correlation[groups[:, None] != groups[None, :]] = 0.0
    return sigma[:, None] * sigma[None, :] * correlation


def _validated_coordinates(
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return finite one-dimensional paired coordinates."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.shape != lon.shape:
        raise ValueError("latitude and longitude must be paired one-dimensional arrays.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitude and longitude must be finite.")
    return lat, lon


def _validated_grid_coordinates(
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return finite non-empty one-dimensional grid-axis coordinates."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size == 0 or lon.size == 0:
        raise ValueError("latitude and longitude must be non-empty one-dimensional grid axes.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitude and longitude must be finite.")
    return lat, lon


def _validated_projection(
    projection: npt.ArrayLike,
    *,
    grid_size: int,
) -> npt.NDArray[np.float64]:
    """Return a finite grid-by-region projection with non-empty columns.

    Args:
        projection: Candidate matrix representing a transposed restriction.
        grid_size: Required number of native-grid rows.

    Returns:
        Validated floating-point projection matrix.

    Raises:
        ValueError: If the matrix shape, entries, or columns are invalid.
    """
    values = np.asarray(projection, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != grid_size or values.shape[1] == 0:
        raise ValueError(f"projection must have shape ({grid_size}, K) with K positive.")
    if not np.isfinite(values).all():
        raise ValueError("projection must be finite.")
    if np.any(np.all(values == 0.0, axis=0)):
        raise ValueError("projection cannot contain an empty regional column.")
    return values


def _validated_length_scale_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Return finite, positive, increasing length-scale bounds."""
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("length_scale_bounds must be finite, positive, and increasing.")
    return lower, upper


def _regional_standard_deviation(
    target: npt.NDArray[np.float64],
    standard_deviation: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    """Return validated target-region marginal standard deviations.

    Args:
        target: Validated target covariance or correlation matrix.
        standard_deviation: Optional explicit regional standard deviations.

    Returns:
        One positive standard deviation per target region.

    Raises:
        ValueError: If the target diagonal or supplied values are invalid.
    """
    if standard_deviation is None:
        diagonal = np.diag(target)
        if (diagonal <= 0.0).any():
            raise ValueError("target_covariance must have a positive diagonal.")
        return np.sqrt(diagonal)
    sigma = np.asarray(standard_deviation, dtype=np.float64)
    if sigma.shape != (target.shape[0],):
        raise ValueError("standard_deviation must contain one value per region.")
    if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
        raise ValueError("standard_deviation must be finite and positive.")
    return sigma


def _covariance_to_correlation(covariance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a positive-diagonal covariance matrix to correlation."""
    diagonal = np.diag(covariance)
    if (diagonal <= 0.0).any():
        raise ValueError("projected covariance must have a positive diagonal.")
    sigma = np.sqrt(diagonal)
    correlation = covariance / (sigma[:, None] * sigma[None, :])
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _fit_diagnostics(
    target_pairs: npt.NDArray[np.float64],
    model_pairs: npt.NDArray[np.float64],
    *,
    length_scale: float,
    converged: bool,
) -> ExponentialLengthScaleFit:
    """Summarize residuals between paired target and model entries.

    Args:
        target_pairs: Unique off-diagonal target entries.
        model_pairs: Corresponding fitted model entries.
        length_scale: Fitted positive scale in coordinate units.
        converged: Whether the numerical optimizer reported success.

    Returns:
        Length-scale fit diagnostics.
    """
    residual = target_pairs - model_pairs
    target_norm = float(np.linalg.norm(target_pairs))
    model_variation = float(np.std(model_pairs))
    target_variation = float(np.std(target_pairs))
    if model_variation > 0.0 and target_variation > 0.0:
        target_model_correlation = float(np.corrcoef(target_pairs, model_pairs)[0, 1])
    else:
        target_model_correlation = None
    return ExponentialLengthScaleFit(
        length_scale=length_scale,
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        relative_rmse=float(np.linalg.norm(residual) / target_norm) if target_norm > 0.0 else 0.0,
        target_model_correlation=target_model_correlation,
        pair_count=int(target_pairs.size),
        converged=converged,
    )


def _validated_covariance(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a finite symmetric square covariance matrix."""
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("target_covariance must be square.")
    if not np.isfinite(values).all():
        raise ValueError("target_covariance must be finite.")
    if not np.allclose(values, values.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("target_covariance must be symmetric.")
    return values


__all__ = [
    "ExponentialLengthScaleFit",
    "fit_projected_exponential_length_scale",
    "fit_separable_exponential_length_scale",
    "grouped_exponential_covariance",
    "projected_exponential_covariance",
    "separable_exponential_correlation",
]
