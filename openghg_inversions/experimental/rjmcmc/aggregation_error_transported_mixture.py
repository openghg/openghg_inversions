"""Normalized transported Gaussian mixtures for aggregation error.

This module is the bounded NumPy foundation for a non-Gaussian successor to
the low-rank aggregation-error moment closure.  It deliberately contains no
mixture fitter and no PyMC integration.  A fitted latent mixture is first
post-centred and whitened to mean zero and covariance identity.  For projected
aggregation covariance ``S`` its principal symmetric square root transports
that latent density into the fixed error-whitened summary space.  Independent
Gaussian measurement noise is then convolved with every component exactly.

If ``r = D**(-1/2) @ (observation - mean)``, ``z = B.T @ r`` and
``r_perp = r - B @ z``, the normalized likelihood is

```
prod_i noise_sd[i]**(-1)
* phi_{n-q}(r_perp)
* sum_c weight[c] * phi_q(
    z; sqrt(S) @ component_mean[c],
    I + sqrt(S) @ component_covariance[c] @ sqrt(S),
)
```

The first implementation intentionally rejects nonzero singular ``S``.
Rank-zero summaries, exactly zero ``S``, and the literal one-component
standard normal mixture have dedicated exact paths.  The latter reproduces
the existing normalized low-rank Gaussian likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .aggregation_error_low_rank import low_rank_gaussian_log_likelihood

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "TransportedGaussianMixture",
    "TransportedMixtureFitterPolicy",
    "postcentre_whiten_gaussian_mixture",
    "principal_symmetric_psd_sqrt",
    "transported_mixture_log_likelihood",
    "transported_summary_moments",
]

_ARTIFACT_SCHEMA = "aggregation-transported-gaussian-mixture-v1"
_FITTER_POLICY_SCHEMA = "aggregation-transported-mixture-fitter-policy-v1"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_MAX_COMPONENT_CONDITION = 1.0e12


def _readonly_float(values: ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    """Return a finite, owned, read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _canonical_json(payload: object) -> str:
    """Return the strict canonical JSON serialization used for identities."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _symmetric(values: ArrayLike, *, name: str) -> FloatArray:
    """Validate one finite square matrix and return its symmetric part."""
    matrix = _readonly_float(values, name=name, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
    tolerance = float(512.0 * np.finfo(np.float64).eps * max(1, matrix.shape[0]) * scale)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    result = 0.5 * (matrix + matrix.T)
    result.setflags(write=False)
    return cast(FloatArray, result)


def _proper_covariance(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a numerically proper full covariance, rejecting collapse."""
    covariance = _symmetric(values, name=name)
    dimension = covariance.shape[0]
    if dimension == 0:
        return covariance
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    scale = max(1.0, abs(largest))
    tolerance = float(512.0 * np.finfo(np.float64).eps * max(1, dimension) * scale)
    if smallest <= tolerance:
        raise ValueError(f"{name} must be positive definite and not numerically collapsed.")
    if largest / smallest > _MAX_COMPONENT_CONDITION:
        raise ValueError(f"{name} condition number exceeds the {_MAX_COMPONENT_CONDITION:.0e} hard limit.")
    return covariance


def _normalized_weights(values: ArrayLike) -> FloatArray:
    """Validate and canonically normalize strictly positive mixture weights."""
    weights = _readonly_float(values, name="weights", ndim=1)
    if weights.size == 0 or np.any(weights <= 0.0):
        raise ValueError("weights must be a non-empty vector of strictly positive values.")
    total = math.fsum(float(value) for value in weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a finite positive total.")
    normalized = np.asarray(weights / total, dtype=np.float64)
    if np.any(normalized <= 0.0):
        raise ValueError("normalizing weights must not underflow a component to zero.")
    normalized.setflags(write=False)
    return cast(FloatArray, normalized)


def _mixture_moments(
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return exact first two central moments of a Gaussian mixture."""
    dimension = means.shape[1]
    mean = np.einsum("c,cq->q", weights, means, optimize=False)
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    for component, weight in enumerate(weights):
        offset = means[component] - mean
        covariance += float(weight) * (covariances[component] + np.outer(offset, offset))
    covariance = 0.5 * (covariance + covariance.T)
    mean.setflags(write=False)
    covariance.setflags(write=False)
    return cast(FloatArray, mean), cast(FloatArray, covariance)


def _component_sort_key(
    weight: float,
    mean: FloatArray,
    covariance: FloatArray,
) -> tuple[float, float, float, str, str]:
    """Return a deterministic key independent of caller component order."""
    return (
        -weight,
        float(np.linalg.norm(mean)),
        float(np.trace(covariance)),
        np.asarray(mean, dtype="<f8").tobytes(order="C").hex(),
        np.asarray(covariance, dtype="<f8").tobytes(order="C").hex(),
    )


def _canonical_components(
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sort components into the artifact's deterministic catalogue order."""
    order = sorted(
        range(weights.size),
        key=lambda index: _component_sort_key(
            float(weights[index]),
            means[index],
            covariances[index],
        ),
    )
    canonical_weights = np.array(weights[order], dtype=np.float64, copy=True)
    canonical_means = np.array(means[order], dtype=np.float64, copy=True)
    canonical_covariances = np.array(
        covariances[order],
        dtype=np.float64,
        copy=True,
    )
    # Division by an fsum-normalizer can leave the stored binary64 weights one
    # ULP below or above one in exact summation.  Close that residual on the
    # first component in canonical catalogue order.  Canonical JSON replay
    # then divides by literal one and is bit-idempotent.
    for _ in range(4):
        residual = 1.0 - math.fsum(float(value) for value in canonical_weights)
        if residual == 0.0:
            break
        adjusted = float(canonical_weights[0]) + residual
        if adjusted == float(canonical_weights[0]):
            adjusted = float(
                np.nextafter(
                    canonical_weights[0],
                    math.inf if residual > 0.0 else -math.inf,
                )
            )
        canonical_weights[0] = adjusted
    if canonical_weights[0] <= 0.0 or math.fsum(float(value) for value in canonical_weights) != 1.0:
        raise ValueError("canonical mixture weights could not be closed to exact unit total.")
    canonical_weights.setflags(write=False)
    canonical_means.setflags(write=False)
    canonical_covariances.setflags(write=False)
    return (
        cast(FloatArray, canonical_weights),
        cast(FloatArray, canonical_means),
        cast(FloatArray, canonical_covariances),
    )


def postcentre_whiten_gaussian_mixture(
    weights: ArrayLike,
    means: ArrayLike,
    covariances: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Post-centre and symmetrically whiten one proper Gaussian mixture.

    The total raw mixture covariance is diagonalized as
    ``V = Q diag(lambda) Q.T``.  The symmetric inverse square root
    ``Q diag(lambda**-1/2) Q.T`` is applied to every component mean and
    covariance.  This is an exact affine transformation of the fitted
    density, not a refit or component-wise standardization.
    """
    normalized = _normalized_weights(weights)
    raw_means = _readonly_float(means, name="means", ndim=2)
    raw_covariances = _readonly_float(
        covariances,
        name="covariances",
        ndim=3,
    )
    component_count, dimension = raw_means.shape
    if normalized.shape != (component_count,):
        raise ValueError("means must have one row per weight.")
    if raw_covariances.shape != (component_count, dimension, dimension):
        raise ValueError("covariances must contain one square matrix per component.")
    proper = np.empty_like(raw_covariances)
    for component in range(component_count):
        proper[component] = _proper_covariance(
            raw_covariances[component],
            name=f"covariances[{component}]",
        )
    normalized, raw_means, proper = _canonical_components(
        normalized,
        raw_means,
        proper,
    )
    if dimension == 0:
        if component_count != 1:
            raise ValueError("rank-zero mixtures must contain exactly one component.")
        return (
            np.asarray([1.0], dtype=np.float64),
            np.empty((1, 0), dtype=np.float64),
            np.empty((1, 0, 0), dtype=np.float64),
        )
    if component_count == 1:
        return (
            np.asarray([1.0], dtype=np.float64),
            np.zeros((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64)[np.newaxis, :, :],
        )

    mean, total_covariance = _mixture_moments(
        normalized,
        raw_means,
        proper,
    )
    total_covariance = _proper_covariance(
        total_covariance,
        name="total mixture covariance",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(total_covariance)
    inverse_sqrt = (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))[np.newaxis, :]) @ eigenvectors.T
    inverse_sqrt = 0.5 * (inverse_sqrt + inverse_sqrt.T)
    whitened_means = (inverse_sqrt @ (raw_means - mean).T).T
    whitened_covariances = np.empty_like(proper)
    for component in range(component_count):
        transformed = inverse_sqrt @ proper[component] @ inverse_sqrt
        whitened_covariances[component] = 0.5 * (transformed + transformed.T)
    return _canonical_components(
        normalized,
        np.asarray(whitened_means, dtype=np.float64),
        np.asarray(whitened_covariances, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TransportedGaussianMixture:
    """Immutable, canonical, moment-standardized latent mixture artifact."""

    weights: FloatArray = field(init=False)
    means: FloatArray = field(init=False)
    covariances: FloatArray = field(init=False)
    sha256: str = field(init=False)

    def __init__(
        self,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
    ) -> None:
        """Validate a pre-standardized mixture and canonicalize its components."""
        normalized = _normalized_weights(weights)
        owned_means = _readonly_float(means, name="means", ndim=2)
        raw_covariances = _readonly_float(
            covariances,
            name="covariances",
            ndim=3,
        )
        component_count, dimension = owned_means.shape
        if normalized.shape != (component_count,):
            raise ValueError("means must have one row per weight.")
        if raw_covariances.shape != (component_count, dimension, dimension):
            raise ValueError("covariances must contain one square matrix per component.")
        if dimension == 0 and component_count != 1:
            raise ValueError("rank-zero mixtures must contain exactly one component.")

        proper = np.empty_like(raw_covariances)
        for component in range(component_count):
            proper[component] = _proper_covariance(
                raw_covariances[component],
                name=f"covariances[{component}]",
            )
        canonical = _canonical_components(
            normalized,
            owned_means,
            proper,
        )
        canonical_weights, canonical_means, canonical_covariances = canonical
        if component_count == 1:
            if not np.array_equal(canonical_weights, np.asarray([1.0])):
                raise ValueError("a one-component standardized mixture must have literal weight one.")
            if not np.array_equal(
                canonical_means,
                np.zeros((1, dimension), dtype=np.float64),
            ):
                raise ValueError("a one-component standardized mixture must have literal zero mean.")
            if not np.array_equal(
                canonical_covariances,
                np.eye(dimension, dtype=np.float64)[np.newaxis, :, :],
            ):
                raise ValueError(
                    "a one-component standardized mixture must have literal identity covariance."
                )

        mixture_mean, mixture_covariance = _mixture_moments(
            canonical_weights,
            canonical_means,
            canonical_covariances,
        )
        tolerance = float(4096.0 * np.finfo(np.float64).eps * max(1, component_count, dimension))
        if not np.allclose(
            mixture_mean,
            np.zeros(dimension, dtype=np.float64),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("latent mixture mean must be zero after post-centring.")
        if not np.allclose(
            mixture_covariance,
            np.eye(dimension, dtype=np.float64),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("latent mixture covariance must be identity after whitening.")

        object.__setattr__(self, "weights", canonical_weights)
        object.__setattr__(self, "means", canonical_means)
        object.__setattr__(self, "covariances", canonical_covariances)
        object.__setattr__(self, "sha256", _sha256_text(self.to_json()))

    @classmethod
    def from_raw(
        cls,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
    ) -> TransportedGaussianMixture:
        """Create an artifact by exact post-centring and whitening."""
        standardized = postcentre_whiten_gaussian_mixture(
            weights,
            means,
            covariances,
        )
        return cls(*standardized)

    @classmethod
    def standard_normal(cls, dimension: int) -> TransportedGaussianMixture:
        """Return the literal one-component standard normal control."""
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be an integer.")
        normalized_dimension = int(dimension)
        if normalized_dimension < 0:
            raise ValueError("dimension must be non-negative.")
        return cls(
            np.asarray([1.0], dtype=np.float64),
            np.zeros((1, normalized_dimension), dtype=np.float64),
            np.eye(normalized_dimension, dtype=np.float64)[np.newaxis, :, :],
        )

    @property
    def component_count(self) -> int:
        """Return the number of mixture components."""
        return int(self.weights.size)

    @property
    def dimension(self) -> int:
        """Return the latent/summary dimension."""
        return int(self.means.shape[1])

    @property
    def is_standard_normal(self) -> bool:
        """Return whether this is the literal Gaussian-closure control."""
        return self.component_count == 1

    @property
    def payload(self) -> dict[str, object]:
        """Return the complete canonical JSON-compatible artifact payload."""
        return {
            "schema": _ARTIFACT_SCHEMA,
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covariances": self.covariances.tolist(),
        }

    def to_json(self) -> str:
        """Return the canonical artifact serialization."""
        return _canonical_json(self.payload)

    @classmethod
    def from_json(
        cls,
        serialized: str,
        *,
        expected_sha256: str | None = None,
    ) -> TransportedGaussianMixture:
        """Load a canonical artifact, optionally requiring an expected hash."""
        if not isinstance(serialized, str):
            raise TypeError("serialized must be a string.")
        observed_sha256 = _sha256_text(serialized)
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str):
                raise TypeError("expected_sha256 must be a string.")
            if observed_sha256 != expected_sha256:
                raise ValueError("serialized mixture SHA-256 does not match the expected identity.")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized mixture must be valid JSON.") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "weights",
            "means",
            "covariances",
        }:
            raise ValueError("serialized mixture has an unexpected schema or fields.")
        if payload["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError("serialized mixture schema is unsupported.")
        result = cls(
            payload["weights"],
            payload["means"],
            payload["covariances"],
        )
        if result.to_json() != serialized:
            raise ValueError("serialized mixture must use canonical JSON and component order.")
        if result.sha256 != observed_sha256:
            raise ValueError("serialized mixture identity did not replay exactly.")
        return result


@dataclass(frozen=True, slots=True)
class TransportedMixtureFitterPolicy:
    """Deterministic bounded fitter contract; fitting is intentionally external."""

    component_counts: tuple[int, ...] = (1, 2, 4, 8)
    restart_seeds: tuple[int, ...] = (1103, 2207, 3301, 4409, 5519)
    maximum_iterations: int = 500
    convergence_tolerance: float = 1.0e-8
    covariance_regularization: float = 1.0e-6
    minimum_component_weight: float = 1.0e-3
    maximum_component_condition: float = 1.0e8

    def __post_init__(self) -> None:
        """Validate that the future fitting search is finite and reproducible."""
        counts = tuple(
            _positive_integer(value, name="component_counts entry") for value in self.component_counts
        )
        if counts != tuple(sorted(set(counts))) or counts[0] != 1:
            raise ValueError("component_counts must be unique, increasing, and begin with one.")
        seeds = tuple(_positive_integer(value, name="restart_seeds entry") for value in self.restart_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("restart_seeds must be a non-empty tuple of unique positive integers.")
        maximum_iterations = _positive_integer(
            self.maximum_iterations,
            name="maximum_iterations",
        )
        for name in (
            "convergence_tolerance",
            "covariance_regularization",
            "minimum_component_weight",
            "maximum_component_condition",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Real):
                raise TypeError(f"{name} must be a non-Boolean real number.")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.minimum_component_weight >= 1.0:
            raise ValueError("minimum_component_weight must be less than one.")
        if self.maximum_component_condition > _MAX_COMPONENT_CONDITION:
            raise ValueError("maximum_component_condition cannot exceed the artifact hard limit.")
        object.__setattr__(self, "component_counts", counts)
        object.__setattr__(self, "restart_seeds", seeds)
        object.__setattr__(self, "maximum_iterations", maximum_iterations)

    @property
    def payload(self) -> dict[str, object]:
        """Return the complete deterministic fitter-policy payload."""
        return {
            "schema": _FITTER_POLICY_SCHEMA,
            "component_counts": list(self.component_counts),
            "restart_seeds": list(self.restart_seeds),
            "maximum_iterations": self.maximum_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "covariance_regularization": self.covariance_regularization,
            "minimum_component_weight": self.minimum_component_weight,
            "maximum_component_condition": self.maximum_component_condition,
        }

    @property
    def sha256(self) -> str:
        """Return the policy's canonical SHA-256 identity."""
        return _sha256_text(_canonical_json(self.payload))


def principal_symmetric_psd_sqrt(values: ArrayLike) -> FloatArray:
    """Return the principal symmetric PSD square root.

    Exactly zero and rank-zero matrices are supported.  Every other singular
    or numerically singular matrix is a hard stop in this first implementation.
    """
    matrix = _symmetric(values, name="summary_covariance")
    dimension = matrix.shape[0]
    if dimension == 0 or not np.any(matrix):
        return _readonly_float(
            np.zeros_like(matrix),
            name="summary_covariance_sqrt",
            ndim=2,
        )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(eigenvalues[-1])
    scale = max(1.0, abs(largest))
    tolerance = float(512.0 * np.finfo(np.float64).eps * max(1, dimension) * scale)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("summary_covariance must be positive semidefinite.")
    if float(eigenvalues[0]) <= tolerance:
        raise ValueError("nonzero singular or numerically singular summary_covariance is unsupported.")
    result = (eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    return _readonly_float(result, name="summary_covariance_sqrt", ndim=2)


def _validated_likelihood_inputs(
    observation: ArrayLike,
    mean: ArrayLike,
    noise_sd: ArrayLike,
    summary_basis: ArrayLike,
    summary_covariance: ArrayLike,
    mixture: TransportedGaussianMixture,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate the scientific likelihood arrays and artifact alignment."""
    if not isinstance(mixture, TransportedGaussianMixture):
        raise TypeError("mixture must be a TransportedGaussianMixture.")
    observed = _readonly_float(observation, name="observation", ndim=1)
    expected = _readonly_float(mean, name="mean", ndim=1)
    if observed.size == 0 or expected.shape != observed.shape:
        raise ValueError("observation and mean must be matching non-empty vectors.")
    raw_scale = np.asarray(noise_sd, dtype=np.float64)
    if raw_scale.ndim == 0:
        raw_scale = np.full(observed.size, float(raw_scale), dtype=np.float64)
    scale = _readonly_float(raw_scale, name="noise_sd", ndim=1)
    if scale.shape != observed.shape or np.any(scale <= 0.0):
        raise ValueError("noise_sd must be positive with one entry per observation.")
    basis = _readonly_float(summary_basis, name="summary_basis", ndim=2)
    if basis.shape[0] != observed.size or basis.shape[1] > observed.size:
        raise ValueError("summary_basis must have shape (observations, q) with q <= observations.")
    tolerance = float(256.0 * np.finfo(np.float64).eps * max(1, *basis.shape))
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[1], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("summary_basis columns must be orthonormal.")
    if mixture.dimension != basis.shape[1]:
        raise ValueError("mixture dimension must match the summary basis.")
    covariance = _symmetric(
        summary_covariance,
        name="summary_covariance",
    )
    if covariance.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("summary_covariance shape must match the summary dimension.")
    return observed, expected, scale, basis, covariance


def transported_mixture_log_likelihood(
    observation: ArrayLike,
    mean: ArrayLike,
    noise_sd: ArrayLike,
    summary_basis: ArrayLike,
    summary_covariance: ArrayLike,
    mixture: TransportedGaussianMixture,
) -> float:
    """Evaluate the normalized direct orthogonal-plus-mixture likelihood."""
    observed, expected, scale, basis, covariance = _validated_likelihood_inputs(
        observation,
        mean,
        noise_sd,
        summary_basis,
        summary_covariance,
        mixture,
    )
    dimension = mixture.dimension
    if dimension == 0:
        return low_rank_gaussian_log_likelihood(
            observed,
            expected,
            scale,
            basis,
            covariance,
        )
    if not np.any(covariance):
        return low_rank_gaussian_log_likelihood(
            observed,
            expected,
            scale,
            basis,
            covariance,
        )
    covariance_sqrt = principal_symmetric_psd_sqrt(covariance)
    if mixture.is_standard_normal:
        return low_rank_gaussian_log_likelihood(
            observed,
            expected,
            scale,
            basis,
            covariance,
        )

    whitened_residual = (observed - expected) / scale
    summary_residual = basis.T @ whitened_residual
    orthogonal_residual = whitened_residual - basis @ summary_residual
    component_log_densities = np.empty(mixture.component_count, dtype=np.float64)
    identity = np.eye(dimension, dtype=np.float64)
    for component in range(mixture.component_count):
        component_mean = covariance_sqrt @ mixture.means[component]
        component_covariance = identity + covariance_sqrt @ mixture.covariances[component] @ covariance_sqrt
        component_covariance = 0.5 * (component_covariance + component_covariance.T)
        try:
            cholesky = np.linalg.cholesky(component_covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "convolved component covariance is not numerically positive definite."
            ) from error
        displacement = summary_residual - component_mean
        solved = np.linalg.solve(cholesky, displacement)
        component_log_densities[component] = math.log(float(mixture.weights[component])) - 0.5 * (
            dimension * _LOG_TWO_PI + 2.0 * float(np.sum(np.log(np.diag(cholesky)))) + float(solved @ solved)
        )
    result = (
        -float(np.sum(np.log(scale)))
        - 0.5 * ((observed.size - dimension) * _LOG_TWO_PI + float(orthogonal_residual @ orthogonal_residual))
        + float(logsumexp(component_log_densities))
    )
    if not math.isfinite(result):
        raise ValueError("transported mixture log likelihood is non-finite.")
    return result


def transported_summary_moments(
    summary_covariance: ArrayLike,
    mixture: TransportedGaussianMixture,
) -> tuple[FloatArray, FloatArray]:
    """Return analytic mean/covariance after transport and measurement noise."""
    if not isinstance(mixture, TransportedGaussianMixture):
        raise TypeError("mixture must be a TransportedGaussianMixture.")
    covariance = _symmetric(
        summary_covariance,
        name="summary_covariance",
    )
    if covariance.shape != (mixture.dimension, mixture.dimension):
        raise ValueError("summary_covariance shape must match the mixture dimension.")
    if mixture.dimension == 0:
        return (
            _readonly_float(np.empty(0), name="summary_mean", ndim=1),
            _readonly_float(np.empty((0, 0)), name="summary_total_covariance", ndim=2),
        )
    if not np.any(covariance):
        return (
            _readonly_float(
                np.zeros(mixture.dimension),
                name="summary_mean",
                ndim=1,
            ),
            _readonly_float(
                np.eye(mixture.dimension),
                name="summary_total_covariance",
                ndim=2,
            ),
        )
    covariance_sqrt = principal_symmetric_psd_sqrt(covariance)
    latent_mean, latent_covariance = _mixture_moments(
        mixture.weights,
        mixture.means,
        mixture.covariances,
    )
    mean = covariance_sqrt @ latent_mean
    total_covariance = (
        np.eye(mixture.dimension, dtype=np.float64) + covariance_sqrt @ latent_covariance @ covariance_sqrt
    )
    total_covariance = 0.5 * (total_covariance + total_covariance.T)
    return (
        _readonly_float(mean, name="summary_mean", ndim=1),
        _readonly_float(
            total_covariance,
            name="summary_total_covariance",
            ndim=2,
        ),
    )
