"""Exact score identities for a standardized projected residual simulator.

For a retained root mass ``T``, projected unit-mass allocation residual
``xi``, independent standard-normal measurement noise ``epsilon``, and
projected unit-mass variances ``lambda``, define

```
s_j(T) = sqrt(1 + T**2 * lambda_j)
x_j = (T * xi_j + epsilon_j) / s_j(T).
```

This module contains only the exact algebra used to construct and validate
score-supervised conditional likelihoods.  The helpers accept either NumPy
arrays or JAX arrays, preserve the selected array backend, and use float64
throughout.  Concrete inputs are checked for finite values.  Value checks
which cannot be evaluated while JAX is tracing are deferred to the caller;
shape checks remain active because traced array shapes are static.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

FloatArray: TypeAlias = NDArray[np.float64] | jax.Array
FloatArrayLike: TypeAlias = ArrayLike | jax.Array

__all__ = [
    "component_observation_score",
    "component_partial_log_mass_score",
    "fixed_observation_log_mass_score",
    "standardization_scale",
    "standardization_scale_log_mass_derivative",
    "standardize_simulator_draw",
]


def _uses_jax(*values: object) -> bool:
    """Return whether any direct input is a JAX array or tracer."""
    return any(isinstance(value, jax.Array) for value in values)


def _namespace(*values: object) -> Any:
    """Return the JAX or NumPy array namespace for a group of inputs."""
    return jnp if _uses_jax(*values) else np


def _concrete_numpy(values: FloatArray) -> NDArray[np.float64] | None:
    """Return a concrete NumPy view, or ``None`` for JAX-managed arrays."""
    if isinstance(values, jax.Array):
        return None
    return np.asarray(values, dtype=np.float64)


def _float64_array(
    values: FloatArrayLike,
    *,
    name: str,
    namespace: Any,
) -> FloatArray:
    """Return one float64 array with concrete finiteness validation."""
    raw = namespace.asarray(values)
    if raw.dtype == namespace.bool_:
        raise TypeError(f"{name} must not be Boolean.")
    result = namespace.asarray(raw, dtype=namespace.float64)
    concrete = _concrete_numpy(result)
    if concrete is not None and not np.all(np.isfinite(concrete)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validated_eigenvalues(
    eigenvalues: FloatArrayLike,
    *,
    namespace: Any,
) -> FloatArray:
    """Return a non-empty vector of finite non-negative eigenvalues."""
    result = _float64_array(
        eigenvalues,
        name="eigenvalues",
        namespace=namespace,
    )
    if result.ndim != 1 or result.size < 1:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array.")
    concrete = _concrete_numpy(result)
    if concrete is not None and np.any(concrete < 0.0):
        raise ValueError("eigenvalues must be non-negative.")
    return result


def _validated_total_mass(
    total_mass: FloatArrayLike,
    *,
    namespace: Any,
) -> FloatArray:
    """Return finite non-negative scalar or batched retained-root masses."""
    result = _float64_array(
        total_mass,
        name="total_mass",
        namespace=namespace,
    )
    concrete = _concrete_numpy(result)
    if concrete is not None and np.any(concrete < 0.0):
        raise ValueError("total_mass must be non-negative.")
    return result


def _validated_coordinate_pair(
    first: FloatArrayLike,
    second: FloatArrayLike,
    eigenvalues: FloatArray,
    *,
    first_name: str,
    second_name: str,
    namespace: Any,
) -> tuple[FloatArray, FloatArray]:
    """Return equally shaped coordinate arrays with the declared final rank."""
    first_array = _float64_array(
        first,
        name=first_name,
        namespace=namespace,
    )
    second_array = _float64_array(
        second,
        name=second_name,
        namespace=namespace,
    )
    if first_array.shape != second_array.shape:
        raise ValueError(f"{first_name} and {second_name} must have identical shapes.")
    if first_array.ndim < 1 or first_array.shape[-1] != eigenvalues.size:
        raise ValueError(
            f"{first_name} and {second_name} must have eigenvalues.size coordinates on their final axis."
        )
    return first_array, second_array


def _validate_mass_batch_shape(
    total_mass: FloatArray,
    coordinate_shape: tuple[int, ...],
) -> None:
    """Require a scalar mass or one mass for every coordinate batch item."""
    leading_shape = coordinate_shape[:-1]
    if total_mass.ndim != 0 and total_mass.shape != leading_shape:
        raise ValueError("total_mass must be scalar or have the coordinate arrays' leading shape.")


def _expanded_mass(total_mass: FloatArray) -> FloatArray:
    """Append the projected-coordinate axis to scalar or batched masses."""
    return total_mass[..., None]


def standardization_scale(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
) -> FloatArray:
    """Return ``sqrt(1 + T**2 * lambda)`` in float64.

    A scalar mass returns a vector.  A mass array with shape ``batch_shape``
    returns ``batch_shape + (q,)``.  At zero mass the result is exactly one.
    """
    namespace = _namespace(total_mass, eigenvalues)
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    return namespace.sqrt(1.0 + namespace.square(_expanded_mass(mass)) * spectrum)


def standardization_scale_log_mass_derivative(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
) -> FloatArray:
    """Return ``d s(T) / d log(T) = T**2 * lambda / s(T)``.

    The continuous limiting derivative at ``T=0`` is returned as an exact
    zero without evaluating ``log(T)``.
    """
    namespace = _namespace(total_mass, eigenvalues)
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    squared_scaled_mass = namespace.square(_expanded_mass(mass)) * spectrum
    return squared_scaled_mass / namespace.sqrt(1.0 + squared_scaled_mass)


def standardize_simulator_draw(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
    allocation_residual: FloatArrayLike,
    gaussian_noise: FloatArrayLike,
) -> FloatArray:
    """Return the standardized simulator draw ``(T * xi + epsilon) / s``."""
    namespace = _namespace(
        total_mass,
        eigenvalues,
        allocation_residual,
        gaussian_noise,
    )
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    allocation, noise = _validated_coordinate_pair(
        allocation_residual,
        gaussian_noise,
        spectrum,
        first_name="allocation_residual",
        second_name="gaussian_noise",
        namespace=namespace,
    )
    _validate_mass_batch_shape(mass, allocation.shape)
    scale = namespace.sqrt(1.0 + namespace.square(_expanded_mass(mass)) * spectrum)
    return (_expanded_mass(mass) * allocation + noise) / scale


def component_observation_score(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
    gaussian_noise: FloatArrayLike,
) -> FloatArray:
    """Return the exact simulated component score ``-s(T) * epsilon``.

    This is the gradient with respect to the standardized coordinate ``x``.
    At zero mass it reduces exactly to the standard-normal score
    ``-epsilon``.
    """
    namespace = _namespace(total_mass, eigenvalues, gaussian_noise)
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    noise = _float64_array(
        gaussian_noise,
        name="gaussian_noise",
        namespace=namespace,
    )
    if noise.ndim < 1 or noise.shape[-1] != spectrum.size:
        raise ValueError("gaussian_noise must have eigenvalues.size coordinates on its final axis.")
    _validate_mass_batch_shape(mass, noise.shape)
    scale = namespace.sqrt(1.0 + namespace.square(_expanded_mass(mass)) * spectrum)
    return -scale * noise


def component_partial_log_mass_score(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
    allocation_residual: FloatArrayLike,
    gaussian_noise: FloatArrayLike,
    standardized_draw: FloatArrayLike,
) -> FloatArray:
    """Return the exact component score at fixed standardized coordinate.

    The returned value is

    ``epsilon.T @ (T * xi - dot(S) * x) + sum(dot(s) / s)``,

    where the dot denotes differentiation with respect to ``log(T)``.
    Leading sample dimensions are preserved and the final coordinate axis is
    reduced.  The score is exactly zero at ``T=0``.
    """
    namespace = _namespace(
        total_mass,
        eigenvalues,
        allocation_residual,
        gaussian_noise,
        standardized_draw,
    )
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    allocation, noise = _validated_coordinate_pair(
        allocation_residual,
        gaussian_noise,
        spectrum,
        first_name="allocation_residual",
        second_name="gaussian_noise",
        namespace=namespace,
    )
    standardized = _float64_array(
        standardized_draw,
        name="standardized_draw",
        namespace=namespace,
    )
    if standardized.shape != allocation.shape:
        raise ValueError("standardized_draw must have the same shape as allocation_residual.")
    _validate_mass_batch_shape(mass, allocation.shape)
    expanded_mass = _expanded_mass(mass)
    squared_scaled_mass = namespace.square(expanded_mass) * spectrum
    scale = namespace.sqrt(1.0 + squared_scaled_mass)
    dot_scale = squared_scaled_mass / scale
    return namespace.sum(
        noise * (expanded_mass * allocation - dot_scale * standardized) + dot_scale / scale,
        axis=-1,
    )


def fixed_observation_log_mass_score(
    total_mass: FloatArrayLike,
    eigenvalues: FloatArrayLike,
    standardized_observation: FloatArrayLike,
    partial_log_mass_score: FloatArrayLike,
    observation_score: FloatArrayLike,
) -> FloatArray:
    """Combine learned scores into the scientific fixed-observation score.

    ``partial_log_mass_score`` is the learned partial derivative at fixed
    standardized coordinate and ``observation_score`` is the learned
    ``x``-gradient.  The result applies both scale-chain-rule terms:

    ``partial_tau - observation_score.T @ (dot(S) / S * x)
                  - sum(dot(s) / s)``.

    At literal zero mass this helper returns the exact limiting value zero,
    independent of the supplied learned scores.  An evaluator can therefore
    branch before attempting to form the undefined conditioner ``log(T)``.
    """
    namespace = _namespace(
        total_mass,
        eigenvalues,
        standardized_observation,
        partial_log_mass_score,
        observation_score,
    )
    mass = _validated_total_mass(total_mass, namespace=namespace)
    spectrum = _validated_eigenvalues(eigenvalues, namespace=namespace)
    standardized, coordinate_score = _validated_coordinate_pair(
        standardized_observation,
        observation_score,
        spectrum,
        first_name="standardized_observation",
        second_name="observation_score",
        namespace=namespace,
    )
    _validate_mass_batch_shape(mass, standardized.shape)
    partial_score = _float64_array(
        partial_log_mass_score,
        name="partial_log_mass_score",
        namespace=namespace,
    )
    leading_shape = standardized.shape[:-1]
    if partial_score.shape != leading_shape:
        raise ValueError("partial_log_mass_score must have the coordinate arrays' leading shape.")

    expanded_mass = _expanded_mass(mass)
    squared_scaled_mass = namespace.square(expanded_mass) * spectrum
    scale_ratio = squared_scaled_mass / (1.0 + squared_scaled_mass)
    combined = (
        partial_score
        - namespace.sum(
            coordinate_score * scale_ratio * standardized,
            axis=-1,
        )
        - namespace.sum(scale_ratio, axis=-1)
    )
    return namespace.where(mass == 0.0, namespace.zeros_like(combined), combined)
