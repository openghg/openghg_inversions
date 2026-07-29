"""Tests for exact score-standardization identities."""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_regularized_flow import (
    component_observation_score,
    component_partial_log_mass_score,
    fixed_observation_log_mass_score,
    standardization_scale,
    standardization_scale_log_mass_derivative,
    standardize_simulator_draw,
)


@pytest.mark.parametrize("rank", [1, 4])
def test_component_scores_match_direct_jax_autodiff(rank: int) -> None:
    """Both simulator targets equal derivatives of the component log density."""
    eigenvalues = jnp.linspace(0.15, 1.2, rank, dtype=jnp.float64)
    allocation = jnp.linspace(-0.7, 0.9, rank, dtype=jnp.float64)
    noise = jnp.linspace(0.6, -0.4, rank, dtype=jnp.float64)
    total_mass = jnp.asarray(1.35, dtype=jnp.float64)
    standardized = standardize_simulator_draw(
        total_mass,
        eigenvalues,
        allocation,
        noise,
    )

    def component_log_density(
        coordinates: jax.Array,
        log_mass: jax.Array,
    ) -> jax.Array:
        mass = jnp.exp(log_mass)
        scale = jnp.sqrt(1.0 + mass**2 * eigenvalues)
        residual = scale * coordinates - mass * allocation
        return (
            -0.5 * jnp.dot(residual, residual) + jnp.log(scale).sum() - 0.5 * rank * math.log(2.0 * math.pi)
        )

    log_mass = jnp.log(total_mass)
    direct_observation = jax.grad(component_log_density, argnums=0)(
        standardized,
        log_mass,
    )
    direct_partial_mass = jax.grad(component_log_density, argnums=1)(
        standardized,
        log_mass,
    )

    np.testing.assert_allclose(
        component_observation_score(total_mass, eigenvalues, noise),
        direct_observation,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        component_partial_log_mass_score(
            total_mass,
            eigenvalues,
            allocation,
            noise,
            standardized,
        ),
        direct_partial_mass,
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def test_batched_standardization_and_scores_reconstruct_declared_identities() -> None:
    """Mass batches preserve shape and implement every formula coordinatewise."""
    masses = np.asarray([0.4, 1.8], dtype=np.float64)
    eigenvalues = np.asarray([0.0, 0.25, 1.5], dtype=np.float64)
    allocation = np.asarray(
        [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]],
        dtype=np.float64,
    )
    noise = np.asarray(
        [[-0.3, 0.8, 0.1], [0.5, -0.7, 0.9]],
        dtype=np.float64,
    )
    scale = standardization_scale(masses, eigenvalues)
    dot_scale = standardization_scale_log_mass_derivative(masses, eigenvalues)
    standardized = standardize_simulator_draw(
        masses,
        eigenvalues,
        allocation,
        noise,
    )

    expected_scale = np.sqrt(1.0 + masses[:, None] ** 2 * eigenvalues)
    expected_dot_scale = masses[:, None] ** 2 * eigenvalues / expected_scale
    expected_standardized = (masses[:, None] * allocation + noise) / expected_scale
    expected_partial = np.sum(
        noise * (masses[:, None] * allocation - expected_dot_scale * expected_standardized)
        + expected_dot_scale / expected_scale,
        axis=-1,
    )

    assert isinstance(scale, np.ndarray)
    assert scale.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(scale, expected_scale, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        dot_scale,
        expected_dot_scale,
        rtol=2.0e-16,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        standardized,
        expected_standardized,
        rtol=2.0e-16,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        component_observation_score(masses, eigenvalues, noise),
        -expected_scale * noise,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        component_partial_log_mass_score(
            masses,
            eigenvalues,
            allocation,
            noise,
            standardized,
        ),
        expected_partial,
        rtol=3.0e-16,
        atol=3.0e-16,
    )


@pytest.mark.parametrize("rank", [1, 3])
def test_fixed_observation_chain_rule_matches_autodiff_and_finite_difference(
    rank: int,
) -> None:
    """The combined score differentiates a transformed density at fixed y."""
    eigenvalues = jnp.linspace(0.2, 0.9, rank, dtype=jnp.float64)
    fixed_observation = jnp.linspace(-0.8, 1.1, rank, dtype=jnp.float64)
    direction = jnp.linspace(0.3, -0.25, rank, dtype=jnp.float64)
    log_mass = jnp.asarray(math.log(1.4), dtype=jnp.float64)

    def learned_log_density(
        coordinates: jax.Array,
        conditioner: jax.Array,
    ) -> jax.Array:
        centered = coordinates - conditioner * direction
        return (
            -0.5 * jnp.dot(centered, centered)
            - 0.5 * rank * math.log(2.0 * math.pi)
            + 0.17 * jnp.sin(conditioner)
        )

    mass = jnp.exp(log_mass)
    scale = standardization_scale(mass, eigenvalues)
    standardized = fixed_observation / scale
    partial_score = jax.grad(learned_log_density, argnums=1)(
        standardized,
        log_mass,
    )
    observation_score = jax.grad(learned_log_density, argnums=0)(
        standardized,
        log_mass,
    )
    combined = fixed_observation_log_mass_score(
        mass,
        eigenvalues,
        standardized,
        partial_score,
        observation_score,
    )

    def transformed_log_density(conditioner: jax.Array) -> jax.Array:
        local_mass = jnp.exp(conditioner)
        local_scale = jnp.sqrt(1.0 + local_mass**2 * eigenvalues)
        coordinates = fixed_observation / local_scale
        return learned_log_density(coordinates, conditioner) - jnp.log(local_scale).sum()

    direct = jax.grad(transformed_log_density)(log_mass)
    step = 1.0e-5
    finite_difference = (
        transformed_log_density(log_mass + step) - transformed_log_density(log_mass - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(combined, direct, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        combined,
        finite_difference,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_zero_mass_limits_are_literal_and_do_not_evaluate_log_mass() -> None:
    """The standard-normal and vanishing log-mass-score limits are exact."""
    eigenvalues = np.asarray([0.3, 2.0], dtype=np.float64)
    allocation = np.asarray([7.0, -9.0], dtype=np.float64)
    noise = np.asarray([0.25, -1.5], dtype=np.float64)
    standardized = standardize_simulator_draw(
        0.0,
        eigenvalues,
        allocation,
        noise,
    )

    np.testing.assert_array_equal(
        standardization_scale(0.0, eigenvalues),
        np.ones(2, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        standardization_scale_log_mass_derivative(0.0, eigenvalues),
        np.zeros(2, dtype=np.float64),
    )
    np.testing.assert_array_equal(standardized, noise)
    np.testing.assert_array_equal(
        component_observation_score(0.0, eigenvalues, noise),
        -noise,
    )
    assert (
        component_partial_log_mass_score(
            0.0,
            eigenvalues,
            allocation,
            noise,
            standardized,
        )
        == 0.0
    )
    assert (
        fixed_observation_log_mass_score(
            0.0,
            eigenvalues,
            standardized,
            123.0,
            np.asarray([4.0, -6.0]),
        )
        == 0.0
    )


def test_helpers_are_jittable_with_traced_mass_and_draws() -> None:
    """Concrete validation does not obstruct JAX tracing."""
    eigenvalues = jnp.asarray([0.1, 0.8], dtype=jnp.float64)

    @jax.jit
    def targets(
        mass: jax.Array,
        allocation: jax.Array,
        noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        standardized = standardize_simulator_draw(
            mass,
            eigenvalues,
            allocation,
            noise,
        )
        observation = component_observation_score(mass, eigenvalues, noise)
        partial = component_partial_log_mass_score(
            mass,
            eigenvalues,
            allocation,
            noise,
            standardized,
        )
        return jnp.asarray(observation), jnp.asarray(partial)

    observation, partial = targets(
        jnp.asarray(0.75, dtype=jnp.float64),
        jnp.asarray([0.4, -0.1], dtype=jnp.float64),
        jnp.asarray([-0.2, 0.6], dtype=jnp.float64),
    )
    assert observation.shape == (2,)
    assert observation.dtype == jnp.float64
    assert partial.shape == ()
    assert partial.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(observation)))
    assert math.isfinite(float(partial))


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: standardization_scale(-0.1, [1.0]),
            "total_mass must be non-negative",
        ),
        (
            lambda: standardization_scale(1.0, [-0.1]),
            "eigenvalues must be non-negative",
        ),
        (
            lambda: standardization_scale(1.0, []),
            "eigenvalues must be a non-empty one-dimensional array",
        ),
        (
            lambda: standardization_scale(1.0, [[0.2]]),
            "eigenvalues must be a non-empty one-dimensional array",
        ),
        (
            lambda: standardization_scale(1.0, [np.inf]),
            "eigenvalues must contain only finite values",
        ),
        (
            lambda: standardize_simulator_draw(
                1.0,
                [0.2, 0.3],
                [0.1],
                [0.4],
            ),
            "eigenvalues.size coordinates",
        ),
        (
            lambda: standardize_simulator_draw(
                1.0,
                [0.2],
                [[0.1], [0.2]],
                [[0.4]],
            ),
            "must have identical shapes",
        ),
        (
            lambda: standardize_simulator_draw(
                [1.0, 2.0, 3.0],
                [0.2],
                [[0.1], [0.2]],
                [[0.4], [0.5]],
            ),
            "total_mass must be scalar or have",
        ),
        (
            lambda: component_partial_log_mass_score(
                1.0,
                [0.2],
                [0.1],
                [0.4],
                [[0.5]],
            ),
            "standardized_draw must have the same shape",
        ),
        (
            lambda: fixed_observation_log_mass_score(
                1.0,
                [0.2, 0.3],
                [[0.1, 0.2], [0.3, 0.4]],
                0.5,
                [[-0.1, -0.2], [-0.3, -0.4]],
            ),
            "partial_log_mass_score must have",
        ),
        (
            lambda: component_observation_score(1.0, [0.2], [True]),
            "gaussian_noise must not be Boolean",
        ),
    ],
)
def test_invalid_shapes_and_values_are_rejected(
    operation: Callable[[], object],
    message: str,
) -> None:
    """Score helpers reject malformed or non-finite scientific inputs."""
    with pytest.raises((TypeError, ValueError), match=message):
        operation()
