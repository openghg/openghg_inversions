"""Focused tests for score-regularized conditional-flow training."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
from flowjax.train import fit_to_data
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import paramax  # noqa: E402
import pytest  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    RawLogMassScoreLoss,
    conditional_log_prob_and_observation_score,
    gamma_log_mass_conditioning,
    make_score_regularized_conditional_flow,
    raw_log_mass_condition_score,
)


def _partition(model: Any) -> tuple[Any, Any]:
    return eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )


def _inexact_leaves(model: Any) -> list[jax.Array]:
    return [leaf for leaf in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(leaf)]


def test_gamma_log_mass_conditioning_is_analytic_and_source_independent() -> None:
    center, scale = gamma_log_mass_conditioning(
        43.742615510366136,
        43.742615510366136,
    )
    assert center == pytest.approx(-0.011474050604083352, abs=1e-17)
    assert scale == pytest.approx(0.15206677909671867, abs=1e-17)
    with pytest.raises(ValueError, match="shape must be positive"):
        gamma_log_mass_conditioning(0.0, 1.0)
    with pytest.raises(ValueError, match="rate must be positive"):
        gamma_log_mass_conditioning(1.0, 0.0)


@pytest.mark.parametrize("dimension", [1, 2])
def test_conditional_flow_is_deterministic_float64_and_finite(dimension: int) -> None:
    first = make_score_regularized_conditional_flow(dimension, source_seed=541)
    second = make_score_regularized_conditional_flow(dimension, source_seed=541)
    first_leaves = _inexact_leaves(first)
    second_leaves = _inexact_leaves(second)

    assert first_leaves
    assert len(first_leaves) == len(second_leaves)
    for left, right in zip(first_leaves, second_leaves, strict=True):
        assert left.dtype == jnp.float64
        np.testing.assert_array_equal(left, right)

    condition = jnp.asarray([0.25], dtype=jnp.float64)
    draw = first.sample(jr.key(19), condition=condition)
    assert draw.shape == (dimension,)
    assert bool(jnp.all(jnp.isfinite(draw)))
    assert bool(jnp.isfinite(first.log_prob(draw, condition)))


def test_one_dimensional_flow_density_numerically_normalizes() -> None:
    flow = make_score_regularized_conditional_flow(1, source_seed=77)
    condition = jnp.asarray([-0.4], dtype=jnp.float64)
    grid = jnp.linspace(-9.0, 9.0, 4001, dtype=jnp.float64)
    points = grid[:, None]
    conditions = jnp.broadcast_to(condition, (grid.size, 1))
    density = jnp.exp(flow.log_prob(points, conditions))

    integral = jnp.trapezoid(density, grid)
    assert float(integral) == pytest.approx(1.0, abs=2e-5)


def test_two_dimensional_flow_density_numerically_normalizes() -> None:
    flow = make_score_regularized_conditional_flow(2, source_seed=79)
    condition = jnp.asarray([0.3], dtype=jnp.float64)
    grid = jnp.linspace(-7.0, 7.0, 121, dtype=jnp.float64)
    first, second = jnp.meshgrid(grid, grid, indexing="ij")
    points = jnp.stack((first.ravel(), second.ravel()), axis=-1)
    conditions = jnp.broadcast_to(condition, (points.shape[0], 1))
    density = jnp.exp(flow.log_prob(points, conditions)).reshape(
        grid.size,
        grid.size,
    )

    integral = jnp.trapezoid(
        jnp.trapezoid(density, grid, axis=1),
        grid,
    )
    assert float(integral) == pytest.approx(1.0, abs=5e-4)


@pytest.mark.parametrize("dimension", [1, 2])
def test_score_loss_is_finite_for_frozen_flows(dimension: int) -> None:
    flow = make_score_regularized_conditional_flow(dimension, source_seed=819)
    params, static = _partition(flow)
    raw_tau = jnp.asarray([-0.7, 0.2], dtype=jnp.float64)
    conditions = ((raw_tau - 0.1) / 1.3)[:, None]
    sample_keys = jr.split(jr.key(23), raw_tau.size)
    projected = jax.vmap(flow.sample)(sample_keys, condition=conditions)
    target_score = jnp.asarray([0.3, -0.4], dtype=jnp.float64)
    loss = RawLogMassScoreLoss(
        condition_center=0.1,
        condition_scale=1.3,
    )

    value = loss(
        params,
        static,
        projected,
        raw_tau,
        target_score,
        key=jr.key(5),
    )
    assert value.dtype == jnp.float64
    assert bool(jnp.isfinite(value))


def test_actual_flow_mixed_parameter_gradients_are_finite_and_score_sensitive() -> None:
    flow = make_score_regularized_conditional_flow(1, source_seed=823)
    params, static = _partition(flow)
    projected = jnp.asarray([[-1.0], [-0.2], [0.6], [1.4]], dtype=jnp.float64)
    raw_tau = jnp.asarray([-0.8, -0.1, 0.5, 1.1], dtype=jnp.float64)
    target_score = jnp.asarray([-0.4, 0.2, 0.7, -0.3], dtype=jnp.float64)
    center = 0.1
    condition_scale = 1.2
    loss = RawLogMassScoreLoss(
        condition_center=center,
        condition_scale=condition_scale,
    )
    _, composite_gradient = eqx.filter_value_and_grad(loss)(
        params,
        static,
        projected,
        raw_tau,
        target_score,
        key=jr.key(7),
    )

    def nll_only(local_params: Any, local_static: Any) -> jax.Array:
        distribution = paramax.unwrap(eqx.combine(local_params, local_static))
        conditions = ((raw_tau - center) / condition_scale)[:, None]
        return -jnp.mean(distribution.log_prob(projected, conditions))

    nll_gradient = eqx.filter_grad(nll_only)(params, static)
    composite_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(composite_gradient) if eqx.is_inexact_array(leaf)
    ]
    nll_leaves = [leaf for leaf in jax.tree_util.tree_leaves(nll_gradient) if eqx.is_inexact_array(leaf)]
    assert composite_leaves
    assert len(composite_leaves) == len(nll_leaves)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in composite_leaves)
    assert any(
        bool(jnp.any(jnp.abs(composite - nll) > 1e-14))
        for composite, nll in zip(composite_leaves, nll_leaves, strict=True)
    )


class _KnownConditionalNormal(eqx.Module):
    coefficient: jax.Array
    log_scale: jax.Array

    def log_prob(
        self,
        value: jax.Array,
        condition: jax.Array,
    ) -> jax.Array:
        scale = jnp.exp(self.log_scale)
        residual = value[..., 0] - self.coefficient * condition[..., 0]
        return -0.5 * jnp.square(residual / scale) - self.log_scale - 0.5 * math.log(2.0 * math.pi)


def test_raw_score_matches_known_conditional_normal_derivative() -> None:
    model = _KnownConditionalNormal(
        coefficient=jnp.asarray(1.7, dtype=jnp.float64),
        log_scale=jnp.asarray(math.log(0.8), dtype=jnp.float64),
    )
    projected = jnp.asarray([[0.2], [-1.1], [1.8]], dtype=jnp.float64)
    raw_tau = jnp.asarray([-0.4, 0.5, 1.2], dtype=jnp.float64)
    center = 0.3
    condition_scale = 1.4
    standardized = (raw_tau - center) / condition_scale
    residual = projected[:, 0] - model.coefficient * standardized
    expected = model.coefficient * residual / (jnp.exp(2.0 * model.log_scale) * condition_scale)

    actual = raw_log_mass_condition_score(
        model,
        projected,
        raw_tau,
        condition_center=center,
        condition_scale=condition_scale,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_forward_mode_raw_score_matches_direct_reverse_mode_on_flow() -> None:
    flow = make_score_regularized_conditional_flow(2, source_seed=829)
    projected = jnp.asarray(
        [[-1.0, 0.4], [-0.2, 1.1], [0.7, -0.5]],
        dtype=jnp.float64,
    )
    raw_tau = jnp.asarray([-0.8, 0.1, 0.9], dtype=jnp.float64)
    center = 0.15
    condition_scale = 1.3

    actual = raw_log_mass_condition_score(
        flow,
        projected,
        raw_tau,
        condition_center=center,
        condition_scale=condition_scale,
    )

    def reverse_mode_score(target: jax.Array, tau: jax.Array) -> jax.Array:
        def log_prob(local_tau: jax.Array) -> jax.Array:
            condition = jnp.asarray(
                [(local_tau - center) / condition_scale],
                dtype=jnp.float64,
            )
            return flow.log_prob(target, condition)

        return jax.grad(log_prob)(tau)

    expected = jax.vmap(reverse_mode_score)(projected, raw_tau)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("dimension", [1, 3])
def test_forward_coordinate_observation_score_matches_reverse_mode(
    dimension: int,
) -> None:
    flow = make_score_regularized_conditional_flow(
        dimension,
        source_seed=831 + dimension,
    )
    projected = jnp.reshape(
        jnp.linspace(
            -0.8,
            0.9,
            2 * dimension,
            dtype=jnp.float64,
        ),
        (2, dimension),
    )
    raw_tau = jnp.asarray((-0.35, 0.55), dtype=jnp.float64)
    center = 0.1
    condition_scale = 1.25
    actual_log_prob, actual_score = conditional_log_prob_and_observation_score(
        flow,
        projected,
        raw_tau,
        condition_center=center,
        condition_scale=condition_scale,
    )
    conditions = ((raw_tau - center) / condition_scale)[:, None]
    expected_log_prob = jax.vmap(flow.log_prob)(projected, conditions)
    expected_score = jax.vmap(jax.grad(flow.log_prob))(
        projected,
        conditions,
    )
    np.testing.assert_allclose(
        actual_log_prob,
        expected_log_prob,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        actual_score,
        expected_score,
        rtol=2e-12,
        atol=2e-12,
    )


def test_forward_mode_mixed_parameter_gradient_matches_prior_schedule() -> None:
    flow = make_score_regularized_conditional_flow(1, source_seed=839)
    params, static = _partition(flow)
    projected = jnp.asarray([[-0.7], [0.9]], dtype=jnp.float64)
    raw_tau = jnp.asarray([-0.45, 0.65], dtype=jnp.float64)
    target_score = jnp.asarray([0.2, -0.35], dtype=jnp.float64)
    center = 0.1
    condition_scale = 1.25
    current_loss = RawLogMassScoreLoss(
        condition_center=center,
        condition_scale=condition_scale,
    )
    current_value, current_gradient = eqx.filter_value_and_grad(current_loss)(
        params,
        static,
        projected,
        raw_tau,
        target_score,
    )

    def prior_schedule_loss(
        local_params: Any,
        local_static: Any,
    ) -> jax.Array:
        distribution = paramax.unwrap(eqx.combine(local_params, local_static))
        conditions = ((raw_tau - center) / condition_scale)[:, None]
        log_probabilities = distribution.log_prob(projected, conditions)

        def log_prob_one(
            target: jax.Array,
            tau: jax.Array,
        ) -> jax.Array:
            condition = jnp.asarray(
                [(tau - center) / condition_scale],
                dtype=jnp.float64,
            )
            return distribution.log_prob(target, condition)

        predicted_score = jax.vmap(jax.grad(log_prob_one, argnums=1))(projected, raw_tau)
        return -jnp.mean(log_probabilities) + jnp.mean(jnp.square(predicted_score - target_score))

    prior_value, prior_gradient = eqx.filter_value_and_grad(prior_schedule_loss)(params, static)
    np.testing.assert_allclose(
        current_value,
        prior_value,
        rtol=2e-13,
        atol=2e-13,
    )
    current_leaves = _inexact_leaves(current_gradient)
    prior_leaves = _inexact_leaves(prior_gradient)
    assert len(current_leaves) == len(prior_leaves)
    for current, prior in zip(current_leaves, prior_leaves, strict=True):
        np.testing.assert_allclose(
            current,
            prior,
            rtol=2e-11,
            atol=2e-12,
        )


def test_composite_parameter_gradient_matches_independent_reference() -> None:
    model = _KnownConditionalNormal(
        coefficient=jnp.asarray(0.43, dtype=jnp.float64),
        log_scale=jnp.asarray(math.log(1.2), dtype=jnp.float64),
    )
    projected = jnp.asarray([[-0.9], [-0.1], [0.8], [1.6]], dtype=jnp.float64)
    raw_tau = jnp.asarray([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float64)
    target_score = jnp.asarray([-0.2, 0.15, 0.55, 0.9], dtype=jnp.float64)
    center = 0.2
    condition_scale = 1.5
    loss = RawLogMassScoreLoss(
        condition_center=center,
        condition_scale=condition_scale,
    )
    params, static = _partition(model)
    actual_value, actual_gradient = eqx.filter_value_and_grad(loss)(
        params,
        static,
        projected,
        raw_tau,
        target_score,
        key=jr.key(2),
    )

    def reference(
        coefficient: jax.Array,
        log_scale: jax.Array,
    ) -> jax.Array:
        standardized_tau = (raw_tau - center) / condition_scale
        variance = jnp.exp(2.0 * log_scale)
        residual = projected[:, 0] - coefficient * standardized_tau
        log_probabilities = -0.5 * jnp.square(residual) / variance - log_scale - 0.5 * math.log(2.0 * math.pi)
        predicted_score = coefficient * residual / (variance * condition_scale)
        return -jnp.mean(log_probabilities) + jnp.mean(jnp.square(predicted_score - target_score))

    reference_value, reference_gradient = jax.value_and_grad(
        reference,
        argnums=(0, 1),
    )(model.coefficient, model.log_scale)

    np.testing.assert_allclose(actual_value, reference_value, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        actual_gradient.coefficient,
        reference_gradient[0],
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        actual_gradient.log_scale,
        reference_gradient[1],
        rtol=3e-13,
        atol=3e-13,
    )

    def nll_only(coefficient: jax.Array) -> jax.Array:
        standardized_tau = (raw_tau - center) / condition_scale
        variance = jnp.exp(2.0 * model.log_scale)
        residual = projected[:, 0] - coefficient * standardized_tau
        return jnp.mean(
            0.5 * jnp.square(residual) / variance + model.log_scale + 0.5 * math.log(2.0 * math.pi)
        )

    assert not np.isclose(
        float(actual_gradient.coefficient),
        float(jax.grad(nll_only)(model.coefficient)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_fit_to_data_takes_step_through_mixed_score_derivative() -> None:
    model = _KnownConditionalNormal(
        coefficient=jnp.asarray(0.35, dtype=jnp.float64),
        log_scale=jnp.asarray(math.log(1.1), dtype=jnp.float64),
    )
    raw_tau = jnp.asarray([-1.0, -0.25, 0.5, 1.25], dtype=jnp.float64)
    projected = jnp.asarray([[-0.9], [-0.1], [0.8], [1.6]], dtype=jnp.float64)
    target_score = jnp.asarray([-0.2, 0.15, 0.55, 0.9], dtype=jnp.float64)
    loss = RawLogMassScoreLoss(
        condition_center=0.2,
        condition_scale=1.5,
    )

    params, static = _partition(model)
    value, gradients = eqx.filter_value_and_grad(loss)(
        params,
        static,
        projected,
        raw_tau,
        target_score,
        key=jr.key(1),
    )
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradients.coefficient))
    assert float(jnp.abs(gradients.coefficient)) > 0.0

    fitted, losses = fit_to_data(
        jr.key(404),
        model,
        data=(projected, raw_tau, target_score),
        loss_fn=loss,
        learning_rate=5e-4,
        max_epochs=1,
        max_patience=0,
        batch_size=2,
        val_prop=0.5,
        return_best=True,
        show_progress=False,
    )
    assert losses.keys() == {"train", "val"}
    assert len(losses["train"]) == len(losses["val"]) == 1
    assert np.all(np.isfinite([losses["train"][0], losses["val"][0]]))
    assert fitted.coefficient.dtype == jnp.float64
    assert float(fitted.coefficient) != float(model.coefficient)
