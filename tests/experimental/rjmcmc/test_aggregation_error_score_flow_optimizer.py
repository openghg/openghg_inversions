"""Focused tests for deterministic score-microbatch optimization."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import paramax  # noqa: E402
import pytest  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_optimizer import (  # noqa: E402
    ScoreRegularizedFitHistory,
    fit_score_regularized_flow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    RawLogMassScoreLoss,
)


class _KnownConditionalNormal(eqx.Module):
    coefficient: jax.Array
    log_scale: jax.Array
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, coefficient: float, log_scale: float) -> None:
        self.coefficient = jnp.asarray(coefficient, dtype=jnp.float64)
        self.log_scale = jnp.asarray(log_scale, dtype=jnp.float64)
        self.shape = (1,)

    def log_prob(
        self,
        value: jax.Array,
        condition: jax.Array,
    ) -> jax.Array:
        scale = jnp.exp(self.log_scale)
        residual = value[..., 0] - self.coefficient * condition[..., 0]
        return -0.5 * jnp.square(residual / scale) - self.log_scale - 0.5 * math.log(2.0 * math.pi)


def _partition(model: Any) -> tuple[Any, Any]:
    return eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )


def _data() -> tuple[jax.Array, jax.Array, jax.Array]:
    projected = jnp.asarray(
        [[-1.4], [-0.8], [-0.1], [0.3], [0.7], [1.2], [1.8], [2.4]],
        dtype=jnp.float64,
    )
    raw_log_mass = jnp.asarray(
        [-1.3, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.4],
        dtype=jnp.float64,
    )
    target_score = jnp.asarray(
        [-0.7, -0.3, 0.0, 0.25, 0.45, 0.65, 0.9, 1.2],
        dtype=jnp.float64,
    )
    return projected, raw_log_mass, target_score


def test_one_outer_batch_matches_one_adam_update_of_mean_microgradients() -> None:
    key = jr.key(1403)
    model = _KnownConditionalNormal(0.31, math.log(1.15))
    projected, raw_log_mass, target_score = _data()
    fitted, history = fit_score_regularized_flow(
        key,
        model,
        projected,
        raw_log_mass,
        target_score,
        condition_center=0.15,
        condition_scale=1.25,
        learning_rate=5e-4,
        batch_size=4,
        score_microbatch_size=2,
        val_prop=0.5,
        max_epochs=1,
        patience=0,
    )

    split_key, train_shuffle_root, _, loss_key_root = jr.split(key, 4)
    split_indices = jr.permutation(split_key, projected.shape[0])
    training_indices = split_indices[:4]
    train_permutation = jr.permutation(
        jr.fold_in(train_shuffle_root, 0),
        4,
    )
    indices = training_indices[train_permutation]
    params, static = _partition(model)
    loss = RawLogMassScoreLoss(
        condition_center=0.15,
        condition_scale=1.25,
    )
    gradients = []
    for micro_index in range(2):
        micro_indices = indices[2 * micro_index : 2 * (micro_index + 1)]
        loss_key = jr.fold_in(
            jr.fold_in(
                jr.fold_in(loss_key_root, 0),
                0,
            ),
            micro_index,
        )
        _, gradient = eqx.filter_value_and_grad(loss)(
            params,
            static,
            projected[micro_indices],
            raw_log_mass[micro_indices],
            target_score[micro_indices],
            key=loss_key,
        )
        gradients.append(gradient)
    mean_gradient = jax.tree_util.tree_map(
        lambda left, right: (left + right) / 2.0,
        gradients[0],
        gradients[1],
    )
    optimizer = optax.adam(5e-4)
    optimizer_state = optimizer.init(params)
    updates, _ = optimizer.update(
        mean_gradient,
        optimizer_state,
        params=params,
    )
    expected = eqx.combine(eqx.apply_updates(params, updates), static)

    np.testing.assert_allclose(
        fitted.coefficient,
        expected.coefficient,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        fitted.log_scale,
        expected.log_scale,
        rtol=0.0,
        atol=2e-15,
    )
    assert history.best_epoch == 0
    assert len(history.train) == len(history.validation) == 1


def test_fit_replays_bitwise_with_identical_key() -> None:
    model = _KnownConditionalNormal(0.22, math.log(0.95))
    data = _data()
    arguments = {
        "condition_center": -0.1,
        "condition_scale": 1.4,
        "batch_size": 4,
        "score_microbatch_size": 2,
        "val_prop": 0.5,
        "max_epochs": 2,
        "patience": 2,
    }
    first_model, first_history = fit_score_regularized_flow(
        jr.key(907),
        model,
        *data,
        **arguments,
    )
    second_model, second_history = fit_score_regularized_flow(
        jr.key(907),
        model,
        *data,
        **arguments,
    )

    np.testing.assert_array_equal(
        first_model.coefficient,
        second_model.coefficient,
    )
    np.testing.assert_array_equal(first_model.log_scale, second_model.log_scale)
    assert first_history == second_history


def test_validation_uses_weighted_final_remainder_microbatch() -> None:
    projected, raw_log_mass, target_score = _data()
    projected = jnp.concatenate([projected, jnp.asarray([[2.9], [3.3]], dtype=jnp.float64)])
    raw_log_mass = jnp.concatenate([raw_log_mass, jnp.asarray([1.8, 2.2], dtype=jnp.float64)])
    target_score = jnp.concatenate([target_score, jnp.asarray([1.4, 1.7], dtype=jnp.float64)])

    _, history = fit_score_regularized_flow(
        jr.key(811),
        _KnownConditionalNormal(0.2, 0.0),
        projected,
        raw_log_mass,
        target_score,
        condition_center=0.0,
        condition_scale=1.0,
        batch_size=4,
        score_microbatch_size=2,
        val_prop=0.3,
        max_epochs=1,
    )

    assert len(history.validation) == 1
    assert math.isfinite(history.validation[0])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"projected_dtype": jnp.float32}, "projected must have dtype float64"),
        ({"target_shape": (2, 4)}, "target_score must have shape"),
        ({"batch_size": 3}, "training effective outer batch size"),
        ({"microbatch_size": 3}, "training effective outer batch size"),
        ({"val_prop": 0.01}, "non-empty train and validation"),
    ],
)
def test_malformed_data_and_divisibility_fail(
    change: dict[str, Any],
    match: str,
) -> None:
    projected, raw_log_mass, target_score = _data()
    if "projected_dtype" in change:
        projected = projected.astype(change["projected_dtype"])
    if "target_shape" in change:
        target_score = target_score.reshape(change["target_shape"])
    batch_size = change.get("batch_size", 4)
    microbatch_size = change.get("microbatch_size", 2)
    val_prop = change.get("val_prop", 0.5)

    with pytest.raises(ValueError, match=match):
        fit_score_regularized_flow(
            jr.key(18),
            _KnownConditionalNormal(0.2, 0.0),
            projected,
            raw_log_mass,
            target_score,
            condition_center=0.0,
            condition_scale=1.0,
            batch_size=batch_size,
            score_microbatch_size=microbatch_size,
            val_prop=val_prop,
            max_epochs=1,
        )


def test_history_shape_and_best_epoch_are_consistent() -> None:
    fitted, history = fit_score_regularized_flow(
        jr.key(505),
        _KnownConditionalNormal(-0.15, math.log(1.3)),
        *_data(),
        condition_center=0.0,
        condition_scale=1.0,
        learning_rate=2e-3,
        batch_size=4,
        score_microbatch_size=2,
        val_prop=0.5,
        max_epochs=4,
        patience=1,
    )

    assert isinstance(history, ScoreRegularizedFitHistory)
    assert 1 <= len(history.train) <= 4
    assert len(history.validation) == len(history.train)
    assert 0 <= history.best_epoch < len(history.validation)
    assert history.validation[history.best_epoch] == min(history.validation)
    assert history.stopped_early is (len(history.train) < 4)
    assert fitted.coefficient.dtype == jnp.float64
    assert all(math.isfinite(value) for value in history.train)
    assert all(math.isfinite(value) for value in history.validation)
