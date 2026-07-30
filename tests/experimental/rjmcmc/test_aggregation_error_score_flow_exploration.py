"""Focused tests for configurable corrected score-flow exploration."""

from __future__ import annotations

import math
from typing import Any, cast

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_exploration import (  # noqa: E402
    ExplorationLossConfig,
    fit_exploratory_score_flow,
    initialization_loss_diagnostics,
    loss_scale_diagnostics,
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


def _data(
    offset: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    projected = jnp.asarray(
        [[-1.4], [-0.8], [-0.1], [0.3], [0.7], [1.2], [1.8], [2.4]],
        dtype=jnp.float64,
    )
    raw_log_mass = jnp.asarray(
        [-1.3, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.4],
        dtype=jnp.float64,
    )
    coefficient = 0.7
    scale = 1.1
    center = 0.15
    condition_scale = 1.25
    condition = (raw_log_mass - center) / condition_scale
    residual = projected[:, 0] - coefficient * condition
    partial = coefficient * residual / (scale**2 * condition_scale) + offset
    observation = (-residual / scale**2 + offset)[:, None]
    return projected, raw_log_mass, partial, observation


def test_loss_scale_diagnostics_report_training_target_rms_and_variance() -> None:
    data = _data()
    diagnostics = loss_scale_diagnostics(data[2], data[3])
    assert diagnostics["sample_count"] == 8
    assert diagnostics["partial_score_rms"] == pytest.approx(math.sqrt(float(np.mean(np.square(data[2])))))
    assert diagnostics["observation_score_rms_by_coordinate"] == pytest.approx(
        [math.sqrt(float(np.mean(np.square(data[3][:, 0]))))]
    )
    assert diagnostics["partial_score_variance"] == pytest.approx(float(np.var(data[2])))


def test_initialization_diagnostics_measure_unweighted_component_gradients() -> None:
    data = _data(offset=0.03)
    scales = cast(
        dict[str, Any],
        loss_scale_diagnostics(data[2], data[3]),
    )
    diagnostics = initialization_loss_diagnostics(
        _KnownConditionalNormal(0.2, math.log(1.3)),
        data,
        condition_center=0.15,
        condition_scale=1.25,
        partial_score_scale=float(scales["partial_score_rms"]),
        observation_score_scales=tuple(
            float(value) for value in scales["observation_score_rms_by_coordinate"]
        ),
        sample_limit=4,
    )
    assert diagnostics["sample_count"] == 4
    assert diagnostics["measured_before_loss_weights_applied"] is True
    components = cast(dict[str, dict[str, Any]], diagnostics["components"])
    assert set(components) == {
        "nll_per_dimension",
        "fisher_scaled_partial_score",
        "fisher_scaled_observation_score",
    }
    for component in components.values():
        assert math.isfinite(component["row_mean"])
        assert component["row_variance"] >= 0.0
        assert component["parameter_gradient_l2_norm"] > 0.0


@pytest.mark.parametrize(
    "config",
    [
        ExplorationLossConfig(
            include_partial_score=False,
            include_observation_score=False,
            partial_score_weight=0.0,
            observation_score_weight=0.0,
            partial_score_scale=1.0,
            observation_score_scales=(1.0,),
        ),
        ExplorationLossConfig(
            include_partial_score=True,
            include_observation_score=False,
            partial_score_weight=1.0,
            observation_score_weight=0.0,
            partial_score_scale=0.8,
            observation_score_scales=(1.0,),
        ),
        ExplorationLossConfig(
            include_partial_score=False,
            include_observation_score=True,
            partial_score_weight=0.0,
            observation_score_weight=1.0,
            partial_score_scale=1.0,
            observation_score_scales=(0.9,),
        ),
    ],
)
def test_each_static_ablation_fits_with_decomposed_external_histories(
    config: ExplorationLossConfig,
) -> None:
    fitted, history = fit_exploratory_score_flow(
        jr.key(501),
        _KnownConditionalNormal(0.2, math.log(1.3)),
        _data(),
        _data(offset=0.01),
        condition_center=0.15,
        condition_scale=1.25,
        loss_config=config,
        learning_rate=1.0e-3,
        batch_size=4,
        microbatch_size=2,
        max_epochs=2,
        patience=1,
    )
    assert len(history.training) == 2
    assert len(history.validation) == 2
    assert history.best_epoch in (0, 1)
    assert history.stop_reason == "maximum_epochs_reached"
    assert history.optimizer_state_reset_at_start
    assert math.isfinite(float(fitted.coefficient))
    for metrics in (*history.training, *history.validation):
        assert math.isfinite(metrics.objective)
        assert math.isfinite(metrics.negative_log_likelihood)
        assert (metrics.scaled_partial_score_mse is not None) == (config.include_partial_score)
        assert (metrics.scaled_observation_score_mse is not None) == (config.include_observation_score)


def test_exploratory_fit_replays_exactly_for_same_key_and_domains() -> None:
    diagnostics = cast(
        dict[str, Any],
        loss_scale_diagnostics(_data()[2], _data()[3]),
    )
    config = ExplorationLossConfig(
        include_partial_score=True,
        include_observation_score=False,
        partial_score_weight=1.0,
        observation_score_weight=0.0,
        partial_score_scale=float(diagnostics["partial_score_rms"]),
        observation_score_scales=(1.0,),
    )
    first_model, first_history = fit_exploratory_score_flow(
        jr.key(1701),
        _KnownConditionalNormal(0.2, math.log(1.3)),
        _data(),
        _data(offset=0.01),
        condition_center=0.15,
        condition_scale=1.25,
        loss_config=config,
        batch_size=4,
        microbatch_size=2,
        max_epochs=2,
        patience=1,
    )
    second_model, second_history = fit_exploratory_score_flow(
        jr.key(1701),
        _KnownConditionalNormal(0.2, math.log(1.3)),
        _data(),
        _data(offset=0.01),
        condition_center=0.15,
        condition_scale=1.25,
        loss_config=config,
        batch_size=4,
        microbatch_size=2,
        max_epochs=2,
        patience=1,
    )
    np.testing.assert_array_equal(
        np.asarray(first_model.coefficient),
        np.asarray(second_model.coefficient),
    )
    np.testing.assert_array_equal(
        np.asarray(first_model.log_scale),
        np.asarray(second_model.log_scale),
    )
    assert first_history == second_history
