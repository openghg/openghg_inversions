"""Configurable exploratory objectives for corrected marginal score flows.

This module is intentionally separate from the frozen v1 optimizer.  It uses
explicit training and model-selection domains, static active loss terms, and
decomposed histories.  NLL-only fits do not compile score derivatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, NamedTuple

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import paramax  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    conditional_log_prob,
    conditional_log_prob_and_observation_score,
    raw_log_mass_condition_log_prob_and_score,
)

__all__ = [
    "ExplorationEpochMetrics",
    "ExplorationFitHistory",
    "ExplorationLossConfig",
    "ExplorationLossTerms",
    "fit_exploratory_score_flow",
    "initialization_loss_diagnostics",
    "loss_scale_diagnostics",
]


@dataclass(frozen=True, slots=True)
class ExplorationLossConfig:
    """Static active terms and training-only Fisher scales."""

    include_partial_score: bool
    include_observation_score: bool
    partial_score_weight: float
    observation_score_weight: float
    partial_score_scale: float
    observation_score_scales: tuple[float, ...]

    def payload(self) -> dict[str, object]:
        return asdict(self)


class ExplorationLossTerms(NamedTuple):
    """JAX scalar components returned with one exploratory objective."""

    negative_log_likelihood: jax.Array
    raw_partial_score_mse: jax.Array
    scaled_partial_score_mse: jax.Array
    raw_observation_score_mse: jax.Array
    scaled_observation_score_mse: jax.Array


@dataclass(frozen=True, slots=True)
class ExplorationEpochMetrics:
    """One finite decomposed epoch record."""

    objective: float
    negative_log_likelihood: float
    raw_partial_score_mse: float | None
    scaled_partial_score_mse: float | None
    raw_observation_score_mse: float | None
    scaled_observation_score_mse: float | None


@dataclass(frozen=True, slots=True)
class ExplorationFitHistory:
    """Complete training and external-validation histories."""

    training: tuple[ExplorationEpochMetrics, ...]
    validation: tuple[ExplorationEpochMetrics, ...]
    best_epoch: int
    stopped_early: bool
    stop_reason: str
    optimizer_state_reset_at_start: bool


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _float64_array(values: Any, *, name: str) -> jax.Array:
    result = jnp.asarray(values)
    if result.dtype != jnp.dtype(jnp.float64):
        raise ValueError(f"{name} must have dtype float64.")
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _batch_vector(values: Any, *, name: str, size: int) -> jax.Array:
    result = _float64_array(values, name=name)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape (n,) or (n, 1).")
    return result


def _validated_key(key: jax.Array) -> jax.Array:
    try:
        data = jr.key_data(key)
    except (TypeError, ValueError) as error:
        raise TypeError("key must be a scalar JAX PRNG key.") from error
    if data.shape != (2,) or data.dtype != jnp.uint32:
        raise ValueError("key must be a scalar JAX PRNG key.")
    return key


def _validate_config(
    config: ExplorationLossConfig,
    *,
    dimension: int,
) -> ExplorationLossConfig:
    if not isinstance(config, ExplorationLossConfig):
        raise TypeError("loss_config must be ExplorationLossConfig.")
    partial_weight = _finite_float(
        config.partial_score_weight,
        name="partial_score_weight",
    )
    observation_weight = _finite_float(
        config.observation_score_weight,
        name="observation_score_weight",
    )
    partial_scale = _finite_float(
        config.partial_score_scale,
        name="partial_score_scale",
    )
    observation_scales = tuple(
        _finite_float(value, name="observation_score_scale") for value in config.observation_score_scales
    )
    if partial_weight < 0.0 or observation_weight < 0.0:
        raise ValueError("loss weights must be non-negative.")
    if partial_scale <= 0.0:
        raise ValueError("partial_score_scale must be positive.")
    if len(observation_scales) != dimension or any(value <= 0.0 for value in observation_scales):
        raise ValueError("observation_score_scales must have one positive value per coordinate.")
    if not config.include_partial_score and partial_weight != 0.0:
        raise ValueError("inactive partial score must have zero weight.")
    if not config.include_observation_score and observation_weight != 0.0:
        raise ValueError("inactive observation score must have zero weight.")
    return ExplorationLossConfig(
        include_partial_score=config.include_partial_score,
        include_observation_score=config.include_observation_score,
        partial_score_weight=partial_weight,
        observation_score_weight=observation_weight,
        partial_score_scale=partial_scale,
        observation_score_scales=observation_scales,
    )


class _ExplorationLoss(eqx.Module):
    condition_center: float = eqx.field(static=True)
    condition_scale: float = eqx.field(static=True)
    include_partial_score: bool = eqx.field(static=True)
    include_observation_score: bool = eqx.field(static=True)
    partial_score_weight: float = eqx.field(static=True)
    observation_score_weight: float = eqx.field(static=True)
    partial_score_scale: float = eqx.field(static=True)
    observation_score_scales: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        condition_center: float,
        condition_scale: float,
        config: ExplorationLossConfig,
        dimension: int,
    ) -> None:
        center = _finite_float(condition_center, name="condition_center")
        scale = _finite_float(condition_scale, name="condition_scale")
        if scale <= 0.0:
            raise ValueError("condition_scale must be positive.")
        normalized = _validate_config(config, dimension=dimension)
        self.condition_center = center
        self.condition_scale = scale
        self.include_partial_score = normalized.include_partial_score
        self.include_observation_score = normalized.include_observation_score
        self.partial_score_weight = normalized.partial_score_weight
        self.observation_score_weight = normalized.observation_score_weight
        self.partial_score_scale = normalized.partial_score_scale
        self.observation_score_scales = normalized.observation_score_scales

    def __call__(
        self,
        params: Any,
        static: Any,
        projected: jax.Array,
        raw_log_mass: jax.Array,
        target_partial_score: jax.Array,
        target_observation_score: jax.Array,
    ) -> tuple[jax.Array, ExplorationLossTerms]:
        distribution = paramax.unwrap(eqx.combine(params, static))
        targets = jnp.asarray(projected)
        dimension = jnp.asarray(targets.shape[1], dtype=jnp.float64)
        predicted_observation = jnp.zeros_like(targets)
        if self.include_partial_score:
            log_probabilities, predicted_partial = raw_log_mass_condition_log_prob_and_score(
                distribution,
                targets,
                raw_log_mass,
                condition_center=self.condition_center,
                condition_scale=self.condition_scale,
            )
            partial_difference = predicted_partial - jnp.asarray(target_partial_score)
            raw_partial = jnp.mean(jnp.square(partial_difference))
            scaled_partial = jnp.mean(jnp.square(partial_difference / self.partial_score_scale))
        elif self.include_observation_score:
            log_probabilities, predicted_observation = conditional_log_prob_and_observation_score(
                distribution,
                targets,
                raw_log_mass,
                condition_center=self.condition_center,
                condition_scale=self.condition_scale,
            )
            raw_partial = jnp.zeros((), dtype=jnp.float64)
            scaled_partial = jnp.zeros((), dtype=jnp.float64)
        else:
            log_probabilities = conditional_log_prob(
                distribution,
                targets,
                raw_log_mass,
                condition_center=self.condition_center,
                condition_scale=self.condition_scale,
            )
            raw_partial = jnp.zeros((), dtype=jnp.float64)
            scaled_partial = jnp.zeros((), dtype=jnp.float64)

        if self.include_observation_score:
            if self.include_partial_score:
                _, predicted_observation = conditional_log_prob_and_observation_score(
                    distribution,
                    targets,
                    raw_log_mass,
                    condition_center=self.condition_center,
                    condition_scale=self.condition_scale,
                )
            observation_difference = predicted_observation - jnp.asarray(target_observation_score)
            raw_observation = jnp.mean(jnp.square(observation_difference))
            observation_scale = jnp.asarray(
                self.observation_score_scales,
                dtype=jnp.float64,
            )
            scaled_observation = jnp.mean(jnp.square(observation_difference / observation_scale))
        else:
            raw_observation = jnp.zeros((), dtype=jnp.float64)
            scaled_observation = jnp.zeros((), dtype=jnp.float64)

        nll = -jnp.mean(log_probabilities) / dimension
        objective = (
            nll
            + self.partial_score_weight * scaled_partial
            + self.observation_score_weight * scaled_observation
        )
        return objective, ExplorationLossTerms(
            negative_log_likelihood=nll,
            raw_partial_score_mse=raw_partial,
            scaled_partial_score_mse=scaled_partial,
            raw_observation_score_mse=raw_observation,
            scaled_observation_score_mse=scaled_observation,
        )


def loss_scale_diagnostics(
    target_partial_score: Any,
    target_observation_score: Any,
    *,
    floor: float = 1.0e-8,
) -> dict[str, object]:
    """Return training-only target variances and Fisher/RMS scales."""
    partial = _float64_array(
        target_partial_score,
        name="target_partial_score",
    )
    observation = _float64_array(
        target_observation_score,
        name="target_observation_score",
    )
    if partial.ndim != 1:
        raise ValueError("target_partial_score must be one-dimensional.")
    if observation.ndim != 2 or observation.shape[0] != partial.shape[0]:
        raise ValueError("target_observation_score must have shape (n, q).")
    minimum = _finite_float(floor, name="floor")
    if minimum <= 0.0:
        raise ValueError("floor must be positive.")
    partial_numpy = np.asarray(partial)
    observation_numpy = np.asarray(observation)
    partial_rms = max(
        math.sqrt(float(np.mean(np.square(partial_numpy)))),
        minimum,
    )
    observation_rms = np.maximum(
        np.sqrt(np.mean(np.square(observation_numpy), axis=0)),
        minimum,
    )
    return {
        "sample_count": int(partial.shape[0]),
        "partial_score_mean": float(np.mean(partial_numpy)),
        "partial_score_variance": float(np.var(partial_numpy)),
        "partial_score_rms": partial_rms,
        "observation_score_mean_by_coordinate": (np.mean(observation_numpy, axis=0).tolist()),
        "observation_score_variance_by_coordinate": (np.var(observation_numpy, axis=0).tolist()),
        "observation_score_rms_by_coordinate": observation_rms.tolist(),
        "scale_floor": minimum,
    }


def _validate_data(
    data: tuple[Any, Any, Any, Any],
    *,
    name: str,
    dimension: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    projected = _float64_array(data[0], name=f"{name}.projected")
    if projected.ndim != 2 or projected.shape[0] < 1 or projected.shape[1] < 1:
        raise ValueError(f"{name}.projected must have shape (n, q).")
    if dimension is not None and projected.shape[1] != dimension:
        raise ValueError("training and validation projected dimensions differ.")
    count = projected.shape[0]
    mass = _batch_vector(data[1], name=f"{name}.raw_log_mass", size=count)
    partial = _batch_vector(
        data[2],
        name=f"{name}.target_partial_score",
        size=count,
    )
    observation = _float64_array(
        data[3],
        name=f"{name}.target_observation_score",
    )
    if observation.shape != projected.shape:
        raise ValueError(f"{name}.target_observation_score must match projected shape.")
    return projected, mass, partial, observation


def _validate_model(model: Any, *, dimension: int) -> tuple[Any, Any]:
    if getattr(model, "shape", None) not in (None, (dimension,)):
        raise ValueError("model shape must match projected dimension.")
    params, static = eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(params) if eqx.is_inexact_array(leaf)]
    if not leaves:
        raise ValueError("model must have trainable floating-point parameters.")
    for leaf in leaves:
        if leaf.dtype != jnp.float64 or not bool(jnp.all(jnp.isfinite(leaf))):
            raise ValueError("model parameters must be finite float64 arrays.")
    return params, static


def initialization_loss_diagnostics(
    model: Any,
    data: tuple[Any, Any, Any, Any],
    *,
    condition_center: float,
    condition_scale: float,
    partial_score_scale: float,
    observation_score_scales: tuple[float, ...],
    sample_limit: int = 256,
) -> dict[str, object]:
    """Measure component risks and parameter-gradient norms before weighting."""
    projected, raw_mass, target_partial, target_observation = _validate_data(
        data,
        name="initialization",
    )
    count = min(
        projected.shape[0],
        _positive_integer(sample_limit, name="sample_limit"),
    )
    projected = projected[:count]
    raw_mass = raw_mass[:count]
    target_partial = target_partial[:count]
    target_observation = target_observation[:count]
    dimension = projected.shape[1]
    config = _validate_config(
        ExplorationLossConfig(
            include_partial_score=True,
            include_observation_score=True,
            partial_score_weight=1.0,
            observation_score_weight=1.0,
            partial_score_scale=partial_score_scale,
            observation_score_scales=observation_score_scales,
        ),
        dimension=dimension,
    )
    params, static = _validate_model(model, dimension=dimension)

    def component_values(
        local_params: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        distribution = paramax.unwrap(eqx.combine(local_params, static))
        log_probabilities, predicted_partial = raw_log_mass_condition_log_prob_and_score(
            distribution,
            projected,
            raw_mass,
            condition_center=condition_center,
            condition_scale=condition_scale,
        )
        _, predicted_observation = conditional_log_prob_and_observation_score(
            distribution,
            projected,
            raw_mass,
            condition_center=condition_center,
            condition_scale=condition_scale,
        )
        nll_rows = -log_probabilities / dimension
        partial_rows = jnp.square((predicted_partial - target_partial) / config.partial_score_scale)
        observation_rows = jnp.mean(
            jnp.square(
                (predicted_observation - target_observation)
                / jnp.asarray(
                    config.observation_score_scales,
                    dtype=jnp.float64,
                )
            ),
            axis=1,
        )
        return nll_rows, partial_rows, observation_rows

    def component_mean(local_params: Any, index: int) -> jax.Array:
        return jnp.mean(component_values(local_params)[index])

    rows = component_values(params)
    gradients = [
        eqx.filter_grad(lambda local_params, index=index: component_mean(local_params, index))(params)
        for index in range(3)
    ]

    def gradient_norm(gradient: Any) -> float:
        squared = sum(
            float(jnp.vdot(leaf, leaf))
            for leaf in jax.tree_util.tree_leaves(gradient)
            if eqx.is_inexact_array(leaf)
        )
        return math.sqrt(squared)

    names = (
        "nll_per_dimension",
        "fisher_scaled_partial_score",
        "fisher_scaled_observation_score",
    )
    return {
        "sample_count": count,
        "measured_before_loss_weights_applied": True,
        "components": {
            name: {
                "row_mean": float(jnp.mean(values)),
                "row_variance": float(jnp.var(values)),
                "row_minimum": float(jnp.min(values)),
                "row_maximum": float(jnp.max(values)),
                "parameter_gradient_l2_norm": gradient_norm(gradient),
            }
            for name, values, gradient in zip(
                names,
                rows,
                gradients,
                strict=True,
            )
        },
    }


def _sequential_add(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _scale_tree(tree: Any, denominator: int) -> Any:
    scale = jnp.asarray(denominator, dtype=jnp.float64)
    return jax.tree_util.tree_map(lambda value: value / scale, tree)


def _finite_scalar_array(value: jax.Array, *, name: str) -> float:
    if value.shape != ():
        raise ValueError(f"{name} must be scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not finite.")
    return result


def _empty_totals() -> list[float]:
    return [0.0] * 6


def _add_weighted_terms(
    totals: list[float],
    objective: jax.Array,
    terms: ExplorationLossTerms,
    weight: int,
) -> None:
    values = (objective, *terms)
    for index, value in enumerate(values):
        totals[index] += (
            _finite_scalar_array(
                value,
                name="loss component",
            )
            * weight
        )


def _epoch_metrics(
    totals: list[float],
    weight: int,
    *,
    config: ExplorationLossConfig,
) -> ExplorationEpochMetrics:
    if weight < 1:
        raise RuntimeError("epoch contains no evaluated rows.")
    values = [value / weight for value in totals]
    return ExplorationEpochMetrics(
        objective=values[0],
        negative_log_likelihood=values[1],
        raw_partial_score_mse=values[2] if config.include_partial_score else None,
        scaled_partial_score_mse=values[3] if config.include_partial_score else None,
        raw_observation_score_mse=(values[4] if config.include_observation_score else None),
        scaled_observation_score_mse=(values[5] if config.include_observation_score else None),
    )


def fit_exploratory_score_flow(
    key: jax.Array,
    model: Any,
    training_data: tuple[Any, Any, Any, Any],
    validation_data: tuple[Any, Any, Any, Any],
    *,
    condition_center: float,
    condition_scale: float,
    loss_config: ExplorationLossConfig,
    learning_rate: float = 5.0e-4,
    batch_size: int = 1024,
    microbatch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[Any, ExplorationFitHistory]:
    """Fit one stage against explicit independent simulator domains."""
    normalized_key = _validated_key(key)
    training = _validate_data(training_data, name="training")
    validation = _validate_data(
        validation_data,
        name="validation",
        dimension=training[0].shape[1],
    )
    dimension = training[0].shape[1]
    config = _validate_config(loss_config, dimension=dimension)
    learning_rate_value = _finite_float(
        learning_rate,
        name="learning_rate",
    )
    if learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be positive.")
    outer_size = min(
        training[0].shape[0],
        _positive_integer(batch_size, name="batch_size"),
    )
    micro_size = _positive_integer(microbatch_size, name="microbatch_size")
    if outer_size % micro_size != 0:
        raise ValueError("effective batch size must be divisible by microbatch_size.")
    epoch_limit = _positive_integer(max_epochs, name="max_epochs")
    patience_value = _nonnegative_integer(patience, name="patience")
    params, static = _validate_model(model, dimension=dimension)
    loss = _ExplorationLoss(
        condition_center=condition_center,
        condition_scale=condition_scale,
        config=config,
        dimension=dimension,
    )
    loss_and_gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
    validation_loss = eqx.filter_jit(loss)
    optimizer = optax.adam(learning_rate_value)
    optimizer_state = optimizer.init(params)

    @eqx.filter_jit
    def apply_optimizer(
        current_params: Any,
        current_state: Any,
        gradients: Any,
    ) -> tuple[Any, Any]:
        updates, next_state = optimizer.update(
            gradients,
            current_state,
            params=current_params,
        )
        return eqx.apply_updates(current_params, updates), next_state

    train_shuffle_root, validation_shuffle_root = jr.split(normalized_key)
    training_history: list[ExplorationEpochMetrics] = []
    validation_history: list[ExplorationEpochMetrics] = []
    best_params = params
    best_epoch = 0
    best_validation = math.inf
    fruitless = 0
    stopped_early = False
    for epoch in range(epoch_limit):
        train_order = jr.permutation(
            jr.fold_in(train_shuffle_root, epoch),
            training[0].shape[0],
        )
        validation_order = jr.permutation(
            jr.fold_in(validation_shuffle_root, epoch),
            validation[0].shape[0],
        )
        shuffled_training = tuple(values[train_order] for values in training)
        shuffled_validation = tuple(values[validation_order] for values in validation)
        train_totals = _empty_totals()
        train_weight = 0
        outer_batches = training[0].shape[0] // outer_size
        for outer_index in range(outer_batches):
            start = outer_index * outer_size
            stop = start + outer_size
            outer = tuple(values[start:stop] for values in shuffled_training)
            accumulated_gradients = None
            microbatch_count = outer_size // micro_size
            for micro_index in range(microbatch_count):
                micro_start = micro_index * micro_size
                micro_stop = micro_start + micro_size
                microbatch = tuple(values[micro_start:micro_stop] for values in outer)
                (objective, terms), gradients = loss_and_gradient(
                    params,
                    static,
                    *microbatch,
                )
                _add_weighted_terms(
                    train_totals,
                    objective,
                    terms,
                    micro_size,
                )
                train_weight += micro_size
                accumulated_gradients = (
                    gradients
                    if accumulated_gradients is None
                    else _sequential_add(accumulated_gradients, gradients)
                )
            if accumulated_gradients is None:  # pragma: no cover
                raise RuntimeError("outer batch contains no microbatches.")
            params, optimizer_state = apply_optimizer(
                params,
                optimizer_state,
                _scale_tree(accumulated_gradients, microbatch_count),
            )
        train_metrics = _epoch_metrics(
            train_totals,
            train_weight,
            config=config,
        )
        training_history.append(train_metrics)

        validation_totals = _empty_totals()
        validation_weight = 0
        validation_batches = math.ceil(validation[0].shape[0] / micro_size)
        for micro_index in range(validation_batches):
            start = micro_index * micro_size
            stop = min(start + micro_size, validation[0].shape[0])
            microbatch = tuple(values[start:stop] for values in shuffled_validation)
            objective, terms = validation_loss(params, static, *microbatch)
            weight = stop - start
            _add_weighted_terms(
                validation_totals,
                objective,
                terms,
                weight,
            )
            validation_weight += weight
        validation_metrics = _epoch_metrics(
            validation_totals,
            validation_weight,
            config=config,
        )
        validation_history.append(validation_metrics)
        if validation_metrics.objective <= best_validation:
            best_validation = validation_metrics.objective
            best_params = params
            best_epoch = epoch
            fruitless = 0
        else:
            fruitless += 1
            if fruitless > patience_value:
                stopped_early = epoch + 1 < epoch_limit
                break

    fitted = eqx.combine(best_params, static)
    return fitted, ExplorationFitHistory(
        training=tuple(training_history),
        validation=tuple(validation_history),
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        stop_reason=("early_stopping_patience_exhausted" if stopped_early else "maximum_epochs_reached"),
        optimizer_state_reset_at_start=True,
    )
