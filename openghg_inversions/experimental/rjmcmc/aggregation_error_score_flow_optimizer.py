"""Deterministic microbatch optimization for score-regularized marginal flows.

The mass-score term in :class:`RawLogMassScoreLoss` creates mixed parameter
and condition derivatives.  Evaluating that term for a full optimizer batch
can require substantially more temporary memory than the model parameters.
This module freezes the memory control used by the score-regularized NLE:
each outer batch is traversed in fixed-size microbatches, their parameter
gradients are added in a fixed sequential order, and exactly one Adam update
is applied to the averaged gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import optax  # noqa: E402
import paramax  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    RawLogMassScoreLoss,
)

__all__ = [
    "ScoreRegularizedFitHistory",
    "fit_score_regularized_flow",
]


@dataclass(frozen=True)
class ScoreRegularizedFitHistory:
    """Immutable optimizer history.

    ``best_epoch`` is the zero-based index into ``train`` and ``validation``.
    Both histories contain the composite loss defined by
    :class:`RawLogMassScoreLoss`.
    """

    train: tuple[float, ...]
    validation: tuple[float, ...]
    best_epoch: int
    stopped_early: bool


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive, non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    """Return one non-negative, non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _finite_float(value: float, *, name: str) -> float:
    """Return one finite, non-Boolean real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validated_key(key: jax.Array) -> jax.Array:
    """Return a scalar JAX PRNG key."""
    try:
        key_data = jr.key_data(key)
    except (TypeError, ValueError) as error:
        raise TypeError("key must be a scalar JAX PRNG key.") from error
    if key_data.shape != (2,) or key_data.dtype != jnp.uint32:
        raise ValueError("key must be a scalar JAX PRNG key.")
    return key


def _float64_array(values: Any, *, name: str) -> jax.Array:
    """Return one concrete, finite float64 JAX array without dtype coercion."""
    result = jnp.asarray(values)
    if result.dtype != jnp.dtype(jnp.float64):
        raise ValueError(f"{name} must have dtype float64.")
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _batch_vector(values: Any, *, name: str, size: int) -> jax.Array:
    """Return a float64 vector from shape ``(n,)`` or ``(n, 1)``."""
    result = _float64_array(values, name=name)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape (n,) or (n, 1).")
    if result.shape[0] != size:
        raise ValueError(f"{name} must have the same batch size as projected.")
    return result


def _validate_model(model: Any, *, projected_dimension: int) -> tuple[Any, Any]:
    """Partition a finite float64 model and validate any declared event shape."""
    declared_shape = getattr(model, "shape", None)
    if declared_shape is not None and tuple(declared_shape) != (projected_dimension,):
        raise ValueError("model shape must match projected.shape[1].")

    params, static = eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    trainable_leaves = [leaf for leaf in jax.tree_util.tree_leaves(params) if eqx.is_inexact_array(leaf)]
    if not trainable_leaves:
        raise ValueError("model must contain trainable floating-point arrays.")
    for leaf in trainable_leaves:
        if leaf.dtype != jnp.dtype(jnp.float64):
            raise ValueError("model trainable parameters must have dtype float64.")
        if not bool(jnp.all(jnp.isfinite(leaf))):
            raise ValueError("model trainable parameters must be finite.")
    return params, static


def _effective_batch_size(
    partition_size: int,
    requested_batch_size: int,
    microbatch_size: int,
    *,
    name: str,
) -> int:
    """Return the FlowJAX-style full outer-batch size for one partition."""
    effective = min(partition_size, requested_batch_size)
    if effective % microbatch_size != 0:
        raise ValueError(
            f"{name} effective outer batch size ({effective}) must be divisible "
            f"by score_microbatch_size ({microbatch_size})."
        )
    return effective


def _sequential_add(left: Any, right: Any) -> Any:
    """Add matching gradient pytrees without changing leaf traversal order."""
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _scale_tree(tree: Any, denominator: int) -> Any:
    """Divide every differentiable gradient leaf by an integer."""
    scale = jnp.asarray(denominator, dtype=jnp.float64)
    return jax.tree_util.tree_map(lambda value: value / scale, tree)


def _finite_loss(value: jax.Array, *, context: str) -> float:
    """Convert a scalar finite loss to a Python float."""
    if value.shape != ():
        raise ValueError(f"{context} loss must be scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{context} loss is not finite.")
    return result


def fit_score_regularized_flow(
    key: jax.Array,
    model: Any,
    projected: Any,
    raw_log_mass: Any,
    target_score: Any,
    *,
    condition_center: float,
    condition_scale: float,
    learning_rate: float = 5e-4,
    batch_size: int = 1024,
    score_microbatch_size: int = 64,
    val_prop: float = 0.1,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[Any, ScoreRegularizedFitHistory]:
    """Fit a score-regularized flow with frozen sequential accumulation.

    The train/validation split is made exactly once.  Each epoch then uses
    independent, epoch-indexed permutations for the two fixed partitions.
    Truncated training outer batches are dropped, matching FlowJAX.  If the
    training partition is smaller than ``batch_size``, its complete size is
    used as one outer batch.  Every used training outer batch must be exactly
    divisible by ``score_microbatch_size``.

    Each training microbatch computes both the loss value and parameter
    gradient.  Gradients are accumulated in increasing microbatch order,
    averaged, and passed to Adam once per outer batch.  Validation performs
    the same fixed microbatch traversal without parameter gradients.  The
    returned model always contains the parameters from the best validation
    epoch; ties select the later epoch, matching FlowJAX's best-epoch rule.
    Validation uses the entire validation partition in ordered fixed-size
    microbatches plus, when needed, one final remainder microbatch.  Its epoch
    loss is weighted by the actual number of rows in each microbatch.
    """
    key = _validated_key(key)
    learning_rate_value = _finite_float(learning_rate, name="learning_rate")
    if learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be positive.")
    batch_size_value = _positive_integer(batch_size, name="batch_size")
    microbatch_size = _positive_integer(
        score_microbatch_size,
        name="score_microbatch_size",
    )
    epoch_limit = _positive_integer(max_epochs, name="max_epochs")
    patience_value = _nonnegative_integer(patience, name="patience")
    validation_fraction = _finite_float(val_prop, name="val_prop")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("val_prop must lie strictly between zero and one.")

    projected_array = _float64_array(projected, name="projected")
    if projected_array.ndim != 2 or projected_array.shape[1] < 1:
        raise ValueError("projected must have shape (n, q) with q positive.")
    sample_count = projected_array.shape[0]
    if sample_count < 2:
        raise ValueError("at least two samples are required.")
    mass_array = _batch_vector(
        raw_log_mass,
        name="raw_log_mass",
        size=sample_count,
    )
    score_array = _batch_vector(
        target_score,
        name="target_score",
        size=sample_count,
    )

    center = _finite_float(condition_center, name="condition_center")
    scale = _finite_float(condition_scale, name="condition_scale")
    if scale <= 0.0:
        raise ValueError("condition_scale must be positive.")

    validation_size = round(validation_fraction * sample_count)
    training_size = sample_count - validation_size
    if training_size < 1 or validation_size < 1:
        raise ValueError("val_prop must produce non-empty train and validation sets.")
    train_batch_size = _effective_batch_size(
        training_size,
        batch_size_value,
        microbatch_size,
        name="training",
    )
    params, static = _validate_model(
        model,
        projected_dimension=projected_array.shape[1],
    )
    loss_function = RawLogMassScoreLoss(
        condition_center=center,
        condition_scale=scale,
    )
    loss_and_gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss_function))
    validation_loss = eqx.filter_jit(loss_function)
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

    split_key, train_shuffle_root, validation_shuffle_root, loss_key_root = jr.split(key, 4)
    split_indices = jr.permutation(split_key, sample_count)
    training_indices = split_indices[:training_size]
    validation_indices = split_indices[training_size:]
    training_data = (
        projected_array[training_indices],
        mass_array[training_indices],
        score_array[training_indices],
    )
    validation_data = (
        projected_array[validation_indices],
        mass_array[validation_indices],
        score_array[validation_indices],
    )

    train_history: list[float] = []
    validation_history: list[float] = []
    best_params = params
    best_epoch = 0
    best_validation_loss = math.inf
    fruitless_epochs = 0
    stopped_early = False

    for epoch in range(epoch_limit):
        train_permutation = jr.permutation(
            jr.fold_in(train_shuffle_root, epoch),
            training_size,
        )
        validation_permutation = jr.permutation(
            jr.fold_in(validation_shuffle_root, epoch),
            validation_size,
        )
        shuffled_training = tuple(values[train_permutation] for values in training_data)
        shuffled_validation = tuple(values[validation_permutation] for values in validation_data)

        train_weighted_loss = 0.0
        train_weight = 0
        train_outer_batches = training_size // train_batch_size
        for outer_index in range(train_outer_batches):
            outer_start = outer_index * train_batch_size
            outer_stop = outer_start + train_batch_size
            outer_batch = tuple(values[outer_start:outer_stop] for values in shuffled_training)
            microbatch_count = train_batch_size // microbatch_size
            accumulated_gradients = None
            outer_weighted_loss = 0.0
            for micro_index in range(microbatch_count):
                micro_start = micro_index * microbatch_size
                micro_stop = micro_start + microbatch_size
                microbatch = tuple(values[micro_start:micro_stop] for values in outer_batch)
                loss_key = jr.fold_in(
                    jr.fold_in(
                        jr.fold_in(loss_key_root, epoch),
                        outer_index,
                    ),
                    micro_index,
                )
                loss_value, gradients = loss_and_gradient(
                    params,
                    static,
                    *microbatch,
                    key=loss_key,
                )
                loss_float = _finite_loss(
                    loss_value,
                    context="training microbatch",
                )
                outer_weighted_loss += loss_float * microbatch_size
                if accumulated_gradients is None:
                    accumulated_gradients = gradients
                else:
                    accumulated_gradients = _sequential_add(
                        accumulated_gradients,
                        gradients,
                    )
            if accumulated_gradients is None:  # pragma: no cover - validated above
                raise RuntimeError("training outer batch had no microbatches.")
            averaged_gradients = _scale_tree(
                accumulated_gradients,
                microbatch_count,
            )
            params, optimizer_state = apply_optimizer(
                params,
                optimizer_state,
                averaged_gradients,
            )
            train_weighted_loss += outer_weighted_loss
            train_weight += train_batch_size
        train_epoch_loss = train_weighted_loss / train_weight
        if not math.isfinite(train_epoch_loss):
            raise FloatingPointError("training epoch loss is not finite.")
        train_history.append(train_epoch_loss)

        validation_weighted_loss = 0.0
        validation_weight = 0
        validation_microbatches = math.ceil(validation_size / microbatch_size)
        for micro_index in range(validation_microbatches):
            micro_start = micro_index * microbatch_size
            micro_stop = min(micro_start + microbatch_size, validation_size)
            microbatch = tuple(values[micro_start:micro_stop] for values in shuffled_validation)
            microbatch_count = micro_stop - micro_start
            loss_key = jr.fold_in(
                jr.fold_in(
                    jr.fold_in(loss_key_root, epoch),
                    train_outer_batches,
                ),
                micro_index,
            )
            loss_value = validation_loss(
                params,
                static,
                *microbatch,
                key=loss_key,
            )
            loss_float = _finite_loss(
                loss_value,
                context="validation microbatch",
            )
            validation_weighted_loss += loss_float * microbatch_count
            validation_weight += microbatch_count
        validation_epoch_loss = validation_weighted_loss / validation_weight
        if not math.isfinite(validation_epoch_loss):
            raise FloatingPointError("validation epoch loss is not finite.")
        validation_history.append(validation_epoch_loss)

        if validation_epoch_loss <= best_validation_loss:
            best_validation_loss = validation_epoch_loss
            best_params = params
            best_epoch = epoch
            fruitless_epochs = 0
        else:
            fruitless_epochs += 1
            if fruitless_epochs > patience_value:
                stopped_early = epoch + 1 < epoch_limit
                break

    fitted_model = eqx.combine(best_params, static)
    history = ScoreRegularizedFitHistory(
        train=tuple(train_history),
        validation=tuple(validation_history),
        best_epoch=best_epoch,
        stopped_early=stopped_early,
    )
    return fitted_model, history
