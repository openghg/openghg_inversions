"""Flow construction and score-regularized training loss for projected marginals.

This optional experimental module requires the ``nle`` dependency group.  The
learner is a normalized FlowJAX conditional density for projected coordinates
``x`` given raw log total mass ``tau``.  Training combines likelihood with the
simulator-derived partial score with respect to raw ``tau``.

The score loss deliberately standardizes ``tau`` internally.  Its target is
therefore always the derivative in the scientifically meaningful raw-log-mass
coordinate, irrespective of the frozen condition standardization used by the
flow.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import equinox as eqx
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import coupling_flow, masked_autoregressive_flow
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import paramax  # noqa: E402
from scipy import special  # noqa: E402

__all__ = [
    "FLOW_INVERT",
    "FLOW_LAYERS",
    "FLOW_NN_DEPTH",
    "FLOW_NN_WIDTH",
    "FLOW_SPLINE_INTERVAL",
    "FLOW_SPLINE_KNOTS",
    "RawLogMassScoreLoss",
    "conditional_log_prob",
    "conditional_log_prob_and_observation_score",
    "gamma_log_mass_conditioning",
    "make_score_regularized_conditional_flow",
    "raw_log_mass_condition_log_prob_and_score",
    "raw_log_mass_condition_score",
]

FLOW_LAYERS = 8
FLOW_SPLINE_KNOTS = 8
FLOW_SPLINE_INTERVAL = (-5.0, 5.0)
FLOW_NN_WIDTH = 128
FLOW_NN_DEPTH = 2
FLOW_INVERT = True


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive, non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _source_seed(value: int) -> int:
    """Return an unsigned 32-bit seed accepted by JAX."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("source_seed must be an integer.")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("source_seed must lie in [0, 2**32).")
    return result


def _finite_scalar(value: float, *, name: str) -> float:
    """Return one finite non-Boolean scalar."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def gamma_log_mass_conditioning(
    shape: float,
    rate: float,
) -> tuple[float, float]:
    """Return the analytic center and scale of ``log(T)`` for Gamma ``T``.

    The second Gamma parameter is a rate.  This observation-blind rule keeps
    condition standardization identical across nested simulation sizes and
    independent source catalogues.
    """
    gamma_shape = _finite_scalar(shape, name="shape")
    gamma_rate = _finite_scalar(rate, name="rate")
    if gamma_shape <= 0.0:
        raise ValueError("shape must be positive.")
    if gamma_rate <= 0.0:
        raise ValueError("rate must be positive.")
    center = float(special.digamma(gamma_shape) - math.log(gamma_rate))
    scale = math.sqrt(float(special.polygamma(1, gamma_shape)))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("analytic Gamma log-mass conditioning is invalid.")
    return center, scale


def _validate_float64_flow(flow: Any) -> None:
    """Require finite float64 trainable leaves in a newly constructed flow."""
    found = False
    for leaf in jax.tree_util.tree_leaves(flow):
        if not eqx.is_inexact_array(leaf):
            continue
        found = True
        if leaf.dtype != jnp.dtype(jnp.float64):
            raise ValueError("flow parameters must use float64.")
        if not bool(jnp.all(jnp.isfinite(leaf))):
            raise ValueError("flow parameters must be finite.")
    if not found:
        raise ValueError("flow must contain trainable floating-point arrays.")


def make_score_regularized_conditional_flow(
    projected_dimension: int,
    *,
    source_seed: int,
) -> Any:
    """Construct the frozen conditional flow for projected dimension ``q``.

    Dimensions of two or more use the predeclared coupling architecture.  A
    coupling split is undefined for ``q=1``, so the exact-oracle specialization
    uses the corresponding one-dimensional conditional masked-autoregressive
    flow.  Both branches use the same spline, conditioner-network, layer, and
    inversion controls.
    """
    dimension = _positive_integer(
        projected_dimension,
        name="projected_dimension",
    )
    seed = _source_seed(source_seed)
    base = Normal(jnp.zeros(dimension, dtype=jnp.float64))
    if dimension == 1:
        result = masked_autoregressive_flow(
            key=jr.key(seed),
            base_dist=base,
            transformer=RationalQuadraticSpline(
                knots=FLOW_SPLINE_KNOTS,
                interval=FLOW_SPLINE_INTERVAL,
            ),
            cond_dim=1,
            flow_layers=FLOW_LAYERS,
            nn_width=FLOW_NN_WIDTH,
            nn_depth=FLOW_NN_DEPTH,
            nn_activation=jnp.tanh,
            invert=FLOW_INVERT,
        )
    else:
        result = coupling_flow(
            key=jr.key(seed),
            base_dist=base,
            transformer=RationalQuadraticSpline(
                knots=FLOW_SPLINE_KNOTS,
                interval=FLOW_SPLINE_INTERVAL,
            ),
            cond_dim=1,
            flow_layers=FLOW_LAYERS,
            nn_width=FLOW_NN_WIDTH,
            nn_depth=FLOW_NN_DEPTH,
            nn_activation=jnp.tanh,
            invert=FLOW_INVERT,
        )
    _validate_float64_flow(result)
    return result


def _batch_scalar(values: jax.Array, *, name: str) -> jax.Array:
    """Return a batch vector from shape ``(n,)`` or ``(n, 1)``."""
    result = jnp.asarray(values)
    if result.ndim == 1:
        return result
    if result.ndim == 2 and result.shape[1] == 1:
        return result[:, 0]
    raise ValueError(f"{name} must have shape (n,) or (n, 1).")


def _raw_log_mass_condition_log_prob_and_score(
    distribution: Any,
    projected: jax.Array,
    raw_log_mass: jax.Array,
    *,
    condition_center: float,
    condition_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate conditional log density and its raw-mass score per batch row.

    The raw-mass derivative is a scalar-input Jacobian-vector product.  A loss
    built from this function may then take the outer parameter gradient,
    retaining the exact mixed derivative while avoiding a reverse-over-reverse
    compilation graph.  Returning the primal value from the same JVP also
    avoids evaluating the conditional flow a second time for the likelihood
    term.
    """
    center = _finite_scalar(condition_center, name="condition_center")
    scale = _finite_scalar(condition_scale, name="condition_scale")
    if scale <= 0.0:
        raise ValueError("condition_scale must be positive.")

    targets = jnp.asarray(projected)
    if targets.ndim != 2 or targets.shape[1] < 1:
        raise ValueError("projected must have shape (n, q) with q positive.")
    tau = _batch_scalar(raw_log_mass, name="raw_log_mass")
    if tau.shape[0] != targets.shape[0]:
        raise ValueError("projected and raw_log_mass batch sizes must match.")

    def value_and_score_one(
        target: jax.Array,
        raw_tau: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def log_prob_one(local_tau: jax.Array) -> jax.Array:
            condition = jnp.expand_dims((local_tau - center) / scale, axis=0)
            return distribution.log_prob(target, condition)

        return jax.jvp(
            log_prob_one,
            (raw_tau,),
            (jnp.ones_like(raw_tau),),
        )

    return jax.vmap(value_and_score_one)(targets, tau)


def conditional_log_prob(
    distribution: Any,
    projected: jax.Array,
    raw_log_mass: jax.Array,
    *,
    condition_center: float,
    condition_scale: float,
) -> jax.Array:
    """Evaluate conditional log density without compiling condition scores."""
    center = _finite_scalar(condition_center, name="condition_center")
    scale = _finite_scalar(condition_scale, name="condition_scale")
    if scale <= 0.0:
        raise ValueError("condition_scale must be positive.")
    targets = jnp.asarray(projected)
    if targets.ndim != 2 or targets.shape[1] < 1:
        raise ValueError("projected must have shape (n, q) with q positive.")
    tau = _batch_scalar(raw_log_mass, name="raw_log_mass")
    if tau.shape[0] != targets.shape[0]:
        raise ValueError("projected and raw_log_mass batch sizes must match.")
    conditions = ((tau - center) / scale)[:, None]
    return jax.vmap(distribution.log_prob)(targets, conditions)


def conditional_log_prob_and_observation_score(
    distribution: Any,
    projected: jax.Array,
    raw_log_mass: jax.Array,
    *,
    condition_center: float,
    condition_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate log density and its projected-coordinate score per row."""
    center = _finite_scalar(condition_center, name="condition_center")
    scale = _finite_scalar(condition_scale, name="condition_scale")
    if scale <= 0.0:
        raise ValueError("condition_scale must be positive.")
    targets = jnp.asarray(projected)
    if targets.ndim != 2 or targets.shape[1] < 1:
        raise ValueError("projected must have shape (n, q) with q positive.")
    tau = _batch_scalar(raw_log_mass, name="raw_log_mass")
    if tau.shape[0] != targets.shape[0]:
        raise ValueError("projected and raw_log_mass batch sizes must match.")
    conditions = ((tau - center) / scale)[:, None]

    def value_and_score_one(
        target: jax.Array,
        condition: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def log_prob(local_target: jax.Array) -> jax.Array:
            return distribution.log_prob(local_target, condition)

        value, linearized = jax.linearize(log_prob, target)
        score = jax.vmap(linearized)(jnp.eye(target.shape[0], dtype=target.dtype))
        return value, score

    return jax.vmap(value_and_score_one)(targets, conditions)


def raw_log_mass_condition_log_prob_and_score(
    distribution: Any,
    projected: jax.Array,
    raw_log_mass: jax.Array,
    *,
    condition_center: float,
    condition_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Public exact likelihood/partial-score pair for exploratory losses."""
    return _raw_log_mass_condition_log_prob_and_score(
        distribution,
        projected,
        raw_log_mass,
        condition_center=condition_center,
        condition_scale=condition_scale,
    )


def raw_log_mass_condition_score(
    distribution: Any,
    projected: jax.Array,
    raw_log_mass: jax.Array,
    *,
    condition_center: float,
    condition_scale: float,
) -> jax.Array:
    """Evaluate ``partial_tau log q(x | standardized(tau))`` per batch row."""
    _, score = _raw_log_mass_condition_log_prob_and_score(
        distribution,
        projected,
        raw_log_mass,
        condition_center=condition_center,
        condition_scale=condition_scale,
    )
    return score


class RawLogMassScoreLoss(eqx.Module):
    """Fit-to-data-compatible NLL plus raw-log-mass score loss.

    Data must be ``(projected, raw_log_mass, target_score)``.  Both terms use
    the frozen unit weight and are divided by projected dimension ``q``:

    ``mean(-log_prob) / q + mean((predicted_score - target_score)**2) / q``.
    """

    condition_center: float = eqx.field(static=True)
    condition_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        condition_center: float,
        condition_scale: float,
    ) -> None:
        center = _finite_scalar(condition_center, name="condition_center")
        scale = _finite_scalar(condition_scale, name="condition_scale")
        if scale <= 0.0:
            raise ValueError("condition_scale must be positive.")
        self.condition_center = center
        self.condition_scale = scale

    def __call__(
        self,
        params: Any,
        static: Any,
        projected: jax.Array,
        raw_log_mass: jax.Array,
        target_score: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Return the composite score-regularized loss; ``key`` is unused."""
        del key
        targets = jnp.asarray(projected)
        if targets.ndim != 2 or targets.shape[1] < 1:
            raise ValueError("projected must have shape (n, q) with q positive.")
        observed_score = _batch_scalar(target_score, name="target_score")
        if observed_score.shape[0] != targets.shape[0]:
            raise ValueError("projected and target_score batch sizes must match.")

        distribution = paramax.unwrap(eqx.combine(params, static))
        tau = _batch_scalar(raw_log_mass, name="raw_log_mass")
        if tau.shape[0] != targets.shape[0]:
            raise ValueError("projected and raw_log_mass batch sizes must match.")
        log_probabilities, predicted_score = _raw_log_mass_condition_log_prob_and_score(
            distribution,
            targets,
            tau,
            condition_center=self.condition_center,
            condition_scale=self.condition_scale,
        )
        dimension = jnp.asarray(targets.shape[1], dtype=log_probabilities.dtype)
        negative_log_likelihood = -jnp.mean(log_probabilities) / dimension
        score_risk = jnp.mean(jnp.square(predicted_score - observed_score)) / dimension
        return negative_log_likelihood + score_risk
