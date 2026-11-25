"""Stable fixed-shape Gamma quantiles for Gaussian-copula states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import grad_not_implemented, zero_grad
from pytensor.graph import Apply, Variable
from pytensor.link.jax.ops import JAXOp
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable
from scipy import special

if TYPE_CHECKING:
    from jax import Array


class _GammaStandardQuantile(JAXOp):
    """Map a standard-normal latent to a unit-scale Gamma with fixed shape.

    PyTensor uses SciPy for values and a log-density ratio for the latent
    gradient. JAX uses its own implementation of the same map and gradient.
    The shape is a prior parameter, so only the latent is differentiated.
    """

    def __init__(self, latent_type: TensorType, shape_type: TensorType) -> None:
        """Set the input and output types without importing JAX."""
        # The inherited node and JAX dispatch require only these types.
        # Import JAX at lowering, so CPU-only graph construction cannot set its precision.
        self.input_types = (latent_type, shape_type)
        self.output_types = (latent_type,)
        self.jax_func = _GammaStandardQuantile.perform_jax
        self.name = None

    def perform(self, node: Apply, inputs: Sequence[Any], output_storage: list[list[Any]]) -> None:
        """Evaluate the quantile in the latent's dtype using both probability tails."""
        latent, shape = inputs
        p = special.ndtr(-np.abs(latent))
        value = np.where(latent > 0, special.gammainccinv(shape, p), special.gammaincinv(shape, p))
        output_storage[0][0] = np.asarray(value, dtype=latent.dtype)

    def pullback(
        self, inputs: Sequence[Variable], outputs: Sequence[Variable],
        output_gradients: Sequence[Variable],
    ) -> list[Variable]:
        """Differentiate the inverse CDF via its density in log space."""
        latent, shape = (cast(TensorVariable, variable) for variable in inputs)
        value = zero_grad(cast(TensorVariable, outputs[0]))
        positive = value > np.finfo(latent.dtype).tiny
        safe = pt.switch(positive, value, 1.0)
        log_density = (shape - 1.0) * pt.log(safe) - safe - pt.gammaln(shape)
        log_normalizer = np.asarray(0.5 * np.log(2.0 * np.pi), dtype=latent.dtype)
        slope = pt.switch(positive, pt.exp(-0.5 * pt.square(latent) - log_normalizer - log_density), 0.0)
        return [cast(TensorVariable, output_gradients[0]) * zero_grad(slope), grad_not_implemented(self, 1, shape)]

    def perform_jax(self, latent: Array, shape: Array) -> Array:
        """Evaluate the quantile with a native JAX derivative for NumPyro and BlackJAX."""
        import jax
        import jax.numpy as jnp
        from jax.scipy import special as jsp
        from tensorflow_probability.substrates.jax import math as tfp_math

        @jax.custom_jvp
        def quantile(u: Array, k: Array) -> Array:
            """Use the inverse CDF centrally and the survival probability in the far right tail."""
            p = 0.5 * jsp.erfc(jnp.abs(u) / np.sqrt(2.0))
            eps = jnp.finfo(u.dtype).eps
            # TFP rounds 1 - p to one in the upper tail before returning infinity.
            use_tail = (u > 0) & (p < eps)
            upper = tfp_math.igammacinv(k, jnp.maximum(p, eps))
            lower = tfp_math.igammainv(k, p)

            def solve_tail() -> Array:
                """Return the bracketed upper-tail root for affected entries."""
                tail_p = jnp.where(use_tail, p, 0.5)
                # Markov's inequality at t=1/2 gives P(X > b) <= 2**k exp(-b/2).
                high = 2 * (k * np.log(2.0) - jnp.log(tail_p))

                def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
                    """Keep the interval that brackets the requested survival probability."""
                    low, high = bounds
                    midpoint = (low + high) * 0.5
                    too_low = jsp.gammaincc(k, midpoint) > tail_p
                    return jnp.where(too_low, midpoint, low), jnp.where(too_low, high, midpoint)

                low, high = jax.lax.fori_loop(0, 32, bisect, (jnp.zeros_like(high), high))
                return jnp.where(use_tail, (low + high) * 0.5, upper)

            upper = jax.lax.cond(jnp.any(use_tail), solve_tail, lambda: upper)
            return jnp.where(u > 0, upper, lower)

        @quantile.defjvp
        def quantile_jvp(primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array, Array]:
            """Apply the quantile's analytic derivative with respect to the latent."""
            u, k = primals
            value = quantile(u, k)
            positive = value > jnp.finfo(u.dtype).tiny
            safe = jnp.where(positive, value, 1.0)
            log_density = (k - 1.0) * jnp.log(safe) - safe - jsp.gammaln(k)
            slope = jnp.where(positive, jnp.exp(-0.5 * u**2 - 0.5 * np.log(2.0 * np.pi) - log_density), 0.0)
            return value, slope * tangents[0]

        return quantile(latent, shape)
