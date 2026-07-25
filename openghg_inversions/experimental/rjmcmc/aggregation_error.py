"""Exact quadrature oracle for aggregation error in a two-cell model.

The native positive masses are

``T ~ Gamma(a, b)``, ``W ~ Beta(r, s)``, and
``X = (T W, T (1 - W))``,

where the Gamma distribution uses shape ``a`` and rate ``b``.  Observations
are conditionally independent Gaussians with mean ``design @ X``.  The
coarse, exact representation retains ``T`` and analytically marginalizes
``W`` by endpoint-aware Gauss--Jacobi quadrature.  The fine representation
keeps both coordinates and integrates their product rule directly.

At finite quadrature order the coarse likelihood is already a normalized
conditional Gaussian mixture: every component is a normalized density and
the explicitly normalized Jacobi weights sum to one.  Consequently no neural
likelihood estimator is needed to establish the exact-representation limit.
The coarse and fine evidences use the likelihood exactly once and exclude any
structural partition prior.  Structural probabilities may be combined with
those evidences afterward using :func:`posterior_partition_probabilities`.

The public API deliberately stays small.  :func:`beta_quadrature` and
:func:`gamma_quadrature` expose the transformed probability rules;
:class:`TwoCellAggregationOracle` evaluates conditional likelihoods,
evidences, and common-total posterior quadrature weights.  Its log-space
methods are authoritative for long observation vectors; linear-space methods
are convenience wrappers.  :func:`log_posterior_partition_probabilities`
combines structural log probabilities with pure log evidences.  The
``nominal_fill_log_evidence`` method is an explicit reduced-model sentinel,
not part of the exact oracle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isfinite, lgamma, log, pi
from numbers import Integral
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import (
    eval_genlaguerre,
    eval_jacobi,
    logsumexp,
    roots_genlaguerre,
    roots_jacobi,
)

FloatArray: TypeAlias = NDArray[np.float64]
Representation: TypeAlias = Literal["coarse", "fine"]

__all__ = [
    "ProbabilityQuadrature",
    "TwoCellAggregationOracle",
    "beta_quadrature",
    "gamma_quadrature",
    "log_posterior_partition_probabilities",
    "posterior_partition_probabilities",
]

_LOG_TWO_PI = log(2.0 * pi)


def _positive_scalar(value: float, *, name: str) -> float:
    """Return one validated finite positive scalar.

    Args:
        value: Candidate scalar.
        name: Parameter name used in validation errors.

    Returns:
        Finite built-in float strictly greater than zero.

    Raises:
        TypeError: If ``value`` is Boolean.
        ValueError: If ``value`` is non-finite or non-positive.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _quadrature_order(value: int, *, name: str = "order") -> int:
    """Return one validated positive quadrature order.

    Args:
        value: Candidate quadrature order.
        name: Parameter name used in validation errors.

    Returns:
        Strictly positive built-in integer.

    Raises:
        TypeError: If ``value`` is Boolean or not integer-like.
        ValueError: If ``value`` is not strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _readonly(values: ArrayLike) -> FloatArray:
    """Return an owned, read-only ``float64`` array."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _stable_logsumexp(values: ArrayLike, *, axis: int | None = None) -> FloatArray:
    """Return SciPy log-sum-exp results with one stable array type.

    Args:
        values: Numerical values to reduce.
        axis: Optional reduction axis; ``None`` reduces all values.

    Returns:
        ``float64`` array, including a zero-dimensional array for scalar
        reductions.
    """
    return np.asarray(logsumexp(values, axis=axis), dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class ProbabilityQuadrature:
    """Finite quadrature rule for an explicitly normalized probability law.

    Args:
        nodes: One-dimensional quadrature nodes.
        weights: Non-negative probability weights corresponding to ``nodes``.

    Attributes:
        nodes: Read-only finite one-dimensional nodes.
        weights: Read-only non-negative weights that sum to one.

    Raises:
        ValueError: If the arrays are not finite one-dimensional vectors of
            equal nonzero length, weights are negative, or weights do not sum
            to one within floating-point tolerance.
    """

    nodes: FloatArray
    weights: FloatArray

    def __post_init__(self) -> None:
        """Validate and own the quadrature arrays."""
        nodes = _readonly(self.nodes)
        weights = _readonly(self.weights)
        if nodes.ndim != 1 or weights.ndim != 1 or nodes.size == 0:
            raise ValueError("nodes and weights must be non-empty one-dimensional arrays.")
        if nodes.shape != weights.shape:
            raise ValueError("nodes and weights must have the same shape.")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
            raise ValueError("nodes and weights must contain only finite values.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")
        if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=5.0e-14):
            raise ValueError("weights must sum to one.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)


def beta_quadrature(first_shape: float, second_shape: float, order: int) -> ProbabilityQuadrature:
    """Return an endpoint-aware Gauss--Jacobi rule for a Beta distribution.

    The Jacobi variable ``z`` is transformed by ``w = (z + 1) / 2``.
    ``scipy.special.roots_jacobi`` uses weight
    ``(1-z)**alpha * (1+z)**beta``; therefore its parameters are supplied in
    the reversed order ``(second_shape - 1, first_shape - 1)``.  Relative
    Christoffel weights are normalized explicitly, removing the Beta
    normalizer without forming an overflow-prone zeroth moment.

    Args:
        first_shape: Positive shape on ``w``.
        second_shape: Positive shape on ``1 - w``.
        order: Positive number of quadrature nodes.

    Returns:
        Probability quadrature on the open unit interval.

    Raises:
        TypeError: If ``order`` is not integer-like or a shape is Boolean.
        ValueError: If a shape or order is invalid, or SciPy produces
            non-finite quadrature values.
    """
    first = _positive_scalar(first_shape, name="first_shape")
    second = _positive_scalar(second_shape, name="second_shape")
    normalized_order = _quadrature_order(order)
    alpha = second - 1.0
    beta = first - 1.0
    # SciPy constructs the unnormalized zeroth moment internally, which can
    # overflow for concentrated Beta laws although the nodes remain finite.
    # Relative Christoffel weights avoid forming that irrelevant constant.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        jacobi_nodes, _ = roots_jacobi(normalized_order, alpha, beta)
    nodes = (np.asarray(jacobi_nodes, dtype=np.float64) + 1.0) / 2.0
    derivative_polynomial = eval_jacobi(
        normalized_order - 1,
        alpha + 1.0,
        beta + 1.0,
        jacobi_nodes,
    )
    log_weights = -np.log1p(-np.square(jacobi_nodes)) - 2.0 * np.log(np.abs(derivative_polynomial))
    weights = np.exp(log_weights - _stable_logsumexp(log_weights))
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
        raise ValueError("SciPy Gauss--Jacobi quadrature produced non-finite nodes or weights.")
    return ProbabilityQuadrature(nodes=nodes, weights=weights)


def gamma_quadrature(shape: float, rate: float, order: int) -> ProbabilityQuadrature:
    """Return a generalized Gauss--Laguerre rule for a Gamma distribution.

    For ``u = rate * t``, the Gamma expectation has generalized Laguerre
    weight ``u**(shape - 1) * exp(-u) / Gamma(shape)``.  Relative Christoffel
    weights are normalized explicitly, removing ``Gamma(shape)`` without
    forming that overflow-prone normalizer.

    Args:
        shape: Positive Gamma shape.
        rate: Positive Gamma rate.
        order: Positive number of quadrature nodes.

    Returns:
        Probability quadrature on positive totals ``t``.

    Raises:
        TypeError: If ``order`` is not integer-like or a parameter is Boolean.
        ValueError: If a parameter or order is not strictly positive and
            finite, or SciPy produces non-finite quadrature values.
    """
    normalized_shape = _positive_scalar(shape, name="shape")
    normalized_rate = _positive_scalar(rate, name="rate")
    normalized_order = _quadrature_order(order)
    alpha = normalized_shape - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        laguerre_nodes, _ = roots_genlaguerre(normalized_order, alpha)
    nodes = np.asarray(laguerre_nodes, dtype=np.float64) / normalized_rate
    derivative_polynomial = eval_genlaguerre(
        normalized_order - 1,
        alpha + 1.0,
        laguerre_nodes,
    )
    log_weights = -np.log(laguerre_nodes) - 2.0 * np.log(np.abs(derivative_polynomial))
    weights = np.exp(log_weights - _stable_logsumexp(log_weights))
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
        raise ValueError("SciPy generalized Gauss--Laguerre quadrature produced non-finite nodes or weights.")
    return ProbabilityQuadrature(nodes=nodes, weights=weights)


def _observation_model(
    observation: ArrayLike,
    design: ArrayLike,
    noise_sd: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate scalar/vector observations and their independent errors.

    Args:
        observation: Scalar or one-dimensional observation vector.
        design: Two-cell vector for scalar data or matrix with one row per
            observation.
        noise_sd: Positive scalar or independent standard deviations.

    Returns:
        Observation vector, two-column design matrix, and standard-deviation
        vector with compatible shapes.

    Raises:
        ValueError: If an input has an incompatible shape, contains
            non-finite values, or includes a non-positive error scale.
    """
    observed = np.asarray(observation, dtype=np.float64)
    if observed.ndim == 0:
        observed = observed.reshape(1)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observation must be a scalar or non-empty one-dimensional vector.")

    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim == 1 and matrix.shape == (2,) and observed.size == 1:
        matrix = matrix.reshape(1, 2)
    if matrix.shape != (observed.size, 2):
        raise ValueError("design must have shape (2,) for scalar data or (number_of_observations, 2).")

    scale = np.asarray(noise_sd, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(observed.size, float(scale), dtype=np.float64)
    if scale.shape != observed.shape:
        raise ValueError("noise_sd must be scalar or have one entry per observation.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(scale)):
        raise ValueError("observation, design, and noise_sd must contain only finite values.")
    if np.any(scale <= 0.0):
        raise ValueError("noise_sd must be strictly positive.")
    return observed, matrix, scale


def _gaussian_log_likelihood_for_mass(
    mass: FloatArray,
    *,
    observation: FloatArray,
    design: FloatArray,
    noise_sd: FloatArray,
) -> FloatArray:
    """Evaluate the normalized independent Gaussian log likelihood at masses.

    Args:
        mass: Array whose final axis contains the two native-cell masses.
        observation: One-dimensional observation vector.
        design: Two-column observation design matrix.
        noise_sd: Positive independent standard deviations.

    Returns:
        Log-likelihood array with the leading shape of ``mass``.
    """
    mean = np.einsum("...c,oc->...o", mass, design)
    residual = (observation - mean) / noise_sd
    log_density = -0.5 * np.sum(residual * residual, axis=-1)
    log_density -= float(np.sum(np.log(noise_sd))) + 0.5 * observation.size * _LOG_TWO_PI
    return cast(FloatArray, log_density)


def _gamma_log_density(total: FloatArray, *, shape: float, rate: float) -> FloatArray:
    """Evaluate a normalized Gamma shape--rate log density.

    Args:
        total: Array of strictly positive evaluation points.
        shape: Positive Gamma shape validated by the oracle.
        rate: Positive Gamma rate validated by the oracle.

    Returns:
        Same-shape Gamma log-density values.
    """
    return cast(
        FloatArray,
        shape * log(rate) - lgamma(shape) + (shape - 1.0) * np.log(total) - rate * total,
    )


def _log_weights(rule: ProbabilityQuadrature) -> FloatArray:
    """Return log probability weights, preserving exact zero weights."""
    with np.errstate(divide="ignore"):
        return cast(FloatArray, np.log(rule.weights))


@dataclass(frozen=True, slots=True)
class TwoCellAggregationOracle:
    """Exact coarse/fine quadrature oracle for two positive native cells.

    Args:
        gamma_shape: Shape of the common total ``T``.
        gamma_rate: Rate of the common total ``T``.
        beta_first_shape: Beta shape for the first-cell fraction ``W``.
        beta_second_shape: Beta shape for the second-cell fraction ``1-W``.
        total_order: Generalized Gauss--Laguerre order used for ``T``.
        fraction_order: Gauss--Jacobi order used for ``W``.

    Attributes:
        gamma_shape: Validated positive Gamma shape.
        gamma_rate: Validated positive Gamma rate.
        beta_first_shape: Validated positive first Beta shape.
        beta_second_shape: Validated positive second Beta shape.
        total_order: Positive Gamma quadrature order.
        fraction_order: Positive Beta quadrature order.

    Raises:
        TypeError: If an order is not integer-like or a scalar is Boolean.
        ValueError: If a distribution parameter or order is invalid.
    """

    gamma_shape: float
    gamma_rate: float
    beta_first_shape: float
    beta_second_shape: float
    total_order: int = 40
    fraction_order: int = 40

    def __post_init__(self) -> None:
        """Validate distribution parameters and quadrature orders."""
        object.__setattr__(self, "gamma_shape", _positive_scalar(self.gamma_shape, name="gamma_shape"))
        object.__setattr__(self, "gamma_rate", _positive_scalar(self.gamma_rate, name="gamma_rate"))
        object.__setattr__(
            self,
            "beta_first_shape",
            _positive_scalar(self.beta_first_shape, name="beta_first_shape"),
        )
        object.__setattr__(
            self,
            "beta_second_shape",
            _positive_scalar(self.beta_second_shape, name="beta_second_shape"),
        )
        object.__setattr__(self, "total_order", _quadrature_order(self.total_order, name="total_order"))
        object.__setattr__(
            self,
            "fraction_order",
            _quadrature_order(self.fraction_order, name="fraction_order"),
        )

    @property
    def nominal_fraction(self) -> float:
        """Return the prior mean first-cell fraction.

        Returns:
            ``beta_first_shape / (beta_first_shape + beta_second_shape)``.
        """
        return self.beta_first_shape / (self.beta_first_shape + self.beta_second_shape)

    def fraction_rule(self) -> ProbabilityQuadrature:
        """Return this oracle's normalized Beta quadrature.

        Returns:
            Gauss--Jacobi probability rule for ``W``.
        """
        return beta_quadrature(
            self.beta_first_shape,
            self.beta_second_shape,
            self.fraction_order,
        )

    def total_rule(self) -> ProbabilityQuadrature:
        """Return this oracle's normalized Gamma quadrature.

        Returns:
            Generalized Gauss--Laguerre probability rule for ``T``.
        """
        return gamma_quadrature(self.gamma_shape, self.gamma_rate, self.total_order)

    def coarse_conditional_log_likelihood(
        self,
        total: ArrayLike,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float | FloatArray:
        """Evaluate the exact coarse log likelihood ``log p(y | T)``.

        The finite Gaussian mixture is accumulated with ``logsumexp``.  A
        scalar total produces a scalar; an array produces the same shape.

        Args:
            total: Non-negative scalar or array of common totals.
            observation: Scalar or one-dimensional observation vector.
            design: Two-cell design vector for scalar data or matrix with
                shape ``(number_of_observations, 2)``.
            noise_sd: Positive scalar or per-observation independent standard
                deviations.

        Returns:
            Conditional log likelihood with the shape of ``total``.

        Raises:
            ValueError: If totals or observation-model inputs are malformed.
        """
        totals = np.asarray(total, dtype=np.float64)
        scalar = totals.ndim == 0
        if not np.all(np.isfinite(totals)) or np.any(totals < 0.0):
            raise ValueError("total must contain only finite non-negative values.")
        observed, matrix, scale = _observation_model(observation, design, noise_sd)
        rule = self.fraction_rule()
        first = totals[..., np.newaxis] * rule.nodes
        mass = np.stack((first, totals[..., np.newaxis] - first), axis=-1)
        log_likelihood = _gaussian_log_likelihood_for_mass(
            mass,
            observation=observed,
            design=matrix,
            noise_sd=scale,
        )
        result = _stable_logsumexp(log_likelihood + _log_weights(rule), axis=-1)
        if scalar:
            return float(result)
        return cast(FloatArray, result)

    def coarse_conditional_likelihood(
        self,
        total: ArrayLike,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float | FloatArray:
        """Evaluate the exact coarse likelihood ``p(y | T)``.

        This convenience wrapper exponentiates
        :meth:`coarse_conditional_log_likelihood`; use the log method when a
        long observation vector could underflow.

        Args:
            total: Non-negative scalar or array of common totals.
            observation: Scalar or one-dimensional observation vector.
            design: Two-cell design vector for scalar data or matrix with
                shape ``(number_of_observations, 2)``.
            noise_sd: Positive scalar or per-observation independent standard
                deviations.

        Returns:
            Conditional likelihood with the shape of ``total``.

        Raises:
            ValueError: If totals or observation-model inputs are malformed.
        """
        log_likelihood = self.coarse_conditional_log_likelihood(
            total,
            observation,
            design,
            noise_sd,
        )
        result = np.exp(log_likelihood)
        if np.ndim(log_likelihood) == 0:
            return float(result)
        return cast(FloatArray, result)

    def coarse_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Integrate the exact coarse conditional likelihood in log space.

        Structural partition probabilities are deliberately absent.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.

        Returns:
            Log normalized marginal data density.

        Raises:
            ValueError: If observation-model inputs have incompatible shapes,
                contain non-finite values, or include non-positive errors.
        """
        rule = self.total_rule()
        conditional = self.coarse_conditional_log_likelihood(
            rule.nodes,
            observation,
            design,
            noise_sd,
        )
        return float(_stable_logsumexp(_log_weights(rule) + conditional))

    def coarse_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Return the exact coarse evidence on the linear scale.

        This convenience wrapper may underflow for long observation vectors;
        :meth:`coarse_log_evidence` is the authoritative calculation.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.

        Returns:
            Normalized marginal data density, possibly zero after underflow.

        Raises:
            ValueError: If observation-model inputs have incompatible shapes,
                contain non-finite values, or include non-positive errors.
        """
        return float(np.exp(self.coarse_log_evidence(observation, design, noise_sd)))

    def fine_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Integrate the native likelihood over ``(T, W)`` in log space.

        This product quadrature is intentionally evaluated independently of
        :meth:`coarse_log_evidence`; equality is the tower identity.
        Structural partition probabilities are deliberately absent.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.

        Returns:
            Log normalized marginal data density.

        Raises:
            ValueError: If observation-model inputs have incompatible shapes,
                contain non-finite values, or include non-positive errors.
        """
        observed, matrix, scale = _observation_model(observation, design, noise_sd)
        total_rule = self.total_rule()
        fraction_rule = self.fraction_rule()
        first = total_rule.nodes[:, np.newaxis] * fraction_rule.nodes[np.newaxis, :]
        mass = np.stack((first, total_rule.nodes[:, np.newaxis] - first), axis=-1)
        log_likelihood = _gaussian_log_likelihood_for_mass(
            mass,
            observation=observed,
            design=matrix,
            noise_sd=scale,
        )
        log_joint_weights = (
            _log_weights(total_rule)[:, np.newaxis] + _log_weights(fraction_rule)[np.newaxis, :]
        )
        return float(_stable_logsumexp(log_joint_weights + log_likelihood))

    def fine_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Return the native two-cell evidence on the linear scale.

        This convenience wrapper may underflow for long observation vectors;
        :meth:`fine_log_evidence` is the authoritative calculation.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.

        Returns:
            Normalized marginal data density, possibly zero after underflow.

        Raises:
            ValueError: If observation-model inputs have incompatible shapes,
                contain non-finite values, or include non-positive errors.
        """
        return float(np.exp(self.fine_log_evidence(observation, design, noise_sd)))

    def nominal_fill_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        fraction: float | None = None,
    ) -> float:
        """Evaluate the deterministic-fill sentinel in log space.

        This approximation replaces ``W`` by one fixed fraction before
        integrating over ``T``.  It is useful for detecting aggregation error
        but is not an exact coarse likelihood except when the observation
        footprints are identical or the Beta law degenerates at that
        fraction.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.
            fraction: Optional first-cell fill fraction.  Defaults to the
                Beta prior mean.

        Returns:
            Log marginal density under deterministic within-region fill.

        Raises:
            TypeError: If ``fraction`` is Boolean.
            ValueError: If ``fraction`` lies outside the closed unit interval
                or observation-model inputs are malformed.
        """
        selected = self.nominal_fraction if fraction is None else float(fraction)
        if isinstance(fraction, bool):
            raise TypeError("fraction must be a real number.")
        if not isfinite(selected) or not 0.0 <= selected <= 1.0:
            raise ValueError("fraction must be finite and lie in [0, 1].")
        observed, matrix, scale = _observation_model(observation, design, noise_sd)
        rule = self.total_rule()
        first = rule.nodes * selected
        mass = np.stack((first, rule.nodes - first), axis=-1)
        log_likelihood = _gaussian_log_likelihood_for_mass(
            mass,
            observation=observed,
            design=matrix,
            noise_sd=scale,
        )
        return float(_stable_logsumexp(_log_weights(rule) + log_likelihood))

    def nominal_fill_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        fraction: float | None = None,
    ) -> float:
        """Return deterministic-fill evidence on the linear scale.

        This convenience wrapper may underflow for long observation vectors;
        :meth:`nominal_fill_log_evidence` is the authoritative calculation.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.
            fraction: Optional first-cell fill fraction.  Defaults to the
                Beta prior mean.

        Returns:
            Marginal density under deterministic within-region fill, possibly
            zero after underflow.

        Raises:
            TypeError: If ``fraction`` is Boolean.
            ValueError: If ``fraction`` lies outside the closed unit interval
                or observation-model inputs are malformed.
        """
        return float(
            np.exp(
                self.nominal_fill_log_evidence(
                    observation,
                    design,
                    noise_sd,
                    fraction=fraction,
                )
            )
        )

    def total_posterior_quadrature(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        representation: Representation,
    ) -> ProbabilityQuadrature:
        """Return posterior probability masses on the common ``T`` rule.

        The coarse path evaluates ``p(y | T)`` and the fine path first forms
        joint ``(T, W)`` likelihood weights and then marginalizes ``W``.  Their
        returned weights should agree up to floating-point summation.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.
            representation: Either ``"coarse"`` or ``"fine"``.

        Returns:
            Common Gamma nodes and normalized posterior masses.

        Raises:
            ValueError: If ``representation`` or model inputs are invalid.
        """
        if representation not in ("coarse", "fine"):
            raise ValueError("representation must be 'coarse' or 'fine'.")
        total_rule = self.total_rule()
        if representation == "coarse":
            log_conditional = np.asarray(
                self.coarse_conditional_log_likelihood(
                    total_rule.nodes,
                    observation,
                    design,
                    noise_sd,
                ),
                dtype=np.float64,
            )
        else:
            observed, matrix, scale = _observation_model(observation, design, noise_sd)
            fraction_rule = self.fraction_rule()
            first = total_rule.nodes[:, np.newaxis] * fraction_rule.nodes[np.newaxis, :]
            mass = np.stack((first, total_rule.nodes[:, np.newaxis] - first), axis=-1)
            log_likelihood = _gaussian_log_likelihood_for_mass(
                mass,
                observation=observed,
                design=matrix,
                noise_sd=scale,
            )
            log_conditional = _stable_logsumexp(
                log_likelihood + _log_weights(fraction_rule),
                axis=-1,
            )
        log_posterior_weights = _log_weights(total_rule) + log_conditional
        weights = np.exp(log_posterior_weights - _stable_logsumexp(log_posterior_weights))
        return ProbabilityQuadrature(nodes=total_rule.nodes, weights=weights)

    def total_posterior_density(
        self,
        total: ArrayLike,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float | FloatArray:
        """Evaluate the common total posterior density on arbitrary points.

        Args:
            total: Strictly positive scalar or array of total values.
            observation: Scalar or one-dimensional observation vector.
            design: Shape ``(2,)`` for scalar data or
                ``(number_of_observations, 2)`` for vector data.
            noise_sd: Positive scalar or shape
                ``(number_of_observations,)``.

        Returns:
            Posterior density with the shape of ``total``.

        Raises:
            ValueError: If totals are non-finite, not strictly positive, or
                model inputs are invalid.
        """
        totals = np.asarray(total, dtype=np.float64)
        scalar = totals.ndim == 0
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise ValueError("total must contain only finite strictly positive values.")
        log_conditional = np.asarray(
            self.coarse_conditional_log_likelihood(
                totals,
                observation,
                design,
                noise_sd,
            ),
            dtype=np.float64,
        )
        log_density = (
            _gamma_log_density(totals, shape=self.gamma_shape, rate=self.gamma_rate)
            + log_conditional
            - self.coarse_log_evidence(observation, design, noise_sd)
        )
        density = np.exp(log_density)
        if scalar:
            return float(density)
        return cast(FloatArray, density)


def log_posterior_partition_probabilities(
    log_structural_prior: ArrayLike,
    log_evidences: ArrayLike,
) -> FloatArray:
    """Combine log structural probabilities with pure log evidences.

    This is the only operation in this module that introduces a partition
    prior.  If exact coarse and fine representations have equal evidence, the
    result equals the supplied structural prior, including when that prior is
    nonuniform.  All normalization uses ``logsumexp``.

    Args:
        log_structural_prior: One-dimensional normalized log probabilities.
            Negative infinity is allowed for zero prior mass.
        log_evidences: Matching one-dimensional pure log evidences.  Negative
            infinity is allowed for zero evidence.

    Returns:
        Read-only normalized posterior log probabilities.

    Raises:
        ValueError: If arrays are malformed, contain NaN or positive infinity,
            the structural log prior is not normalized, or every posterior
            mass is zero.
    """
    log_prior = np.asarray(log_structural_prior, dtype=np.float64)
    log_likelihood = np.asarray(log_evidences, dtype=np.float64)
    if log_prior.ndim != 1 or log_prior.size == 0 or log_likelihood.shape != log_prior.shape:
        raise ValueError("log_structural_prior and log_evidences must be matching non-empty vectors.")
    if np.any(np.isnan(log_prior)) or np.any(np.isposinf(log_prior)):
        raise ValueError("log_structural_prior must contain finite values or negative infinity.")
    if not np.isclose(
        float(_stable_logsumexp(log_prior)),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("log_structural_prior must represent probabilities summing to one.")
    if np.any(np.isnan(log_likelihood)) or np.any(np.isposinf(log_likelihood)):
        raise ValueError("log_evidences must contain finite values or negative infinity.")
    log_unnormalized = log_prior + log_likelihood
    log_normalizer = float(_stable_logsumexp(log_unnormalized))
    if not isfinite(log_normalizer):
        raise ValueError("posterior partition log normalizer must be finite.")
    return _readonly(log_unnormalized - log_normalizer)


def posterior_partition_probabilities(
    structural_prior: ArrayLike,
    evidences: ArrayLike,
) -> FloatArray:
    """Combine structural probabilities with pure evidences.

    This linear convenience wrapper validates its inputs and delegates to
    :func:`log_posterior_partition_probabilities`, so the product and
    normalization remain stable when positive values are very small.

    Args:
        structural_prior: One-dimensional non-negative probabilities summing
            to one.
        evidences: Matching one-dimensional finite non-negative pure
            likelihood evidences.

    Returns:
        Read-only normalized posterior partition probabilities.

    Raises:
        ValueError: If arrays are malformed, the prior is not normalized, or
            the posterior normalizer is not positive and finite.
    """
    prior = np.asarray(structural_prior, dtype=np.float64)
    likelihood = np.asarray(evidences, dtype=np.float64)
    if prior.ndim != 1 or prior.size == 0 or likelihood.shape != prior.shape:
        raise ValueError("structural_prior and evidences must be matching non-empty vectors.")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("structural_prior must contain finite non-negative values.")
    if not np.isclose(float(prior.sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("structural_prior must sum to one.")
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
        raise ValueError("evidences must contain finite non-negative values.")
    with np.errstate(divide="ignore"):
        log_prior = np.log(prior)
        log_likelihood = np.log(likelihood)
    return _readonly(np.exp(log_posterior_partition_probabilities(log_prior, log_likelihood)))
