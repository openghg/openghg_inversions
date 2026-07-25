"""Exact quadrature oracles for aggregation error in small Gamma models.

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
evidences, and common-total posterior quadrature weights.
:class:`FourCellAggregationOracle` extends that construction to a row-major
``2 x 2`` native grid of independent, common-rate Gamma masses.  It evaluates
the root through independent row-first and column-first Beta charts and
compares root, row, column, and fine projection frontiers without depending
on a particular tree implementation.  The four-cell likelihood is reduced
to Gaussian sufficient statistics before quadrature and evaluated in bounded
chunks, so no quadrature-node by observation array is materialized.

Log-space methods are authoritative for long observation vectors;
linear-space methods are convenience wrappers.
:func:`log_posterior_partition_probabilities` combines structural log
probabilities with pure log evidences.  The ``nominal_fill_log_evidence``
methods are explicit reduced-model sentinels, not part of either exact
oracle.
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
FourCellTiling: TypeAlias = Literal["root", "row", "column", "fine"]
FourCellChart: TypeAlias = Literal["row-first", "column-first"]

__all__ = [
    "ProbabilityQuadrature",
    "FourCellAggregationOracle",
    "FourCellChart",
    "FourCellTiling",
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


def _observation_model_with_width(
    observation: ArrayLike,
    design: ArrayLike,
    noise_sd: ArrayLike,
    *,
    width: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate an independent-Gaussian observation model of fixed width.

    Args:
        observation: Scalar or one-dimensional observation vector.
        design: Width-vector for scalar data or matrix with one row per
            observation.
        noise_sd: Positive scalar or one standard deviation per observation.
        width: Required number of design columns.

    Returns:
        Observation vector, design matrix, and standard-deviation vector.

    Raises:
        ValueError: If an input has an incompatible shape, is non-finite, or
            includes a non-positive standard deviation.
    """
    observed = np.asarray(observation, dtype=np.float64)
    if observed.ndim == 0:
        observed = observed.reshape(1)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observation must be a scalar or non-empty one-dimensional vector.")

    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim == 1 and matrix.shape == (width,) and observed.size == 1:
        matrix = matrix.reshape(1, width)
    if matrix.shape != (observed.size, width):
        raise ValueError(
            f"design must have shape ({width},) for scalar data or (number_of_observations, {width})."
        )

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


@dataclass(frozen=True, slots=True)
class _GaussianSufficientStatistics:
    """Centered sufficient statistics for a diagonal Gaussian likelihood.

    A weighted least-squares reference mass is computed once per observation
    model. Quadratic likelihoods are evaluated in displacement coordinates
    around that reference rather than by subtracting three large
    origin-centered terms. This costs one four-column least-squares solve and
    retains standardized inputs only for direct-residual fallbacks. A fallback
    is used for non-positive cancellation artifacts and fits whose residual
    quadratic is below ``sqrt(machine epsilon)`` times the standardized
    observation quadratic. This may make exceptionally close fits cost
    ``O(number_of_observations)`` each; ordinary quadrature nodes retain the
    bounded sufficient-statistic fast path.
    """

    log_normalizer: float
    observation_quadratic: float
    reference_mass: FloatArray
    reference_quadratic: float
    centered_linear: FloatArray
    gram: FloatArray
    standardized_observation: FloatArray
    standardized_design: FloatArray

    @classmethod
    def from_observation_model(
        cls,
        observation: FloatArray,
        design: FloatArray,
        noise_sd: FloatArray,
    ) -> _GaussianSufficientStatistics:
        """Reduce observations to a constant, linear term, and Gram matrix."""
        standardized_observation = observation / noise_sd
        standardized_design = design / noise_sd[:, np.newaxis]
        reference_mass, _, _, _ = np.linalg.lstsq(
            standardized_design,
            standardized_observation,
            rcond=None,
        )
        reference_residual = standardized_observation - standardized_design @ reference_mass
        reference_quadratic = float(np.dot(reference_residual, reference_residual))
        observation_quadratic = float(np.dot(standardized_observation, standardized_observation))
        log_normalizer = -float(np.sum(np.log(noise_sd))) - 0.5 * observation.size * _LOG_TWO_PI
        return cls(
            log_normalizer=log_normalizer,
            observation_quadratic=observation_quadratic,
            reference_mass=_readonly(reference_mass),
            reference_quadratic=reference_quadratic,
            centered_linear=_readonly(standardized_design.T @ reference_residual),
            gram=_readonly(standardized_design.T @ standardized_design),
            standardized_observation=_readonly(standardized_observation),
            standardized_design=_readonly(standardized_design),
        )

    def log_likelihood(self, mass: FloatArray) -> FloatArray:
        """Evaluate normalized Gaussian log likelihoods from native masses."""
        displacement = mass - self.reference_mass
        linear_term = displacement @ self.centered_linear
        quadratic_term = np.einsum("...i,ij,...j->...", displacement, self.gram, displacement)
        residual_quadratic = self.reference_quadratic - 2.0 * linear_term + quadratic_term
        cancellation_scale = (
            abs(self.reference_quadratic) + 2.0 * np.abs(linear_term) + np.abs(quadratic_term)
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * cancellation_scale
        # Even a centered quadratic loses a few digits when the requested mass
        # is extremely close to a huge fitted signal. Direct evaluation is
        # bounded to relative near-fits; ordinary quadrature nodes retain the
        # sufficient-statistic fast path.
        near_fit_tolerance = np.sqrt(np.finfo(np.float64).eps) * self.observation_quadratic
        needs_direct = (
            (residual_quadratic < 0.0)
            | ((cancellation_scale > 0.0) & (np.abs(residual_quadratic) <= tolerance))
            | ((self.observation_quadratic > 0.0) & (residual_quadratic <= near_fit_tolerance))
        )
        if np.any(needs_direct):
            flat_mass = mass.reshape(-1, mass.shape[-1])
            flat_quadratic = np.asarray(residual_quadratic).reshape(-1).copy()
            for index in np.flatnonzero(np.asarray(needs_direct).reshape(-1)):
                residual = self.standardized_observation - self.standardized_design @ flat_mass[index]
                flat_quadratic[index] = float(np.dot(residual, residual))
            residual_quadratic = flat_quadratic.reshape(np.shape(residual_quadratic))
        return cast(
            FloatArray,
            self.log_normalizer - 0.5 * residual_quadratic,
        )


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


@dataclass(frozen=True, slots=True, eq=False)
class FourCellAggregationOracle:
    """Exact projection oracle for a row-major ``2 x 2`` Gamma native grid.

    The four native masses are independent
    ``X_i ~ Gamma(native_shapes[i], gamma_rate)``.  Their total is therefore
    Gamma with shape ``sum(native_shapes)`` and the normalized masses are
    Dirichlet.  Two neutral-to-the-right factorizations of that Dirichlet law
    provide independent-Beta charts:

    - row-first: total first-row share, then the two within-row shares;
    - column-first: total first-column share, then the two within-column
      shares.

    Cell order is ``(top-left, top-right, bottom-left, bottom-right)``.  The
    four projection frontiers are root, two row totals, two column totals, and
    the four native cells.  Every exact frontier analytically (by quadrature)
    integrates the native conditional allocation it hides.  Consequently its
    evidence and total posterior are properties of one common native model,
    not of the chosen projection.

    Args:
        native_shapes: Four positive native-cell Gamma shapes in row-major
            order.
        gamma_rate: Positive common Gamma rate.
        total_order: Generalized Gauss--Laguerre order for the common total.
        fraction_order: Gauss--Jacobi order for every independent Beta share.
        chunk_size: Maximum number of four-cell mass vectors evaluated in one
            likelihood batch.

    Attributes:
        native_shapes: Read-only four-vector of validated Gamma shapes.
        gamma_rate: Validated positive common Gamma rate.
        total_order: Positive total quadrature order.
        fraction_order: Positive share quadrature order.
        chunk_size: Positive bound on a likelihood evaluation batch.

    Raises:
        TypeError: If an order is not integer-like or a scalar is Boolean.
        ValueError: If shapes, rate, orders, or chunk size are invalid.

    Notes:
        Gaussian likelihood evaluation uses the sufficient statistics
        ``H.T @ D^-1 @ H`` and ``H.T @ D^-1 @ y``.  Quadrature therefore does
        not construct an array whose dimensions include both observations and
        quadrature nodes.
    """

    native_shapes: FloatArray
    gamma_rate: float
    total_order: int = 24
    fraction_order: int = 16
    chunk_size: int = 16_384

    def __post_init__(self) -> None:
        """Validate and own distribution and quadrature configuration."""
        raw_shapes = np.asarray(self.native_shapes, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in raw_shapes.flat):
            raise TypeError("native_shapes must contain real numbers, not Boolean values.")
        shapes = _readonly(self.native_shapes)
        if shapes.shape != (4,) or not np.all(np.isfinite(shapes)) or np.any(shapes <= 0.0):
            raise ValueError("native_shapes must contain four finite strictly positive values.")
        object.__setattr__(self, "native_shapes", shapes)
        object.__setattr__(self, "gamma_rate", _positive_scalar(self.gamma_rate, name="gamma_rate"))
        object.__setattr__(self, "total_order", _quadrature_order(self.total_order, name="total_order"))
        object.__setattr__(
            self,
            "fraction_order",
            _quadrature_order(self.fraction_order, name="fraction_order"),
        )
        object.__setattr__(
            self,
            "chunk_size",
            _quadrature_order(self.chunk_size, name="chunk_size"),
        )

    @property
    def gamma_shape(self) -> float:
        """Return the common-total Gamma shape."""
        return float(self.native_shapes.sum())

    @property
    def nominal_fractions(self) -> FloatArray:
        """Return prior-mean native fractions of the common total."""
        return _readonly(self.native_shapes / self.gamma_shape)

    def total_rule(self) -> ProbabilityQuadrature:
        """Return normalized quadrature for the common native total."""
        return gamma_quadrature(self.gamma_shape, self.gamma_rate, self.total_order)

    def projection(self, tiling: FourCellTiling) -> FloatArray:
        """Return the linear native-to-region projection for one frontier.

        Args:
            tiling: ``"root"``, ``"row"``, ``"column"``, or ``"fine"``.

        Returns:
            Read-only matrix mapping the row-major four-cell mass vector to
            the active region totals.

        Raises:
            ValueError: If ``tiling`` is unknown.
        """
        if tiling == "root":
            return _readonly([[1.0, 1.0, 1.0, 1.0]])
        if tiling == "row":
            return _readonly([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        if tiling == "column":
            return _readonly([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        if tiling == "fine":
            return _readonly(np.eye(4, dtype=np.float64))
        raise ValueError("tiling must be 'root', 'row', 'column', or 'fine'.")

    def _chart_rules(
        self,
        chart: FourCellChart,
    ) -> tuple[ProbabilityQuadrature, ProbabilityQuadrature, ProbabilityQuadrature]:
        """Return the three independent Beta rules for one Dirichlet chart."""
        a0, a1, a2, a3 = self.native_shapes
        if chart == "row-first":
            return (
                beta_quadrature(a0 + a1, a2 + a3, self.fraction_order),
                beta_quadrature(a0, a1, self.fraction_order),
                beta_quadrature(a2, a3, self.fraction_order),
            )
        if chart == "column-first":
            return (
                beta_quadrature(a0 + a2, a1 + a3, self.fraction_order),
                beta_quadrature(a0, a2, self.fraction_order),
                beta_quadrature(a1, a3, self.fraction_order),
            )
        raise ValueError("chart must be 'row-first' or 'column-first'.")

    def _fraction_chunk(
        self,
        chart: FourCellChart,
        start: int,
        stop: int,
        *,
        rules: tuple[
            ProbabilityQuadrature,
            ProbabilityQuadrature,
            ProbabilityQuadrature,
        ]
        | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Materialize one bounded chunk of Dirichlet fractions and weights."""
        aggregate_rule, first_rule, second_rule = self._chart_rules(chart) if rules is None else rules
        order = self.fraction_order
        flat_index = np.arange(start, stop, dtype=np.int64)
        aggregate_index = flat_index // (order * order)
        first_index = (flat_index // order) % order
        second_index = flat_index % order
        aggregate = aggregate_rule.nodes[aggregate_index]
        first = first_rule.nodes[first_index]
        second = second_rule.nodes[second_index]

        fractions = np.empty((flat_index.size, 4), dtype=np.float64)
        if chart == "row-first":
            fractions[:, 0] = aggregate * first
            fractions[:, 1] = aggregate * (1.0 - first)
            fractions[:, 2] = (1.0 - aggregate) * second
            fractions[:, 3] = (1.0 - aggregate) * (1.0 - second)
        else:
            fractions[:, 0] = aggregate * first
            fractions[:, 2] = aggregate * (1.0 - first)
            fractions[:, 1] = (1.0 - aggregate) * second
            fractions[:, 3] = (1.0 - aggregate) * (1.0 - second)

        log_weights = (
            _log_weights(aggregate_rule)[aggregate_index]
            + _log_weights(first_rule)[first_index]
            + _log_weights(second_rule)[second_index]
        )
        return fractions, cast(FloatArray, log_weights)

    def _conditional_log_likelihood_by_total(
        self,
        statistics: _GaussianSufficientStatistics,
        *,
        chart: FourCellChart,
    ) -> FloatArray:
        """Integrate all three chart shares conditional on every total node."""
        return cast(
            FloatArray,
            self._conditional_log_likelihood_for_totals(
                self.total_rule().nodes,
                statistics,
                chart=chart,
            ),
        )

    def _conditional_log_likelihood_for_totals(
        self,
        totals: FloatArray,
        statistics: _GaussianSufficientStatistics,
        *,
        chart: FourCellChart,
    ) -> FloatArray:
        """Integrate all three chart shares at arbitrary total values."""
        flat_totals = totals.reshape(-1)
        fraction_count = self.fraction_order**3
        rules = self._chart_rules(chart)
        conditional = np.full(flat_totals.size, -np.inf, dtype=np.float64)
        for total_index, total in enumerate(flat_totals):
            accumulator = -np.inf
            for start in range(0, fraction_count, self.chunk_size):
                stop = min(start + self.chunk_size, fraction_count)
                fractions, log_weights = self._fraction_chunk(
                    chart,
                    start,
                    stop,
                    rules=rules,
                )
                log_likelihood = statistics.log_likelihood(total * fractions)
                chunk_value = float(_stable_logsumexp(log_weights + log_likelihood))
                accumulator = float(np.logaddexp(accumulator, chunk_value))
            conditional[total_index] = accumulator
        return conditional.reshape(totals.shape)

    def _conditional_log_likelihood_for_pairs(
        self,
        region_masses: FloatArray,
        statistics: _GaussianSufficientStatistics,
        *,
        tiling: Literal["row", "column"],
        rules: tuple[ProbabilityQuadrature, ProbabilityQuadrature] | None = None,
    ) -> FloatArray:
        """Integrate two within-region shares at row or column totals."""
        a0, a1, a2, a3 = self.native_shapes
        if rules is not None:
            first_rule, second_rule = rules
        elif tiling == "row":
            first_rule = beta_quadrature(a0, a1, self.fraction_order)
            second_rule = beta_quadrature(a2, a3, self.fraction_order)
        else:
            first_rule = beta_quadrature(a0, a2, self.fraction_order)
            second_rule = beta_quadrature(a1, a3, self.fraction_order)
        pair_count = self.fraction_order**2
        flat_masses = region_masses.reshape(-1, 2)
        conditional = np.full(flat_masses.shape[0], -np.inf, dtype=np.float64)
        for mass_index, (first_mass, second_mass) in enumerate(flat_masses):
            accumulator = -np.inf
            for start in range(0, pair_count, self.chunk_size):
                stop = min(start + self.chunk_size, pair_count)
                flat_index = np.arange(start, stop, dtype=np.int64)
                first_index = flat_index // self.fraction_order
                second_index = flat_index % self.fraction_order
                first = first_rule.nodes[first_index]
                second = second_rule.nodes[second_index]
                native = np.empty((flat_index.size, 4), dtype=np.float64)
                if tiling == "row":
                    native[:, 0] = first_mass * first
                    native[:, 1] = first_mass * (1.0 - first)
                    native[:, 2] = second_mass * second
                    native[:, 3] = second_mass * (1.0 - second)
                else:
                    native[:, 0] = first_mass * first
                    native[:, 2] = first_mass * (1.0 - first)
                    native[:, 1] = second_mass * second
                    native[:, 3] = second_mass * (1.0 - second)
                log_weights = _log_weights(first_rule)[first_index] + _log_weights(second_rule)[second_index]
                chunk_value = float(_stable_logsumexp(log_weights + statistics.log_likelihood(native)))
                accumulator = float(np.logaddexp(accumulator, chunk_value))
            conditional[mass_index] = accumulator
        return conditional.reshape(region_masses.shape[:-1])

    def _pair_frontier_conditional_by_total(
        self,
        statistics: _GaussianSufficientStatistics,
        *,
        tiling: Literal["row", "column"],
    ) -> FloatArray:
        """Integrate a pair-frontier likelihood over its aggregate share."""
        a0, a1, a2, a3 = self.native_shapes
        if tiling == "row":
            aggregate_rule = beta_quadrature(
                a0 + a1,
                a2 + a3,
                self.fraction_order,
            )
        else:
            aggregate_rule = beta_quadrature(
                a0 + a2,
                a1 + a3,
                self.fraction_order,
            )
        total_rule = self.total_rule()
        conditional = np.empty(total_rule.nodes.size, dtype=np.float64)
        log_aggregate_weights = _log_weights(aggregate_rule)
        if tiling == "row":
            within_rules = (
                beta_quadrature(a0, a1, self.fraction_order),
                beta_quadrature(a2, a3, self.fraction_order),
            )
        else:
            within_rules = (
                beta_quadrature(a0, a2, self.fraction_order),
                beta_quadrature(a1, a3, self.fraction_order),
            )
        for total_index, total in enumerate(total_rule.nodes):
            region_masses = np.stack(
                (
                    total * aggregate_rule.nodes,
                    total * (1.0 - aggregate_rule.nodes),
                ),
                axis=-1,
            )
            pair_conditional = self._conditional_log_likelihood_for_pairs(
                region_masses,
                statistics,
                tiling=tiling,
                rules=within_rules,
            )
            conditional[total_index] = float(_stable_logsumexp(log_aggregate_weights + pair_conditional))
        return conditional

    @staticmethod
    def _statistics(
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> _GaussianSufficientStatistics:
        """Validate a four-cell model and construct Gaussian statistics."""
        observed, matrix, scale = _observation_model_with_width(
            observation,
            design,
            noise_sd,
            width=4,
        )
        return _GaussianSufficientStatistics.from_observation_model(observed, matrix, scale)

    def _chart_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        chart: FourCellChart,
    ) -> float:
        """Evaluate common-native evidence through one independent-Beta chart."""
        statistics = self._statistics(observation, design, noise_sd)
        total_rule = self.total_rule()
        conditional = self._conditional_log_likelihood_by_total(statistics, chart=chart)
        return float(_stable_logsumexp(_log_weights(total_rule) + conditional))

    def root_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        chart: FourCellChart = "row-first",
    ) -> float:
        """Return exact root evidence using a selected Dirichlet chart.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Four-cell vector or matrix with four columns.
            noise_sd: Positive scalar or per-observation standard deviations.
            chart: Independent-Beta factorization used to marginalize all
                native fractions.

        Returns:
            Log normalized marginal data density.

        Raises:
            ValueError: If model inputs or ``chart`` are invalid.
        """
        return self._chart_log_evidence(observation, design, noise_sd, chart=chart)

    def conditional_log_likelihood(
        self,
        region_masses: ArrayLike,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling,
        root_chart: FourCellChart = "row-first",
    ) -> float | FloatArray:
        """Evaluate the exact native likelihood conditional on region totals.

        Hidden native allocations are integrated under their normalized
        conditional Dirichlet laws.  This makes the projective tower
        explicit without exposing tree histories or chart coordinates.

        Args:
            region_masses: Scalar or arbitrary array of totals for root.
                The final axis has length two, two, or four for row, column,
                or fine respectively.
            observation: Scalar or one-dimensional observation vector.
            design: Four-cell vector or matrix with four columns.
            noise_sd: Positive scalar or per-observation standard deviations.
            tiling: Projection frontier conditioning the likelihood.
            root_chart: Auxiliary chart for the root hidden allocation.

        Returns:
            Scalar or array of log likelihoods with the leading shape of
            ``region_masses``.

        Raises:
            ValueError: If masses, inputs, tiling, or chart are invalid.
        """
        masses = np.asarray(region_masses, dtype=np.float64)
        statistics = self._statistics(observation, design, noise_sd)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("region_masses must contain finite non-negative values.")

        if tiling == "root":
            totals = masses
            scalar = totals.ndim == 0
            result = self._conditional_log_likelihood_for_totals(
                np.asarray(totals, dtype=np.float64),
                statistics,
                chart=root_chart,
            )
        elif tiling in ("row", "column"):
            if masses.ndim == 0 or masses.shape[-1] != 2:
                raise ValueError(f"{tiling} region_masses must have final axis length 2.")
            scalar = masses.ndim == 1
            result = self._conditional_log_likelihood_for_pairs(
                masses,
                statistics,
                tiling=tiling,
            )
        elif tiling == "fine":
            if masses.ndim == 0 or masses.shape[-1] != 4:
                raise ValueError("fine region_masses must have final axis length 4.")
            scalar = masses.ndim == 1
            result = statistics.log_likelihood(masses)
        else:
            raise ValueError("tiling must be 'root', 'row', 'column', or 'fine'.")
        if scalar:
            return float(result)
        return cast(FloatArray, result)

    def row_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Return exact evidence at the two-row projection frontier."""
        statistics = self._statistics(observation, design, noise_sd)
        total_rule = self.total_rule()
        conditional = self._pair_frontier_conditional_by_total(
            statistics,
            tiling="row",
        )
        return float(_stable_logsumexp(_log_weights(total_rule) + conditional))

    def column_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
    ) -> float:
        """Return exact evidence at the two-column projection frontier."""
        statistics = self._statistics(observation, design, noise_sd)
        total_rule = self.total_rule()
        conditional = self._pair_frontier_conditional_by_total(
            statistics,
            tiling="column",
        )
        return float(_stable_logsumexp(_log_weights(total_rule) + conditional))

    def fine_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        chart: FourCellChart = "row-first",
    ) -> float:
        """Return native four-cell evidence through one Dirichlet chart."""
        return self._chart_log_evidence(observation, design, noise_sd, chart=chart)

    def log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling,
        root_chart: FourCellChart = "row-first",
    ) -> float:
        """Return exact log evidence for any projection frontier.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Four-cell vector or matrix with four columns.
            noise_sd: Positive scalar or per-observation standard deviations.
            tiling: Root, row, column, or fine projection.
            root_chart: Chart used for root and fine evaluations.  Row and
                column frontiers use their corresponding chart.

        Returns:
            Log normalized marginal data density.

        Raises:
            ValueError: If ``tiling``, ``root_chart``, or inputs are invalid.
        """
        if tiling == "root":
            return self.root_log_evidence(
                observation,
                design,
                noise_sd,
                chart=root_chart,
            )
        if tiling == "row":
            return self.row_log_evidence(observation, design, noise_sd)
        if tiling == "column":
            return self.column_log_evidence(observation, design, noise_sd)
        if tiling == "fine":
            return self.fine_log_evidence(
                observation,
                design,
                noise_sd,
                chart=root_chart,
            )
        raise ValueError("tiling must be 'root', 'row', 'column', or 'fine'.")

    def evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling,
        root_chart: FourCellChart = "row-first",
    ) -> float:
        """Return exact evidence on the linear scale.

        This convenience method may underflow for long observation vectors;
        :meth:`log_evidence` is authoritative.
        """
        return float(
            np.exp(
                self.log_evidence(
                    observation,
                    design,
                    noise_sd,
                    tiling=tiling,
                    root_chart=root_chart,
                )
            )
        )

    def total_posterior_quadrature(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling,
        root_chart: FourCellChart = "row-first",
    ) -> ProbabilityQuadrature:
        """Return posterior probability masses on the common-total rule.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Four-cell vector or matrix with four columns.
            noise_sd: Positive scalar or per-observation standard deviations.
            tiling: Root, row, column, or fine projection.
            root_chart: Chart for root and fine frontiers.

        Returns:
            Common Gamma nodes and normalized posterior masses.

        Raises:
            ValueError: If a model input, tiling, or chart is invalid.
        """
        if tiling in ("root", "fine"):
            chart = root_chart
        elif tiling == "row":
            chart = "row-first"
        elif tiling == "column":
            chart = "column-first"
        else:
            raise ValueError("tiling must be 'root', 'row', 'column', or 'fine'.")
        statistics = self._statistics(observation, design, noise_sd)
        total_rule = self.total_rule()
        if tiling in ("root", "fine"):
            conditional = self._conditional_log_likelihood_by_total(
                statistics,
                chart=chart,
            )
        else:
            conditional = self._pair_frontier_conditional_by_total(
                statistics,
                tiling=tiling,
            )
        log_weights = _log_weights(total_rule) + conditional
        weights = np.exp(log_weights - _stable_logsumexp(log_weights))
        return ProbabilityQuadrature(nodes=total_rule.nodes, weights=weights)

    def _nominal_conditional_by_total(
        self,
        statistics: _GaussianSufficientStatistics,
        *,
        tiling: FourCellTiling,
    ) -> FloatArray:
        """Evaluate deterministic-fill conditional likelihoods by total node."""
        total_rule = self.total_rule()
        if tiling == "root":
            mass = total_rule.nodes[:, np.newaxis] * self.nominal_fractions
            return statistics.log_likelihood(mass)
        if tiling == "fine":
            return self._conditional_log_likelihood_by_total(
                statistics,
                chart="row-first",
            )

        a0, a1, a2, a3 = self.native_shapes
        if tiling == "row":
            aggregate_rule = beta_quadrature(
                a0 + a1,
                a2 + a3,
                self.fraction_order,
            )
            aggregate = aggregate_rule.nodes
            fractions = np.stack(
                (
                    aggregate * a0 / (a0 + a1),
                    aggregate * a1 / (a0 + a1),
                    (1.0 - aggregate) * a2 / (a2 + a3),
                    (1.0 - aggregate) * a3 / (a2 + a3),
                ),
                axis=-1,
            )
        elif tiling == "column":
            aggregate_rule = beta_quadrature(
                a0 + a2,
                a1 + a3,
                self.fraction_order,
            )
            aggregate = aggregate_rule.nodes
            fractions = np.stack(
                (
                    aggregate * a0 / (a0 + a2),
                    (1.0 - aggregate) * a1 / (a1 + a3),
                    aggregate * a2 / (a0 + a2),
                    (1.0 - aggregate) * a3 / (a1 + a3),
                ),
                axis=-1,
            )
        else:
            raise ValueError("tiling must be 'root', 'row', 'column', or 'fine'.")

        conditional = np.empty(total_rule.nodes.size, dtype=np.float64)
        log_fraction_weights = _log_weights(aggregate_rule)
        for index, total in enumerate(total_rule.nodes):
            conditional[index] = float(
                _stable_logsumexp(log_fraction_weights + statistics.log_likelihood(total * fractions))
            )
        return conditional

    def nominal_fill_log_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling = "root",
    ) -> float:
        """Return a deterministic hidden-allocation sentinel.

        For root, every native fraction is fixed at its Dirichlet mean.  At
        row or column frontiers the retained aggregate share is integrated
        while each hidden within-region split is fixed at its conditional
        mean.  Fine has no hidden allocation and therefore returns exact
        common-native evidence.

        Args:
            observation: Scalar or one-dimensional observation vector.
            design: Four-cell vector or matrix with four columns.
            noise_sd: Positive scalar or per-observation standard deviations.
            tiling: Projection frontier whose hidden allocations are filled.

        Returns:
            Log marginal density under deterministic hidden allocation.

        Raises:
            ValueError: If model inputs or ``tiling`` are invalid.
        """
        statistics = self._statistics(observation, design, noise_sd)
        total_rule = self.total_rule()
        conditional = self._nominal_conditional_by_total(statistics, tiling=tiling)
        return float(_stable_logsumexp(_log_weights(total_rule) + conditional))

    def nominal_fill_evidence(
        self,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        tiling: FourCellTiling = "root",
    ) -> float:
        """Return deterministic hidden-allocation evidence on a linear scale."""
        return float(
            np.exp(
                self.nominal_fill_log_evidence(
                    observation,
                    design,
                    noise_sd,
                    tiling=tiling,
                )
            )
        )


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
