"""Dense linear-Gaussian oracle for exact Bocquet projections.

This experimental module implements the finite-dimensional reference algebra
for a native state ``x ~ N(mu, B)`` and observations
``y | x ~ N(H x, R)``.  A full-row-rank restriction ``Gamma`` defines the
reported coefficient ``alpha = Gamma x``.  Its Bayesian conditional
prolongation and unresolved covariance are retained explicitly, so direct
inference for ``alpha`` has exactly the same moments as restricting the native
posterior.

The implementation intentionally uses dense NumPy arrays and Cholesky solves.
It is a correctness oracle for small projection problems, including correlated
native and observation covariances, rather than a production large-grid
solver.  Result arrays are copied and marked read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    """Immutable moments and innovation from Gaussian conditioning.

    Attributes:
        mean: Posterior state mean.
        covariance: Posterior state covariance.
        innovation: Observation minus its prior mean.
        innovation_covariance: Prior-predictive observation covariance used
            for conditioning.
    """

    mean: np.ndarray
    covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray

    def __post_init__(self) -> None:
        """Copy all result arrays and make them read-only."""
        _freeze_array_fields(
            self,
            ("mean", "covariance", "innovation", "innovation_covariance"),
        )


@dataclass(frozen=True, slots=True)
class BocquetProjection:
    """Exact prior-conditional reduction induced by one restriction.

    The reduced coefficient is ``alpha = restriction @ x``.  Its conditional
    native mean is
    ``mu + conditional_prolongation @ (alpha - projected_prior_mean)``.
    Consequently, ``reduced_design`` acts on coefficient departures from the
    projected prior mean, while ``reduced_likelihood_offset`` supplies the
    equivalent absolute-coefficient likelihood intercept.

    Attributes:
        restriction: Full-row-rank matrix ``Gamma`` with shape ``(K, M)``.
        native_prior_mean: Native prior mean ``mu``.
        observation_prior_mean: Native prior prediction ``H @ mu``.
        reduced_likelihood_offset: Reduced likelihood intercept
            ``H @ (mu - Lambda @ Gamma @ mu)``.
        projected_prior_mean: Restricted prior mean ``Gamma @ mu``.
        projected_prior_covariance: Restricted covariance
            ``B_Gamma = Gamma @ B @ Gamma.T``.
        conditional_prolongation: Bayesian prolongation
            ``Lambda = B @ Gamma.T @ B_Gamma^-1``.
        unresolved_covariance: Conditional covariance
            ``B_c = B - Lambda @ B_Gamma @ Lambda.T``.
        reduced_design: Conditional reduced design ``H @ Lambda``.
        base_observation_covariance: Declared covariance ``R``, before adding
            unresolved-state uncertainty.
        effective_observation_covariance: Exact reduced covariance
            ``R + H @ B_c @ H.T``.
        resolved_signal_covariance: Observation covariance represented by the
            reduced coefficient.
        native_signal_covariance: Full native signal covariance ``H @ B @ H.T``.
        innovation_covariance: Partition-invariant covariance
            ``R + H @ B @ H.T``.
    """

    restriction: np.ndarray
    native_prior_mean: np.ndarray
    observation_prior_mean: np.ndarray
    reduced_likelihood_offset: np.ndarray
    projected_prior_mean: np.ndarray
    projected_prior_covariance: np.ndarray
    conditional_prolongation: np.ndarray
    unresolved_covariance: np.ndarray
    reduced_design: np.ndarray
    base_observation_covariance: np.ndarray
    effective_observation_covariance: np.ndarray
    resolved_signal_covariance: np.ndarray
    native_signal_covariance: np.ndarray
    innovation_covariance: np.ndarray

    def __post_init__(self) -> None:
        """Copy all result arrays and make them read-only."""
        _freeze_array_fields(
            self,
            (
                "restriction",
                "native_prior_mean",
                "observation_prior_mean",
                "reduced_likelihood_offset",
                "projected_prior_mean",
                "projected_prior_covariance",
                "conditional_prolongation",
                "unresolved_covariance",
                "reduced_design",
                "base_observation_covariance",
                "effective_observation_covariance",
                "resolved_signal_covariance",
                "native_signal_covariance",
                "innovation_covariance",
            ),
        )


@dataclass(frozen=True, slots=True)
class GaussianProjectionAnalysis:
    """Native and directly reduced analyses for one projection problem.

    Attributes:
        native_posterior: Posterior moments in the native state coordinates.
        projection: Exact Bocquet prior restriction and likelihood terms.
        reduced_posterior: Posterior moments of ``Gamma @ x`` computed directly
            from the reduced likelihood.
    """

    native_posterior: GaussianPosterior
    projection: BocquetProjection
    reduced_posterior: GaussianPosterior


def restriction_for_prolongation(
    B: npt.ArrayLike,
    prolongation: npt.ArrayLike,
) -> np.ndarray:
    """Return the prior-weighted restriction that preserves a fixed prolongation.

    For a full-column-rank regional prolongation ``U``, the returned matrix is

    ``Gamma_U = (U.T @ B**-1 @ U)**-1 @ U.T @ B**-1``.

    Passing this restriction to :func:`build_bocquet_projection` makes the
    conditional prolongation equal ``U``. This is the appropriate convention
    when the forward model must retain fixed piecewise-regional amplitudes and
    summed regional design columns. It differs from preserving a caller-chosen
    literal arithmetic or area-weighted aggregate under a correlated prior.

    Args:
        B: Symmetric positive-definite native prior covariance with shape
            ``(M, M)``.
        prolongation: Full-column-rank native-to-regional amplitude map ``U``
            with shape ``(M, K)``.

    Returns:
        Read-only full-row-rank restriction with shape ``(K, M)``.

    Raises:
        ValueError: If inputs are non-finite, complex, incompatible, the prior
            covariance is not symmetric positive definite, or ``prolongation``
            does not have full column rank.
    """
    candidate = _finite_float_array(prolongation, name="prolongation")
    if candidate.ndim != 2:
        raise ValueError("prolongation must be two-dimensional with shape (native, projected).")
    if 0 in candidate.shape:
        raise ValueError("prolongation dimensions must be non-empty.")
    if candidate.shape[1] > candidate.shape[0] or np.linalg.matrix_rank(candidate) != candidate.shape[1]:
        raise ValueError("prolongation must have full column rank.")
    prior_covariance = _validated_spd_matrix(B, name="B", size=candidate.shape[0])
    prior_cholesky = _positive_definite_cholesky(prior_covariance, name="B")
    precision_weighted_prolongation = _cholesky_solve(prior_cholesky, candidate)
    projected_precision = _symmetrize(candidate.T @ precision_weighted_prolongation)
    projected_precision_cholesky = _positive_definite_cholesky(
        projected_precision,
        name="projected precision",
    )
    restriction = _cholesky_solve(
        projected_precision_cholesky,
        precision_weighted_prolongation.T,
    )
    restriction.setflags(write=False)
    return restriction


def native_gaussian_posterior(
    H: npt.ArrayLike,
    B: npt.ArrayLike,
    R: npt.ArrayLike,
    mu: npt.ArrayLike,
    y: npt.ArrayLike,
) -> GaussianPosterior:
    """Condition a dense native Gaussian model in observation space.

    The posterior mean uses the observation-space covariance
    ``S = H B H.T + R``.  The covariance is reconstructed from a
    prior-whitened information factor using ``R`` solves, avoiding an unstable
    subtraction from ``B`` under strong information.

    Args:
        H: Observation design with shape ``(N, M)``.
        B: Symmetric positive-definite native prior covariance with shape
            ``(M, M)``.
        R: Symmetric positive-definite observation covariance with shape
            ``(N, N)``.  Non-diagonal covariances are supported.
        mu: Native prior mean with shape ``(M,)``.
        y: Observation vector with shape ``(N,)``.

    Returns:
        Read-only native posterior moments and innovation terms.

    Raises:
        ValueError: If an input is non-finite, complex, dimensionally
            incompatible, non-symmetric, or a covariance is not positive
            definite.
    """
    design, prior_covariance, observation_covariance, prior_mean = _validated_model(H, B, R, mu)
    observations = _validated_vector(y, name="y", length=design.shape[0])

    innovation = observations - design @ prior_mean
    cross_covariance = prior_covariance @ design.T
    innovation_covariance = _symmetrize(design @ cross_covariance + observation_covariance)
    innovation_cholesky = _positive_definite_cholesky(
        innovation_covariance,
        name="innovation covariance",
    )
    posterior_mean = prior_mean + cross_covariance @ _cholesky_solve(
        innovation_cholesky,
        innovation,
    )
    posterior_covariance = _stable_posterior_covariance(
        prior_covariance,
        design,
        observation_covariance,
        name="native posterior covariance",
    )
    return GaussianPosterior(
        mean=posterior_mean,
        covariance=posterior_covariance,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
    )


def build_bocquet_projection(
    H: npt.ArrayLike,
    B: npt.ArrayLike,
    R: npt.ArrayLike,
    restriction: npt.ArrayLike,
    mu: npt.ArrayLike,
) -> BocquetProjection:
    """Construct the exact conditional reduction for ``alpha = Gamma @ x``.

    Args:
        H: Observation design with shape ``(N, M)``.
        B: Symmetric positive-definite native prior covariance with shape
            ``(M, M)``.
        R: Symmetric positive-definite base observation covariance with shape
            ``(N, N)``.
        restriction: Full-row-rank restriction ``Gamma`` with shape ``(K, M)``.
            Rows may overlap and need not describe disjoint grid regions.
        mu: Native prior mean with shape ``(M,)``.

    Returns:
        Read-only projected prior, conditional prolongation, unresolved
        covariance, and exact reduced observation model.

    Raises:
        ValueError: If an input is non-finite, complex, dimensionally
            incompatible, non-symmetric, not positive definite, or the
            restriction does not have full row rank.
        ArithmeticError: If roundoff produces a materially non-positive
            unresolved covariance or breaks innovation-covariance closure.
    """
    design, prior_covariance, observation_covariance, prior_mean = _validated_model(H, B, R, mu)
    gamma = _validated_restriction(restriction, native_dimension=design.shape[1])

    projected_prior_mean = gamma @ prior_mean
    projected_prior_covariance = _symmetrize(gamma @ prior_covariance @ gamma.T)
    projected_cholesky = _positive_definite_cholesky(
        projected_prior_covariance,
        name="projected prior covariance",
    )
    conditional_prolongation = _cholesky_solve(
        projected_cholesky,
        gamma @ prior_covariance,
    ).T
    prior_cholesky = _positive_definite_cholesky(prior_covariance, name="B")
    whitened_restriction = gamma @ prior_cholesky
    unresolved_projector = _symmetrize(
        np.eye(design.shape[1])
        - whitened_restriction.T @ _cholesky_solve(projected_cholesky, whitened_restriction)
    )
    unresolved_factor = prior_cholesky @ unresolved_projector
    unresolved_covariance = _symmetrize(unresolved_factor @ unresolved_factor.T)
    _validate_positive_semidefinite(unresolved_covariance, name="unresolved covariance")

    reduced_design = design @ conditional_prolongation
    observation_prior_mean = design @ prior_mean
    reduced_likelihood_offset = observation_prior_mean - reduced_design @ projected_prior_mean
    resolved_signal_covariance = _symmetrize(reduced_design @ projected_prior_covariance @ reduced_design.T)
    native_signal_covariance = _symmetrize(design @ prior_covariance @ design.T)
    effective_observation_covariance = _symmetrize(
        observation_covariance + design @ unresolved_covariance @ design.T
    )
    _positive_definite_cholesky(
        effective_observation_covariance,
        name="effective observation covariance",
    )
    innovation_covariance = _symmetrize(observation_covariance + native_signal_covariance)
    _positive_definite_cholesky(innovation_covariance, name="innovation covariance")
    if not np.allclose(
        effective_observation_covariance + resolved_signal_covariance,
        innovation_covariance,
        rtol=2e-10,
        atol=_matrix_tolerance(innovation_covariance),
    ):
        raise ArithmeticError("projected and native innovation covariances do not agree.")

    return BocquetProjection(
        restriction=gamma,
        native_prior_mean=prior_mean,
        observation_prior_mean=observation_prior_mean,
        reduced_likelihood_offset=reduced_likelihood_offset,
        projected_prior_mean=projected_prior_mean,
        projected_prior_covariance=projected_prior_covariance,
        conditional_prolongation=conditional_prolongation,
        unresolved_covariance=unresolved_covariance,
        reduced_design=reduced_design,
        base_observation_covariance=observation_covariance,
        effective_observation_covariance=effective_observation_covariance,
        resolved_signal_covariance=resolved_signal_covariance,
        native_signal_covariance=native_signal_covariance,
        innovation_covariance=innovation_covariance,
    )


def reduced_gaussian_posterior(
    projection: BocquetProjection,
    y: npt.ArrayLike,
) -> GaussianPosterior:
    """Condition the exact reduced model directly.

    The posterior mean retains the observation-space update.  The covariance
    uses the effective observation covariance to form a prior-whitened
    information factor, avoiding cancellation in a covariance downdate.

    Args:
        projection: Exact reduction returned by
            :func:`build_bocquet_projection`.
        y: Observation vector with shape ``(N,)``.

    Returns:
        Read-only posterior moments for ``alpha = Gamma @ x``.  These moments
        equal the corresponding restriction of the native posterior, up to
        floating-point error.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If a stored innovation covariance is not positive definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
        ValueError: If ``y`` is non-finite, complex, or has the wrong shape.
    """
    if not isinstance(projection, BocquetProjection):
        raise TypeError("projection must be a BocquetProjection.")
    observations = _validated_vector(
        y,
        name="y",
        length=projection.observation_prior_mean.size,
    )
    reduced_prior_prediction = (
        projection.reduced_likelihood_offset + projection.reduced_design @ projection.projected_prior_mean
    )
    innovation = observations - reduced_prior_prediction
    cross_covariance = projection.projected_prior_covariance @ projection.reduced_design.T
    innovation_cholesky = _positive_definite_cholesky(
        projection.innovation_covariance,
        name="innovation covariance",
    )
    posterior_mean = projection.projected_prior_mean + cross_covariance @ _cholesky_solve(
        innovation_cholesky,
        innovation,
    )
    posterior_covariance = _stable_posterior_covariance(
        projection.projected_prior_covariance,
        projection.reduced_design,
        projection.effective_observation_covariance,
        name="reduced posterior covariance",
    )
    return GaussianPosterior(
        mean=posterior_mean,
        covariance=posterior_covariance,
        innovation=innovation,
        innovation_covariance=projection.innovation_covariance,
    )


def gaussian_projection_oracle(
    H: npt.ArrayLike,
    B: npt.ArrayLike,
    R: npt.ArrayLike,
    restriction: npt.ArrayLike,
    mu: npt.ArrayLike,
    y: npt.ArrayLike,
) -> GaussianProjectionAnalysis:
    """Run native and exact reduced analyses for one dense Gaussian problem.

    Args:
        H: Observation design with shape ``(N, M)``.
        B: Symmetric positive-definite native prior covariance.
        R: Symmetric positive-definite base observation covariance.
        restriction: Full-row-rank restriction with shape ``(K, M)``.
        mu: Native prior mean with shape ``(M,)``.
        y: Observation vector with shape ``(N,)``.

    Returns:
        A verify-friendly bundle containing native posterior moments, every
        exact reduction term, and directly fitted reduced posterior moments.

    Raises:
        ValueError: If any model input is invalid or incompatible.
        ArithmeticError: If generated covariance identities fail beyond
            floating-point tolerance.
    """
    native_posterior = native_gaussian_posterior(H, B, R, mu, y)
    projection = build_bocquet_projection(H, B, R, restriction, mu)
    reduced_posterior = reduced_gaussian_posterior(projection, y)
    return GaussianProjectionAnalysis(
        native_posterior=native_posterior,
        projection=projection,
        reduced_posterior=reduced_posterior,
    )


def projected_dfs(projection: BocquetProjection) -> float:
    """Return projected degrees of freedom for signal without a half factor.

    This is ``trace(S^-1 C_projected)``, where ``S`` is the invariant
    innovation covariance and ``C_projected`` is the resolved observation
    signal covariance.  The convention matches Equation 38 in the project
    report and does **not** include a factor of ``1/2``.

    Args:
        projection: Exact Bocquet reduction to score.

    Returns:
        Non-negative projected DFS.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If the stored base observation covariance is not positive
            definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
    """
    _require_projection(projection)
    cholesky = _positive_definite_cholesky(
        projection.innovation_covariance,
        name="innovation covariance",
    )
    return _checked_nonnegative(
        float(np.trace(_cholesky_solve(cholesky, projection.resolved_signal_covariance))),
        name="projected DFS",
    )


def projected_fisher_base_r(projection: BocquetProjection) -> float:
    """Return the base-``R`` projected Fisher objective without a half factor.

    The value is ``trace(R^-1 C_projected)``.  It uses the declared base
    observation covariance ``R``, not the projection-dependent effective
    covariance, and follows the Equation 36 convention with no factor of
    ``1/2``.

    Args:
        projection: Exact Bocquet reduction to score.

    Returns:
        Non-negative projected Fisher trace.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If the stored effective observation covariance is not
            positive definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
    """
    _require_projection(projection)
    cholesky = _positive_definite_cholesky(
        projection.base_observation_covariance,
        name="base observation covariance",
    )
    return _checked_nonnegative(
        float(np.trace(_cholesky_solve(cholesky, projection.resolved_signal_covariance))),
        name="projected Fisher objective",
    )


def projected_fisher_aggregation_aware(projection: BocquetProjection) -> float:
    """Return the aggregation-aware projected Fisher objective.

    The value is ``trace(R_Gamma^-1 C_projected)``, where
    ``R_Gamma = R + H B_c H.T``.  It is the Equation 37 counterpart to
    :func:`projected_fisher_base_r` and has no factor of ``1/2``.  Because
    ``R_Gamma`` depends on the restriction, this objective is generally not
    additive across projected coordinates.

    Args:
        projection: Exact Bocquet reduction to score.

    Returns:
        Non-negative aggregation-aware projected Fisher trace.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
    """
    _require_projection(projection)
    cholesky = _positive_definite_cholesky(
        projection.effective_observation_covariance,
        name="effective observation covariance",
    )
    return _checked_nonnegative(
        float(np.trace(_cholesky_solve(cholesky, projection.resolved_signal_covariance))),
        name="aggregation-aware projected Fisher objective",
    )


def equation_45_objective(projection: BocquetProjection, y: npt.ArrayLike) -> float:
    """Return the data-dependent Equation 45 mean-update score.

    The score is
    ``delta_alpha.T @ B_Gamma^-1 @ delta_alpha``, where ``delta_alpha`` is the
    reduced posterior-mean update.  This implementation deliberately omits
    ``1/2`` to match the report's Equation 45 ``J`` convention; the related
    Equation 41 relative entropy term would include ``1/2``.

    Args:
        projection: Exact Bocquet reduction to score.
        y: Observation vector used to calculate the posterior-mean update.

    Returns:
        Non-negative squared prior-normalized projected mean update.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If ``y`` is invalid or incompatible, or a required stored
            covariance is not positive definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
    """
    posterior = reduced_gaussian_posterior(projection, y)
    update = posterior.mean - projection.projected_prior_mean
    cholesky = _positive_definite_cholesky(
        projection.projected_prior_covariance,
        name="projected prior covariance",
    )
    value = float(update @ _cholesky_solve(cholesky, update))
    return _checked_nonnegative(value, name="Equation 45 objective")


def projected_bayesian_kl(projection: BocquetProjection, y: npt.ArrayLike) -> float:
    """Return projected posterior-to-prior Gaussian KL information gain.

    This is ``KL[N(m_a, P_a) || N(m_0, B_Gamma)]`` in projected coordinates.
    The standard Gaussian ``1/2`` factor is included.  Thus the function is the
    realized projected Bayesian information gain, not its prior-predictive
    expectation.  Its covariance contribution is evaluated eigenwise from the
    prior-whitened averaging kernel, using a stable weak-mode expansion of
    ``-log1p(-a) - a``.

    Args:
        projection: Exact Bocquet reduction to score.
        y: Observation vector defining the projected posterior.

    Returns:
        Non-negative KL divergence, including the conventional ``1/2`` factor.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If ``y`` is invalid or incompatible, or a required stored
            covariance is not positive definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
    """
    posterior = reduced_gaussian_posterior(projection, y)
    prior_covariance = projection.projected_prior_covariance
    prior_cholesky = _positive_definite_cholesky(
        prior_covariance,
        name="projected prior covariance",
    )
    update = posterior.mean - projection.projected_prior_mean
    mean_term = float(update @ _cholesky_solve(prior_cholesky, update))
    whitened_design = projection.reduced_design @ prior_cholesky
    innovation_cholesky = _positive_definite_cholesky(
        projection.innovation_covariance,
        name="innovation covariance",
    )
    averaging_kernel = _symmetrize(whitened_design.T @ _cholesky_solve(innovation_cholesky, whitened_design))
    covariance_term = _covariance_kl_from_averaging_kernel(averaging_kernel)
    value = covariance_term + 0.5 * mean_term
    return _checked_nonnegative(value, name="projected Bayesian KL")


def projected_bayesian_information_gain(
    projection: BocquetProjection,
    y: npt.ArrayLike,
) -> float:
    """Return the projected Bayesian information gain, including ``1/2``.

    This is a descriptive alias for :func:`projected_bayesian_kl` and uses the
    posterior-to-prior KL direction.

    Args:
        projection: Exact Bocquet reduction to score.
        y: Observation vector defining the projected posterior.

    Returns:
        The projected posterior-to-prior Gaussian KL divergence.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
        ValueError: If ``y`` is invalid or incompatible, or a required stored
            covariance is not positive definite.
        ArithmeticError: If numerical failure produces a non-finite or
            materially negative result.
    """
    return projected_bayesian_kl(projection, y)


def _validated_model(
    H: npt.ArrayLike,
    B: npt.ArrayLike,
    R: npt.ArrayLike,
    mu: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and copy the shared dense Gaussian model inputs.

    Args:
        H: Candidate observation design.
        B: Candidate native prior covariance.
        R: Candidate observation covariance.
        mu: Candidate native prior mean.

    Returns:
        Validated design, prior covariance, observation covariance, and prior
        mean arrays.

    Raises:
        ValueError: If dimensions, finiteness, symmetry, or positive
            definiteness requirements are violated.
    """
    design = _finite_float_array(H, name="H")
    if design.ndim != 2:
        raise ValueError("H must be two-dimensional with shape (observation, native state).")
    if 0 in design.shape:
        raise ValueError("H dimensions must be non-empty.")
    prior_covariance = _validated_spd_matrix(B, name="B", size=design.shape[1])
    observation_covariance = _validated_spd_matrix(R, name="R", size=design.shape[0])
    prior_mean = _validated_vector(mu, name="mu", length=design.shape[1])
    return design, prior_covariance, observation_covariance, prior_mean


def _validated_restriction(values: npt.ArrayLike, *, native_dimension: int) -> np.ndarray:
    """Validate and copy one non-empty full-row-rank restriction.

    Args:
        values: Candidate restriction array.
        native_dimension: Required number of native-state columns.

    Returns:
        A copied, finite, two-dimensional restriction.

    Raises:
        ValueError: If the restriction has invalid dimensions, values, or row
            rank.
    """
    restriction = _finite_float_array(values, name="restriction")
    if restriction.ndim != 2:
        raise ValueError("restriction must be two-dimensional with shape (projected, native).")
    if restriction.shape[0] == 0:
        raise ValueError("restriction must contain at least one row.")
    if restriction.shape[1] != native_dimension:
        raise ValueError(f"restriction must have {native_dimension} columns; got {restriction.shape[1]}.")
    if restriction.shape[0] > native_dimension or np.linalg.matrix_rank(restriction) != restriction.shape[0]:
        raise ValueError("restriction must have full row rank.")
    return restriction


def _validated_spd_matrix(values: npt.ArrayLike, *, name: str, size: int) -> np.ndarray:
    """Validate and copy a finite symmetric positive-definite matrix.

    Args:
        values: Candidate covariance-like matrix.
        name: Input name used in validation messages.
        size: Required matrix extent on both axes.

    Returns:
        A copied symmetric positive-definite matrix.

    Raises:
        ValueError: If values are invalid, dimensions do not match, the matrix
            is asymmetric, or Cholesky factorization fails.
    """
    matrix = _finite_float_array(values, name=name)
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}); got {matrix.shape}.")
    tolerance = _matrix_tolerance(matrix)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    matrix = _symmetrize(matrix)
    _positive_definite_cholesky(matrix, name=name)
    return matrix


def _validated_vector(values: npt.ArrayLike, *, name: str, length: int) -> np.ndarray:
    """Validate and copy a finite vector with one required length.

    Args:
        values: Candidate vector values.
        name: Input name used in validation messages.
        length: Required vector length.

    Returns:
        A copied finite floating-point vector.

    Raises:
        ValueError: If values are invalid or the vector has the wrong shape.
    """
    vector = _finite_float_array(values, name=name)
    if vector.ndim != 1 or vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},); got {vector.shape}.")
    return vector


def _finite_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a copied finite real float array.

    Args:
        values: Candidate numeric values.
        name: Input name used in validation messages.

    Returns:
        A copied real floating-point array.

    Raises:
        ValueError: If conversion fails or values are complex or non-finite.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _positive_definite_cholesky(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Return a Cholesky factor or raise a named validation error.

    Args:
        matrix: Symmetric matrix to factorize.
        name: Matrix name used in the validation message.

    Returns:
        Lower-triangular Cholesky factor.

    Raises:
        ValueError: If ``matrix`` is not positive definite.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc


def _stable_posterior_covariance(
    prior_covariance: np.ndarray,
    design: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return posterior covariance from a prior-whitened information factor.

    Args:
        prior_covariance: Positive-definite state prior covariance.
        design: Observation-by-state design matrix.
        observation_covariance: Positive-definite observation-error covariance.
        name: Posterior covariance name used in validation errors.

    Returns:
        Symmetric positive-definite posterior covariance.

    Raises:
        ValueError: If a required covariance or the resulting posterior
            covariance is not positive definite.
        ArithmeticError: If the factor calculation produces non-finite values.
    """
    prior_cholesky = _positive_definite_cholesky(prior_covariance, name="prior covariance")
    observation_cholesky = _positive_definite_cholesky(
        observation_covariance,
        name="observation covariance",
    )
    prior_whitened_design = design @ prior_cholesky
    error_whitened_design = np.linalg.solve(observation_cholesky, prior_whitened_design)
    information = _symmetrize(
        np.eye(prior_covariance.shape[0]) + error_whitened_design.T @ error_whitened_design
    )
    information_cholesky = _positive_definite_cholesky(
        information,
        name="prior-whitened posterior information",
    )
    posterior_factor = np.linalg.solve(information_cholesky, prior_cholesky.T).T
    posterior_covariance = _symmetrize(posterior_factor @ posterior_factor.T)
    if not np.all(np.isfinite(posterior_covariance)):
        raise ArithmeticError(f"{name} is not finite.")
    _positive_definite_cholesky(posterior_covariance, name=name)
    return posterior_covariance


def _cholesky_solve(cholesky: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a positive-definite system from its lower Cholesky factor.

    Args:
        cholesky: Lower-triangular Cholesky factor.
        right_hand_side: Vector or matrix right-hand side.

    Returns:
        Solution to the represented positive-definite linear system.
    """
    intermediate = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.T, intermediate)


def _covariance_kl_from_averaging_kernel(averaging_kernel: np.ndarray) -> float:
    """Return the covariance-only Gaussian KL from averaging-kernel modes.

    Args:
        averaging_kernel: Symmetric prior-whitened averaging kernel whose
            eigenvalues are theoretically in the half-open interval ``[0, 1)``.

    Returns:
        Covariance KL with the conventional factor ``1/2``.

    Raises:
        ArithmeticError: If the kernel has non-finite eigenvalues or a mode
            outside its positive-definite Gaussian range beyond roundoff.
    """
    eigenvalues = np.linalg.eigvalsh(averaging_kernel)
    if not np.all(np.isfinite(eigenvalues)):
        raise ArithmeticError("averaging-kernel eigenvalues must be finite.")
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(eigenvalues < -tolerance) or np.any(eigenvalues > 1.0 + tolerance):
        raise ArithmeticError("averaging-kernel eigenvalues must lie in [0, 1).")
    upper = np.nextafter(1.0, 0.0)
    eigenvalues = np.clip(eigenvalues, 0.0, upper)
    return 0.5 * float(np.sum(_negative_log1p_minus_identity(eigenvalues)))


def _negative_log1p_minus_identity(values: np.ndarray) -> np.ndarray:
    """Evaluate ``-log1p(-x) - x`` without losing weak quadratic terms."""
    result = np.empty_like(values)
    small = values < 1e-4
    small_values = values[small]
    result[small] = np.square(small_values) * (
        0.5 + small_values * (1.0 / 3.0 + small_values * (0.25 + small_values * (0.2 + small_values / 6.0)))
    )
    regular_values = values[~small]
    result[~small] = -np.log1p(-regular_values) - regular_values
    return result


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove floating-point antisymmetry from a covariance calculation."""
    return 0.5 * (matrix + matrix.T)


def _matrix_tolerance(matrix: np.ndarray) -> float:
    """Return a scale-aware absolute tolerance for dense matrix identities."""
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    return 1e-12 * scale


def _validate_positive_semidefinite(matrix: np.ndarray, *, name: str) -> None:
    """Reject a covariance with a materially negative eigenvalue.

    Args:
        matrix: Symmetric covariance candidate.
        name: Covariance name used in an error message.

    Raises:
        ArithmeticError: If an eigenvalue is negative beyond a scale-aware
            floating-point tolerance.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ArithmeticError(f"{name} is not positive semidefinite within numerical tolerance.")


def _checked_nonnegative(value: float, *, name: str) -> float:
    """Return a non-negative objective, clipping negative scalar roundoff.

    Args:
        value: Computed scalar objective.
        name: Objective name used in an error message.

    Returns:
        ``value`` with negligible negative roundoff replaced by zero.

    Raises:
        ArithmeticError: If ``value`` is non-finite or materially negative.
    """
    if not np.isfinite(value):
        raise ArithmeticError(f"{name} is not finite.")
    tolerance = 1e-12 * max(1.0, abs(value))
    if value < -tolerance:
        raise ArithmeticError(f"{name} is materially negative.")
    return max(0.0, value)


def _require_projection(projection: BocquetProjection) -> None:
    """Raise if an objective receives the wrong projection type.

    Args:
        projection: Candidate projection object.

    Raises:
        TypeError: If ``projection`` is not a :class:`BocquetProjection`.
    """
    if not isinstance(projection, BocquetProjection):
        raise TypeError("projection must be a BocquetProjection.")


def _freeze_array_fields(instance: object, field_names: tuple[str, ...]) -> None:
    """Copy named NumPy array fields and make each copy read-only.

    Args:
        instance: Frozen dataclass instance whose fields are replaced.
        field_names: Names of array fields to copy and freeze.

    Side Effects:
        Replaces each named field through ``object.__setattr__`` during frozen
        dataclass initialization.
    """
    for field_name in field_names:
        array = np.array(getattr(instance, field_name), dtype=float, copy=True)
        array.setflags(write=False)
        object.__setattr__(instance, field_name, array)
