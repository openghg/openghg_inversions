"""Validate observation-error inputs and prepare aggregation-error covariances.

Shared checks cover observations, reported errors and minimum errors.

Aggregation error is fixed input data, separate from measurement error and
the inferred RHIME model-error term.  Prepared inversion inputs may represent
it exactly as a dense covariance, efficiently as a low-rank-plus-diagonal
covariance. Marginal standard deviations may accompany either representation.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, TypeAlias
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import xarray as xr

from openghg_inversions._linalg import cholesky_with_condition
from openghg_inversions.array_ops import materialize_together, validate_covariance_coordinates

#: ``"auto"`` selects the supplied dense or low-rank covariance, or ``"none"``
#: when there is no aggregation-error input. It raises if both covariances
#: are present or only marginal standard deviations are supplied.
AggregationErrorMode: TypeAlias = Literal["auto", "none", "dense", "low_rank"]

AGGREGATION_ERROR_SD = "aggregation_error_sd"
AGGREGATION_ERROR_COVARIANCE = "aggregation_error_covariance"
LOW_RANK_FACTOR = "low_rank_factor"
DIAGONAL_RESIDUAL_VARIANCE = "diagonal_residual_variance"
OBSERVATION_ERROR_INPUT_NAMES = ("mf", "mf_error", "min_error")


@dataclass(frozen=True)
class AggregationError:
    """Validated aggregation-error representation selected for a likelihood.

    Model builders trust this value as already validated. Externally supplied
    representations should be constructed through
    :func:`resolve_aggregation_error`, which selects and validates a
    coherent-reduction representation. Package preparation helpers may also
    construct it directly from values they have just produced. Other direct
    construction is an expert seam, and the caller then owns the coherence of
    the mode, payload, marginal variance, coordinates, and numerical
    covariance properties.

    Args:
        mode: Concrete covariance representation.
        marginal_variance: Observation-aligned covariance diagonal.
        covariance: Optional dense covariance matrix.
        factor: Optional low-rank covariance factor.
        diagonal_variance: Optional independent residual variance for a
            low-rank representation.
    """

    mode: Literal["none", "dense", "low_rank"]
    marginal_variance: np.ndarray
    covariance: xr.DataArray | None = None
    factor: xr.DataArray | None = None
    diagonal_variance: xr.DataArray | None = None


@dataclass(frozen=True)
class LowRankAggregationErrorApproximation:
    """A prepared LRPD representation and its approximation diagnostics.

    Attributes:
        aggregation_error: Labelled low-rank factor and non-negative diagonal
            residual accepted by the shared likelihood components.
        diagnostics: JSON-safe mapping containing the method, requested rank,
            actual retained factor width, retained spectral fraction, relative
            Frobenius error, diagonal-preservation error, tolerances, and
            diagonal-tail clipping diagnostics. These values describe the
            approximation but do not certify a rank for a particular
            likelihood.
    """

    aggregation_error: AggregationError
    diagnostics: dict[str, str | int | float]


@dataclass(frozen=True)
class _SpectralLrpd:
    """Numerical outputs from diagonal-preserving spectral truncation.

    Attributes:
        factor: Retained spectral factor with shape
            ``(nmeasure, actual_rank)``.
        diagonal: Non-negative diagonal covariance tail with shape
            ``(nmeasure,)``.
        retained_spectrum: Sum of the retained eigenvalues.
        positive_spectrum: Sum of all positive source eigenvalues.
        source_frobenius_norm: Frobenius norm of the source covariance.
        reconstruction_error: Frobenius norm of the difference between the
            source covariance and the factor-plus-diagonal approximation.
        diagonal_tail_clipped_count: Number of small negative diagonal-tail
            entries clipped to zero.
        diagonal_tail_max_clipped_magnitude: Largest magnitude clipped from
            the diagonal tail.
    """

    factor: np.ndarray
    diagonal: np.ndarray
    retained_spectrum: float
    positive_spectrum: float
    source_frobenius_norm: float
    reconstruction_error: float
    diagonal_tail_clipped_count: int
    diagonal_tail_max_clipped_magnitude: float


def prepare_low_rank_aggregation_error(
    covariance: xr.DataArray,
    *,
    rank: int,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> LowRankAggregationErrorApproximation:
    """Approximate a labelled covariance by a low-rank-plus-diagonal form.

    This is an eager numerical boundary. Sparse payloads are densified, Dask
    payloads are materialized, and the complete dense covariance receives one
    eigendecomposition. The leading eigenmodes form the low-rank factor and
    the discarded marginal variance is retained on the diagonal, so the source
    covariance diagonal is preserved up to accepted roundoff and tail
    clipping. The factor width is the lesser of ``rank`` and the numerical
    positive rank, and may be zero when the covariance has no positive
    numerical modes. The factor has square-root covariance units and the
    diagonal tail has covariance units, but this helper does not attach unit
    attributes to the returned arrays.

    Args:
        covariance: Symmetric positive-semidefinite covariance matrix.
        rank: Maximum number of leading eigenmodes to retain.
        output_dim: Observation dimension on the covariance rows.
        covariance_dim: Repeated observation dimension on the columns.

    Returns:
        Constructed low-rank aggregation error and JSON-safe approximation
        diagnostics.

    Raises:
        ValueError: If dimensions, coordinates, rank, or numerical covariance
            properties are invalid.
    """
    validate_covariance_coordinates(
        covariance,
        dim=output_dim,
        covariance_dim=covariance_dim,
    )
    nmeasure = covariance.sizes[output_dim]
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= nmeasure:
        raise ValueError(f"`rank` must be an integer from 1 to {nmeasure}; got {rank!r}.")

    (covariance,) = materialize_together(covariance, purpose="aggregation-error covariance for factorization")
    values = np.asarray(
        _numeric_finite("covariance", covariance, owner="Low-rank approximation input"),
        dtype=float,
    )
    roundoff_tolerance = _validate_dense_covariance_symmetry(
        values,
        owner="Low-rank approximation input covariance",
    )
    return _factorize_aggregation_error(
        covariance, values, rank=rank, roundoff_tolerance=roundoff_tolerance, output_dim=output_dim
    )


def _factorize_aggregation_error(
    covariance: xr.DataArray,
    values: np.ndarray,
    *,
    rank: int,
    roundoff_tolerance: float,
    output_dim: str,
) -> LowRankAggregationErrorApproximation:
    """Factorize a validated aggregation-error covariance into a labelled factor and diagonal tail.

    Args:
        covariance: Materialized labelled covariance supplying the row labels.
        values: Its finite symmetric values.
        rank: Maximum number of leading eigenmodes to retain.
        roundoff_tolerance: Scale-relative tolerance from the symmetry check.
        output_dim: Observation dimension on the covariance rows.

    Returns:
        Constructed low-rank aggregation error and JSON-safe diagnostics.
    """
    approximation = _diagonal_preserving_spectral_lrpd(
        values,
        rank=rank,
        roundoff_tolerance=roundoff_tolerance,
        owner="Low-rank approximation input covariance",
    )

    labels = covariance.coords[output_dim]
    factor = xr.DataArray(
        approximation.factor,
        dims=(output_dim, "agg_rank"),
        coords={output_dim: labels, "agg_rank": np.arange(approximation.factor.shape[1])},
        name=LOW_RANK_FACTOR,
    )
    diagonal = xr.DataArray(
        approximation.diagonal,
        dims=(output_dim,),
        coords={output_dim: labels},
        name=DIAGONAL_RESIDUAL_VARIANCE,
    )
    marginal_variance = (
        np.einsum("ij,ij->i", approximation.factor, approximation.factor) + approximation.diagonal
    )
    aggregation_error = AggregationError(
        mode="low_rank",
        marginal_variance=marginal_variance,
        factor=factor,
        diagonal_variance=diagonal,
    )

    retained_fraction = (
        approximation.retained_spectrum / approximation.positive_spectrum
        if approximation.positive_spectrum
        else 1.0
    )
    return LowRankAggregationErrorApproximation(
        aggregation_error=aggregation_error,
        diagnostics={
            "method": "descending_eigendecomposition_with_diagonal_tail",
            "requested_rank": rank,
            "actual_rank": approximation.factor.shape[1],
            "retained_positive_spectral_fraction": retained_fraction,
            "relative_frobenius_reconstruction_error": (
                approximation.reconstruction_error / approximation.source_frobenius_norm
                if approximation.source_frobenius_norm
                else 0.0
            ),
            "diagonal_preservation_error": float(np.max(np.abs(np.diag(values) - marginal_variance))),
            "roundoff_tolerance": roundoff_tolerance,
            "symmetry_relative_tolerance": 1e-10,
            "psd_absolute_tolerance": roundoff_tolerance,
            "diagonal_tail_clipped_count": approximation.diagonal_tail_clipped_count,
            "diagonal_tail_max_clipped_magnitude": approximation.diagonal_tail_max_clipped_magnitude,
        },
    )


def aggregation_error_as_low_rank(
    aggregation_error: AggregationError,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize fixed aggregation covariance as factor-plus-diagonal data.

    Dense inputs use every eigenmode above the scale-relative numerical
    tolerance and retain the remaining marginal variance on the diagonal.
    This is exact when every positive mode exceeds that tolerance, but it is
    not the low-rank performance path.

    Args:
        aggregation_error: Validated fixed aggregation-error representation.

    Returns:
        A factor with shape ``(nmeasure, actual_rank)`` and non-negative
        diagonal variance with shape ``(nmeasure,)``. The factor may have zero
        columns.
    """
    nmeasure = aggregation_error.marginal_variance.size
    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        covariance = np.asarray(aggregation_error.covariance.values, dtype=float)
        approximation = _diagonal_preserving_spectral_lrpd(
            covariance,
            rank=None,
            roundoff_tolerance=_covariance_roundoff_tolerance(covariance),
            owner="Dense aggregation-error covariance",
        )
        return approximation.factor, approximation.diagonal
    if aggregation_error.mode == "low_rank":
        assert aggregation_error.factor is not None and aggregation_error.diagonal_variance is not None
        return (
            np.asarray(aggregation_error.factor.values, dtype=float),
            np.asarray(aggregation_error.diagonal_variance.values, dtype=float),
        )
    return np.empty((nmeasure, 0)), np.zeros(nmeasure)


@dataclass(frozen=True)
class WhitenedRankRule:
    r"""Choose an LRPD aggregation-error rank from the R-whitened spectrum.

    The rule solves :math:`A u_i = \mu_i R_0 u_i` for the aggregation
    covariance :math:`A` against a reference mismatch covariance :math:`R_0`
    supplied separately (in CO2 preparation, ``reference_covariance``).
    :math:`\mu_i` is the variance of mode :math:`i` of :math:`A` in units of
    reference noise. Dropping the modes after the leading :math:`k` from the
    whitened representation costs

    .. math::

        \mathrm{KL}\big(N(0, R_0 + A)\,\|\,N(0, R_0 + A_k)\big)
        = \tfrac12 \sum_{i>k} \big[\mu_i - \log(1 + \mu_i)\big].

    The rank is the larger of the count of :math:`\mu_i` above
    ``mu_threshold`` and the smallest :math:`k` whose dropped-mode KL per
    observation is at most ``kl_budget_per_observation``; either criterion
    may be disabled with ``None``. The stored factor is still the leading
    eigenmodes of :math:`A` with the discarded marginal variance on the
    diagonal; only the number of modes comes from the whitened spectrum.

    The count depends strongly on :math:`R_0`: larger reference variance
    whitens more of :math:`A` away. The rule is a screening heuristic, not a
    bound on posterior error. It was checked only on four Verification Games
    CO2 cases (January and July 2021, two prior bases, Gaussian analytic
    posteriors), where a rank at least the :math:`\mu > 0.1` count, against
    either reported error plus 1 ppm independent mismatch or prior-median
    within-site OU mismatch, gave country totals within 0.5 Tg C/month of
    exact dense :math:`A`. Check the chosen rank for the intended likelihood.

    Args:
        mu_threshold: Positive whitened-eigenvalue threshold for the count
            criterion, or ``None`` to disable it.
        kl_budget_per_observation: Positive dropped-mode KL budget in nats per
            observation, or ``None`` to disable it. This spectral KL ignores
            the diagonal tail, so it is not the KL of the stored
            representation. In the four cases above, at the prior-median
            reference, it was about 3-10 times smaller than the stored
            representation's KL: a budget of ``1e-3`` kept far too few modes
            (worst country error 0.6-1.3 Tg C/month), while the
            ``mu > 0.1`` count corresponded to about ``2e-5``-``1.2e-4``.
            Calibrate any budget for the intended reference.
        rank: Optional explicit rank for benchmarking. The spectrum and rule
            rank are still computed and recorded, and a ``UserWarning`` is
            issued when ``rank`` is below the rule rank.

    Raises:
        ValueError: If a value is invalid or no criterion or rank is set.
    """

    mu_threshold: float | None = 0.1
    kl_budget_per_observation: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        """Validate rule values at construction."""
        for name in ("mu_threshold", "kl_budget_per_observation"):
            value = getattr(self, name)
            if value is not None and not _is_positive_real(value):
                raise ValueError(f"WhitenedRankRule {name} must be a positive finite number or None.")
        if self.rank is not None and (
            isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1
        ):
            raise ValueError("WhitenedRankRule rank must be a positive integer or None.")
        if self.rank is None and self.mu_threshold is None and self.kl_budget_per_observation is None:
            raise ValueError(
                "WhitenedRankRule needs mu_threshold, kl_budget_per_observation or an explicit rank."
            )


def whitened_eigenvalue_kl(eigenvalues: np.ndarray) -> float:
    r"""Return the Gaussian KL carried by whitened covariance eigenvalues.

    For eigenvalues :math:`\lambda_i > -1` of a covariance difference
    whitened by the approximate total covariance, or the generalized
    eigenvalues of dropped modes, the KL divergence is
    :math:`\tfrac12 \sum_i (\lambda_i - \log(1 + \lambda_i))`. It is
    non-negative and zero only when every eigenvalue is zero.

    Args:
        eigenvalues: Whitened eigenvalues, each greater than ``-1``.

    Returns:
        KL divergence in nats.

    Raises:
        ValueError: If any eigenvalue is not finite or is at most ``-1``.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= -1.0):
        raise ValueError("Whitened eigenvalues must be finite and greater than -1.")
    return 0.5 * float(np.sum(values - np.log1p(values)))


def select_whitened_rank(
    eigenvalues: np.ndarray,
    rule: WhitenedRankRule,
) -> tuple[int, dict[str, Any]]:
    """Apply a rank rule to whitened eigenvalues.

    Args:
        eigenvalues: Non-empty, finite, non-negative whitened eigenvalues in
            descending order.
        rule: Rank criteria and optional explicit override.

    Returns:
        The chosen rank, at least one and at most the eigenvalue count, and a
        JSON-safe record of the criteria, counts and dropped-mode KL. Issues a
        ``UserWarning`` when an explicit rank is below the rule rank.

    Raises:
        ValueError: If ``eigenvalues`` is not a non-empty, finite,
            non-negative vector in descending order.
    """
    mu = np.asarray(eigenvalues, dtype=float)
    if mu.ndim != 1 or mu.size == 0 or not np.isfinite(mu).all() or np.any(mu < 0.0):
        raise ValueError("Whitened eigenvalues must be a non-empty vector of finite non-negative values.")
    if np.any(np.diff(mu) > 0.0):
        raise ValueError("Whitened eigenvalues must be in descending order.")
    n = mu.size
    # tail_kl[k] is the KL of dropping modes k, k + 1, ..., n - 1 (zero-based).
    terms = 0.5 * (mu - np.log1p(mu))
    tail_kl = np.append(np.cumsum(terms[::-1])[::-1], 0.0)
    count = None if rule.mu_threshold is None else int(np.count_nonzero(mu > rule.mu_threshold))
    kl_rank = None
    if rule.kl_budget_per_observation is not None:
        kl_rank = int(np.argmax(tail_kl <= rule.kl_budget_per_observation * n))
    criteria = [value for value in (count, kl_rank) if value is not None]
    rule_rank = min(max(*criteria, 1), n) if criteria else None
    if rule.rank is not None:
        chosen = min(rule.rank, n)
    else:
        assert rule_rank is not None  # the rule requires a criterion without an explicit rank
        chosen = rule_rank
    if rule.rank is not None and rule_rank is not None and rule.rank < rule_rank:
        warnings.warn(
            f"Explicit aggregation_error rank {rule.rank} is below the whitened-spectrum rule rank "
            f"{rule_rank} (mu > {rule.mu_threshold}: {count}; KL-budget rank: {kl_rank}). "
            f"Dropped whitened modes carry {tail_kl[chosen]:.4g} nats "
            f"({tail_kl[chosen] / n:.3g} per observation) at the reference covariance.",
            UserWarning,
            stacklevel=2,
        )
    return chosen, {
        "method": "r_whitened_generalized_eigenvalues",
        "n_observation": n,
        "mu_threshold": rule.mu_threshold,
        "count_mu_above_threshold": count,
        "kl_budget_per_observation": rule.kl_budget_per_observation,
        "kl_budget_rank": kl_rank,
        "rule_rank": rule_rank,
        "override_rank": rule.rank,
        "chosen_rank": chosen,
        "largest_mu": float(mu[0]),
        "first_dropped_mu": float(mu[chosen]) if chosen < n else 0.0,
        "dropped_mode_kl": float(tail_kl[chosen]),
        "dropped_mode_kl_per_observation": float(tail_kl[chosen] / n),
        "dropped_mode_kl_scope": (
            "KL(N(0, R0 + A) || N(0, R0 + A_k)) for the leading k R0-whitened modes, without the "
            "diagonal tail; the stored factor uses the leading eigenmodes of A with a diagonal tail"
        ),
    }


REFERENCE_MIN_RECIPROCAL_CONDITION = float(np.sqrt(np.finfo(np.float64).eps))


def _prepare_whitened_low_rank_aggregation_error(
    aggregation_error_covariance: xr.DataArray,
    reference_covariance: np.ndarray,
    rule: WhitenedRankRule,
    *,
    reference_description: Mapping[str, Any],
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> tuple[LowRankAggregationErrorApproximation, dict[str, Any]]:
    r"""Choose the LRPD rank from the whitened spectrum and factorize once.

    Solves :math:`A u = \mu R_0 u` by forming :math:`L^{-1} A L^{-T}` with the
    Cholesky factor :math:`L` of the dense reference :math:`R_0`, then applies
    ``rule`` and factorizes :math:`A` at the chosen rank. The caller owns the
    labels and units of :math:`R_0`: it must already be ordered like the rows
    of :math:`A` and expressed in the same units squared.

    Validation has one owner per property. :math:`A` is checked once for
    finite and symmetric values at its own scale; positive semidefiniteness
    and roundoff of :math:`A` are owned by the LRPD factorization, so negative
    whitened eigenvalues are clipped to zero for rank scoring. :math:`R_0` is
    checked here to be finite, symmetric and numerically positive definite,
    using its Cholesky factor and the LAPACK reciprocal condition estimate.

    Args:
        aggregation_error_covariance: Labelled covariance :math:`A`.
        reference_covariance: Dense reference covariance :math:`R_0` values.
        rule: Rank criteria and optional explicit rank.
        reference_description: JSON-safe description of :math:`R_0` from its
            owner, stored with the rank diagnostics.
        output_dim: Observation dimension on the covariance rows.
        covariance_dim: Repeated observation dimension on the columns.

    Returns:
        The low-rank approximation at the chosen rank, as from
        :func:`prepare_low_rank_aggregation_error`, and a JSON-safe record of
        the reference description, criteria, counts, the spectral
        dropped-mode KL and the KL of the stored representation at
        :math:`R_0`. The stored KL is ``None``, with the reason under
        ``stored_representation_kl_unavailable``, when a total cannot be
        Cholesky factorized, for example because of roundoff.

    Raises:
        ValueError: If coordinates or values are invalid, :math:`R_0` is not
            numerically positive definite, or the description is not
            JSON-safe.
    """
    try:
        description = json.loads(json.dumps(dict(reference_description), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("The reference covariance description must be JSON-safe.") from exc
    validate_covariance_coordinates(
        aggregation_error_covariance, dim=output_dim, covariance_dim=covariance_dim
    )
    (covariance,) = materialize_together(
        aggregation_error_covariance,
        purpose="aggregation-error covariance for rank selection and factorization",
    )
    values = np.asarray(
        _numeric_finite("covariance", covariance, owner="Low-rank approximation input"), dtype=float
    )
    tolerance = _validate_dense_covariance_symmetry(values, owner="Low-rank approximation input covariance")
    cholesky = _reference_cholesky(reference_covariance, values.shape[0])
    eigenvalues = _whitened_spectrum(values, cholesky)
    del cholesky
    rank, selection = select_whitened_rank(eigenvalues, rule)
    approximation = _factorize_aggregation_error(
        covariance, values, rank=rank, roundoff_tolerance=tolerance, output_dim=output_dim
    )
    stored = approximation.aggregation_error
    assert stored.factor is not None and stored.diagonal_variance is not None
    n = values.shape[0]
    try:
        stored_kl: float | None = _stored_representation_kl(
            values,
            np.asarray(reference_covariance, dtype=float),
            np.asarray(stored.factor.values, dtype=float),
            np.asarray(stored.diagonal_variance.values, dtype=float),
        )
    except np.linalg.LinAlgError:
        stored_kl = None
    kl_record: dict[str, Any] = {
        "stored_representation_kl": stored_kl,
        "stored_representation_kl_per_observation": None if stored_kl is None else stored_kl / n,
    }
    if stored_kl is None:
        kl_record["stored_representation_kl_unavailable"] = (
            "R0 + A or R0 + F F^T + diag(d) could not be Cholesky factorized. A possible cause is roundoff: "
            "negative eigenvalues of A accepted within the LRPD tolerance, or R0 too small to change the "
            "total in floating point"
        )
    return approximation, {
        "reference_covariance": description,
        **selection,
        **kl_record,
        "stored_representation_kl_scope": (
            "KL(N(0, R0 + A) || N(0, R0 + F F^T + diag(d))) for the stored factor F (leading eigenmodes of A) "
            "and diagonal tail d, at the reference covariance R0"
        ),
    }


def _stored_representation_kl(
    values: np.ndarray,
    reference_covariance: np.ndarray,
    factor: np.ndarray,
    diagonal: np.ndarray,
) -> float:
    r"""Return the KL divergence of the stored LRPD representation at the reference.

    Computes :math:`\mathrm{KL}(N(0, R_0 + A)\,\|\,N(0, R_0 + A_s))` with
    :math:`A_s = F F^\top + \mathrm{diag}(d)`, as
    :math:`\tfrac12[\mathrm{tr}((R_0 + A_s)^{-1}(A - A_s)) + \log|R_0 + A_s| - \log|R_0 + A|]`.
    Both totals are positive definite when :math:`A` is positive
    semidefinite. :math:`A` is accepted with negative eigenvalues down to the
    LRPD roundoff tolerance, which can exceed the smallest eigenvalue of
    :math:`R_0`, and an :math:`R_0` far below the scale of :math:`A` can
    round away when added. The caller then records the KL as unavailable. Cost is two
    dense Cholesky factorizations and one dense solve, with three extra
    ``n x n`` arrays.

    Raises:
        numpy.linalg.LinAlgError: If either total is not positive definite.
    """
    stored = factor @ factor.T
    stored[np.diag_indices_from(stored)] += diagonal
    difference = values - stored
    stored += reference_covariance
    exact = stored + difference
    exact_factor = cho_factor(exact, lower=True, overwrite_a=True, check_finite=False)
    logdet_exact = 2.0 * float(np.log(np.diag(exact_factor[0])).sum())
    del exact, exact_factor
    stored_factor = cho_factor(stored, lower=True, overwrite_a=True, check_finite=False)
    logdet_stored = 2.0 * float(np.log(np.diag(stored_factor[0])).sum())
    trace = float(np.trace(cho_solve(stored_factor, difference, overwrite_b=True, check_finite=False)))
    return max(0.5 * (trace + logdet_stored - logdet_exact), 0.0)


def _reference_cholesky(reference_covariance: np.ndarray, n_observation: int) -> np.ndarray:
    """Check a dense reference covariance and return its lower Cholesky factor.

    A successful Cholesky factorization alone does not show that the matrix is
    numerically nonsingular, so the reciprocal condition estimate from the
    same factor is also required to exceed
    :data:`REFERENCE_MIN_RECIPROCAL_CONDITION` (see
    ``docs/development/validation_and_xarray.rst``).

    Raises:
        ValueError: If the reference is not a finite symmetric
            ``n_observation`` square matrix or is not numerically positive
            definite.
    """
    values = np.asarray(reference_covariance, dtype=float)
    if values.shape != (n_observation, n_observation):
        raise ValueError(
            f"The reference covariance must be {n_observation} x {n_observation}; got shape {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("The reference covariance must be finite.")
    _validate_dense_covariance_symmetry(values, owner="The reference covariance")
    try:
        (factor, _), reciprocal_condition = cholesky_with_condition(0.5 * (values + values.T))
    except np.linalg.LinAlgError as exc:
        raise ValueError("The reference covariance must be positive definite.") from exc
    if not reciprocal_condition > REFERENCE_MIN_RECIPROCAL_CONDITION:
        raise ValueError(
            "The reference covariance is numerically singular: estimated reciprocal 1-norm condition "
            f"{reciprocal_condition:.3e} is not above {REFERENCE_MIN_RECIPROCAL_CONDITION:.3e}."
        )
    return factor


def _whitened_spectrum(values: np.ndarray, cholesky: np.ndarray) -> np.ndarray:
    """Return descending whitened eigenvalues of checked ``A`` with negative roundoff clipped.

    Forms ``L^{-1} A L^{-T}`` with two triangular solves: the first gives
    ``L^{-1} A`` and the second, applied to its transpose, ``L^{-1} A L^{-T}``.
    Only the lower triangle of ``cholesky`` is read.
    """
    whitened = solve_triangular(cholesky, values, lower=True, check_finite=False)
    whitened = solve_triangular(cholesky, whitened.T, lower=True, check_finite=False)
    whitened += whitened.T
    whitened *= 0.5
    eigenvalues = np.linalg.eigvalsh(whitened)[::-1]
    del whitened
    # The LRPD factorization owns positive semidefiniteness of A at its own scale.
    return np.maximum(eigenvalues, 0.0)


def _is_positive_real(value: object) -> bool:
    """Return whether ``value`` is a positive finite real number, not a bool."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(np.isfinite(float(value)))
        and float(value) > 0.0
    )


def _diagonal_preserving_spectral_lrpd(
    values: np.ndarray,
    *,
    rank: int | None,
    roundoff_tolerance: float,
    owner: str,
) -> _SpectralLrpd:
    """Construct a coupled spectral factor and diagonal covariance tail.

    Args:
        values: Eager finite symmetric covariance values.
        rank: Maximum retained numerical rank, or ``None`` for every
            numerically positive mode.
        roundoff_tolerance: Scale-relative absolute tolerance established at
            the covariance boundary.
        owner: Scientific owner named in validation errors.

    Returns:
        Factor, diagonal tail, and algebraic approximation diagnostics.

    Raises:
        ValueError: If the covariance is materially indefinite or produces a
            materially negative diagonal tail.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(values, UPLO="L")
    if float(eigenvalues.min()) < -roundoff_tolerance:
        raise ValueError(f"{owner} must be positive semidefinite.")
    positive_count = int(np.count_nonzero(eigenvalues > roundoff_tolerance))
    keep_count = positive_count if rank is None else min(rank, positive_count)
    if keep_count:
        retained = eigenvalues[-keep_count:][::-1]
        factor = eigenvectors[:, -keep_count:][:, ::-1] * np.sqrt(retained)
        retained_diagonal = np.einsum("ij,ij->i", factor, factor)
        discarded = eigenvalues[:-keep_count]
    else:
        retained = eigenvalues[:0]
        factor = np.empty((values.shape[0], 0), dtype=float)
        retained_diagonal = np.zeros(values.shape[0], dtype=float)
        discarded = eigenvalues
    del eigenvectors

    raw_diagonal = np.diag(values) - retained_diagonal
    if float(raw_diagonal.min()) < -roundoff_tolerance:
        raise ValueError("Low-rank approximation produced a negative diagonal residual variance.")
    clipped = raw_diagonal < 0.0
    clipped_count = int(np.count_nonzero(clipped))
    max_clipped_magnitude = float(np.max(-raw_diagonal[clipped])) if clipped_count else 0.0
    diagonal = np.maximum(raw_diagonal, 0.0)
    source_norm = float(np.sqrt(np.dot(eigenvalues, eigenvalues)))
    error_squared = float(
        np.dot(discarded, discarded) - 2.0 * np.dot(raw_diagonal, diagonal) + np.dot(diagonal, diagonal)
    )
    reconstruction_error = float(np.sqrt(max(error_squared, 0.0)))
    return _SpectralLrpd(
        factor=factor,
        diagonal=diagonal,
        retained_spectrum=float(retained.sum()),
        positive_spectrum=float(np.maximum(eigenvalues, 0.0).sum()),
        source_frobenius_norm=source_norm,
        reconstruction_error=reconstruction_error,
        diagonal_tail_clipped_count=clipped_count,
        diagonal_tail_max_clipped_magnitude=max_clipped_magnitude,
    )


def _covariance_roundoff_tolerance(values: np.ndarray) -> float:
    """Return the covariance-scale absolute numerical tolerance."""
    scale = float(np.max(np.abs(values)))
    return 1e-10 * scale if scale else 0.0


def _validate_dense_covariance_symmetry(
    values: np.ndarray,
    *,
    owner: str,
) -> float:
    """Require a materialized dense covariance to be symmetric.

    Args:
        values: Materialized square covariance values.
        owner: Scientific owner named in validation errors.

    Returns:
        Scale-relative absolute roundoff tolerance.

    Raises:
        ValueError: If ``values`` is not symmetric within the scale-based
            numerical tolerance.
    """
    tolerance = _covariance_roundoff_tolerance(values)
    block_rows = 256
    for start in range(0, values.shape[0], block_rows):
        stop = min(start + block_rows, values.shape[0])
        if not np.allclose(
            values[start:stop],
            values[:, start:stop].T,
            rtol=1e-10,
            atol=tolerance,
        ):
            raise ValueError(f"{owner} must be symmetric.")
    return tolerance


def _validate_dense_covariance_values(
    values: np.ndarray,
    *,
    owner: str,
) -> float:
    """Require a materialized dense covariance to be symmetric and PSD.

    Args:
        values: Materialized square covariance values.
        owner: Scientific owner named in validation errors.

    Returns:
        Scale-relative absolute roundoff tolerance.

    Raises:
        ValueError: If ``values`` is not symmetric or positive semidefinite
            within the scale-based numerical tolerance.
    """
    tolerance = _validate_dense_covariance_symmetry(values, owner=owner)
    eigenvalues = np.linalg.eigvalsh(values, UPLO="L")
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(f"{owner} must be positive semidefinite.")
    return tolerance


def _numeric_finite(
    name: str,
    array: xr.DataArray,
    *,
    owner: str = "Aggregation-error input",
) -> np.ndarray:
    """Return numeric finite array values or raise a labelled error.

    Args:
        name: Scientific input name used in diagnostics.
        array: Labelled values to materialize and validate.
        owner: Component label used in diagnostics.

    Returns:
        Materialized NumPy values.

    Raises:
        ValueError: If values are non-numeric or non-finite.
    """
    values = np.asarray(array.values)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{owner} {name!r} must be numeric.")
    if not np.isfinite(values).all():
        raise ValueError(f"{owner} {name!r} must contain only finite values.")
    return values


def _validate_vector(
    data: xr.Dataset,
    name: str,
    *,
    output_dim: str,
    nonnegative: bool = True,
) -> tuple[xr.DataArray, np.ndarray]:
    """Validate one observation-aligned aggregation-error vector.

    Args:
        data: Dataset owning the vector and observation dimension.
        name: Vector variable name.
        output_dim: Required observation dimension.
        nonnegative: Whether negative values are invalid.

    Returns:
        Original labelled vector and its materialized numeric values.

    Raises:
        ValueError: If the vector has invalid dimensions, alignment, values,
            or sign.
    """
    array = data[name]
    if array.dims != (output_dim,):
        raise ValueError(
            f"Aggregation-error input {name!r} must have dims ({output_dim!r},); got {array.dims!r}."
        )
    if array.sizes[output_dim] != data.sizes[output_dim]:
        raise ValueError(f"Aggregation-error input {name!r} is not observation-aligned.")
    values = _numeric_finite(name, array)
    if nonnegative and (values < 0).any():
        raise ValueError(f"Aggregation-error input {name!r} must contain only non-negative values.")
    return array, values


def validate_observation_error_arrays(
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray | None,
    *,
    owner: str,
    output_dim: str = "nmeasure",
) -> None:
    """Validate the named scientific arrays consumed by an error component.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Optional minimum total-error standard deviations.
        owner: Name of the likelihood/error component consuming the arrays.
        output_dim: Required observation dimension.

    Raises:
        ValueError: If an input is not an aligned observation vector or an
            error array is non-numeric, non-finite, or negative.
    """
    if observations.dims != (output_dim,):
        raise ValueError(
            f"{owner} input 'observations' must have dims ({output_dim!r},); got {observations.dims!r}."
        )
    _numeric_finite("observations", observations, owner=f"{owner} input")
    nmeasure = observations.sizes[output_dim]
    arrays = [("observation_error", observation_error)]
    if minimum_error is not None:
        arrays.append(("minimum_error", minimum_error))
    for name, array in arrays:
        if array.dims != (output_dim,):
            raise ValueError(f"{owner} input {name!r} must have dims ({output_dim!r},); got {array.dims!r}.")
        if array.sizes[output_dim] != nmeasure:
            raise ValueError(f"{owner} input {name!r} is not observation-aligned.")
        values = _numeric_finite(name, array, owner=f"{owner} input")
        if (values < 0).any():
            raise ValueError(f"{owner} input {name!r} must contain only non-negative values.")


def select_aggregation_error_mode(
    data: xr.Dataset, requested: AggregationErrorMode
) -> Literal["none", "dense", "low_rank"]:
    """Select an aggregation-error representation without materializing it.

    Args:
        data: Prepared inversion inputs containing any available aggregation-error
            representations.
        requested: Requested representation, or ``"auto"`` to infer one from
            the available inputs.

    Returns:
        The selected concrete aggregation-error representation.

    Raises:
        ValueError: If ``requested`` is invalid or ``"auto"`` finds both dense
            and low-rank representations or a standalone standard deviation.
    """
    if requested not in ("auto", "none", "dense", "low_rank"):
        raise ValueError(
            "`aggregation_error_mode` must be one of 'auto', 'none', 'dense', "
            f"or 'low_rank'; got {requested!r}."
        )
    if requested != "auto":
        return requested

    dense = AGGREGATION_ERROR_COVARIANCE in data
    low_rank = LOW_RANK_FACTOR in data or DIAGONAL_RESIDUAL_VARIANCE in data
    if dense and low_rank:
        raise ValueError(
            "Prepared inputs contain both dense and low-rank aggregation-error covariance; "
            "set `aggregation_error_mode` explicitly."
        )
    if dense:
        return "dense"
    if low_rank:
        return "low_rank"
    if AGGREGATION_ERROR_SD in data:
        raise ValueError(
            f"Prepared inputs contain {AGGREGATION_ERROR_SD!r} without an aggregation-error covariance. "
            "Supply a covariance or select 'none' explicitly."
        )
    return "none"


def aggregation_error_input_names(
    data: xr.Dataset,
    requested: AggregationErrorMode,
) -> tuple[str, ...]:
    """Return labelled arrays required by the selected error component.

    Args:
        data: Prepared inputs containing available aggregation-error products.
        requested: Requested representation, or ``"auto"``.

    Returns:
        Variable names required to materialize the selected representation.

    Raises:
        ValueError: If the requested mode is invalid or automatic selection is
            ambiguous or finds only a standalone standard deviation.
    """
    selected = select_aggregation_error_mode(data, requested)
    if selected == "dense":
        names = [AGGREGATION_ERROR_COVARIANCE]
        if AGGREGATION_ERROR_SD in data:
            names.append(AGGREGATION_ERROR_SD)
        return tuple(names)
    if selected == "low_rank":
        names = [LOW_RANK_FACTOR, DIAGONAL_RESIDUAL_VARIANCE]
        if AGGREGATION_ERROR_SD in data:
            names.append(AGGREGATION_ERROR_SD)
        return tuple(names)
    return ()


def resolve_aggregation_error(
    data: xr.Dataset,
    mode: AggregationErrorMode = "auto",
    *,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> AggregationError:
    """Validate and select an aggregation-error covariance representation.

    In ``"auto"`` mode, ``aggregation_error_sd`` is a marginal diagnostic beside
    a dense or low-rank covariance. Automatic selection chooses the only
    covariance present, or ``"none"`` when there is no aggregation-error input.
    Supplying both covariances or a standalone standard deviation requires an
    explicit selection.

    Args:
        data: Prepared inversion inputs containing the requested aggregation-
            error representation.
        mode: Representation to use, or ``"auto"`` to select from available
            inputs.
        output_dim: Observation dimension used by error vectors and the first
            covariance dimension.
        covariance_dim: Second dimension required for a dense covariance.

    Returns:
        Validated aggregation-error arrays and their marginal variance.

    Raises:
        ValueError: If the selected inputs are absent, malformed, inconsistent,
            non-finite, or not a valid covariance representation.
    """
    if output_dim not in data.dims:
        raise ValueError(f"Prepared inputs have no observation dimension {output_dim!r}.")
    selected = select_aggregation_error_mode(data, mode)
    nmeasure = data.sizes[output_dim]

    if selected == "none":
        return AggregationError(mode="none", marginal_variance=np.zeros(nmeasure))

    if selected == "dense":
        if AGGREGATION_ERROR_COVARIANCE not in data:
            raise ValueError(
                f"Dense aggregation error requires {AGGREGATION_ERROR_COVARIANCE!r} in prepared inputs."
            )
        covariance = data[AGGREGATION_ERROR_COVARIANCE]
        if covariance.dims != (output_dim, covariance_dim):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must have dims "
                f"({output_dim!r}, {covariance_dim!r}); got {covariance.dims!r}."
            )
        if covariance.shape != (nmeasure, nmeasure):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r} must be square and "
                f"match {output_dim!r}; got shape {covariance.shape!r}."
            )
        observation_labels = np.asarray(data.get_index(output_dim).values)
        missing_coords = {
            name: observation_labels for name in (output_dim, covariance_dim) if name not in covariance.coords
        }
        if missing_coords:
            covariance = covariance.assign_coords(missing_coords)
        validate_covariance_coordinates(
            covariance,
            dim=output_dim,
            covariance_dim=covariance_dim,
        )
        (covariance,) = materialize_together(covariance, purpose="aggregation-error covariance")
        values = _numeric_finite(AGGREGATION_ERROR_COVARIANCE, covariance)
        _validate_dense_covariance_values(
            values,
            owner=f"Aggregation-error input {AGGREGATION_ERROR_COVARIANCE!r}",
        )
        marginal_variance = np.diag(values).copy()
        _validate_marginal_sd(data, marginal_variance, output_dim=output_dim)
        return AggregationError(
            mode="dense",
            marginal_variance=marginal_variance,
            covariance=covariance,
        )

    missing = [name for name in (LOW_RANK_FACTOR, DIAGONAL_RESIDUAL_VARIANCE) if name not in data]
    if missing:
        raise ValueError(f"Low-rank aggregation error is missing input(s): {missing!r}.")
    factor = data[LOW_RANK_FACTOR]
    if factor.ndim != 2 or factor.dims[0] != output_dim:
        raise ValueError(
            f"Aggregation-error input {LOW_RANK_FACTOR!r} must be a two-dimensional array "
            f"whose first dimension is {output_dim!r}; got {factor.dims!r}."
        )
    if factor.sizes[output_dim] != nmeasure:
        raise ValueError(f"Aggregation-error input {LOW_RANK_FACTOR!r} is not observation-aligned.")
    diagonal = data[DIAGONAL_RESIDUAL_VARIANCE]
    if diagonal.dims != (output_dim,):
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} must have dims "
            f"({output_dim!r},); got {diagonal.dims!r}."
        )
    if diagonal.sizes[output_dim] != nmeasure:
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} is not observation-aligned."
        )

    factor, diagonal = materialize_together(
        factor, diagonal, purpose="low-rank aggregation-error factor and diagonal"
    )
    factor_values = _numeric_finite(LOW_RANK_FACTOR, factor)
    diagonal_values = _numeric_finite(DIAGONAL_RESIDUAL_VARIANCE, diagonal)
    if (diagonal_values < 0).any():
        raise ValueError(
            f"Aggregation-error input {DIAGONAL_RESIDUAL_VARIANCE!r} must contain only non-negative values."
        )
    marginal_variance = np.sum(factor_values**2, axis=1) + diagonal_values
    _validate_marginal_sd(data, marginal_variance, output_dim=output_dim)
    return AggregationError(
        mode="low_rank",
        marginal_variance=marginal_variance,
        factor=factor,
        diagonal_variance=diagonal,
    )


def _validate_marginal_sd(
    data: xr.Dataset,
    marginal_variance: np.ndarray,
    *,
    output_dim: str,
) -> None:
    """Validate an optional marginal-SD diagnostic beside structured input.

    Args:
        data: Prepared inputs that may contain ``aggregation_error_sd``.
        marginal_variance: Diagonal of the selected structured covariance.
        output_dim: Observation dimension required on the diagnostic.

    Raises:
        ValueError: If the diagnostic is invalid or disagrees with the selected
            covariance diagonal.
    """
    if AGGREGATION_ERROR_SD not in data:
        return
    _, values = _validate_vector(data, AGGREGATION_ERROR_SD, output_dim=output_dim)
    expected = np.sqrt(marginal_variance)
    if not np.allclose(values, expected, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Aggregation-error diagnostic {AGGREGATION_ERROR_SD!r} must equal the square root "
            "of the selected covariance diagonal."
        )
