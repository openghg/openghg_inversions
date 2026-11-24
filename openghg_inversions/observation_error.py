"""Backend-neutral aggregation-error covariance contracts.

Aggregation error is fixed input data, separate from measurement error and
the inferred RHIME model-error term.  Prepared inversion inputs may represent
it exactly as a dense covariance, efficiently as a low-rank-plus-diagonal
covariance, or diagnostically as independent standard deviations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, TypeAlias
import warnings

import numpy as np
from scipy.linalg import solve_triangular
import xarray as xr

from openghg_inversions.array_ops import materialize_together, validate_covariance_coordinates

AggregationErrorMode: TypeAlias = Literal["auto", "none", "dense", "low_rank", "diagonal"]

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

    mode: Literal["none", "dense", "low_rank", "diagonal"]
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
    if aggregation_error.mode == "diagonal":
        assert aggregation_error.diagonal_variance is not None
        return np.empty((nmeasure, 0)), np.asarray(aggregation_error.diagonal_variance.values, dtype=float)
    return np.empty((nmeasure, 0)), np.zeros(nmeasure)


@dataclass(frozen=True)
class WhitenedRankRule:
    r"""Choose an LRPD aggregation-error rank from the R-whitened spectrum.

    The rule solves :math:`A u_i = \mu_i R_0 u_i` for the aggregation
    covariance :math:`A` against a reference mismatch covariance

    .. math::

        R_0 = \mathrm{diag}(e^2) + S_0 C S_0,

    where :math:`e` is the reported observation error, :math:`S_0` holds the
    reference mismatch amplitudes and :math:`C` is the identity or a
    within-site OU correlation. :math:`\mu_i` is the variance of mode
    :math:`i` of :math:`A` in units of reference noise. Dropping the modes
    after the leading :math:`k` from the whitened representation costs

    .. math::

        \mathrm{KL}\big(N(0, R_0 + A)\,\|\,N(0, R_0 + A_k)\big)
        = \tfrac12 \sum_{i>k} \big[\mu_i - \log(1 + \mu_i)\big].

    The rank is the larger of the count of :math:`\mu_i` above
    ``mu_threshold`` and the smallest :math:`k` whose dropped-mode KL per
    observation is at most ``kl_budget_per_observation``; either criterion
    may be disabled with ``None``. The stored factor is still the leading
    eigenmodes of :math:`A` with the discarded marginal variance on the
    diagonal; only the number of modes comes from the whitened spectrum.

    The count depends strongly on :math:`R_0`: larger amplitudes whiten more
    of :math:`A` away. Use floored or prior-median amplitudes, never zero.
    The rule is a screening heuristic, not a bound on posterior error. It was
    checked only on four Verification Games CO2 cases (January and July 2021,
    two prior bases, Gaussian analytic posteriors), where a rank at least the
    :math:`\mu > 0.1` count, at either the default reference or prior-median
    within-site OU amplitudes, gave country totals within 0.5 Tg C/month of
    exact dense :math:`A`. Check the chosen rank for the intended likelihood.

    The reference covariance is built from these settings by
    :func:`reference_mismatch_covariance`. To screen against a different
    :math:`R_0`, pass a :class:`ReferenceMismatchCovariance` to
    :func:`whitened_aggregation_spectrum` and apply
    :func:`select_whitened_rank` to the result.

    Args:
        reference_sd: Positive reference mismatch amplitude, either one value
            for every site or a mapping covering every observation site.
            The default 1 ppm matches the observation-noise floor used in the
            VG low-rank investigation.
        units: Mole-fraction units of ``reference_sd``. They are converted to
            the observation units before whitening.
        tau_hours: Optional positive OU correlation time scale in hours, one
            value or a site mapping. ``None`` uses independent reference
            mismatch.
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

    reference_sd: float | Mapping[str, float] = 1.0
    units: str = "ppm"
    tau_hours: float | Mapping[str, float] | None = None
    mu_threshold: float | None = 0.1
    kl_budget_per_observation: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        """Validate rule values at construction."""
        _positive_site_values(self.reference_sd, "reference_sd")
        if self.tau_hours is not None:
            _positive_site_values(self.tau_hours, "tau_hours")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("WhitenedRankRule units must be a non-empty string.")
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

    def reference_record(self) -> dict[str, Any]:
        """Return the JSON-safe reference settings as supplied."""
        return {
            "reference_sd": _json_site_values(self.reference_sd),
            "units": self.units,
            "tau_hours": None if self.tau_hours is None else _json_site_values(self.tau_hours),
            "correlation": "independent" if self.tau_hours is None else "within-site OU",
        }


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


@dataclass(frozen=True, eq=False)
class ReferenceMismatchCovariance:
    r"""Site-block reference mismatch covariance :math:`R_0` for rank screening.

    :math:`R_0` is either diagonal (independent reference mismatch) or block
    diagonal with one dense block per observation site. Build the usual
    reference with :func:`reference_mismatch_covariance`; any other
    numerical covariance with this structure, for example a mismatch model
    evaluated at fixed parameter values, can be supplied directly to
    :func:`whitened_aggregation_spectrum`.

    Attributes:
        variance: Per-row variance, the diagonal of :math:`R_0`.
        site_rows: Row indices of each site block, partitioning every row.
            Empty for a diagonal reference.
        blocks: Dense symmetric positive definite block for each entry of
            ``site_rows``, with diagonal equal to ``variance`` on those rows.
        record: JSON-safe description of how the reference was built.
        minimum_eigenvalue: Smallest eigenvalue of :math:`R_0`, computed at
            construction.

    Raises:
        ValueError: If the variance is not finite and positive, the blocks do
            not partition the rows, or a block is not symmetric positive
            definite with the stated diagonal.
    """

    variance: np.ndarray
    site_rows: tuple[np.ndarray, ...] = ()
    blocks: tuple[np.ndarray, ...] = ()
    record: Mapping[str, Any] = field(default_factory=dict)
    minimum_eigenvalue: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate the structure and record the smallest eigenvalue."""
        variance = np.asarray(self.variance, dtype=float)
        if (
            variance.ndim != 1
            or variance.size == 0
            or not np.isfinite(variance).all()
            or np.any(variance <= 0.0)
        ):
            raise ValueError(
                "Reference mismatch variance must be a non-empty vector of positive finite values."
            )
        if len(self.site_rows) != len(self.blocks):
            raise ValueError("Reference mismatch covariance needs one block per site row set.")
        site_rows = tuple(np.asarray(rows) for rows in self.site_rows)
        for rows in site_rows:
            if rows.ndim != 1 or rows.size == 0 or not np.issubdtype(rows.dtype, np.integer):
                raise ValueError(
                    "Each reference mismatch block's row indices must be a non-empty one-dimensional "
                    "integer array."
                )
        minimum = float(variance.min())
        if site_rows:
            covered = np.sort(np.concatenate(site_rows))
            if not np.array_equal(covered, np.arange(variance.size)):
                raise ValueError(
                    "Reference mismatch site blocks must partition every observation row exactly once."
                )
            minimum = np.inf
            for rows, block in zip(site_rows, self.blocks, strict=True):
                values = np.asarray(block, dtype=float)
                if values.shape != (len(rows), len(rows)) or not np.isfinite(values).all():
                    raise ValueError("Each reference mismatch block must be finite and match its rows.")
                if not np.allclose(values, values.T, rtol=1e-12, atol=0.0) or not np.allclose(
                    np.diag(values), variance[rows], rtol=1e-12, atol=0.0
                ):
                    raise ValueError(
                        "Each reference mismatch block must be symmetric with diagonal equal to variance."
                    )
                minimum = min(minimum, float(np.linalg.eigvalsh(values)[0]))
            if minimum <= 0.0:
                raise ValueError(
                    "Reference mismatch covariance is not positive definite; check for repeated observation "
                    "times with zero observation error."
                )
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "site_rows", site_rows)
        object.__setattr__(self, "minimum_eigenvalue", minimum)


def reference_mismatch_covariance(
    observation_error_sd: np.ndarray,
    sites: np.ndarray,
    *,
    reference_sd: float | Mapping[str, float],
    tau_hours: float | Mapping[str, float] | None = None,
    observation_hours: np.ndarray | None = None,
) -> ReferenceMismatchCovariance:
    r"""Build the reference :math:`R_0 = \mathrm{diag}(e^2) + S_0 C S_0`.

    :math:`e` is the reported observation error, :math:`S_0` holds one
    reference amplitude per site and :math:`C` is the identity or, with
    ``tau_hours``, a within-site OU correlation
    :math:`\exp(-|t_i - t_j| / \tau_s)` with one time scale per site. Values
    are resolved by site label, so the result does not depend on row order.

    Args:
        observation_error_sd: Per-row reported observation error, finite and
            non-negative.
        sites: Per-row site labels.
        reference_sd: Positive reference amplitude in the units of
            ``observation_error_sd``: one value or a mapping covering every
            site.
        tau_hours: Optional positive OU time scale in hours: one value or a
            mapping covering every site. ``None`` gives a diagonal reference.
        observation_hours: Per-row observation times in hours. Required with
            ``tau_hours``.

    Returns:
        The reference covariance with a JSON-safe ``record`` of its settings.

    Raises:
        ValueError: If shapes disagree, values are invalid, a mapping omits a
            site, or a site block is not positive definite.
    """
    labels = np.asarray(sites).astype(str)
    error = np.asarray(observation_error_sd, dtype=float)
    n = labels.size
    if labels.ndim != 1 or n == 0 or error.shape != (n,):
        raise ValueError("observation_error_sd and sites must be non-empty and one per row.")
    if not np.isfinite(error).all() or np.any(error < 0.0):
        raise ValueError("observation_error_sd must be finite and non-negative.")
    _positive_site_values(reference_sd, "reference_sd", owner="Reference mismatch covariance")
    amplitude = _site_rows(reference_sd, labels, "reference_sd")
    record: dict[str, Any] = {
        "reference_sd": _json_site_values(reference_sd),
        "tau_hours": None if tau_hours is None else _json_site_values(tau_hours),
        "correlation": "independent" if tau_hours is None else "within-site OU",
        "observation_error": "reported observation-error variance included",
        "observation_error_sd_min": float(error.min()),
        "observation_error_sd_median": float(np.median(error)),
    }
    variance = np.square(error) + np.square(amplitude)
    if tau_hours is None:
        return ReferenceMismatchCovariance(variance=variance, record=record)

    _positive_site_values(tau_hours, "tau_hours", owner="Reference mismatch covariance")
    tau = _site_rows(tau_hours, labels, "tau_hours")
    if observation_hours is None:
        raise ValueError("An OU reference mismatch covariance requires observation_hours.")
    hours = np.asarray(observation_hours, dtype=float)
    if hours.shape != (n,) or not np.isfinite(hours).all():
        raise ValueError("observation_hours must be finite and one per row.")
    site_rows = []
    blocks = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        block = np.square(amplitude[rows[0]]) * np.exp(
            -np.abs(hours[rows, None] - hours[None, rows]) / tau[rows[0]]
        )
        block[np.diag_indices(rows.size)] = variance[rows]
        site_rows.append(rows)
        blocks.append(block)
    return ReferenceMismatchCovariance(
        variance=variance, site_rows=tuple(site_rows), blocks=tuple(blocks), record=record
    )


def whitened_aggregation_spectrum(
    covariance: np.ndarray,
    reference: ReferenceMismatchCovariance,
) -> np.ndarray:
    r"""Return generalized eigenvalues of aggregation error against a reference.

    Solves :math:`A u = \mu R_0 u` by forming :math:`L^{-1} A L^{-T}`, where
    :math:`L` is the block-diagonal Cholesky factor of :math:`R_0`, and taking
    its eigenvalues. Only eigenvalues are computed. Cost is one dense
    ``n x n`` copy, block triangular solves and one symmetric eigenvalue
    decomposition, :math:`O(n^3)`.

    :math:`A` must be a finite, square and symmetric matrix at its own scale
    (the tolerance used by :func:`prepare_low_rank_aggregation_error`).
    Positive semidefiniteness is owned by that factorization: a negative
    whitened eigenvalue is rejected only when it implies an eigenvalue of
    :math:`A` below the negative roundoff tolerance, and accepted roundoff is
    clipped to zero.

    Args:
        covariance: Dense symmetric aggregation covariance :math:`A`.
        reference: Reference mismatch covariance :math:`R_0` in the same
            units squared.

    Returns:
        Whitened eigenvalues in descending order, with accepted roundoff
        below zero clipped to zero.

    Raises:
        ValueError: If ``covariance`` is not a finite square symmetric matrix
            matching ``reference``, or is materially indefinite.
    """
    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("Aggregation covariance must be a non-empty square matrix.")
    if values.shape[0] != reference.variance.size:
        raise ValueError(
            f"Aggregation covariance has {values.shape[0]} rows but the reference covariance has "
            f"{reference.variance.size}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Aggregation covariance must be finite.")
    tolerance = _validate_dense_covariance_symmetry(values, owner="Aggregation covariance")

    if not reference.site_rows:
        scale = np.sqrt(reference.variance)
        whitened = values / scale[:, None]
        whitened /= scale[None, :]
    else:
        # Form L^{-1} A L^{-T} for the block-diagonal Cholesky factor L of R0: the first pass solves on
        # each site's rows (L^{-1} A), the second on each site's columns ((L^{-1} A) L^{-T}).
        whitened = values.copy()
        factors = [
            (rows, np.linalg.cholesky(block)) for rows, block in zip(reference.site_rows, reference.blocks)
        ]
        for rows, cholesky in factors:
            whitened[rows, :] = solve_triangular(cholesky, whitened[rows, :], lower=True)
        for rows, cholesky in factors:
            whitened[:, rows] = solve_triangular(cholesky, whitened[:, rows].T, lower=True).T
        whitened += whitened.T
        whitened *= 0.5

    eigenvalues = np.linalg.eigvalsh(whitened)[::-1]
    del whitened
    # v'Av >= lambda_min(A) |v|^2 and v'R0v >= lambda_min(R0) |v|^2, so a negative whitened eigenvalue
    # satisfies mu >= lambda_min(A) / lambda_min(R0). Every A accepted at its own roundoff tolerance
    # therefore has mu >= -tolerance / lambda_min(R0); allow the whitening's own roundoff on top.
    floor = tolerance / reference.minimum_eigenvalue + 1e-10 * float(np.max(np.abs(eigenvalues)))
    if float(eigenvalues[-1]) < -floor:
        raise ValueError("Aggregation covariance must be positive semidefinite.")
    return np.maximum(eigenvalues, 0.0)


def select_whitened_rank(
    eigenvalues: np.ndarray,
    rule: WhitenedRankRule,
) -> tuple[int, dict[str, Any]]:
    """Apply a rank rule to whitened eigenvalues.

    Args:
        eigenvalues: Non-negative whitened eigenvalues in descending order.
        rule: Rank criteria and optional explicit override.

    Returns:
        The chosen rank, at least one and at most the eigenvalue count, and a
        JSON-safe record of counts, criteria and dropped-mode KL. Issues a
        ``UserWarning`` when an explicit rank is below the rule rank.
    """
    mu = np.asarray(eigenvalues, dtype=float)
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
            stacklevel=3,
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


def choose_whitened_aggregation_rank(
    covariance: np.ndarray,
    rule: WhitenedRankRule,
    *,
    observation_error_sd: np.ndarray,
    sites: np.ndarray,
    observation_times: np.ndarray | None = None,
    reference_unit_scale: float = 1.0,
    observation_units: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Choose an LRPD rank from a dense aggregation covariance and a rule.

    This is an eager numerical boundary: it builds the rule's reference with
    :func:`reference_mismatch_covariance`, computes the whitened spectrum with
    :func:`whitened_aggregation_spectrum` and applies
    :func:`select_whitened_rank`. To screen against another reference
    covariance, call those two functions directly.

    Args:
        covariance: Dense symmetric aggregation covariance in observation
            units squared.
        rule: Reference covariance, criteria and optional explicit rank.
        observation_error_sd: Per-row reported observation error in
            observation units, included in the reference covariance.
        sites: Per-row site labels used for site mappings and OU blocks.
        observation_times: Per-row ``datetime64`` times or hours. Required
            when ``rule.tau_hours`` is set.
        reference_unit_scale: Multiplier converting ``rule.units`` to the
            observation units.
        observation_units: Observation units, recorded in the diagnostics.

    Returns:
        The chosen rank and a JSON-safe record of the reference covariance
        (supplied and per-site amplitudes in observation units, the
        observation-error summary), counts, criteria and dropped-mode KL.

    Raises:
        ValueError: If inputs are invalid or a site mapping omits a site.
    """
    labels = np.asarray(sites).astype(str)
    unit_scale = float(reference_unit_scale)
    reference_sd = (
        {site: float(value) * unit_scale for site, value in rule.reference_sd.items()}
        if isinstance(rule.reference_sd, Mapping)
        else float(rule.reference_sd) * unit_scale
    )
    hours = None
    if rule.tau_hours is not None:
        if observation_times is None:
            raise ValueError("An OU whitened-rank reference requires observation times.")
        times = np.asarray(observation_times)
        if np.issubdtype(times.dtype, np.datetime64):
            nanoseconds = times.astype("datetime64[ns]")
            hours = (nanoseconds - nanoseconds.min()) / np.timedelta64(1, "h")
        else:
            hours = times.astype(float)
    reference = reference_mismatch_covariance(
        observation_error_sd,
        labels,
        reference_sd=reference_sd,
        tau_hours=rule.tau_hours,
        observation_hours=hours,
    )
    eigenvalues = whitened_aggregation_spectrum(covariance, reference)
    rank, selection = select_whitened_rank(eigenvalues, rule)
    amplitude = _site_rows(reference_sd, labels, "reference_sd")
    site_amplitudes = {
        label: float(amplitude[np.flatnonzero(labels == label)[0]]) for label in np.unique(labels)
    }
    record = {
        **reference.record,
        **rule.reference_record(),
        "observation_units": observation_units,
        "reference_sd_observation_units_by_site": site_amplitudes,
    }
    return rank, {"reference_covariance": record, **selection}


def _is_positive_real(value: object) -> bool:
    """Return whether ``value`` is a positive finite real number, not a bool."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(np.isfinite(float(value)))
        and float(value) > 0.0
    )


def _positive_site_values(value: object, name: str, *, owner: str = "WhitenedRankRule") -> None:
    """Validate a positive scalar or a non-empty mapping of positive site values."""
    if isinstance(value, Mapping):
        if not value or not all(
            isinstance(site, str) and _is_positive_real(item) for site, item in value.items()
        ):
            raise ValueError(
                f"{owner} {name} mapping must be non-empty with string sites and positive values."
            )
    elif not _is_positive_real(value):
        raise ValueError(f"{owner} {name} must be a positive finite number or a site mapping.")


def _json_site_values(value: float | Mapping[str, float]) -> float | dict[str, float]:
    """Return a scalar or site mapping as JSON-safe floats."""
    if isinstance(value, Mapping):
        return {str(site): float(item) for site, item in value.items()}
    return float(value)


def _site_rows(
    value: float | Mapping[str, float],
    sites: np.ndarray,
    name: str,
) -> np.ndarray:
    """Expand a scalar or site mapping to one value per observation row.

    Args:
        value: One value or a mapping keyed by site label.
        sites: Per-row site labels.
        name: Option name used in errors.

    Returns:
        Float values aligned with ``sites``.

    Raises:
        ValueError: If the mapping omits an observation site.
    """
    labels = np.asarray(sites).astype(str)
    if not isinstance(value, Mapping):
        return np.full(labels.size, float(value))
    missing = sorted(str(label) for label in set(labels) - set(value))
    if missing:
        raise ValueError(f"{name} site mapping is missing observation site(s): {missing!r}.")
    return np.asarray([float(value[label]) for label in labels])


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
) -> Literal["none", "dense", "low_rank", "diagonal"]:
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
            and low-rank representations.
    """
    if requested not in ("auto", "none", "dense", "low_rank", "diagonal"):
        raise ValueError(
            "`aggregation_error_mode` must be one of 'auto', 'none', 'dense', "
            f"'low_rank', or 'diagonal'; got {requested!r}."
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
        return "diagonal"
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
            ambiguous.
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
    if selected == "diagonal":
        return (AGGREGATION_ERROR_SD,)
    return ()


def resolve_aggregation_error(
    data: xr.Dataset,
    mode: AggregationErrorMode = "auto",
    *,
    output_dim: str = "nmeasure",
    covariance_dim: str = "nmeasure_cov",
) -> AggregationError:
    """Validate and select an aggregation-error covariance representation.

    In ``"auto"`` mode, a structured representation takes precedence over
    ``aggregation_error_sd`` because that vector is commonly retained as a
    marginal diagnostic beside the exact covariance.  Supplying both dense and
    low-rank forms is ambiguous and therefore requires an explicit selection.

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

    if selected == "diagonal":
        if AGGREGATION_ERROR_SD not in data:
            raise ValueError(
                f"Diagonal aggregation error requires {AGGREGATION_ERROR_SD!r} in prepared inputs."
            )
        standard_deviation = data[AGGREGATION_ERROR_SD]
        if standard_deviation.dims != (output_dim,):
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_SD!r} must have dims "
                f"({output_dim!r},); got {standard_deviation.dims!r}."
            )
        if standard_deviation.sizes[output_dim] != nmeasure:
            raise ValueError(f"Aggregation-error input {AGGREGATION_ERROR_SD!r} is not observation-aligned.")
        (standard_deviation,) = materialize_together(
            standard_deviation, purpose="aggregation-error standard deviation"
        )
        values = _numeric_finite(AGGREGATION_ERROR_SD, standard_deviation)
        if (values < 0).any():
            raise ValueError(
                f"Aggregation-error input {AGGREGATION_ERROR_SD!r} must contain only non-negative values."
            )
        return AggregationError(
            mode="diagonal",
            marginal_variance=values**2,
            diagonal_variance=standard_deviation**2,
        )

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
