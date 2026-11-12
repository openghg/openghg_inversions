"""Backend-neutral aggregation-error covariance contracts.

Aggregation error is fixed input data, separate from measurement error and
the inferred RHIME model-error term.  Prepared inversion inputs may represent
it exactly as a dense covariance, efficiently as a low-rank-plus-diagonal
covariance, or diagnostically as independent standard deviations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

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


def dense_aggregation_covariance(aggregation_error: AggregationError) -> np.ndarray:
    """Materialize any fixed aggregation-error representation as a dense matrix.

    This is a diagnostic and test helper with quadratic memory. Low-rank
    inputs return ``factor @ factor.T + diag(diagonal_variance)``; diagonal
    inputs return their diagonal and ``"none"`` returns zeros.

    Args:
        aggregation_error: Validated fixed aggregation-error representation.

    Returns:
        Dense covariance with shape ``(nmeasure, nmeasure)``.
    """
    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        return np.asarray(aggregation_error.covariance.values, dtype=float)
    factor, diagonal = aggregation_error_as_low_rank(aggregation_error)
    return factor @ factor.T + np.diag(diagonal)


def block_ou_correlation(
    observation_times: np.ndarray,
    group_index: np.ndarray,
    tau_hours: float,
) -> np.ndarray:
    """Return a dense within-group exponential (OU) correlation matrix.

    Rows in the same group are correlated by ``exp(-|t_i - t_j| / tau)``;
    rows in different groups are uncorrelated. This is the correlation
    template of the fixed-OU likelihood, built densely for diagnostics.

    Args:
        observation_times: One time per row, as ``datetime64`` values or
            finite hours.
        group_index: One group label per row, such as a species/site position.
        tau_hours: Positive finite correlation time scale in hours.

    Returns:
        Symmetric correlation matrix with unit diagonal.

    Raises:
        ValueError: If shapes differ or ``tau_hours`` is not positive and finite.
    """
    times = np.asarray(observation_times)
    groups = np.asarray(group_index)
    if times.ndim != 1 or groups.shape != times.shape:
        raise ValueError("observation_times and group_index must be one-dimensional and aligned.")
    if isinstance(tau_hours, bool) or not np.isfinite(tau_hours) or tau_hours <= 0:
        raise ValueError("tau_hours must be a positive finite number.")
    if np.issubdtype(times.dtype, np.datetime64):
        times = (times - times.min()) / np.timedelta64(1, "h")
    hours = np.asarray(times, dtype=float)
    if not np.isfinite(hours).all():
        raise ValueError("observation_times must be finite.")
    correlation = np.exp(-np.abs(hours[:, None] - hours[None, :]) / float(tau_hours))
    return np.where(groups[:, None] == groups[None, :], correlation, 0.0)


def aggregation_error_kl_per_observation(
    reference_covariance: np.ndarray | xr.DataArray,
    approximation: AggregationError,
    *,
    sigma: Sequence[float | np.ndarray],
    independent_error_sd: np.ndarray | None = None,
    sigma_correlation: np.ndarray | None = None,
) -> np.ndarray:
    r"""Gaussian KL per observation lost by approximating aggregation error.

    For each entry of ``sigma`` this evaluates

    .. math::

        \frac{1}{n} \mathrm{KL}\big(N(0, A + D) \,\|\, N(0, \tilde A + D)\big),
        \qquad D = \mathrm{diag}(e^2) + S C S,

    where :math:`A` is the dense reference aggregation covariance,
    :math:`\tilde A` the approximation, :math:`e` the independent error,
    :math:`S = \mathrm{diag}(\sigma)` and :math:`C` the model-data mismatch
    correlation (identity by default, or e.g. :func:`block_ou_correlation`).
    With :math:`L L^T = \tilde A + D` and :math:`\lambda` the eigenvalues of
    :math:`L^{-1}(A - \tilde A)L^{-T}`, the KL is
    :math:`\tfrac12\sum_i(\lambda_i - \log(1 + \lambda_i))`, which is
    non-negative and exactly zero when :math:`\tilde A = A`.

    The approximation error matters most when :math:`D` is small, so evaluate
    it over the posterior range of the mismatch amplitude, not at one value.

    Cost: exact dense algebra, one Cholesky factorization and one symmetric
    eigendecomposition of an ``n x n`` matrix per sigma value, i.e.
    :math:`O(n^3)` time and :math:`O(n^2)` memory. Use it for tests, small
    cases or one-off checks on a large node, not inside a sampler.

    Args:
        reference_covariance: Dense reference aggregation covariance
            :math:`A`; xarray inputs are materialized.
        approximation: Approximate representation :math:`\tilde A` on the same
            rows, in any mode.
        sigma: Mismatch amplitudes to evaluate. Each entry is a scalar applied
            to every row or a per-row array (for example, for mixed channel units).
        independent_error_sd: Optional per-row independent standard
            deviations :math:`e`; omitted means zero.
        sigma_correlation: Optional dense mismatch correlation :math:`C`;
            omitted means independent rows (``diag(sigma**2)``).

    Returns:
        KL per observation in nats, one value per ``sigma`` entry.

    Raises:
        ValueError: If shapes disagree, inputs are non-finite or negative, or
            either total covariance is not positive definite.
    """
    reference = np.asarray(reference_covariance, dtype=float)
    n = reference.shape[0]
    if reference.ndim != 2 or reference.shape != (n, n) or n == 0:
        raise ValueError("reference_covariance must be a non-empty square matrix.")
    approximate = dense_aggregation_covariance(approximation)
    if approximate.shape != reference.shape:
        raise ValueError("approximation must have the same rows as reference_covariance.")
    base = np.zeros(n) if independent_error_sd is None else np.asarray(independent_error_sd, dtype=float)
    if base.shape != (n,) or not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("independent_error_sd must be finite, non-negative and one per row.")
    correlation = None if sigma_correlation is None else np.asarray(sigma_correlation, dtype=float)
    if correlation is not None and correlation.shape != reference.shape:
        raise ValueError("sigma_correlation must have the same shape as reference_covariance.")
    difference = reference - approximate
    result = []
    for value in sigma:
        amplitude = np.broadcast_to(np.asarray(value, dtype=float), (n,))
        if not np.isfinite(amplitude).all() or np.any(amplitude < 0):
            raise ValueError("sigma values must be finite and non-negative.")
        if correlation is None:
            noise = np.diag(np.square(base) + np.square(amplitude))
        else:
            noise = np.diag(np.square(base)) + amplitude[:, None] * correlation * amplitude[None, :]
        try:
            cholesky = np.linalg.cholesky(approximate + noise)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Approximate total covariance is not positive definite; add independent error or sigma."
            ) from error
        whitened = solve_triangular(cholesky, difference, lower=True)
        whitened = solve_triangular(cholesky, whitened.T, lower=True)
        eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
        if eigenvalues.min() <= -1.0:
            raise ValueError("Reference total covariance is not positive definite.")
        result.append(0.5 * float(np.sum(eigenvalues - np.log1p(eigenvalues))) / n)
    return np.asarray(result)


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
