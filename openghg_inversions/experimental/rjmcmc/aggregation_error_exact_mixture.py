"""Structure-preserving compression of root Gamma--Dirichlet mixtures.

The exact root aggregation-error likelihood is a continuous Gaussian location
mixture over one Dirichlet allocation.  This module provides a
structure-preserving approximation with separate, computable bounds for
projection omission and finite-bank cluster compression:

* rank observation-space directions with the exact analytic, noise-whitened
  residual covariance;
* construct a direct equal-weight allocation bank with
  :class:`~openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture.ConditionalAllocationMixture`;
* cluster its leading residual locations with SciPy's tested ``kmeans2``
  implementation; and
* replace every cluster by a Gaussian with the cluster's exact population
  mean and covariance.

For large native grids, :func:`build_chunked_projected_root_bank` constructs
the frozen bank directly in the leading non-Gaussian coordinates while
retaining the complete spectrum for the analytic Gaussian complement.

The resulting finite mixture preserves the source bank's normalization, mean,
and covariance.  Measurement noise is convolved analytically.  Residual
eigendirections retained outside the non-Gaussian mixture use a Gaussian
approximation matching the analytic first two moments, while directions beyond
the retained spectrum use measurement noise alone.  The reported bounds do
not cover scrambled-Sobol discretization error, this complementary Gaussian
approximation, or pointwise log-likelihood error.

Only a single retained root is supported.  With several retained regions the
within-cluster covariance is a quadratic function of all region masses and
cannot be represented by the scalar scaling used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.cluster.vq import ClusterError, kmeans2
from scipy.special import logsumexp

from .aggregation_error_conditional_mixture import ConditionalAllocationMixture
from .aggregation_error_low_rank import AdditiveDirichletAggregation

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "CompressedRootMixture",
    "RootResidualSpectrum",
    "build_chunked_projected_root_bank",
    "compressed_root_mixture_log_likelihood",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


def _readonly_float(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    """Return a finite owned read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _readonly_integer(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> IntArray:
    """Return an owned read-only int64 array after integer-kind validation."""
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(f"{name} must contain integers.")
    result = np.array(raw, dtype=np.int64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    result.setflags(write=False)
    return result


def _positive_integer(value: int, *, name: str) -> int:
    """Return one validated positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _unsigned_seed(value: int, *, name: str) -> int:
    """Return one validated unsigned 64-bit seed."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if not 0 <= result < 2**64:
        raise ValueError(f"{name} must lie in [0, 2**64).")
    return result


def _array_sha256(values: FloatArray) -> str:
    """Return a shape- and value-sensitive canonical float64 array digest."""
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {
            "dtype": "<f8",
            "shape": list(values.shape),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonicalize_eigenvector_signs(vectors: FloatArray) -> FloatArray:
    """Return eigenvectors with a deterministic largest-entry sign."""
    result = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True, slots=True, eq=False, init=False)
class RootResidualSpectrum:
    """Numerically retained spectrum of one analytic root covariance.

    The spectrum is constructed from the complete native observation operator
    and Dirichlet concentrations.  It does not depend on the realized
    observation or current root mass.  At root mass ``T`` its eigenvalues are
    ``T**2 * eigenvalues``.

    Attributes:
        observation_mean_design: Observation-space conditional mean per unit
            root mass.
        noise_sd: Measurement-error standard deviations.
        basis: Leading orthonormal eigenvectors in noise-whitened observation
            space, with shape ``(observations, retained_rank)``.
        eigenvalues: Descending positive analytic covariance eigenvalues per
            squared unit root mass.
        total_variance: Trace of the roundoff-clipped analytic covariance.
        discarded_variance: Trace not represented by :attr:`basis`, including
            tolerance-small numerical eigenvalues.
        requested_retained_variance_fraction: Construction threshold.
        retained_variance_fraction: Actual retained analytic trace fraction.
        eigenvalue_tolerance: Scale-aware numerical zero threshold.
        cell_alphas_sha256: Native Dirichlet concentration identity.
        design_sha256: Native observation-operator identity.
        noise_sd_sha256: Measurement-scale identity.
    """

    observation_mean_design: FloatArray = field(init=False)
    noise_sd: FloatArray = field(init=False)
    basis: FloatArray = field(init=False)
    eigenvalues: FloatArray = field(init=False)
    total_variance: float = field(init=False)
    discarded_variance: float = field(init=False)
    requested_retained_variance_fraction: float = field(init=False)
    retained_variance_fraction: float = field(init=False)
    eigenvalue_tolerance: float = field(init=False)
    cell_alphas_sha256: str = field(init=False)
    design_sha256: str = field(init=False)
    noise_sd_sha256: str = field(init=False)

    def __init__(
        self,
        observation_mean_design: ArrayLike,
        noise_sd: ArrayLike,
        basis: ArrayLike,
        eigenvalues: ArrayLike,
        *,
        total_variance: float,
        discarded_variance: float,
        requested_retained_variance_fraction: float,
        eigenvalue_tolerance: float,
        cell_alphas_sha256: str,
        design_sha256: str,
        noise_sd_sha256: str,
    ) -> None:
        """Validate and own a constructed root residual spectrum.

        Args:
            observation_mean_design: Unit-root conditional observation mean.
            noise_sd: Positive observation-error standard deviations.
            basis: Retained whitened orthonormal eigenvectors.
            eigenvalues: Positive descending retained eigenvalues.
            total_variance: Analytic covariance trace after roundoff clipping.
            discarded_variance: Trace outside ``basis``.
            requested_retained_variance_fraction: Requested trace fraction.
            eigenvalue_tolerance: Numerical eigenvalue-zero threshold.
            cell_alphas_sha256: Native concentration-array identity.
            design_sha256: Native design-array identity.
            noise_sd_sha256: Noise-scale-array identity.

        Raises:
            ValueError: If arrays, eigenpairs, variances, or fractions violate
                the spectrum contract.
        """
        mean = _readonly_float(
            observation_mean_design,
            name="observation_mean_design",
            ndim=1,
        )
        scale = _readonly_float(noise_sd, name="noise_sd", ndim=1)
        if scale.shape != mean.shape or np.any(scale <= 0.0):
            raise ValueError("noise_sd must be positive with one value per observation.")
        retained_basis = _readonly_float(basis, name="basis", ndim=2)
        retained_values = _readonly_float(
            eigenvalues,
            name="eigenvalues",
            ndim=1,
        )
        if retained_basis.shape != (mean.size, retained_values.size):
            raise ValueError(
                "basis must have one row per observation and one column per retained eigenvalue."
            )
        if retained_values.size and (
            np.any(retained_values <= 0.0) or np.any(retained_values[:-1] < retained_values[1:])
        ):
            raise ValueError("eigenvalues must be strictly positive and non-increasing.")
        gram_tolerance = float(512.0 * np.finfo(np.float64).eps * max(1, *retained_basis.shape))
        if not np.allclose(
            retained_basis.T @ retained_basis,
            np.eye(retained_values.size, dtype=np.float64),
            rtol=0.0,
            atol=gram_tolerance,
        ):
            raise ValueError("basis columns must be orthonormal.")
        total = float(total_variance)
        discarded = float(discarded_variance)
        requested = float(requested_retained_variance_fraction)
        tolerance = float(eigenvalue_tolerance)
        if (
            not math.isfinite(total)
            or not math.isfinite(discarded)
            or total < 0.0
            or discarded < 0.0
            or discarded > total + tolerance
        ):
            raise ValueError("total_variance and discarded_variance must be finite and consistent.")
        if not math.isfinite(requested) or not 0.0 < requested <= 1.0:
            raise ValueError("requested_retained_variance_fraction must lie in (0, 1].")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("eigenvalue_tolerance must be finite and non-negative.")
        retained = float(np.sum(retained_values))
        closure_tolerance = max(
            tolerance * max(1, mean.size),
            1024.0 * np.finfo(np.float64).eps * max(1.0, total),
        )
        if abs((retained + discarded) - total) > closure_tolerance:
            raise ValueError("retained and discarded eigenvalue sums do not reconstruct total_variance.")
        actual_fraction = 1.0 if total == 0.0 else retained / total
        object.__setattr__(self, "observation_mean_design", mean)
        object.__setattr__(self, "noise_sd", scale)
        object.__setattr__(self, "basis", retained_basis)
        object.__setattr__(self, "eigenvalues", retained_values)
        object.__setattr__(self, "total_variance", total)
        object.__setattr__(self, "discarded_variance", discarded)
        object.__setattr__(
            self,
            "requested_retained_variance_fraction",
            requested,
        )
        object.__setattr__(
            self,
            "retained_variance_fraction",
            actual_fraction,
        )
        object.__setattr__(self, "eigenvalue_tolerance", tolerance)
        object.__setattr__(
            self,
            "cell_alphas_sha256",
            str(cell_alphas_sha256),
        )
        object.__setattr__(self, "design_sha256", str(design_sha256))
        object.__setattr__(self, "noise_sd_sha256", str(noise_sd_sha256))

    @classmethod
    def from_aggregation(
        cls,
        aggregation: AdditiveDirichletAggregation,
        *,
        retained_variance_fraction: float = 1.0,
        maximum_rank: int | None = None,
    ) -> RootResidualSpectrum:
        """Construct the root spectrum from exact Gamma--Dirichlet moments.

        Args:
            aggregation: Common native aggregation model.  Its existing
                summary basis is ignored; the complete design and noise scales
                define this root spectrum.
            retained_variance_fraction: Required fraction of analytic
                aggregation variance to retain.
            maximum_rank: Optional hard rank budget.  Construction fails if
                this budget cannot meet ``retained_variance_fraction``.

        Returns:
            Frozen value object with initially read-only owned arrays and
            descending retained eigenpairs.  Eigenvalues at or below the
            numerical tolerance can be omitted even when the requested
            fraction is one.

        Raises:
            TypeError: If ``aggregation`` or ``maximum_rank`` has the wrong
                type.
            ValueError: If the requested fraction is invalid, the covariance
                is materially indefinite, or the rank budget is insufficient.
        """
        if not isinstance(aggregation, AdditiveDirichletAggregation):
            raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
        requested = float(retained_variance_fraction)
        if not math.isfinite(requested) or not 0.0 < requested <= 1.0:
            raise ValueError("retained_variance_fraction must lie in (0, 1].")
        if maximum_rank is None:
            rank_budget: int | None = None
        else:
            if isinstance(maximum_rank, bool) or not isinstance(
                maximum_rank,
                Integral,
            ):
                raise TypeError("maximum_rank must be an integer.")
            rank_budget = int(maximum_rank)
            if rank_budget < 0:
                raise ValueError("maximum_rank must be non-negative.")

        alphas = cast(FloatArray, aggregation.cell_alphas.reshape(-1))
        alpha_total = math.fsum(float(value) for value in alphas)
        proportions = alphas / alpha_total
        design = cast(FloatArray, aggregation.design)
        noise = cast(FloatArray, aggregation.noise_sd)
        mean = design @ proportions
        centered = (design - mean[:, np.newaxis]) / noise[:, np.newaxis]
        covariance_factor = centered * np.sqrt(proportions / (alpha_total + 1.0))[np.newaxis, :]
        covariance = covariance_factor @ covariance_factor.T
        covariance = 0.5 * (covariance + covariance.T)
        scale = max(1.0, float(np.max(np.abs(covariance), initial=0.0)))
        eigenvalue_tolerance = float(
            1024.0 * np.finfo(np.float64).eps * max(1, *covariance.shape, alphas.size) * scale
        )
        eigenvalues, eigenvectors = linalg.eigh(
            covariance,
            check_finite=False,
            driver="evr",
        )
        if eigenvalues.size and float(eigenvalues[0]) < -eigenvalue_tolerance:
            raise ValueError("analytic root residual covariance is materially indefinite.")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        all_nonnegative_values = np.asarray(
            eigenvalues[::-1],
            dtype=np.float64,
        )
        positive = eigenvalues > eigenvalue_tolerance
        positive_values = np.asarray(eigenvalues[positive][::-1], dtype=np.float64)
        positive_vectors = np.asarray(
            eigenvectors[:, positive][:, ::-1],
            dtype=np.float64,
        )
        positive_vectors = _canonicalize_eigenvector_signs(positive_vectors)
        total_variance = float(np.sum(all_nonnegative_values))

        if positive_values.size == 0:
            retained_rank = 0
        elif requested == 1.0:
            retained_rank = int(positive_values.size)
        else:
            target = requested * total_variance
            cumulative = np.cumsum(positive_values)
            retained_rank = (
                int(np.searchsorted(cumulative, target)) + 1
                if cumulative.size and target <= float(cumulative[-1])
                else int(positive_values.size)
            )
        if rank_budget is not None and retained_rank > rank_budget:
            raise ValueError("maximum_rank cannot satisfy retained_variance_fraction.")
        retained_values = positive_values[:retained_rank]
        retained_vectors = positive_vectors[:, :retained_rank]
        discarded_variance = max(
            0.0,
            total_variance - float(np.sum(retained_values)),
        )
        return cls(
            mean,
            noise,
            retained_vectors,
            retained_values,
            total_variance=total_variance,
            discarded_variance=discarded_variance,
            requested_retained_variance_fraction=requested,
            eigenvalue_tolerance=eigenvalue_tolerance,
            cell_alphas_sha256=_array_sha256(cast(FloatArray, aggregation.cell_alphas)),
            design_sha256=_array_sha256(design),
            noise_sd_sha256=_array_sha256(noise),
        )

    @property
    def retained_rank(self) -> int:
        """Return the retained analytic residual rank."""
        return int(self.eigenvalues.size)

    @property
    def projection_kl_upper_bound_per_squared_mass(self) -> float:
        """Bound exact-versus-projected mixture KL per squared root mass.

        The comparison uses the same unit Gaussian measurement convolution
        and replaces omitted aggregation-residual coordinates by zero.  It
        excludes finite-bank and Gaussian-complement approximation error.
        """
        return 0.5 * self.discarded_variance

    @property
    def projection_tv_upper_bound_per_absolute_mass(self) -> float:
        """Bound exact-versus-projected mixture TV per absolute root mass.

        This is Pinsker's consequence of
        :attr:`projection_kl_upper_bound_per_squared_mass`; it is not a
        pointwise log-density bound.
        """
        return 0.5 * math.sqrt(self.discarded_variance)


def build_chunked_projected_root_bank(
    aggregation: AdditiveDirichletAggregation,
    spectrum: RootResidualSpectrum,
    *,
    mixture_rank: int,
    sample_count: int,
    sample_chunk_size: int,
    projection_chunk_size: int,
    source_seed: int,
    source_provenance: str,
    cell_ids: ArrayLike | None = None,
) -> ConditionalAllocationMixture:
    """Build a root bank directly in leading spectrum coordinates.

    The complete analytic ``spectrum`` remains authoritative for directions
    beyond ``mixture_rank``.  Native allocation shares are materialized only
    for one sample chunk, projected immediately, and discarded.  This avoids
    the all-at-once ``sample_count * native_cell_count`` allocation array.

    Args:
        aggregation: Common additive native Gamma--Dirichlet model.
        spectrum: Analytic root spectrum from the same native model.
        mixture_rank: Number of leading spectrum coordinates stored.
        sample_count: Power-of-two scrambled-Sobol bank size.
        sample_chunk_size: Power-of-two rows materialized at once, no greater
            than ``sample_count``.
        projection_chunk_size: Fixed power-of-two projection microbatch, no
            greater than ``sample_chunk_size``.
        source_seed: Seed for the frozen scrambled-Sobol catalogue.
        source_provenance: Human-readable source description.
        cell_ids: Optional unique stable scientific native-cell identifiers.

    Returns:
        Immutable one-root projected source bank.

    Raises:
        TypeError: If model, spectrum, rank, or integer controls are invalid.
        ValueError: If native identities disagree or ranks are out of range.
    """
    if not isinstance(aggregation, AdditiveDirichletAggregation):
        raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
    if not isinstance(spectrum, RootResidualSpectrum):
        raise TypeError("spectrum must be a RootResidualSpectrum.")
    if isinstance(mixture_rank, bool) or not isinstance(mixture_rank, Integral):
        raise TypeError("mixture_rank must be an integer.")
    rank = int(mixture_rank)
    if not 0 <= rank <= spectrum.retained_rank:
        raise ValueError("mixture_rank must lie between zero and spectrum rank.")
    if (
        _array_sha256(cast(FloatArray, aggregation.cell_alphas)) != spectrum.cell_alphas_sha256
        or _array_sha256(cast(FloatArray, aggregation.design)) != spectrum.design_sha256
        or _array_sha256(cast(FloatArray, aggregation.noise_sd)) != spectrum.noise_sd_sha256
    ):
        raise ValueError("aggregation native identities do not match spectrum.")

    projected_aggregation = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design,
        aggregation.noise_sd,
        spectrum.basis[:, :rank],
    )
    labels = np.zeros(aggregation.cell_shape, dtype=np.int64)
    return ConditionalAllocationMixture.from_aggregation(
        projected_aggregation,
        labels,
        sample_count=sample_count,
        source_seed=source_seed,
        source_provenance=source_provenance,
        cell_ids=cell_ids,
        construction_method=("scrambled_sobol_balanced_dirichlet_chunked_projected"),
        sample_chunk_size=sample_chunk_size,
        projection_chunk_size=projection_chunk_size,
    )


def _cluster_moments(
    locations: FloatArray,
    labels: IntArray,
    component_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Return exact population moments for non-empty hard clusters."""
    sample_count, dimension = locations.shape
    counts = np.bincount(labels, minlength=component_count).astype(
        np.int64,
        copy=False,
    )
    if counts.shape != (component_count,) or np.any(counts <= 0):
        raise ClusterError("hard clustering produced an empty component")
    weights = counts.astype(np.float64) / float(sample_count)
    means = np.empty((component_count, dimension), dtype=np.float64)
    covariances = np.empty(
        (component_count, dimension, dimension),
        dtype=np.float64,
    )
    for component in range(component_count):
        selected = locations[labels == component]
        mean = np.mean(selected, axis=0)
        difference = selected - mean
        covariance = difference.T @ difference / float(selected.shape[0])
        means[component] = mean
        covariances[component] = 0.5 * (covariance + covariance.T)
    return (
        cast(FloatArray, weights),
        cast(FloatArray, means),
        cast(FloatArray, covariances),
        cast(IntArray, counts),
    )


def _component_order(
    weights: FloatArray,
    means: FloatArray,
) -> IntArray:
    """Return a canonical lexicographic component order."""
    if means.shape[1] == 0:
        return np.arange(weights.size, dtype=np.int64)
    keys: list[FloatArray] = [
        cast(FloatArray, means[:, column]) for column in reversed(range(means.shape[1]))
    ]
    # Weight and original index make exact coincident means deterministic.
    keys.insert(0, -weights)
    keys.insert(0, np.arange(weights.size, dtype=np.float64))
    return np.asarray(np.lexsort(tuple(keys)), dtype=np.int64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CompressedRootMixture:
    """Gaussian-cluster compression of a finite root allocation bank.

    Compression preserves the empirical bank's normalization, mean, and
    covariance in the leading :attr:`mixture_rank` coordinates.  It does not
    make that bank exact for the continuous Dirichlet law.  Coordinates
    between ``mixture_rank`` and the retained spectrum rank use a Gaussian
    approximation matching analytic moments; spectrum directions beyond that
    rank are omitted.

    Attributes:
        spectrum: Analytic root eigenspectrum and observation context.
        weights: Positive compressed component weights summing to one.
        means: Unit-root component means in leading whitened coordinates.
        covariances: Positive-semidefinite unit-root within-cluster
            covariances.
        covariance_eigenvalues: Non-negative component covariance eigenvalues
            cached for batched likelihood evaluation.
        covariance_eigenvectors: Corresponding orthonormal component
            eigenvectors, stored column-wise.
        cluster_counts: Number of equal-weight source locations in each
            component.
        source_sha256: Authenticated source-bank identity.
        source_seed: Source-bank scramble seed.
        source_sample_count: Number of source locations.
        selected_restart: Selected SciPy k-means restart, or ``-1`` for a
            literal no-clustering path.
        restart_inertias: Inertia for each successful/failed fixed restart;
            failed values use the maximum finite float64 sentinel.
        kl_upper_bound: Unit-root KL upper bound for replacing the finite bank
            in the leading coordinates by these cluster Gaussians.  It
            excludes source-bank, projection, and Gaussian-complement errors.
    """

    spectrum: RootResidualSpectrum = field(init=False)
    weights: FloatArray = field(init=False)
    means: FloatArray = field(init=False)
    covariances: FloatArray = field(init=False)
    covariance_eigenvalues: FloatArray = field(init=False)
    covariance_eigenvectors: FloatArray = field(init=False)
    cluster_counts: IntArray = field(init=False)
    source_sha256: str = field(init=False)
    source_seed: int = field(init=False)
    source_sample_count: int = field(init=False)
    selected_restart: int = field(init=False)
    restart_inertias: FloatArray = field(init=False)
    kl_upper_bound: float = field(init=False)

    def __init__(
        self,
        spectrum: RootResidualSpectrum,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
        cluster_counts: ArrayLike,
        *,
        source_sha256: str,
        source_seed: int,
        source_sample_count: int,
        selected_restart: int,
        restart_inertias: ArrayLike,
        kl_upper_bound: float,
    ) -> None:
        """Validate and own one compressed root mixture.

        Args:
            spectrum: Analytic root residual spectrum.
            weights: Positive component weights.
            means: Unit-root component means.
            covariances: Unit-root within-cluster covariances.
            cluster_counts: Equal-weight source counts by component.
            source_sha256: Source artifact identity.
            source_seed: Source scramble seed.
            source_sample_count: Source-bank size.
            selected_restart: Selected restart or ``-1``.
            restart_inertias: Per-restart clustering inertias.
            kl_upper_bound: Leading-coordinate unit-root compression bound.

        Raises:
            TypeError: If ``spectrum`` or integer arrays have wrong types.
            ValueError: If shapes, weights, covariance support, counts,
                restart metadata, or the bound violate the artifact contract.
        """
        if not isinstance(spectrum, RootResidualSpectrum):
            raise TypeError("spectrum must be a RootResidualSpectrum.")
        owned_weights = _readonly_float(weights, name="weights", ndim=1)
        owned_means = _readonly_float(means, name="means", ndim=2)
        owned_covariances = _readonly_float(
            covariances,
            name="covariances",
            ndim=3,
        )
        owned_counts = _readonly_integer(
            cluster_counts,
            name="cluster_counts",
            ndim=1,
        )
        component_count = owned_weights.size
        if component_count == 0:
            raise ValueError("weights must contain at least one component.")
        if (
            owned_means.shape[0] != component_count
            or owned_covariances.shape
            != (
                component_count,
                owned_means.shape[1],
                owned_means.shape[1],
            )
            or owned_counts.shape != (component_count,)
        ):
            raise ValueError("weights, means, covariances, and cluster_counts have incompatible shapes.")
        if (
            owned_means.shape[1] > spectrum.retained_rank
            or np.any(owned_weights <= 0.0)
            or np.any(owned_counts <= 0)
        ):
            raise ValueError(
                "component weights/counts must be positive and mixture rank cannot exceed spectrum rank."
            )
        weight_tolerance = float(64.0 * np.finfo(np.float64).eps * component_count)
        if not math.isclose(
            float(np.sum(owned_weights)),
            1.0,
            rel_tol=0.0,
            abs_tol=weight_tolerance,
        ):
            raise ValueError("component weights must sum to one.")
        covariance_tolerance = float(1024.0 * np.finfo(np.float64).eps * max(1, owned_means.shape[1]))
        for covariance in owned_covariances:
            if not np.allclose(
                covariance,
                covariance.T,
                rtol=0.0,
                atol=covariance_tolerance,
            ):
                raise ValueError("component covariances must be symmetric.")
            if covariance.size:
                minimum = float(
                    linalg.eigvalsh(
                        covariance,
                        check_finite=False,
                        subset_by_index=[0, 0],
                    )[0]
                )
                scale = max(1.0, float(np.max(np.abs(covariance))))
                if minimum < -covariance_tolerance * scale:
                    raise ValueError("component covariances must be positive semidefinite.")
        normalized_source_count = _positive_integer(
            source_sample_count,
            name="source_sample_count",
        )
        if int(np.sum(owned_counts)) != normalized_source_count:
            raise ValueError("cluster_counts must sum to source_sample_count.")
        normalized_seed = _unsigned_seed(source_seed, name="source_seed")
        if (
            not isinstance(selected_restart, Integral)
            or isinstance(selected_restart, bool)
            or int(selected_restart) < -1
        ):
            raise ValueError("selected_restart must be -1 or a non-negative integer.")
        inertias = _readonly_float(
            restart_inertias,
            name="restart_inertias",
            ndim=1,
        )
        if np.any(inertias < 0.0):
            raise ValueError("restart_inertias must be non-negative.")
        selected = int(selected_restart)
        if selected >= inertias.size:
            raise ValueError("selected_restart must index restart_inertias.")
        bound = float(kl_upper_bound)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("kl_upper_bound must be finite and non-negative.")
        covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
            owned_covariances,
        )
        covariance_eigenvalues = np.maximum(
            covariance_eigenvalues,
            0.0,
        )
        for component in range(component_count):
            covariance_eigenvectors[component] = _canonicalize_eigenvector_signs(
                covariance_eigenvectors[component],
            )
        owned_covariance_eigenvalues = _readonly_float(
            covariance_eigenvalues,
            name="covariance_eigenvalues",
            ndim=2,
        )
        owned_covariance_eigenvectors = _readonly_float(
            covariance_eigenvectors,
            name="covariance_eigenvectors",
            ndim=3,
        )
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "weights", owned_weights)
        object.__setattr__(self, "means", owned_means)
        object.__setattr__(self, "covariances", owned_covariances)
        object.__setattr__(
            self,
            "covariance_eigenvalues",
            owned_covariance_eigenvalues,
        )
        object.__setattr__(
            self,
            "covariance_eigenvectors",
            owned_covariance_eigenvectors,
        )
        object.__setattr__(self, "cluster_counts", owned_counts)
        object.__setattr__(self, "source_sha256", str(source_sha256))
        object.__setattr__(self, "source_seed", normalized_seed)
        object.__setattr__(
            self,
            "source_sample_count",
            normalized_source_count,
        )
        object.__setattr__(self, "selected_restart", selected)
        object.__setattr__(self, "restart_inertias", inertias)
        object.__setattr__(self, "kl_upper_bound", bound)

    @classmethod
    def from_source(
        cls,
        source: ConditionalAllocationMixture,
        spectrum: RootResidualSpectrum,
        *,
        mixture_rank: int,
        component_count: int,
        restart_count: int = 3,
        random_seed: int = 731,
        maximum_iterations: int = 100,
    ) -> CompressedRootMixture:
        """Compress one equal-weight root allocation bank.

        Args:
            source: Direct allocation bank expressed in a leading prefix of
                ``spectrum.basis``.  Its rank must cover ``mixture_rank``;
                later spectrum directions use the analytic Gaussian
                complement and need not be stored in the source.
            spectrum: Analytic root spectrum used by the source.
            mixture_rank: Number of leading spectrum directions represented by
                the non-Gaussian mixture.
            component_count: Number of hard clusters.
            restart_count: Number of deterministic k-means++ starts.
            random_seed: Base unsigned seed for restart generators.
            maximum_iterations: SciPy ``kmeans2`` iteration cap.
        Returns:
            Frozen empirical-moment-preserving compressed mixture with
            initially read-only owned arrays.  :attr:`kl_upper_bound` applies
            only to unit-root cluster replacement in the leading coordinates.

        Raises:
            TypeError: If source, spectrum, rank, integer controls, or seeds
                have invalid types.
            ValueError: If source identities, ranks, or controls are invalid.
            RuntimeError: If every fixed clustering restart fails.
        """
        if not isinstance(source, ConditionalAllocationMixture):
            raise TypeError("source must be a ConditionalAllocationMixture.")
        if not isinstance(spectrum, RootResidualSpectrum):
            raise TypeError("spectrum must be a RootResidualSpectrum.")
        if source.region_count != 1:
            raise ValueError("compressed root mixtures require one region.")
        if isinstance(mixture_rank, bool) or not isinstance(
            mixture_rank,
            Integral,
        ):
            raise TypeError("mixture_rank must be an integer.")
        rank = int(mixture_rank)
        if not 0 <= rank <= spectrum.retained_rank:
            raise ValueError("mixture_rank must lie between zero and spectrum rank.")
        if not rank <= source.summary_rank <= spectrum.retained_rank:
            raise ValueError("source summary rank must lie between mixture_rank and spectrum retained rank.")
        identity_tolerance = float(
            512.0 * np.finfo(np.float64).eps * max(1, source.observation_count, spectrum.retained_rank)
        )
        if not np.array_equal(source.noise_sd, spectrum.noise_sd):
            raise ValueError("source noise scales do not match spectrum.")
        expected_source_basis_sha256 = _array_sha256(
            cast(
                FloatArray,
                spectrum.basis[:, : source.summary_rank],
            )
        )
        if source.summary_basis_sha256 != expected_source_basis_sha256:
            raise ValueError("source summary basis is not a leading prefix of spectrum.")
        if not np.allclose(
            source.observation_mean_design[:, 0],
            spectrum.observation_mean_design,
            rtol=0.0,
            atol=identity_tolerance,
        ):
            raise ValueError("source conditional mean does not match spectrum.")
        if (
            source.cell_alphas_sha256 != spectrum.cell_alphas_sha256
            or source.design_sha256 != spectrum.design_sha256
            or source.noise_sd_sha256 != spectrum.noise_sd_sha256
        ):
            raise ValueError("source native identities do not match spectrum.")
        components = _positive_integer(
            component_count,
            name="component_count",
        )
        restarts = _positive_integer(restart_count, name="restart_count")
        iterations = _positive_integer(
            maximum_iterations,
            name="maximum_iterations",
        )
        seed = _unsigned_seed(random_seed, name="random_seed")
        sample_count = source.sample_count
        if components > sample_count:
            raise ValueError("component_count cannot exceed source sample count.")
        locations = np.asarray(
            source.projected_unit_mass_residual_factors[:, :rank, 0],
            dtype=np.float64,
        )

        if rank == 0:
            labels = np.zeros(sample_count, dtype=np.int64)
            weights = np.ones(1, dtype=np.float64)
            means = np.empty((1, 0), dtype=np.float64)
            covariances = np.empty((1, 0, 0), dtype=np.float64)
            counts = np.asarray([sample_count], dtype=np.int64)
            selected_restart = -1
            inertias = np.empty(0, dtype=np.float64)
        elif components == sample_count:
            labels = np.arange(sample_count, dtype=np.int64)
            weights = np.full(
                sample_count,
                1.0 / sample_count,
                dtype=np.float64,
            )
            means = np.array(locations, copy=True)
            covariances = np.zeros(
                (sample_count, rank, rank),
                dtype=np.float64,
            )
            counts = np.ones(sample_count, dtype=np.int64)
            selected_restart = -1
            inertias = np.empty(0, dtype=np.float64)
        elif components == 1:
            labels = np.zeros(sample_count, dtype=np.int64)
            weights, means, covariances, counts = _cluster_moments(
                locations,
                labels,
                1,
            )
            selected_restart = -1
            inertias = np.empty(0, dtype=np.float64)
        else:
            successful: list[tuple[float, int, IntArray]] = []
            restart_inertias = np.full(
                restarts,
                np.finfo(np.float64).max,
                dtype=np.float64,
            )
            for restart in range(restarts):
                restart_seed = np.random.SeedSequence([seed, restart])
                generator = np.random.default_rng(restart_seed)
                try:
                    _, raw_labels = kmeans2(
                        locations,
                        components,
                        iter=iterations,
                        minit="++",
                        missing="raise",
                        check_finite=False,
                        rng=generator,
                    )
                    candidate_labels = np.asarray(
                        raw_labels,
                        dtype=np.int64,
                    )
                    (
                        _,
                        candidate_means,
                        _,
                        _,
                    ) = _cluster_moments(
                        locations,
                        candidate_labels,
                        components,
                    )
                except (ClusterError, FloatingPointError, ValueError):
                    continue
                difference = locations - candidate_means[candidate_labels]
                inertia = float(
                    np.einsum(
                        "sq,sq->",
                        difference,
                        difference,
                        optimize=False,
                    )
                )
                if not math.isfinite(inertia) or inertia < 0.0:
                    continue
                restart_inertias[restart] = inertia
                successful.append(
                    (
                        inertia,
                        restart,
                        candidate_labels,
                    )
                )
            if not successful:
                raise RuntimeError("all deterministic k-means restarts failed.")
            _, selected_restart, labels = min(
                successful,
                key=lambda item: (item[0], item[1]),
            )
            weights, means, covariances, counts = _cluster_moments(
                locations,
                labels,
                components,
            )
            inertias = restart_inertias

        order = _component_order(weights, means)
        weights = np.asarray(weights[order], dtype=np.float64)
        means = np.asarray(means[order], dtype=np.float64)
        covariances = np.asarray(
            covariances[order],
            dtype=np.float64,
        )
        counts = np.asarray(counts[order], dtype=np.int64)
        logdet_terms = np.empty(weights.size, dtype=np.float64)
        identity = np.eye(rank, dtype=np.float64)
        for component, covariance in enumerate(covariances):
            sign, logdet = np.linalg.slogdet(identity + covariance)
            if sign <= 0.0 or not math.isfinite(float(logdet)):
                raise ValueError("compressed component covariance produced an invalid KL bound.")
            logdet_terms[component] = float(logdet)
        kl_upper_bound = 0.5 * float(weights @ logdet_terms)
        return cls(
            spectrum,
            weights,
            means,
            covariances,
            counts,
            source_sha256=source.sha256,
            source_seed=source.source_seed,
            source_sample_count=sample_count,
            selected_restart=selected_restart,
            restart_inertias=inertias,
            kl_upper_bound=kl_upper_bound,
        )

    @property
    def component_count(self) -> int:
        """Return the compressed number of Gaussian components."""
        return int(self.weights.size)

    @property
    def mixture_rank(self) -> int:
        """Return the non-Gaussian retained-coordinate rank."""
        return int(self.means.shape[1])

    @property
    def storage_nbytes(self) -> int:
        """Return bytes owned by the artifact's numerical arrays."""
        return int(
            self.spectrum.observation_mean_design.nbytes
            + self.spectrum.noise_sd.nbytes
            + self.spectrum.basis.nbytes
            + self.spectrum.eigenvalues.nbytes
            + self.weights.nbytes
            + self.means.nbytes
            + self.covariances.nbytes
            + self.covariance_eigenvalues.nbytes
            + self.covariance_eigenvectors.nbytes
            + self.cluster_counts.nbytes
            + self.restart_inertias.nbytes
        )

    def log_likelihood(
        self,
        observation: ArrayLike,
        root_mass: float,
        *,
        offset: ArrayLike = 0.0,
    ) -> float:
        """Evaluate the normalized hybrid root likelihood.

        The leading coordinates use the analytically noise-convolved
        compressed mixture, remaining retained spectrum coordinates use the
        Gaussian moment approximation, and the orthogonal observation space
        uses measurement noise alone.  The returned density is normalized
        with respect to the original observation units.

        Args:
            observation: Finite vector aligned with the source operator.
            root_mass: Finite non-negative retained root mass.
            offset: Scalar or observation-aligned fixed contribution.

        Returns:
            Normalized observation log density.

        Raises:
            TypeError: If ``root_mass`` is not a real scalar.
            ValueError: If inputs are malformed or evaluation is non-finite.
        """
        if isinstance(root_mass, bool) or not isinstance(root_mass, Real):
            raise TypeError("root_mass must be a real scalar.")
        mass = float(root_mass)
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError("root_mass must be finite and non-negative.")
        observed = np.asarray(observation, dtype=np.float64)
        if observed.shape != self.spectrum.observation_mean_design.shape or not np.all(np.isfinite(observed)):
            raise ValueError("observation must be finite with one value per observation.")
        raw_offset = np.asarray(offset, dtype=np.float64)
        if raw_offset.ndim == 0:
            fixed_offset = np.full(
                observed.size,
                float(raw_offset),
                dtype=np.float64,
            )
        else:
            fixed_offset = raw_offset
        if fixed_offset.shape != observed.shape or not np.all(np.isfinite(fixed_offset)):
            raise ValueError("offset must be finite and scalar or observation-aligned.")
        residual = (
            observed - fixed_offset - mass * self.spectrum.observation_mean_design
        ) / self.spectrum.noise_sd
        coordinates = self.spectrum.basis.T @ residual
        orthogonal_squared = float(residual @ residual - coordinates @ coordinates)
        orthogonal_tolerance = float(
            2048.0 * np.finfo(np.float64).eps * max(1, observed.size) * max(1.0, float(residual @ residual))
        )
        if orthogonal_squared < -orthogonal_tolerance:
            raise ValueError("orthogonal residual norm became materially negative.")
        orthogonal_squared = max(0.0, orthogonal_squared)
        retained_rank = self.spectrum.retained_rank
        mixture_rank = self.mixture_rank
        result = -float(np.sum(np.log(self.spectrum.noise_sd)))
        result -= 0.5 * ((observed.size - retained_rank) * _LOG_TWO_PI + orthogonal_squared)

        complement_values = self.spectrum.eigenvalues[mixture_rank:]
        if complement_values.size:
            complement_coordinates = coordinates[mixture_rank:]
            complement_variances = 1.0 + mass * mass * complement_values
            result -= 0.5 * float(
                np.sum(
                    _LOG_TWO_PI
                    + np.log(complement_variances)
                    + complement_coordinates * complement_coordinates / complement_variances
                )
            )

        if mixture_rank:
            displacement = coordinates[np.newaxis, :mixture_rank] - mass * self.means
            rotated = np.einsum(
                "mji,mj->mi",
                self.covariance_eigenvectors,
                displacement,
                optimize=False,
            )
            component_variances = 1.0 + mass * mass * self.covariance_eigenvalues
            component_terms = np.log(self.weights) - 0.5 * (
                mixture_rank * _LOG_TWO_PI
                + np.sum(np.log(component_variances), axis=1)
                + np.sum(
                    rotated * rotated / component_variances,
                    axis=1,
                )
            )
            result += float(cast(Real, logsumexp(component_terms)))
        if not math.isfinite(result):
            raise ValueError("compressed root mixture likelihood is non-finite.")
        return result


def compressed_root_mixture_log_likelihood(
    observation: ArrayLike,
    root_mass: float,
    artifact: CompressedRootMixture,
    *,
    offset: ArrayLike = 0.0,
) -> float:
    """Evaluate a normalized compressed root aggregation likelihood.

    Args:
        observation: Finite observation vector.
        root_mass: Finite non-negative retained root mass.
        artifact: Validated compressed root artifact.
        offset: Scalar or observation-aligned fixed contribution.

    Returns:
        Observation log density in original units.

    Raises:
        TypeError: If ``artifact`` or ``root_mass`` has an invalid type.
        ValueError: If evaluation inputs are malformed or non-finite.
    """
    if not isinstance(artifact, CompressedRootMixture):
        raise TypeError("artifact must be a CompressedRootMixture.")
    return artifact.log_likelihood(
        observation,
        root_mass,
        offset=offset,
    )
