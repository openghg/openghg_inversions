"""Portable conditional mixture densities for aggregation residuals.

The classes in this module represent one fixed-partition approximation to the
conditional observation law obtained after unresolved within-region
allocations are marginalized.  They are deliberately independent of PyTorch,
``sbi``, PyTensor, and PyMC: training may happen elsewhere, while scientific
evaluation uses immutable float64 arrays and an authenticated JSON artifact.

For retained region masses ``m``, total ``T = sum(m)``, error-whitened
residual ``r``, and an orthonormal basis ``Q`` for the exact aggregation-error
image, the implemented density is

```
z = Q.T @ r
v = r - Q @ z

p(y | m) = prod_i noise_sd[i]**(-1) * phi(v)
           * sum_l pi_l(w)
               N(z; T * mu_l(w), I + T**2 * Sigma_l(w)).
```

The orthogonal Gaussian factor and observation-scale Jacobian are part of the
normalization.  There is no additional ``-q log(T)`` after convolution with
measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    PartitionSummaryFactors,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ConditionalResidualImageMDN",
    "RESIDUAL_IMAGE_BASIS_RULE",
    "RESIDUAL_IMAGE_CONTEXT_SCHEMA",
    "ResidualImageContext",
    "conditional_residual_image_mdn_log_likelihood",
]

RESIDUAL_IMAGE_CONTEXT_SCHEMA = "aggregation-conditional-residual-image-context-v2"
RESIDUAL_IMAGE_BASIS_RULE = "stable-id-column-portable-two-pass-mgs-v2"
_MDN_SCHEMA = "aggregation-conditional-residual-image-mdn-v1"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64


def _canonical_json(payload: object) -> str:
    """Return strict, stable JSON suitable for hashing."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _array_sha256(values: FloatArray | IntArray) -> str:
    """Return a dtype-, shape-, and value-sensitive array digest."""
    if np.issubdtype(values.dtype, np.floating):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        dtype = "<f8"
    else:
        canonical = np.ascontiguousarray(values, dtype="<i8")
        dtype = "<i8"
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": dtype,
                "shape": list(values.shape),
            }
        ).encode("ascii")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_float(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    """Return one finite float64 array backed by immutable bytes."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.frombuffer(result.tobytes(order="C"), dtype=np.float64)
    return cast(FloatArray, immutable.reshape(result.shape))


def _readonly_integer(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> IntArray:
    """Return one immutable int64 array without lossy coercion."""
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(f"{name} must be an integer array.")
    if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} values must fit in signed int64.")
    result = np.array(raw, dtype=np.int64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    immutable = np.frombuffer(result.tobytes(order="C"), dtype=np.int64)
    return cast(IntArray, immutable.reshape(result.shape))


def _validated_sha256(value: str, *, name: str) -> str:
    """Return one canonical lower-case SHA-256 string."""
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def _source_provenance(value: str) -> str:
    """Return one non-empty source description."""
    if not isinstance(value, str):
        raise TypeError("source_provenance must be a string.")
    if not value.strip():
        raise ValueError("source_provenance must be non-empty.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _stable_logsumexp(values: FloatArray) -> float:
    """Return log-sum-exp for one non-empty finite vector."""
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _softplus(value: float) -> float:
    """Evaluate softplus without avoidable overflow."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _portable_dot(left: list[float], right: list[float]) -> float:
    """Return one fixed-order binary64 dot product without BLAS dispatch."""
    if len(left) != len(right):
        raise ValueError("portable dot-product inputs must have the same length.")
    return math.fsum(left[index] * right[index] for index in range(len(left)))


def _portable_norm(values: list[float]) -> float:
    """Return one fixed-order Euclidean norm without BLAS dispatch."""
    squared = _portable_dot(values, values)
    if squared < 0.0:
        raise FloatingPointError("portable squared norm must be non-negative.")
    return math.sqrt(squared)


def _canonical_residual_basis(
    whitened_centered_design: FloatArray,
) -> tuple[FloatArray, float]:
    """Return a portable canonical basis for the supplied column image.

    SVD is used only to determine rank behind a wide ambiguity gate. Its
    hardware-dependent singular vectors never enter the retained chart.
    Canonically ordered design columns are instead orthonormalized with
    fixed-order scalar binary64 operations and :func:`math.fsum`. This avoids
    BLAS/LAPACK kernel selection changing authenticated context bytes.
    """
    observation_count, column_count = whitened_centered_design.shape
    if column_count == 0:
        return _readonly_float(
            np.empty((observation_count, 0), dtype=np.float64),
            name="residual_basis",
            ndim=2,
        ), 0.0
    singular_values = np.linalg.svd(
        whitened_centered_design,
        full_matrices=False,
        compute_uv=False,
    )
    squared_image_scale = math.fsum(
        float(whitened_centered_design[row, column]) ** 2
        for row in range(observation_count)
        for column in range(column_count)
    )
    image_scale = math.sqrt(squared_image_scale)
    tolerance = float(
        256.0 * np.finfo(np.float64).eps * max(1, observation_count, column_count) * image_scale
    )
    if image_scale == 0.0:
        rank = 0
    else:
        ambiguous = (singular_values >= tolerance / 100.0) & (singular_values <= 100.0 * tolerance)
        if np.any(ambiguous):
            raise ValueError("residual-image rank is numerically ambiguous under the frozen rule.")
        rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        return _readonly_float(
            np.empty((observation_count, 0), dtype=np.float64),
            name="residual_basis",
            ndim=2,
        ), tolerance

    accepted: list[list[float]] = []
    for column in range(column_count):
        candidate = [float(whitened_centered_design[row, column]) for row in range(observation_count)]
        for _ in range(2):
            for previous in accepted:
                coefficient = _portable_dot(previous, candidate)
                candidate = [candidate[row] - previous[row] * coefficient for row in range(observation_count)]
        norm = _portable_norm(candidate)
        if tolerance / 100.0 <= norm <= 100.0 * tolerance:
            raise ValueError("residual-image pivot selection is numerically ambiguous under the frozen rule.")
        if norm < tolerance / 100.0:
            continue
        candidate = [value / norm for value in candidate]
        pivot = max(range(observation_count), key=lambda row: abs(candidate[row]))
        if candidate[pivot] < 0.0:
            candidate = [-value for value in candidate]
        accepted.append(candidate)
        if len(accepted) == rank:
            break
    if len(accepted) != rank:
        raise ValueError("failed to construct the canonical residual-image basis.")
    basis = np.asarray(accepted, dtype=np.float64).T
    orthonormality_error = max(
        abs(_portable_dot(accepted[left_index], accepted[right_index]) - float(left_index == right_index))
        for left_index in range(rank)
        for right_index in range(rank)
    )
    span_error = 0.0
    for column in range(column_count):
        source = [float(whitened_centered_design[row, column]) for row in range(observation_count)]
        coefficients = [_portable_dot(vector, source) for vector in accepted]
        for row in range(observation_count):
            reconstructed = math.fsum(accepted[index][row] * coefficients[index] for index in range(rank))
            span_error = max(span_error, abs(source[row] - reconstructed))
    scale = max(1.0, float(np.max(np.abs(whitened_centered_design))))
    if orthonormality_error > 1.0e-12 or span_error > 1.0e-12 * scale:
        raise ValueError("canonical residual-image basis failed its span audit.")
    return _readonly_float(basis, name="residual_basis", ndim=2), tolerance


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResidualImageContext:
    """Authenticated scientific context for one fixed labelled partition.

    Regions are ordered canonically by the sorted stable-cell-ID signature of
    their members.  Consequently, masses supplied to an evaluator must follow
    :attr:`labels`, not necessarily the incidental integer labels passed to
    :meth:`from_aggregation`.  Use :meth:`canonicalize_masses` when masses are
    still aligned with an external labelling.
    """

    observation_mean_design: FloatArray = field(init=False)
    noise_sd: FloatArray = field(init=False)
    residual_basis: FloatArray = field(init=False)
    labels: IntArray = field(init=False)
    cell_ids: IntArray = field(init=False)
    alpha_totals: FloatArray = field(init=False)
    rank_tolerance: float = field(init=False)
    rank_rule: str = field(init=False)
    source_provenance: str = field(init=False)
    cell_alphas_sha256: str = field(init=False)
    design_sha256: str = field(init=False)
    noise_sd_sha256: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        observation_mean_design: ArrayLike,
        noise_sd: ArrayLike,
        residual_basis: ArrayLike,
        labels: ArrayLike,
        cell_ids: ArrayLike,
        alpha_totals: ArrayLike,
        *,
        rank_tolerance: float,
        rank_rule: str,
        source_provenance: str,
        cell_alphas_sha256: str,
        design_sha256: str,
        noise_sd_sha256: str,
    ) -> None:
        """Validate, own, and fingerprint a residual-image context."""
        mean_design = _readonly_float(
            observation_mean_design,
            name="observation_mean_design",
            ndim=2,
        )
        if mean_design.shape[0] == 0 or mean_design.shape[1] == 0:
            raise ValueError("observation_mean_design must be non-empty.")
        scale = _readonly_float(noise_sd, name="noise_sd", ndim=1)
        if scale.shape != (mean_design.shape[0],) or np.any(scale <= 0.0):
            raise ValueError("noise_sd must be positive with one value per observation.")
        basis = _readonly_float(
            residual_basis,
            name="residual_basis",
            ndim=2,
        )
        if basis.shape[0] != mean_design.shape[0]:
            raise ValueError("residual_basis must have one row per observation.")
        basis_tolerance = float(256.0 * np.finfo(np.float64).eps * max(1, *basis.shape))
        if not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1], dtype=np.float64),
            rtol=0.0,
            atol=basis_tolerance,
        ):
            raise ValueError("residual_basis columns must be orthonormal.")
        owned_labels = _readonly_integer(labels, name="labels")
        if owned_labels.ndim == 0 or owned_labels.size == 0:
            raise ValueError("labels must be a non-empty native-cell array.")
        region_count = mean_design.shape[1]
        if not np.array_equal(
            np.unique(owned_labels),
            np.arange(region_count, dtype=np.int64),
        ):
            raise ValueError("labels must use every contiguous region identifier.")
        owned_ids = _readonly_integer(cell_ids, name="cell_ids")
        if owned_ids.shape != owned_labels.shape:
            raise ValueError("cell_ids must have the same shape as labels.")
        if np.unique(owned_ids).size != owned_ids.size:
            raise ValueError("cell_ids must be unique.")
        totals = _readonly_float(alpha_totals, name="alpha_totals", ndim=1)
        if totals.shape != (region_count,) or np.any(totals <= 0.0):
            raise ValueError("alpha_totals must have one positive value per region.")
        normalized_tolerance = float(rank_tolerance)
        if not np.isfinite(normalized_tolerance) or normalized_tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and non-negative.")
        if rank_rule != RESIDUAL_IMAGE_BASIS_RULE:
            raise ValueError(f"rank_rule must be {RESIDUAL_IMAGE_BASIS_RULE!r}.")
        provenance = _source_provenance(source_provenance)
        alpha_identity = _validated_sha256(
            cell_alphas_sha256,
            name="cell_alphas_sha256",
        )
        design_identity = _validated_sha256(
            design_sha256,
            name="design_sha256",
        )
        noise_identity = _validated_sha256(
            noise_sd_sha256,
            name="noise_sd_sha256",
        )
        if _array_sha256(scale) != noise_identity:
            raise ValueError("noise_sd does not match noise_sd_sha256.")

        object.__setattr__(self, "observation_mean_design", mean_design)
        object.__setattr__(self, "noise_sd", scale)
        object.__setattr__(self, "residual_basis", basis)
        object.__setattr__(self, "labels", owned_labels)
        object.__setattr__(self, "cell_ids", owned_ids)
        object.__setattr__(self, "alpha_totals", totals)
        object.__setattr__(self, "rank_tolerance", normalized_tolerance)
        object.__setattr__(self, "rank_rule", rank_rule)
        object.__setattr__(self, "source_provenance", provenance)
        object.__setattr__(self, "cell_alphas_sha256", alpha_identity)
        object.__setattr__(self, "design_sha256", design_identity)
        object.__setattr__(self, "noise_sd_sha256", noise_identity)
        object.__setattr__(self, "artifact_sha256", _sha256_text(self.to_json()))

    @classmethod
    def from_aggregation(
        cls,
        aggregation: AdditiveDirichletAggregation,
        partition: ArrayLike | PartitionSummaryFactors,
        cell_ids: ArrayLike,
        *,
        source_provenance: str,
    ) -> ResidualImageContext:
        """Construct the exact residual-image context for one partition."""
        if not isinstance(aggregation, AdditiveDirichletAggregation):
            raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
        if isinstance(partition, PartitionSummaryFactors):
            supplied = partition
            rebuilt = aggregation.partition_factors(supplied.labels)
            for name in (
                "labels",
                "alpha_totals",
                "observation_mean_design",
                "summary_mean_design",
                "summary_covariance_factors",
            ):
                if not np.array_equal(
                    getattr(supplied, name),
                    getattr(rebuilt, name),
                ):
                    raise ValueError("partition factors do not exactly match the supplied aggregation.")
            factors = supplied
        else:
            factors = aggregation.partition_factors(partition)
        owned_ids = _readonly_integer(cell_ids, name="cell_ids")
        if owned_ids.shape != aggregation.cell_shape:
            raise ValueError("cell_ids must have the same shape as cell_alphas.")
        if np.unique(owned_ids).size != owned_ids.size:
            raise ValueError("cell_ids must be unique.")

        flat_labels = factors.labels.reshape(-1)
        flat_ids = owned_ids.reshape(-1)
        region_signatures = [
            tuple(sorted(int(value) for value in flat_ids[np.flatnonzero(flat_labels == region)]))
            for region in range(factors.region_count)
        ]
        canonical_original_regions = np.asarray(
            sorted(
                range(factors.region_count),
                key=lambda region: region_signatures[region],
            ),
            dtype=np.int64,
        )
        original_to_canonical = np.empty(
            factors.region_count,
            dtype=np.int64,
        )
        original_to_canonical[canonical_original_regions] = np.arange(
            factors.region_count,
            dtype=np.int64,
        )
        canonical_labels = original_to_canonical[flat_labels].reshape(factors.labels.shape)
        flat_alphas = aggregation.cell_alphas.reshape(-1)
        design = aggregation.design
        canonical_means: list[list[float]] = []
        canonical_totals: list[float] = []
        centered_columns: list[FloatArray] = []
        for original_region in canonical_original_regions:
            selected = np.flatnonzero(flat_labels == original_region)
            selected = selected[np.argsort(flat_ids[selected], kind="stable")]
            alpha_total = math.fsum(float(flat_alphas[index]) for index in selected)
            canonical_totals.append(alpha_total)
            mean = np.asarray(
                [
                    float(design[row, selected[0]])
                    + math.fsum(
                        (float(design[row, index]) - float(design[row, selected[0]]))
                        * float(flat_alphas[index])
                        for index in selected
                    )
                    / alpha_total
                    for row in range(design.shape[0])
                ],
                dtype=np.float64,
            )
            canonical_means.append(mean.tolist())
            centered = (aggregation.design[:, selected] - mean[:, np.newaxis]) / aggregation.noise_sd[
                :, np.newaxis
            ]
            centered_columns.append(cast(FloatArray, centered))
        canonical_mean_design = np.asarray(canonical_means, dtype=np.float64).T
        canonical_alpha_totals = np.asarray(canonical_totals, dtype=np.float64)
        whitened_centered = cast(
            FloatArray,
            np.concatenate(centered_columns, axis=1),
        )
        basis, rank_tolerance = _canonical_residual_basis(whitened_centered)
        alphas = cast(FloatArray, aggregation.cell_alphas)
        design = cast(FloatArray, aggregation.design)
        noise = cast(FloatArray, aggregation.noise_sd)
        return cls(
            canonical_mean_design,
            noise,
            basis,
            canonical_labels,
            owned_ids,
            canonical_alpha_totals,
            rank_tolerance=rank_tolerance,
            rank_rule=RESIDUAL_IMAGE_BASIS_RULE,
            source_provenance=source_provenance,
            cell_alphas_sha256=_array_sha256(alphas),
            design_sha256=_array_sha256(design),
            noise_sd_sha256=_array_sha256(noise),
        )

    @property
    def observation_count(self) -> int:
        """Return the observation dimension."""
        return int(self.observation_mean_design.shape[0])

    @property
    def region_count(self) -> int:
        """Return the number of retained regions."""
        return int(self.observation_mean_design.shape[1])

    @property
    def conditioner_dimension(self) -> int:
        """Return the additive-log-ratio conditioner dimension."""
        return self.region_count - 1

    @property
    def residual_rank(self) -> int:
        """Return the exact aggregation-residual image rank."""
        return int(self.residual_basis.shape[1])

    def canonicalize_masses(
        self,
        masses: ArrayLike,
        source_labels: ArrayLike,
    ) -> FloatArray:
        """Reorder external labelled masses into canonical context order.

        Args:
            masses: One finite strictly positive mass per contiguous source
                label, ordered by that source label.
            source_labels: External contiguous labels aligned cell-for-cell
                with :attr:`labels`.

        Returns:
            Read-only masses ordered by this context's canonical region IDs.

        Raises:
            TypeError: If labels are not an integer array.
            ValueError: If labels, partitions, or masses are malformed or the
                source partition differs from this context.
        """
        external_labels = _readonly_integer(
            source_labels,
            name="source_labels",
        )
        if external_labels.shape != self.labels.shape:
            raise ValueError("source_labels must have the same native-cell shape as context labels.")
        if not np.array_equal(
            np.unique(external_labels),
            np.arange(self.region_count, dtype=np.int64),
        ):
            raise ValueError("source_labels must use every contiguous region identifier.")
        external_masses = np.asarray(masses, dtype=np.float64)
        if (
            external_masses.shape != (self.region_count,)
            or not np.all(np.isfinite(external_masses))
            or np.any(external_masses <= 0.0)
        ):
            raise ValueError("masses must contain one finite strictly positive value per source region.")
        result = np.empty(self.region_count, dtype=np.float64)
        used_canonical = np.zeros(self.region_count, dtype=np.bool_)
        for source_region in range(self.region_count):
            selected = external_labels == source_region
            canonical_regions = np.unique(self.labels[selected])
            if canonical_regions.size != 1:
                raise ValueError("source_labels do not describe the same partition as the context.")
            canonical_region = int(canonical_regions[0])
            if used_canonical[canonical_region]:
                raise ValueError("source_labels do not map bijectively to context regions.")
            used_canonical[canonical_region] = True
            result[canonical_region] = external_masses[source_region]
        if not np.all(used_canonical):
            raise ValueError("source_labels do not map bijectively to context regions.")
        return _readonly_float(
            result,
            name="canonical_masses",
            ndim=1,
        )

    @property
    def sha256(self) -> str:
        """Return the artifact digest under the conventional short name."""
        return self.artifact_sha256

    @property
    def payload(self) -> dict[str, object]:
        """Return the strict JSON-compatible artifact payload."""
        return {
            "schema": RESIDUAL_IMAGE_CONTEXT_SCHEMA,
            "observation_mean_design": self.observation_mean_design.tolist(),
            "noise_sd": self.noise_sd.tolist(),
            "residual_basis": self.residual_basis.tolist(),
            "labels": self.labels.tolist(),
            "cell_ids": self.cell_ids.tolist(),
            "alpha_totals": self.alpha_totals.tolist(),
            "rank_tolerance": self.rank_tolerance,
            "rank_rule": self.rank_rule,
            "source_provenance": self.source_provenance,
            "cell_alphas_sha256": self.cell_alphas_sha256,
            "design_sha256": self.design_sha256,
            "noise_sd_sha256": self.noise_sd_sha256,
        }

    def to_json(self) -> str:
        """Serialize the complete context as canonical JSON."""
        return _canonical_json(self.payload)

    @classmethod
    def from_json(
        cls,
        serialized: str,
        *,
        expected_sha256: str,
    ) -> ResidualImageContext:
        """Reconstruct one context after authenticating canonical JSON bytes."""
        if not isinstance(serialized, str):
            raise TypeError("serialized context must be a string.")
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if _sha256_text(serialized) != expected:
            raise ValueError("context SHA-256 fingerprint does not match.")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized context is not valid JSON.") from error
        if not isinstance(payload, dict) or payload.get("schema") != RESIDUAL_IMAGE_CONTEXT_SCHEMA:
            raise ValueError("serialized context has an unexpected schema.")
        required = {
            "schema",
            "observation_mean_design",
            "noise_sd",
            "residual_basis",
            "labels",
            "cell_ids",
            "alpha_totals",
            "rank_tolerance",
            "rank_rule",
            "source_provenance",
            "cell_alphas_sha256",
            "design_sha256",
            "noise_sd_sha256",
        }
        if set(payload) != required:
            raise ValueError("serialized context fields do not match the schema.")
        result = cls(
            payload["observation_mean_design"],
            payload["noise_sd"],
            payload["residual_basis"],
            payload["labels"],
            payload["cell_ids"],
            payload["alpha_totals"],
            rank_tolerance=payload["rank_tolerance"],
            rank_rule=payload["rank_rule"],
            source_provenance=payload["source_provenance"],
            cell_alphas_sha256=payload["cell_alphas_sha256"],
            design_sha256=payload["design_sha256"],
            noise_sd_sha256=payload["noise_sd_sha256"],
        )
        if result.to_json() != serialized:
            raise ValueError("serialized context must use canonical JSON.")
        if result.artifact_sha256 != expected:  # pragma: no cover - implied above
            raise ValueError("context SHA-256 fingerprint does not match.")
        return result


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConditionalResidualImageMDN:
    """Portable float64 mixture-density network for one fixed context."""

    context: ResidualImageContext = field(init=False)
    hidden_weight_1: FloatArray = field(init=False)
    hidden_bias_1: FloatArray = field(init=False)
    hidden_weight_2: FloatArray = field(init=False)
    hidden_bias_2: FloatArray = field(init=False)
    output_weight: FloatArray = field(init=False)
    output_bias: FloatArray = field(init=False)
    input_center: FloatArray = field(init=False)
    input_scale: FloatArray = field(init=False)
    component_count: int = field(init=False)
    cholesky_diagonal_floor: float = field(init=False)
    source_provenance: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        context: ResidualImageContext,
        hidden_weight_1: ArrayLike,
        hidden_bias_1: ArrayLike,
        hidden_weight_2: ArrayLike,
        hidden_bias_2: ArrayLike,
        output_weight: ArrayLike,
        output_bias: ArrayLike,
        *,
        component_count: int,
        cholesky_diagonal_floor: float,
        input_center: ArrayLike | None = None,
        input_scale: ArrayLike | None = None,
        source_provenance: str,
    ) -> None:
        """Validate, own, and fingerprint network and mixture parameters."""
        if not isinstance(context, ResidualImageContext):
            raise TypeError("context must be a ResidualImageContext.")
        components = _positive_integer(
            component_count,
            name="component_count",
        )
        floor = float(cholesky_diagonal_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("cholesky_diagonal_floor must be finite and positive.")
        weight_1 = _readonly_float(
            hidden_weight_1,
            name="hidden_weight_1",
            ndim=2,
        )
        bias_1 = _readonly_float(
            hidden_bias_1,
            name="hidden_bias_1",
            ndim=1,
        )
        if weight_1.shape[0] == 0 or weight_1.shape != (
            bias_1.size,
            context.conditioner_dimension,
        ):
            raise ValueError("hidden_weight_1 must have shape (first_hidden_size, conditioner_dimension).")
        weight_2 = _readonly_float(
            hidden_weight_2,
            name="hidden_weight_2",
            ndim=2,
        )
        bias_2 = _readonly_float(
            hidden_bias_2,
            name="hidden_bias_2",
            ndim=1,
        )
        if weight_2.shape[0] == 0 or weight_2.shape != (
            bias_2.size,
            bias_1.size,
        ):
            raise ValueError("hidden_weight_2 must have shape (second_hidden_size, first_hidden_size).")
        output = _readonly_float(
            output_weight,
            name="output_weight",
            ndim=2,
        )
        output_offset = _readonly_float(
            output_bias,
            name="output_bias",
            ndim=1,
        )
        rank = context.residual_rank
        triangle_size = rank * (rank + 1) // 2
        expected_output_size = components * (1 + rank + triangle_size)
        if output.shape != (expected_output_size, bias_2.size):
            raise ValueError("output_weight has the wrong mixture-output or hidden dimension.")
        if output_offset.shape != (expected_output_size,):
            raise ValueError("output_bias has the wrong mixture-output dimension.")
        if input_center is None:
            center = _readonly_float(
                np.zeros(context.conditioner_dimension, dtype=np.float64),
                name="input_center",
                ndim=1,
            )
        else:
            center = _readonly_float(
                input_center,
                name="input_center",
                ndim=1,
            )
        if input_scale is None:
            scale = _readonly_float(
                np.ones(context.conditioner_dimension, dtype=np.float64),
                name="input_scale",
                ndim=1,
            )
        else:
            scale = _readonly_float(
                input_scale,
                name="input_scale",
                ndim=1,
            )
        if center.shape != (context.conditioner_dimension,):
            raise ValueError("input_center must match conditioner_dimension.")
        if scale.shape != (context.conditioner_dimension,) or np.any(scale <= 0.0):
            raise ValueError("input_scale must be positive and match conditioner_dimension.")
        provenance = _source_provenance(source_provenance)

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "hidden_weight_1", weight_1)
        object.__setattr__(self, "hidden_bias_1", bias_1)
        object.__setattr__(self, "hidden_weight_2", weight_2)
        object.__setattr__(self, "hidden_bias_2", bias_2)
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(self, "output_bias", output_offset)
        object.__setattr__(self, "input_center", center)
        object.__setattr__(self, "input_scale", scale)
        object.__setattr__(self, "component_count", components)
        object.__setattr__(self, "cholesky_diagonal_floor", floor)
        object.__setattr__(self, "source_provenance", provenance)
        object.__setattr__(self, "artifact_sha256", _sha256_text(self.to_json()))

    @property
    def input_dimension(self) -> int:
        """Return the network conditioning dimension."""
        return self.context.conditioner_dimension

    @property
    def residual_rank(self) -> int:
        """Return the residual-image rank."""
        return self.context.residual_rank

    @property
    def region_count(self) -> int:
        """Return the number of retained masses."""
        return self.context.region_count

    @property
    def sha256(self) -> str:
        """Return the artifact digest under the conventional short name."""
        return self.artifact_sha256

    def _raw_output(self, masses: FloatArray) -> FloatArray:
        """Evaluate the two-layer float64 tanh network."""
        conditioner = np.log(masses[:-1]) - math.log(float(masses[-1]))
        standardized = (conditioner - self.input_center) / self.input_scale
        hidden_1 = np.tanh(self.hidden_weight_1 @ standardized + self.hidden_bias_1)
        hidden_2 = np.tanh(self.hidden_weight_2 @ hidden_1 + self.hidden_bias_2)
        result = self.output_weight @ hidden_2 + self.output_bias
        if not np.all(np.isfinite(result)):
            raise ValueError("conditional MDN network output is non-finite.")
        return cast(FloatArray, result)

    def _components(
        self,
        masses: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return log weights, means, and lower Cholesky factors."""
        raw = self._raw_output(masses)
        components = self.component_count
        rank = self.residual_rank
        triangle_size = rank * (rank + 1) // 2
        logits = raw[:components]
        log_weights = logits - _stable_logsumexp(cast(FloatArray, logits))
        means_start = components
        means_stop = means_start + components * rank
        means = raw[means_start:means_stop].reshape(components, rank)
        packed = raw[means_stop:].reshape(components, triangle_size)
        factors = np.zeros((components, rank, rank), dtype=np.float64)
        for component in range(components):
            cursor = 0
            for row in range(rank):
                for column in range(row + 1):
                    value = float(packed[component, cursor])
                    cursor += 1
                    if row == column:
                        value = self.cholesky_diagonal_floor + _softplus(value)
                    factors[component, row, column] = value
        return (
            cast(FloatArray, log_weights),
            cast(FloatArray, means),
            cast(FloatArray, factors),
        )

    def log_likelihood(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> float:
        """Evaluate the normalized conditional observation log density.

        ``masses`` must follow the canonical region order represented by
        ``context.labels``.  External label order can be converted with
        :meth:`ResidualImageContext.canonicalize_masses`.
        """
        observed = np.asarray(observation, dtype=np.float64)
        if observed.shape != (self.context.observation_count,) or not np.all(np.isfinite(observed)):
            raise ValueError("observation must be finite with one value per context observation.")
        retained = np.asarray(masses, dtype=np.float64)
        if (
            retained.shape != (self.region_count,)
            or not np.all(np.isfinite(retained))
            or np.any(retained <= 0.0)
        ):
            raise ValueError("masses must contain one finite strictly positive value per region.")
        raw_offset = np.asarray(offset, dtype=np.float64)
        if raw_offset.ndim == 0:
            fixed_offset = np.full(
                self.context.observation_count,
                float(raw_offset),
                dtype=np.float64,
            )
        else:
            fixed_offset = raw_offset
        if fixed_offset.shape != observed.shape or not np.all(np.isfinite(fixed_offset)):
            raise ValueError("offset must be finite and scalar or aligned with observation.")
        total = float(np.sum(retained))
        if not np.isfinite(total):
            raise ValueError("mass total must be finite.")
        residual = (
            observed - fixed_offset - self.context.observation_mean_design @ retained
        ) / self.context.noise_sd
        coordinates = self.context.residual_basis.T @ residual
        orthogonal = residual - self.context.residual_basis @ coordinates
        orthogonal_log_density = -0.5 * (
            (self.context.observation_count - self.residual_rank) * _LOG_TWO_PI
            + float(orthogonal @ orthogonal)
        )
        log_weights, means, factors = self._components(cast(FloatArray, retained))
        component_terms = np.empty(self.component_count, dtype=np.float64)
        identity = np.eye(self.residual_rank, dtype=np.float64)
        for component in range(self.component_count):
            factor = factors[component]
            covariance = identity + total * total * (factor @ factor.T)
            try:
                cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:  # pragma: no cover
                raise ValueError("convolved component covariance is not positive definite.") from error
            displacement = coordinates - total * means[component]
            solved = np.linalg.solve(cholesky, displacement)
            component_terms[component] = log_weights[component] - 0.5 * (
                self.residual_rank * _LOG_TWO_PI
                + 2.0 * float(np.sum(np.log(np.diag(cholesky))))
                + float(solved @ solved)
            )
        result = (
            -float(np.sum(np.log(self.context.noise_sd)))
            + orthogonal_log_density
            + _stable_logsumexp(component_terms)
        )
        if not np.isfinite(result):
            raise ValueError("conditional residual-image log density is non-finite.")
        return result

    @property
    def payload(self) -> dict[str, object]:
        """Return the strict JSON-compatible fitted-artifact payload."""
        return {
            "schema": _MDN_SCHEMA,
            "context": self.context.payload,
            "context_sha256": self.context.artifact_sha256,
            "hidden_weight_1": self.hidden_weight_1.tolist(),
            "hidden_bias_1": self.hidden_bias_1.tolist(),
            "hidden_weight_2": self.hidden_weight_2.tolist(),
            "hidden_bias_2": self.hidden_bias_2.tolist(),
            "output_weight": self.output_weight.tolist(),
            "output_bias": self.output_bias.tolist(),
            "input_center": self.input_center.tolist(),
            "input_scale": self.input_scale.tolist(),
            "component_count": self.component_count,
            "cholesky_diagonal_floor": self.cholesky_diagonal_floor,
            "source_provenance": self.source_provenance,
        }

    def to_json(self) -> str:
        """Serialize the complete fitted artifact as canonical JSON."""
        return _canonical_json(self.payload)

    @classmethod
    def from_json(
        cls,
        serialized: str,
        *,
        expected_sha256: str,
    ) -> ConditionalResidualImageMDN:
        """Reconstruct a fitted artifact after authenticating canonical JSON."""
        if not isinstance(serialized, str):
            raise TypeError("serialized MDN must be a string.")
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if _sha256_text(serialized) != expected:
            raise ValueError("MDN SHA-256 fingerprint does not match.")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized MDN is not valid JSON.") from error
        if not isinstance(payload, dict) or payload.get("schema") != _MDN_SCHEMA:
            raise ValueError("serialized MDN has an unexpected schema.")
        required = {
            "schema",
            "context",
            "context_sha256",
            "hidden_weight_1",
            "hidden_bias_1",
            "hidden_weight_2",
            "hidden_bias_2",
            "output_weight",
            "output_bias",
            "input_center",
            "input_scale",
            "component_count",
            "cholesky_diagonal_floor",
            "source_provenance",
        }
        if set(payload) != required:
            raise ValueError("serialized MDN fields do not match the schema.")
        context = ResidualImageContext.from_json(
            _canonical_json(payload["context"]),
            expected_sha256=payload["context_sha256"],
        )
        result = cls(
            context,
            payload["hidden_weight_1"],
            payload["hidden_bias_1"],
            payload["hidden_weight_2"],
            payload["hidden_bias_2"],
            payload["output_weight"],
            payload["output_bias"],
            component_count=payload["component_count"],
            cholesky_diagonal_floor=payload["cholesky_diagonal_floor"],
            input_center=payload["input_center"],
            input_scale=payload["input_scale"],
            source_provenance=payload["source_provenance"],
        )
        if result.to_json() != serialized:
            raise ValueError("serialized MDN must use canonical JSON.")
        if result.artifact_sha256 != expected:  # pragma: no cover - implied above
            raise ValueError("MDN SHA-256 fingerprint does not match.")
        return result


def conditional_residual_image_mdn_log_likelihood(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalResidualImageMDN,
    *,
    offset: ArrayLike = 0.0,
) -> float:
    """Evaluate a portable conditional residual-image fitted artifact."""
    if not isinstance(artifact, ConditionalResidualImageMDN):
        raise TypeError("artifact must be a ConditionalResidualImageMDN.")
    return artifact.log_likelihood(
        observation,
        masses,
        offset=offset,
    )
