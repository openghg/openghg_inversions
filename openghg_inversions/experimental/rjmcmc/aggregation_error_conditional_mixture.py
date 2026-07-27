"""Frozen conditional-allocation mixtures for aggregation error.

This module provides a deterministic finite-mixture baseline between exact
small-problem quadrature and a learned neural likelihood.  For one fixed,
labelled partition it draws a frozen bank of within-region Dirichlet
allocations from the common additive native concentration field.  The PCG64
construction uses independent draws; the scrambled-Sobol construction uses
randomized low-discrepancy components whose rows are not independent.  The
resulting projected unit-mass aggregation residuals are stored in an
immutable, canonical JSON artifact.

Let ``A`` contain the conditional observation mean per unit retained region
mass, and let ``F[s, :, j]`` be the projected aggregation residual generated
by frozen allocation sample ``s`` for one unit of mass in region ``j``.  For
retained masses ``m``, independent measurement covariance
``D = diag(noise_sd**2)``, and a fixed orthonormal summary basis ``B`` in
error-whitened observation space, the implemented density is

```
r = D**(-1/2) @ (y - offset - A @ m)
z = B.T @ r
mu_s(m) = F[s] @ m

p(y | m) ~= prod_i noise_sd[i]**(-1)
             * phi_{n-q}(r - B @ z)
             * mean_s phi_q(z; mu_s(m), I_q).
```

Every finite-bank density is normalized in observation space.  The Gaussian
complement outside ``B`` is nevertheless an explicit approximation unless
the summary basis spans every aggregation-error direction.  Consequently
this object is a conditional likelihood approximation for a fixed
partition, not raw conditional model evidence for comparing partitions or
dimensions.

Artifact construction is the only operation that uses randomness.  The
legacy/default construction uses seeded PCG64 streams keyed by stable
native-cell identifiers.  An explicit quasi-Monte Carlo construction uses a
stable-ID balanced Dirichlet inverse and a joint scrambled Sobol net within
each canonical dimension block.  Catalogues above SciPy's Sobol dimension
limit use independently scrambled blocks, whose combined discrepancy is an
empirical property.  Log-density and gradient evaluation never draw and are
exactly replayable from the serialized artifact.

A version-three single-root construction traverses the same frozen Sobol
catalogue in sample chunks and projects each allocation immediately.  It
stores only the requested summary coordinates rather than an
``samples x native_cells`` allocation array.  A separate fixed projection
microbatch makes the projected numerical result independent of the chosen
memory chunk.
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
import scipy
from scipy import special
from scipy.stats import qmc

from .aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    PartitionSummaryFactors,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ConditionalAllocationMixture",
    "conditional_allocation_mixture_log_likelihood",
    "conditional_allocation_mixture_log_likelihood_and_gradient",
]

_ARTIFACT_SCHEMA_V1 = "aggregation-conditional-allocation-mixture-v1"
_ARTIFACT_SCHEMA_V2 = "aggregation-conditional-allocation-mixture-v2"
_ARTIFACT_SCHEMA_V3 = "aggregation-conditional-allocation-mixture-v3"
_PCG64_CONSTRUCTION_METHOD = "keyed_pcg64_dirichlet"
_SOBOL_CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
_CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet_chunked_projected"
_SUPPORTED_CONSTRUCTION_METHODS = frozenset(
    {
        _PCG64_CONSTRUCTION_METHOD,
        _SOBOL_CONSTRUCTION_METHOD,
        _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD,
    }
)
_SOBOL_BITS = 52
_SOBOL_MAX_DIMENSION = 21_201
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON without non-finite extensions."""
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


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _source_seed(value: int) -> int:
    """Return one unsigned binary64-width seed."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("source_seed must be an integer.")
    result = int(value)
    if result < 0 or result >= 2**64:
        raise ValueError("source_seed must lie in [0, 2**64).")
    return result


def _construction_method(value: str) -> str:
    """Return one supported frozen-bank construction method."""
    if not isinstance(value, str):
        raise TypeError("construction_method must be a string.")
    if value not in _SUPPORTED_CONSTRUCTION_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_CONSTRUCTION_METHODS))
        raise ValueError(f"construction_method must be one of: {supported}.")
    return value


def _require_power_of_two(value: int, *, name: str) -> None:
    """Require one positive integer to be an exact power of two."""
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two for the Sobol construction.")


def _readonly_float(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    """Return one finite ``float64`` array backed by immutable bytes."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)
    return cast(FloatArray, immutable)


def _readonly_integer(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> IntArray:
    """Return one immutable-bytes integer array without lossy coercion."""
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
    immutable = np.frombuffer(result.tobytes(order="C"), dtype=np.int64).reshape(result.shape)
    return cast(IntArray, immutable)


def _validated_sha256(value: str, *, name: str) -> str:
    """Return one lower-case SHA-256 identity string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def _array_payload(values: FloatArray | IntArray) -> dict[str, object]:
    """Return a shape-preserving canonical array payload."""
    if np.issubdtype(values.dtype, np.floating):
        dtype = "<f8"
    elif np.issubdtype(values.dtype, np.integer):
        dtype = "<i8"
    else:  # pragma: no cover - all callers are validated first
        raise TypeError("only float64 and int64 artifact arrays are supported.")
    return {
        "dtype": dtype,
        "shape": list(values.shape),
        "values": values.reshape(-1).tolist(),
    }


def _array_from_payload(
    payload: object,
    *,
    name: str,
    dtype: str,
) -> FloatArray | IntArray:
    """Decode one strict shape-preserving array payload."""
    if not isinstance(payload, dict) or set(payload) != {
        "dtype",
        "shape",
        "values",
    }:
        raise ValueError(f"{name} has an unexpected serialized array schema.")
    if payload["dtype"] != dtype:
        raise ValueError(f"{name} must use serialized dtype {dtype}.")
    shape_payload = payload["shape"]
    values_payload = payload["values"]
    if not isinstance(shape_payload, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape_payload
    ):
        raise ValueError(f"{name} shape must contain non-negative integers.")
    if not isinstance(values_payload, list):
        raise ValueError(f"{name} values must be a list.")
    expected_size = math.prod(shape_payload)
    if len(values_payload) != expected_size:
        raise ValueError(f"{name} serialized value count does not match its shape.")
    try:
        result = np.asarray(values_payload, dtype=np.dtype(dtype)).reshape(tuple(shape_payload))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} contains invalid serialized values.") from error
    if dtype == "<f8":
        return _readonly_float(result, name=name)
    return _readonly_integer(result, name=name)


def _array_sha256(values: FloatArray | IntArray) -> str:
    """Return a platform-independent dtype/shape/value array identity."""
    if np.issubdtype(values.dtype, np.floating):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        dtype = "<f8"
    else:
        canonical = np.ascontiguousarray(values, dtype="<i8")
        dtype = "<i8"
    header = _canonical_json({"dtype": dtype, "shape": list(values.shape)})
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _orthonormal_basis(values: ArrayLike, observation_count: int) -> FloatArray:
    """Return a validated error-whitened orthonormal summary basis."""
    basis = _readonly_float(values, name="summary_basis", ndim=2)
    if basis.shape[0] != observation_count or basis.shape[1] > observation_count:
        raise ValueError("summary_basis must have shape (number_of_observations, summary_rank).")
    tolerance = float(256.0 * np.finfo(np.float64).eps * max(1, *basis.shape))
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[1], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("summary_basis columns must be orthonormal.")
    return basis


def _region_generator(
    source_seed: int,
    sorted_cell_ids: IntArray,
) -> np.random.Generator:
    """Return the PCG64 stream keyed by a region's stable cell catalogue."""
    digest = hashlib.sha256(_ARTIFACT_SCHEMA_V1.encode("ascii"))
    digest.update(source_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(np.ascontiguousarray(sorted_cell_ids, dtype="<i8").tobytes())
    region_seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    return np.random.Generator(np.random.PCG64(region_seed))


def _sobol_block_seed(
    source_seed: int,
    *,
    node_count: int,
    block_index: int,
    catalogue_sha256: str,
) -> int:
    """Return a deterministic seed for one oversized Sobol catalogue block."""
    digest = hashlib.sha256(_ARTIFACT_SCHEMA_V2.encode("ascii"))
    digest.update(source_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(node_count.to_bytes(8, byteorder="little", signed=False))
    digest.update(block_index.to_bytes(8, byteorder="little", signed=False))
    digest.update(bytes.fromhex(catalogue_sha256))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _sobol_dimension_count(labels: IntArray) -> int:
    """Return the number of balanced-tree split coordinates."""
    counts = np.bincount(labels.reshape(-1))
    return int(np.sum(np.maximum(counts - 1, 0), dtype=np.int64))


def _sobol_catalogue_sha256(labels: IntArray, cell_ids: IntArray) -> str:
    """Return the canonical stable-ID region/node catalogue identity."""
    flat_labels = labels.reshape(-1)
    flat_ids = cell_ids.reshape(-1)
    region_signatures = [
        sorted(int(value) for value in flat_ids[np.flatnonzero(flat_labels == region)])
        for region in range(int(np.max(flat_labels)) + 1)
    ]
    region_signatures.sort(key=tuple)
    return _sha256_text(
        _canonical_json(
            {
                "balanced_tree": "count-balanced-breadth-first",
                "region_cell_id_signatures": region_signatures,
            }
        )
    )


def _sobol_block_dimensions(node_count: int) -> list[int]:
    """Return the deterministic contiguous Sobol block dimensions."""
    return [
        min(_SOBOL_MAX_DIMENSION, node_count - start) for start in range(0, node_count, _SOBOL_MAX_DIMENSION)
    ]


def _balanced_tree_ranges(cell_count: int) -> list[tuple[int, int, int]]:
    """Return count-balanced internal nodes in breadth-first order."""
    if cell_count <= 1:
        return []
    result: list[tuple[int, int, int]] = []
    queue = [(0, cell_count)]
    cursor = 0
    while cursor < len(queue):
        lower, upper = queue[cursor]
        cursor += 1
        middle = lower + (upper - lower) // 2
        result.append((lower, middle, upper))
        if middle - lower > 1:
            queue.append((lower, middle))
        if upper - middle > 1:
            queue.append((middle, upper))
    return result


def _source_operator_identity(
    *,
    design_sha256: str,
    noise_sd_sha256: str,
    summary_basis_sha256: str,
) -> str:
    """Return one identity for the fixed observation/operator projection."""
    return _sha256_text(
        _canonical_json(
            {
                "design_sha256": design_sha256,
                "noise_sd_sha256": noise_sd_sha256,
                "summary_basis_sha256": summary_basis_sha256,
            }
        )
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConditionalAllocationMixture:
    """Immutable frozen bank for one fixed labelled partition.

    Use :meth:`from_aggregation` for scientific construction.  The public
    constructor exists to support strict JSON replay and validates all
    dimensions and source identities.

    Args:
        projected_unit_mass_residual_factors: Array with shape
            ``(samples, summary_rank, regions)``.
        observation_mean_design: Conditional observation mean per unit mass,
            with shape ``(observations, regions)``.
        noise_sd: Independent measurement standard deviations.
        summary_basis: Frozen orthonormal basis in whitened observation space.
        labels: Fixed contiguous region labels on the native grid.
        cell_ids: Unique stable native-cell identifiers aligned with labels.
        alpha_totals: Additive concentration total in each labelled region.
        source_seed: Unsigned seed used to derive the region PCG64 streams.
        source_provenance: Non-empty human-readable source description.
        cell_alphas_sha256: Identity of the additive native concentration.
        design_sha256: Identity of the native observation operator.
        noise_sd_sha256: Identity of the error scales.
        summary_basis_sha256: Identity of the summary projection.
        construction_method: Frozen allocation-bank construction contract.
        construction_scipy_version: SciPy version that constructed a Sobol
            artifact.  This is omitted from the legacy PCG64 artifact.
        construction_sample_chunk_size: Number of Sobol rows materialized at
            once by the chunked projected construction.  This is part of the
            version-three numerical identity.
        construction_projection_chunk_size: Fixed row microbatch used for
            each native-to-summary matrix multiplication.
    """

    projected_unit_mass_residual_factors: FloatArray = field(init=False)
    observation_mean_design: FloatArray = field(init=False)
    noise_sd: FloatArray = field(init=False)
    summary_basis: FloatArray = field(init=False)
    labels: IntArray = field(init=False)
    cell_ids: IntArray = field(init=False)
    alpha_totals: FloatArray = field(init=False)
    source_seed: int = field(init=False)
    source_provenance: str = field(init=False)
    cell_alphas_sha256: str = field(init=False)
    design_sha256: str = field(init=False)
    noise_sd_sha256: str = field(init=False)
    summary_basis_sha256: str = field(init=False)
    source_operator_sha256: str = field(init=False)
    partition_sha256: str = field(init=False)
    construction_method: str = field(init=False)
    construction_scipy_version: str | None = field(init=False)
    construction_sample_chunk_size: int | None = field(init=False)
    construction_projection_chunk_size: int | None = field(init=False)
    sha256: str = field(init=False)

    def __init__(
        self,
        projected_unit_mass_residual_factors: ArrayLike,
        observation_mean_design: ArrayLike,
        noise_sd: ArrayLike,
        summary_basis: ArrayLike,
        labels: ArrayLike,
        cell_ids: ArrayLike,
        alpha_totals: ArrayLike,
        source_seed: int,
        source_provenance: str,
        cell_alphas_sha256: str,
        design_sha256: str,
        noise_sd_sha256: str,
        summary_basis_sha256: str,
        construction_method: str = _PCG64_CONSTRUCTION_METHOD,
        construction_scipy_version: str | None = None,
        construction_sample_chunk_size: int | None = None,
        construction_projection_chunk_size: int | None = None,
    ) -> None:
        """Validate, own, and fingerprint a frozen conditional mixture."""
        residual_factors = _readonly_float(
            projected_unit_mass_residual_factors,
            name="projected_unit_mass_residual_factors",
            ndim=3,
        )
        if residual_factors.shape[0] == 0:
            raise ValueError("projected_unit_mass_residual_factors must contain at least one sample.")
        observation_design = _readonly_float(
            observation_mean_design,
            name="observation_mean_design",
            ndim=2,
        )
        if observation_design.shape[0] == 0 or observation_design.shape[1] == 0:
            raise ValueError("observation_mean_design must be non-empty.")
        sample_count, summary_rank, region_count = residual_factors.shape
        if observation_design.shape[1] != region_count:
            raise ValueError("observation_mean_design and residual factors must have the same region count.")
        raw_scale = np.asarray(noise_sd, dtype=np.float64)
        if raw_scale.ndim == 0:
            raw_scale = np.full(
                observation_design.shape[0],
                float(raw_scale),
                dtype=np.float64,
            )
        scale = _readonly_float(raw_scale, name="noise_sd", ndim=1)
        if scale.shape != (observation_design.shape[0],) or np.any(scale <= 0.0):
            raise ValueError("noise_sd must be positive with one entry per observation.")
        basis = _orthonormal_basis(summary_basis, observation_design.shape[0])
        if basis.shape[1] != summary_rank:
            raise ValueError("summary_basis rank must match the residual-factor summary rank.")
        owned_labels = _readonly_integer(labels, name="labels")
        if owned_labels.ndim == 0 or owned_labels.size == 0:
            raise ValueError("labels must be a non-empty native-cell array.")
        unique_labels = np.unique(owned_labels)
        if not np.array_equal(
            unique_labels,
            np.arange(region_count, dtype=np.int64),
        ):
            raise ValueError("labels must use every contiguous region identifier from zero.")
        owned_cell_ids = _readonly_integer(cell_ids, name="cell_ids")
        if owned_cell_ids.shape != owned_labels.shape:
            raise ValueError("cell_ids must have the same shape as labels.")
        if np.unique(owned_cell_ids).size != owned_cell_ids.size:
            raise ValueError("cell_ids must be unique.")
        normalized_method = _construction_method(construction_method)
        if normalized_method in {
            _SOBOL_CONSTRUCTION_METHOD,
            _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD,
        }:
            _require_power_of_two(sample_count, name="sample_count")
            node_count = _sobol_dimension_count(owned_labels)
            if any(
                dimension > _SOBOL_MAX_DIMENSION for dimension in _sobol_block_dimensions(node_count)
            ):  # pragma: no cover - construction makes this impossible
                raise ValueError("Sobol block dimension exceeds the supported maximum.")
            if construction_scipy_version is None:
                normalized_scipy_version: str | None = scipy.__version__
            elif not isinstance(construction_scipy_version, str):
                raise TypeError("construction_scipy_version must be a string.")
            elif not construction_scipy_version.strip():
                raise ValueError("construction_scipy_version must be non-empty.")
            else:
                normalized_scipy_version = construction_scipy_version
            if normalized_method == _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD:
                normalized_chunk_size = _positive_integer(
                    construction_sample_chunk_size,  # type: ignore[arg-type]
                    name="construction_sample_chunk_size",
                )
                _require_power_of_two(
                    normalized_chunk_size,
                    name="construction_sample_chunk_size",
                )
                if normalized_chunk_size > sample_count:
                    raise ValueError("construction_sample_chunk_size cannot exceed sample_count.")
                if region_count != 1:
                    raise ValueError(
                        "the chunked projected Sobol construction currently "
                        "supports exactly one retained root region."
                    )
                normalized_projection_chunk_size = _positive_integer(
                    construction_projection_chunk_size,  # type: ignore[arg-type]
                    name="construction_projection_chunk_size",
                )
                _require_power_of_two(
                    normalized_projection_chunk_size,
                    name="construction_projection_chunk_size",
                )
                if normalized_projection_chunk_size > normalized_chunk_size:
                    raise ValueError(
                        "construction_projection_chunk_size cannot exceed construction_sample_chunk_size."
                    )
            else:
                if (
                    construction_sample_chunk_size is not None
                    or construction_projection_chunk_size is not None
                ):
                    raise ValueError(
                        "construction chunk sizes are only valid for the "
                        "chunked projected Sobol construction."
                    )
                normalized_chunk_size = None
                normalized_projection_chunk_size = None
        else:
            if construction_scipy_version is not None:
                raise ValueError("construction_scipy_version is only valid for Sobol artifacts.")
            if construction_sample_chunk_size is not None or construction_projection_chunk_size is not None:
                raise ValueError(
                    "construction chunk sizes are only valid for the chunked projected Sobol construction."
                )
            normalized_scipy_version = None
            normalized_chunk_size = None
            normalized_projection_chunk_size = None
        totals = _readonly_float(alpha_totals, name="alpha_totals", ndim=1)
        if totals.shape != (region_count,) or np.any(totals <= 0.0):
            raise ValueError("alpha_totals must contain one finite positive value per region.")
        normalized_seed = _source_seed(source_seed)
        if not isinstance(source_provenance, str):
            raise TypeError("source_provenance must be a string.")
        if not source_provenance.strip():
            raise ValueError("source_provenance must be non-empty.")
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
        basis_identity = _validated_sha256(
            summary_basis_sha256,
            name="summary_basis_sha256",
        )
        if _array_sha256(scale) != noise_identity:
            raise ValueError("noise_sd does not match noise_sd_sha256.")
        if _array_sha256(basis) != basis_identity:
            raise ValueError("summary_basis does not match summary_basis_sha256.")
        operator_identity = _source_operator_identity(
            design_sha256=design_identity,
            noise_sd_sha256=noise_identity,
            summary_basis_sha256=basis_identity,
        )
        partition_identity = _sha256_text(
            _canonical_json(
                {
                    "labels_sha256": _array_sha256(owned_labels),
                    "cell_ids_sha256": _array_sha256(owned_cell_ids),
                    "alpha_totals_sha256": _array_sha256(totals),
                }
            )
        )

        object.__setattr__(
            self,
            "projected_unit_mass_residual_factors",
            residual_factors,
        )
        object.__setattr__(self, "observation_mean_design", observation_design)
        object.__setattr__(self, "noise_sd", scale)
        object.__setattr__(self, "summary_basis", basis)
        object.__setattr__(self, "labels", owned_labels)
        object.__setattr__(self, "cell_ids", owned_cell_ids)
        object.__setattr__(self, "alpha_totals", totals)
        object.__setattr__(self, "source_seed", normalized_seed)
        object.__setattr__(self, "source_provenance", source_provenance)
        object.__setattr__(self, "cell_alphas_sha256", alpha_identity)
        object.__setattr__(self, "design_sha256", design_identity)
        object.__setattr__(self, "noise_sd_sha256", noise_identity)
        object.__setattr__(self, "summary_basis_sha256", basis_identity)
        object.__setattr__(self, "source_operator_sha256", operator_identity)
        object.__setattr__(self, "partition_sha256", partition_identity)
        object.__setattr__(self, "construction_method", normalized_method)
        object.__setattr__(
            self,
            "construction_scipy_version",
            normalized_scipy_version,
        )
        object.__setattr__(
            self,
            "construction_sample_chunk_size",
            normalized_chunk_size,
        )
        object.__setattr__(
            self,
            "construction_projection_chunk_size",
            normalized_projection_chunk_size,
        )
        object.__setattr__(self, "sha256", _sha256_text(self.to_json()))
        # Keep sample_count referenced here: it documents that the leading
        # dimension was deliberately validated before fingerprinting.
        assert sample_count == self.sample_count

    @classmethod
    def from_aggregation(
        cls,
        aggregation: AdditiveDirichletAggregation,
        partition: ArrayLike | PartitionSummaryFactors,
        *,
        sample_count: int,
        source_seed: int,
        source_provenance: str,
        cell_ids: ArrayLike | None = None,
        construction_method: str = _PCG64_CONSTRUCTION_METHOD,
        sample_chunk_size: int | None = None,
        projection_chunk_size: int | None = None,
    ) -> ConditionalAllocationMixture:
        """Construct a replayable frozen bank from one additive native model.

        If ``partition`` is a :class:`PartitionSummaryFactors` object, its
        labels and cached factors must exactly replay factors built from
        ``aggregation``.  Stable ``cell_ids`` make the generated bank
        invariant to a simultaneous permutation of native cells and their
        scientific identities.  The default identifiers are row-major native
        indices.

        Args:
            aggregation: Common additive native aggregation model.
            partition: Fixed labels or cached factors for those labels.
            sample_count: Number of allocation components.  These are
                independent draws for PCG64 and randomized low-discrepancy
                points for the Sobol construction.
            source_seed: Seed used to derive keyed PCG64 streams or scrambled
                Sobol dimension blocks.
            source_provenance: Human-readable origin of the bank.
            cell_ids: Optional unique integer scientific cell identifiers.
            construction_method: Allocation-bank construction.  The default
                ``keyed_pcg64_dirichlet`` preserves the version-one artifact
                and numerical contract.  The
                ``scrambled_sobol_balanced_dirichlet`` method uses a
                canonical joint scrambled Sobol catalogue, split into
                independently scrambled blocks only above the engine's
                dimension limit, and stable-ID balanced-tree Dirichlet
                inversion.
            sample_chunk_size: Number of sample rows materialized at once by
                ``scrambled_sobol_balanced_dirichlet_chunked_projected``.
                It must be a power of two no greater than ``sample_count``.
                The chunked method currently requires one root region.
            projection_chunk_size: Fixed power-of-two row microbatch for
                native-to-summary projection.  It cannot exceed
                ``sample_chunk_size``.  Holding this fixed makes projected
                floating-point values independent of the memory chunk.

        Returns:
            Immutable conditional-allocation mixture artifact.

        Raises:
            TypeError: If model, integer, or array types are invalid.
            ValueError: If shapes, identities, or cached factors disagree.
        """
        if not isinstance(aggregation, AdditiveDirichletAggregation):
            raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
        normalized_sample_count = _positive_integer(
            sample_count,
            name="sample_count",
        )
        normalized_seed = _source_seed(source_seed)
        normalized_method = _construction_method(construction_method)
        if normalized_method in {
            _SOBOL_CONSTRUCTION_METHOD,
            _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD,
        }:
            _require_power_of_two(
                normalized_sample_count,
                name="sample_count",
            )
        if normalized_method == _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD:
            normalized_chunk_size = _positive_integer(
                sample_chunk_size,  # type: ignore[arg-type]
                name="sample_chunk_size",
            )
            _require_power_of_two(
                normalized_chunk_size,
                name="sample_chunk_size",
            )
            if normalized_chunk_size > normalized_sample_count:
                raise ValueError("sample_chunk_size cannot exceed sample_count.")
            normalized_projection_chunk_size = _positive_integer(
                projection_chunk_size,  # type: ignore[arg-type]
                name="projection_chunk_size",
            )
            _require_power_of_two(
                normalized_projection_chunk_size,
                name="projection_chunk_size",
            )
            if normalized_projection_chunk_size > normalized_chunk_size:
                raise ValueError("projection_chunk_size cannot exceed sample_chunk_size.")
        elif sample_chunk_size is not None or projection_chunk_size is not None:
            raise ValueError("chunk sizes are only valid for the chunked projected Sobol construction.")
        else:
            normalized_chunk_size = None
            normalized_projection_chunk_size = None
        if isinstance(partition, PartitionSummaryFactors):
            supplied_factors = partition
            labels = supplied_factors.labels
        else:
            supplied_factors = None
            labels = partition
        rebuilt_factors = aggregation.partition_factors(labels)
        if supplied_factors is not None:
            for name in (
                "labels",
                "alpha_totals",
                "observation_mean_design",
                "summary_mean_design",
                "summary_covariance_factors",
            ):
                if not np.array_equal(
                    getattr(supplied_factors, name),
                    getattr(rebuilt_factors, name),
                ):
                    raise ValueError("partition factors do not exactly match the supplied aggregation.")
            factors = supplied_factors
        else:
            factors = rebuilt_factors

        if cell_ids is None:
            normalized_cell_ids = np.arange(
                aggregation.cell_alphas.size,
                dtype=np.int64,
            ).reshape(aggregation.cell_shape)
        else:
            normalized_cell_ids = _readonly_integer(cell_ids, name="cell_ids")
            if normalized_cell_ids.shape != aggregation.cell_shape:
                raise ValueError("cell_ids must have the same shape as cell_alphas.")
            if np.unique(normalized_cell_ids).size != normalized_cell_ids.size:
                raise ValueError("cell_ids must be unique.")

        flat_labels = factors.labels.reshape(-1)
        flat_ids = normalized_cell_ids.reshape(-1)
        flat_alphas = aggregation.cell_alphas.reshape(-1)
        summary_design = aggregation.summary_design
        residual_factors = np.empty(
            (
                normalized_sample_count,
                factors.summary_dimension,
                factors.region_count,
            ),
            dtype=np.float64,
        )
        if normalized_method == _PCG64_CONSTRUCTION_METHOD:
            for region in range(factors.region_count):
                selected_indices = np.flatnonzero(flat_labels == region)
                order = np.argsort(flat_ids[selected_indices], kind="stable")
                sorted_indices = selected_indices[order]
                if sorted_indices.size == 1:
                    # A one-cell Dirichlet distribution is the exact point
                    # mass at one.  NumPy's generic Gamma normalization can
                    # return a value a few ULP from one, which must not
                    # manufacture aggregation error in the fine-cell limit.
                    residual_factors[:, :, region] = 0.0
                    continue
                sorted_ids = np.asarray(flat_ids[sorted_indices], dtype=np.int64)
                region_alphas = flat_alphas[sorted_indices]
                generator = _region_generator(normalized_seed, sorted_ids)
                shares = generator.dirichlet(
                    region_alphas,
                    size=normalized_sample_count,
                )
                region_columns = summary_design[:, sorted_indices]
                expected = region_columns @ (region_alphas / float(np.sum(region_alphas)))
                residual_factors[:, :, region] = shares @ region_columns.T - expected[np.newaxis, :]
        elif normalized_method == _SOBOL_CONSTRUCTION_METHOD:
            cls._fill_scrambled_sobol_residual_factors(
                residual_factors=residual_factors,
                flat_labels=flat_labels,
                flat_ids=flat_ids,
                flat_alphas=flat_alphas,
                summary_design=summary_design,
                region_count=factors.region_count,
                sample_count=normalized_sample_count,
                source_seed=normalized_seed,
            )
        else:
            if factors.region_count != 1:
                raise ValueError(
                    "the chunked projected Sobol construction currently "
                    "supports exactly one retained root region."
                )
            assert normalized_chunk_size is not None
            assert normalized_projection_chunk_size is not None
            cls._fill_chunked_projected_sobol_residual_factors(
                residual_factors=residual_factors,
                flat_ids=flat_ids,
                flat_alphas=flat_alphas,
                summary_design=summary_design,
                sample_count=normalized_sample_count,
                sample_chunk_size=normalized_chunk_size,
                projection_chunk_size=normalized_projection_chunk_size,
                source_seed=normalized_seed,
            )

        alphas = cast(FloatArray, aggregation.cell_alphas)
        design = cast(FloatArray, aggregation.design)
        noise = cast(FloatArray, aggregation.noise_sd)
        basis = cast(FloatArray, aggregation.summary_basis)
        return cls(
            residual_factors,
            factors.observation_mean_design,
            noise,
            basis,
            factors.labels,
            normalized_cell_ids,
            factors.alpha_totals,
            normalized_seed,
            source_provenance,
            _array_sha256(alphas),
            _array_sha256(design),
            _array_sha256(noise),
            _array_sha256(basis),
            normalized_method,
            None,
            normalized_chunk_size,
            normalized_projection_chunk_size,
        )

    @staticmethod
    def _fill_chunked_projected_sobol_residual_factors(
        *,
        residual_factors: FloatArray,
        flat_ids: IntArray,
        flat_alphas: FloatArray,
        summary_design: FloatArray,
        sample_count: int,
        sample_chunk_size: int,
        projection_chunk_size: int,
        source_seed: int,
    ) -> None:
        """Fill one root bank without retaining sample-by-native allocations."""
        if int(qmc.Sobol.MAXDIM) < _SOBOL_MAX_DIMENSION:
            raise RuntimeError(
                "installed SciPy Sobol implementation has a smaller dimension "
                "limit than the recorded construction contract."
            )
        order = np.argsort(flat_ids, kind="stable")
        sorted_indices = np.asarray(order, dtype=np.int64)
        sorted_ids = np.asarray(flat_ids[sorted_indices], dtype=np.int64)
        region_alphas = np.asarray(flat_alphas[sorted_indices], dtype=np.float64)
        cell_count = int(sorted_indices.size)
        if cell_count == 1:
            residual_factors[:, :, 0] = 0.0
            return

        nodes: list[tuple[int, int, int, float, float]] = []
        for lower, middle, upper in _balanced_tree_ranges(cell_count):
            left_alpha = math.fsum(float(value) for value in region_alphas[lower:middle])
            right_alpha = math.fsum(float(value) for value in region_alphas[middle:upper])
            if (
                not math.isfinite(left_alpha)
                or not math.isfinite(right_alpha)
                or left_alpha <= 0.0
                or right_alpha <= 0.0
            ):
                raise ValueError("balanced Dirichlet node concentrations must be finite and positive.")
            nodes.append((lower, middle, upper, left_alpha, right_alpha))

        labels = np.zeros(cell_count, dtype=np.int64)
        catalogue_sha256 = _sobol_catalogue_sha256(labels, sorted_ids)
        node_count = len(nodes)
        block_dimensions = _sobol_block_dimensions(node_count)
        engines: list[qmc.Sobol] = []
        for block_index, dimension in enumerate(block_dimensions):
            engines.append(
                qmc.Sobol(
                    d=dimension,
                    scramble=True,
                    bits=_SOBOL_BITS,
                    rng=_sobol_block_seed(
                        source_seed,
                        node_count=node_count,
                        block_index=block_index,
                        catalogue_sha256=catalogue_sha256,
                    ),
                    optimization=None,
                )
            )

        region_columns = summary_design[:, sorted_indices]
        # Match PartitionSummaryFactors exactly: the stored observation mean
        # and the projected residual use one represented concentration total.
        expected = region_columns @ (region_alphas / float(np.sum(region_alphas)))
        for start in range(0, sample_count, sample_chunk_size):
            stop = min(start + sample_chunk_size, sample_count)
            chunk_count = stop - start
            shares = np.empty((chunk_count, cell_count), dtype=np.float64)
            active_masses: dict[tuple[int, int], FloatArray] = {
                (0, cell_count): cast(
                    FloatArray,
                    np.ones(chunk_count, dtype=np.float64),
                )
            }
            node_cursor = 0
            for engine, dimension in zip(
                engines,
                block_dimensions,
                strict=True,
            ):
                uniforms = engine.random(chunk_count)
                if uniforms.shape != (chunk_count, dimension):
                    raise RuntimeError("Sobol engine returned an unexpected sample shape.")
                if not np.all(np.isfinite(uniforms)) or np.any((uniforms < 0.0) | (uniforms > 1.0)):
                    raise ValueError("Sobol coordinates must be finite and lie in [0, 1].")
                block_nodes = nodes[node_cursor : node_cursor + dimension]
                left_parameters = np.asarray(
                    [node[3] for node in block_nodes],
                    dtype=np.float64,
                )
                right_parameters = np.asarray(
                    [node[4] for node in block_nodes],
                    dtype=np.float64,
                )
                fractions = special.betaincinv(
                    left_parameters[np.newaxis, :],
                    right_parameters[np.newaxis, :],
                    uniforms,
                )
                if not np.all(np.isfinite(fractions)) or np.any((fractions < 0.0) | (fractions > 1.0)):
                    raise ValueError("balanced Dirichlet inverse produced an invalid split fraction.")
                for local_coordinate in range(dimension):
                    lower, middle, upper, _, _ = block_nodes[local_coordinate]
                    parent_mass = active_masses.pop((lower, upper))
                    left_mass = parent_mass * fractions[:, local_coordinate]
                    right_mass = parent_mass - left_mass
                    if (
                        not np.all(np.isfinite(left_mass))
                        or not np.all(np.isfinite(right_mass))
                        or np.any(left_mass < 0.0)
                        or np.any(right_mass < 0.0)
                        or np.any(left_mass > parent_mass)
                        or np.any(right_mass > parent_mass)
                    ):
                        raise ValueError("balanced Dirichlet inverse produced invalid child masses.")
                    if np.any(np.abs((left_mass + right_mass) - parent_mass) > np.spacing(parent_mass)):
                        raise ValueError("balanced Dirichlet inverse did not conserve parent mass.")
                    if middle - lower == 1:
                        shares[:, lower] = left_mass
                    else:
                        active_masses[(lower, middle)] = cast(
                            FloatArray,
                            left_mass,
                        )
                    if upper - middle == 1:
                        shares[:, middle] = right_mass
                    else:
                        active_masses[(middle, upper)] = cast(
                            FloatArray,
                            right_mass,
                        )
                node_cursor += dimension
            if node_cursor != node_count or active_masses:
                raise RuntimeError("balanced Dirichlet tree catalogue was not exhausted.")
            if not np.all(np.isfinite(shares)) or np.any((shares < 0.0) | (shares > 1.0)):
                raise ValueError("balanced Dirichlet inverse produced invalid allocation shares.")
            for projection_start in range(0, chunk_count, projection_chunk_size):
                projection_stop = min(
                    projection_start + projection_chunk_size,
                    chunk_count,
                )
                output_start = start + projection_start
                output_stop = start + projection_stop
                residual_factors[output_start:output_stop, :, 0] = (
                    shares[projection_start:projection_stop] @ region_columns.T - expected[np.newaxis, :]
                )

    @staticmethod
    def _fill_scrambled_sobol_residual_factors(
        *,
        residual_factors: FloatArray,
        flat_labels: IntArray,
        flat_ids: IntArray,
        flat_alphas: FloatArray,
        summary_design: FloatArray,
        region_count: int,
        sample_count: int,
        source_seed: int,
    ) -> None:
        """Fill factors using a canonical balanced-tree Sobol inverse."""
        if int(qmc.Sobol.MAXDIM) < _SOBOL_MAX_DIMENSION:
            raise RuntimeError(
                "installed SciPy Sobol implementation has a smaller dimension "
                "limit than the recorded construction contract."
            )

        # Region labels are incidental.  Sort regions by their complete
        # stable-ID signatures, and cells within each region by stable ID.
        region_catalogue: list[tuple[int, IntArray, FloatArray]] = []
        for region in range(region_count):
            selected_indices = np.flatnonzero(flat_labels == region)
            order = np.argsort(flat_ids[selected_indices], kind="stable")
            sorted_indices = np.asarray(
                selected_indices[order],
                dtype=np.int64,
            )
            region_alphas = np.asarray(
                flat_alphas[sorted_indices],
                dtype=np.float64,
            )
            region_catalogue.append(
                (
                    region,
                    sorted_indices,
                    region_alphas,
                )
            )
        region_catalogue.sort(key=lambda entry: tuple(int(value) for value in flat_ids[entry[1]]))
        catalogue_sha256 = _sobol_catalogue_sha256(
            flat_labels,
            flat_ids,
        )

        # Each entry identifies one internal balanced-tree node.  Concatenating
        # breadth-first region catalogues gives one canonical joint Sobol
        # coordinate system.
        nodes: list[tuple[int, int, int, int, float, float]] = []
        shares_by_catalogue_region: list[FloatArray] = []
        active_masses: list[dict[tuple[int, int], FloatArray]] = []
        for catalogue_region, (_, sorted_indices, region_alphas) in enumerate(region_catalogue):
            cell_count = int(sorted_indices.size)
            shares = np.empty((sample_count, cell_count), dtype=np.float64)
            shares_by_catalogue_region.append(cast(FloatArray, shares))
            if cell_count == 1:
                shares[:, 0] = 1.0
                active_masses.append({})
                continue
            active_masses.append(
                {
                    (0, cell_count): cast(
                        FloatArray,
                        np.ones(sample_count, dtype=np.float64),
                    )
                }
            )
            for lower, middle, upper in _balanced_tree_ranges(cell_count):
                # Direct compensated sums avoid subtracting nearly equal
                # prefixes when one late subtree is tiny relative to an
                # earlier cell's concentration.
                left_alpha = math.fsum(float(value) for value in region_alphas[lower:middle])
                right_alpha = math.fsum(float(value) for value in region_alphas[middle:upper])
                if (
                    not math.isfinite(left_alpha)
                    or not math.isfinite(right_alpha)
                    or left_alpha <= 0.0
                    or right_alpha <= 0.0
                ):
                    raise ValueError("balanced Dirichlet node concentrations must be finite and positive.")
                nodes.append(
                    (
                        catalogue_region,
                        lower,
                        middle,
                        upper,
                        left_alpha,
                        right_alpha,
                    )
                )

        node_count = len(nodes)
        block_dimensions = _sobol_block_dimensions(node_count)
        base_two_exponent = sample_count.bit_length() - 1
        node_cursor = 0
        for block_index, dimension in enumerate(block_dimensions):
            block_seed = _sobol_block_seed(
                source_seed,
                node_count=node_count,
                block_index=block_index,
                catalogue_sha256=catalogue_sha256,
            )
            engine = qmc.Sobol(
                d=dimension,
                scramble=True,
                bits=_SOBOL_BITS,
                rng=block_seed,
                optimization=None,
            )
            uniforms = engine.random_base2(base_two_exponent)
            if uniforms.shape != (sample_count, dimension):
                raise RuntimeError("Sobol engine returned an unexpected sample shape.")
            if not np.all(np.isfinite(uniforms)) or np.any((uniforms < 0.0) | (uniforms > 1.0)):
                raise ValueError("Sobol coordinates must be finite and lie in [0, 1].")

            for local_coordinate in range(dimension):
                (
                    catalogue_region,
                    lower,
                    middle,
                    upper,
                    left_alpha,
                    right_alpha,
                ) = nodes[node_cursor + local_coordinate]
                fractions = special.betaincinv(
                    left_alpha,
                    right_alpha,
                    uniforms[:, local_coordinate],
                )
                if not np.all(np.isfinite(fractions)) or np.any((fractions < 0.0) | (fractions > 1.0)):
                    raise ValueError("balanced Dirichlet inverse produced an invalid split fraction.")
                parent_mass = active_masses[catalogue_region].pop((lower, upper))
                left_mass = parent_mass * fractions
                # Subtraction makes the right child the exact complement in
                # the represented parent coordinate rather than a second,
                # independently rounded product.
                right_mass = parent_mass - left_mass
                if (
                    not np.all(np.isfinite(left_mass))
                    or not np.all(np.isfinite(right_mass))
                    or np.any(left_mass < 0.0)
                    or np.any(right_mass < 0.0)
                    or np.any(left_mass > parent_mass)
                    or np.any(right_mass > parent_mass)
                ):
                    raise ValueError("balanced Dirichlet inverse produced invalid child masses.")
                reconstructed = left_mass + right_mass
                conservation_tolerance = np.spacing(parent_mass)
                if np.any(np.abs(reconstructed - parent_mass) > conservation_tolerance):
                    raise ValueError("balanced Dirichlet inverse did not conserve parent mass.")

                if middle - lower == 1:
                    shares_by_catalogue_region[catalogue_region][
                        :,
                        lower,
                    ] = left_mass
                else:
                    active_masses[catalogue_region][(lower, middle)] = cast(FloatArray, left_mass)
                if upper - middle == 1:
                    shares_by_catalogue_region[catalogue_region][
                        :,
                        middle,
                    ] = right_mass
                else:
                    active_masses[catalogue_region][(middle, upper)] = cast(FloatArray, right_mass)
            node_cursor += dimension

        if node_cursor != node_count or any(bool(region_masses) for region_masses in active_masses):
            raise RuntimeError("balanced Dirichlet tree catalogue was not exhausted.")

        for catalogue_region, (
            output_region,
            sorted_indices,
            region_alphas,
        ) in enumerate(region_catalogue):
            shares = shares_by_catalogue_region[catalogue_region]
            if not np.all(np.isfinite(shares)) or np.any((shares < 0.0) | (shares > 1.0)):
                raise ValueError("balanced Dirichlet inverse produced invalid allocation shares.")
            if sorted_indices.size == 1:
                residual_factors[:, :, output_region] = 0.0
                continue
            region_columns = summary_design[:, sorted_indices]
            expected = region_columns @ (region_alphas / float(np.sum(region_alphas)))
            residual_factors[:, :, output_region] = shares @ region_columns.T - expected[np.newaxis, :]

    @property
    def sample_count(self) -> int:
        """Return the frozen number of equal-weight mixture components."""
        return int(self.projected_unit_mass_residual_factors.shape[0])

    @property
    def summary_rank(self) -> int:
        """Return the fixed summary dimension."""
        return int(self.projected_unit_mass_residual_factors.shape[1])

    @property
    def region_count(self) -> int:
        """Return the fixed number of labelled retained regions."""
        return int(self.projected_unit_mass_residual_factors.shape[2])

    @property
    def observation_count(self) -> int:
        """Return the number of observation rows."""
        return int(self.observation_mean_design.shape[0])

    @property
    def storage_nbytes(self) -> int:
        """Return bytes owned by the artifact's scientific arrays."""
        return int(
            self.projected_unit_mass_residual_factors.nbytes
            + self.observation_mean_design.nbytes
            + self.noise_sd.nbytes
            + self.summary_basis.nbytes
            + self.labels.nbytes
            + self.cell_ids.nbytes
            + self.alpha_totals.nbytes
        )

    @property
    def payload(self) -> dict[str, object]:
        """Return the complete canonical JSON-compatible artifact payload."""
        common_payload: dict[str, object] = {
            "bit_generator": "PCG64",
            "factor_axes": ["sample", "summary", "region"],
            "projected_unit_mass_residual_factors": _array_payload(self.projected_unit_mass_residual_factors),
            "observation_mean_design": _array_payload(self.observation_mean_design),
            "noise_sd": _array_payload(self.noise_sd),
            "summary_basis": _array_payload(self.summary_basis),
            "labels": _array_payload(self.labels),
            "cell_ids": _array_payload(self.cell_ids),
            "alpha_totals": _array_payload(self.alpha_totals),
            "source_seed": self.source_seed,
            "source_provenance": self.source_provenance,
            "cell_alphas_sha256": self.cell_alphas_sha256,
            "design_sha256": self.design_sha256,
            "noise_sd_sha256": self.noise_sd_sha256,
            "summary_basis_sha256": self.summary_basis_sha256,
            "source_operator_sha256": self.source_operator_sha256,
            "partition_sha256": self.partition_sha256,
        }
        if self.construction_method == _PCG64_CONSTRUCTION_METHOD:
            # This exact field set and schema are the immutable v1 contract.
            # Do not add construction_method to it: existing JSON and SHA-256
            # identities must remain byte-for-byte stable.
            return {
                "schema": _ARTIFACT_SCHEMA_V1,
                **common_payload,
            }

        del common_payload["bit_generator"]
        node_count = _sobol_dimension_count(self.labels)
        sobol_payload: dict[str, object] = {
            "schema": _ARTIFACT_SCHEMA_V2,
            "construction_method": _SOBOL_CONSTRUCTION_METHOD,
            "construction_scipy_version": self.construction_scipy_version,
            "quasi_random_engine": "scipy.stats.qmc.Sobol",
            "sobol_scramble": True,
            "sobol_bits": _SOBOL_BITS,
            "sobol_optimization": None,
            "inverse_transform": "scipy.special.betaincinv",
            "dimension_order": ("stable-id-region-signature/count-balanced-breadth-first"),
            "sobol_block_rule": ("contiguous-canonical-node-catalogue/max-dimension-21201"),
            "sobol_seed_derivation": (
                "sha256(schema-v2,source-seed,node-count,block-index,catalogue-sha256)/little-endian-first-64"
            ),
            "sobol_catalogue_sha256": _sobol_catalogue_sha256(
                self.labels,
                self.cell_ids,
            ),
            "sobol_block_dimensions": _sobol_block_dimensions(node_count),
            **common_payload,
        }
        if self.construction_method == _SOBOL_CONSTRUCTION_METHOD:
            return sobol_payload
        sobol_payload.update(
            {
                "schema": _ARTIFACT_SCHEMA_V3,
                "construction_method": (_CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD),
                "construction_sample_chunk_size": (self.construction_sample_chunk_size),
                "construction_projection_chunk_size": (self.construction_projection_chunk_size),
                "sample_traversal": ("ascending-contiguous-sample-chunks/persistent-sobol-engines"),
                "projection_rule": ("project-fixed-row-microbatches-then-discard-native-allocations"),
            }
        )
        return sobol_payload

    def to_json(self) -> str:
        """Return the canonical artifact serialization.

        Returns:
            Strict JSON containing the frozen bank and all recorded source
            identities.
        """
        return _canonical_json(self.payload)

    @classmethod
    def from_json(
        cls,
        serialized: str,
        *,
        expected_sha256: str,
    ) -> ConditionalAllocationMixture:
        """Replay an artifact whose whole identity is pinned by the caller.

        Args:
            serialized: Strict canonical JSON produced by :meth:`to_json`.
            expected_sha256: Required trusted whole-artifact fingerprint.  It
                pins the external native-alpha and design identities, which
                cannot be reconstructed from the reduced artifact alone.

        Returns:
            Immutable artifact reconstructed from ``serialized``.

        Raises:
            TypeError: If a text or digest argument has the wrong type.
            ValueError: If JSON, schema, arrays, identities, canonical form,
                or the required fingerprint is invalid.
        """
        if not isinstance(serialized, str):
            raise TypeError("serialized must be a string.")
        observed_sha256 = _sha256_text(serialized)
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if observed_sha256 != expected:
            raise ValueError("serialized artifact SHA-256 does not match the expected identity.")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized artifact must be valid JSON.") from error
        version_one_fields = {
            "schema",
            "bit_generator",
            "factor_axes",
            "projected_unit_mass_residual_factors",
            "observation_mean_design",
            "noise_sd",
            "summary_basis",
            "labels",
            "cell_ids",
            "alpha_totals",
            "source_seed",
            "source_provenance",
            "cell_alphas_sha256",
            "design_sha256",
            "noise_sd_sha256",
            "summary_basis_sha256",
            "source_operator_sha256",
            "partition_sha256",
        }
        version_two_fields = (version_one_fields - {"bit_generator"}) | {
            "construction_method",
            "construction_scipy_version",
            "quasi_random_engine",
            "sobol_scramble",
            "sobol_bits",
            "sobol_optimization",
            "inverse_transform",
            "dimension_order",
            "sobol_block_rule",
            "sobol_seed_derivation",
            "sobol_catalogue_sha256",
            "sobol_block_dimensions",
        }
        version_three_fields = version_two_fields | {
            "construction_sample_chunk_size",
            "construction_projection_chunk_size",
            "sample_traversal",
            "projection_rule",
        }
        if not isinstance(payload, dict):
            raise ValueError("serialized artifact has unexpected fields.")
        schema = payload.get("schema")
        if schema == _ARTIFACT_SCHEMA_V1:
            expected_fields = version_one_fields
            construction_method = _PCG64_CONSTRUCTION_METHOD
        elif schema == _ARTIFACT_SCHEMA_V2:
            expected_fields = version_two_fields
            construction_method = _SOBOL_CONSTRUCTION_METHOD
        elif schema == _ARTIFACT_SCHEMA_V3:
            expected_fields = version_three_fields
            construction_method = _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD
        else:
            raise ValueError("serialized artifact schema is unsupported.")
        if set(payload) != expected_fields:
            raise ValueError("serialized artifact has unexpected fields.")
        if construction_method == _PCG64_CONSTRUCTION_METHOD:
            if payload["bit_generator"] != "PCG64":
                raise ValueError("serialized artifact bit generator is unsupported.")
        else:
            if payload["construction_method"] != construction_method:
                raise ValueError("serialized artifact construction method is unsupported.")
            if payload["quasi_random_engine"] != "scipy.stats.qmc.Sobol":
                raise ValueError("serialized artifact quasi-random engine is unsupported.")
            if payload["sobol_scramble"] is not True:
                raise ValueError("serialized artifact Sobol scrambling is unsupported.")
            if payload["sobol_bits"] != _SOBOL_BITS:
                raise ValueError("serialized artifact Sobol bit depth is unsupported.")
            if payload["sobol_optimization"] is not None:
                raise ValueError("serialized artifact Sobol optimization is unsupported.")
            if payload["inverse_transform"] != "scipy.special.betaincinv":
                raise ValueError("serialized artifact inverse transform is unsupported.")
            if payload["dimension_order"] != ("stable-id-region-signature/count-balanced-breadth-first"):
                raise ValueError("serialized artifact Sobol dimension order is unsupported.")
            if payload["sobol_block_rule"] != ("contiguous-canonical-node-catalogue/max-dimension-21201"):
                raise ValueError("serialized artifact Sobol block rule is unsupported.")
            if payload["sobol_seed_derivation"] != (
                "sha256(schema-v2,source-seed,node-count,block-index,catalogue-sha256)/little-endian-first-64"
            ):
                raise ValueError("serialized artifact Sobol seed derivation is unsupported.")
            if construction_method == _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD:
                if payload["sample_traversal"] != (
                    "ascending-contiguous-sample-chunks/persistent-sobol-engines"
                ):
                    raise ValueError("serialized artifact sample traversal is unsupported.")
                if payload["projection_rule"] != (
                    "project-fixed-row-microbatches-then-discard-native-allocations"
                ):
                    raise ValueError("serialized artifact projection rule is unsupported.")
        if payload["factor_axes"] != ["sample", "summary", "region"]:
            raise ValueError("serialized artifact factor axes are unsupported.")
        residual_factors = _array_from_payload(
            payload["projected_unit_mass_residual_factors"],
            name="projected_unit_mass_residual_factors",
            dtype="<f8",
        )
        observation_design = _array_from_payload(
            payload["observation_mean_design"],
            name="observation_mean_design",
            dtype="<f8",
        )
        noise_sd = _array_from_payload(
            payload["noise_sd"],
            name="noise_sd",
            dtype="<f8",
        )
        summary_basis = _array_from_payload(
            payload["summary_basis"],
            name="summary_basis",
            dtype="<f8",
        )
        labels = _array_from_payload(
            payload["labels"],
            name="labels",
            dtype="<i8",
        )
        cell_ids = _array_from_payload(
            payload["cell_ids"],
            name="cell_ids",
            dtype="<i8",
        )
        alpha_totals = _array_from_payload(
            payload["alpha_totals"],
            name="alpha_totals",
            dtype="<f8",
        )
        result = cls(
            residual_factors,
            observation_design,
            noise_sd,
            summary_basis,
            labels,
            cell_ids,
            alpha_totals,
            payload["source_seed"],
            payload["source_provenance"],
            payload["cell_alphas_sha256"],
            payload["design_sha256"],
            payload["noise_sd_sha256"],
            payload["summary_basis_sha256"],
            construction_method,
            (
                payload["construction_scipy_version"]
                if construction_method
                in {
                    _SOBOL_CONSTRUCTION_METHOD,
                    _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD,
                }
                else None
            ),
            (
                payload["construction_sample_chunk_size"]
                if construction_method == _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD
                else None
            ),
            (
                payload["construction_projection_chunk_size"]
                if construction_method == _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD
                else None
            ),
        )
        if construction_method in {
            _SOBOL_CONSTRUCTION_METHOD,
            _CHUNKED_PROJECTED_SOBOL_CONSTRUCTION_METHOD,
        }:
            expected_catalogue_sha256 = _sobol_catalogue_sha256(
                result.labels,
                result.cell_ids,
            )
            if payload["sobol_catalogue_sha256"] != expected_catalogue_sha256:
                raise ValueError("serialized Sobol catalogue identity does not replay.")
            expected_block_dimensions = _sobol_block_dimensions(_sobol_dimension_count(result.labels))
            if payload["sobol_block_dimensions"] != expected_block_dimensions:
                raise ValueError("serialized Sobol block dimensions do not replay.")
        if payload["source_operator_sha256"] != result.source_operator_sha256:
            raise ValueError("serialized source operator identity does not replay.")
        if payload["partition_sha256"] != result.partition_sha256:
            raise ValueError("serialized partition identity does not replay.")
        if result.to_json() != serialized:
            raise ValueError("serialized artifact must use canonical JSON.")
        if result.sha256 != observed_sha256:
            raise ValueError("serialized artifact identity did not replay.")
        return result

    def log_likelihood(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> float:
        """Evaluate this frozen conditional likelihood approximation.

        Args:
            observation: Finite vector aligned with artifact observations.
            masses: One finite non-negative retained mass per region.
            mean_offset: Optional fixed observation-space contribution.

        Returns:
            Normalized approximate conditional log density.
        """
        return conditional_allocation_mixture_log_likelihood(
            observation,
            masses,
            self,
            mean_offset=mean_offset,
        )

    def log_likelihood_and_mass_gradient(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> tuple[float, FloatArray]:
        """Evaluate log likelihood and its analytic retained-mass gradient.

        Args:
            observation: Finite vector aligned with artifact observations.
            masses: One finite non-negative retained mass per region.
            mean_offset: Optional fixed observation-space contribution.

        Returns:
            Normalized approximate log density and read-only mass gradient.
        """
        return conditional_allocation_mixture_log_likelihood_and_gradient(
            observation,
            masses,
            self,
            mean_offset=mean_offset,
        )


def _validated_evaluation(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalAllocationMixture,
    mean_offset: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate one deterministic evaluation and return owned vectors."""
    if not isinstance(artifact, ConditionalAllocationMixture):
        raise TypeError("artifact must be a ConditionalAllocationMixture.")
    observed = _readonly_float(observation, name="observation", ndim=1)
    if observed.shape != (artifact.observation_count,):
        raise ValueError("observation must have one entry per artifact row.")
    retained_masses = _readonly_float(masses, name="masses", ndim=1)
    if retained_masses.shape != (artifact.region_count,) or np.any(retained_masses < 0.0):
        raise ValueError("masses must contain one finite non-negative value per region.")
    if mean_offset is None:
        offset = np.zeros(artifact.observation_count, dtype=np.float64)
        offset.setflags(write=False)
    else:
        offset = _readonly_float(mean_offset, name="mean_offset", ndim=1)
        if offset.shape != (artifact.observation_count,):
            raise ValueError("mean_offset must have one entry per observation.")
    return observed, retained_masses, cast(FloatArray, offset)


def _conditional_log_density_terms(
    observed: FloatArray,
    retained_masses: FloatArray,
    offset: FloatArray,
    artifact: ConditionalAllocationMixture,
) -> tuple[
    float,
    FloatArray,
    FloatArray | None,
    FloatArray | None,
    float | None,
]:
    """Return log density and only the intermediates needed by its gradient."""
    expected = offset + artifact.observation_mean_design @ retained_masses
    whitened_residual = (observed - expected) / artifact.noise_sd
    if not np.any(artifact.projected_unit_mass_residual_factors):
        result = -float(np.sum(np.log(artifact.noise_sd))) - 0.5 * (
            artifact.observation_count * _LOG_TWO_PI + float(whitened_residual @ whitened_residual)
        )
        if not math.isfinite(result):
            raise ValueError("conditional mixture evaluation produced a non-finite result.")
        return (
            result,
            cast(FloatArray, whitened_residual),
            None,
            None,
            None,
        )

    summary_residual = artifact.summary_basis.T @ whitened_residual
    orthogonal_residual = whitened_residual - artifact.summary_basis @ summary_residual
    component_means = np.einsum(
        "sqr,r->sq",
        artifact.projected_unit_mass_residual_factors,
        retained_masses,
        optimize=False,
    )
    displacements = summary_residual[np.newaxis, :] - component_means
    component_log_densities = -0.5 * (
        artifact.summary_rank * _LOG_TWO_PI
        + np.einsum(
            "sq,sq->s",
            displacements,
            displacements,
            optimize=False,
        )
    )
    log_component_total = float(np.logaddexp.reduce(component_log_densities))
    log_summary_density = log_component_total - math.log(artifact.sample_count)
    log_orthogonal_density = -0.5 * (
        (artifact.observation_count - artifact.summary_rank) * _LOG_TWO_PI
        + float(orthogonal_residual @ orthogonal_residual)
    )
    result = -float(np.sum(np.log(artifact.noise_sd))) + log_orthogonal_density + log_summary_density
    if not math.isfinite(result):
        raise ValueError("conditional mixture evaluation produced a non-finite result.")
    return (
        result,
        cast(FloatArray, whitened_residual),
        cast(FloatArray, orthogonal_residual),
        cast(FloatArray, displacements),
        log_component_total,
    )


def conditional_allocation_mixture_log_likelihood_and_gradient(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalAllocationMixture,
    *,
    mean_offset: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    """Evaluate the normalized frozen mixture and analytic mass gradient.

    The equal mixture weights and the complement Gaussian normalizer are
    included.  The returned derivative accounts for both the conditional mean
    ``A @ masses`` and every component mean ``F[s] @ masses``.

    Args:
        observation: Finite observation vector.
        masses: One finite non-negative retained mass per artifact region.
        artifact: Frozen conditional-allocation mixture.
        mean_offset: Optional fixed observation-space contribution.

    Returns:
        Pair of normalized approximate log density and a read-only gradient
        with respect to ``masses``.

    Raises:
        TypeError: If ``artifact`` has the wrong type.
        ValueError: If evaluation arrays are malformed or the result is
            non-finite.
    """
    observed, retained_masses, offset = _validated_evaluation(
        observation,
        masses,
        artifact,
        mean_offset,
    )
    (
        result,
        whitened_residual,
        orthogonal_residual,
        displacements,
        log_component_total,
    ) = _conditional_log_density_terms(
        observed,
        retained_masses,
        offset,
        artifact,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        whitened_mean_design = artifact.observation_mean_design / artifact.noise_sd[:, np.newaxis]
    if orthogonal_residual is None:
        gradient = whitened_mean_design.T @ whitened_residual
        if not np.all(np.isfinite(gradient)):
            raise ValueError("conditional mixture mass gradient is non-finite.")
        return result, _readonly_float(
            gradient,
            name="mass_gradient",
            ndim=1,
        )
    assert displacements is not None
    assert log_component_total is not None
    component_log_densities = -0.5 * (
        artifact.summary_rank * _LOG_TWO_PI
        + np.einsum(
            "sq,sq->s",
            displacements,
            displacements,
            optimize=False,
        )
    )
    responsibilities = np.exp(component_log_densities - log_component_total)
    summary_mean_design = artifact.summary_basis.T @ whitened_mean_design
    weighted_displacement = responsibilities @ displacements
    bank_gradient = np.einsum(
        "s,sqr,sq->r",
        responsibilities,
        artifact.projected_unit_mass_residual_factors,
        displacements,
        optimize=False,
    )
    gradient = (
        whitened_mean_design.T @ orthogonal_residual
        + summary_mean_design.T @ weighted_displacement
        + bank_gradient
    )
    if not np.all(np.isfinite(gradient)):
        raise ValueError("conditional mixture mass gradient is non-finite.")
    return result, _readonly_float(gradient, name="mass_gradient", ndim=1)


def conditional_allocation_mixture_log_likelihood(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalAllocationMixture,
    *,
    mean_offset: ArrayLike | None = None,
) -> float:
    """Evaluate only the normalized frozen conditional mixture log density.

    Args:
        observation: Finite vector aligned with artifact observations.
        masses: One finite non-negative retained mass per region.
        artifact: Frozen conditional-allocation mixture.
        mean_offset: Optional fixed observation-space contribution.

    Returns:
        Normalized approximate conditional log density.
    """
    observed, retained_masses, offset = _validated_evaluation(
        observation,
        masses,
        artifact,
        mean_offset,
    )
    result, _, _, _, _ = _conditional_log_density_terms(
        observed,
        retained_masses,
        offset,
        artifact,
    )
    return result
