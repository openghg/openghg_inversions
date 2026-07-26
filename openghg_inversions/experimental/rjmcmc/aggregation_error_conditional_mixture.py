"""Frozen conditional-allocation mixtures for aggregation error.

This module provides a deterministic finite-mixture baseline between exact
small-problem quadrature and a learned neural likelihood.  For one fixed,
labelled partition it draws a frozen bank of independent within-region
Dirichlet allocations from the common additive native concentration field.
The resulting projected unit-mass aggregation residuals are stored in an
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

Artifact construction is the only operation that uses randomness.  It uses
seeded PCG64 streams keyed by stable native-cell identifiers.  Log-density
and gradient evaluation never draw and are exactly replayable from the
serialized artifact.
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
    "ConditionalAllocationMixture",
    "conditional_allocation_mixture_log_likelihood",
    "conditional_allocation_mixture_log_likelihood_and_gradient",
]

_ARTIFACT_SCHEMA = "aggregation-conditional-allocation-mixture-v1"
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
    digest = hashlib.sha256(_ARTIFACT_SCHEMA.encode("ascii"))
    digest.update(source_seed.to_bytes(8, byteorder="little", signed=False))
    digest.update(np.ascontiguousarray(sorted_cell_ids, dtype="<i8").tobytes())
    region_seed = int.from_bytes(digest.digest(), byteorder="little", signed=False)
    return np.random.Generator(np.random.PCG64(region_seed))


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
            sample_count: Number of independent allocation draws.
            source_seed: Seed used to derive independent per-region PCG64
                streams.
            source_provenance: Human-readable origin of the bank.
            cell_ids: Optional unique integer scientific cell identifiers.

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
        for region in range(factors.region_count):
            selected_indices = np.flatnonzero(flat_labels == region)
            order = np.argsort(flat_ids[selected_indices], kind="stable")
            sorted_indices = selected_indices[order]
            if sorted_indices.size == 1:
                # A one-cell Dirichlet distribution is the exact point mass
                # at one.  NumPy's generic Gamma normalization can return a
                # value a few ULP from one, which must not manufacture
                # aggregation error in the fine-cell limit.
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
        )

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
        return {
            "schema": _ARTIFACT_SCHEMA,
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
        expected_fields = {
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
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("serialized artifact has unexpected fields.")
        if payload["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError("serialized artifact schema is unsupported.")
        if payload["bit_generator"] != "PCG64":
            raise ValueError("serialized artifact bit generator is unsupported.")
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
        )
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
