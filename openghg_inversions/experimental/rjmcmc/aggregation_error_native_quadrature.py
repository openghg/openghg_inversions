"""Support-exact native-allocation quadrature marginals.

This module constructs a deterministic weighted Gaussian mixture for the
conditional observation law induced by a root Gamma--Dirichlet allocation.
The allocation coordinates use normalized Gauss--Jacobi probability rules,
so boundary singularities remain in their native Beta measure rather than
being fitted in Euclidean residual coordinates.

For retained masses ``m``, error-whitened residual ``r``, complete residual
image basis ``Q``, deterministic component factors ``F_s``, and normalized
weights ``w_s``, the implemented density is

```
z = Q.T @ r
v = r - Q @ z

p(y | m) = prod_i noise_sd[i]**(-1) * phi(v)
           * sum_s w_s phi(z; F_s @ m, I).
```

The same weighted components are used by :meth:`ConditionalNativeQuadrature.sample`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
import struct
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy

from .aggregation_error import beta_quadrature
from .aggregation_error_conditional_mdn import ResidualImageContext
from .aggregation_error_low_rank import AdditiveDirichletAggregation

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
QuadratureChart = Literal["single", "row-first", "column-first"]

__all__ = [
    "ConditionalNativeQuadrature",
    "QuadratureChart",
    "native_quadrature_log_likelihood",
    "native_quadrature_log_likelihood_and_gradient",
]

_SCHEMA = "aggregation-conditional-native-quadrature-v1"
_MAGIC = b"OGI-NATIVE-QUADRATURE-V1\n"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64
_GIT_SHA_LENGTH = 40
_DEFAULT_CHUNK_SIZE = 16_384


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    """Return the lower-case SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


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
                "shape": list(canonical.shape),
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
    """Return one finite immutable float64 array."""
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


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _validated_sha256(value: str, *, name: str) -> str:
    """Return one canonical lower-case SHA-256."""
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def _validated_git_sha(value: str) -> str:
    """Return one complete lower-case Git SHA."""
    if not isinstance(value, str) or len(value) != _GIT_SHA_LENGTH:
        raise ValueError("source_git_revision must be a complete lower-case Git SHA.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("source_git_revision must be a complete lower-case Git SHA.") from error
    if decoded.hex() != value:
        raise ValueError("source_git_revision must be a complete lower-case Git SHA.")
    return value


def _stable_logsumexp(values: FloatArray) -> float:
    """Return log-sum-exp of one non-empty finite vector."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log-sum-exp input must be a non-empty vector.")
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _chart(value: str, *, cell_count: int) -> QuadratureChart:
    """Return one supported chart for a two- or four-cell root."""
    if cell_count == 2 and value == "single":
        return "single"
    if cell_count == 4 and value in ("row-first", "column-first"):
        return cast(QuadratureChart, value)
    raise ValueError(
        "two-cell roots require 'single'; four-cell roots require 'row-first' or 'column-first'."
    )


def _ordered_indices(
    cell_ids: IntArray,
    *,
    chart: QuadratureChart,
) -> IntArray:
    """Return canonical native indices in the selected chart order."""
    sorted_indices = np.argsort(cell_ids.reshape(-1), kind="stable")
    if chart == "column-first":
        sorted_indices = sorted_indices[np.asarray([0, 2, 1, 3], dtype=np.int64)]
    return cast(IntArray, np.asarray(sorted_indices, dtype=np.int64))


def _root_fraction_quadrature(
    cell_alphas: FloatArray,
    cell_ids: IntArray,
    *,
    quadrature_order: int,
    chart: QuadratureChart,
) -> tuple[FloatArray, FloatArray]:
    """Return native fractions and log weights for one root allocation."""
    flat_alphas = cell_alphas.reshape(-1)
    flat_ids = cell_ids.reshape(-1)
    cell_count = flat_alphas.size
    normalized_chart = _chart(chart, cell_count=cell_count)
    ordered = _ordered_indices(flat_ids, chart=normalized_chart)
    ordered_alphas = flat_alphas[ordered]
    order = _positive_integer(quadrature_order, name="quadrature_order")

    if cell_count == 2:
        rule = beta_quadrature(
            float(ordered_alphas[0]),
            float(ordered_alphas[1]),
            order,
        )
        ordered_fractions = np.column_stack((rule.nodes, 1.0 - rule.nodes))
        log_weights = np.log(rule.weights)
    else:
        aggregate_rule = beta_quadrature(
            float(ordered_alphas[0] + ordered_alphas[1]),
            float(ordered_alphas[2] + ordered_alphas[3]),
            order,
        )
        first_rule = beta_quadrature(
            float(ordered_alphas[0]),
            float(ordered_alphas[1]),
            order,
        )
        second_rule = beta_quadrature(
            float(ordered_alphas[2]),
            float(ordered_alphas[3]),
            order,
        )
        flat_index = np.arange(order**3, dtype=np.int64)
        aggregate_index = flat_index // (order * order)
        first_index = (flat_index // order) % order
        second_index = flat_index % order
        aggregate = aggregate_rule.nodes[aggregate_index]
        first = first_rule.nodes[first_index]
        second = second_rule.nodes[second_index]
        ordered_fractions = np.empty((flat_index.size, 4), dtype=np.float64)
        ordered_fractions[:, 0] = aggregate * first
        ordered_fractions[:, 1] = aggregate * (1.0 - first)
        ordered_fractions[:, 2] = (1.0 - aggregate) * second
        ordered_fractions[:, 3] = (1.0 - aggregate) * (1.0 - second)
        log_weights = (
            np.log(aggregate_rule.weights[aggregate_index])
            + np.log(first_rule.weights[first_index])
            + np.log(second_rule.weights[second_index])
        )

    fractions = np.empty_like(ordered_fractions)
    fractions[:, ordered] = ordered_fractions
    row_sums = np.sum(fractions, axis=1)
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(fractions < 0.0) or np.any(fractions > 1.0) or np.max(np.abs(row_sums - 1.0)) > tolerance:
        raise RuntimeError("native quadrature did not preserve simplex support.")
    log_normalizer = _stable_logsumexp(cast(FloatArray, log_weights))
    normalized_log_weights = log_weights - log_normalizer
    return (
        _readonly_float(fractions, name="fractions", ndim=2),
        _readonly_float(
            normalized_log_weights,
            name="log_weights",
            ndim=1,
        ),
    )


def _array_metadata(values: FloatArray | IntArray) -> dict[str, object]:
    """Return canonical raw-array metadata."""
    dtype = "<f8" if np.issubdtype(values.dtype, np.floating) else "<i8"
    return {
        "dtype": dtype,
        "shape": list(values.shape),
        "sha256": _array_sha256(values),
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConditionalNativeQuadrature:
    """Authenticated support-aware weighted conditional mixture."""

    context: ResidualImageContext = field(init=False)
    projected_unit_mass_residual_factors: FloatArray = field(init=False)
    log_weights: FloatArray = field(init=False)
    cell_alphas: FloatArray = field(init=False)
    quadrature_order: int = field(init=False)
    chart: QuadratureChart = field(init=False)
    scipy_version: str = field(init=False)
    source_git_revision: str = field(init=False)
    driver_sha256: str = field(init=False)
    protocol_sha256: str = field(init=False)
    source_provenance: str = field(init=False)
    chunk_size: int = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        context: ResidualImageContext,
        projected_unit_mass_residual_factors: ArrayLike,
        log_weights: ArrayLike,
        cell_alphas: ArrayLike,
        *,
        quadrature_order: int,
        chart: QuadratureChart,
        scipy_version: str,
        source_git_revision: str,
        driver_sha256: str,
        protocol_sha256: str,
        source_provenance: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Validate, own, and fingerprint one quadrature artifact."""
        if not isinstance(context, ResidualImageContext):
            raise TypeError("context must be a ResidualImageContext.")
        factors = _readonly_float(
            projected_unit_mass_residual_factors,
            name="projected_unit_mass_residual_factors",
            ndim=3,
        )
        if (
            factors.shape[0] == 0
            or factors.shape[1] != context.residual_rank
            or factors.shape[2] != context.region_count
        ):
            raise ValueError("projected residual factors have incompatible dimensions.")
        weights = _readonly_float(log_weights, name="log_weights", ndim=1)
        if weights.shape != (factors.shape[0],):
            raise ValueError("log_weights must have one entry per component.")
        if abs(_stable_logsumexp(weights)) > 1.0e-12:
            raise ValueError("quadrature log weights must be normalized.")
        alphas = _readonly_float(cell_alphas, name="cell_alphas")
        if alphas.shape != context.labels.shape or np.any(alphas <= 0.0):
            raise ValueError("cell_alphas must match context labels and be positive.")
        order = _positive_integer(quadrature_order, name="quadrature_order")
        normalized_chart = _chart(chart, cell_count=alphas.size)
        expected_components = order if alphas.size == 2 else order**3
        if factors.shape[0] != expected_components:
            raise ValueError("component count does not match chart and quadrature order.")
        if not isinstance(scipy_version, str) or not scipy_version:
            raise ValueError("scipy_version must be a non-empty string.")
        provenance = str(source_provenance)
        if not provenance.strip():
            raise ValueError("source_provenance must be non-empty.")

        object.__setattr__(self, "context", context)
        object.__setattr__(
            self,
            "projected_unit_mass_residual_factors",
            factors,
        )
        object.__setattr__(self, "log_weights", weights)
        object.__setattr__(self, "cell_alphas", alphas)
        object.__setattr__(self, "quadrature_order", order)
        object.__setattr__(self, "chart", normalized_chart)
        object.__setattr__(self, "scipy_version", scipy_version)
        object.__setattr__(
            self,
            "source_git_revision",
            _validated_git_sha(source_git_revision),
        )
        object.__setattr__(
            self,
            "driver_sha256",
            _validated_sha256(driver_sha256, name="driver_sha256"),
        )
        object.__setattr__(
            self,
            "protocol_sha256",
            _validated_sha256(protocol_sha256, name="protocol_sha256"),
        )
        object.__setattr__(self, "source_provenance", provenance)
        object.__setattr__(
            self,
            "chunk_size",
            _positive_integer(chunk_size, name="chunk_size"),
        )
        object.__setattr__(self, "artifact_sha256", _sha256_bytes(self.to_bytes()))

    @classmethod
    def from_aggregation(
        cls,
        aggregation: AdditiveDirichletAggregation,
        labels: ArrayLike,
        cell_ids: ArrayLike,
        *,
        quadrature_order: int,
        chart: QuadratureChart,
        source_git_revision: str,
        driver_sha256: str,
        protocol_sha256: str,
        source_provenance: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> ConditionalNativeQuadrature:
        """Construct one root support-quadrature artifact."""
        if not isinstance(aggregation, AdditiveDirichletAggregation):
            raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
        owned_labels = _readonly_integer(labels, name="labels")
        if owned_labels.shape != aggregation.cell_shape or np.unique(owned_labels).size != 1:
            raise ValueError("native quadrature currently requires one root region.")
        owned_ids = _readonly_integer(cell_ids, name="cell_ids")
        if owned_ids.shape != aggregation.cell_shape:
            raise ValueError("cell_ids must match the native cell shape.")
        context = ResidualImageContext.from_aggregation(
            aggregation,
            owned_labels,
            owned_ids,
            source_provenance=source_provenance,
        )
        alphas = cast(FloatArray, aggregation.cell_alphas)
        fractions, log_weights = _root_fraction_quadrature(
            alphas,
            owned_ids,
            quadrature_order=quadrature_order,
            chart=chart,
        )
        summary_design = (context.residual_basis / context.noise_sd[:, np.newaxis]).T @ aggregation.design
        normalized_alphas = alphas.reshape(-1) / float(np.sum(alphas))
        expected = summary_design @ normalized_alphas
        residual = fractions @ summary_design.T - expected[np.newaxis, :]
        factors = residual[:, :, np.newaxis]
        return cls(
            context,
            factors,
            log_weights,
            alphas,
            quadrature_order=quadrature_order,
            chart=chart,
            scipy_version=scipy.__version__,
            source_git_revision=source_git_revision,
            driver_sha256=driver_sha256,
            protocol_sha256=protocol_sha256,
            source_provenance=source_provenance,
            chunk_size=chunk_size,
        )

    @property
    def component_count(self) -> int:
        """Return the number of weighted quadrature components."""
        return int(self.log_weights.size)

    @property
    def region_count(self) -> int:
        """Return the retained region count."""
        return self.context.region_count

    @property
    def residual_rank(self) -> int:
        """Return the complete aggregation-residual rank."""
        return self.context.residual_rank

    @property
    def observation_count(self) -> int:
        """Return the observation dimension."""
        return self.context.observation_count

    @property
    def normalized_weights(self) -> FloatArray:
        """Return immutable positive component probabilities."""
        result = np.exp(self.log_weights - _stable_logsumexp(self.log_weights))
        return _readonly_float(result, name="normalized_weights", ndim=1)

    @property
    def metadata(self) -> dict[str, object]:
        """Return canonical artifact metadata excluding raw array bytes."""
        arrays: dict[str, FloatArray | IntArray] = {
            "cell_alphas": self.cell_alphas,
            "log_weights": self.log_weights,
            "projected_unit_mass_residual_factors": (self.projected_unit_mass_residual_factors),
        }
        return {
            "schema": _SCHEMA,
            "context": self.context.payload,
            "quadrature_order": self.quadrature_order,
            "chart": self.chart,
            "scipy_version": self.scipy_version,
            "source_git_revision": self.source_git_revision,
            "driver_sha256": self.driver_sha256,
            "protocol_sha256": self.protocol_sha256,
            "source_provenance": self.source_provenance,
            "chunk_size": self.chunk_size,
            "array_order": sorted(arrays),
            "arrays": {name: _array_metadata(arrays[name]) for name in sorted(arrays)},
        }

    def to_bytes(self) -> bytes:
        """Return canonical non-pickle artifact bytes."""
        arrays = {
            "cell_alphas": self.cell_alphas,
            "log_weights": self.log_weights,
            "projected_unit_mass_residual_factors": (self.projected_unit_mass_residual_factors),
        }
        metadata = _canonical_json(self.metadata).encode("ascii")
        body = b"".join(
            np.ascontiguousarray(arrays[name], dtype="<f8").tobytes(order="C") for name in sorted(arrays)
        )
        return _MAGIC + struct.pack("<Q", len(metadata)) + metadata + body

    @classmethod
    def from_bytes(
        cls,
        serialized: bytes,
        *,
        expected_sha256: str,
    ) -> ConditionalNativeQuadrature:
        """Authenticate and replay canonical artifact bytes."""
        if not isinstance(serialized, bytes):
            raise TypeError("serialized artifact must be bytes.")
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if _sha256_bytes(serialized) != expected:
            raise ValueError("serialized artifact SHA-256 does not match.")
        prefix_size = len(_MAGIC) + 8
        if len(serialized) < prefix_size or not serialized.startswith(_MAGIC):
            raise ValueError("serialized artifact has an invalid magic value.")
        metadata_size = struct.unpack(
            "<Q",
            serialized[len(_MAGIC) : prefix_size],
        )[0]
        metadata_stop = prefix_size + metadata_size
        if metadata_stop > len(serialized):
            raise ValueError("serialized artifact metadata is truncated.")
        try:
            metadata_text = serialized[prefix_size:metadata_stop].decode("ascii")
            metadata = json.loads(metadata_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("serialized artifact metadata is invalid.") from error
        if _canonical_json(metadata) != metadata_text:
            raise ValueError("serialized artifact metadata must be canonical JSON.")
        if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
            raise ValueError("serialized artifact schema is unsupported.")
        required = {
            "schema",
            "context",
            "quadrature_order",
            "chart",
            "scipy_version",
            "source_git_revision",
            "driver_sha256",
            "protocol_sha256",
            "source_provenance",
            "chunk_size",
            "array_order",
            "arrays",
        }
        if set(metadata) != required:
            raise ValueError("serialized artifact metadata fields are invalid.")
        array_order = metadata["array_order"]
        arrays_metadata = metadata["arrays"]
        expected_order = [
            "cell_alphas",
            "log_weights",
            "projected_unit_mass_residual_factors",
        ]
        if array_order != expected_order or not isinstance(arrays_metadata, dict):
            raise ValueError("serialized artifact array order is invalid.")
        cursor = metadata_stop
        decoded: dict[str, FloatArray] = {}
        for name in expected_order:
            description = arrays_metadata.get(name)
            if (
                not isinstance(description, dict)
                or set(description) != {"dtype", "shape", "sha256"}
                or description["dtype"] != "<f8"
                or not isinstance(description["shape"], list)
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in description["shape"]
                )
            ):
                raise ValueError("serialized artifact array metadata is invalid.")
            shape = tuple(description["shape"])
            byte_count = math.prod(shape) * 8
            stop = cursor + byte_count
            if stop > len(serialized):
                raise ValueError("serialized artifact array data are truncated.")
            array = np.frombuffer(
                serialized[cursor:stop],
                dtype="<f8",
            ).reshape(shape)
            immutable = _readonly_float(array, name=name)
            if _array_sha256(immutable) != description["sha256"]:
                raise ValueError("serialized artifact array digest does not match.")
            decoded[name] = immutable
            cursor = stop
        if cursor != len(serialized):
            raise ValueError("serialized artifact has trailing bytes.")
        context_text = _canonical_json(metadata["context"])
        context = ResidualImageContext.from_json(
            context_text,
            expected_sha256=_sha256_bytes(context_text.encode("utf-8")),
        )
        result = cls(
            context,
            decoded["projected_unit_mass_residual_factors"],
            decoded["log_weights"],
            decoded["cell_alphas"],
            quadrature_order=metadata["quadrature_order"],
            chart=metadata["chart"],
            scipy_version=metadata["scipy_version"],
            source_git_revision=metadata["source_git_revision"],
            driver_sha256=metadata["driver_sha256"],
            protocol_sha256=metadata["protocol_sha256"],
            source_provenance=metadata["source_provenance"],
            chunk_size=metadata["chunk_size"],
        )
        if result.to_bytes() != serialized or result.artifact_sha256 != expected:
            raise ValueError("serialized artifact did not replay canonically.")
        return result

    def log_likelihood(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> float:
        """Evaluate the normalized finite conditional density."""
        return native_quadrature_log_likelihood(
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
        """Evaluate log density and analytic retained-mass gradient."""
        return native_quadrature_log_likelihood_and_gradient(
            observation,
            masses,
            self,
            mean_offset=mean_offset,
        )

    def log_likelihood_batch(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> FloatArray:
        """Evaluate one observation at a batch of retained states."""
        states = np.asarray(masses, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != self.region_count:
            raise ValueError("masses must have shape (states, regions).")
        result = np.asarray(
            [
                self.log_likelihood(
                    observation,
                    state,
                    mean_offset=mean_offset,
                )
                for state in states
            ],
            dtype=np.float64,
        )
        return _readonly_float(result, name="log_likelihood_batch", ndim=1)

    def sample_with_component_indices(
        self,
        masses: ArrayLike,
        *,
        sample_count: int,
        rng: np.random.Generator,
        mean_offset: ArrayLike | None = None,
    ) -> tuple[FloatArray, IntArray]:
        """Draw from the same weighted Gaussian mixture used by the density."""
        retained = _validated_masses(masses, self)
        count = _positive_integer(sample_count, name="sample_count")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        offset = _validated_offset(mean_offset, self)
        indices = np.asarray(
            rng.choice(
                self.component_count,
                size=count,
                replace=True,
                p=self.normalized_weights,
            ),
            dtype=np.int64,
        )
        component_means = np.einsum(
            "nqr,r->nq",
            self.projected_unit_mass_residual_factors[indices],
            retained,
            optimize=False,
        )
        whitened = rng.standard_normal((count, self.observation_count))
        if self.residual_rank:
            whitened += component_means @ self.context.residual_basis.T
        expected = offset + self.context.observation_mean_design @ retained
        observations = expected[np.newaxis, :] + whitened * self.context.noise_sd
        return (
            _readonly_float(observations, name="samples", ndim=2),
            _readonly_integer(indices, name="component_indices", ndim=1),
        )

    def sample(
        self,
        masses: ArrayLike,
        *,
        sample_count: int,
        rng: np.random.Generator,
        mean_offset: ArrayLike | None = None,
    ) -> FloatArray:
        """Draw observations from the finite quadrature marginal."""
        samples, _ = self.sample_with_component_indices(
            masses,
            sample_count=sample_count,
            rng=rng,
            mean_offset=mean_offset,
        )
        return samples

    def analytic_mean_and_covariance(
        self,
        masses: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the exact first two moments of this finite mixture."""
        retained = _validated_masses(masses, self)
        offset = _validated_offset(mean_offset, self)
        probabilities = self.normalized_weights
        component_means = np.einsum(
            "sqr,r->sq",
            self.projected_unit_mass_residual_factors,
            retained,
            optimize=False,
        )
        weighted_mean = probabilities @ component_means
        centered = component_means - weighted_mean[np.newaxis, :]
        covariance_image = np.einsum(
            "s,si,sj->ij",
            probabilities,
            centered,
            centered,
            optimize=False,
        )
        whitened_covariance = np.eye(
            self.observation_count,
            dtype=np.float64,
        )
        if self.residual_rank:
            whitened_covariance += (
                self.context.residual_basis @ covariance_image @ self.context.residual_basis.T
            )
        mean = (
            offset
            + self.context.observation_mean_design @ retained
            + self.context.noise_sd * (self.context.residual_basis @ weighted_mean)
        )
        covariance = (
            self.context.noise_sd[:, np.newaxis] * whitened_covariance * self.context.noise_sd[np.newaxis, :]
        )
        return (
            _readonly_float(mean, name="analytic_mean", ndim=1),
            _readonly_float(
                covariance,
                name="analytic_covariance",
                ndim=2,
            ),
        )


def _validated_masses(
    masses: ArrayLike,
    artifact: ConditionalNativeQuadrature,
) -> FloatArray:
    """Return one valid non-negative retained-mass vector."""
    retained = _readonly_float(masses, name="masses", ndim=1)
    if retained.shape != (artifact.region_count,) or np.any(retained < 0.0):
        raise ValueError("masses must contain one non-negative value per region.")
    return retained


def _validated_offset(
    mean_offset: ArrayLike | None,
    artifact: ConditionalNativeQuadrature,
) -> FloatArray:
    """Return one valid fixed observation offset."""
    if mean_offset is None:
        return _readonly_float(
            np.zeros(artifact.observation_count, dtype=np.float64),
            name="mean_offset",
            ndim=1,
        )
    offset = _readonly_float(mean_offset, name="mean_offset", ndim=1)
    if offset.shape != (artifact.observation_count,):
        raise ValueError("mean_offset must have one entry per observation.")
    return offset


def _validated_observation(
    observation: ArrayLike,
    artifact: ConditionalNativeQuadrature,
) -> FloatArray:
    """Return one valid observation vector."""
    observed = _readonly_float(observation, name="observation", ndim=1)
    if observed.shape != (artifact.observation_count,):
        raise ValueError("observation must have one entry per observation row.")
    return observed


def native_quadrature_log_likelihood_and_gradient(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalNativeQuadrature,
    *,
    mean_offset: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    """Evaluate normalized log density and analytic retained-mass gradient."""
    if not isinstance(artifact, ConditionalNativeQuadrature):
        raise TypeError("artifact must be a ConditionalNativeQuadrature.")
    observed = _validated_observation(observation, artifact)
    retained = _validated_masses(masses, artifact)
    offset = _validated_offset(mean_offset, artifact)
    expected = offset + artifact.context.observation_mean_design @ retained
    residual = (observed - expected) / artifact.context.noise_sd
    basis = artifact.context.residual_basis
    coordinates = basis.T @ residual
    orthogonal = residual - basis @ coordinates
    whitened_mean_design = artifact.context.observation_mean_design / artifact.context.noise_sd[:, np.newaxis]
    summary_mean_design = basis.T @ whitened_mean_design
    log_weight_normalizer = _stable_logsumexp(artifact.log_weights)

    log_component_total = -math.inf
    weighted_displacement = np.zeros(artifact.residual_rank, dtype=np.float64)
    bank_gradient = np.zeros(artifact.region_count, dtype=np.float64)
    component_chunks: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for start in range(0, artifact.component_count, artifact.chunk_size):
        stop = min(start + artifact.chunk_size, artifact.component_count)
        factors = artifact.projected_unit_mass_residual_factors[start:stop]
        component_means = np.einsum(
            "sqr,r->sq",
            factors,
            retained,
            optimize=False,
        )
        displacements = coordinates[np.newaxis, :] - component_means
        log_terms = artifact.log_weights[start:stop] - 0.5 * (
            artifact.residual_rank * _LOG_TWO_PI
            + np.einsum(
                "sq,sq->s",
                displacements,
                displacements,
                optimize=False,
            )
        )
        chunk_total = _stable_logsumexp(cast(FloatArray, log_terms))
        log_component_total = float(np.logaddexp(log_component_total, chunk_total))
        component_chunks.append(
            (
                cast(FloatArray, log_terms),
                cast(FloatArray, displacements),
                cast(FloatArray, factors),
            )
        )

    for log_terms, displacements, factors in component_chunks:
        responsibilities = np.exp(log_terms - log_component_total)
        weighted_displacement += responsibilities @ displacements
        bank_gradient += np.einsum(
            "s,sqr,sq->r",
            responsibilities,
            factors,
            displacements,
            optimize=False,
        )
    log_summary_density = log_component_total - log_weight_normalizer
    log_orthogonal_density = -0.5 * (
        (artifact.observation_count - artifact.residual_rank) * _LOG_TWO_PI + float(orthogonal @ orthogonal)
    )
    result = -float(np.sum(np.log(artifact.context.noise_sd))) + log_orthogonal_density + log_summary_density
    gradient = (
        whitened_mean_design.T @ orthogonal + summary_mean_design.T @ weighted_displacement + bank_gradient
    )
    if not math.isfinite(result) or not np.all(np.isfinite(gradient)):
        raise ValueError("native quadrature evaluation produced non-finite values.")
    return result, _readonly_float(gradient, name="mass_gradient", ndim=1)


def native_quadrature_log_likelihood(
    observation: ArrayLike,
    masses: ArrayLike,
    artifact: ConditionalNativeQuadrature,
    *,
    mean_offset: ArrayLike | None = None,
) -> float:
    """Evaluate only the normalized finite conditional log density."""
    result, _ = native_quadrature_log_likelihood_and_gradient(
        observation,
        masses,
        artifact,
        mean_offset=mean_offset,
    )
    return result
