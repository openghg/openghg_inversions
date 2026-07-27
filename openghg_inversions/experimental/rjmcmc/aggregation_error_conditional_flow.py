"""Normalized conditional-flow likelihood and simulator for aggregation error.

This optional experimental module represents the complete aggregation-residual
image with a conditional normalizing flow.  The projected measurement noise is
included in the simulations used to train the flow.  The exact Gaussian
orthogonal complement remains analytic.

For retained masses ``m``, error-whitened observation residual ``r``, complete
residual-image basis ``Q``, and exact unit-mass residual covariances ``C_j``,
define

```
z = Q.T @ r
V(m) = I + sum_j m[j]**2 * C_j
L(m) @ L(m).T = V(m)
u = solve(L(m), z).
```

The fitted flow is a normalized density ``q(u | c(m))``.  Consequently the
implemented observation density is normalized and has an exact forward
simulator:

```
p(y | m) = prod_i noise_sd[i]**(-1)
           * phi(r - Q @ z)
           * q(u | c(m))
           * det(L(m))**(-1).
```

Importing this module requires the optional ``nle`` dependency group.
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import io
import json
import math
from numbers import Integral
import struct
from typing import Any, TypeAlias, cast

import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import triangular_spline_flow
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .aggregation_error_conditional_mdn import ResidualImageContext
from .aggregation_error_low_rank import AdditiveDirichletAggregation

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ConditionalResidualImageFlow",
    "FLOW_ARCHITECTURE",
    "FLOW_LAYERS",
    "FLOW_KNOTS",
    "FLOW_TANH_MAX",
    "conditional_residual_unit_covariances",
    "make_conditional_residual_flow",
]

_ARTIFACT_SCHEMA = "aggregation-conditional-residual-image-flow-v1"
_ARTIFACT_MAGIC = b"OpenGHG-conditional-residual-flow-v1\0"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64

FLOW_ARCHITECTURE = "flowjax-triangular-spline-flow-v1"
FLOW_LAYERS = 8
FLOW_KNOTS = 8
FLOW_TANH_MAX = 3.0
FLOW_INVERT = True
FLOWJAX_VERSION = version("flowjax")
JAX_VERSION = version("jax")
JAXLIB_VERSION = version("jaxlib")
EQUINOX_VERSION = version("equinox")
OPTAX_VERSION = version("optax")
PARAMAX_VERSION = version("paramax")


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    """Return the lower-case SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: str, *, name: str) -> str:
    """Return a validated lower-case SHA-256 digest."""
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def _source_seed(value: int) -> int:
    """Return an unsigned 32-bit seed accepted by JAX."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("source_seed must be an integer.")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("source_seed must lie in [0, 2**32).")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _readonly_float(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    """Return an immutable finite float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)
    return cast(FloatArray, immutable)


def _array_sha256(values: ArrayLike) -> str:
    """Return the context-compatible float64 array digest."""
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": "<f8",
                "shape": list(canonical.shape),
            }
        ).encode("ascii")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validate_flow_arrays(flow: Any) -> None:
    """Require all inexact flow leaves to be finite float64 arrays."""
    found_array = False
    for leaf in jax.tree_util.tree_leaves(flow):
        if not eqx.is_inexact_array(leaf):
            continue
        found_array = True
        array = np.asarray(leaf)
        if array.dtype != np.dtype(np.float64):
            raise ValueError("flow parameters must use float64.")
        if not np.all(np.isfinite(array)):
            raise ValueError("flow parameters must be finite.")
    if not found_array:
        raise ValueError("flow must contain trainable floating-point arrays.")


def make_conditional_residual_flow(
    residual_rank: int,
    conditioner_dimension: int,
    *,
    source_seed: int,
) -> Any:
    """Construct the source-pinned conditional triangular spline flow."""
    rank = _positive_integer(residual_rank, name="residual_rank")
    conditioner = _positive_integer(
        conditioner_dimension,
        name="conditioner_dimension",
    )
    seed = _source_seed(source_seed)
    base = Normal(jnp.zeros(rank, dtype=jnp.float64))
    result = triangular_spline_flow(
        key=jr.key(seed),
        base_dist=base,
        cond_dim=conditioner,
        flow_layers=FLOW_LAYERS,
        knots=FLOW_KNOTS,
        tanh_max_val=FLOW_TANH_MAX,
        invert=FLOW_INVERT,
    )
    _validate_flow_arrays(result)
    return result


def conditional_residual_unit_covariances(
    aggregation: AdditiveDirichletAggregation,
    context: ResidualImageContext,
) -> FloatArray:
    """Return exact per-region unit-mass residual-image covariances.

    The context's canonical labels align cell-for-cell with the aggregation
    arrays.  For Dirichlet proportions ``p`` with total concentration
    ``alpha_0``,

    ``Cov(p) = (diag(alpha / alpha_0) - mean outer mean) / (alpha_0 + 1)``.
    """
    if not isinstance(aggregation, AdditiveDirichletAggregation):
        raise TypeError("aggregation must be an AdditiveDirichletAggregation.")
    if not isinstance(context, ResidualImageContext):
        raise TypeError("context must be a ResidualImageContext.")
    if _array_sha256(aggregation.cell_alphas) != context.cell_alphas_sha256:
        raise ValueError("aggregation cell_alphas do not match the context.")
    if _array_sha256(aggregation.design) != context.design_sha256:
        raise ValueError("aggregation design does not match the context.")
    if _array_sha256(aggregation.noise_sd) != context.noise_sd_sha256:
        raise ValueError("aggregation noise_sd does not match the context.")

    flat_alphas = np.asarray(aggregation.cell_alphas, dtype=np.float64).reshape(-1)
    flat_labels = np.asarray(context.labels, dtype=np.int64).reshape(-1)
    whitened_design = np.asarray(
        aggregation.design / aggregation.noise_sd[:, np.newaxis],
        dtype=np.float64,
    )
    covariances = np.empty(
        (context.region_count, context.residual_rank, context.residual_rank),
        dtype=np.float64,
    )
    for region in range(context.region_count):
        selected = np.flatnonzero(flat_labels == region)
        alpha = flat_alphas[selected]
        alpha_total = float(np.sum(alpha))
        mean = alpha / alpha_total
        proportion_covariance = (
            np.diag(mean) - np.outer(mean, mean)
        ) / (alpha_total + 1.0)
        projected_design = (
            context.residual_basis.T @ whitened_design[:, selected]
        )
        covariance = (
            projected_design @ proportion_covariance @ projected_design.T
        )
        covariances[region] = 0.5 * (covariance + covariance.T)

    tolerance = 512.0 * np.finfo(np.float64).eps * max(
        1,
        context.residual_rank,
        context.region_count,
    )
    for covariance in covariances:
        if float(np.min(np.linalg.eigvalsh(covariance))) < -tolerance:
            raise ValueError("computed unit residual covariance is not positive semidefinite.")
    return _readonly_float(
        covariances,
        name="unit_residual_covariances",
        ndim=3,
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConditionalResidualImageFlow:
    """Authenticated normalized conditional likelihood and forward simulator."""

    context: ResidualImageContext = field(init=False)
    unit_residual_covariances: FloatArray = field(init=False)
    conditioner_center: FloatArray = field(init=False)
    conditioner_scale: FloatArray = field(init=False)
    flow: Any = field(init=False, repr=False)
    initialization_seed: int = field(init=False)
    source_provenance: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        context: ResidualImageContext,
        unit_residual_covariances: ArrayLike,
        conditioner_center: ArrayLike,
        conditioner_scale: ArrayLike,
        flow: Any,
        *,
        initialization_seed: int,
        source_provenance: str,
    ) -> None:
        """Validate and own the complete fitted artifact."""
        if not isinstance(context, ResidualImageContext):
            raise TypeError("context must be a ResidualImageContext.")
        covariances = _readonly_float(
            unit_residual_covariances,
            name="unit_residual_covariances",
            ndim=3,
        )
        expected_covariance_shape = (
            context.region_count,
            context.residual_rank,
            context.residual_rank,
        )
        if covariances.shape != expected_covariance_shape:
            raise ValueError(
                "unit_residual_covariances must have shape "
                "(region_count, residual_rank, residual_rank)."
            )
        tolerance = 512.0 * np.finfo(np.float64).eps * max(
            1,
            context.region_count,
            context.residual_rank,
        )
        for covariance in covariances:
            if not np.allclose(
                covariance,
                covariance.T,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("unit residual covariances must be symmetric.")
            if float(np.min(np.linalg.eigvalsh(covariance))) < -tolerance:
                raise ValueError("unit residual covariances must be positive semidefinite.")

        center = _readonly_float(
            conditioner_center,
            name="conditioner_center",
            ndim=1,
        )
        scale = _readonly_float(
            conditioner_scale,
            name="conditioner_scale",
            ndim=1,
        )
        if center.shape != (context.region_count,):
            raise ValueError("conditioner_center must have one value per retained region.")
        if scale.shape != center.shape or np.any(scale <= 0.0):
            raise ValueError("conditioner_scale must be positive and match conditioner_center.")
        if not isinstance(source_provenance, str) or not source_provenance:
            raise ValueError("source_provenance must be a non-empty string.")
        if len(source_provenance) > 4_096:
            raise ValueError("source_provenance is too long.")
        seed = _source_seed(initialization_seed)

        if getattr(flow, "shape", None) != (context.residual_rank,):
            raise ValueError("flow event shape must match the residual rank.")
        if getattr(flow, "cond_shape", None) != (context.region_count,):
            raise ValueError("flow condition shape must match the retained-region count.")
        _validate_flow_arrays(flow)

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "unit_residual_covariances", covariances)
        object.__setattr__(self, "conditioner_center", center)
        object.__setattr__(self, "conditioner_scale", scale)
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "initialization_seed", seed)
        object.__setattr__(self, "source_provenance", source_provenance)
        object.__setattr__(self, "artifact_sha256", _sha256_bytes(self.to_bytes()))

    @property
    def residual_rank(self) -> int:
        """Return the residual-image rank."""
        return self.context.residual_rank

    @property
    def region_count(self) -> int:
        """Return the retained-region count."""
        return self.context.region_count

    @property
    def sha256(self) -> str:
        """Return the authenticated artifact digest."""
        return self.artifact_sha256

    def _validated_masses(self, masses: ArrayLike) -> FloatArray:
        """Return finite positive masses in canonical context order."""
        result = np.asarray(masses, dtype=np.float64)
        if (
            result.shape != (self.region_count,)
            or not np.all(np.isfinite(result))
            or np.any(result <= 0.0)
        ):
            raise ValueError("masses must contain one finite strictly positive value per region.")
        if not np.isfinite(float(np.sum(result))):
            raise ValueError("mass total must be finite.")
        return cast(FloatArray, result)

    def conditioner(self, masses: ArrayLike) -> FloatArray:
        """Return the standardized log-total and additive-log-ratio context."""
        retained = self._validated_masses(masses)
        return cast(FloatArray, self.conditioners(retained[np.newaxis, :])[0])

    def conditioners(self, masses: ArrayLike) -> FloatArray:
        """Return standardized conditioners for a matrix of canonical masses."""
        retained = np.asarray(masses, dtype=np.float64)
        if (
            retained.ndim != 2
            or retained.shape[1:] != (self.region_count,)
            or not np.all(np.isfinite(retained))
            or np.any(retained <= 0.0)
        ):
            raise ValueError(
                "masses must be a matrix with one finite strictly positive column per region."
            )
        totals = np.sum(retained, axis=1)
        if not np.all(np.isfinite(totals)):
            raise ValueError("mass totals must be finite.")
        raw = np.empty_like(retained)
        raw[:, 0] = np.log(totals)
        if self.region_count > 1:
            raw[:, 1:] = (
                np.log(retained[:, :-1])
                - np.log(retained[:, -1:])
            )
        result = (
            raw - self.conditioner_center[np.newaxis, :]
        ) / self.conditioner_scale[np.newaxis, :]
        if not np.all(np.isfinite(result)):
            raise ValueError("standardized flow conditioners are non-finite.")
        return cast(FloatArray, result)

    def projected_cholesky(self, masses: ArrayLike) -> FloatArray:
        """Return the exact projected noisy-residual Cholesky factor."""
        retained = self._validated_masses(masses)
        return cast(
            FloatArray,
            self.projected_choleskies(retained[np.newaxis, :])[0],
        )

    def projected_choleskies(self, masses: ArrayLike) -> FloatArray:
        """Return projected Cholesky factors for a matrix of canonical masses."""
        retained = np.asarray(masses, dtype=np.float64)
        if (
            retained.ndim != 2
            or retained.shape[1:] != (self.region_count,)
            or not np.all(np.isfinite(retained))
            or np.any(retained <= 0.0)
        ):
            raise ValueError(
                "masses must be a matrix with one finite strictly positive column per region."
            )
        covariance = np.eye(
            self.residual_rank,
            dtype=np.float64,
        )[np.newaxis, :, :] + np.einsum(
            "nj,jab->nab",
            retained * retained,
            self.unit_residual_covariances,
        )
        try:
            result = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:  # pragma: no cover - guarded by construction
            raise ValueError("projected residual covariance is not positive definite.") from error
        return cast(FloatArray, result)

    @staticmethod
    def _offset(
        offset: ArrayLike,
        observation_count: int,
    ) -> FloatArray:
        """Return a finite observation-aligned offset."""
        raw = np.asarray(offset, dtype=np.float64)
        if raw.ndim == 0:
            result = np.full(observation_count, float(raw), dtype=np.float64)
        else:
            result = raw
        if result.shape != (observation_count,) or not np.all(np.isfinite(result)):
            raise ValueError("offset must be finite and scalar or aligned with observation.")
        return cast(FloatArray, result)

    def log_likelihood(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> float:
        """Evaluate the normalized fitted observation log density."""
        observed = np.asarray(observation, dtype=np.float64)
        if (
            observed.shape != (self.context.observation_count,)
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError("observation must be finite with one value per context observation.")
        retained = self._validated_masses(masses)
        fixed_offset = self._offset(offset, self.context.observation_count)
        residual = (
            observed
            - fixed_offset
            - self.context.observation_mean_design @ retained
        ) / self.context.noise_sd
        coordinates = self.context.residual_basis.T @ residual
        orthogonal = residual - self.context.residual_basis @ coordinates
        cholesky = self.projected_cholesky(retained)
        standardized = np.linalg.solve(cholesky, coordinates)
        flow_log_density = float(
            self.flow.log_prob(
                jnp.asarray(standardized, dtype=jnp.float64),
                jnp.asarray(self.conditioner(retained), dtype=jnp.float64),
            )
        )
        result = (
            -float(np.sum(np.log(self.context.noise_sd)))
            - 0.5
            * (
                (self.context.observation_count - self.residual_rank) * _LOG_TWO_PI
                + float(orthogonal @ orthogonal)
            )
            + flow_log_density
            - float(np.sum(np.log(np.diag(cholesky))))
        )
        if not np.isfinite(result):
            raise ValueError("conditional residual-image flow log density is non-finite.")
        return result

    def log_likelihood_batch(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> FloatArray:
        """Evaluate one observation at a matrix of retained-mass states."""
        observed = np.asarray(observation, dtype=np.float64)
        if (
            observed.shape != (self.context.observation_count,)
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError("observation must be finite with one value per context observation.")
        retained = np.asarray(masses, dtype=np.float64)
        conditioners = self.conditioners(retained)
        fixed_offset = self._offset(offset, self.context.observation_count)
        residual = (
            observed[np.newaxis, :]
            - fixed_offset[np.newaxis, :]
            - retained @ self.context.observation_mean_design.T
        ) / self.context.noise_sd[np.newaxis, :]
        coordinates = residual @ self.context.residual_basis
        orthogonal = (
            residual
            - coordinates @ self.context.residual_basis.T
        )
        cholesky = self.projected_choleskies(retained)
        standardized = np.linalg.solve(
            cholesky,
            coordinates[:, :, np.newaxis],
        )[:, :, 0]
        flow_log_density = np.asarray(
            self.flow.log_prob(
                jnp.asarray(standardized, dtype=jnp.float64),
                jnp.asarray(conditioners, dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        result = (
            -float(np.sum(np.log(self.context.noise_sd)))
            - 0.5
            * (
                (self.context.observation_count - self.residual_rank) * _LOG_TWO_PI
                + np.sum(orthogonal * orthogonal, axis=1)
            )
            + flow_log_density
            - np.sum(
                np.log(
                    np.diagonal(
                        cholesky,
                        axis1=1,
                        axis2=2,
                    )
                ),
                axis=1,
            )
        )
        if result.shape != (retained.shape[0],) or not np.all(np.isfinite(result)):
            raise ValueError("conditional residual-image flow log densities are non-finite.")
        return cast(FloatArray, result)

    def sample_observation(
        self,
        masses: ArrayLike,
        *,
        sample_count: int,
        source_seed: int,
        offset: ArrayLike = 0.0,
    ) -> FloatArray:
        """Draw observations from the same normalized fitted marginal model."""
        retained = self._validated_masses(masses)
        count = _positive_integer(sample_count, name="sample_count")
        seed = _source_seed(source_seed)
        fixed_offset = self._offset(offset, self.context.observation_count)
        flow_key, orthogonal_key = jr.split(jr.key(seed))
        standardized = np.asarray(
            self.flow.sample(
                flow_key,
                (count,),
                condition=jnp.asarray(
                    self.conditioner(retained),
                    dtype=jnp.float64,
                ),
            ),
            dtype=np.float64,
        )
        if standardized.shape != (count, self.residual_rank):
            raise RuntimeError("flow returned an unexpected sample shape.")
        cholesky = self.projected_cholesky(retained)
        coordinates = standardized @ cholesky.T
        gaussian = np.asarray(
            jr.normal(
                orthogonal_key,
                (count, self.context.observation_count),
                dtype=jnp.float64,
            ),
            dtype=np.float64,
        )
        basis = self.context.residual_basis
        orthogonal = gaussian - (gaussian @ basis) @ basis.T
        residual = coordinates @ basis.T + orthogonal
        mean = fixed_offset + self.context.observation_mean_design @ retained
        result = mean + residual * self.context.noise_sd
        if not np.all(np.isfinite(result)):
            raise ValueError("conditional residual-image flow samples are non-finite.")
        return cast(FloatArray, result)

    @property
    def metadata_payload(self) -> dict[str, object]:
        """Return the strict JSON-compatible non-parameter metadata."""
        return {
            "schema": _ARTIFACT_SCHEMA,
            "context": self.context.payload,
            "context_sha256": self.context.artifact_sha256,
            "unit_residual_covariances": self.unit_residual_covariances.tolist(),
            "conditioner_center": self.conditioner_center.tolist(),
            "conditioner_scale": self.conditioner_scale.tolist(),
            "architecture": {
                "name": FLOW_ARCHITECTURE,
                "flow_layers": FLOW_LAYERS,
                "knots": FLOW_KNOTS,
                "tanh_max_val": FLOW_TANH_MAX,
                "invert": FLOW_INVERT,
            },
            "runtime": {
                "flowjax": FLOWJAX_VERSION,
                "jax": JAX_VERSION,
                "jaxlib": JAXLIB_VERSION,
                "equinox": EQUINOX_VERSION,
                "optax": OPTAX_VERSION,
                "paramax": PARAMAX_VERSION,
            },
            "initialization_seed": self.initialization_seed,
            "source_provenance": self.source_provenance,
        }

    def to_bytes(self) -> bytes:
        """Serialize metadata and fitted flow leaves to canonical bytes."""
        metadata = _canonical_json(self.metadata_payload).encode("utf-8")
        buffer = io.BytesIO()
        buffer.write(_ARTIFACT_MAGIC)
        buffer.write(struct.pack("<Q", len(metadata)))
        buffer.write(metadata)
        eqx.tree_serialise_leaves(buffer, self.flow)
        return buffer.getvalue()

    @classmethod
    def from_bytes(
        cls,
        serialized: bytes,
        *,
        expected_sha256: str,
    ) -> ConditionalResidualImageFlow:
        """Authenticate and reconstruct a fitted flow artifact."""
        if not isinstance(serialized, bytes):
            raise TypeError("serialized flow artifact must be bytes.")
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if _sha256_bytes(serialized) != expected:
            raise ValueError("flow artifact SHA-256 fingerprint does not match.")
        buffer = io.BytesIO(serialized)
        if buffer.read(len(_ARTIFACT_MAGIC)) != _ARTIFACT_MAGIC:
            raise ValueError("serialized flow artifact has an unexpected magic header.")
        encoded_length = buffer.read(8)
        if len(encoded_length) != 8:
            raise ValueError("serialized flow artifact metadata length is truncated.")
        metadata_length = struct.unpack("<Q", encoded_length)[0]
        metadata_bytes = buffer.read(metadata_length)
        if len(metadata_bytes) != metadata_length:
            raise ValueError("serialized flow artifact metadata is truncated.")
        try:
            payload = json.loads(metadata_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("serialized flow artifact metadata is not valid JSON.") from error
        if _canonical_json(payload).encode("utf-8") != metadata_bytes:
            raise ValueError("serialized flow artifact metadata must use canonical JSON.")
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "context",
            "context_sha256",
            "unit_residual_covariances",
            "conditioner_center",
            "conditioner_scale",
            "architecture",
            "runtime",
            "initialization_seed",
            "source_provenance",
        }:
            raise ValueError("serialized flow artifact metadata has an unexpected schema.")
        if payload["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError("serialized flow artifact has an unexpected schema.")
        expected_architecture = {
            "name": FLOW_ARCHITECTURE,
            "flow_layers": FLOW_LAYERS,
            "knots": FLOW_KNOTS,
            "tanh_max_val": FLOW_TANH_MAX,
            "invert": FLOW_INVERT,
        }
        if payload["architecture"] != expected_architecture:
            raise ValueError("serialized flow artifact architecture does not match.")
        expected_runtime = {
            "flowjax": FLOWJAX_VERSION,
            "jax": JAX_VERSION,
            "jaxlib": JAXLIB_VERSION,
            "equinox": EQUINOX_VERSION,
            "optax": OPTAX_VERSION,
            "paramax": PARAMAX_VERSION,
        }
        if payload["runtime"] != expected_runtime:
            raise ValueError("serialized flow artifact runtime does not match.")
        context = ResidualImageContext.from_json(
            _canonical_json(payload["context"]),
            expected_sha256=payload["context_sha256"],
        )
        template = make_conditional_residual_flow(
            context.residual_rank,
            context.region_count,
            source_seed=payload["initialization_seed"],
        )
        try:
            fitted_flow = eqx.tree_deserialise_leaves(buffer, template)
        except Exception as error:
            raise ValueError("serialized flow parameter leaves are invalid.") from error
        result = cls(
            context,
            payload["unit_residual_covariances"],
            payload["conditioner_center"],
            payload["conditioner_scale"],
            fitted_flow,
            initialization_seed=payload["initialization_seed"],
            source_provenance=payload["source_provenance"],
        )
        if result.to_bytes() != serialized:
            raise ValueError("serialized flow artifact does not replay canonically.")
        if result.artifact_sha256 != expected:  # pragma: no cover - implied above
            raise ValueError("flow artifact SHA-256 fingerprint does not match.")
        return result
