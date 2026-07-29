"""Authenticated score-regularized likelihood for one projected root.

The artifact models the leading standardized projected residual with one
conditional FlowJAX density.  Remaining authenticated spectrum directions use
their analytic Gaussian moment closure and the orthogonal observation-space
complement remains standard Gaussian after measurement-noise whitening.

The mass-score methods differentiate the residual density while holding the
projected residual fixed.  They deliberately do not include the derivative
of the conditional observation mean when a raw observation is held fixed.
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import io
import json
import math
from numbers import Integral, Real
import struct
from typing import Any, TypeAlias, cast

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .aggregation_error_exact_mixture import RootResidualSpectrum
from .aggregation_error_score_flow_training import (
    FLOW_INVERT,
    FLOW_LAYERS,
    FLOW_NN_DEPTH,
    FLOW_NN_WIDTH,
    FLOW_SPLINE_INTERVAL,
    FLOW_SPLINE_KNOTS,
    gamma_log_mass_conditioning,
    make_score_regularized_conditional_flow,
)
from .aggregation_error_score_regularized_flow import (
    fixed_observation_log_mass_score,
    standardization_scale,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "GAMMA_LOG_MASS_CONDITIONING_RULE",
    "ScoreRegularizedRootFlow",
]

_ARTIFACT_SCHEMA = "score-regularized-root-flow-v1"
_ARTIFACT_MAGIC = b"OpenGHG-score-regularized-root-flow-v1\0"
_ARCHITECTURE_NAME = "flowjax-score-regularized-projected-root-v1"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64

GAMMA_LOG_MASS_CONDITIONING_RULE = "gamma-log-mass-digamma-trigamma-rate-v1"

_RUNTIME = {
    "equinox": version("equinox"),
    "flowjax": version("flowjax"),
    "jax": version("jax"),
    "jaxlib": version("jaxlib"),
    "optax": version("optax"),
    "paramax": version("paramax"),
}


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON text."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    """Return a lower-case SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    """Return one strict lower-case SHA-256 digest."""
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def _array_sha256(values: ArrayLike) -> str:
    """Return a shape- and value-sensitive canonical float64 array digest."""
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": "<f8",
                "shape": list(array.shape),
            }
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_scalar(value: object, *, name: str) -> float:
    """Return a finite non-Boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one non-negative non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _source_seed(value: object) -> int:
    """Return one unsigned 32-bit JAX seed."""
    result = _nonnegative_integer(value, name="initialization_seed")
    if result >= 2**32:
        raise ValueError("initialization_seed must lie in [0, 2**32).")
    return result


def _readonly_float(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
) -> FloatArray:
    """Return an owned, finite, immutable float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.frombuffer(
        result.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(result.shape)
    return cast(FloatArray, immutable)


def _validate_flow(flow: Any, *, leading_rank: int) -> None:
    """Validate event/condition shapes and all fitted floating leaves."""
    if getattr(flow, "shape", None) != (leading_rank,):
        raise ValueError("flow event shape must match leading_rank.")
    if getattr(flow, "cond_shape", None) != (1,):
        raise ValueError("flow condition shape must be (1,).")
    found = False
    for leaf in jax.tree_util.tree_leaves(flow):
        if not eqx.is_inexact_array(leaf):
            continue
        found = True
        array = np.asarray(leaf)
        if array.dtype != np.dtype(np.float64):
            raise ValueError("flow parameters must use float64.")
        if not np.all(np.isfinite(array)):
            raise ValueError("flow parameters must be finite.")
    if not found:
        raise ValueError("flow must contain floating-point parameter leaves.")


def _spectrum_payload(spectrum: RootResidualSpectrum) -> dict[str, object]:
    """Return the complete authenticated spectrum metadata."""
    arrays = {
        "basis": spectrum.basis,
        "eigenvalues": spectrum.eigenvalues,
        "noise_sd": spectrum.noise_sd,
        "observation_mean_design": spectrum.observation_mean_design,
    }
    return {
        "arrays": {
            name: {
                "sha256": _array_sha256(values),
                "values": values.tolist(),
            }
            for name, values in arrays.items()
        },
        "cell_alphas_sha256": _validated_sha256(
            spectrum.cell_alphas_sha256,
            name="spectrum.cell_alphas_sha256",
        ),
        "design_sha256": _validated_sha256(
            spectrum.design_sha256,
            name="spectrum.design_sha256",
        ),
        "discarded_variance": spectrum.discarded_variance,
        "eigenvalue_tolerance": spectrum.eigenvalue_tolerance,
        "noise_sd_sha256": _validated_sha256(
            spectrum.noise_sd_sha256,
            name="spectrum.noise_sd_sha256",
        ),
        "requested_retained_variance_fraction": (spectrum.requested_retained_variance_fraction),
        "retained_variance_fraction": spectrum.retained_variance_fraction,
        "total_variance": spectrum.total_variance,
    }


def _spectrum_from_payload(payload: object) -> RootResidualSpectrum:
    """Validate and reconstruct a complete spectrum payload."""
    if not isinstance(payload, dict) or set(payload) != {
        "arrays",
        "cell_alphas_sha256",
        "design_sha256",
        "discarded_variance",
        "eigenvalue_tolerance",
        "noise_sd_sha256",
        "requested_retained_variance_fraction",
        "retained_variance_fraction",
        "total_variance",
    }:
        raise ValueError("serialized spectrum has an unexpected schema.")
    encoded_arrays = payload["arrays"]
    if not isinstance(encoded_arrays, dict) or set(encoded_arrays) != {
        "basis",
        "eigenvalues",
        "noise_sd",
        "observation_mean_design",
    }:
        raise ValueError("serialized spectrum arrays have an unexpected schema.")
    arrays: dict[str, FloatArray] = {}
    expected_ndim = {
        "basis": 2,
        "eigenvalues": 1,
        "noise_sd": 1,
        "observation_mean_design": 1,
    }
    for name, encoded in encoded_arrays.items():
        if not isinstance(encoded, dict) or set(encoded) != {"sha256", "values"}:
            raise ValueError(f"serialized spectrum {name} has an unexpected schema.")
        values = _readonly_float(
            encoded["values"],
            name=f"spectrum.{name}",
            ndim=expected_ndim[name],
        )
        expected_hash = _validated_sha256(
            encoded["sha256"],
            name=f"spectrum.{name}.sha256",
        )
        if _array_sha256(values) != expected_hash:
            raise ValueError(f"serialized spectrum {name} hash does not match.")
        arrays[name] = values
    spectrum = RootResidualSpectrum(
        arrays["observation_mean_design"],
        arrays["noise_sd"],
        arrays["basis"],
        arrays["eigenvalues"],
        total_variance=payload["total_variance"],
        discarded_variance=payload["discarded_variance"],
        requested_retained_variance_fraction=payload["requested_retained_variance_fraction"],
        eigenvalue_tolerance=payload["eigenvalue_tolerance"],
        cell_alphas_sha256=_validated_sha256(
            payload["cell_alphas_sha256"],
            name="spectrum.cell_alphas_sha256",
        ),
        design_sha256=_validated_sha256(
            payload["design_sha256"],
            name="spectrum.design_sha256",
        ),
        noise_sd_sha256=_validated_sha256(
            payload["noise_sd_sha256"],
            name="spectrum.noise_sd_sha256",
        ),
    )
    retained_fraction = _finite_scalar(
        payload["retained_variance_fraction"],
        name="spectrum.retained_variance_fraction",
    )
    tolerance = 32.0 * np.finfo(np.float64).eps
    if not math.isclose(
        spectrum.retained_variance_fraction,
        retained_fraction,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("serialized spectrum retained variance fraction does not match.")
    return spectrum


def _architecture_payload(leading_rank: int) -> dict[str, object]:
    """Return the frozen architecture identity for one leading rank."""
    return {
        "conditional_dimension": 1,
        "flow_layers": FLOW_LAYERS,
        "invert": FLOW_INVERT,
        "leading_rank": leading_rank,
        "name": _ARCHITECTURE_NAME,
        "nn_depth": FLOW_NN_DEPTH,
        "nn_width": FLOW_NN_WIDTH,
        "specialization": (
            "gaussian-rank-zero"
            if leading_rank == 0
            else "masked-autoregressive-rational-quadratic-spline"
            if leading_rank == 1
            else "rational-quadratic-spline-coupling"
        ),
        "spline_interval": list(FLOW_SPLINE_INTERVAL),
        "spline_knots": FLOW_SPLINE_KNOTS,
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ScoreRegularizedRootFlow:
    """One strict normalized projected-root likelihood artifact."""

    spectrum: RootResidualSpectrum = field(init=False)
    leading_rank: int = field(init=False)
    gamma_shape: float = field(init=False)
    gamma_rate: float = field(init=False)
    conditioning_rule_id: str = field(init=False)
    condition_center: float = field(init=False)
    condition_scale: float = field(init=False)
    flow: Any = field(init=False, repr=False)
    initialization_seed: int = field(init=False)
    source_provenance: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        spectrum: RootResidualSpectrum,
        leading_rank: int,
        gamma_shape: float,
        gamma_rate: float,
        flow: Any | None,
        *,
        conditioning_rule_id: str,
        initialization_seed: int,
        source_provenance: str,
    ) -> None:
        """Validate and own a fitted leading-flow plus complete spectrum."""
        if not isinstance(spectrum, RootResidualSpectrum):
            raise TypeError("spectrum must be a RootResidualSpectrum.")
        rank = _nonnegative_integer(leading_rank, name="leading_rank")
        if rank > spectrum.eigenvalues.size:
            raise ValueError("leading_rank must not exceed the retained spectrum rank.")
        shape = _finite_scalar(gamma_shape, name="gamma_shape")
        rate = _finite_scalar(gamma_rate, name="gamma_rate")
        if (
            not isinstance(conditioning_rule_id, str)
            or conditioning_rule_id != GAMMA_LOG_MASS_CONDITIONING_RULE
        ):
            raise ValueError("conditioning_rule_id does not match the frozen analytic Gamma rule.")
        center, scale = gamma_log_mass_conditioning(shape, rate)
        seed = _source_seed(initialization_seed)
        if not isinstance(source_provenance, str) or not source_provenance:
            raise ValueError("source_provenance must be a non-empty string.")
        if len(source_provenance) > 4_096:
            raise ValueError("source_provenance is too long.")
        _spectrum_payload(spectrum)
        if rank == 0:
            if flow is not None:
                raise ValueError("flow must be None when leading_rank is zero.")
        else:
            if flow is None:
                raise ValueError("flow is required when leading_rank is positive.")
            _validate_flow(flow, leading_rank=rank)

        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "leading_rank", rank)
        object.__setattr__(self, "gamma_shape", shape)
        object.__setattr__(self, "gamma_rate", rate)
        object.__setattr__(
            self,
            "conditioning_rule_id",
            conditioning_rule_id,
        )
        object.__setattr__(self, "condition_center", center)
        object.__setattr__(self, "condition_scale", scale)
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "initialization_seed", seed)
        object.__setattr__(self, "source_provenance", source_provenance)
        object.__setattr__(self, "artifact_sha256", _sha256_bytes(self.to_bytes()))

    @property
    def observation_count(self) -> int:
        """Return the number of native observations."""
        return int(self.spectrum.observation_mean_design.size)

    @property
    def retained_rank(self) -> int:
        """Return the complete authenticated analytic spectrum rank."""
        return int(self.spectrum.eigenvalues.size)

    @property
    def sha256(self) -> str:
        """Return the canonical serialized artifact digest."""
        return self.artifact_sha256

    def _total_mass(self, total_mass: object) -> float:
        """Return one finite non-negative root mass."""
        result = _finite_scalar(total_mass, name="total_mass")
        if result < 0.0:
            raise ValueError("total_mass must be non-negative.")
        return result

    def _condition(self, total_mass: float) -> jax.Array:
        """Return the standardized raw-log-mass condition for positive mass."""
        if total_mass <= 0.0:
            raise ValueError("a flow condition requires positive total_mass.")
        return jnp.asarray(
            [(math.log(total_mass) - self.condition_center) / self.condition_scale],
            dtype=jnp.float64,
        )

    def _observation(self, observation: ArrayLike) -> FloatArray:
        """Return one finite observation vector."""
        result = np.asarray(observation, dtype=np.float64)
        if result.shape != (self.observation_count,) or not np.all(np.isfinite(result)):
            raise ValueError("observation must be finite with one value per spectrum observation.")
        return cast(FloatArray, result)

    def _offset(self, offset: ArrayLike) -> FloatArray:
        """Return a finite scalar-expanded or observation-aligned offset."""
        raw = np.asarray(offset, dtype=np.float64)
        result = np.full(self.observation_count, float(raw), dtype=np.float64) if raw.ndim == 0 else raw
        if result.shape != (self.observation_count,) or not np.all(np.isfinite(result)):
            raise ValueError("offset must be finite and scalar or aligned with observation.")
        return cast(FloatArray, result)

    def _standardized_leading(
        self,
        standardized: ArrayLike,
    ) -> FloatArray:
        """Return one finite leading standardized coordinate vector."""
        result = np.asarray(standardized, dtype=np.float64)
        if result.shape != (self.leading_rank,) or not np.all(np.isfinite(result)):
            raise ValueError("standardized coordinate must be finite with shape (leading_rank,).")
        return cast(FloatArray, result)

    def _leading_residual(self, residual: ArrayLike) -> FloatArray:
        """Return one finite unstandardized leading projected residual."""
        result = np.asarray(residual, dtype=np.float64)
        if result.shape != (self.leading_rank,) or not np.all(np.isfinite(result)):
            raise ValueError("leading_residual must be finite with shape (leading_rank,).")
        return cast(FloatArray, result)

    def log_likelihood(
        self,
        observation: ArrayLike,
        total_mass: float,
        *,
        offset: ArrayLike = 0.0,
    ) -> float:
        """Evaluate the normalized density in native observation units."""
        observed = self._observation(observation)
        mass = self._total_mass(total_mass)
        fixed_offset = self._offset(offset)
        residual = (
            observed - fixed_offset - mass * self.spectrum.observation_mean_design
        ) / self.spectrum.noise_sd

        if mass == 0.0:
            result = -0.5 * (self.observation_count * _LOG_TWO_PI + float(residual @ residual)) - float(
                np.log(self.spectrum.noise_sd).sum()
            )
            return result

        coordinates = self.spectrum.basis.T @ residual
        orthogonal = residual - self.spectrum.basis @ coordinates
        scales = np.sqrt(1.0 + mass * mass * self.spectrum.eigenvalues)
        result = -float(np.log(self.spectrum.noise_sd).sum()) - 0.5 * (
            (self.observation_count - self.retained_rank) * _LOG_TWO_PI + float(orthogonal @ orthogonal)
        )
        if self.leading_rank:
            standardized = coordinates[: self.leading_rank] / scales[: self.leading_rank]
            result += float(
                self.flow.log_prob(
                    jnp.asarray(standardized, dtype=jnp.float64),
                    self._condition(mass),
                )
            )
            result -= float(np.log(scales[: self.leading_rank]).sum())
        if self.leading_rank < self.retained_rank:
            complement = coordinates[self.leading_rank :] / scales[self.leading_rank :]
            result -= 0.5 * (
                (self.retained_rank - self.leading_rank) * _LOG_TWO_PI + float(complement @ complement)
            )
            result -= float(np.log(scales[self.leading_rank :]).sum())
        if not math.isfinite(result):
            raise ValueError("score-regularized root-flow log density is non-finite.")
        return result

    def log_likelihood_batch(
        self,
        observation: ArrayLike,
        total_masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> FloatArray:
        """Deterministically evaluate one observation over a mass vector."""
        masses = np.asarray(total_masses, dtype=np.float64)
        if masses.ndim != 1 or not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("total_masses must be a finite non-negative vector.")
        result = np.asarray(
            [
                self.log_likelihood(
                    observation,
                    float(mass),
                    offset=offset,
                )
                for mass in masses
            ],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return cast(FloatArray, result)

    def log_likelihood_observation_batch(
        self,
        observations: ArrayLike,
        total_mass: float,
        *,
        offset: ArrayLike = 0.0,
    ) -> FloatArray:
        """Deterministically evaluate an observation matrix at one mass."""
        observed = np.asarray(observations, dtype=np.float64)
        if (
            observed.ndim != 2
            or observed.shape[1] != self.observation_count
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError("observations must be a finite matrix with one column per observation.")
        mass = self._total_mass(total_mass)
        fixed_offset = self._offset(offset)
        residual = (
            observed
            - fixed_offset[np.newaxis, :]
            - mass * self.spectrum.observation_mean_design[np.newaxis, :]
        ) / self.spectrum.noise_sd[np.newaxis, :]
        noise_jacobian = -float(np.log(self.spectrum.noise_sd).sum())
        if mass == 0.0:
            result = noise_jacobian - 0.5 * (
                self.observation_count * _LOG_TWO_PI + np.sum(residual * residual, axis=1)
            )
        else:
            coordinates = residual @ self.spectrum.basis
            orthogonal = residual - coordinates @ self.spectrum.basis.T
            scales = np.sqrt(1.0 + mass * mass * self.spectrum.eigenvalues)
            result = noise_jacobian - 0.5 * (
                (self.observation_count - self.retained_rank) * _LOG_TWO_PI
                + np.sum(orthogonal * orthogonal, axis=1)
            )
            if self.leading_rank:
                standardized = coordinates[:, : self.leading_rank] / scales[np.newaxis, : self.leading_rank]
                conditions = jnp.broadcast_to(
                    self._condition(mass),
                    (observed.shape[0], 1),
                )
                result += np.asarray(
                    self.flow.log_prob(
                        jnp.asarray(standardized, dtype=jnp.float64),
                        conditions,
                    ),
                    dtype=np.float64,
                )
                result -= float(np.log(scales[: self.leading_rank]).sum())
            if self.leading_rank < self.retained_rank:
                complement = coordinates[:, self.leading_rank :] / scales[np.newaxis, self.leading_rank :]
                result -= 0.5 * (
                    (self.retained_rank - self.leading_rank) * _LOG_TWO_PI
                    + np.sum(complement * complement, axis=1)
                )
                result -= float(np.log(scales[self.leading_rank :]).sum())
        if result.shape != (observed.shape[0],) or not np.all(np.isfinite(result)):
            raise ValueError("score-regularized root-flow log densities are non-finite.")
        result.setflags(write=False)
        return cast(FloatArray, result)

    def sample_observation(
        self,
        total_mass: float,
        *,
        sample_count: int,
        source_seed: int,
        offset: ArrayLike = 0.0,
    ) -> FloatArray:
        """Draw exact samples from the same normalized hybrid likelihood."""
        mass = self._total_mass(total_mass)
        count = _positive_integer(sample_count, name="sample_count")
        seed = _nonnegative_integer(source_seed, name="source_seed")
        if seed >= 2**32:
            raise ValueError("source_seed must lie in [0, 2**32).")
        fixed_offset = self._offset(offset)
        mean = fixed_offset + mass * self.spectrum.observation_mean_design

        if mass == 0.0:
            residual = np.asarray(
                jr.normal(
                    jr.key(seed),
                    (count, self.observation_count),
                    dtype=jnp.float64,
                ),
                dtype=np.float64,
            )
        else:
            flow_key, gaussian_key = jr.split(jr.key(seed))
            gaussian = np.asarray(
                jr.normal(
                    gaussian_key,
                    (count, self.observation_count),
                    dtype=jnp.float64,
                ),
                dtype=np.float64,
            )
            basis = self.spectrum.basis
            gaussian_coordinates = gaussian @ basis
            orthogonal = gaussian - gaussian_coordinates @ basis.T
            scales = np.sqrt(1.0 + mass * mass * self.spectrum.eigenvalues)
            coordinates = gaussian_coordinates * scales
            if self.leading_rank:
                standardized = np.asarray(
                    self.flow.sample(
                        flow_key,
                        (count,),
                        condition=self._condition(mass),
                    ),
                    dtype=np.float64,
                )
                if standardized.shape != (count, self.leading_rank):
                    raise RuntimeError("flow returned an unexpected sample shape.")
                coordinates[:, : self.leading_rank] = standardized * scales[: self.leading_rank]
            residual = orthogonal + coordinates @ basis.T
        result = mean + residual * self.spectrum.noise_sd
        if not np.all(np.isfinite(result)):
            raise ValueError("score-regularized root-flow samples are non-finite.")
        return cast(FloatArray, result)

    def leading_standardized_partial_log_mass_score(
        self,
        standardized: ArrayLike,
        total_mass: float,
    ) -> float:
        """Return ``partial_tau log q(x|tau)`` at fixed standardized ``x``."""
        point = self._standardized_leading(standardized)
        mass = self._total_mass(total_mass)
        if mass == 0.0 or self.leading_rank == 0:
            return 0.0
        raw_tau = jnp.asarray(math.log(mass), dtype=jnp.float64)
        projected = jnp.asarray(point, dtype=jnp.float64)

        def log_prob(raw_log_mass: jax.Array) -> jax.Array:
            condition = jnp.asarray(
                [(raw_log_mass - self.condition_center) / self.condition_scale],
                dtype=jnp.float64,
            )
            return self.flow.log_prob(projected, condition)

        result = float(jax.grad(log_prob)(raw_tau))
        if not math.isfinite(result):
            raise ValueError("flow partial log-mass score is non-finite.")
        return result

    def leading_standardized_observation_score(
        self,
        standardized: ArrayLike,
        total_mass: float,
    ) -> FloatArray:
        """Return ``grad_x log q(x|tau)`` in leading standardized coordinates."""
        point = self._standardized_leading(standardized)
        mass = self._total_mass(total_mass)
        if self.leading_rank == 0:
            return point
        if mass == 0.0:
            result = -point
        else:
            condition = self._condition(mass)

            def log_prob(coordinates: jax.Array) -> jax.Array:
                return self.flow.log_prob(coordinates, condition)

            result = np.asarray(
                jax.grad(log_prob)(jnp.asarray(point, dtype=jnp.float64)),
                dtype=np.float64,
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("flow observation score is non-finite.")
        result.setflags(write=False)
        return cast(FloatArray, result)

    def leading_fixed_residual_log_mass_score(
        self,
        leading_residual: ArrayLike,
        total_mass: float,
    ) -> float:
        """Return the leading-density ``tau`` score at fixed projected residual.

        This chain-rule score includes the leading scale Jacobian.  It does not
        include the conditional-mean derivative needed when holding a raw
        observation, rather than its residual, fixed.
        """
        residual = self._leading_residual(leading_residual)
        mass = self._total_mass(total_mass)
        if mass == 0.0 or self.leading_rank == 0:
            return 0.0
        leading_eigenvalues = self.spectrum.eigenvalues[: self.leading_rank]
        scales = np.asarray(
            standardization_scale(mass, leading_eigenvalues),
            dtype=np.float64,
        )
        standardized = residual / scales
        partial_score = self.leading_standardized_partial_log_mass_score(
            standardized,
            mass,
        )
        observation_score = self.leading_standardized_observation_score(
            standardized,
            mass,
        )
        result = float(
            fixed_observation_log_mass_score(
                mass,
                leading_eigenvalues,
                standardized,
                partial_score,
                observation_score,
            )
        )
        if not math.isfinite(result):
            raise ValueError("fixed-residual log-mass score is non-finite.")
        return result

    @property
    def metadata_payload(self) -> dict[str, object]:
        """Return strict non-parameter metadata for canonical serialization."""
        spectrum = _spectrum_payload(self.spectrum)
        return {
            "architecture": _architecture_payload(self.leading_rank),
            "condition_center": self.condition_center,
            "condition_scale": self.condition_scale,
            "conditioning_rule_id": self.conditioning_rule_id,
            "gamma_rate": self.gamma_rate,
            "gamma_shape": self.gamma_shape,
            "initialization_seed": self.initialization_seed,
            "leading_rank": self.leading_rank,
            "runtime": _RUNTIME,
            "schema": _ARTIFACT_SCHEMA,
            "source_provenance": self.source_provenance,
            "spectrum": spectrum,
            "spectrum_sha256": _sha256_bytes(_canonical_json(spectrum).encode("utf-8")),
        }

    def to_bytes(self) -> bytes:
        """Serialize strict metadata followed by canonical fitted-flow leaves."""
        metadata = _canonical_json(self.metadata_payload).encode("utf-8")
        buffer = io.BytesIO()
        buffer.write(_ARTIFACT_MAGIC)
        buffer.write(struct.pack("<Q", len(metadata)))
        buffer.write(metadata)
        if self.leading_rank:
            eqx.tree_serialise_leaves(buffer, self.flow)
        return buffer.getvalue()

    @classmethod
    def from_bytes(
        cls,
        serialized: bytes,
        *,
        expected_sha256: str,
    ) -> ScoreRegularizedRootFlow:
        """Authenticate, reconstruct, and exactly reserialize a v1 artifact."""
        if not isinstance(serialized, bytes):
            raise TypeError("serialized root-flow artifact must be bytes.")
        expected = _validated_sha256(
            expected_sha256,
            name="expected_sha256",
        )
        if _sha256_bytes(serialized) != expected:
            raise ValueError("root-flow artifact SHA-256 fingerprint does not match.")
        buffer = io.BytesIO(serialized)
        if buffer.read(len(_ARTIFACT_MAGIC)) != _ARTIFACT_MAGIC:
            raise ValueError("serialized root-flow artifact has an unexpected magic header.")
        encoded_length = buffer.read(8)
        if len(encoded_length) != 8:
            raise ValueError("serialized root-flow artifact metadata length is truncated.")
        metadata_length = struct.unpack("<Q", encoded_length)[0]
        metadata_bytes = buffer.read(metadata_length)
        if len(metadata_bytes) != metadata_length:
            raise ValueError("serialized root-flow artifact metadata is truncated.")
        try:
            payload = json.loads(metadata_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("serialized root-flow artifact metadata is not valid JSON.") from error
        if _canonical_json(payload).encode("utf-8") != metadata_bytes:
            raise ValueError("serialized root-flow artifact metadata must use canonical JSON.")
        if not isinstance(payload, dict) or set(payload) != {
            "architecture",
            "condition_center",
            "condition_scale",
            "conditioning_rule_id",
            "gamma_rate",
            "gamma_shape",
            "initialization_seed",
            "leading_rank",
            "runtime",
            "schema",
            "source_provenance",
            "spectrum",
            "spectrum_sha256",
        }:
            raise ValueError("serialized root-flow artifact metadata has an unexpected schema.")
        if payload["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError("serialized root-flow artifact has an unexpected schema.")
        rank = _nonnegative_integer(payload["leading_rank"], name="leading_rank")
        if payload["architecture"] != _architecture_payload(rank):
            raise ValueError("serialized root-flow artifact architecture does not match.")
        if payload["runtime"] != _RUNTIME:
            raise ValueError("serialized root-flow artifact runtime does not match.")
        if payload["conditioning_rule_id"] != GAMMA_LOG_MASS_CONDITIONING_RULE:
            raise ValueError("serialized root-flow conditioning rule does not match.")
        analytic_center, analytic_scale = gamma_log_mass_conditioning(
            payload["gamma_shape"],
            payload["gamma_rate"],
        )
        if payload["condition_center"] != analytic_center or payload["condition_scale"] != analytic_scale:
            raise ValueError("serialized root-flow conditioning does not replay analytically.")
        spectrum_bytes = _canonical_json(payload["spectrum"]).encode("utf-8")
        spectrum_sha = _validated_sha256(
            payload["spectrum_sha256"],
            name="spectrum_sha256",
        )
        if _sha256_bytes(spectrum_bytes) != spectrum_sha:
            raise ValueError("serialized spectrum SHA-256 fingerprint does not match.")
        spectrum = _spectrum_from_payload(payload["spectrum"])
        if rank:
            template = make_score_regularized_conditional_flow(
                rank,
                source_seed=payload["initialization_seed"],
            )
            try:
                fitted_flow = eqx.tree_deserialise_leaves(buffer, template)
            except Exception as error:
                raise ValueError("serialized root-flow parameter leaves are invalid.") from error
        else:
            fitted_flow = None
        if buffer.read(1):
            raise ValueError("serialized root-flow artifact has trailing bytes.")
        result = cls(
            spectrum,
            rank,
            payload["gamma_shape"],
            payload["gamma_rate"],
            fitted_flow,
            conditioning_rule_id=payload["conditioning_rule_id"],
            initialization_seed=payload["initialization_seed"],
            source_provenance=payload["source_provenance"],
        )
        if result.to_bytes() != serialized:
            raise ValueError("serialized root-flow artifact does not replay canonically.")
        if result.artifact_sha256 != expected:  # pragma: no cover
            raise ValueError("root-flow artifact SHA-256 fingerprint does not match.")
        return result
