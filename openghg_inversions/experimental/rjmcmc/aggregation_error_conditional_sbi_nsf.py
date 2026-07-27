"""Normalized sbi neural-spline likelihood for aggregation error.

The optional artifact in this module models the complete projected noisy
aggregation residual with an ``sbi`` conditional neural spline flow.  Exact
conditional covariance whitening and the Gaussian residual-image complement
remain outside the learner.  The same authenticated model supplies normalized
log densities, retained-mass gradients, and forward observations.

Importing this module requires the optional ``nle`` dependency group.
"""

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

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sbi.neural_nets import likelihood_nn
import torch

from .aggregation_error_conditional_flow import (
    conditional_residual_unit_covariances,
)
from .aggregation_error_conditional_mdn import ResidualImageContext

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ConditionalResidualImageSbiNsf",
    "NSF_HIDDEN_FEATURES",
    "NSF_NUM_BINS",
    "NSF_NUM_TRANSFORMS",
    "conditional_residual_unit_covariances",
    "make_conditional_residual_nsf",
]

_ARTIFACT_MAGIC = b"OpenGHG-conditional-residual-sbi-nsf-v1\0"
_ARTIFACT_SCHEMA = "aggregation-conditional-residual-image-sbi-nsf-v1"
_ARCHITECTURE = "sbi-autoregressive-neural-spline-flow-v1"
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SHA256_HEX_LENGTH = 64
_SUPPORTED_STATE_DTYPES = {"<f8", "<i8"}

NSF_HIDDEN_FEATURES = 128
NSF_NUM_TRANSFORMS = 8
NSF_NUM_BINS = 16
SBI_VERSION = version("sbi")
TORCH_VERSION = version("torch")
NFLOWS_VERSION = version("nflows")


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
    """Return an unsigned 32-bit Torch seed."""
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
    immutable = np.frombuffer(
        result.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(result.shape)
    return cast(FloatArray, immutable)


def _architecture_payload() -> dict[str, object]:
    """Return the source-pinned estimator declaration."""
    return {
        "name": _ARCHITECTURE,
        "model": "nsf",
        "hidden_features": NSF_HIDDEN_FEATURES,
        "num_transforms": NSF_NUM_TRANSFORMS,
        "num_bins": NSF_NUM_BINS,
        "z_score_theta": "none",
        "z_score_x": "none",
    }


def _runtime_payload() -> dict[str, str]:
    """Return the exact runtime versions bound to an artifact."""
    return {
        "nflows": NFLOWS_VERSION,
        "sbi": SBI_VERSION,
        "torch": TORCH_VERSION,
    }


def _validate_model(model: Any) -> None:
    """Require a nonempty finite float64 state with only known dtypes."""
    state = model.state_dict()
    if not state:
        raise ValueError("NSF state dictionary must not be empty.")
    found_float = False
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("NSF state names must be nonempty strings.")
        if tensor.device.type != "cpu":
            raise ValueError("NSF state tensors must be on CPU.")
        if tensor.dtype == torch.float64:
            found_float = True
            if not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError("NSF floating-point state must be finite.")
        elif tensor.dtype != torch.int64:
            raise ValueError("NSF state tensors must use float64 or int64.")
    if not found_float:
        raise ValueError("NSF state must contain floating-point parameters.")


def make_conditional_residual_nsf(
    residual_rank: int,
    conditioner_dimension: int,
    *,
    source_seed: int,
) -> Any:
    """Build the source-pinned conditional autoregressive NSF."""
    rank = _positive_integer(residual_rank, name="residual_rank")
    conditioner = _positive_integer(
        conditioner_dimension,
        name="conditioner_dimension",
    )
    seed = _source_seed(source_seed)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            builder = likelihood_nn(
                model="nsf",
                hidden_features=NSF_HIDDEN_FEATURES,
                num_transforms=NSF_NUM_TRANSFORMS,
                num_bins=NSF_NUM_BINS,
                z_score_theta="none",
                z_score_x="none",
            )
            model = builder(
                torch.zeros((2, conditioner), dtype=torch.float64),
                torch.zeros((2, rank), dtype=torch.float64),
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    model = model.cpu().to(dtype=torch.float64)
    model.eval()
    _validate_model(model)
    if tuple(model.input_shape) != (rank,):
        raise ValueError("NSF input shape does not match residual rank.")
    if tuple(model.condition_shape) != (conditioner,):
        raise ValueError("NSF condition shape does not match retained-region count.")
    return model


def _state_array(tensor: torch.Tensor) -> NDArray[Any]:
    """Return one canonical little-endian state array."""
    raw = tensor.detach().cpu().contiguous().numpy()
    if tensor.dtype == torch.float64:
        return np.ascontiguousarray(raw, dtype="<f8")
    if tensor.dtype == torch.int64:
        return np.ascontiguousarray(raw, dtype="<i8")
    raise ValueError("NSF state tensors must use float64 or int64.")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConditionalResidualImageSbiNsf:
    """Authenticated normalized NSF likelihood and marginal simulator."""

    context: ResidualImageContext = field(init=False)
    unit_residual_covariances: FloatArray = field(init=False)
    conditioner_center: FloatArray = field(init=False)
    conditioner_scale: FloatArray = field(init=False)
    model: Any = field(init=False, repr=False)
    initialization_seed: int = field(init=False)
    source_provenance: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __init__(
        self,
        context: ResidualImageContext,
        unit_residual_covariances: ArrayLike,
        conditioner_center: ArrayLike,
        conditioner_scale: ArrayLike,
        model: Any,
        *,
        initialization_seed: int,
        source_provenance: str,
    ) -> None:
        """Validate and own a complete fitted NSF artifact."""
        if not isinstance(context, ResidualImageContext):
            raise TypeError("context must be a ResidualImageContext.")
        covariances = _readonly_float(
            unit_residual_covariances,
            name="unit_residual_covariances",
            ndim=3,
        )
        expected_shape = (
            context.region_count,
            context.residual_rank,
            context.residual_rank,
        )
        if covariances.shape != expected_shape:
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
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a Torch module.")
        model = model.cpu().to(dtype=torch.float64)
        model.eval()
        _validate_model(model)
        if tuple(getattr(model, "input_shape", ())) != (context.residual_rank,):
            raise ValueError("NSF input shape does not match residual rank.")
        if tuple(getattr(model, "condition_shape", ())) != (context.region_count,):
            raise ValueError("NSF condition shape does not match retained-region count.")
        if not isinstance(source_provenance, str) or not source_provenance:
            raise ValueError("source_provenance must be a nonempty string.")
        if len(source_provenance) > 4_096:
            raise ValueError("source_provenance is too long.")

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "unit_residual_covariances", covariances)
        object.__setattr__(self, "conditioner_center", center)
        object.__setattr__(self, "conditioner_scale", scale)
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "initialization_seed",
            _source_seed(initialization_seed),
        )
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
            raise ValueError(
                "masses must contain one finite strictly positive value per region."
            )
        if not np.isfinite(float(np.sum(result))):
            raise ValueError("mass total must be finite.")
        return cast(FloatArray, result)

    def conditioners(self, masses: ArrayLike) -> FloatArray:
        """Return standardized conditioners for canonical mass rows."""
        retained = np.asarray(masses, dtype=np.float64)
        if (
            retained.ndim != 2
            or retained.shape[1:] != (self.region_count,)
            or not np.all(np.isfinite(retained))
            or np.any(retained <= 0.0)
        ):
            raise ValueError(
                "masses must be a matrix with one finite positive column per region."
            )
        totals = np.sum(retained, axis=1)
        if not np.all(np.isfinite(totals)):
            raise ValueError("mass totals must be finite.")
        raw = np.empty_like(retained)
        raw[:, 0] = np.log(totals)
        if self.region_count > 1:
            raw[:, 1:] = np.log(retained[:, :-1]) - np.log(retained[:, -1:])
        result = (
            raw - self.conditioner_center[np.newaxis, :]
        ) / self.conditioner_scale[np.newaxis, :]
        if not np.all(np.isfinite(result)):
            raise ValueError("standardized NSF conditioners are non-finite.")
        return cast(FloatArray, result)

    def conditioner(self, masses: ArrayLike) -> FloatArray:
        """Return one standardized NSF conditioner."""
        retained = self._validated_masses(masses)
        return cast(FloatArray, self.conditioners(retained[np.newaxis, :])[0])

    def projected_choleskies(self, masses: ArrayLike) -> FloatArray:
        """Return exact projected noisy-residual Cholesky factors."""
        retained = np.asarray(masses, dtype=np.float64)
        self.conditioners(retained)
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
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "projected residual covariance is not positive definite."
            ) from error
        return cast(FloatArray, result)

    def projected_cholesky(self, masses: ArrayLike) -> FloatArray:
        """Return one exact projected noisy-residual Cholesky factor."""
        retained = self._validated_masses(masses)
        return cast(
            FloatArray,
            self.projected_choleskies(retained[np.newaxis, :])[0],
        )

    @staticmethod
    def _offset(offset: ArrayLike, observation_count: int) -> FloatArray:
        """Return a finite observation-aligned offset."""
        raw = np.asarray(offset, dtype=np.float64)
        if raw.ndim == 0:
            result = np.full(observation_count, float(raw), dtype=np.float64)
        else:
            result = raw
        if result.shape != (observation_count,) or not np.all(np.isfinite(result)):
            raise ValueError(
                "offset must be finite and scalar or aligned with observation."
            )
        return cast(FloatArray, result)

    def _torch_log_likelihood(
        self,
        observation: torch.Tensor,
        masses: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one differentiable observation log density."""
        noise_sd = torch.tensor(
            np.array(self.context.noise_sd, copy=True),
            dtype=torch.float64,
        )
        mean_design = torch.tensor(
            np.array(self.context.observation_mean_design, copy=True),
            dtype=torch.float64,
        )
        basis = torch.tensor(
            np.array(self.context.residual_basis, copy=True),
            dtype=torch.float64,
        )
        covariances = torch.tensor(
            np.array(self.unit_residual_covariances, copy=True),
            dtype=torch.float64,
        )
        residual = (observation - offset - mean_design @ masses) / noise_sd
        coordinates = basis.T @ residual
        orthogonal = residual - basis @ coordinates
        covariance = torch.eye(
            self.residual_rank,
            dtype=torch.float64,
        ) + torch.einsum("j,jab->ab", masses * masses, covariances)
        cholesky = torch.linalg.cholesky(covariance)
        standardized = torch.linalg.solve_triangular(
            cholesky,
            coordinates[:, None],
            upper=False,
        )[:, 0]
        total = torch.sum(masses)
        raw = torch.empty(
            self.region_count,
            dtype=torch.float64,
        )
        raw[0] = torch.log(total)
        if self.region_count > 1:
            raw[1:] = torch.log(masses[:-1]) - torch.log(masses[-1])
        conditioner = (
            raw
            - torch.tensor(
                np.array(self.conditioner_center, copy=True),
                dtype=torch.float64,
            )
        ) / torch.tensor(
            np.array(self.conditioner_scale, copy=True),
            dtype=torch.float64,
        )
        flow_log_density = self.model.log_prob(
            standardized[None, None, :],
            condition=conditioner[None, :],
        )[0, 0]
        return (
            -torch.sum(torch.log(noise_sd))
            - 0.5
            * (
                (self.context.observation_count - self.residual_rank) * _LOG_TWO_PI
                + torch.dot(orthogonal, orthogonal)
            )
            + flow_log_density
            - torch.sum(torch.log(torch.diagonal(cholesky)))
        )

    def log_likelihood_and_mass_gradient(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> tuple[float, FloatArray]:
        """Return the normalized log density and analytic mass gradient."""
        observed = np.asarray(observation, dtype=np.float64)
        if (
            observed.shape != (self.context.observation_count,)
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError(
                "observation must be finite with one value per context observation."
            )
        retained = self._validated_masses(masses)
        fixed_offset = self._offset(offset, self.context.observation_count)
        observed_tensor = torch.as_tensor(observed, dtype=torch.float64)
        mass_tensor = torch.tensor(
            retained,
            dtype=torch.float64,
            requires_grad=True,
        )
        offset_tensor = torch.as_tensor(fixed_offset, dtype=torch.float64)
        value = self._torch_log_likelihood(
            observed_tensor,
            mass_tensor,
            offset_tensor,
        )
        gradient = torch.autograd.grad(value, mass_tensor)[0]
        result = float(value.detach())
        gradient_array = np.asarray(
            gradient.detach().cpu(),
            dtype=np.float64,
        )
        if not np.isfinite(result) or not np.all(np.isfinite(gradient_array)):
            raise ValueError("NSF likelihood value or mass gradient is non-finite.")
        return result, cast(FloatArray, gradient_array)

    def log_likelihood(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
    ) -> float:
        """Evaluate the normalized fitted observation log density."""
        return self.log_likelihood_and_mass_gradient(
            observation,
            masses,
            offset=offset,
        )[0]

    def log_likelihood_batch(
        self,
        observation: ArrayLike,
        masses: ArrayLike,
        *,
        offset: ArrayLike = 0.0,
        batch_size: int = 8_192,
    ) -> FloatArray:
        """Evaluate one observation at a matrix of retained-mass states."""
        observed = np.asarray(observation, dtype=np.float64)
        if (
            observed.shape != (self.context.observation_count,)
            or not np.all(np.isfinite(observed))
        ):
            raise ValueError(
                "observation must be finite with one value per context observation."
            )
        retained = np.asarray(masses, dtype=np.float64)
        conditioners = self.conditioners(retained)
        fixed_offset = self._offset(offset, self.context.observation_count)
        chunk_size = _positive_integer(batch_size, name="batch_size")
        residual = (
            observed[np.newaxis, :]
            - fixed_offset[np.newaxis, :]
            - retained @ self.context.observation_mean_design.T
        ) / self.context.noise_sd[np.newaxis, :]
        coordinates = residual @ self.context.residual_basis
        orthogonal = residual - coordinates @ self.context.residual_basis.T
        cholesky = self.projected_choleskies(retained)
        standardized = np.linalg.solve(
            cholesky,
            coordinates[:, :, np.newaxis],
        )[:, :, 0]
        values: list[NDArray[np.float64]] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, retained.shape[0], chunk_size):
                stop = min(start + chunk_size, retained.shape[0])
                target = torch.as_tensor(
                    standardized[start:stop],
                    dtype=torch.float64,
                )
                condition = torch.as_tensor(
                    conditioners[start:stop],
                    dtype=torch.float64,
                )
                log_prob = self.model.log_prob(
                    target[None, :, :],
                    condition=condition,
                )[0]
                values.append(
                    np.asarray(log_prob.cpu(), dtype=np.float64)
                )
        flow_log_density = np.concatenate(values)
        result = (
            -float(np.sum(np.log(self.context.noise_sd)))
            - 0.5
            * (
                (self.context.observation_count - self.residual_rank) * _LOG_TWO_PI
                + np.sum(orthogonal * orthogonal, axis=1)
            )
            + flow_log_density
            - np.sum(
                np.log(np.diagonal(cholesky, axis1=1, axis2=2)),
                axis=1,
            )
        )
        if result.shape != (retained.shape[0],) or not np.all(np.isfinite(result)):
            raise ValueError("NSF log densities are non-finite.")
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
        condition = torch.as_tensor(
            self.conditioner(retained)[None, :],
            dtype=torch.float64,
        )
        self.model.eval()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                standardized_tensor = self.model.sample(
                    torch.Size([count]),
                    condition=condition,
                )[:, 0, :]
                gaussian_tensor = torch.randn(
                    (count, self.context.observation_count),
                    dtype=torch.float64,
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        standardized = np.asarray(
            standardized_tensor.cpu(),
            dtype=np.float64,
        )
        gaussian = np.asarray(gaussian_tensor.cpu(), dtype=np.float64)
        if standardized.shape != (count, self.residual_rank):
            raise RuntimeError("NSF returned an unexpected sample shape.")
        cholesky = self.projected_cholesky(retained)
        coordinates = standardized @ cholesky.T
        basis = self.context.residual_basis
        orthogonal = gaussian - (gaussian @ basis) @ basis.T
        residual = coordinates @ basis.T + orthogonal
        mean = fixed_offset + self.context.observation_mean_design @ retained
        result = mean + residual * self.context.noise_sd
        if not np.all(np.isfinite(result)):
            raise ValueError("NSF observation samples are non-finite.")
        return cast(FloatArray, result)

    @property
    def metadata_payload(self) -> dict[str, object]:
        """Return strict JSON-compatible metadata including state manifest."""
        tensors: list[dict[str, object]] = []
        for name in sorted(self.model.state_dict()):
            array = _state_array(self.model.state_dict()[name])
            payload = array.tobytes(order="C")
            tensors.append(
                {
                    "dtype": array.dtype.str,
                    "name": name,
                    "nbytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "shape": list(array.shape),
                }
            )
        return {
            "architecture": _architecture_payload(),
            "conditioner_center": self.conditioner_center.tolist(),
            "conditioner_scale": self.conditioner_scale.tolist(),
            "context": self.context.payload,
            "context_sha256": self.context.artifact_sha256,
            "initialization_seed": self.initialization_seed,
            "runtime": _runtime_payload(),
            "schema": _ARTIFACT_SCHEMA,
            "source_provenance": self.source_provenance,
            "state_tensors": tensors,
            "unit_residual_covariances": self.unit_residual_covariances.tolist(),
        }

    def to_bytes(self) -> bytes:
        """Serialize metadata and state tensors to canonical non-pickle bytes."""
        metadata = _canonical_json(self.metadata_payload).encode("utf-8")
        buffer = io.BytesIO()
        buffer.write(_ARTIFACT_MAGIC)
        buffer.write(struct.pack("<Q", len(metadata)))
        buffer.write(metadata)
        state = self.model.state_dict()
        for name in sorted(state):
            buffer.write(_state_array(state[name]).tobytes(order="C"))
        return buffer.getvalue()

    @classmethod
    def from_bytes(
        cls,
        serialized: bytes,
        *,
        expected_sha256: str,
    ) -> ConditionalResidualImageSbiNsf:
        """Authenticate and reconstruct a fitted NSF artifact."""
        if not isinstance(serialized, bytes):
            raise TypeError("serialized NSF artifact must be bytes.")
        expected = _validated_sha256(expected_sha256, name="expected_sha256")
        if _sha256_bytes(serialized) != expected:
            raise ValueError("NSF artifact SHA-256 fingerprint does not match.")
        buffer = io.BytesIO(serialized)
        if buffer.read(len(_ARTIFACT_MAGIC)) != _ARTIFACT_MAGIC:
            raise ValueError("serialized NSF artifact has an unexpected magic header.")
        encoded_length = buffer.read(8)
        if len(encoded_length) != 8:
            raise ValueError("serialized NSF metadata length is truncated.")
        metadata_length = struct.unpack("<Q", encoded_length)[0]
        metadata_bytes = buffer.read(metadata_length)
        if len(metadata_bytes) != metadata_length:
            raise ValueError("serialized NSF metadata is truncated.")
        try:
            payload = json.loads(metadata_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("serialized NSF metadata is not valid JSON.") from error
        if _canonical_json(payload).encode("utf-8") != metadata_bytes:
            raise ValueError("serialized NSF metadata must use canonical JSON.")
        expected_keys = {
            "architecture",
            "conditioner_center",
            "conditioner_scale",
            "context",
            "context_sha256",
            "initialization_seed",
            "runtime",
            "schema",
            "source_provenance",
            "state_tensors",
            "unit_residual_covariances",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("serialized NSF metadata has an unexpected schema.")
        if payload["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError("serialized NSF artifact has an unexpected schema.")
        if payload["architecture"] != _architecture_payload():
            raise ValueError("serialized NSF architecture does not match.")
        if payload["runtime"] != _runtime_payload():
            raise ValueError("serialized NSF runtime does not match.")
        context = ResidualImageContext.from_json(
            _canonical_json(payload["context"]),
            expected_sha256=payload["context_sha256"],
        )
        model = make_conditional_residual_nsf(
            context.residual_rank,
            context.region_count,
            source_seed=payload["initialization_seed"],
        )
        manifest = payload["state_tensors"]
        if not isinstance(manifest, list):
            raise ValueError("serialized NSF state manifest must be a list.")
        expected_names = sorted(model.state_dict())
        names = [
            item.get("name") if isinstance(item, dict) else None
            for item in manifest
        ]
        if names != expected_names:
            raise ValueError("serialized NSF state keys do not match the architecture.")
        rebuilt: dict[str, torch.Tensor] = {}
        for item in manifest:
            if not isinstance(item, dict) or set(item) != {
                "dtype",
                "name",
                "nbytes",
                "sha256",
                "shape",
            }:
                raise ValueError("serialized NSF tensor manifest entry is malformed.")
            dtype = item["dtype"]
            if dtype not in _SUPPORTED_STATE_DTYPES:
                raise ValueError("serialized NSF tensor dtype is unsupported.")
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or any(
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    for size in shape
                )
            ):
                raise ValueError("serialized NSF tensor shape is malformed.")
            expected_nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            if item["nbytes"] != expected_nbytes:
                raise ValueError("serialized NSF tensor byte count is inconsistent.")
            raw = buffer.read(expected_nbytes)
            if len(raw) != expected_nbytes:
                raise ValueError("serialized NSF tensor bytes are truncated.")
            if _sha256_bytes(raw) != item["sha256"]:
                raise ValueError("serialized NSF tensor digest does not match.")
            array = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
            rebuilt[item["name"]] = torch.from_numpy(array)
        if buffer.read(1):
            raise ValueError("serialized NSF artifact has trailing bytes.")
        try:
            model.load_state_dict(rebuilt, strict=True)
        except (RuntimeError, ValueError) as error:
            raise ValueError("serialized NSF state is incompatible.") from error
        result = cls(
            context,
            payload["unit_residual_covariances"],
            payload["conditioner_center"],
            payload["conditioner_scale"],
            model,
            initialization_seed=payload["initialization_seed"],
            source_provenance=payload["source_provenance"],
        )
        if result.to_bytes() != serialized:
            raise ValueError("serialized NSF artifact does not replay canonically.")
        if result.artifact_sha256 != expected:
            raise ValueError("NSF artifact SHA-256 fingerprint does not match.")
        return result
