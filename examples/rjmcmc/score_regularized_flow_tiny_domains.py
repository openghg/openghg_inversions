"""Observation-blind simulator domains for the score-regularized tiny screen.

This module builds only the six frozen root simulators used by N1.  It does
not fit a density, inspect a realized observation, evaluate an oracle, or
publish a scientific decision.

For one root case the simulator is

```
T ~ Gamma(sum(alpha), rate)
xi ~ projected unit-mass Dirichlet allocation residual
epsilon ~ Normal(0, I)
x = (T * xi + epsilon) / sqrt(1 + T**2 * lambda).
```

The allocation, mass, and measurement-noise catalogues use genuinely
independent, domain-separated PCG64 streams.  All catalogues are exact
prefixes when a larger power-of-two sample count is reconstructed with the
same case, public domain, and base seed.  This corrected v2 protocol must not
be confused with the historical v1 row-aligned Sobol construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy import special

from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (
    gamma_log_mass_conditioning,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_regularized_flow import (
    component_observation_score,
    component_partial_log_mass_score,
    standardize_simulator_draw,
)
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Family = Literal["two_cell", "four_cell"]
PublicDomain = Literal[
    "training",
    "model-selection-validation",
    "development-reporting-test",
]

PROTOCOL = "score-regularized-projected-root-nle-pcg64-iid-v2"
SCHEMA = "score-regularized-flow-tiny-domain-v2"
EVIDENCE_SCHEMA = "score-regularized-flow-tiny-domain-evidence-v2"
CONSTRUCTION_METHOD = "keyed_pcg64_dirichlet"
UNIFORM_GENERATOR = "numpy.random.Generator(numpy.random.PCG64(seed))"
UNIFORM_DRAW_CONTRACT = "row-major Generator.random((sample_count, dimension))"
ROOT_TOTAL_TRANSFORM = "scipy.special.gammaincinv(shape, uniform) / rate"
STANDARD_NORMAL_TRANSFORM = "scipy.special.ndtri(uniform)"

TRAINING_DOMAIN = "training"
MODEL_SELECTION_VALIDATION_DOMAIN = "model-selection-validation"
DEVELOPMENT_REPORTING_TEST_DOMAIN = "development-reporting-test"
PUBLIC_DOMAINS = (
    TRAINING_DOMAIN,
    MODEL_SELECTION_VALIDATION_DOMAIN,
    DEVELOPMENT_REPORTING_TEST_DOMAIN,
)

ALLOCATION_STREAM = "balanced-dirichlet-allocation"
ROOT_TOTAL_STREAM = "gamma-root-total"
STANDARD_NORMAL_STREAM = "projected-standard-normal"
SIMULATOR_STREAMS = (
    ALLOCATION_STREAM,
    ROOT_TOTAL_STREAM,
    STANDARD_NORMAL_STREAM,
)

DEVELOPMENT_MATRIX = tuple(
    (regime, family, "root")
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family in ("two_cell", "four_cell")
)
CASE_IDS = tuple(f"{regime}__{family}__root" for regime, family, _ in DEVELOPMENT_MATRIX)

_LOWER_OPEN_UNIT = np.nextafter(np.float64(0.0), np.float64(1.0))
_UPPER_OPEN_UNIT = np.nextafter(np.float64(1.0), np.float64(0.0))


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON text."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    """Return the SHA-256 identity of one canonical JSON value."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _array_sha256(values: NDArray[Any]) -> str:
    """Return a dtype-, shape-, and value-sensitive array identity."""
    contiguous = np.ascontiguousarray(values)
    header = _canonical_json(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    )
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_float64(values: ArrayLike, *, name: str) -> FloatArray:
    """Return one finite, owned, read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return cast(FloatArray, result)


def _unsigned_64(value: int, *, name: str) -> int:
    """Return a validated unsigned 64-bit integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if not 0 <= result < 2**64:
        raise ValueError(f"{name} must lie in [0, 2**64).")
    return result


def _power_of_two(value: int, *, name: str) -> int:
    """Return a validated positive power-of-two integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1 or result & (result - 1):
        raise ValueError(f"{name} must be a positive power of two.")
    return result


def _public_domain(domain: str) -> PublicDomain:
    """Return one allowed public domain and reject all other names."""
    if not isinstance(domain, str):
        raise TypeError("domain must be a string.")
    if domain not in PUBLIC_DOMAINS:
        raise ValueError("protected or unknown sample domains cannot be opened.")
    return cast(PublicDomain, domain)


def _case(case_id: str) -> tuple[str, Family]:
    """Return the frozen regime and family represented by ``case_id``."""
    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string.")
    for regime, family, tiling in DEVELOPMENT_MATRIX:
        expected = f"{regime}__{family}__{tiling}"
        if case_id == expected:
            return regime, cast(Family, family)
    raise ValueError("case_id is not one of the six frozen root cases.")


def domain_stream_seed(
    base_seed: int,
    *,
    case_id: str,
    domain: str,
    stream_name: str,
) -> int:
    """Derive the frozen unsigned 64-bit seed for one public simulator stream.

    The byte contract is exactly

    ``SHA256(PROTOCOL || uint64_le(seed) || case_id || domain || stream)``;

    the returned seed is the little-endian integer in its first eight bytes.
    Domain validation deliberately precedes hashing.
    """
    normalized_base = _unsigned_64(base_seed, name="base_seed")
    _case(case_id)
    normalized_domain = _public_domain(domain)
    if not isinstance(stream_name, str):
        raise TypeError("stream_name must be a string.")
    if stream_name not in SIMULATOR_STREAMS:
        raise ValueError("unknown simulator stream name.")
    digest = hashlib.sha256(PROTOCOL.encode("ascii"))
    digest.update(normalized_base.to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(normalized_domain.encode("ascii"))
    digest.update(stream_name.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _permutation(
    values: ArrayLike | None,
    size: int,
    *,
    name: str,
) -> IntArray:
    """Return a validated permutation, defaulting to canonical order."""
    if values is None:
        result = np.arange(size, dtype=np.int64)
    else:
        raw = np.asarray(values)
        if (
            raw.ndim != 1
            or raw.shape != (size,)
            or not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)
        ):
            raise ValueError(f"{name} must be an integer permutation of length {size}.")
        result = np.asarray(raw, dtype=np.int64)
        if not np.array_equal(np.sort(result), np.arange(size, dtype=np.int64)):
            raise ValueError(f"{name} must be an integer permutation of length {size}.")
        result = np.array(result, copy=True)
    result.setflags(write=False)
    return cast(IntArray, result)


def _canonicalize_spectrum_signs(
    spectrum: RootResidualSpectrum,
    observation_ids: IntArray,
) -> RootResidualSpectrum:
    """Make eigenvector signs invariant to a permutation of observations."""
    basis = np.array(spectrum.basis, dtype=np.float64, copy=True)
    canonical_order = np.argsort(observation_ids, kind="stable")
    canonical_basis = basis[canonical_order]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(canonical_basis[:, column])))
        if canonical_basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
            canonical_basis[:, column] *= -1.0
    return RootResidualSpectrum(
        spectrum.observation_mean_design,
        spectrum.noise_sd,
        basis,
        spectrum.eigenvalues,
        total_variance=spectrum.total_variance,
        discarded_variance=spectrum.discarded_variance,
        requested_retained_variance_fraction=(
            spectrum.requested_retained_variance_fraction
        ),
        eigenvalue_tolerance=spectrum.eigenvalue_tolerance,
        cell_alphas_sha256=spectrum.cell_alphas_sha256,
        design_sha256=spectrum.design_sha256,
        noise_sd_sha256=spectrum.noise_sd_sha256,
    )


@dataclass(frozen=True, slots=True)
class TinyScoreDomainEvidence:
    """Strict identities for one constructed simulator domain."""

    schema: str
    protocol: str
    case_id: str
    domain: PublicDomain
    base_seed: int
    sample_count: int
    gamma_shape: float
    gamma_rate: float
    conditioning_center: float
    conditioning_scale: float
    construction_method: str
    uniform_generator: str
    uniform_draw_contract: str
    root_total_transform: str
    standard_normal_transform: str
    numpy_version: str
    scipy_version: str
    stream_seeds: tuple[tuple[str, int], ...]
    scientific_input_sha256: str
    spectrum_sha256: str
    allocation_artifact_sha256: str
    array_sha256: tuple[tuple[str, str], ...]
    sha256: str

    def payload(self, *, include_sha256: bool = True) -> dict[str, object]:
        """Return the strict JSON-compatible evidence payload."""
        result: dict[str, object] = {
            "schema": self.schema,
            "protocol": self.protocol,
            "case_id": self.case_id,
            "domain": self.domain,
            "base_seed": self.base_seed,
            "sample_count": self.sample_count,
            "gamma_shape": self.gamma_shape,
            "gamma_rate": self.gamma_rate,
            "conditioning_center": self.conditioning_center,
            "conditioning_scale": self.conditioning_scale,
            "construction_method": self.construction_method,
            "uniform_generator": self.uniform_generator,
            "uniform_draw_contract": self.uniform_draw_contract,
            "root_total_transform": self.root_total_transform,
            "standard_normal_transform": self.standard_normal_transform,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "stream_seeds": dict(self.stream_seeds),
            "scientific_input_sha256": self.scientific_input_sha256,
            "spectrum_sha256": self.spectrum_sha256,
            "allocation_artifact_sha256": self.allocation_artifact_sha256,
            "array_sha256": dict(self.array_sha256),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def verify(self) -> None:
        """Raise if the evidence envelope does not match its own digest."""
        observed = _sha256_json(self.payload(include_sha256=False))
        if observed != self.sha256:
            raise ValueError("tiny-domain evidence SHA-256 does not replay.")


@dataclass(frozen=True, slots=True)
class TinyScoreDomain:
    """One immutable observation-blind tiny simulator domain."""

    case_id: str
    domain: PublicDomain
    spectrum: RootResidualSpectrum
    root_total_uniform: FloatArray
    gaussian_noise_uniform: FloatArray
    total_mass: FloatArray
    raw_log_mass: FloatArray
    allocation_residual: FloatArray
    gaussian_noise: FloatArray
    standardized_draw: FloatArray
    mass_score_target: FloatArray
    observation_score_target: FloatArray
    evidence: TinyScoreDomainEvidence

    @property
    def T(self) -> FloatArray:
        """Alias for the sampled root totals."""
        return self.total_mass

    @property
    def raw_tau(self) -> FloatArray:
        """Alias for the unstandardized ``log(T)`` conditioner."""
        return self.raw_log_mass

    @property
    def xi(self) -> FloatArray:
        """Alias for the unit-mass allocation residual."""
        return self.allocation_residual

    @property
    def epsilon(self) -> FloatArray:
        """Alias for the projected standard-normal noise."""
        return self.gaussian_noise

    @property
    def x(self) -> FloatArray:
        """Alias for the standardized noisy projected residual."""
        return self.standardized_draw

    @property
    def conditioning(self) -> tuple[float, float]:
        """Return the analytic center and scale of the raw log mass."""
        return (
            self.evidence.conditioning_center,
            self.evidence.conditioning_scale,
        )

    @property
    def hashes(self) -> Mapping[str, str]:
        """Return a read-only view of the authenticated array identities."""
        return MappingProxyType(dict(self.evidence.array_sha256))

    def verify(self) -> None:
        """Raise if any returned array or evidence identity has drifted."""
        self.evidence.verify()
        arrays = {
            "root_total_uniform": self.root_total_uniform,
            "gaussian_noise_uniform": self.gaussian_noise_uniform,
            "total_mass": self.total_mass,
            "raw_log_mass": self.raw_log_mass,
            "allocation_residual": self.allocation_residual,
            "gaussian_noise": self.gaussian_noise,
            "standardized_draw": self.standardized_draw,
            "mass_score_target": self.mass_score_target,
            "observation_score_target": self.observation_score_target,
        }
        observed = tuple((name, _array_sha256(values)) for name, values in arrays.items())
        if observed != self.evidence.array_sha256:
            raise ValueError("tiny-domain arrays do not match their authenticated hashes.")


def _pcg64_uniforms(sample_count: int, dimension: int, seed: int) -> FloatArray:
    """Return one independent PCG64 catalogue in the open unit cube."""
    generator = np.random.Generator(np.random.PCG64(seed))
    uniforms = np.asarray(
        generator.random((sample_count, dimension)),
        dtype=np.float64,
    )
    uniforms = np.clip(uniforms, _LOWER_OPEN_UNIT, _UPPER_OPEN_UNIT)
    return _readonly_float64(uniforms, name="PCG64 uniforms")


def _spectrum_sha256(spectrum: RootResidualSpectrum) -> str:
    """Return one strict identity for the complete numerical spectrum."""
    return _sha256_json(
        {
            "observation_mean_design_sha256": _array_sha256(
                spectrum.observation_mean_design
            ),
            "noise_sd_sha256": _array_sha256(spectrum.noise_sd),
            "basis_sha256": _array_sha256(spectrum.basis),
            "eigenvalues_sha256": _array_sha256(spectrum.eigenvalues),
            "total_variance": spectrum.total_variance,
            "discarded_variance": spectrum.discarded_variance,
            "requested_retained_variance_fraction": (
                spectrum.requested_retained_variance_fraction
            ),
            "retained_variance_fraction": spectrum.retained_variance_fraction,
            "eigenvalue_tolerance": spectrum.eigenvalue_tolerance,
            "cell_alphas_sha256": spectrum.cell_alphas_sha256,
            "design_sha256": spectrum.design_sha256,
            "noise_sd_identity": spectrum.noise_sd_sha256,
        }
    )


def simulate_tiny_score_domain(
    case_id: str,
    *,
    domain: str,
    sample_count: int,
    base_seed: int,
    cell_permutation: ArrayLike | None = None,
    observation_permutation: ArrayLike | None = None,
) -> TinyScoreDomain:
    """Construct one complete observation-blind N1 simulator domain.

    ``cell_permutation`` and ``observation_permutation`` are scientific
    invariance controls.  Stable native-cell and observation identifiers keep
    the generated source catalogue aligned with the canonical case.
    """
    regime_name, family = _case(case_id)
    normalized_domain = _public_domain(domain)
    count = _power_of_two(sample_count, name="sample_count")
    normalized_base_seed = _unsigned_64(base_seed, name="base_seed")
    del regime_name, family
    root_case = aggregation_error_tiny_oracle.tiny_root_case(case_id)
    shapes, gamma_rate, design, _, noise = root_case.arrays()
    canonical_shapes = np.asarray(shapes, dtype=np.float64)
    canonical_design = np.asarray(design, dtype=np.float64)
    canonical_noise = np.asarray(noise, dtype=np.float64)

    cell_order = _permutation(
        cell_permutation,
        canonical_shapes.size,
        name="cell_permutation",
    )
    observation_order = _permutation(
        observation_permutation,
        canonical_design.shape[0],
        name="observation_permutation",
    )
    cell_ids = np.arange(canonical_shapes.size, dtype=np.int64)[cell_order]
    observation_ids = np.arange(canonical_design.shape[0], dtype=np.int64)[
        observation_order
    ]
    permuted_shapes = canonical_shapes[cell_order]
    permuted_design = canonical_design[observation_order][:, cell_order]
    permuted_noise = canonical_noise[observation_order]

    identity_aggregation = AdditiveDirichletAggregation(
        permuted_shapes,
        permuted_design,
        permuted_noise,
        np.eye(permuted_design.shape[0], dtype=np.float64),
    )
    raw_spectrum = RootResidualSpectrum.from_aggregation(
        identity_aggregation,
        retained_variance_fraction=1.0,
    )
    spectrum = _canonicalize_spectrum_signs(raw_spectrum, observation_ids)
    if spectrum.retained_rank < 1:
        raise RuntimeError("a frozen tiny root case has no non-Gaussian residual direction.")

    seeds = {
        stream: domain_stream_seed(
            normalized_base_seed,
            case_id=case_id,
            domain=normalized_domain,
            stream_name=stream,
        )
        for stream in SIMULATOR_STREAMS
    }
    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("simulator stream seed collision.")
    projected_aggregation = AdditiveDirichletAggregation(
        permuted_shapes,
        permuted_design,
        permuted_noise,
        spectrum.basis,
    )
    allocation_artifact = ConditionalAllocationMixture.from_aggregation(
        projected_aggregation,
        np.zeros(permuted_shapes.shape, dtype=np.int64),
        sample_count=count,
        source_seed=seeds[ALLOCATION_STREAM],
        source_provenance=(
            f"{PROTOCOL}:{case_id}:{normalized_domain}:"
            f"{ALLOCATION_STREAM}:S={count}"
        ),
        cell_ids=cell_ids,
        construction_method=CONSTRUCTION_METHOD,
    )
    allocation = _readonly_float64(
        allocation_artifact.projected_unit_mass_residual_factors[:, :, 0],
        name="allocation_residual",
    )
    if allocation.shape != (count, spectrum.retained_rank):
        raise RuntimeError("allocation simulator returned an unexpected shape.")

    gamma_shape = float(math.fsum(float(value) for value in canonical_shapes))
    rate = float(gamma_rate)
    mass_uniform = _pcg64_uniforms(
        count,
        1,
        seeds[ROOT_TOTAL_STREAM],
    )[:, 0]
    total_mass = _readonly_float64(
        special.gammaincinv(gamma_shape, mass_uniform) / rate,
        name="total_mass",
    )
    if np.any(total_mass <= 0.0):
        raise FloatingPointError("Gamma inverse returned a non-positive root total.")
    raw_log_mass = _readonly_float64(np.log(total_mass), name="raw_log_mass")

    noise_uniform = _pcg64_uniforms(
        count,
        spectrum.retained_rank,
        seeds[STANDARD_NORMAL_STREAM],
    )
    gaussian_noise = _readonly_float64(
        special.ndtri(noise_uniform),
        name="gaussian_noise",
    )
    standardized = _readonly_float64(
        standardize_simulator_draw(
            total_mass,
            spectrum.eigenvalues,
            allocation,
            gaussian_noise,
        ),
        name="standardized_draw",
    )
    mass_score = _readonly_float64(
        component_partial_log_mass_score(
            total_mass,
            spectrum.eigenvalues,
            allocation,
            gaussian_noise,
            standardized,
        ),
        name="mass_score_target",
    )
    observation_score = _readonly_float64(
        component_observation_score(
            total_mass,
            spectrum.eigenvalues,
            gaussian_noise,
        ),
        name="observation_score_target",
    )
    conditioning_center, conditioning_scale = gamma_log_mass_conditioning(
        gamma_shape,
        rate,
    )

    arrays = {
        "root_total_uniform": mass_uniform,
        "gaussian_noise_uniform": noise_uniform,
        "total_mass": total_mass,
        "raw_log_mass": raw_log_mass,
        "allocation_residual": allocation,
        "gaussian_noise": gaussian_noise,
        "standardized_draw": standardized,
        "mass_score_target": mass_score,
        "observation_score_target": observation_score,
    }
    array_hashes = tuple((name, _array_sha256(values)) for name, values in arrays.items())
    scientific_input_sha256 = _sha256_json(
        {
            "tiny_root_definitions_sha256": (
                aggregation_error_tiny_oracle.definitions_sha256()
            ),
            "case_id": case_id,
            "cell_alphas_sha256": _array_sha256(permuted_shapes),
            "design_sha256": _array_sha256(permuted_design),
            "noise_sd_sha256": _array_sha256(permuted_noise),
            "cell_ids_sha256": _array_sha256(cell_ids),
            "observation_ids_sha256": _array_sha256(observation_ids),
            "gamma_shape": gamma_shape,
            "gamma_rate": rate,
        }
    )
    evidence_without_sha: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": PROTOCOL,
        "case_id": case_id,
        "domain": normalized_domain,
        "base_seed": normalized_base_seed,
        "sample_count": count,
        "gamma_shape": gamma_shape,
        "gamma_rate": rate,
        "conditioning_center": conditioning_center,
        "conditioning_scale": conditioning_scale,
        "construction_method": CONSTRUCTION_METHOD,
        "uniform_generator": UNIFORM_GENERATOR,
        "uniform_draw_contract": UNIFORM_DRAW_CONTRACT,
        "root_total_transform": ROOT_TOTAL_TRANSFORM,
        "standard_normal_transform": STANDARD_NORMAL_TRANSFORM,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "stream_seeds": seeds,
        "scientific_input_sha256": scientific_input_sha256,
        "spectrum_sha256": _spectrum_sha256(spectrum),
        "allocation_artifact_sha256": allocation_artifact.sha256,
        "array_sha256": dict(array_hashes),
    }
    evidence = TinyScoreDomainEvidence(
        schema=EVIDENCE_SCHEMA,
        protocol=PROTOCOL,
        case_id=case_id,
        domain=normalized_domain,
        base_seed=normalized_base_seed,
        sample_count=count,
        gamma_shape=gamma_shape,
        gamma_rate=rate,
        conditioning_center=conditioning_center,
        conditioning_scale=conditioning_scale,
        construction_method=CONSTRUCTION_METHOD,
        uniform_generator=UNIFORM_GENERATOR,
        uniform_draw_contract=UNIFORM_DRAW_CONTRACT,
        root_total_transform=ROOT_TOTAL_TRANSFORM,
        standard_normal_transform=STANDARD_NORMAL_TRANSFORM,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        stream_seeds=tuple(seeds.items()),
        scientific_input_sha256=scientific_input_sha256,
        spectrum_sha256=cast(str, evidence_without_sha["spectrum_sha256"]),
        allocation_artifact_sha256=allocation_artifact.sha256,
        array_sha256=array_hashes,
        sha256=_sha256_json(evidence_without_sha),
    )
    result = TinyScoreDomain(
        case_id=case_id,
        domain=normalized_domain,
        spectrum=spectrum,
        root_total_uniform=mass_uniform,
        gaussian_noise_uniform=noise_uniform,
        total_mass=total_mass,
        raw_log_mass=raw_log_mass,
        allocation_residual=allocation,
        gaussian_noise=gaussian_noise,
        standardized_draw=standardized,
        mass_score_target=mass_score,
        observation_score_target=observation_score,
        evidence=evidence,
    )
    result.verify()
    return result


__all__ = [
    "ALLOCATION_STREAM",
    "CASE_IDS",
    "DEVELOPMENT_MATRIX",
    "DEVELOPMENT_REPORTING_TEST_DOMAIN",
    "MODEL_SELECTION_VALIDATION_DOMAIN",
    "PROTOCOL",
    "PUBLIC_DOMAINS",
    "ROOT_TOTAL_STREAM",
    "SIMULATOR_STREAMS",
    "STANDARD_NORMAL_STREAM",
    "TRAINING_DOMAIN",
    "TinyScoreDomain",
    "TinyScoreDomainEvidence",
    "domain_stream_seed",
    "simulate_tiny_score_domain",
]
