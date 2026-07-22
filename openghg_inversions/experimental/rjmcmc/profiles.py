"""Experimental run-profile and provenance primitives for spatial RJMCMC.

The classes in this module describe the scalar settings needed to identify and
replay an experimental run. They deliberately do not serialize observations,
sensitivities, coordinates, or other scientific data arrays. Callers should
instead identify those inputs with :class:`InputReference` records and stable
checksums.

The manifest schema is intentionally small. Future retained-draw and checkpoint
formats can embed the canonical JSON manifest or its checksum without depending
on Python dataclass serialization details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from json import dumps
from math import fsum, isclose, isfinite
from numbers import Integral
from string import hexdigits
from typing import ClassVar, TypeAlias

from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
    LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
    LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
    SamplerConfig,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

RUN_MANIFEST_SCHEMA_VERSION = 2

_INDEPENDENT_ERROR_MODEL = "independent_gaussian"
_OU_ERROR_MODEL = "independent_site_ou_nugget"
_SHARED_HIERARCHY_PARAMETERIZATION = "shared_arithmetic_moments_log_state"


def _positive_float(value: float, *, name: str) -> float:
    """Return ``value`` as a finite positive float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _non_negative_integer(value: int, *, name: str) -> int:
    """Return ``value`` as a non-negative built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _non_empty_text(value: str, *, name: str) -> str:
    """Return stripped non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _bounded_uniform_limits(
    lower: tuple[float, ...] | None,
    upper: tuple[float, ...] | None,
    *,
    name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate paired positive bounds for a vector of uniform priors."""
    if lower is None or upper is None:
        raise ValueError(f"{name}_prior_lower and {name}_prior_upper are required.")
    lower_values = tuple(_positive_float(value, name=f"{name}_prior_lower") for value in lower)
    upper_values = tuple(_positive_float(value, name=f"{name}_prior_upper") for value in upper)
    if not lower_values or len(lower_values) != len(upper_values):
        raise ValueError(f"{name} prior bounds must be non-empty tuples with matching shapes.")
    if any(lower_value >= upper_value for lower_value, upper_value in zip(lower_values, upper_values)):
        raise ValueError(f"Each {name} prior lower bound must be smaller than its upper bound.")
    return lower_values, upper_values


@dataclass(frozen=True, slots=True)
class TargetSettings:
    """Scalar target settings that materially define an RJMCMC run.

    The coefficient-prior parameters are arithmetic moments of the lognormal
    distribution, matching :class:`~openghg_inversions.experimental.rjmcmc.core.TransDimensionalProblem`.
    Observation vectors, their standard deviations, sensitivities, grid
    coordinates, and OU site/group alignment arrays remain external data inputs
    and are not copied into this value object. They should be identified by
    :class:`InputReference` records and checksums.

    Args:
        k_min: Smallest supported active-region count.
        k_max: Largest supported active-region count.
        k_prior_probabilities: Normalized probabilities ordered from ``k_min``
            through ``k_max``.
        coefficient_prior_mean: Arithmetic lognormal prior mean.
        coefficient_prior_sd: Arithmetic lognormal prior standard deviation.
        observation_error_model: Stable identifier for either independent
            Gaussian errors or the independent-site OU mismatch plus known
            measurement-nugget model.
        observation_time_unit: Unit of the observation times and correlation
            timescales. Required only for the OU model.
        mismatch_sd_prior_lower: Positive lower bounds for the OU mismatch-SD
            bounded-uniform priors. Required only for the OU model.
        mismatch_sd_prior_upper: Upper bounds paired with
            ``mismatch_sd_prior_lower``.
        correlation_timescale_prior_lower: Positive lower bounds for the OU
            timescale bounded-uniform priors. Required only for the OU model.
        correlation_timescale_prior_upper: Upper bounds paired with
            ``correlation_timescale_prior_lower``.
        shared_coefficient_hierarchy: Whether the dynamic Voronoi coefficients
            use a shared arithmetic-moment hierarchy. Fixed outer coefficients
            always remain outside this hierarchy.
        coefficient_hierarchy_parameterization: Stable hierarchy state
            parameterization identifier. Required exactly when the hierarchy
            is enabled.
        mean_hyperprior_median: Median of the shared arithmetic-mean lognormal
            hyperprior.
        mean_hyperprior_log_sd: Log-space standard deviation of that
            hyperprior.
        sd_hyperprior_median: Median of the shared arithmetic-SD lognormal
            hyperprior.
        sd_hyperprior_log_sd: Log-space standard deviation of that hyperprior.

    Raises:
        ValueError: If bounds, probabilities, moments, or the likelihood model
            are malformed.
    """

    k_min: int
    k_max: int
    k_prior_probabilities: tuple[float, ...]
    coefficient_prior_mean: float
    coefficient_prior_sd: float
    observation_error_model: str = _INDEPENDENT_ERROR_MODEL
    observation_time_unit: str | None = None
    mismatch_sd_prior_lower: tuple[float, ...] | None = None
    mismatch_sd_prior_upper: tuple[float, ...] | None = None
    correlation_timescale_prior_lower: tuple[float, ...] | None = None
    correlation_timescale_prior_upper: tuple[float, ...] | None = None
    shared_coefficient_hierarchy: bool = False
    coefficient_hierarchy_parameterization: str | None = None
    mean_hyperprior_median: float | None = None
    mean_hyperprior_log_sd: float | None = None
    sd_hyperprior_median: float | None = None
    sd_hyperprior_log_sd: float | None = None

    def __post_init__(self) -> None:
        """Validate and own an immutable normalized target description."""
        k_min = _non_negative_integer(self.k_min, name="k_min")
        k_max = _non_negative_integer(self.k_max, name="k_max")
        if k_min < 1 or k_max < k_min:
            raise ValueError("Require 1 <= k_min <= k_max.")

        probabilities: list[float] = []
        for value in self.k_prior_probabilities:
            if isinstance(value, bool):
                raise ValueError("k_prior_probabilities must contain finite non-negative values.")
            probability = float(value)
            if not isfinite(probability) or probability < 0.0:
                raise ValueError("k_prior_probabilities must contain finite non-negative values.")
            probabilities.append(probability)
        if len(probabilities) != k_max - k_min + 1:
            raise ValueError("k_prior_probabilities must contain one value for each supported k.")
        if not isclose(fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("k_prior_probabilities must sum to one.")

        observation_error_model = _non_empty_text(
            self.observation_error_model,
            name="observation_error_model",
        )
        if observation_error_model not in (_INDEPENDENT_ERROR_MODEL, _OU_ERROR_MODEL):
            raise ValueError(
                "observation_error_model must be 'independent_gaussian' or 'independent_site_ou_nugget'."
            )

        ou_names = (
            "observation_time_unit",
            "mismatch_sd_prior_lower",
            "mismatch_sd_prior_upper",
            "correlation_timescale_prior_lower",
            "correlation_timescale_prior_upper",
        )
        if observation_error_model == _INDEPENDENT_ERROR_MODEL:
            if any(getattr(self, name) is not None for name in ou_names):
                raise ValueError("OU error-model settings require 'independent_site_ou_nugget'.")
        else:
            if self.observation_time_unit is None:
                raise ValueError("observation_time_unit is required for the OU error model.")
            object.__setattr__(
                self,
                "observation_time_unit",
                _non_empty_text(self.observation_time_unit, name="observation_time_unit"),
            )
            mismatch_lower, mismatch_upper = _bounded_uniform_limits(
                self.mismatch_sd_prior_lower,
                self.mismatch_sd_prior_upper,
                name="mismatch_sd",
            )
            timescale_lower, timescale_upper = _bounded_uniform_limits(
                self.correlation_timescale_prior_lower,
                self.correlation_timescale_prior_upper,
                name="correlation_timescale",
            )
            object.__setattr__(self, "mismatch_sd_prior_lower", mismatch_lower)
            object.__setattr__(self, "mismatch_sd_prior_upper", mismatch_upper)
            object.__setattr__(self, "correlation_timescale_prior_lower", timescale_lower)
            object.__setattr__(self, "correlation_timescale_prior_upper", timescale_upper)

        if not isinstance(self.shared_coefficient_hierarchy, bool):
            raise ValueError("shared_coefficient_hierarchy must be a boolean.")
        hierarchy_names = (
            "coefficient_hierarchy_parameterization",
            "mean_hyperprior_median",
            "mean_hyperprior_log_sd",
            "sd_hyperprior_median",
            "sd_hyperprior_log_sd",
        )
        if not self.shared_coefficient_hierarchy:
            if any(getattr(self, name) is not None for name in hierarchy_names):
                raise ValueError("Hierarchy settings require shared_coefficient_hierarchy=True.")
        else:
            if self.coefficient_hierarchy_parameterization != _SHARED_HIERARCHY_PARAMETERIZATION:
                raise ValueError(
                    f"coefficient_hierarchy_parameterization must be {_SHARED_HIERARCHY_PARAMETERIZATION!r}."
                )
            for name in hierarchy_names[1:]:
                value = getattr(self, name)
                if value is None:
                    raise ValueError(f"{name} is required for the shared coefficient hierarchy.")
                object.__setattr__(self, name, _positive_float(value, name=name))

        object.__setattr__(self, "k_min", k_min)
        object.__setattr__(self, "k_max", k_max)
        object.__setattr__(self, "k_prior_probabilities", tuple(probabilities))
        object.__setattr__(
            self,
            "coefficient_prior_mean",
            _positive_float(self.coefficient_prior_mean, name="coefficient_prior_mean"),
        )
        object.__setattr__(
            self,
            "coefficient_prior_sd",
            _positive_float(self.coefficient_prior_sd, name="coefficient_prior_sd"),
        )
        object.__setattr__(self, "observation_error_model", observation_error_model)


@dataclass(frozen=True, slots=True)
class InputReference:
    """Stable provenance for one external run input without embedding its data.

    Args:
        role: Scientific role, such as ``"observations"`` or ``"footprints"``.
        identifier: Stable path, URI, catalogue identifier, or synthetic-input
            name.
        sha256: Optional hexadecimal SHA-256 checksum. Checksums are strongly
            recommended for file-backed inputs.

    Raises:
        ValueError: If text fields or the checksum are malformed.
    """

    role: str
    identifier: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Normalize text and validate an optional SHA-256 digest."""
        role = _non_empty_text(self.role, name="role")
        identifier = _non_empty_text(self.identifier, name="identifier")
        checksum = self.sha256
        if checksum is not None:
            if not isinstance(checksum, str):
                raise ValueError("sha256 must be a 64-character hexadecimal string.")
            checksum = checksum.strip().lower()
            if len(checksum) != 64 or any(character not in hexdigits for character in checksum):
                raise ValueError("sha256 must be a 64-character hexadecimal string.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "sha256", checksum)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Code and external-input identities associated with a run.

    Input references are stored in canonical role/identifier order so callers
    constructing the same provenance in different orders obtain identical
    manifests.

    Args:
        code_revision: Optional source-control revision or release identifier.
        inputs: External input references. Duplicate role/identifier pairs are
            rejected.

    Raises:
        ValueError: If the revision, input types, or input identities are
            malformed.
    """

    code_revision: str | None = None
    inputs: tuple[InputReference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the code revision and canonicalize input ordering."""
        code_revision = self.code_revision
        if code_revision is not None:
            code_revision = _non_empty_text(code_revision, name="code_revision")

        inputs = tuple(self.inputs)
        if any(not isinstance(reference, InputReference) for reference in inputs):
            raise ValueError("inputs must contain only InputReference values.")
        inputs = tuple(sorted(inputs, key=lambda reference: (reference.role, reference.identifier)))
        identities = [(reference.role, reference.identifier) for reference in inputs]
        if len(identities) != len(set(identities)):
            raise ValueError("inputs must not repeat a role/identifier pair.")

        object.__setattr__(self, "code_revision", code_revision)
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True, slots=True)
class RunProfile:
    """Validated experimental RJMCMC settings and deterministic provenance.

    ``RunProfile`` requires an explicit non-negative sampler seed. A
    :class:`~openghg_inversions.experimental.rjmcmc.sampling.SamplerConfig` with ``seed=None``
    remains useful for exploratory sampling but cannot identify a reproducible
    run.

    Args:
        name: Stable human-readable profile identifier.
        target: Scalar prior and likelihood-model settings.
        sampler: Complete transition schedule and random-number settings.
        retention: Collection-time warmup and thinning declaration.
        provenance: Source revision and external input identities.

    Raises:
        ValueError: If component types, seed, or retention range are malformed.
    """

    schema_version: ClassVar[int] = RUN_MANIFEST_SCHEMA_VERSION

    name: str
    target: TargetSettings
    sampler: SamplerConfig
    retention: RetentionSettings = field(default_factory=RetentionSettings)
    provenance: RunProvenance = field(default_factory=RunProvenance)

    def __post_init__(self) -> None:
        """Validate component types and cross-component run constraints."""
        name = _non_empty_text(self.name, name="name")
        if not isinstance(self.target, TargetSettings):
            raise ValueError("target must be a TargetSettings instance.")
        if not isinstance(self.sampler, SamplerConfig):
            raise ValueError("sampler must be a SamplerConfig instance.")
        if not isinstance(self.retention, RetentionSettings):
            raise ValueError("retention must be a RetentionSettings instance.")
        if not isinstance(self.provenance, RunProvenance):
            raise ValueError("provenance must be a RunProvenance instance.")

        seed = self.sampler.seed
        if seed is None:
            raise ValueError("sampler.seed must be explicit for a reproducible run profile.")
        _non_negative_integer(seed, name="sampler.seed")
        if self.retention.warmup_transitions > self.sampler.iterations:
            raise ValueError("warmup_transitions must not exceed sampler.iterations.")

        schedule_profile = self.sampler.schedule_profile
        has_ou = self.target.observation_error_model == _OU_ERROR_MODEL
        has_hierarchy = self.target.shared_coefficient_hierarchy
        if schedule_profile in ("default", LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE):
            if has_ou or has_hierarchy:
                raise ValueError(
                    "The default and 14-slot schedules require an independent, nonhierarchical target."
                )
        elif schedule_profile == LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE:
            if not has_ou or has_hierarchy:
                raise ValueError("The 16-slot schedule requires OU errors without a hierarchy.")
        elif schedule_profile == LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE:
            if not has_ou or not has_hierarchy:
                raise ValueError("The 17-slot schedule requires OU errors and a shared hierarchy.")
        object.__setattr__(self, "name", name)

    def to_manifest(self) -> dict[str, JsonValue]:
        """Return the versioned manifest as JSON-serializable built-in values."""
        sampler = self.sampler
        target = self.target
        seed = sampler.seed
        if seed is None:  # guarded by __post_init__; keeps type narrowing local
            raise RuntimeError("A validated run profile must have an explicit sampler seed.")
        return {
            "schema_version": self.schema_version,
            "profile_name": self.name,
            "target": {
                "active_region_count": {
                    "minimum": target.k_min,
                    "maximum": target.k_max,
                    "prior_probabilities": list(target.k_prior_probabilities),
                },
                "coefficient_prior": {
                    "distribution": "lognormal",
                    "parameterization": "arithmetic_moments",
                    "mean": target.coefficient_prior_mean,
                    "standard_deviation": target.coefficient_prior_sd,
                },
                "observation_error_model": target.observation_error_model,
                "observation_error_model_settings": (
                    None
                    if target.observation_error_model == _INDEPENDENT_ERROR_MODEL
                    else {
                        "time_unit": target.observation_time_unit,
                        "mismatch_sd_prior": {
                            "distribution": "bounded_uniform",
                            "lower": list(target.mismatch_sd_prior_lower or ()),
                            "upper": list(target.mismatch_sd_prior_upper or ()),
                        },
                        "correlation_timescale_prior": {
                            "distribution": "bounded_uniform",
                            "lower": list(target.correlation_timescale_prior_lower or ()),
                            "upper": list(target.correlation_timescale_prior_upper or ()),
                        },
                    }
                ),
                "dynamic_coefficient_hierarchy": (
                    None
                    if not target.shared_coefficient_hierarchy
                    else {
                        "parameterization": target.coefficient_hierarchy_parameterization,
                        "includes_fixed_outer_coefficients": False,
                        "mean_hyperprior_median": target.mean_hyperprior_median,
                        "mean_hyperprior_log_sd": target.mean_hyperprior_log_sd,
                        "sd_hyperprior_median": target.sd_hyperprior_median,
                        "sd_hyperprior_log_sd": target.sd_hyperprior_log_sd,
                    }
                ),
            },
            "sampler": {
                "iterations": sampler.iterations,
                "coefficient_proposal_sd": sampler.coefficient_proposal_sd,
                "birth_proposal_sd": sampler.birth_proposal_sd,
                "fixed_coefficient_proposal_sd": sampler.fixed_coefficient_proposal_sd,
                "schedule_profile": sampler.schedule_profile,
                "mismatch_sd_proposal_sd": sampler.mismatch_sd_proposal_sd,
                "correlation_timescale_proposal_sd": sampler.correlation_timescale_proposal_sd,
                "eta_proposal_sd": sampler.eta_proposal_sd,
                "zeta_proposal_sd": sampler.zeta_proposal_sd,
                "seed": int(seed),
                "backend": sampler.backend,
                "nucleus_move": sampler.nucleus_move,
                "local_move_scale": sampler.local_move_scale,
            },
            "retention": {
                "warmup_transitions": self.retention.warmup_transitions,
                "thin": self.retention.thin,
            },
            "provenance": {
                "code_revision": self.provenance.code_revision,
                "inputs": [
                    {
                        "role": reference.role,
                        "identifier": reference.identifier,
                        "sha256": reference.sha256,
                    }
                    for reference in self.provenance.inputs
                ],
            },
        }

    def to_json(self) -> str:
        """Return a canonical compact JSON representation of the manifest."""
        return dumps(
            self.to_manifest(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
