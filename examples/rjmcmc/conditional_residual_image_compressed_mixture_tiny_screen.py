#!/usr/bin/env python3
"""Screen structure-preserving compression of exact root allocation mixtures.

This development-only driver evaluates six source-pinned tiny root cases from
the C1 conditional-allocation oracle.  It first constructs the complete
analytic residual spectrum without consulting the realized observation, then
builds equal-weight scrambled-Sobol allocation banks in that fixed basis.  A
source bank must pass a predeclared sample-size suffix before any compressed
mixture is assessed.  Compression uses deterministic hard clustering and
exact within-cluster population moments.

The source and compressed likelihoods are scored separately against the exact
quadrature oracle.  Compression is also scored incrementally against its
locked source bank.  This program cannot read protected catalogues or publish
production inversions, and no result licenses structural inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy import __version__ as scipy_version

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    CompressedRootMixture,
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
Family = Literal["two_cell", "four_cell"]
Profile = Literal["smoke", "development"]
Stage = Literal["source", "compression"]
Likelihood = Callable[[float], float]

SCHEMA = "rjmcmc-conditional-residual-image-compressed-mixture-tiny-screen-v1"
PROTOCOL = "root-exact-spectrum-sobol-moment-compression-v1"
SOURCE_LOCK_SCHEMA = "rjmcmc-compressed-mixture-common-source-lock-v1"
CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
DEVELOPMENT_MATRIX = tuple(
    (regime, family, "root")
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family in ("two_cell", "four_cell")
)
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)

SMOKE_SOURCE_SAMPLE_COUNTS = (4_096,)
SMOKE_COMPONENT_COUNTS = (8, 16)
DEVELOPMENT_SOURCE_SAMPLE_COUNTS = (65_536, 262_144, 1_048_576)
DEVELOPMENT_COMPONENT_COUNTS = (16, 32, 64, 128, 256, 512, 1_024)
DEVELOPMENT_SEED = 731
CONFIRMATION_SEEDS = (1_877, 4_099, 8_317)
SMOKE_MINIMUM_SOURCE_SUFFIX = 1
SMOKE_MINIMUM_COMPRESSION_SUFFIX = 1
DEVELOPMENT_MINIMUM_SOURCE_SUFFIX = 2
DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX = 2

RETAINED_VARIANCE_FRACTION = 1.0
CLUSTER_RESTART_COUNT = 3
CLUSTER_MAXIMUM_ITERATIONS = 100
PROTECTED_CATALOGUE_ACCESSED = False
PRODUCTION_OUTPUT_WRITTEN = False


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
    """Return the SHA-256 digest of one canonical JSON value."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _array_sha256(values: NDArray[Any]) -> str:
    """Return a dtype-, shape-, and value-sensitive array digest."""
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


def _protocol_payload() -> dict[str, Any]:
    """Return every setting that defines the scientific protocol."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stages": ("source", "compression"),
        "source_lock_schema": SOURCE_LOCK_SCHEMA,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix": DEVELOPMENT_MATRIX,
        "source_sample_counts": DEVELOPMENT_SOURCE_SAMPLE_COUNTS,
        "component_counts": DEVELOPMENT_COMPONENT_COUNTS,
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds_deferred": CONFIRMATION_SEEDS,
        "source_minimum_passing_suffix": DEVELOPMENT_MINIMUM_SOURCE_SUFFIX,
        "compression_minimum_passing_suffix": (DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX),
        "retained_variance_fraction": RETAINED_VARIANCE_FRACTION,
        "construction_method": CONSTRUCTION_METHOD,
        "mixture_rank": "full_exact_numerical_residual_rank",
        "cluster_restart_count": CLUSTER_RESTART_COUNT,
        "cluster_maximum_iterations": CLUSTER_MAXIMUM_ITERATIONS,
        "gradient_step": c1.GRADIENT_STEP,
        "thresholds": c1.THRESHOLDS,
        "pointwise_scoring": "c1-checkerboard-by-total-and-share-index-v1",
        "source_lock": (
            "a separate all-six merger selects the smallest common S starting "
            "an all-larger scientific-pass suffix of the required length"
        ),
        "compression_lock": (
            "smallest predeclared M starting an all-larger exact-scientific-"
            "pass suffix of the required length, using only the locked source"
        ),
        "confirmation": (
            "deferred; later stage must use the frozen lock and all three confirmation seeds without retuning"
        ),
    }


def _validate_protocol() -> None:
    """Fail closed if inherited exact definitions or local constants drift."""
    if c1.a1_definitions_sha256() != c1.A1_DEFINITIONS_SHA256:
        raise RuntimeError("inherited A1 numerical definitions no longer match their pin")
    counts = DEVELOPMENT_SOURCE_SAMPLE_COUNTS
    components = DEVELOPMENT_COMPONENT_COUNTS
    if (
        tuple(sorted(counts)) != counts
        or len(set(counts)) != len(counts)
        or any(value < 1 or value & (value - 1) for value in counts)
    ):
        raise RuntimeError("development source sizes must be unique increasing powers of two")
    if tuple(sorted(components)) != components or len(set(components)) != len(components):
        raise RuntimeError("development component counts must be unique and increasing")
    if CONFIRMATION_SEEDS != (1_877, 4_099, 8_317):
        raise RuntimeError("confirmation seed catalogue drifted")


def _git_revision() -> str | None:
    """Return the current Git revision when Git is available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_revision(expected: str | None) -> str:
    """Return a source revision cross-checked against the checkout."""
    if expected is not None and (
        len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("source_revision must be a 40-character lower-case Git SHA")
    observed = _git_revision()
    if expected is not None and observed is not None and observed != expected:
        raise RuntimeError("source_revision does not match the current Git checkout")
    revision = expected if expected is not None else observed
    if revision is None:
        raise RuntimeError("source_revision is required when Git is unavailable")
    return revision


def _driver_sha256() -> str:
    """Return the exact digest of this executable source."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _stable_lock(
    values: Sequence[int],
    passes: Sequence[bool],
    *,
    minimum_suffix_length: int,
) -> int | None:
    """Return the smallest value starting a sufficiently long passing suffix."""
    return c1._stable_lock_sample_count(
        values,
        passes,
        minimum_suffix_length=minimum_suffix_length,
    )


def _finite_difference_log_mass_gradient(
    likelihood: Likelihood,
    mass: float,
) -> float:
    """Return a centred finite-difference derivative with respect to log mass."""
    coordinate = np.asarray([math.log(mass)], dtype=np.float64)

    def in_coordinate(value: FloatArray) -> float:
        return likelihood(math.exp(float(value[0])))

    return float(c1._centered_gradient(in_coordinate, coordinate)[0])


def _gradient_catalogue(
    *,
    shapes: FloatArray,
    rate: float,
    masses: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_likelihood: Likelihood,
) -> list[dict[str, Any]]:
    """Return C1's frozen root gradient states with exact derivatives."""
    anchor = np.asarray([math.log(float(np.sum(shapes) / rate))], dtype=np.float64)
    states = c1._gradient_state_coordinates(
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_log_likelihood,
        prior_mean_coordinate=anchor,
    )
    return [
        {
            "state_id": state_id,
            "coordinate": coordinate.tolist(),
            "mass": math.exp(float(coordinate[0])),
            "exact_coordinate_gradient": _finite_difference_log_mass_gradient(
                exact_likelihood,
                math.exp(float(coordinate[0])),
            ),
        }
        for state_id, coordinate in states
    ]


def _scientific_evaluation(
    *,
    likelihood: Likelihood,
    likelihood_name: str,
    masses: FloatArray,
    observation: FloatArray,
    log_prior: FloatArray,
    exact_log_likelihood: FloatArray,
    exact_summary: dict[str, Any],
    gradient_states: Sequence[dict[str, Any]],
    validation_mask: BoolArray,
    include_timings: bool,
) -> dict[str, Any]:
    """Score one generic root likelihood against the exact C1 oracle."""
    del observation  # The closed likelihood callable owns the observation.
    started = time.perf_counter()
    candidate = np.asarray(
        [likelihood(float(state[0])) for state in masses],
        dtype=np.float64,
    )
    evaluation_seconds = time.perf_counter() - started if include_timings else None
    error = np.abs(candidate - exact_log_likelihood)
    summary = c1._posterior_summary(masses, log_prior, candidate)
    summary_errors, summary_errors_by_coordinate = c1._summary_errors(
        exact_summary,
        summary,
    )
    validation_prior_log_weights = log_prior[validation_mask]
    validation_prior_weights = np.exp(
        validation_prior_log_weights - c1._stable_logsumexp(validation_prior_log_weights)
    )
    validation_posterior_log_weights = validation_prior_log_weights + exact_log_likelihood[validation_mask]
    validation_posterior_weights = np.exp(
        validation_posterior_log_weights - c1._stable_logsumexp(validation_posterior_log_weights)
    )
    validation_error = error[validation_mask]
    gradient_audits: list[dict[str, Any]] = []
    for state in gradient_states:
        mass = float(state["mass"])
        exact_gradient = float(state["exact_coordinate_gradient"])
        candidate_gradient = _finite_difference_log_mass_gradient(
            likelihood,
            mass,
        )
        scaled_error = abs(candidate_gradient - exact_gradient) / (1.0 + abs(exact_gradient))
        gradient_audits.append(
            {
                "state_id": state["state_id"],
                "coordinate": state["coordinate"],
                "exact_coordinate_gradient": exact_gradient,
                f"{likelihood_name}_coordinate_gradient": candidate_gradient,
                "scaled_error": scaled_error,
            }
        )
    metrics = {
        "median_absolute_conditional_log_likelihood_error_nat": (
            c1._weighted_quantile(
                validation_error,
                validation_prior_weights,
                0.5,
            )
        ),
        "p99_absolute_conditional_log_likelihood_error_nat": (
            c1._weighted_quantile(
                validation_error,
                validation_posterior_weights,
                0.99,
            )
        ),
        "scaled_coordinate_gradient_error": max(float(audit["scaled_error"]) for audit in gradient_audits),
        "absolute_log_evidence_error_nat": abs(
            float(summary["log_evidence"]) - float(exact_summary["log_evidence"])
        ),
        **summary_errors,
    }
    checks = {
        name: bool(metrics[name] <= threshold) for name, threshold in c1.THRESHOLDS.items() if name in metrics
    }
    return {
        "likelihood_name": likelihood_name,
        "log_likelihood_sha256": _array_sha256(candidate),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_states_per_second": (
            None
            if evaluation_seconds is None or evaluation_seconds == 0.0
            else masses.shape[0] / evaluation_seconds
        ),
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "posterior_summary": summary,
        "posterior_errors_by_coordinate": summary_errors_by_coordinate,
        "gradient_audits": gradient_audits,
        "diagnostics": {
            "unweighted_full_grid_median_absolute_error_nat": float(np.median(error)),
            "unweighted_full_grid_p99_absolute_error_nat": float(np.quantile(error, 0.99)),
            "unweighted_full_grid_maximum_absolute_error_nat": float(np.max(error)),
            "pointwise_gate_weighting": {
                "median": ("normalized quadrature prior weights on the C1 checkerboard development view"),
                "p99": (
                    "normalized exact-posterior quadrature weights on the C1 checkerboard development view"
                ),
            },
        },
    }


def _incremental_evaluation(
    *,
    source_likelihood: Likelihood,
    compressed_likelihood: Likelihood,
    masses: FloatArray,
    log_prior: FloatArray,
    validation_mask: BoolArray,
    source_evaluation: dict[str, Any],
    compressed_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Measure only the extra error introduced by compression."""
    source_values = np.asarray(
        [source_likelihood(float(state[0])) for state in masses],
        dtype=np.float64,
    )
    compressed_values = np.asarray(
        [compressed_likelihood(float(state[0])) for state in masses],
        dtype=np.float64,
    )
    difference = np.abs(compressed_values - source_values)
    validation_prior_log_weights = log_prior[validation_mask]
    validation_prior_weights = np.exp(
        validation_prior_log_weights - c1._stable_logsumexp(validation_prior_log_weights)
    )
    validation_source_posterior_log_weights = validation_prior_log_weights + source_values[validation_mask]
    validation_source_posterior_weights = np.exp(
        validation_source_posterior_log_weights
        - c1._stable_logsumexp(validation_source_posterior_log_weights)
    )
    summary_errors, by_coordinate = c1._summary_errors(
        source_evaluation["posterior_summary"],
        compressed_evaluation["posterior_summary"],
    )
    gradient_errors = [
        abs(float(compressed["compressed_coordinate_gradient"]) - float(source["source_coordinate_gradient"]))
        / (1.0 + abs(float(source["source_coordinate_gradient"])))
        for source, compressed in zip(
            source_evaluation["gradient_audits"],
            compressed_evaluation["gradient_audits"],
            strict=True,
        )
    ]
    return {
        "median_absolute_conditional_log_likelihood_difference_nat": (
            c1._weighted_quantile(
                difference[validation_mask],
                validation_prior_weights,
                0.5,
            )
        ),
        "p99_absolute_conditional_log_likelihood_difference_nat": (
            c1._weighted_quantile(
                difference[validation_mask],
                validation_source_posterior_weights,
                0.99,
            )
        ),
        "maximum_scaled_coordinate_gradient_difference": max(gradient_errors),
        "absolute_log_evidence_difference_nat": abs(
            float(compressed_evaluation["posterior_summary"]["log_evidence"])
            - float(source_evaluation["posterior_summary"]["log_evidence"])
        ),
        **summary_errors,
        "posterior_errors_by_coordinate": by_coordinate,
    }


def _population_moments(samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the population mean and covariance of row-wise samples."""
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / float(samples.shape[0])
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64),
    )


def _compressed_moments(
    artifact: CompressedRootMixture,
) -> tuple[FloatArray, FloatArray]:
    """Return the global mean and covariance of a compressed mixture."""
    mean = artifact.weights @ artifact.means
    covariance = np.zeros(
        (artifact.mixture_rank, artifact.mixture_rank),
        dtype=np.float64,
    )
    for weight, component_mean, component_covariance in zip(
        artifact.weights,
        artifact.means,
        artifact.covariances,
        strict=True,
    ):
        displacement = component_mean - mean
        covariance += float(weight) * (component_covariance + np.outer(displacement, displacement))
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64),
    )


def _moment_diagnostics(
    source: ConditionalAllocationMixture,
    spectrum: RootResidualSpectrum,
    compressed: CompressedRootMixture | None = None,
) -> dict[str, Any]:
    """Report source analytic accuracy and exact compression moment closure."""
    locations = np.asarray(
        source.projected_unit_mass_residual_factors[:, :, 0],
        dtype=np.float64,
    )
    source_mean, source_covariance = _population_moments(locations)
    analytic_covariance = np.diag(spectrum.eigenvalues)
    result: dict[str, Any] = {
        "source_mean_maximum_absolute": float(np.max(np.abs(source_mean), initial=0.0)),
        "source_covariance_maximum_absolute_error_vs_analytic": float(
            np.max(
                np.abs(source_covariance - analytic_covariance),
                initial=0.0,
            )
        ),
        "source_covariance_relative_frobenius_error_vs_analytic": (
            float(np.linalg.norm(source_covariance - analytic_covariance))
            / max(
                float(np.linalg.norm(analytic_covariance)),
                float(np.finfo(np.float64).tiny),
            )
        ),
        "source_mean_sha256": _array_sha256(source_mean),
        "source_covariance_sha256": _array_sha256(source_covariance),
        "analytic_covariance_sha256": _array_sha256(analytic_covariance),
    }
    if compressed is not None:
        compressed_mean, compressed_covariance = _compressed_moments(compressed)
        result["compression"] = {
            "mean_maximum_absolute_difference_from_source": float(
                np.max(
                    np.abs(compressed_mean - source_mean[: compressed.mixture_rank]),
                    initial=0.0,
                )
            ),
            "covariance_maximum_absolute_difference_from_source": float(
                np.max(
                    np.abs(
                        compressed_covariance
                        - source_covariance[
                            : compressed.mixture_rank,
                            : compressed.mixture_rank,
                        ]
                    ),
                    initial=0.0,
                )
            ),
            "mean_sha256": _array_sha256(compressed_mean),
            "covariance_sha256": _array_sha256(compressed_covariance),
        }
    return result


def _case_inputs(
    regime_name: str,
    family: Family,
    profile: Profile,
) -> dict[str, Any]:
    """Build one frozen root case and its exact oracle state."""
    regime = c1._regime(regime_name)
    shapes, rate, design, observation, noise = c1._case_arrays(regime, family)
    total_order = 8 if profile == "smoke" else regime.total_order
    fraction_order = 6 if profile == "smoke" else regime.fraction_order
    masses, log_prior = c1._mass_grid(
        shapes=shapes,
        rate=rate,
        family=family,
        tiling="root",
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_values = c1._exact_log_likelihood(
        masses=masses,
        shapes=shapes,
        rate=rate,
        design=design,
        observation=observation,
        noise=noise,
        family=family,
        tiling="root",
        total_order=total_order,
        fraction_order=fraction_order,
    )

    def exact_likelihood(mass: float) -> float:
        values = c1._exact_log_likelihood(
            masses=np.asarray([[mass]], dtype=np.float64),
            shapes=shapes,
            rate=rate,
            design=design,
            observation=observation,
            noise=noise,
            family=family,
            tiling="root",
            total_order=total_order,
            fraction_order=fraction_order,
        )
        return float(values[0])

    return {
        "regime": regime,
        "shapes": shapes,
        "rate": rate,
        "design": design,
        "observation": observation,
        "noise": noise,
        "total_order": total_order,
        "fraction_order": fraction_order,
        "masses": masses,
        "log_prior": log_prior,
        "exact_values": exact_values,
        "exact_likelihood": exact_likelihood,
        "exact_summary": c1._posterior_summary(masses, log_prior, exact_values),
        "validation_mask": c1._development_validation_state_mask(
            masses,
            total_order=total_order,
            fraction_order=fraction_order,
        ),
    }


def _source_artifact(
    *,
    shapes: FloatArray,
    design: FloatArray,
    noise: FloatArray,
    spectrum: RootResidualSpectrum,
    sample_count: int,
    case_id: str,
) -> ConditionalAllocationMixture:
    """Build one equal-weight Sobol source in the exact spectrum basis."""
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        spectrum.basis,
    )
    return ConditionalAllocationMixture.from_aggregation(
        aggregation,
        np.zeros(shapes.size, dtype=np.int64),
        sample_count=sample_count,
        source_seed=DEVELOPMENT_SEED,
        source_provenance=(f"{PROTOCOL}:{case_id}:S={sample_count}:seed={DEVELOPMENT_SEED}"),
        construction_method=CONSTRUCTION_METHOD,
    )


def _expected_development_case_ids() -> tuple[str, ...]:
    """Return the exact ordered case catalogue required by a common lock."""
    return tuple("__".join(case) for case in DEVELOPMENT_MATRIX)


def _source_lock_sha256(payload: dict[str, Any]) -> str:
    """Return the authenticated digest of a source lock excluding itself."""
    unsigned = dict(payload)
    unsigned.pop("source_lock_sha256", None)
    return _sha256_json(unsigned)


def _load_source_lock(
    path: Path,
    *,
    source_revision: str,
    driver_sha256: str,
) -> dict[str, Any]:
    """Load and authenticate one all-six common source-lock certificate."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("source lock must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source lock must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("source lock must contain one JSON object")
    if raw != (_canonical_json(payload) + "\n").encode("ascii"):
        raise ValueError("source lock must use exact canonical JSON plus one newline")
    recorded_digest = payload.get("source_lock_sha256")
    if not isinstance(recorded_digest, str) or recorded_digest != _source_lock_sha256(payload):
        raise ValueError("source lock self-digest is absent or invalid")
    expected_scalars = {
        "schema": SOURCE_LOCK_SCHEMA,
        "protocol_sha256": _sha256_json(_protocol_payload()),
        "source_git_revision": source_revision,
        "source_driver_sha256": driver_sha256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "development_seed": DEVELOPMENT_SEED,
        "minimum_passing_suffix_length": DEVELOPMENT_MINIMUM_SOURCE_SUFFIX,
        "eligible": True,
        "structural_inference_licensed": False,
    }
    for name, expected in expected_scalars.items():
        if payload.get(name) != expected:
            raise ValueError(f"source lock field {name!r} does not match this protocol")
    if payload.get("source_sample_counts") != list(DEVELOPMENT_SOURCE_SAMPLE_COUNTS):
        raise ValueError("source lock sample-count ladder does not match this protocol")
    expected_case_ids = _expected_development_case_ids()
    if payload.get("matrix_case_ids") != list(expected_case_ids):
        raise ValueError("source lock does not certify the exact ordered six-case matrix")
    locked_sample_count = payload.get("locked_sample_count")
    if (
        isinstance(locked_sample_count, bool)
        or not isinstance(locked_sample_count, int)
        or locked_sample_count not in DEVELOPMENT_SOURCE_SAMPLE_COUNTS
    ):
        raise ValueError("source lock has no valid common locked sample count")
    certificates = payload.get("case_certificates")
    if not isinstance(certificates, dict) or set(certificates) != set(expected_case_ids):
        raise ValueError("source lock must contain exactly six case certificates")
    for case_id, certificate in certificates.items():
        if not isinstance(certificate, dict):
            raise ValueError(f"source lock certificate for {case_id!r} is malformed")
        if certificate.get("case_id") != case_id:
            raise ValueError(f"source lock certificate identity disagrees for {case_id!r}")
        if certificate.get("sample_count") != locked_sample_count:
            raise ValueError(f"source lock sample count disagrees for {case_id!r}")
        if certificate.get("scientific_pass") is not True:
            raise ValueError(f"source lock contains a failing certificate for {case_id!r}")
    return cast(dict[str, Any], payload)


def _source_certificate(
    *,
    case_id: str,
    input_sha256: str,
    spectrum: RootResidualSpectrum,
    source: ConditionalAllocationMixture,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact per-case identity fields consumed by the merger."""
    return {
        "case_id": case_id,
        "sample_count": source.sample_count,
        "input_sha256": input_sha256,
        "cell_alphas_sha256": spectrum.cell_alphas_sha256,
        "design_sha256": spectrum.design_sha256,
        "noise_sd_sha256": spectrum.noise_sd_sha256,
        "spectrum_basis_sha256": _array_sha256(spectrum.basis),
        "spectrum_eigenvalues_sha256": _array_sha256(spectrum.eigenvalues),
        "source_artifact_sha256": source.sha256,
        "source_operator_sha256": source.source_operator_sha256,
        "partition_sha256": source.partition_sha256,
        "scientific_pass": bool(evaluation["scientific_pass"]),
    }


def _validate_case_source_certificate(
    certificate: object,
    *,
    expected: dict[str, Any],
) -> None:
    """Fail if a rebuilt source differs from its common-lock certificate."""
    if not isinstance(certificate, dict):
        raise ValueError("case source certificate is malformed")
    if certificate != expected:
        differing = sorted(
            key for key in set(certificate) | set(expected) if certificate.get(key) != expected.get(key)
        )
        raise ValueError("rebuilt source does not match its common-lock certificate: " + ", ".join(differing))


def run_case(
    *,
    regime_name: str,
    family: Family,
    profile: Profile,
    stage: Stage,
    source_sample_counts: Sequence[int],
    component_counts: Sequence[int],
    source_lock_payload: dict[str, Any] | None = None,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run one stage of a root compressed-mixture scientific screen.

    Args:
        regime_name: Frozen C1 regime identifier.
        family: Two- or four-cell exact-oracle family.
        profile: Bounded smoke or source-pinned development profile.
        stage: Source-bank ladder or authenticated compression ladder.
        source_sample_counts: Increasing Sobol bank sizes.
        component_counts: Increasing compressed mixture sizes.
        source_lock_payload: Authenticated all-six lock for compression.
        include_timings: Whether to retain non-replayable wall-clock timings.

    Returns:
        Canonical-JSON-compatible case report.

    Raises:
        ValueError: If the case or protocol controls are invalid.
        RuntimeError: If inherited numerical identities have drifted.
    """
    _validate_protocol()
    if stage not in ("source", "compression"):
        raise ValueError("stage must be 'source' or 'compression'")
    if stage == "compression" and (profile != "development" or source_lock_payload is None):
        raise ValueError("compression requires a development common source lock")
    if stage == "source" and source_lock_payload is not None:
        raise ValueError("source stage cannot consume a source lock")
    case_key = (regime_name, family, "root")
    matrix = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if case_key not in matrix:
        raise ValueError(f"case {case_key!r} is not available in {profile}")
    counts = tuple(int(value) for value in source_sample_counts)
    components = tuple(int(value) for value in component_counts)
    if (
        not counts
        or tuple(sorted(counts)) != counts
        or len(set(counts)) != len(counts)
        or any(value < 1 or value & (value - 1) for value in counts)
    ):
        raise ValueError("source_sample_counts must be unique increasing powers of two")
    if (
        not components
        or tuple(sorted(components)) != components
        or len(set(components)) != len(components)
        or any(value < 1 for value in components)
    ):
        raise ValueError("component_counts must be unique increasing positive integers")
    if profile == "development" and (
        counts != DEVELOPMENT_SOURCE_SAMPLE_COUNTS or components != DEVELOPMENT_COMPONENT_COUNTS
    ):
        raise ValueError("development sample and component counts are source-pinned")
    effective_counts = (
        counts
        if stage == "source"
        else (int(cast(dict[str, Any], source_lock_payload)["locked_sample_count"]),)
    )

    case_id = f"{regime_name}__{family}__root"
    inputs = _case_inputs(regime_name, family, profile)
    shapes = cast(FloatArray, inputs["shapes"])
    design = cast(FloatArray, inputs["design"])
    observation = cast(FloatArray, inputs["observation"])
    noise = cast(FloatArray, inputs["noise"])
    masses = cast(FloatArray, inputs["masses"])
    log_prior = cast(FloatArray, inputs["log_prior"])
    exact_values = cast(FloatArray, inputs["exact_values"])
    validation_mask = cast(BoolArray, inputs["validation_mask"])
    exact_likelihood = cast(Likelihood, inputs["exact_likelihood"])
    exact_summary = cast(dict[str, Any], inputs["exact_summary"])
    identity_aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )
    spectrum = RootResidualSpectrum.from_aggregation(
        identity_aggregation,
        retained_variance_fraction=RETAINED_VARIANCE_FRACTION,
    )
    gradient_states = _gradient_catalogue(
        shapes=shapes,
        rate=float(inputs["rate"]),
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_values,
        exact_likelihood=exact_likelihood,
    )

    source_records: list[dict[str, Any]] = []
    source_artifacts: dict[int, ConditionalAllocationMixture] = {}
    source_evaluations: dict[int, dict[str, Any]] = {}
    input_sha256 = c1._case_input_sha256(
        inputs["regime"],
        family,
        "root",
        int(inputs["total_order"]),
        int(inputs["fraction_order"]),
    )
    for sample_count in effective_counts:
        started = time.perf_counter()
        source = _source_artifact(
            shapes=shapes,
            design=design,
            noise=noise,
            spectrum=spectrum,
            sample_count=sample_count,
            case_id=case_id,
        )
        build_seconds = time.perf_counter() - started if include_timings else None

        def source_likelihood(
            mass: float,
            artifact: ConditionalAllocationMixture = source,
        ) -> float:
            return artifact.log_likelihood(
                observation,
                np.asarray([mass], dtype=np.float64),
            )

        evaluation = _scientific_evaluation(
            likelihood=source_likelihood,
            likelihood_name="source",
            masses=masses,
            observation=observation,
            log_prior=log_prior,
            exact_log_likelihood=exact_values,
            exact_summary=exact_summary,
            gradient_states=gradient_states,
            validation_mask=validation_mask,
            include_timings=include_timings,
        )
        source_artifacts[sample_count] = source
        source_evaluations[sample_count] = evaluation
        certificate = _source_certificate(
            case_id=case_id,
            input_sha256=input_sha256,
            spectrum=spectrum,
            source=source,
            evaluation=evaluation,
        )
        source_records.append(
            {
                "sample_count": sample_count,
                "source_seed": source.source_seed,
                "artifact_sha256": source.sha256,
                "source_operator_sha256": source.source_operator_sha256,
                "partition_sha256": source.partition_sha256,
                "construction_method": source.construction_method,
                "construction_scipy_version": source.construction_scipy_version,
                "storage_nbytes": source.storage_nbytes,
                "build_seconds": build_seconds,
                "moment_diagnostics": _moment_diagnostics(source, spectrum),
                "exact_vs_source": evaluation,
                "merger_certificate": certificate,
            }
        )
    common_report = {
        "case_id": case_id,
        "profile": profile,
        "stage": stage,
        "input_sha256": input_sha256,
        "source_identities": {
            "a1_source_revision": c1.A1_SOURCE_REVISION,
            "a1_numerical_source_sha256": c1.A1_NUMERICAL_SOURCE_SHA256,
            "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
            "cell_alphas_sha256": spectrum.cell_alphas_sha256,
            "design_sha256": spectrum.design_sha256,
            "noise_sd_sha256": spectrum.noise_sd_sha256,
            "mass_grid_sha256": _array_sha256(masses),
            "exact_log_likelihood_sha256": _array_sha256(exact_values),
            "validation_mask_sha256": _array_sha256(validation_mask.astype(np.uint8)),
        },
        "spectrum": {
            "selection": (
                "complete analytic design/noise/prior spectrum; observed residual unavailable to selection"
            ),
            "observation_count": int(observation.size),
            "retained_rank": spectrum.retained_rank,
            "requested_retained_variance_fraction": (spectrum.requested_retained_variance_fraction),
            "retained_variance_fraction": spectrum.retained_variance_fraction,
            "total_variance": spectrum.total_variance,
            "discarded_variance": spectrum.discarded_variance,
            "eigenvalues": spectrum.eigenvalues.tolist(),
            "basis_sha256": _array_sha256(spectrum.basis),
            "eigenvalues_sha256": _array_sha256(spectrum.eigenvalues),
            "projection_kl_upper_bound_per_squared_mass": (
                spectrum.projection_kl_upper_bound_per_squared_mass
            ),
        },
        "quadrature": {
            "total_order": int(inputs["total_order"]),
            "fraction_order": int(inputs["fraction_order"]),
            "mass_state_count": int(masses.shape[0]),
            "pointwise_scoring": {
                "scheme": "c1-checkerboard-by-total-and-share-index-v1",
                "validation_state_count": int(np.count_nonzero(validation_mask)),
                "alters_evidence_or_posterior_quadrature": False,
            },
        },
        "exact_posterior_summary": exact_summary,
        "gradient_state_catalogue": gradient_states,
        "confirmation": {
            "status": "deferred_to_later_protocol_stage",
            "seeds": list(CONFIRMATION_SEEDS),
            "may_retune_source_or_compression_lock": False,
        },
        "observed_residual_used_for_basis_selection": False,
        "protected_catalogue_accessed": PROTECTED_CATALOGUE_ACCESSED,
        "production_output_written": PRODUCTION_OUTPUT_WRITTEN,
        "structural_inference_licensed": False,
    }
    source_minimum_suffix = (
        SMOKE_MINIMUM_SOURCE_SUFFIX if profile == "smoke" else DEVELOPMENT_MINIMUM_SOURCE_SUFFIX
    )
    case_suffix_start = _stable_lock(
        effective_counts,
        [bool(record["exact_vs_source"]["scientific_pass"]) for record in source_records],
        minimum_suffix_length=(source_minimum_suffix if stage == "source" else 1),
    )
    if stage == "source":
        return {
            **common_report,
            "source_bank": {
                "sample_counts": list(counts),
                "development_seed": DEVELOPMENT_SEED,
                "minimum_common_passing_suffix_length": source_minimum_suffix,
                "evaluations": source_records,
                "case_passing_suffix_start": case_suffix_start,
                "case_passes_suffix_requirement": case_suffix_start is not None,
                "common_source_lock_issued": False,
                "common_source_lock_requires_separate_all_six_merger": True,
                "merger_rule": (
                    "smallest common predeclared S starting an all-larger "
                    "six-case scientific-pass suffix of the required length"
                ),
            },
            "compression_evaluated": False,
            "scientific_pass": case_suffix_start is not None,
        }

    assert source_lock_payload is not None
    locked_source_count = int(source_lock_payload["locked_sample_count"])
    locked_source = source_artifacts[locked_source_count]
    locked_source_evaluation = source_evaluations[locked_source_count]
    _validate_case_source_certificate(
        source_lock_payload["case_certificates"][case_id],
        expected=source_records[0]["merger_certificate"],
    )
    if not bool(locked_source_evaluation["scientific_pass"]):
        raise RuntimeError("authenticated locked source failed replayed scientific gates")

    compression_records: list[dict[str, Any]] = []

    def locked_source_likelihood(mass: float) -> float:
        return locked_source.log_likelihood(
            observation,
            np.asarray([mass], dtype=np.float64),
        )

    for component_count in components:
        started = time.perf_counter()
        compressed = CompressedRootMixture.from_source(
            locked_source,
            spectrum,
            mixture_rank=spectrum.retained_rank,
            component_count=component_count,
            restart_count=CLUSTER_RESTART_COUNT,
            random_seed=DEVELOPMENT_SEED,
            maximum_iterations=CLUSTER_MAXIMUM_ITERATIONS,
        )
        build_seconds = time.perf_counter() - started if include_timings else None

        def compressed_likelihood(
            mass: float,
            artifact: CompressedRootMixture = compressed,
        ) -> float:
            return artifact.log_likelihood(observation, mass)

        evaluation = _scientific_evaluation(
            likelihood=compressed_likelihood,
            likelihood_name="compressed",
            masses=masses,
            observation=observation,
            log_prior=log_prior,
            exact_log_likelihood=exact_values,
            exact_summary=exact_summary,
            gradient_states=gradient_states,
            validation_mask=validation_mask,
            include_timings=include_timings,
        )
        compression_records.append(
            {
                "component_count": component_count,
                "mixture_rank": compressed.mixture_rank,
                "source_sample_count": compressed.source_sample_count,
                "source_sha256": compressed.source_sha256,
                "selected_restart": compressed.selected_restart,
                "restart_inertias": [
                    float(value) if math.isfinite(float(value)) else None
                    for value in compressed.restart_inertias
                ],
                "cluster_counts": compressed.cluster_counts.tolist(),
                "kl_upper_bound": compressed.kl_upper_bound,
                "storage_nbytes": compressed.storage_nbytes,
                "storage_fraction_of_source": (compressed.storage_nbytes / locked_source.storage_nbytes),
                "build_seconds": build_seconds,
                "moment_diagnostics": _moment_diagnostics(
                    locked_source,
                    spectrum,
                    compressed,
                ),
                "exact_vs_compressed": evaluation,
                "compression_incremental": _incremental_evaluation(
                    source_likelihood=locked_source_likelihood,
                    compressed_likelihood=compressed_likelihood,
                    masses=masses,
                    log_prior=log_prior,
                    validation_mask=validation_mask,
                    source_evaluation=locked_source_evaluation,
                    compressed_evaluation=evaluation,
                ),
            }
        )
    compression_minimum_suffix = DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX
    locked_component_count = (
        _stable_lock(
            [int(record["component_count"]) for record in compression_records],
            [bool(record["exact_vs_compressed"]["scientific_pass"]) for record in compression_records],
            minimum_suffix_length=compression_minimum_suffix,
        )
        if compression_records
        else None
    )
    compression_lock_eligible = locked_component_count is not None
    return {
        **common_report,
        "authenticated_common_source_lock": {
            "schema": SOURCE_LOCK_SCHEMA,
            "source_lock_sha256": source_lock_payload["source_lock_sha256"],
            "locked_sample_count": locked_source_count,
            "rebuild_certificate_matched": True,
            "all_six_cases_certified": True,
            "private_per_case_source_selection_used": False,
        },
        "locked_source_replay": source_records[0],
        "compression": {
            "component_counts": list(components),
            "minimum_passing_suffix_length": compression_minimum_suffix,
            "evaluations": compression_records,
            "locked_component_count": locked_component_count,
            "lock_eligible": compression_lock_eligible,
            "lock_rule": (
                "smallest predeclared M starting an all-larger exact-"
                "scientific-pass suffix of the required length"
            ),
            "evaluated_only_after_authenticated_common_source_lock": True,
        },
        "scientific_pass": compression_lock_eligible,
    }


def matrix_catalogue() -> dict[str, Any]:
    """Return the executable smoke and development case catalogue."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stages": ["source", "compression"],
        "stage_profiles": {
            "source": ["smoke", "development"],
            "compression": ["development"],
        },
        "source_lock_schema": SOURCE_LOCK_SCHEMA,
        "smoke": [list(case) for case in SMOKE_MATRIX],
        "development": [list(case) for case in DEVELOPMENT_MATRIX],
        "protected_catalogue": {
            "present": False,
            "accessible": False,
        },
        "structural_inference_licensed": False,
    }


def run_screen(
    *,
    profile: Profile,
    stage: Stage,
    case_id: str,
    source_revision: str | None = None,
    source_lock_path: Path | None = None,
    source_sample_counts: Sequence[int] | None = None,
    component_counts: Sequence[int] | None = None,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run one atomic smoke or development case.

    Args:
        profile: Bounded smoke or frozen development protocol.
        stage: Source-bank ladder or authenticated compression ladder.
        case_id: Exact ``REGIME__FAMILY__root`` case identifier.
        source_revision: Optional expected Git revision.
        source_lock_path: Canonical all-six lock required by compression.
        source_sample_counts: Optional smoke-only source-size override.
        component_counts: Optional smoke-only component-count override.
        include_timings: Whether to include non-replayable timings.

    Returns:
        Canonical-JSON-compatible one-case report.

    Raises:
        ValueError: If the profile, case, or overrides are invalid.
        RuntimeError: If source identities disagree.
    """
    _validate_protocol()
    if stage == "compression" and source_lock_path is None:
        raise ValueError("compression stage requires source_lock_path")
    if stage == "source" and source_lock_path is not None:
        raise ValueError("source stage cannot consume source_lock_path")
    if stage == "compression" and profile != "development":
        raise ValueError("compression stage is available only for development")
    if stage == "source" and component_counts is not None:
        raise ValueError("component_counts are not inputs to the source stage")
    if stage == "compression" and source_sample_counts is not None:
        raise ValueError("compression must obtain its source size from the common lock")
    if profile == "development" and (source_sample_counts is not None or component_counts is not None):
        raise ValueError("development source and component counts cannot be overridden")
    matrix = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    matches = [case for case in matrix if "__".join(case) == case_id]
    if len(matches) != 1:
        raise ValueError(f"case_id {case_id!r} is not available in {profile}")
    regime_name, family, _ = matches[0]
    selected_source_counts = (
        SMOKE_SOURCE_SAMPLE_COUNTS
        if profile == "smoke" and source_sample_counts is None
        else DEVELOPMENT_SOURCE_SAMPLE_COUNTS
        if profile == "development"
        else tuple(cast(Sequence[int], source_sample_counts))
    )
    selected_component_counts = (
        SMOKE_COMPONENT_COUNTS
        if profile == "smoke" and component_counts is None
        else DEVELOPMENT_COMPONENT_COUNTS
        if profile == "development"
        else tuple(cast(Sequence[int], component_counts))
    )
    resolved_revision = _source_revision(source_revision)
    driver_sha256 = _driver_sha256()
    source_lock_payload = (
        _load_source_lock(
            cast(Path, source_lock_path),
            source_revision=resolved_revision,
            driver_sha256=driver_sha256,
        )
        if stage == "compression"
        else None
    )
    case = run_case(
        regime_name=regime_name,
        family=cast(Family, family),
        profile=profile,
        stage=stage,
        source_sample_counts=selected_source_counts,
        component_counts=selected_component_counts,
        source_lock_payload=source_lock_payload,
        include_timings=include_timings,
    )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "profile": profile,
        "stage": stage,
        "source_git_revision": resolved_revision,
        "driver_sha256": driver_sha256,
        "protocol_sha256": _sha256_json(_protocol_payload()),
        "protocol_payload": _protocol_payload(),
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_lock_sha256": (
            None if source_lock_payload is None else source_lock_payload["source_lock_sha256"]
        ),
        "thresholds": c1.THRESHOLDS,
        "observed_residual_used_for_basis_selection": False,
        "protected_catalogue_accessed": PROTECTED_CATALOGUE_ACCESSED,
        "production_output_written": PRODUCTION_OUTPUT_WRITTEN,
        "structural_inference_licensed": False,
        "confirmation_status": "deferred_to_later_protocol_stage",
        "runtime": {
            "numpy_version": np.__version__,
            "scipy_version": scipy_version,
        },
        "case": case,
        "scientific_pass": bool(case["scientific_pass"]),
    }


def _positive_csv(
    value: str,
    *,
    name: str,
    upper_bound: int,
) -> tuple[int, ...]:
    """Parse a unique positive integer CSV argument."""
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from error
    if (
        not parsed
        or any(item < 1 or item > upper_bound for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError(f"{name} must contain unique integers in [1, {upper_bound}]")
    return parsed


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish canonical JSON once without partial or overwritten output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    """Build the fail-closed command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        default="smoke",
    )
    parser.add_argument(
        "--stage",
        choices=("source", "compression"),
        default="source",
    )
    parser.add_argument(
        "--case-id",
        required=False,
        help="Required run case as REGIME__FAMILY__root.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-lock",
        type=Path,
        help="Authenticated common source-lock JSON required by compression.",
    )
    parser.add_argument(
        "--source-revision",
        help="Expected full lower-case Git SHA; required if Git is unavailable.",
    )
    parser.add_argument(
        "--sample-counts",
        type=lambda value: _positive_csv(
            value,
            name="sample-counts",
            upper_bound=2**24,
        ),
    )
    parser.add_argument(
        "--component-counts",
        type=lambda value: _positive_csv(
            value,
            name="component-counts",
            upper_bound=2**16,
        ),
    )
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--no-timings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI controls, run one case, and publish it atomically."""
    args = _parser().parse_args(argv)
    if args.list_matrix:
        if (
            args.output is not None
            or args.case_id is not None
            or args.source_revision is not None
            or args.source_lock is not None
            or args.sample_counts is not None
            or args.component_counts is not None
        ):
            raise SystemExit("--list-matrix cannot be combined with run options")
        print(_canonical_json(matrix_catalogue()))
        return 0
    if args.output is None or args.case_id is None:
        raise SystemExit("--output and --case-id are required unless --list-matrix is used")
    if args.profile == "development" and (
        args.sample_counts is not None or args.component_counts is not None
    ):
        raise SystemExit("development source and component counts are source-pinned")
    if args.stage == "compression" and args.source_lock is None:
        raise SystemExit("--source-lock is required for compression")
    if args.stage == "source" and args.source_lock is not None:
        raise SystemExit("--source-lock is only valid for compression")
    if args.stage == "compression" and args.profile != "development":
        raise SystemExit("compression is available only under the development profile")
    if args.stage == "source" and args.component_counts is not None:
        raise SystemExit("--component-counts is not an input to the source stage")
    if args.stage == "compression" and args.sample_counts is not None:
        raise SystemExit("--sample-counts is not an input to compression")
    report = run_screen(
        profile=args.profile,
        stage=args.stage,
        case_id=args.case_id,
        source_revision=args.source_revision,
        source_lock_path=args.source_lock,
        source_sample_counts=args.sample_counts,
        component_counts=args.component_counts,
        include_timings=not args.no_timings,
    )
    _write_atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
