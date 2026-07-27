#!/usr/bin/env python3
"""Confirm a frozen exact-mixture compression on independent source scrambles.

The development run selected one direct-bank size and one compression size.
This driver authenticates those immutable development decisions, rebuilds the
same six exact-oracle cases at one untouched Sobol scramble, and evaluates the
source and compressed likelihoods without retuning.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Sequence, cast

import numpy as np
from scipy import __version__ as scipy_version

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_certify as development_certify,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_tiny_screen as development,
)
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

SCHEMA = "rjmcmc-conditional-residual-image-compressed-mixture-confirmation-v1"
PROTOCOL = "root-exact-spectrum-sobol-moment-compression-confirmation-v1"

DEVELOPMENT_REVISION = "d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea"
DEVELOPMENT_SOURCE_DECISION_RAW_SHA256 = "61fa3ff7bd2aee439b532a8c70633df05cc3cc452c6a44aff377ac4ad613fa9a"
DEVELOPMENT_SOURCE_LOCK_SHA256 = "80d631763473f7a9262ef0314b4efcab73e44a1d85697bc9aee243a4e3f41ead"
DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256 = (
    "932efada6a7ad1894696c40fe78c515f2da661cfa780ddb7c659304c370bdf5e"
)
LOCKED_SOURCE_SAMPLE_COUNT = 65_536
LOCKED_COMPONENT_COUNT = 256
SOURCE_SEEDS = (1_877, 4_099, 8_317)
CLUSTER_SEED = development.DEVELOPMENT_SEED
MOMENT_MEAN_ABSOLUTE_TOLERANCE = 5.0e-13
MOMENT_COVARIANCE_ABSOLUTE_TOLERANCE = 5.0e-12

PROTECTED_CATALOGUE_ACCESSED = False
PRODUCTION_OUTPUT_WRITTEN = False


def _driver_sha256() -> str:
    """Return the exact digest of this executable source."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _protocol_payload() -> dict[str, Any]:
    """Return every frozen confirmation setting."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "development_revision": DEVELOPMENT_REVISION,
        "development_source_decision_raw_sha256": (DEVELOPMENT_SOURCE_DECISION_RAW_SHA256),
        "development_source_lock_sha256": DEVELOPMENT_SOURCE_LOCK_SHA256,
        "development_compression_decision_raw_sha256": (DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256),
        "development_protocol_sha256": development._sha256_json(development._protocol_payload()),
        "development_driver_sha256": development._driver_sha256(),
        "locked_source_sample_count": LOCKED_SOURCE_SAMPLE_COUNT,
        "locked_component_count": LOCKED_COMPONENT_COUNT,
        "source_seeds": SOURCE_SEEDS,
        "cluster_seed": CLUSTER_SEED,
        "cluster_restart_count": development.CLUSTER_RESTART_COUNT,
        "cluster_maximum_iterations": development.CLUSTER_MAXIMUM_ITERATIONS,
        "retained_variance_fraction": development.RETAINED_VARIANCE_FRACTION,
        "moment_mean_absolute_tolerance": MOMENT_MEAN_ABSOLUTE_TOLERANCE,
        "moment_covariance_absolute_tolerance": (MOMENT_COVARIANCE_ABSOLUTE_TOLERANCE),
        "thresholds": development.c1.THRESHOLDS,
        "selection": "all-six-cases-times-all-three-source-seeds-must-pass",
        "retuning_permitted": False,
        "protected_catalogue_present": False,
    }


def _load_development_decisions(
    source_decision_path: Path,
    compression_decision_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the exact development lock and compression decision."""
    source_decision, source_raw_sha256 = development_certify._read_canonical_json(source_decision_path)
    if source_raw_sha256 != DEVELOPMENT_SOURCE_DECISION_RAW_SHA256:
        raise ValueError("development source decision raw SHA-256 is not frozen")
    source_lock = development._load_source_lock(
        source_decision_path,
        source_revision=DEVELOPMENT_REVISION,
        driver_sha256=development._driver_sha256(),
    )
    if (
        source_lock["source_lock_sha256"] != DEVELOPMENT_SOURCE_LOCK_SHA256
        or source_lock["locked_sample_count"] != LOCKED_SOURCE_SAMPLE_COUNT
    ):
        raise ValueError("development source lock does not match the frozen selection")

    compression_decision, compression_raw_sha256 = development_certify._read_canonical_json(
        compression_decision_path
    )
    if compression_raw_sha256 != DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256:
        raise ValueError("development compression decision raw SHA-256 is not frozen")
    expected = {
        "schema": development_certify.COMPRESSION_DECISION_SCHEMA,
        "protocol": development.PROTOCOL,
        "protocol_sha256": development._sha256_json(development._protocol_payload()),
        "source_git_revision": DEVELOPMENT_REVISION,
        "source_driver_sha256": development._driver_sha256(),
        "source_lock_sha256": DEVELOPMENT_SOURCE_LOCK_SHA256,
        "locked_source_sample_count": LOCKED_SOURCE_SAMPLE_COUNT,
        "locked_component_count": LOCKED_COMPONENT_COUNT,
        "eligible": True,
        "confirmation_status": "deferred_to_later_protocol_stage",
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
    }
    for name, value in expected.items():
        if compression_decision.get(name) != value:
            raise ValueError(f"development compression field {name!r} is not frozen")
    if compression_decision.get("component_counts") != list(development.DEVELOPMENT_COMPONENT_COUNTS):
        raise ValueError("development component ladder is not frozen")
    return source_lock, compression_decision


def _case_identity_matches_lock(
    *,
    case_id: str,
    family: development.Family,
    inputs: dict[str, Any],
    spectrum: RootResidualSpectrum,
    source_lock: dict[str, Any],
) -> None:
    """Fail if a confirmation case differs from its development definition."""
    certificate = source_lock["case_certificates"][case_id]
    observed = {
        "input_sha256": development.c1._case_input_sha256(
            inputs["regime"],
            family,
            "root",
            int(inputs["total_order"]),
            int(inputs["fraction_order"]),
        ),
        "cell_alphas_sha256": spectrum.cell_alphas_sha256,
        "design_sha256": spectrum.design_sha256,
        "noise_sd_sha256": spectrum.noise_sd_sha256,
        "spectrum_basis_sha256": development._array_sha256(spectrum.basis),
        "spectrum_eigenvalues_sha256": development._array_sha256(spectrum.eigenvalues),
    }
    differing = sorted(name for name, value in observed.items() if certificate.get(name) != value)
    if differing:
        raise ValueError("confirmation case differs from development lock: " + ", ".join(differing))


def validate_development_inputs(
    *,
    source_decision_path: Path,
    compression_decision_path: Path,
) -> dict[str, Any]:
    """Return a compact certificate after authenticating development inputs."""
    source_lock, compression_decision = _load_development_decisions(
        source_decision_path,
        compression_decision_path,
    )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "protocol_sha256": development._sha256_json(_protocol_payload()),
        "development_revision": DEVELOPMENT_REVISION,
        "development_source_lock_sha256": source_lock["source_lock_sha256"],
        "development_locked_source_sample_count": source_lock["locked_sample_count"],
        "development_locked_component_count": compression_decision["locked_component_count"],
        "matrix_case_ids": list(development._expected_development_case_ids()),
        "source_seeds": list(SOURCE_SEEDS),
        "eligible": True,
        "structural_inference_licensed": False,
    }


def run_confirmation(
    *,
    case_id: str,
    source_seed: int,
    source_decision_path: Path,
    compression_decision_path: Path,
    source_revision: str | None = None,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run one frozen case and independent source-scramble confirmation."""
    if source_seed not in SOURCE_SEEDS:
        raise ValueError("source_seed is not in the frozen confirmation catalogue")
    case_ids = development._expected_development_case_ids()
    if case_id not in case_ids:
        raise ValueError("case_id is not in the frozen development matrix")
    source_lock, compression_decision = _load_development_decisions(
        source_decision_path,
        compression_decision_path,
    )
    regime_name, family, _ = case_id.split("__")
    inputs = development._case_inputs(
        regime_name,
        cast(development.Family, family),
        "development",
    )
    shapes = cast(development.FloatArray, inputs["shapes"])
    design = cast(development.FloatArray, inputs["design"])
    observation = cast(development.FloatArray, inputs["observation"])
    noise = cast(development.FloatArray, inputs["noise"])
    masses = cast(development.FloatArray, inputs["masses"])
    log_prior = cast(development.FloatArray, inputs["log_prior"])
    exact_values = cast(development.FloatArray, inputs["exact_values"])
    validation_mask = cast(development.BoolArray, inputs["validation_mask"])
    exact_likelihood = cast(development.Likelihood, inputs["exact_likelihood"])
    exact_summary = cast(dict[str, Any], inputs["exact_summary"])

    identity_aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )
    spectrum = RootResidualSpectrum.from_aggregation(
        identity_aggregation,
        retained_variance_fraction=development.RETAINED_VARIANCE_FRACTION,
    )
    _case_identity_matches_lock(
        case_id=case_id,
        family=cast(development.Family, family),
        inputs=inputs,
        spectrum=spectrum,
        source_lock=source_lock,
    )
    gradient_states = development._gradient_catalogue(
        shapes=shapes,
        rate=float(inputs["rate"]),
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_values,
        exact_likelihood=exact_likelihood,
    )

    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        spectrum.basis,
    )
    source_started = time.perf_counter()
    source = ConditionalAllocationMixture.from_aggregation(
        aggregation,
        np.zeros(shapes.size, dtype=np.int64),
        sample_count=LOCKED_SOURCE_SAMPLE_COUNT,
        source_seed=source_seed,
        source_provenance=(f"{PROTOCOL}:{case_id}:S={LOCKED_SOURCE_SAMPLE_COUNT}:seed={source_seed}"),
        construction_method=development.CONSTRUCTION_METHOD,
    )
    source_build_seconds = time.perf_counter() - source_started if include_timings else None

    def source_likelihood(mass: float) -> float:
        return source.log_likelihood(
            observation,
            np.asarray([mass], dtype=np.float64),
        )

    source_evaluation = development._scientific_evaluation(
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

    compression_started = time.perf_counter()
    compressed = CompressedRootMixture.from_source(
        source,
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=LOCKED_COMPONENT_COUNT,
        restart_count=development.CLUSTER_RESTART_COUNT,
        random_seed=CLUSTER_SEED,
        maximum_iterations=development.CLUSTER_MAXIMUM_ITERATIONS,
    )
    compression_build_seconds = time.perf_counter() - compression_started if include_timings else None

    def compressed_likelihood(mass: float) -> float:
        return compressed.log_likelihood(observation, mass)

    compressed_evaluation = development._scientific_evaluation(
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
    moment_diagnostics = development._moment_diagnostics(
        source,
        spectrum,
        compressed,
    )
    compression_moments = moment_diagnostics["compression"]
    confirmation_checks = {
        "source_scientific_pass": bool(source_evaluation["scientific_pass"]),
        "compressed_scientific_pass": bool(compressed_evaluation["scientific_pass"]),
        "compression_mean_closure": bool(
            compression_moments["mean_maximum_absolute_difference_from_source"]
            <= MOMENT_MEAN_ABSOLUTE_TOLERANCE
        ),
        "compression_covariance_closure": bool(
            compression_moments["covariance_maximum_absolute_difference_from_source"]
            <= MOMENT_COVARIANCE_ABSOLUTE_TOLERANCE
        ),
        "compression_kl_bound_finite": bool(
            math.isfinite(compressed.kl_upper_bound) and compressed.kl_upper_bound >= 0.0
        ),
    }
    resolved_revision = development._source_revision(source_revision)
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "protocol_sha256": development._sha256_json(_protocol_payload()),
        "protocol_payload": _protocol_payload(),
        "source_git_revision": resolved_revision,
        "driver_sha256": _driver_sha256(),
        "development": {
            "revision": DEVELOPMENT_REVISION,
            "source_decision_raw_sha256": (DEVELOPMENT_SOURCE_DECISION_RAW_SHA256),
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "compression_decision_raw_sha256": (DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256),
            "locked_source_sample_count": source_lock["locked_sample_count"],
            "locked_component_count": compression_decision["locked_component_count"],
        },
        "case_id": case_id,
        "source_seed": source_seed,
        "cluster_seed": CLUSTER_SEED,
        "input_sha256": source_lock["case_certificates"][case_id]["input_sha256"],
        "retained_rank": spectrum.retained_rank,
        "source": {
            "artifact_sha256": source.sha256,
            "build_seconds": source_build_seconds,
            "storage_nbytes": source.storage_nbytes,
            "moment_diagnostics": {
                name: value for name, value in moment_diagnostics.items() if name != "compression"
            },
            "exact_evaluation": source_evaluation,
        },
        "compression": {
            "component_count": compressed.component_count,
            "build_seconds": compression_build_seconds,
            "storage_nbytes": compressed.storage_nbytes,
            "storage_fraction_of_source": (compressed.storage_nbytes / source.storage_nbytes),
            "selected_restart": compressed.selected_restart,
            "restart_inertias": compressed.restart_inertias.tolist(),
            "kl_upper_bound": compressed.kl_upper_bound,
            "moment_diagnostics": compression_moments,
            "exact_evaluation": compressed_evaluation,
            "incremental_evaluation": development._incremental_evaluation(
                source_likelihood=source_likelihood,
                compressed_likelihood=compressed_likelihood,
                masses=masses,
                log_prior=log_prior,
                validation_mask=validation_mask,
                source_evaluation=source_evaluation,
                compressed_evaluation=compressed_evaluation,
            ),
        },
        "confirmation_checks": confirmation_checks,
        "observed_residual_used_for_basis_selection": False,
        "retuning_performed": False,
        "protected_catalogue_accessed": PROTECTED_CATALOGUE_ACCESSED,
        "production_output_written": PRODUCTION_OUTPUT_WRITTEN,
        "structural_inference_licensed": False,
        "runtime": {
            "numpy_version": np.__version__,
            "scipy_version": scipy_version,
        },
        "scientific_pass": all(confirmation_checks.values()),
    }


def _source_seed(value: str) -> int:
    """Parse one frozen confirmation seed."""
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source seed must be an integer") from error
    if seed not in SOURCE_SEEDS:
        raise argparse.ArgumentTypeError(f"source seed must be one of {SOURCE_SEEDS}")
    return seed


def _parser() -> argparse.ArgumentParser:
    """Build the validation and confirmation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-development")
    run = subparsers.add_parser("run")
    for subparser in (validate, run):
        subparser.add_argument(
            "--development-source-decision",
            type=Path,
            required=True,
        )
        subparser.add_argument(
            "--development-compression-decision",
            type=Path,
            required=True,
        )
    run.add_argument("--case-id", required=True)
    run.add_argument("--source-seed", type=_source_seed, required=True)
    run.add_argument("--source-revision")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--no-timings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate development inputs or publish one confirmation artifact."""
    args = _parser().parse_args(argv)
    if args.command == "validate-development":
        payload = validate_development_inputs(
            source_decision_path=args.development_source_decision,
            compression_decision_path=args.development_compression_decision,
        )
        print(development._canonical_json(payload))
        return 0
    report = run_confirmation(
        case_id=args.case_id,
        source_seed=args.source_seed,
        source_decision_path=args.development_source_decision,
        compression_decision_path=args.development_compression_decision,
        source_revision=args.source_revision,
        include_timings=not args.no_timings,
    )
    development._write_atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
