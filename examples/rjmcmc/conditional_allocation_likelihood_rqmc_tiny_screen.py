#!/usr/bin/env python3
"""Run the checksum-pinned C1-RQMC conditional-allocation tiny screen.

This is a separate development protocol from
``conditional_allocation_likelihood_tiny_screen``.  It keeps that driver's
exact quadrature definitions, nine-case matrix, thresholds, bank-size ladder,
seed allocation, and blindness rules, but constructs each frozen allocation
bank with an independently scrambled balanced-Dirichlet-tree Sobol rule.

The driver has no held-out operator/partition mode.  It cannot use an
observed residual, approximate evidence, or protected result to select a
basis, partition, dimension, bank size, or scramble.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Literal, Sequence, cast

import numpy as np
from scipy import __version__ as scipy_version

# Direct Slurm invocation uses this file path rather than ``python -m``.  Add
# only the resolved repository root in that execution mode so the sibling C1
# module is found under the same import name used by tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

SCHEMA = "rjmcmc-conditional-allocation-c1-rqmc-tiny-screen-v1"
PROTOCOL = "conditional-allocation-c1-full-rank-scrambled-sobol-balanced-dirichlet-bank-v1"
BANK_CONSTRUCTION_METHOD = "scrambled_sobol_balanced_dirichlet"
DEVELOPMENT_SCIPY_VERSION = "1.15.2"

# Importing the original definitions, rather than copying them, makes numeric
# drift between the PCG64 and RQMC comparisons fail visibly at the shared A1
# checksum pin.
A1_SOURCE_REVISION = c1.A1_SOURCE_REVISION
A1_NUMERICAL_SOURCE_SHA256 = c1.A1_NUMERICAL_SOURCE_SHA256
A1_DEFINITIONS_SHA256 = c1.A1_DEFINITIONS_SHA256
THRESHOLDS = c1.THRESHOLDS
MERGER_THRESHOLDS = c1.MERGER_THRESHOLDS
DEVELOPMENT_MATRIX = c1.DEVELOPMENT_MATRIX
SMOKE_MATRIX = c1.SMOKE_MATRIX
DEVELOPMENT_SAMPLE_COUNTS = c1.DEVELOPMENT_SAMPLE_COUNTS
DEVELOPMENT_REPEAT_SEEDS = c1.DEVELOPMENT_REPEAT_SEEDS
DEVELOPMENT_SELECTION_SEED = c1.DEVELOPMENT_SELECTION_SEED
CONFIRMATION_SEEDS = c1.CONFIRMATION_SEEDS
SMOKE_SAMPLE_COUNTS = c1.SMOKE_SAMPLE_COUNTS
SMOKE_REPEAT_SEEDS = c1.SMOKE_REPEAT_SEEDS
HELD_OUT_CATALOGUE_ID = c1.HELD_OUT_CATALOGUE_ID
HELD_OUT_CATALOGUE_SHA256 = c1.HELD_OUT_CATALOGUE_SHA256
DEVELOPMENT_PROTOCOL_SHA256 = "dcb2ef2bebb0c7eefafbd49a225c864e1b8a7478c568c168ed1640dd91ea9f4b"

Profile = Literal["smoke", "development"]

_ARTIFACT_PROVENANCE_FIELDS = (
    "construction_scipy_version",
    "quasi_random_engine",
    "sobol_catalogue_sha256",
    "sobol_block_dimensions",
    "sobol_bits",
    "sobol_scramble",
    "sobol_optimization",
    "inverse_transform",
    "dimension_order",
    "sobol_block_rule",
    "sobol_seed_derivation",
)


def _bank_method(artifact: ConditionalAllocationMixture) -> str:
    """Return and validate the frozen artifact construction method."""
    observed = artifact.construction_method
    if observed != BANK_CONSTRUCTION_METHOD:
        raise RuntimeError(f"allocation bank used an unexpected construction method: {observed!r}")
    return observed


def _artifact_construction_provenance(
    artifact: ConditionalAllocationMixture,
) -> dict[str, object]:
    """Return the exact compact Sobol-construction subset from the artifact."""
    payload = artifact.payload
    if payload.get("schema") != "aggregation-conditional-allocation-mixture-v2":
        raise RuntimeError("RQMC bank did not use artifact schema v2")
    if payload.get("construction_method") != BANK_CONSTRUCTION_METHOD:
        raise RuntimeError("RQMC artifact payload construction method disagrees")
    try:
        provenance = {field: payload[field] for field in _ARTIFACT_PROVENANCE_FIELDS}
    except KeyError as error:  # pragma: no cover - core schema tests own this
        raise RuntimeError(f"RQMC artifact omitted construction provenance {error.args[0]!r}") from error
    if not isinstance(provenance["construction_scipy_version"], str):
        raise RuntimeError("RQMC artifact omitted its SciPy construction version")
    return provenance


def _protocol_sha256(
    *,
    sample_counts: Sequence[int],
    repeat_seeds: Sequence[int],
    matrix: Sequence[tuple[str, str, str]],
) -> str:
    """Return one invocation identity under the frozen RQMC protocol."""
    return c1._sha256_json(
        {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "bank_construction_method": BANK_CONSTRUCTION_METHOD,
            "required_development_construction_scipy_version": (DEVELOPMENT_SCIPY_VERSION),
            "thresholds": THRESHOLDS,
            "merger_thresholds": MERGER_THRESHOLDS,
            "gradient_step": c1.GRADIENT_STEP,
            "sample_counts": list(sample_counts),
            "repeat_seeds": list(repeat_seeds),
            "matrix": matrix,
        }
    )


def _validate_frozen_development_protocol() -> None:
    """Fail before evaluation if any full development setting has drifted."""
    observed = _protocol_sha256(
        sample_counts=DEVELOPMENT_SAMPLE_COUNTS,
        repeat_seeds=DEVELOPMENT_REPEAT_SEEDS,
        matrix=DEVELOPMENT_MATRIX,
    )
    if observed != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen RQMC development protocol identity changed")


def _build_artifact(
    aggregation: AdditiveDirichletAggregation,
    labels: c1.IntArray,
    *,
    sample_count: int,
    seed: int,
    case_id: str,
) -> ConditionalAllocationMixture:
    """Build one source-pinned independently scrambled RQMC artifact."""
    return ConditionalAllocationMixture.from_aggregation(
        aggregation,
        labels,
        sample_count=sample_count,
        source_seed=seed,
        source_provenance=(f"{PROTOCOL}:{case_id}:S={sample_count}:scramble_seed={seed}"),
        construction_method=BANK_CONSTRUCTION_METHOD,
    )


def run_case(
    *,
    regime_name: str,
    family: c1.Family,
    tiling: c1.Tiling,
    sample_counts: Sequence[int],
    repeat_seeds: Sequence[int],
    profile: Profile,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run one exact-versus-RQMC-bank case under the frozen C1 protocol."""
    if profile not in ("smoke", "development"):
        raise ValueError("profile must be 'smoke' or 'development'")
    if profile == "development":
        _validate_frozen_development_protocol()
        if scipy_version != DEVELOPMENT_SCIPY_VERSION:
            raise RuntimeError(
                f"development requires SciPy {DEVELOPMENT_SCIPY_VERSION}, observed {scipy_version}"
            )
    if not sample_counts or any(
        isinstance(value, bool) or int(value) != value or value < 1 for value in sample_counts
    ):
        raise ValueError("sample_counts must contain positive integers")
    normalized_counts = tuple(int(value) for value in sample_counts)
    if len(set(normalized_counts)) != len(normalized_counts):
        raise ValueError("sample_counts must be unique")
    if normalized_counts != tuple(sorted(normalized_counts)):
        raise ValueError("sample_counts must be strictly increasing")
    if any(count & (count - 1) for count in normalized_counts):
        raise ValueError("RQMC sample_counts must be powers of two")
    if not repeat_seeds or any(
        isinstance(value, bool) or int(value) != value or not 0 <= value < 2**64 for value in repeat_seeds
    ):
        raise ValueError("repeat_seeds must contain unsigned integer seeds")
    normalized_seeds = tuple(int(value) for value in repeat_seeds)
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("repeat_seeds must be unique")
    if profile == "development" and (
        normalized_counts != DEVELOPMENT_SAMPLE_COUNTS or normalized_seeds != DEVELOPMENT_REPEAT_SEEDS
    ):
        raise ValueError("development uses the source-pinned sample counts and seeds")

    case_key = (regime_name, family, tiling)
    allowed = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if case_key not in allowed:
        raise ValueError(f"case {case_key!r} is not available in {profile}")

    regime = c1._regime(regime_name)
    shapes, rate, design, observation, noise = c1._case_arrays(regime, family)
    common_projection = np.asarray(
        regime.projection2 if family == "two_cell" else regime.projection4,
        dtype=np.float64,
    )
    labels = c1.labels_for_tiling(family, tiling)
    if profile == "smoke":
        total_order, fraction_order = 8, 6
    else:
        total_order, fraction_order = regime.total_order, regime.fraction_order
    masses, log_prior = c1._mass_grid(
        shapes=shapes,
        rate=rate,
        family=family,
        tiling=tiling,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_log_likelihood = c1._exact_log_likelihood(
        masses=masses,
        shapes=shapes,
        rate=rate,
        design=design,
        observation=observation,
        noise=noise,
        family=family,
        tiling=tiling,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_summary = c1._posterior_summary(
        masses,
        log_prior,
        exact_log_likelihood,
    )
    prior_mean_coordinate = c1._anchor_coordinate(shapes, rate, labels)

    def exact_function(value: c1.FloatArray) -> float:
        state = c1.coordinate_to_masses(value)
        return float(
            c1._exact_log_likelihood(
                masses=state[None, :],
                shapes=shapes,
                rate=rate,
                design=design,
                observation=observation,
                noise=noise,
                family=family,
                tiling=tiling,
                total_order=total_order,
                fraction_order=fraction_order,
            )[0]
        )

    gradient_states = [
        {
            "state_id": state_id,
            "coordinate": state_coordinate.tolist(),
            "exact_coordinate_gradient": c1._centered_gradient(
                exact_function,
                state_coordinate,
            ).tolist(),
        }
        for state_id, state_coordinate in c1._gradient_state_coordinates(
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            prior_mean_coordinate=prior_mean_coordinate,
        )
    ]
    validation_state_mask = c1._development_validation_state_mask(
        masses,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )
    case_id = f"{regime_name}__{family}__{tiling}"

    def evaluate(sample_count: int, seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        artifact = _build_artifact(
            aggregation,
            labels,
            sample_count=sample_count,
            seed=seed,
            case_id=case_id,
        )
        build_seconds = time.perf_counter() - started if include_timings else None
        result = c1._evaluate_bank(
            artifact=artifact,
            observation=observation,
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            exact_summary=exact_summary,
            gradient_states=gradient_states,
            validation_state_mask=validation_state_mask,
            include_timings=include_timings,
        )
        result["build_seconds"] = build_seconds
        result["construction_method"] = _bank_method(artifact)
        result["bank_construction_provenance"] = _artifact_construction_provenance(artifact)
        result["scramble_seed"] = seed
        return result

    development_seed = normalized_seeds[0]
    confirmation_seeds = list(normalized_seeds[1:])
    development_evaluations = [evaluate(sample_count, development_seed) for sample_count in normalized_counts]
    development_pass_pattern = [
        {
            "sample_count": int(result["sample_count"]),
            "pass": bool(result["scientific_pass_without_repeat_evidence_gate"]),
        }
        for result in development_evaluations
    ]
    minimum_passing_suffix_length = 1 if profile == "smoke" else 2
    locked_sample_count = c1._stable_lock_sample_count(
        normalized_counts,
        [bool(result["scientific_pass_without_repeat_evidence_gate"]) for result in development_evaluations],
        minimum_suffix_length=minimum_passing_suffix_length,
    )
    confirmation_evaluations = (
        [evaluate(locked_sample_count, seed) for seed in confirmation_seeds]
        if locked_sample_count is not None
        else []
    )
    locked_development = next(
        (result for result in development_evaluations if result["sample_count"] == locked_sample_count),
        None,
    )
    locked_results = [locked_development, *confirmation_evaluations] if locked_development is not None else []
    locked_evidence = [result["posterior_summary"]["log_evidence"] for result in locked_results]
    evidence_range = float(max(locked_evidence) - min(locked_evidence)) if locked_evidence else None
    evidence_check = (
        bool(evidence_range <= THRESHOLDS["between_bank_log_evidence_range_nat"])
        if evidence_range is not None
        else False
    )
    confirmation_checks = [
        bool(result["scientific_pass_without_repeat_evidence_gate"]) for result in confirmation_evaluations
    ]
    confirmation_pass = (
        bool(all(confirmation_checks) and evidence_check) if confirmation_evaluations else None
    )
    development_lock_eligible = locked_sample_count is not None
    if profile == "smoke":
        case_pass = bool(development_lock_eligible and evidence_check)
    else:
        case_pass = bool(development_lock_eligible and confirmation_pass is True and evidence_check)
    construction_provenance = development_evaluations[0]["bank_construction_provenance"]
    if any(
        result["bank_construction_provenance"] != construction_provenance
        for result in [
            *development_evaluations,
            *confirmation_evaluations,
        ]
    ):
        raise RuntimeError("RQMC construction provenance changed within one frozen case")

    return {
        "case_id": case_id,
        "profile": profile,
        "input_sha256": c1._case_input_sha256(
            regime,
            family,
            tiling,
            total_order,
            fraction_order,
        ),
        "bank": {
            "construction_method": BANK_CONSTRUCTION_METHOD,
            "randomization": "independent_scrambled_sobol_per_source_seed",
            "sample_count_requirement": "power_of_two",
            **construction_provenance,
        },
        "regime": regime_name,
        "family": family,
        "tiling": tiling,
        "summary_basis": {
            "kind": "identity",
            "rank": int(observation.size),
            "observation_count": int(observation.size),
            "selection": ("fixed_full_rank_independent_of_observed_residual"),
        },
        "quadrature": {
            "total_order": total_order,
            "fraction_order": fraction_order,
            "mass_state_count": int(masses.shape[0]),
        },
        "mass_grid": {
            "integration_role": "complete_pinned_quadrature",
            "sha256": hashlib.sha256(np.ascontiguousarray(masses, dtype="<f8").tobytes()).hexdigest(),
            "pointwise_gate_split": {
                "scheme": "c1-checkerboard-by-total-and-share-index-v1",
                "new_in_c1_not_in_a1_or_t2": True,
                "used_for_development_pointwise_scoring": True,
                "is_protected_operator_or_partition_data": False,
                "validation_state_count": int(np.count_nonzero(validation_state_mask)),
                "validation_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        validation_state_mask,
                        dtype=np.uint8,
                    ).tobytes()
                ).hexdigest(),
                "alters_evidence_or_posterior_quadrature": False,
            },
        },
        "coordinate_names": (
            ["log_total"]
            if prior_mean_coordinate.size == 1
            else [
                "log_total",
                "log_first_to_second_region_mass_ratio",
            ]
        ),
        "gradient_state_catalogue": gradient_states,
        "common_native_projection": {
            "partition_invariant": True,
            "bank_posterior_summary_available": False,
            "definition_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    common_projection,
                    dtype="<f8",
                ).tobytes()
            ).hexdigest(),
            "status": (
                "deferred: the frozen observation bank does not retain the "
                "underlying allocation shares or projection factors"
            ),
        },
        "declared_structural_prior_weight": c1._structural_prior_weight(
            family,
            tiling,
        ),
        "evidence_merger_group_id": f"{regime_name}__{family}",
        "evidence_merger_thresholds": MERGER_THRESHOLDS,
        "independent_evidence_merger": {
            "status": "pending_not_implemented",
            "emitted_values_are_inputs_not_a_certificate": True,
        },
        "structural_evidence_use": ("diagnostic merger only; must not update partition or dimension"),
        "exact_posterior_summary": exact_summary,
        "development_seed": development_seed,
        "confirmation_seeds": confirmation_seeds,
        "development_evaluations": development_evaluations,
        "development_pass_pattern": development_pass_pattern,
        "minimum_passing_suffix_length": minimum_passing_suffix_length,
        "confirmation_evaluations": confirmation_evaluations,
        "locked_sample_count": locked_sample_count,
        "development_lock_eligible": development_lock_eligible,
        "lock_selection_rule": (
            "smallest predeclared S for which it and every larger S pass all "
            "development gates under the single development scramble"
        ),
        "confirmation_can_retune_lock": False,
        "lock_certificate": {
            "schema": "conditional-allocation-c1-rqmc-bank-lock-v1",
            "eligible": development_lock_eligible,
            "locked_sample_count": locked_sample_count,
            "development_seed": development_seed,
            "selection_rule_satisfied": development_lock_eligible,
            "minimum_passing_suffix_length": minimum_passing_suffix_length,
            "confirmation_requested": bool(confirmation_seeds),
            "confirmation_complete": bool(confirmation_evaluations)
            and len(confirmation_evaluations) == len(confirmation_seeds),
            "confirmation_pass": confirmation_pass,
            "confirmation_can_retune": False,
            "full_c1_promotion_licensed": False,
        },
        "between_bank_log_evidence_range_nat": evidence_range,
        "between_bank_log_evidence_range_pass": evidence_check,
        "confirmation_pass": confirmation_pass,
        "scientific_pass": case_pass,
    }


def _driver_sha256() -> str:
    """Return the exact identity of this executable source file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def matrix_catalogue() -> dict[str, Any]:
    """Return the immutable development and inaccessible held-out catalogue."""
    return {
        "schema": SCHEMA,
        "bank_construction_method": BANK_CONSTRUCTION_METHOD,
        "required_development_construction_scipy_version": (DEVELOPMENT_SCIPY_VERSION),
        "development": [list(case) for case in DEVELOPMENT_MATRIX],
        "smoke": [list(case) for case in SMOKE_MATRIX],
        "held_out_catalogue": {
            "id": HELD_OUT_CATALOGUE_ID,
            "sha256": HELD_OUT_CATALOGUE_SHA256,
            "numerical_values_present": False,
            "executable_here": False,
        },
        "held_out_information_read": False,
    }


def run_screen(
    *,
    profile: Profile,
    sample_counts: Sequence[int] | None = None,
    repeat_seeds: Sequence[int] | None = None,
    case_id: str | None = None,
    source_revision: str | None = None,
    include_timings: bool = True,
) -> dict[str, Any]:
    """Run a smoke or one/all predeclared RQMC development cases."""
    if profile not in ("smoke", "development"):
        raise ValueError("held-out execution is deliberately unavailable")
    observed_definitions_sha = c1.a1_definitions_sha256()
    if observed_definitions_sha != A1_DEFINITIONS_SHA256:
        raise RuntimeError("shared A1 numerical definitions no longer match their pin")
    if profile == "development" and (sample_counts is not None or repeat_seeds is not None):
        raise ValueError(
            "development uses the source-pinned sample counts and seeds; "
            "overrides require a new reviewed protocol revision"
        )
    if profile == "development":
        _validate_frozen_development_protocol()
        if scipy_version != DEVELOPMENT_SCIPY_VERSION:
            raise RuntimeError(
                f"development requires SciPy {DEVELOPMENT_SCIPY_VERSION}, observed {scipy_version}"
            )
    resolved_source_revision = c1._source_revision(source_revision)
    if profile == "smoke":
        matrix = SMOKE_MATRIX
        selected_counts = SMOKE_SAMPLE_COUNTS if sample_counts is None else tuple(sample_counts)
        selected_seeds = SMOKE_REPEAT_SEEDS if repeat_seeds is None else tuple(repeat_seeds)
    elif profile == "development":
        matrix = DEVELOPMENT_MATRIX
        selected_counts = DEVELOPMENT_SAMPLE_COUNTS
        selected_seeds = DEVELOPMENT_REPEAT_SEEDS
    else:  # pragma: no cover - validated above
        raise AssertionError("unreachable profile")
    if case_id is not None:
        matches = [case for case in matrix if "__".join(case) == case_id]
        if len(matches) != 1:
            raise ValueError(f"case_id {case_id!r} is not available in profile {profile}")
        matrix = tuple(matches)
    cases = [
        run_case(
            regime_name=regime,
            family=cast(c1.Family, family),
            tiling=cast(c1.Tiling, tiling),
            sample_counts=selected_counts,
            repeat_seeds=selected_seeds,
            profile=profile,
            include_timings=include_timings,
        )
        for regime, family, tiling in matrix
    ]
    shared_provenance_fields = tuple(
        field
        for field in _ARTIFACT_PROVENANCE_FIELDS
        if field not in ("sobol_catalogue_sha256", "sobol_block_dimensions")
    )
    report_bank: dict[str, object] = {
        "construction_method": BANK_CONSTRUCTION_METHOD,
        "randomization": "independent_scrambled_sobol_per_source_seed",
        "sample_count_requirement": "power_of_two",
    }
    for field in shared_provenance_fields:
        values = {c1._canonical_json(case["bank"][field]) for case in cases}
        if len(values) != 1:
            raise RuntimeError(f"RQMC report cases disagree on construction provenance {field}")
        report_bank[field] = cases[0]["bank"][field]
    report_bank["sobol_catalogue_sha256"] = {
        case["case_id"]: case["bank"]["sobol_catalogue_sha256"] for case in cases
    }
    report_bank["sobol_block_dimensions"] = {
        case["case_id"]: case["bank"]["sobol_block_dimensions"] for case in cases
    }
    report_bank["protocol_hash_inclusion"] = {
        "algorithm_token_included": True,
        "required_development_scipy_version_included": True,
        "observed_runtime_and_derived_provenance_included": False,
        "reason": (
            "development fails unless the observed runtime matches the hashed "
            "SciPy requirement; catalogue/block identities are deterministic "
            "consequences of already pinned case inputs"
        ),
    }
    protocol_sha256 = _protocol_sha256(
        sample_counts=selected_counts,
        repeat_seeds=selected_seeds,
        matrix=matrix,
    )
    return {
        "schema": SCHEMA,
        "completion_scope": "rqmc_smoke_and_development_only_not_full_c1",
        "protocol": PROTOCOL,
        "profile": profile,
        "selected_case_id": case_id,
        "per_case_atomic_output": case_id is not None,
        "source_git_revision": resolved_source_revision,
        "driver_sha256": _driver_sha256(),
        "a1_source_revision": A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": observed_definitions_sha,
        "required_development_construction_scipy_version": (DEVELOPMENT_SCIPY_VERSION),
        "bank": report_bank,
        "protocol_sha256": protocol_sha256,
        "frozen_full_development_protocol_sha256": (DEVELOPMENT_PROTOCOL_SHA256),
        "held_out_information_read": False,
        "held_out_operator_partition_information_read": False,
        "c1_pointwise_validation_subset_evaluated": True,
        "held_out_execution_available": False,
        "observed_residual_used_for_basis_selection": False,
        "structural_inference_licensed": False,
        "full_c1_pass": False,
        "full_c1_pass_reason": (
            "held-out operators, partitions, retained-mass grids, and "
            "independent held-out invocation are intentionally not implemented"
        ),
        "independent_evidence_merger_status": "pending_not_implemented",
        "sample_counts": list(selected_counts),
        "repeat_seeds": list(selected_seeds),
        "bank_lock_protocol": {
            "development_seed": int(selected_seeds[0]),
            "confirmation_seeds": [int(seed) for seed in selected_seeds[1:]],
            "selection_rule": ("smallest S passing development gates under development scramble"),
            "confirmation_can_retune": False,
        },
        "thresholds": THRESHOLDS,
        "merger_thresholds": MERGER_THRESHOLDS,
        "matrix_catalogue": matrix_catalogue(),
        "cases": cases,
        "scientific_pass": all(case["scientific_pass"] for case in cases),
    }


def _positive_csv(
    value: str,
    *,
    name: str,
    upper_bound: int,
) -> tuple[int, ...]:
    """Parse one strict comma-separated positive integer sequence."""
    return c1._positive_csv(value, name=name, upper_bound=upper_bound)


def _power_of_two_csv(value: str) -> tuple[int, ...]:
    """Parse the RQMC bank-size list and reject non-power-of-two values."""
    parsed = _positive_csv(
        value,
        name="sample-counts",
        upper_bound=1_048_576,
    )
    if any(item & (item - 1) for item in parsed):
        raise argparse.ArgumentTypeError("sample-counts must contain powers of two")
    return parsed


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish canonical JSON once, without partial or overwritten output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
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
            stream.write(c1._canonical_json(payload))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        default="smoke",
    )
    parser.add_argument(
        "--case-id",
        help=(
            "Run one profile case as REGIME__FAMILY__TILING; required for "
            "independent Slurm array outputs but optional for a full profile"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-revision",
        help=("Expected full lower-case Git SHA; required when Git is absent from the execution environment"),
    )
    parser.add_argument(
        "--sample-counts",
        type=_power_of_two_csv,
    )
    parser.add_argument(
        "--repeat-seeds",
        type=lambda value: _positive_csv(
            value,
            name="repeat-seeds",
            upper_bound=2**63 - 1,
        ),
    )
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--no-timings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the CLI, run the selected screen, and publish only on success."""
    args = _parser().parse_args(argv)
    if args.list_matrix:
        if (
            args.output is not None
            or args.sample_counts
            or args.repeat_seeds
            or args.case_id
            or args.source_revision
        ):
            raise SystemExit("--list-matrix cannot be combined with run options")
        print(c1._canonical_json(matrix_catalogue()))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list-matrix is used")
    if args.profile == "development" and (args.sample_counts is not None or args.repeat_seeds is not None):
        raise SystemExit("development sample counts and seeds are source-pinned and cannot be overridden")
    report = run_screen(
        profile=args.profile,
        sample_counts=args.sample_counts,
        repeat_seeds=args.repeat_seeds,
        case_id=args.case_id,
        source_revision=args.source_revision,
        include_timings=not args.no_timings,
    )
    _write_atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
