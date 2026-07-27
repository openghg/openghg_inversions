#!/usr/bin/env python
"""Build and score the predeclared BP1 native-quadrature marginal.

The driver constructs support-exact Gauss--Jacobi allocation coordinates,
analytically convolves their weighted pushforward with Gaussian measurement
noise, and applies the unchanged C1 scientific gates. Protected domains are
unreachable from this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import scipy

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as flow_screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
    QuadratureChart,
)

FloatArray: TypeAlias = NDArray[np.float64]
Profile = Literal["smoke", "development"]

SCHEMA = "rjmcmc-conditional-native-quadrature-tiny-screen-v1"
PROTOCOL = "conditional-native-support-quadrature-v1"
DEVELOPMENT_PROTOCOL_SHA256 = "d11ba4b37c973772a0ad2bc6caff1e29c56deac0ef6d94fd348a25f468c9b49c"
PROTECTED_HOLDOUT_CATALOGUE_ID = "conditional-native-quadrature-protected-v1"

DEVELOPMENT_MATRIX = tuple(
    (regime, family, "root")
    for regime in ("near_gaussian", "skewed", "boundary_heavy")
    for family in ("two_cell", "four_cell")
)
SMOKE_MATRIX = (("near_gaussian", "two_cell", "root"),)
DEVELOPMENT_QUADRATURE_ORDERS = (24, 32, 40, 48)
SMOKE_QUADRATURE_ORDERS = (8,)
DEVELOPMENT_SELECTION_SEED = 731
CONFIRMATION_SEEDS = (1_877, 4_099, 8_317)
CONFIRMATION_SAMPLE_COUNT = 131_072
MOMENT_MEAN_TOLERANCE = 2.0e-10
MOMENT_COVARIANCE_TOLERANCE = 5.0e-10


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    """Return the SHA-256 of canonical JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _array_sha256(values: NDArray[np.generic]) -> str:
    """Return a shape-aware canonical array digest."""
    if np.issubdtype(values.dtype, np.floating):
        array = np.ascontiguousarray(values, dtype="<f8")
        dtype = "<f8"
    elif np.issubdtype(values.dtype, np.integer):
        array = np.ascontiguousarray(values, dtype="<i8")
        dtype = "<i8"
    else:
        raise TypeError("only floating and integer arrays can be hashed")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": dtype,
                "shape": list(array.shape),
            }
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    """Return the complete frozen development declaration."""
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "architecture": {
            "allocation_coordinates": ("exact-beta-gauss-jacobi-neutral-to-the-right"),
            "two_cell_chart": "single",
            "four_cell_published_chart": "column-first",
            "four_cell_audit_chart": "row-first",
            "gaussian_convolution": "analytic-unit-covariance-components",
            "residual_image": "complete-canonical",
            "dtype": "float64",
            "component_weights": "positive-normalized-product-rule",
            "training": None,
        },
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "development_matrix": [list(case) for case in DEVELOPMENT_MATRIX],
        "quadrature_orders": list(DEVELOPMENT_QUADRATURE_ORDERS),
        "moment_tolerances": {
            "absolute_weighted_residual_mean": MOMENT_MEAN_TOLERANCE,
            "absolute_weighted_residual_covariance": (MOMENT_COVARIANCE_TOLERANCE),
        },
        "scientific_thresholds": c1.THRESHOLDS,
        "common_lock": ("smallest common six-case all-larger passing suffix of length at least two"),
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "sample_count": CONFIRMATION_SAMPLE_COUNT,
            "state": "root-prior-mean-mass",
            "batch_count": 128,
            "batch_sample_count": 1_024,
            "mean_mcse_multiplier": 5.0,
            "covariance_mcse_multiplier": 5.0,
            "finite_density_index_stratified_count": 256,
            "component_frequency_target_bin_probability": 1.0 / 256.0,
            "minimum_merged_component_expected_count": 20,
            "component_frequency_survival_probability_minimum": 1.0e-6,
        },
        "protected": {
            "catalogue_id": PROTECTED_HOLDOUT_CATALOGUE_ID,
            "protected_action_authorized": False,
            "g3_requires_passing_g2_holdout_eligible_certificate": True,
        },
    }


def _protocol_sha256() -> str:
    """Return the development declaration digest."""
    return _sha256_json(_protocol_payload())


def _validate_development_protocol() -> None:
    """Fail closed until the complete declaration digest is frozen."""
    if not DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the native-quadrature protocol is not frozen")
    if _protocol_sha256() != DEVELOPMENT_PROTOCOL_SHA256:
        raise RuntimeError("the frozen native-quadrature protocol changed")


def _validate_source_identity(
    source_git_revision: str,
    driver_sha256: str,
) -> None:
    """Validate complete source identities."""
    if len(source_git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_git_revision
    ):
        raise ValueError("source_git_revision must be a complete lower-case Git SHA")
    if len(driver_sha256) != 64 or any(character not in "0123456789abcdef" for character in driver_sha256):
        raise ValueError("driver_sha256 must be a lower-case SHA-256 digest")


def _candidate_chart(family: c1.Family) -> QuadratureChart:
    """Return the frozen published chart for one family."""
    return "single" if family == "two_cell" else "column-first"


def _exact_case(
    *,
    regime_name: str,
    family: c1.Family,
    profile: Profile,
) -> dict[str, Any]:
    """Build the unchanged C1 comparator inputs for one root case."""
    regime = c1._regime(regime_name)  # pyright: ignore[reportPrivateUsage]
    shapes, rate, design, observation, noise = c1._case_arrays(  # pyright: ignore[reportPrivateUsage]
        regime,
        family,
    )
    labels = c1.labels_for_tiling(family, "root")
    total_order = 8 if profile == "smoke" else regime.total_order
    fraction_order = 6 if profile == "smoke" else regime.fraction_order
    masses, log_prior = c1._mass_grid(  # pyright: ignore[reportPrivateUsage]
        shapes=shapes,
        rate=rate,
        family=family,
        tiling="root",
        total_order=total_order,
        fraction_order=fraction_order,
    )
    exact_log_likelihood = c1._exact_log_likelihood(  # pyright: ignore[reportPrivateUsage]
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
    exact_summary = c1._posterior_summary(  # pyright: ignore[reportPrivateUsage]
        masses,
        log_prior,
        exact_log_likelihood,
    )
    prior_mean_coordinate = c1._anchor_coordinate(  # pyright: ignore[reportPrivateUsage]
        shapes,
        rate,
        labels,
    )

    def exact_function(value: FloatArray) -> float:
        return float(
            c1._exact_log_likelihood(  # pyright: ignore[reportPrivateUsage]
                masses=c1.coordinate_to_masses(value)[np.newaxis, :],
                shapes=shapes,
                rate=rate,
                design=design,
                observation=observation,
                noise=noise,
                family=family,
                tiling="root",
                total_order=total_order,
                fraction_order=fraction_order,
            )[0]
        )

    gradient_states = [
        {
            "state_id": state_id,
            "coordinate": coordinate.tolist(),
            "exact_coordinate_gradient": c1._centered_gradient(  # pyright: ignore[reportPrivateUsage]
                exact_function,
                coordinate,
            ).tolist(),
        }
        for state_id, coordinate in c1._gradient_state_coordinates(  # pyright: ignore[reportPrivateUsage]
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            prior_mean_coordinate=prior_mean_coordinate,
        )
    ]
    validation_state_mask = c1._development_validation_state_mask(  # pyright: ignore[reportPrivateUsage]
        masses,
        total_order=total_order,
        fraction_order=fraction_order,
    )
    return {
        "regime": regime,
        "shapes": shapes,
        "rate": rate,
        "design": design,
        "observation": observation,
        "noise": noise,
        "labels": labels,
        "total_order": total_order,
        "fraction_order": fraction_order,
        "masses": masses,
        "log_prior": log_prior,
        "exact_log_likelihood": exact_log_likelihood,
        "exact_summary": exact_summary,
        "gradient_states": gradient_states,
        "validation_state_mask": validation_state_mask,
    }


def _construct_artifact(
    case: dict[str, Any],
    *,
    family: c1.Family,
    quadrature_order: int,
    chart: QuadratureChart,
    source_git_revision: str,
    driver_sha256: str,
    case_id: str,
) -> ConditionalNativeQuadrature:
    """Construct one authenticated artifact from exact native inputs."""
    shapes = cast(FloatArray, case["shapes"])
    design = cast(FloatArray, case["design"])
    noise = cast(FloatArray, case["noise"])
    labels = cast(NDArray[np.int64], case["labels"])
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(design.shape[0], dtype=np.float64),
    )
    return ConditionalNativeQuadrature.from_aggregation(
        aggregation,
        labels,
        np.arange(shapes.size, dtype=np.int64),
        quadrature_order=quadrature_order,
        chart=chart,
        source_git_revision=source_git_revision,
        driver_sha256=driver_sha256,
        protocol_sha256=_protocol_sha256(),
        source_provenance=(f"{PROTOCOL}:{case_id}:O{quadrature_order}:{chart}:source-{source_git_revision}"),
    )


def _moment_audit(
    artifact: ConditionalNativeQuadrature,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Compare quadrature residual moments with exact Dirichlet moments."""
    weights = artifact.normalized_weights
    factors = artifact.projected_unit_mass_residual_factors[:, :, 0]
    weighted_mean = weights @ factors
    centered = factors - weighted_mean[np.newaxis, :]
    weighted_covariance = np.einsum(
        "s,si,sj->ij",
        weights,
        centered,
        centered,
        optimize=False,
    )
    projected = AdditiveDirichletAggregation(
        case["shapes"],
        case["design"],
        case["noise"],
        artifact.context.residual_basis,
    )
    exact_covariance = projected.partition_factors(artifact.context.labels).summary_covariance_factors[0]
    mean_error = float(np.max(np.abs(weighted_mean), initial=0.0))
    covariance_error = float(
        np.max(
            np.abs(weighted_covariance - exact_covariance),
            initial=0.0,
        )
    )
    return {
        "weighted_residual_mean": weighted_mean.tolist(),
        "weighted_residual_covariance_sha256": _array_sha256(cast(NDArray[np.generic], weighted_covariance)),
        "exact_residual_covariance_sha256": _array_sha256(cast(NDArray[np.generic], exact_covariance)),
        "maximum_absolute_mean_error": mean_error,
        "maximum_absolute_covariance_error": covariance_error,
        "mean_tolerance": MOMENT_MEAN_TOLERANCE,
        "covariance_tolerance": MOMENT_COVARIANCE_TOLERANCE,
        "pass": bool(mean_error <= MOMENT_MEAN_TOLERANCE and covariance_error <= MOMENT_COVARIANCE_TOLERANCE),
    }


def _chart_audit(
    artifact: ConditionalNativeQuadrature,
    case: dict[str, Any],
    *,
    family: c1.Family,
    quadrature_order: int,
    source_git_revision: str,
    driver_sha256: str,
    case_id: str,
) -> dict[str, Any]:
    """Record finite same-order chart discrepancies."""
    if family == "two_cell":
        return {
            "applicable": False,
            "maximum_absolute_log_likelihood_difference_nat": 0.0,
            "maximum_absolute_log_mass_gradient_difference": 0.0,
            "finite": True,
        }
    alternate = _construct_artifact(
        case,
        family=family,
        quadrature_order=quadrature_order,
        chart="row-first",
        source_git_revision=source_git_revision,
        driver_sha256=driver_sha256,
        case_id=case_id,
    )
    observation = cast(FloatArray, case["observation"])
    masses = cast(FloatArray, case["masses"])
    published_values = artifact.log_likelihood_batch(observation, masses)
    alternate_values = alternate.log_likelihood_batch(observation, masses)
    gradient_difference = 0.0
    for state in case["gradient_states"]:
        retained = c1.coordinate_to_masses(np.asarray(state["coordinate"], dtype=np.float64))
        published_gradient = artifact.log_likelihood_and_mass_gradient(
            observation,
            retained,
        )[1]
        alternate_gradient = alternate.log_likelihood_and_mass_gradient(
            observation,
            retained,
        )[1]
        gradient_difference = max(
            gradient_difference,
            float(np.max(np.abs(published_gradient - alternate_gradient))),
        )
    likelihood_difference = float(np.max(np.abs(published_values - alternate_values)))
    return {
        "applicable": True,
        "published_chart": artifact.chart,
        "alternate_chart": alternate.chart,
        "alternate_artifact_sha256": alternate.artifact_sha256,
        "maximum_absolute_log_likelihood_difference_nat": likelihood_difference,
        "maximum_absolute_log_mass_gradient_difference": gradient_difference,
        "finite": bool(math.isfinite(likelihood_difference) and math.isfinite(gradient_difference)),
    }


def _smoke_sample_audit(
    artifact: ConditionalNativeQuadrature,
    *,
    base_seed: int,
) -> dict[str, Any]:
    """Run one bounded density/simulator consistency check."""
    masses = np.ones(artifact.region_count, dtype=np.float64)
    expected_mean, expected_covariance = artifact.analytic_mean_and_covariance(masses)
    sample_count = 4_096
    samples = artifact.sample(
        masses,
        sample_count=sample_count,
        rng=np.random.default_rng(base_seed),
    )
    mcse = np.sqrt(np.diag(expected_covariance) / sample_count)
    scaled_error = float(np.max(np.abs(np.mean(samples, axis=0) - expected_mean) / mcse))
    return {
        "sample_count": sample_count,
        "sample_sha256": _array_sha256(cast(NDArray[np.generic], samples)),
        "maximum_mean_error_mcse": scaled_error,
        "pass": bool(np.all(np.isfinite(samples)) and scaled_error <= 6.0),
    }


def run_case(
    *,
    regime_name: str,
    family: c1.Family,
    quadrature_order: int,
    base_seed: int,
    profile: Profile,
    source_git_revision: str,
    driver_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Construct and score one source-pinned root case and order."""
    case_key = (regime_name, family, "root")
    allowed = SMOKE_MATRIX if profile == "smoke" else DEVELOPMENT_MATRIX
    if case_key not in allowed:
        raise ValueError(f"case {case_key!r} is not available in {profile}")
    orders = SMOKE_QUADRATURE_ORDERS if profile == "smoke" else DEVELOPMENT_QUADRATURE_ORDERS
    if quadrature_order not in orders:
        raise ValueError("quadrature_order is not source-pinned")
    _validate_source_identity(source_git_revision, driver_sha256)
    if profile == "development":
        _validate_development_protocol()
    case = _exact_case(
        regime_name=regime_name,
        family=family,
        profile=profile,
    )
    case_id = f"{regime_name}__{family}__root"
    artifact = _construct_artifact(
        case,
        family=family,
        quadrature_order=quadrature_order,
        chart=_candidate_chart(family),
        source_git_revision=source_git_revision,
        driver_sha256=driver_sha256,
        case_id=case_id,
    )
    artifact_bytes = artifact.to_bytes()
    replay = ConditionalNativeQuadrature.from_bytes(
        artifact_bytes,
        expected_sha256=artifact.artifact_sha256,
    )
    replay_pass = bool(
        replay.to_bytes() == artifact_bytes and replay.artifact_sha256 == artifact.artifact_sha256
    )
    moment_audit = _moment_audit(artifact, case)
    chart_audit = _chart_audit(
        artifact,
        case,
        family=family,
        quadrature_order=quadrature_order,
        source_git_revision=source_git_revision,
        driver_sha256=driver_sha256,
        case_id=case_id,
    )
    evaluation = flow_screen._evaluate_artifact(  # pyright: ignore[reportPrivateUsage]
        artifact=cast(Any, artifact),
        observation=case["observation"],
        masses=case["masses"],
        log_prior=case["log_prior"],
        exact_log_likelihood=case["exact_log_likelihood"],
        exact_summary=case["exact_summary"],
        gradient_states=case["gradient_states"],
        validation_state_mask=case["validation_state_mask"],
    )
    sample_audit = _smoke_sample_audit(artifact, base_seed=base_seed) if profile == "smoke" else None
    operational_pass = bool(
        replay_pass
        and moment_audit["pass"]
        and chart_audit["finite"]
        and (sample_audit is None or sample_audit["pass"])
    )
    task_pass = bool(operational_pass and (profile == "smoke" or bool(evaluation["scientific_pass"])))
    return {
        "schema": SCHEMA,
        "protocol": {
            "name": PROTOCOL,
            "sha256": _protocol_sha256(),
            "payload": _protocol_payload(),
        },
        "profile": profile,
        "source": {
            "git_revision": source_git_revision,
            "driver_sha256": driver_sha256,
        },
        "case_id": case_id,
        "quadrature_order": quadrature_order,
        "component_count": artifact.component_count,
        "base_seed": base_seed,
        "chart": artifact.chart,
        "context_sha256": artifact.context.artifact_sha256,
        "selected_artifact_sha256": artifact.artifact_sha256,
        "artifact_replay_pass": replay_pass,
        "moment_audit": moment_audit,
        "chart_audit": chart_audit,
        "sample_audit": sample_audit,
        "evaluation": evaluation,
        "task_pass": task_pass,
    }, artifact_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes atomically within an existing output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_result(
    output_directory: Path,
    result: dict[str, Any],
    artifact_bytes: bytes,
) -> dict[str, str]:
    """Publish artifact, report, and completion marker in that order."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['case_id']}__O{result['quadrature_order']}__base{result['base_seed']}"
    artifact_path = output_directory / f"{stem}.nq"
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != result["selected_artifact_sha256"]:
        raise RuntimeError("artifact bytes do not match the result identity")
    _atomic_write(artifact_path, artifact_bytes)
    envelope_payload = {
        "result": result,
        "artifact": {
            "path": artifact_path.name,
            "sha256": artifact_sha256,
        },
    }
    envelope = {
        "payload": envelope_payload,
        "sha256": _sha256_json(envelope_payload),
    }
    report_path = output_directory / f"{stem}.json"
    report_bytes = (_canonical_json(envelope) + "\n").encode("utf-8")
    _atomic_write(report_path, report_bytes)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    marker_payload = {
        "schema": "rjmcmc-conditional-native-quadrature-task-complete-v1",
        "case_id": result["case_id"],
        "quadrature_order": result["quadrature_order"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
    }
    marker_path = output_directory / f"{stem}.complete.json"
    _atomic_write(
        marker_path,
        (_canonical_json(marker_payload) + "\n").encode("utf-8"),
    )
    return {
        "artifact": str(artifact_path),
        "report": str(report_path),
        "completion_marker": str(marker_path),
    }


def main() -> None:
    """Run one independent smoke or development task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        required=True,
    )
    parser.add_argument(
        "--regime",
        choices=("near_gaussian", "skewed", "boundary_heavy"),
        required=True,
    )
    parser.add_argument(
        "--family",
        choices=("two_cell", "four_cell"),
        required=True,
    )
    parser.add_argument("--quadrature-order", type=int, required=True)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEVELOPMENT_SELECTION_SEED,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    args = parser.parse_args()
    result, artifact = run_case(
        regime_name=args.regime,
        family=cast(c1.Family, args.family),
        quadrature_order=args.quadrature_order,
        base_seed=args.base_seed,
        profile=cast(Profile, args.profile),
        source_git_revision=args.source_git_revision,
        driver_sha256=args.driver_sha256,
    )
    paths = _write_result(args.output_directory, result, artifact)
    print(_canonical_json({"task_pass": result["task_pass"], "paths": paths}))


if __name__ == "__main__":
    main()
