#!/usr/bin/env python
"""Run one permitted native-quadrature G2 simulator confirmation shard."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_native_quadrature_certify as development
from examples.rjmcmc import conditional_native_quadrature_tiny_screen as screen
from examples.rjmcmc import conditional_residual_image_flow_certify as common
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as flow_screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
)

SCHEMA = "rjmcmc-conditional-native-quadrature-confirmation-v1"
MARKER_SCHEMA = "rjmcmc-conditional-native-quadrature-confirmation-complete-v1"
_BATCH_COUNT = 128
_BATCH_SAMPLE_COUNT = 1_024
_DENSITY_AUDIT_COUNT = 256
_FREQUENCY_TARGET_PROBABILITY = 1.0 / 256.0
_FREQUENCY_MINIMUM_EXPECTED = 20.0
_FREQUENCY_MINIMUM_SURVIVAL = 1.0e-6


def _authenticate_lock(
    lock_path: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], int]:
    """Authenticate one complete six-case G1 lock."""
    envelope = common._read_json(lock_path)  # pyright: ignore[reportPrivateUsage]
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
        raise ValueError("G1 lock envelope is malformed")
    payload = envelope["payload"]
    if envelope["sha256"] != common._sha256_json(  # pyright: ignore[reportPrivateUsage]
        payload
    ):
        raise ValueError("G1 lock envelope digest does not match")
    expected_cases = {
        f"{regime}__{family}__root"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    }
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != development.LOCK_SCHEMA
        or payload.get("source")
        != {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        }
        or payload.get("protocol_sha256") != expected_protocol_sha256
        or payload.get("selection_seed") != screen.DEVELOPMENT_SELECTION_SEED
        or payload.get("confirmation_seeds") != list(screen.CONFIRMATION_SEEDS)
        or payload.get("confirmation_sample_count") != screen.CONFIRMATION_SAMPLE_COUNT
        or set(payload.get("selected_artifacts", {})) != expected_cases
    ):
        raise ValueError("G1 lock does not authenticate every case and protocol")
    order = payload.get("locked_order")
    if order not in screen.DEVELOPMENT_QUADRATURE_ORDERS:
        raise ValueError("G1 lock order is not source-pinned")
    return payload, int(order)


def _frequency_groups(
    probabilities: NDArray[np.float64],
    *,
    sample_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return canonical adjacent component groups and their probabilities."""
    group_ids = np.empty(probabilities.size, dtype=np.int64)
    group_probabilities: list[float] = []
    group = 0
    running = 0.0
    for index, probability in enumerate(probabilities):
        group_ids[index] = group
        running += float(probability)
        if running >= _FREQUENCY_TARGET_PROBABILITY:
            group_probabilities.append(running)
            group += 1
            running = 0.0
    if running:
        if not group_probabilities:
            group_probabilities.append(running)
        else:
            group_ids[group_ids == group] = group - 1
            group_probabilities[-1] += running
    normalized = np.asarray(group_probabilities, dtype=np.float64)
    normalized /= float(np.sum(normalized))
    expected = sample_count * normalized
    if np.any(expected < _FREQUENCY_MINIMUM_EXPECTED):
        raise RuntimeError("component-frequency grouping has a sparse expected bin")
    return group_ids, normalized


def _simulator_audit(
    artifact: ConditionalNativeQuadrature,
    *,
    masses: NDArray[np.float64],
    seed: int,
) -> dict[str, Any]:
    """Apply the predeclared G2 simulator and frequency checks."""
    samples, component_indices = artifact.sample_with_component_indices(
        masses,
        sample_count=screen.CONFIRMATION_SAMPLE_COUNT,
        rng=np.random.default_rng(seed),
    )
    analytic_mean, analytic_covariance = artifact.analytic_mean_and_covariance(masses)
    batches = samples.reshape(
        _BATCH_COUNT,
        _BATCH_SAMPLE_COUNT,
        artifact.observation_count,
    )
    batch_means = np.mean(batches, axis=1)
    batch_covariances = np.asarray(
        [np.cov(batch, rowvar=False, ddof=1) for batch in batches],
        dtype=np.float64,
    )
    if artifact.observation_count == 1:
        batch_covariances = batch_covariances.reshape(_BATCH_COUNT, 1, 1)
    estimated_mean = np.mean(batch_means, axis=0)
    estimated_covariance = np.mean(batch_covariances, axis=0)
    mean_mcse = np.std(batch_means, axis=0, ddof=1) / math.sqrt(_BATCH_COUNT)
    covariance_mcse = np.std(batch_covariances, axis=0, ddof=1) / math.sqrt(_BATCH_COUNT)
    if np.any(mean_mcse <= 0.0) or np.any(covariance_mcse <= 0.0):
        raise RuntimeError("simulator batch MCSE is not positive")
    mean_scaled_error = float(np.max(np.abs(estimated_mean - analytic_mean) / mean_mcse))
    covariance_scaled_error = float(
        np.max(np.abs(estimated_covariance - analytic_covariance) / covariance_mcse)
    )

    audit_indices = np.linspace(
        0,
        samples.shape[0] - 1,
        _DENSITY_AUDIT_COUNT,
        dtype=np.int64,
    )
    audit_log_likelihoods = np.asarray(
        [artifact.log_likelihood(samples[index], masses) for index in audit_indices],
        dtype=np.float64,
    )
    group_ids, group_probabilities = _frequency_groups(
        artifact.normalized_weights,
        sample_count=samples.shape[0],
    )
    observed_counts = np.bincount(
        group_ids[component_indices],
        minlength=group_probabilities.size,
    ).astype(np.float64)
    expected_counts = samples.shape[0] * group_probabilities
    chi_square = float(np.sum(np.square(observed_counts - expected_counts) / expected_counts))
    degrees_of_freedom = int(expected_counts.size - 1)
    survival = float(stats.chi2.sf(chi_square, degrees_of_freedom))
    finite_density_pass = bool(np.all(np.isfinite(audit_log_likelihoods)))
    return {
        "sample_count": int(samples.shape[0]),
        "seed": seed,
        "state_masses": masses.tolist(),
        "sample_sha256": screen._array_sha256(  # pyright: ignore[reportPrivateUsage]
            cast(NDArray[np.generic], samples)
        ),
        "component_indices_sha256": screen._array_sha256(  # pyright: ignore[reportPrivateUsage]
            cast(NDArray[np.generic], component_indices)
        ),
        "batch_count": _BATCH_COUNT,
        "batch_sample_count": _BATCH_SAMPLE_COUNT,
        "maximum_mean_error_mcse": mean_scaled_error,
        "maximum_covariance_error_mcse": covariance_scaled_error,
        "mean_pass": bool(mean_scaled_error <= 5.0),
        "covariance_pass": bool(covariance_scaled_error <= 5.0),
        "density_audit_count": _DENSITY_AUDIT_COUNT,
        "density_audit_sha256": screen._array_sha256(  # pyright: ignore[reportPrivateUsage]
            cast(NDArray[np.generic], audit_log_likelihoods)
        ),
        "finite_density_pass": finite_density_pass,
        "component_frequency": {
            "group_count": int(group_probabilities.size),
            "minimum_expected_count": float(np.min(expected_counts)),
            "chi_square": chi_square,
            "degrees_of_freedom": degrees_of_freedom,
            "survival_probability": survival,
            "minimum_survival_probability": _FREQUENCY_MINIMUM_SURVIVAL,
            "pass": bool(survival >= _FREQUENCY_MINIMUM_SURVIVAL),
        },
        "pass": bool(
            mean_scaled_error <= 5.0
            and covariance_scaled_error <= 5.0
            and finite_density_pass
            and survival >= _FREQUENCY_MINIMUM_SURVIVAL
        ),
    }


def run_confirmation(
    *,
    regime_name: str,
    family: c1.Family,
    base_seed: int,
    lock_path: Path,
    artifact_path: Path,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Run one authenticated independent G2 shard."""
    if base_seed not in screen.CONFIRMATION_SEEDS:
        raise ValueError("base_seed is not a source-pinned confirmation seed")
    lock, locked_order = _authenticate_lock(
        lock_path,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    case_id = f"{regime_name}__{family}__root"
    selected = lock["selected_artifacts"].get(case_id)
    if not isinstance(selected, dict):
        raise ValueError("G1 lock does not select the requested case")
    artifact_bytes = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != selected.get("artifact_sha256"):
        raise ValueError("G1 selected artifact digest does not match")
    artifact = ConditionalNativeQuadrature.from_bytes(
        artifact_bytes,
        expected_sha256=artifact_sha256,
    )
    if (
        artifact.quadrature_order != locked_order
        or artifact.source_git_revision != expected_source_revision
        or artifact.driver_sha256 != expected_driver_sha256
        or artifact.protocol_sha256 != expected_protocol_sha256
    ):
        raise ValueError("G1 artifact identity does not match the lock")
    case = screen._exact_case(  # pyright: ignore[reportPrivateUsage]
        regime_name=regime_name,
        family=family,
        profile="development",
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
    shapes = cast(NDArray[np.float64], case["shapes"])
    prior_mean_mass = np.asarray(
        [float(np.sum(shapes)) / float(case["rate"])],
        dtype=np.float64,
    )
    simulator_audit = _simulator_audit(
        artifact,
        masses=prior_mean_mass,
        seed=base_seed,
    )
    task_pass = bool(evaluation["scientific_pass"] and simulator_audit["pass"])
    return {
        "schema": SCHEMA,
        "source": {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        },
        "protocol_sha256": expected_protocol_sha256,
        "case_id": case_id,
        "quadrature_order": locked_order,
        "base_seed": base_seed,
        "artifact_sha256": artifact_sha256,
        "lock_sha256": common._sha256_path(  # pyright: ignore[reportPrivateUsage]
            lock_path
        ),
        "evaluation": evaluation,
        "simulator_audit": simulator_audit,
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
) -> dict[str, str]:
    """Publish report and completion marker, with the marker last."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['case_id']}__O{result['quadrature_order']}__base{result['base_seed']}"
    report_path = output_directory / f"{stem}.json"
    envelope = {
        "payload": result,
        "sha256": common._sha256_json(  # pyright: ignore[reportPrivateUsage]
            result
        ),
    }
    report_bytes = (
        common._canonical_json(envelope) + "\n"  # pyright: ignore[reportPrivateUsage]
    ).encode("utf-8")
    _atomic_write(report_path, report_bytes)
    marker = {
        "schema": MARKER_SCHEMA,
        "case_id": result["case_id"],
        "quadrature_order": result["quadrature_order"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": result["artifact_sha256"],
        "lock_sha256": result["lock_sha256"],
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    marker_path = output_directory / f"{stem}.complete.json"
    _atomic_write(
        marker_path,
        (
            common._canonical_json(marker) + "\n"  # pyright: ignore[reportPrivateUsage]
        ).encode("utf-8"),
    )
    return {
        "report": str(report_path),
        "completion_marker": str(marker_path),
    }


def main() -> None:
    """Run one G2 confirmation shard."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    result, _artifact_bytes = run_confirmation(
        regime_name=args.regime,
        family=cast(c1.Family, args.family),
        base_seed=args.base_seed,
        lock_path=args.lock,
        artifact_path=args.artifact,
        expected_source_revision=args.expected_source_revision,
        expected_driver_sha256=args.expected_driver_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    paths = _write_result(args.output_directory, result)
    print(
        common._canonical_json(  # pyright: ignore[reportPrivateUsage]
            {"task_pass": result["task_pass"], "paths": paths}
        )
    )


if __name__ == "__main__":
    main()
