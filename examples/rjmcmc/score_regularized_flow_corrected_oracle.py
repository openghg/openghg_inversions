#!/usr/bin/env python3
"""Build the observation-blind corrected tiny-root oracle bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_tiny_oracle as oracle,
)

SCHEMA = "rjmcmc-score-nle-corrected-oracle-bundle-v1"
SELECTED_CASE_ORDERS = {
    "near_gaussian__two_cell__root": (16, 32),
    "skewed__four_cell__root": (8, 12, 16),
}
LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT = 0.0025


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(compact.encode("ascii")).hexdigest()


def _pretty_json_bytes(payload: object) -> bytes:
    """Return the exact on-disk JSON representation for an oracle file."""
    return f"{_canonical_json(payload)}\n".encode("ascii")


def _atomic_json(path: Path, payload: object) -> str:
    """Create one JSON file atomically and return its exact file-byte digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing evidence: {path}")
    content = _pretty_json_bytes(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _publish_bundle(
    output_root: Path,
    bundle: dict[str, Any],
    source_git_revision: str,
) -> tuple[Path, Path]:
    """Publish one create-only bundle and bind its payload and file bytes."""
    report = output_root / "oracle" / "oracle_bundle.json"
    completion = output_root / "oracle" / "COMPLETE.json"
    for path in (report, completion):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace existing oracle evidence: {path}")
    without_sha = dict(bundle)
    payload_sha256 = without_sha.pop("sha256", None)
    if payload_sha256 != _sha256_json(without_sha):
        raise ValueError("oracle bundle canonical payload SHA-256 does not replay.")
    if bundle.get("source_git_revision") != source_git_revision:
        raise ValueError("oracle bundle source revision does not match publication.")
    report_file_sha256 = _atomic_json(report, bundle)
    if not bundle["pass"]:
        raise RuntimeError("corrected oracle bundle did not converge.")
    _atomic_json(
        completion,
        {
            "schema": SCHEMA,
            "source_git_revision": source_git_revision,
            "report_path": str(report),
            "oracle_bundle_payload_sha256": payload_sha256,
            "oracle_bundle_file_sha256": report_file_sha256,
            "completion_marker_published_last": True,
        },
    )
    return report, completion


def build_bundle(source_git_revision: str) -> dict[str, Any]:
    """Build all selected references and the independent boundary certificate."""
    started = time.perf_counter()
    selected: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for case_id, orders in SELECTED_CASE_ORDERS.items():
        ladder = [
            oracle.adaptive_log_total_summary(
                case_id,
                fraction_order=order,
            )
            for order in orders
        ]
        reference = ladder[-1]
        previous = ladder[-2]
        delta = abs(reference.log_evidence - previous.log_evidence)
        location_delta = (
            abs(reference.posterior_mean_total - previous.posterior_mean_total) / reference.posterior_sd_total
        )
        sd_delta = (
            abs(reference.posterior_sd_total - previous.posterior_sd_total) / reference.posterior_sd_total
        )
        endpoint_delta = (
            max(
                abs(reference.posterior_lower_0_025 - previous.posterior_lower_0_025),
                abs(reference.posterior_median - previous.posterior_median),
                abs(reference.posterior_upper_0_975 - previous.posterior_upper_0_975),
            )
            / reference.posterior_sd_total
        )
        case_checks = {
            "log_evidence_converged": (delta <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT),
            "posterior_mean_converged": location_delta <= 0.005,
            "posterior_sd_converged": sd_delta <= 0.002,
            "posterior_endpoints_converged": endpoint_delta <= 0.005,
            "represented_prior_mass": (reference.represented_prior_mass >= 1.0 - 1.0e-12),
            "represented_posterior_mass": (reference.represented_posterior_mass >= 1.0 - 1.0e-6),
            "posterior_mode_included": reference.mode_included,
            "scaled_quadrature_error_small": (reference.scaled_quadrature_error <= 1.0e-6),
        }
        case_pass = all(case_checks.values())
        selected[case_id] = {
            "order_ladder": [summary.payload() for summary in ladder],
            "reference": reference.payload(),
            "last_two_log_evidence_delta_nat": delta,
            "last_two_posterior_mean_delta_reference_sd": (location_delta),
            "last_two_posterior_sd_relative_delta": sd_delta,
            "last_two_posterior_endpoint_delta_reference_sd": (endpoint_delta),
            "checks": case_checks,
            "pass": case_pass,
        }
        checks[f"{case_id}__converged"] = case_pass
    boundary = oracle.boundary_oracle_certificate()
    selected[oracle.BOUNDARY_CASE_ID] = {
        "order_ladder": boundary["primary_order_ladder"],
        "reference": boundary["primary_order_ladder"][-1],
        "last_two_log_evidence_delta_nat": boundary["diagnostics"]["primary_log_evidence_delta_nat"],
        "pass": boundary["pass"],
    }
    checks["boundary_independent_certificate"] = bool(boundary["pass"])
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": oracle.definitions_sha256(),
        "selected_cases": selected,
        "boundary_independent_certificate": boundary,
        "checks": checks,
        "pass": all(checks.values()),
        "runtime_seconds": time.perf_counter() - started,
    }
    return {
        **without_sha,
        "sha256": _sha256_json(without_sha),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.source_git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in arguments.source_git_revision
    ):
        raise ValueError("source_git_revision must be a full lower-case Git SHA.")
    bundle = build_bundle(arguments.source_git_revision)
    _publish_bundle(
        arguments.output_root,
        bundle,
        arguments.source_git_revision,
    )


if __name__ == "__main__":
    main()
