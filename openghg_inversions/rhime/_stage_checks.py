"""Stable staged check envelopes and the shared posterior convergence check."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from openghg_inversions.inference.diagnostics import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME as CONVERGENCE_CHECK_NAME,
    posterior_convergence_check,
)
import numpy as np

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.serialization import load_trace, reset_serialisation_multiindexes

from ._stage_artifacts import (
    artifact_path as _artifact_path,
    file_identity as _file_identity,
    write_json as _write_json,
    _output_path,
    _stage_output_directory,
)
from ._stage_authentication import _verify_sample_manifest

PREPARATION_CHECK_NAME = "prior-predictive-readiness"


def _check_result(
    *,
    name: str,
    status: str,
    measured_values: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    message: str,
    artifact_paths: list[str],
    stage: str,
) -> dict[str, Any]:
    stage = _check_stage(stage)
    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "name": name,
        "status": status,
        "producer": "openghg_inversions",
        "measured_values": dict(measured_values),
        "thresholds": dict(thresholds),
        "message": message,
        "artifact_paths": artifact_paths,
        "stage": stage,
    }


def _check_stage(stage: str) -> str:
    """Validate the non-empty stage label required by OGR CheckResult v1."""
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("Scientific check stage must be a non-empty string.")
    return stage.strip()


def _validate_diagnostic_thresholds(
    *,
    max_rhat: float,
    min_bulk_ess: float,
    min_tail_ess: float,
    max_divergences: int,
) -> None:
    """Reject thresholds that cannot define a valid convergence policy."""
    for name, value, minimum in (
        ("max_rhat", max_rhat, 1.0),
        ("min_bulk_ess", min_bulk_ess, 0.0),
        ("min_tail_ess", min_tail_ess, 0.0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < minimum
        ):
            raise ValueError(f"Diagnostic threshold {name} must be finite and at least {minimum:g}.")
    if isinstance(max_divergences, bool) or not isinstance(max_divergences, Integral) or max_divergences < 0:
        raise ValueError("Diagnostic threshold max_divergences must be a non-negative integer.")


def diagnose_rhime_stage(
    *,
    posterior: str | Path,
    output_dir: str | Path,
    sample_manifest: str | Path | None = None,
    check_output: str | Path | None = None,
    max_rhat: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_divergences: int = 0,
    stage: str = "posterior",
) -> dict[str, Any]:
    """Calculate the stable posterior convergence check."""
    stage = _check_stage(stage)
    _validate_diagnostic_thresholds(
        max_rhat=max_rhat,
        min_bulk_ess=min_bulk_ess,
        min_tail_ess=min_tail_ess,
        max_divergences=max_divergences,
    )
    posterior_path = Path(posterior).resolve()
    sample_contract = None
    if sample_manifest is not None:
        sample_contract = _verify_sample_manifest(sample_manifest, posterior=posterior_path)
    idata = load_trace(posterior_path)
    destination = _stage_output_directory(output_dir)
    summary_path = _output_path(destination, None, "posterior-diagnostics.nc")
    summary, result = posterior_convergence_check(
        idata,
        max_rhat=max_rhat,
        min_bulk_ess=min_bulk_ess,
        min_tail_ess=min_tail_ess,
        max_divergences=max_divergences,
        stage=stage,
        artifact_paths=[_artifact_path(summary_path)],
    )
    reset_serialisation_multiindexes(summary).to_netcdf(summary_path)
    check_path = _write_json(_output_path(destination, check_output, "sampler-convergence.json"), result)
    if (
        sample_contract is not None
        and sample_contract.get("effective_configuration", {}).get("model") == "co2"
    ):
        _write_json(
            destination / "diagnose-manifest.json",
            {
                "schema_version": 1,
                "producer": "openghg_inversions",
                "stage": "diagnose",
                "configuration_identity": sample_contract["configuration_identity"],
                "ogi": installed_ogi_provenance(),
                "input_identities": {"posterior": _file_identity(posterior_path)},
                "artifacts": {
                    "diagnostics": _artifact_path(summary_path),
                    "check": _artifact_path(check_path),
                },
                "artifact_identities": {
                    "diagnostics": _file_identity(summary_path),
                    "check": _file_identity(check_path),
                },
            },
        )
    return result
