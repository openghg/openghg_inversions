"""Stable staged check envelopes and the shared posterior convergence check."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping

from openghg_inversions.inference.diagnostics import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    posterior_convergence_check,
)
import numpy as np

from openghg_inversions.serialization import load_trace, reset_serialisation_multiindexes

from ._stage_artifacts import (
    artifact_path as _artifact_path,
    write_json as _write_json,
    _output_path,
    _stage_output_directory,
    reference,
)
from ._stage_authentication import _load_stage_manifest

PREPARATION_CHECK_NAME = "prior-predictive-readiness"

__all__ = [
    "CHECK_SCHEMA_VERSION",
    "CONVERGENCE_CHECK_NAME",
    "PREPARATION_CHECK_NAME",
    "diagnose_rhime_stage",
    "prior_predictive_check",
]


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


def prior_predictive_check(
    build: Callable[[], Any],
    *,
    draws: int,
    destination: Path,
    report: Path,
    stage: str,
) -> dict[str, Any]:
    """Draw prior predictions and write the evidence and a readiness report.

    The old report is invalidated before building or writing evidence, so a
    failed overwrite cannot leave an earlier successful readiness claim.

    Args:
        build: Builds the model to check; its readiness errors become a failed check.
        draws: Positive number of prior-predictive draws.
        destination: Existing check directory receiving the evidence and report.
        report: Resolved report path selected and checked by the caller.
        stage: Validated stage label.

    Returns:
        The readiness check written to the report.

    Raises:
        OSError: If writing the evidence or report fails.
    """
    report.unlink(missing_ok=True)
    import pymc as pm

    from openghg_inversions.serialization import save_trace

    try:
        built = build()
        with built.model:
            prior = pm.sample_prior_predictive(draws, built.model)
    except (KeyError, ValueError) as exc:
        non_finite, status, artifacts = None, "fail", []
        message = f"Prior-predictive readiness failed: {type(exc).__name__}: {exc}"
    else:
        concentration = built.output_contract.variable_roles.get("concentration")
        predictive = prior.children.get("prior_predictive")
        values = [np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars]
        non_finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in values)
        status = "fail"
        if predictive is None or concentration not in predictive:
            message = "Prior prediction is missing the declared concentration draws."
        elif predictive[concentration].size == 0:
            message = "Prior prediction contains no concentration draws."
        elif non_finite:
            message = f"Prior predictive contains {non_finite} non-finite values."
        else:
            status = "pass"
            message = f"Prior predictive produced {draws} finite draws."
        prior_path = destination / "prior-predictive.nc"
        save_trace(prior, prior_path)
        artifacts = [str(prior_path)]
    result = _check_result(
        name=PREPARATION_CHECK_NAME, status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0}, message=message, artifact_paths=artifacts, stage=stage,
    )
    _write_json(report, result)
    return result


def _check_draws(draws: Any) -> None:
    """Reject non-positive or non-integer prior-predictive draw counts."""
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError(f"Prior-predictive draws must be a positive integer; got {draws!r}.")


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
    posterior: str | Path | None = None,
    output_dir: str | Path,
    sample_manifest: str | Path | None = None,
    check_output: str | Path | None = None,
    max_rhat: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_divergences: int = 0,
    stage: str = "posterior",
) -> dict[str, Any]:
    """Calculate convergence from a posterior path or current sample manifest.

    An optional sample manifest supplies descriptive run context, without
    authenticating the posterior file. When ``posterior`` is omitted, resolve
    its reference beside the sample manifest. Diagnosis needs no preparation,
    model construction or inference; explicit posterior paths take precedence.
    A draw diverges when either its ordinary or cached-amplitude NUTS flag is
    true. Missing both flags leaves divergence assessment unknown.
    """
    stage = _check_stage(stage)
    _validate_diagnostic_thresholds(
        max_rhat=max_rhat,
        min_bulk_ess=min_bulk_ess,
        min_tail_ess=min_tail_ess,
        max_divergences=max_divergences,
    )
    sample_contract = None
    if sample_manifest is not None:
        manifest_path, sample_contract = _load_stage_manifest(sample_manifest, stage="sample")
        if posterior is None:
            posterior = reference(manifest_path, sample_contract.get("posterior"))
    if posterior is None:
        raise ValueError("Pass a posterior path or a current sample manifest containing its reference.")
    posterior_path = Path(posterior).resolve()
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
    _write_json(_output_path(destination, check_output, "sampler-convergence.json"), result)
    return result
