"""Stable staged check envelopes and the shared posterior convergence check."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import arviz as az
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
CONVERGENCE_CHECK_NAME = "sampler-convergence"
CHECK_SCHEMA_VERSION = 1


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
    summary = az.summary(
        idata["posterior"].to_dataset(),
        kind="diagnostics",
        fmt="xarray",
        round_to="none",
    )
    if "summary" in summary.dims:
        summary = summary.rename(summary="metric")
    destination = _stage_output_directory(output_dir)
    summary_path = _output_path(destination, None, "posterior-diagnostics.nc")
    reset_serialisation_multiindexes(summary).to_netcdf(summary_path)

    def finite_extreme(name: str, operation: str) -> tuple[float | None, str | None, list[str]]:
        candidates = []
        if name in summary:
            candidates.append((name, summary[name]))
        else:
            for variable, values in summary.data_vars.items():
                for dim in values.dims:
                    if dim in values.coords and name in values.coords[dim].values:
                        candidates.append((str(variable), values.sel({dim: name}, drop=True)))
                        break
        best: tuple[float, str] | None = None
        unassessable: list[str] = []
        for variable, values in candidates:
            array = np.asarray(values.values, dtype=float)
            finite = np.isfinite(array)
            for positions in np.argwhere(~finite):
                index = tuple(positions)
                labels = [
                    f"{dim}={values.coords[dim].values[position]}"
                    for dim, position in zip(values.dims, index)
                    if dim in values.coords
                ]
                unassessable.append(",".join([variable, *labels]))
            if not finite.any():
                continue
            masked = np.where(finite, array, -np.inf if operation == "max" else np.inf)
            flat_index = int(masked.argmax() if operation == "max" else masked.argmin())
            index = np.unravel_index(flat_index, array.shape)
            labels = [
                f"{dim}={values.coords[dim].values[position]}"
                for dim, position in zip(values.dims, index)
                if dim in values.coords
            ]
            candidate = (float(array[index]), ",".join([variable, *labels]))
            if best is None or (candidate[0] > best[0] if operation == "max" else candidate[0] < best[0]):
                best = candidate
        if best is None:
            return None, None, unassessable
        return best[0], best[1], unassessable

    rhat, rhat_variable, unassessable_rhat = finite_extreme("r_hat", "max")
    bulk_ess, bulk_ess_variable, unassessable_bulk_ess = finite_extreme("ess_bulk", "min")
    tail_ess, tail_ess_variable, unassessable_tail_ess = finite_extreme("ess_tail", "min")
    posterior_group = idata["posterior"].to_dataset() if "posterior" in idata.children else None
    sample_stats = idata["sample_stats"].to_dataset() if "sample_stats" in idata.children else None
    divergences_by_chain = (
        np.asarray(sample_stats["diverging"].sum("draw").values, dtype=int).tolist()
        if sample_stats is not None and "diverging" in sample_stats
        else None
    )
    divergences: int | None = sum(divergences_by_chain) if divergences_by_chain is not None else None

    measured = {
        "chains": posterior_group.sizes.get("chain") if posterior_group is not None else None,
        "draws_per_chain": posterior_group.sizes.get("draw") if posterior_group is not None else None,
        "max_rhat": rhat,
        "max_rhat_variable": rhat_variable,
        "unassessable_rhat": unassessable_rhat,
        "min_bulk_ess": bulk_ess,
        "min_bulk_ess_variable": bulk_ess_variable,
        "unassessable_bulk_ess": unassessable_bulk_ess,
        "min_tail_ess": tail_ess,
        "min_tail_ess_variable": tail_ess_variable,
        "unassessable_tail_ess": unassessable_tail_ess,
        "divergences": divergences,
        "divergences_by_chain": divergences_by_chain,
    }
    thresholds = {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "divergences": max_divergences,
    }
    assessed_names = ("max_rhat", "min_bulk_ess", "min_tail_ess", "divergences")
    partial = {
        "max_rhat": unassessable_rhat,
        "min_bulk_ess": unassessable_bulk_ess,
        "min_tail_ess": unassessable_tail_ess,
    }
    missing = [name for name in assessed_names if measured[name] is None or partial.get(name)]
    failed = (
        (rhat is not None and rhat > max_rhat)
        or (bulk_ess is not None and bulk_ess < min_bulk_ess)
        or (tail_ess is not None and tail_ess < min_tail_ess)
        or (divergences is not None and divergences > max_divergences)
    )
    status = "fail" if failed else "unknown" if missing else "pass"
    if failed:
        message = "Posterior convergence thresholds were exceeded."
        if missing:
            message += f" Also unavailable: {', '.join(missing)}."
    elif missing:
        message = f"Convergence metrics unavailable: {', '.join(missing)}."
    else:
        message = "Posterior convergence thresholds were met."
    result = _check_result(
        name=CONVERGENCE_CHECK_NAME,
        status=status,
        measured_values=measured,
        thresholds=thresholds,
        message=message,
        artifact_paths=[_artifact_path(summary_path)],
        stage=stage,
    )
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
