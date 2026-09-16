"""Posterior convergence diagnostics shared by RHIME execution paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

CONVERGENCE_CHECK_NAME = "sampler-convergence"
CHECK_SCHEMA_VERSION = 1


def posterior_convergence_check(
    idata: az.InferenceData,
    *,
    max_rhat: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_divergences: int = 0,
    stage: str = "posterior",
    artifact_paths: Sequence[str] = (),
    summary: xr.Dataset | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Calculate a detailed ArviZ summary and compact convergence check."""
    if summary is None:
        summary = az.summary(idata, kind="diagnostics", fmt="xarray")

    def finite_extreme(name: str, operation: str) -> tuple[float | None, str | None, list[str]]:
        candidates = []
        if name in summary:
            candidates.append((name, summary[name]))
        else:
            for variable, values in summary.data_vars.items():
                if "metric" in values.dims and name in values.coords["metric"]:
                    candidates.append((str(variable), values.sel(metric=name, drop=True)))
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
    posterior = getattr(idata, "posterior", None)
    sample_stats = getattr(idata, "sample_stats", None)
    chains = posterior.sizes.get("chain") if posterior is not None else None
    divergences_by_chain = (
        np.asarray(sample_stats["diverging"].sum("draw").values, dtype=int).tolist()
        if sample_stats is not None and "diverging" in sample_stats
        else None
    )
    if chains == 1:
        rhat = None
        rhat_variable = None
        if not unassessable_rhat:
            unassessable_rhat = ["between-chain convergence requires at least two chains"]

    measured = {
        "chains": chains,
        "draws_per_chain": posterior.sizes.get("draw") if posterior is not None else None,
        "max_rhat": rhat,
        "max_rhat_variable": rhat_variable,
        "unassessable_rhat": unassessable_rhat,
        "min_bulk_ess": bulk_ess,
        "min_bulk_ess_variable": bulk_ess_variable,
        "unassessable_bulk_ess": unassessable_bulk_ess,
        "min_tail_ess": tail_ess,
        "min_tail_ess_variable": tail_ess_variable,
        "unassessable_tail_ess": unassessable_tail_ess,
        "divergences": sum(divergences_by_chain) if divergences_by_chain is not None else None,
        "divergences_by_chain": divergences_by_chain,
    }
    thresholds = {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "divergences": max_divergences,
    }
    partial = {
        "max_rhat": unassessable_rhat,
        "min_bulk_ess": unassessable_bulk_ess,
        "min_tail_ess": unassessable_tail_ess,
    }
    assessed_names = ("max_rhat", "min_bulk_ess", "min_tail_ess", "divergences")
    missing = [name for name in assessed_names if measured[name] is None or partial.get(name)]
    failed = (
        (rhat is not None and rhat > max_rhat)
        or (bulk_ess is not None and bulk_ess < min_bulk_ess)
        or (tail_ess is not None and tail_ess < min_tail_ess)
        or (measured["divergences"] is not None and measured["divergences"] > max_divergences)
    )
    status = "fail" if failed else "unknown" if missing else "pass"
    if chains == 1:
        message = "Between-chain convergence is not assessable with one chain."
        if failed:
            message += " Other posterior convergence thresholds were exceeded."
        elif missing[1:]:
            message += f" Also unavailable: {', '.join(missing[1:])}."
    elif failed:
        message = "Posterior convergence thresholds were exceeded."
        if missing:
            message += f" Also unavailable: {', '.join(missing)}."
    elif missing:
        message = f"Convergence metrics unavailable: {', '.join(missing)}."
    else:
        message = "Posterior convergence thresholds were met."

    return summary, {
        "schema_version": CHECK_SCHEMA_VERSION,
        "name": CONVERGENCE_CHECK_NAME,
        "status": status,
        "producer": "openghg_inversions",
        "measured_values": measured,
        "thresholds": thresholds,
        "message": message,
        "artifact_paths": list(artifact_paths),
        "stage": stage,
    }
