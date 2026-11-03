"""Posterior convergence calculations without recipe, graph, or output objects."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

CONVERGENCE_CHECK_NAME = "sampler-convergence"
CHECK_SCHEMA_VERSION = 1


def posterior_summary(
    posterior: xr.Dataset,
    *,
    var_names: Sequence[str] | None = None,
) -> xr.Dataset:
    """Calculate unrounded ArviZ diagnostics on the original chain/draw axes.

    Args:
        posterior: Posterior variables with labelled ``chain`` and ``draw``
            dimensions. Samples are not pooled or modified before assessment.
        var_names: Optional posterior variables to assess.

    Returns:
        ArviZ diagnostic values with a ``metric`` axis. Thresholds, product
        names, missing-value policy and check-artifact writing belong to callers.
    """
    summary_kwargs = {"kind": "diagnostics", "fmt": "xarray", "round_to": "none"}
    if var_names is not None:
        summary_kwargs["var_names"] = list(var_names)
    result = az.summary(posterior, **summary_kwargs)
    if "summary" in result.dims:
        result = result.rename(summary="metric")
    return result


def posterior_convergence_check(
    trace: xr.DataTree,
    *,
    max_rhat: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_divergences: int = 0,
    stage: str = "posterior",
    artifact_paths: Sequence[str] = (),
    summary: xr.Dataset | None = None,
    variable_names: Sequence[str] | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Summarize posterior diagnostics and assess convergence thresholds.

    Args:
        trace: Trace whose ``posterior`` variables have ``chain`` and ``draw``
            dimensions. Optional ``sample_stats`` supplies ``diverging`` and/or
            ``sigma_nuts_diverging`` flags; a draw flagged by either counts once.
        max_rhat: Largest passing R-hat value.
        min_bulk_ess: Smallest passing bulk effective sample size.
        min_tail_ess: Smallest passing tail effective sample size.
        max_divergences: Largest passing total number of divergent draws.
        stage: Stage label recorded in the check result.
        artifact_paths: Artifact paths recorded in the result; no files are written.
        summary: Precomputed diagnostics to assess instead of calling
            :func:`posterior_summary`. Accepts variables indexed by a ``metric``
            coordinate or diagnostic variables named ``r_hat``, ``ess_bulk`` and
            ``ess_tail``. Used as supplied, without filtering by ``variable_names``.
        variable_names: Posterior variables to summarize and record as assessed.
            Defaults to the JSON list in the trace's
            ``sampler_convergence_variables`` attribute, or all posterior variables
            when that attribute is absent. An empty sequence skips summary
            calculation; it does not suppress a supplied summary.

    Returns:
        The diagnostic summary and a JSON-compatible CheckResult dictionary
        containing measurements, thresholds, status and a message. Measurements
        include the largest finite R-hat, smallest finite effective sample sizes,
        their variable/coordinate labels, non-finite entries and divergence counts.
        Status is ``fail`` when any measured threshold is exceeded, otherwise
        ``unknown`` when diagnostics are missing or contain non-finite entries,
        and ``pass`` otherwise. R-hat is unassessable with one chain. Values equal
        to their thresholds pass.

    Notes:
        Summary diagnostics and divergence counts are read eagerly. The trace and
        any supplied summary are not modified.
    """
    # Prefer the saved latent-variable scope; legacy traces assess all posterior variables.
    if variable_names is None and "sampler_convergence_variables" in trace.attrs:
        variable_names = json.loads(trace.attrs["sampler_convergence_variables"])

    posterior = trace["posterior"].to_dataset() if "posterior" in trace.children else None
    sample_stats = trace["sample_stats"].to_dataset() if "sample_stats" in trace.children else None
    if summary is None:
        if posterior is None or (variable_names is not None and not variable_names):
            summary = xr.Dataset()
        else:
            summary = posterior_summary(posterior, var_names=variable_names)

    # Reduce each diagnostic to its worst finite value without hiding unavailable entries.
    chains = posterior.sizes.get("chain") if posterior is not None else None
    if chains == 1:
        rhat, rhat_variable = None, None
        unassessable_rhat = ["between-chain convergence requires at least two chains"]
    else:
        rhat, rhat_variable, unassessable_rhat = _finite_extreme(summary, "r_hat", "max")
    bulk_ess, bulk_ess_variable, unassessable_bulk_ess = _finite_extreme(summary, "ess_bulk", "min")
    tail_ess, tail_ess_variable, unassessable_tail_ess = _finite_extreme(summary, "ess_tail", "min")

    # A retained draw counts once if either NUTS step reports a divergence.
    diverging = None
    if sample_stats is not None:
        for name in ("diverging", "sigma_nuts_diverging"):
            if name in sample_stats:
                flags = sample_stats[name].astype(bool)
                diverging = flags if diverging is None else diverging | flags
    divergences_by_chain = (
        np.asarray(diverging.sum("draw").values, dtype=int).tolist()
        if diverging is not None
        else None
    )

    measured = {
        "assessed_variables": (
            list(variable_names) if variable_names is not None else [str(name) for name in summary.data_vars]
        ),
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

    # A finite failure takes precedence over missing or partially non-finite metrics.
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


def _finite_extreme(summary: xr.Dataset, name: str, operation: str) -> tuple[float | None, str | None, list[str]]:
    """Find a finite extreme and label it along with every non-finite entry."""
    # Accept metric variables as well as ArviZ variables with a metric axis.
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
            unassessable.append(_variable_label(variable, values, index))
        if not finite.any():
            continue
        # Exclude non-finite entries from the extreme, but retain their labels above.
        masked = np.where(finite, array, -np.inf if operation == "max" else np.inf)
        flat_index = int(masked.argmax() if operation == "max" else masked.argmin())
        index = np.unravel_index(flat_index, array.shape)
        candidate = (float(array[index]), _variable_label(variable, values, index))
        if best is None or (candidate[0] > best[0] if operation == "max" else candidate[0] < best[0]):
            best = candidate
    if best is None:
        return None, None, unassessable
    return best[0], best[1], unassessable


def _variable_label(variable: str, values: xr.DataArray, index: tuple[int, ...]) -> str:
    """Identify a scalar or indexed parameter using its available coordinate labels."""
    labels = [
        f"{dim}={values.coords[dim].values[position]}"
        for dim, position in zip(values.dims, index)
        if dim in values.coords
    ]
    return ",".join([variable, *labels])
