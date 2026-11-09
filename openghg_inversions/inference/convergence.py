"""ArviZ convergence assessment for retained posterior samples."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from .diagnostics import posterior_summary


def posterior_convergence_check(
    trace: xr.DataTree,
    *,
    variable_names: Sequence[str] | None = None,
    rhat_max: float | None = None,
    ess_min_ratio: float | None = None,
    ess_threshold: int | None = None,
    bfmi_threshold: float | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Assess retained posterior samples using ArviZ's diagnostic policy.

    Args:
        trace: Trace with ``chain`` and ``draw`` posterior dimensions and optional
            ``sample_stats``. Ordinary ``diverging`` and cached-amplitude
            ``sigma_nuts_diverging`` flags are combined; either flags a draw once.
        variable_names: Variables to assess for R-hat and effective sample size
            (ESS). Defaults to the JSON list in the trace's
            ``sampler_convergence_variables`` attribute, or all posterior variables
            when absent. An empty sequence leaves the assessment incomplete.
        rhat_max: Maximum acceptable R-hat; ``None`` uses the ArviZ default.
        ess_min_ratio: Minimum acceptable ratio of the smaller bulk/tail ESS to
            total retained samples across chains; ``None`` uses the ArviZ default.
        ess_threshold: Minimum acceptable bulk ESS, independently of the ratio
            check. ``None`` uses ArviZ's default of 100 times the number of chains.
        bfmi_threshold: Minimum acceptable energy Bayesian fraction of missing
            information (E-BFMI); ``None`` uses the ArviZ default.

    Returns:
        Detailed diagnostics from :func:`~openghg_inversions.inference.diagnostics.posterior_summary`
        and a plain report with ``status``, ``measured_values``, ``thresholds`` and
        ``message``. Status is ``fail`` when :func:`arviz.diagnose` reports a
        sampling problem; otherwise it is ``unknown`` for missing divergence
        flags, absent variables/draws, non-finite summary diagnostics or a single
        chain, and ``pass`` otherwise. The report records the ArviZ version,
        assessed variables, chain/draw counts, divergence count, unavailable
        diagnostics and explicit threshold overrides.

    Notes:
        Diagnostics are computed eagerly without modifying the trace. ArviZ also
        checks tree depth and E-BFMI when their sampler statistics are available;
        their absence alone does not make the assessment incomplete. The report
        contains no staged-workflow envelope and this function writes no files.
    """
    options = {
        name: value
        for name, value in {
            "rhat_max": rhat_max,
            "ess_min_ratio": ess_min_ratio,
            "ess_threshold": ess_threshold,
            "bfmi_threshold": bfmi_threshold,
        }.items()
        if value is not None
    }
    if variable_names is None and "sampler_convergence_variables" in trace.attrs:
        variable_names = json.loads(trace.attrs["sampler_convergence_variables"])
    posterior = trace["posterior"].to_dataset() if "posterior" in trace.children else xr.Dataset()
    assessed = list(variable_names) if variable_names is not None else list(posterior.data_vars)
    chains, draws = posterior.sizes.get("chain"), posterior.sizes.get("draw")
    summary = xr.Dataset()
    unavailable = []
    details: dict[str, Any] = {}
    has_errors = False

    stats = trace["sample_stats"].to_dataset() if "sample_stats" in trace.children else xr.Dataset()
    diverging = None
    for name in ("diverging", "sigma_nuts_diverging"):
        if name in stats:
            flags = stats[name].astype(bool)
            diverging = flags if diverging is None else diverging | flags
    if diverging is None:
        unavailable.append("divergence flags unavailable")
    else:
        # Either NUTS step can diverge; count each retained draw only once.
        stats = stats.assign(diverging=diverging)

    if assessed and draws:
        # diagnose returns only flagged values; retain the full summary for investigation and NaN checks.
        summary = posterior_summary(posterior, var_names=assessed)
        # ArviZ's chain-based ESS default requires coordinates on unindexed sample axes.
        diagnostic_posterior = posterior.assign_coords(
            {
                dim: np.arange(posterior.sizes[dim])
                for dim in ("chain", "draw")
                if dim in posterior.dims and dim not in posterior.coords
            }
        )
        diagnostic_trace = xr.DataTree.from_dict({"posterior": diagnostic_posterior, "sample_stats": stats})
        has_errors, details = az.diagnose(
            diagnostic_trace,
            var_names=assessed,
            show_diagnostics=False,
            return_diagnostics=True,
            **options,
        )
        # ArviZ's threshold comparisons do not flag NaN diagnostics as unavailable.
        unavailable.extend(
            f"non-finite diagnostics for {name}"
            for name, values in summary.data_vars.items()
            if not np.isfinite(values).all().item()
        )
    else:
        unavailable.append("no posterior variables or retained draws")
    if chains == 1:
        unavailable.append("between-chain convergence requires at least two chains")

    status = "fail" if has_errors else "unknown" if unavailable else "pass"
    message = {
        "fail": "ArviZ detected sampling problems.",
        "unknown": "Sampling convergence could not be fully assessed.",
        "pass": "ArviZ detected no sampling problems.",
    }[status]
    if unavailable:
        message += f" Assessment incomplete: {'; '.join(unavailable)}."
    return summary, {
        "status": status,
        "measured_values": {
            "assessed_variables": assessed,
            "chains": chains,
            "draws_per_chain": draws,
            "arviz_version": az.__version__,
            "arviz_has_errors": bool(has_errors),
            "divergences": details.get("divergent", {}).get("n_divergent"),
            "unavailable": unavailable,
        },
        "thresholds": {"policy": "arviz.diagnose defaults", **options},
        "message": message,
    }
