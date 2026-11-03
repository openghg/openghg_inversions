"""Posterior convergence calculations without recipe, graph, or output objects."""

from __future__ import annotations

from collections.abc import Sequence

import arviz as az
import xarray as xr

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
