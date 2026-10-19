"""Posterior convergence calculations without recipe, graph, or output objects."""

import arviz as az
import xarray as xr


def posterior_summary(posterior: xr.Dataset) -> xr.Dataset:
    """Calculate unrounded ArviZ diagnostics on the original chain/draw axes.

    Args:
        posterior: Posterior variables with labelled ``chain`` and ``draw``
            dimensions. Samples are not pooled or modified before assessment.

    Returns:
        ArviZ diagnostic values with a ``metric`` axis. Thresholds, product
        names, missing-value policy and check-artifact writing belong to callers.
    """
    result = az.summary(posterior, kind="diagnostics", fmt="xarray", round_to="none")
    if "summary" in result.dims:
        result = result.rename(summary="metric")
    return result
