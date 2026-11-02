"""Reconstruct observation-aligned sigma values from inversion output."""

from __future__ import annotations

import xarray as xr

from openghg_inversions.postprocessing.inversion_output import trace_group
from openghg_inversions.sigma import SigmaAlignment


def reconstruct_sigma_aligned(
    trace: xr.DataTree,
    *,
    model_data: xr.Dataset | None = None,
) -> xr.DataArray:
    """Reconstruct posterior sigma values on the observation dimension.

    Args:
        trace: Trace tree containing posterior ``sigma`` and, unless
            ``model_data`` is supplied, registered constant model data.
        model_data: Optional canonical sigma alignment data.

    Returns:
        Posterior sigma values indexed onto ``nmeasure``.

    Raises:
        KeyError: If required trace groups, posterior sigma, or registered
            alignment data are absent.
        ValueError: If registered alignment data are invalid.
    """
    if model_data is None:
        model_data = trace_group(trace, "constant_data")
    alignment = SigmaAlignment.from_model_data(model_data)
    return alignment.align(trace_group(trace, "posterior")["sigma"])
