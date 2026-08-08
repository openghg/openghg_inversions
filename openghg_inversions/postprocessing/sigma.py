"""Reconstruct observation-aligned sigma values from inversion output."""

from __future__ import annotations

from typing import Any, cast

import xarray as xr

from openghg_inversions.sigma import SigmaAlignment


def reconstruct_sigma_aligned(
    trace: xr.DataTree,
    *,
    model_data: xr.Dataset | None = None,
) -> xr.DataArray:
    """Reconstruct posterior sigma values on the observation dimension.

    Args:
        trace: Inference data containing posterior ``sigma`` and, unless
            ``model_data`` is supplied, registered constant model data.
        model_data: Optional canonical sigma alignment data.

    Returns:
        Posterior sigma values indexed onto ``nmeasure``.

    Raises:
        AttributeError: If required inference-data groups are absent.
        KeyError: If posterior sigma or registered alignment data are absent.
        ValueError: If registered alignment data are invalid.
    """
    trace_data = cast(Any, trace)
    if model_data is None:
        model_data = cast(xr.Dataset, trace_data.constant_data)
    alignment = SigmaAlignment.from_model_data(model_data)
    return alignment.align(cast(xr.DataArray, trace_data.posterior["sigma"]))
