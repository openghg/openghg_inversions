"""Shared normalization helpers for retained ArviZ sampling results.

This module contains sampler-independent operations applied after burn slicing
and before predictive groups are attached. Keeping this boundary neutral lets
the modern RHIME and fixed-basis compatibility samplers share identical draw
coordinate and metadata behavior.

``_reset_retained_draws`` mutates the supplied ``InferenceData``: every
draw-bearing dataset is relabelled with consecutive zero-based draws, ``burn``
is recorded on the root and those groups, and groups without a draw dimension
are unchanged.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr


def _reset_retained_draws(idata: az.InferenceData, *, burn: int) -> az.InferenceData:
    """Relabel retained draws and preserve the discarded burn-in count.

    Args:
        idata: Inference data whose draw-bearing groups are relabelled in place.
        burn: Number of discarded burn-in draws to record in metadata.

    Returns:
        The mutated inference data, with each draw coordinate reset to
        consecutive zero-based integers and burn stored on the root and
        draw-bearing groups.
    """
    idata.attrs["burn"] = burn
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if not isinstance(group, xr.Dataset) or "draw" not in group.dims:
            continue
        group = group.assign_coords(draw=np.arange(group.sizes["draw"]))
        group.attrs["burn"] = burn
        setattr(idata, group_name, group)
    return idata
