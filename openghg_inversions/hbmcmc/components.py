"""Compatibility wrappers for HBMCMC model components."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pytensor.tensor import TensorVariable

from openghg_inversions.models.components import add_offset_component


def make_offset(
    site_indicator: np.ndarray,
    prior_args: dict,
    name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
    offset_freq: str | None = None,
    offset_freq_indicator: xr.DataArray | np.ndarray | None = None,
) -> TensorVariable:
    """Create an offset inside a PyMC model.

    This compatibility wrapper keeps the historical import path while delegating
    to the new shared component implementation.
    """
    site_indicator_da = xr.DataArray(site_indicator, dims=(output_dim,), name="site_indicator")

    # Previously `offset_freq` was ignored, and without adding another argument to this function,
    # we cannot make use of it
    if offset_freq_indicator is None:
        offset_freq = None

    return add_offset_component(
        site_indicator_da,
        prior_args=prior_args,
        offset_freq_indicator=offset_freq_indicator,
        offset_freq=offset_freq,
        var_name=f"{name}_latent",
        output_name=name,
        output_dim=output_dim,
        drop_first=drop_first,
    )
