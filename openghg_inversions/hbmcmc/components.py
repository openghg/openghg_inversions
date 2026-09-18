"""Compatibility wrappers for HBMCMC model components."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_offset_component


def make_offset(
    site_indicator: np.ndarray,
    prior_args: dict,
    name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
    offset_freq: str | None = None,
) -> TensorVariable:
    """Create an offset inside a PyMC model.

    This compatibility wrapper keeps the historical import path while delegating
    to the new shared component implementation. ``offset_freq`` remains an
    ignored compatibility argument because this site-only interface has no
    observation times from which to derive periods.
    """
    del offset_freq
    observations = xr.DataArray(
        np.empty(site_indicator.size),
        dims=(output_dim,),
        coords={"site": (output_dim, site_indicator)},
    )

    return add_offset_component(
        observations,
        prior_args=prior_args,
        var_name=f"{name}_latent",
        output_name=name,
        output_dim=output_dim,
        drop_first=drop_first,
    )
