"""RHIME build-result adapter for the shared inference layer.

The established sampler import remains an alias. Scientific recipes retain
control of graph construction and pass the completed graph to inference.
"""

from __future__ import annotations

import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.inference.sampling import (
    NutsSampler as NutsSampler,
    RhimeSampler as RhimeSampler,
)
from openghg_inversions.rhime.builders import RhimeModelBuildResult


def sample_rhime_model(
    model_build_result: RhimeModelBuildResult,
    sampler: RhimeSampler,
) -> xr.DataTree:
    """Sample a built RHIME graph at the named sampler boundary.

    Args:
        model_build_result: Concrete graph and semantic variable roles.
        sampler: Configured sampler used for posterior and predictive draws.

    Returns:
        Sampled posterior and predictive groups.
    """
    timing_start = timer_start()
    idata = sampler.sample(
        model_build_result.model,
        variable_roles=model_build_result.variable_roles,
    )
    log_timing(
        "rhime.sampler_total",
        timer_seconds(timing_start),
        draws=sampler.draws,
        burn=sampler.burn,
        tune=sampler.tune,
        chains=sampler.chains,
        nuts_sampler=sampler.nuts_sampler,
    )
    return idata
