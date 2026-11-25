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
    *,
    defer_deterministics: bool = False,
) -> xr.DataTree:
    """Sample a built RHIME graph at the named sampler boundary.

    Args:
        model_build_result: Concrete graph and semantic variable roles.
        sampler: Configured sampler used for posterior and predictive draws.
        defer_deterministics: Evaluate deterministics for the retained
            posterior and any saved warmup draws after PyMC returns. The default
            records them during sampling. Samplers that override
            :meth:`~openghg_inversions.inference.sampling.RhimeSampler.sample`
            retain their own recording behavior.

    Returns:
        Sampled posterior and predictive groups.
    """
    model = model_build_result.model
    roles = model_build_result.output_contract.variable_roles

    timing_start = timer_start()
    if defer_deterministics and type(sampler).sample is RhimeSampler.sample:
        idata = sampler.sample(model, variable_roles=roles, defer_deterministics=True)
    else:
        idata = sampler.sample(model, variable_roles=roles)

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
