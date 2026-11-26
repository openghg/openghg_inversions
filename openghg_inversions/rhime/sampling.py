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
    return _sample_rhime_model(model_build_result, sampler, defer_deterministics=False)


def _sample_linked_rhime_model(
    model_build_result: RhimeModelBuildResult, sampler: RhimeSampler,
) -> xr.DataTree:
    """Sample a linked CO2/O2 recipe with deferred recording where supported."""
    return _sample_rhime_model(model_build_result, sampler, defer_deterministics=True)


def _sample_rhime_model(
    model_build_result: RhimeModelBuildResult,
    sampler: RhimeSampler,
    *,
    defer_deterministics: bool,
) -> xr.DataTree:
    """Run the recipe's sampler and record its total elapsed time."""
    model = model_build_result.model
    roles = model_build_result.output_contract.variable_roles
    sample = sampler.sample

    timing_start = timer_start()
    if defer_deterministics and getattr(sample, "__func__", None) is RhimeSampler.sample:
        idata = RhimeSampler._sample(sampler, model, variable_roles=roles, defer_deterministics=True)
    else:
        idata = sample(model, variable_roles=roles)

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
