"""Backend-neutral preparation of operands owned by RHIME likelihoods."""

from __future__ import annotations

import json

from dask import is_dask_collection
import xarray as xr

from openghg_inversions.model_error import (
    MinimumError,
    minimum_error_input_names,
    minimum_error_source_policy,
    validate_minimum_error_policy,
)
from .specs import AdditiveSigmaSettings, LikelihoodSettings, PollutionEventSettings


def _floor_settings(
    likelihood: LikelihoodSettings | None,
) -> PollutionEventSettings | AdditiveSigmaSettings | None:
    """Select settings only when the likelihood consumes a floor."""
    if isinstance(likelihood, PollutionEventSettings):
        return likelihood
    if isinstance(likelihood, AdditiveSigmaSettings) and likelihood.use_minimum_error_floor:
        return likelihood
    return None


def likelihood_input_names(
    inputs: xr.Dataset, likelihood: LikelihoodSettings | None, *, materialized: bool = False,
) -> tuple[str, ...]:
    """Select likelihood-owned operands for numerical execution or building."""
    settings = _floor_settings(likelihood)
    if settings is None:
        return ()
    if materialized:
        return ("min_error",)
    return minimum_error_input_names(inputs, settings.min_error)


def prepare_likelihood_inputs(inputs: xr.Dataset, likelihood: LikelihoodSettings | None) -> xr.Dataset:
    """Resolve an enabled likelihood floor from retained eager observations."""
    settings = _floor_settings(likelihood)
    if settings is None:
        return inputs
    names = minimum_error_input_names(inputs, settings.min_error)
    if any(is_dask_collection(inputs[name]) for name in names):
        raise ValueError(
            "Model input resolution requires selected operands and coordinates to be materialized; "
            "use the concrete family materialize_*_inputs helper first."
        )
    floor = MinimumError.from_observations(
        inputs, settings.min_error, by_site=settings.min_error_options["by_site"],
    )
    floor.values.attrs["minimum_error_source_policy"] = json.dumps(
        minimum_error_source_policy(
            settings.min_error, floor.sites, by_site=settings.min_error_options["by_site"],
        ), sort_keys=True,
    )
    return inputs.assign(min_error=floor.values.variable)


def validate_likelihood_inputs(inputs: xr.Dataset, likelihood: LikelihoodSettings | None) -> None:
    """Check source policy only for the floor consumed by this likelihood."""
    settings = _floor_settings(likelihood)
    if settings is not None:
        validate_minimum_error_policy(
            inputs["min_error"], settings.min_error, by_site=settings.min_error_options["by_site"],
        )
