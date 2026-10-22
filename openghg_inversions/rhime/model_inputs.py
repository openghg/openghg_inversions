"""Backend-neutral resolution and policy checks for final RHIME operands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import cast

from dask import is_dask_collection
import pandas as pd
import xarray as xr

from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.inversion_inputs import normalise_frequency
from openghg_inversions.model_error import MinimumError
from .specs import AdditiveSigmaSettings, LikelihoodSettings, PollutionEventSettings, RhimeModelSpec

__all__ = ["resolve_model_inputs"]


def minimum_error_input_names(
    inputs: xr.Dataset, likelihood: LikelihoodSettings | None, *, materialized: bool = False,
) -> tuple[str, ...]:
    """Select only operands consumed by the selected likelihood's floor."""
    enabled = isinstance(likelihood, PollutionEventSettings) or (
        isinstance(likelihood, AdditiveSigmaSettings) and likelihood.use_minimum_error_floor
    )
    if not enabled:
        return ()
    if materialized:
        return ("min_error",)
    likelihood = cast(PollutionEventSettings | AdditiveSigmaSettings, likelihood)
    if likelihood.min_error == "residual":
        return ("mf_mod", "bc_mod") if "bc_mod" in inputs else ("mf_mod",)
    return ()


def _minimum_error_policy(
    likelihood: PollutionEventSettings | AdditiveSigmaSettings, sites: Sequence[str],
) -> dict[str, object]:
    """Describe just the choices that produced a consumed floor operand."""
    value = likelihood.min_error
    if isinstance(value, Mapping):
        return {"sites": {site: value.get(site) for site in sites}}
    if isinstance(value, str):
        policy: dict[str, object] = {"method": value}
        if value == "residual":
            policy["by_site"] = likelihood.min_error_options["by_site"]
        return policy
    return {"scalar": value}


def resolve_minimum_error(inputs: xr.Dataset, likelihood: LikelihoodSettings | None) -> xr.Dataset:
    """Calculate an enabled floor from retained eager observations."""
    if not minimum_error_input_names(inputs, likelihood, materialized=True):
        return inputs
    likelihood = cast(PollutionEventSettings | AdditiveSigmaSettings, likelihood)
    floor = MinimumError.from_observations(
        inputs, likelihood.min_error, by_site=likelihood.min_error_options["by_site"],
    )
    floor.values.attrs["minimum_error_source_policy"] = json.dumps(
        _minimum_error_policy(likelihood, floor.sites), sort_keys=True,
    )
    return inputs.assign(min_error=floor.values.variable)


def validate_minimum_error_policy(inputs: xr.Dataset, likelihood: LikelihoodSettings | None) -> None:
    """Reject changed producer policies without recomputing a consumed floor.

    Independently supplied final arrays without source-policy metadata remain
    authoritative. Mapping comparison covers retained sites only.
    """
    if not minimum_error_input_names(inputs, likelihood, materialized=True):
        return
    source = inputs["min_error"].attrs.get("minimum_error_source_policy")
    if source is None:
        return
    policy = json.loads(source)
    likelihood = cast(PollutionEventSettings | AdditiveSigmaSettings, likelihood)
    if policy != _minimum_error_policy(likelihood, tuple(policy.get("sites", {}))):
        raise ValueError(
            "min_error was resolved with a different minimum-error policy; "
            "materialize the prepared inputs again with the requested likelihood settings."
        )


def _validate_builder_boundary_inputs(
    inputs: xr.Dataset, model_spec: RhimeModelSpec, *, materializer: str,
) -> None:
    """Check policies only for consumed, producer-resolved operands."""
    validate_minimum_error_policy(inputs, model_spec.likelihood)
    if not model_spec.use_bc:
        return
    boundary = inputs["H_bc"]
    preparation = boundary.attrs.get("boundary_sensitivity_preparation")
    if preparation == "raw-spatial" or (
        model_spec.bc_freq is not None
        and "bc_period" not in boundary.coords
        and preparation != "sampled-period-expansion"
    ):
        raise ValueError(
            "H_bc requires model-owned boundary alignment before building. "
            f"Call {materializer}(prepared, model_spec) even for eager raw inputs; "
            "generic materialize_pymc_inputs only densifies inputs and does not resolve BC periods."
        )
    if "bc_frequency" not in boundary.attrs:
        return
    source_frequency = boundary.attrs["bc_frequency"]
    source_frequency = normalise_frequency(None if source_frequency == "none" else source_frequency)
    frequency = normalise_frequency(model_spec.bc_freq)
    mismatch = source_frequency != frequency
    if not mismatch and isinstance(frequency, pd.Timedelta):
        # The producer's default anchor is the earliest retained observation.
        # Lazy coordinates are rejected by the builder's eager-input check.
        anchor = model_spec.bc_freq_anchor
        if anchor is None and not is_dask_collection(inputs.time):
            anchor = inputs.time.values.min()
        if anchor is not None:
            mismatch = pd.Timestamp(boundary.attrs["bc_anchor_time"]) != pd.Timestamp(anchor)
    if mismatch:
        raise ValueError(
            "H_bc was resolved with a different boundary-period policy (bc_freq/bc_freq_anchor); "
            f"call {materializer}(prepared, model_spec) again with the requested settings."
        )


def resolve_model_inputs(inputs: xr.Dataset, model_spec: RhimeModelSpec) -> xr.Dataset:
    """Resolve final boundary sensitivities and likelihood floors from eager inputs.

    This backend-neutral step performs no graph construction or materialization.
    Pass a Dataset whose selected observations, auxiliary coordinates and raw BC
    operands are already eager. Unselected arrays remain borrowed, including lazy
    diagnostics. The returned Dataset carries final ``H_bc`` and ``min_error``
    operands with their source policies, for analysis or a concrete model builder.
    Boundary expansion requires raw spatial ``H_bc`` when BCs are enabled.

    Args:
        inputs: Borrowed Dataset with consumed operands and coordinates eager.
        model_spec: Scientific choices for BC periods and the likelihood floor.

    Returns:
        A Dataset with final enabled BC/floor operands and source-policy attrs.
        Unselected variables retain their borrowed payloads.

    Raises:
        ValueError: If consumed operands or coordinates are lazy, BC sensitivity
            is already expanded, or floor inputs or values are invalid.
    """
    selected = ["H_bc"] if model_spec.use_bc else []
    if minimum_error_input_names(inputs, model_spec.likelihood, materialized=True):
        selected.extend(("mf", *minimum_error_input_names(inputs, model_spec.likelihood)))
    if any(is_dask_collection(inputs[name]) for name in selected):
        raise ValueError("Model input resolution requires selected operands and coordinates to be materialized; "
            "use the concrete family materialize_*_inputs helper first.")
    result = resolve_minimum_error(inputs, model_spec.likelihood)
    if model_spec.use_bc:
        raw = inputs["H_bc"]
        if (
            raw.attrs.get("boundary_sensitivity_preparation") == "sampled-period-expansion"
            or "bc_period" in raw.coords
        ):
            raise ValueError("Model boundary alignment requires raw spatial H_bc; prepare the inputs again.")
        boundary = BoundaryAlignment.prepare(
            raw, frequency=model_spec.bc_freq, anchor_time=model_spec.bc_freq_anchor,
        ).data
        # Linear model construction owns zero-state pruning. The model state
        # axis differs from the raw spatial axis, which diagnostics may share.
        # Keep those variables and labels under an explicit raw namespace.
        result = result.drop_vars("H_bc")
        if "bc_region" in result.dims:
            result = result.rename_dims(bc_region="raw_bc_region")
            if "bc_region" in result.coords:
                result = result.rename_vars(bc_region="raw_bc_region")
        result["H_bc"] = xr.DataArray(
            boundary.data, dims=boundary.dims, attrs=boundary.attrs,
            coords=xr.Coordinates.from_pandas_multiindex(boundary.indexes["bc_region"], "bc_region"),
        )
    return result
