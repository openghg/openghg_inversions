"""Backend-neutral resolution and policy checks for final RHIME operands.

Observation-error standard deviation ``mf_error`` is already prepared and is
carried through unchanged. Likelihood preparation calculates ``min_error`` as
the selected likelihood's minimum total-error floor.
"""

from __future__ import annotations

from dask import is_dask_collection
import pandas as pd
import xarray as xr

from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.inversion_inputs import normalise_frequency
from .likelihood_inputs import prepare_likelihood_inputs, validate_likelihood_inputs
from .specs import RhimeModelSpec

__all__ = ["resolve_model_inputs"]


def validate_model_inputs(
    inputs: xr.Dataset, model_spec: RhimeModelSpec, *, materializer: str,
) -> None:
    """Validate consumed likelihood and boundary operands before model building."""
    validate_likelihood_inputs(inputs, model_spec.likelihood)
    _validate_boundary_inputs(inputs, model_spec, materializer=materializer)


def _validate_boundary_inputs(
    inputs: xr.Dataset, model_spec: RhimeModelSpec, *, materializer: str,
) -> None:
    """Check model-owned BC expansion and boundary-period policy."""
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
    Observation-error standard deviation ``mf_error`` is carried through unchanged;
    likelihood preparation calculates the minimum total-error floor ``min_error``
    when selected by the likelihood.

    Args:
        inputs: Borrowed Dataset with consumed operands and coordinates eager.
        model_spec: Scientific choices for BC periods and the likelihood floor.

    Returns:
        A Dataset with final enabled BC/floor operands and source-policy attrs.
        When BCs are enabled, ``H_bc`` uses a ``bc_region`` MultiIndex with levels
        ``bc_curtain`` and ``bc_period``; other variables and diagnostics on the
        raw BC axis are preserved under ``raw_bc_region``. Unselected variables
        retain their borrowed payloads.

    Raises:
        ValueError: If consumed operands or coordinates are lazy, BC sensitivity
            is already expanded, or floor inputs or values are invalid.
    """
    result = prepare_likelihood_inputs(inputs, model_spec.likelihood)
    if model_spec.use_bc:
        raw = inputs["H_bc"]
        if is_dask_collection(raw):
            raise ValueError(
                "Model input resolution requires selected operands and coordinates to be materialized; "
                "use the concrete family materialize_*_inputs helper first."
            )
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
