"""Explicit eager boundaries between labelled RHIME inputs and model backends."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.model_error import MinimumError
from .specs import AdditiveSigmaSettings, PollutionEventSettings

if TYPE_CHECKING:
    from .specs import RhimeModelSpec
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.priors import PriorArgs

__all__ = ["materialize_pymc_inputs"]


def materialize_pymc_inputs(
    prepared: RhimePreparedInputs,
    *,
    variable_names: Collection[str],
) -> xr.Dataset:
    """Materialize related PyMC arrays together without mutating preparation.

    ``variable_names`` comes from the concrete recipe and its selected
    components. Sparse chunk payloads are converted with :func:`to_dense`;
    those arrays and their lazy auxiliary coordinates are computed in one
    shared Dask operation and installed in a shallow dataset copy. Unselected
    prepared products and the canonical prepared artifact remain unchanged.

    Args:
        prepared: Borrowed backend-neutral prepared inputs.
        variable_names: Names selected by the concrete recipe and its active
            components.

    Returns:
        A shallow dataset copy in which selected arrays and their lazy
        coordinates have been materialized together.

    Raises:
        ValueError: If a selected variable is absent from the prepared data.
    """
    return _materialize_pymc_inputs_and_priors(prepared, variable_names=variable_names)[0]


def _materialize_pymc_inputs_and_priors(
    prepared: RhimePreparedInputs,
    *,
    variable_names: Collection[str],
    prior_parameters: Mapping[str, PriorArgs] | None = None,
) -> tuple[xr.Dataset, dict[str, PriorArgs]]:
    """Materialize selected inputs and prior parameters together without mutation.

    Args:
        prepared: Borrowed backend-neutral prepared inputs.
        variable_names: Prepared variables consumed by the concrete model.
            Their sparse payloads are densified; those values, lazy auxiliary
            coordinates and prior parameters share one Dask computation.
        prior_parameters: Prior parameter mappings keyed by model variable name,
            such as ``x`` or ``sigma``. Omission selects no prior parameters.

    Returns:
        A shallow Dataset copy retaining every prepared variable, with selected
        inputs and their coordinates materialized, and a new dictionary of
        prior parameter mappings under the supplied model-variable keys.
        Unselected Dataset variables remain borrowed and may remain lazy.
        Prior selection precedes zero-sensitivity pruning; model construction
        decides which selected priors actually enter the graph and its metadata.

    Raises:
        ValueError: If a selected variable is absent from the prepared data.
    """
    timing_start = timer_start()
    inv_inputs = prepared.inv_inputs
    names = list(dict.fromkeys(variable_names))
    missing = [name for name in names if name not in inv_inputs]
    if missing:
        raise ValueError(f"Selected PyMC component inputs are missing from prepared data: {missing!r}.")
    coordinate_names = sorted(
        {
            str(coordinate_name)
            for name in names
            for coordinate_name, coordinate in inv_inputs[name].coords.items()
            if isinstance(coordinate.data, DaskArray)
        }
    )
    priors = {name: {key: to_dense(value) if isinstance(value, xr.DataArray) else value
                     for key, value in params.items()}
              for name, params in (prior_parameters or {}).items()}
    *computed, computed_priors = dask_compute(
        *(to_dense(inv_inputs[name]).data for name in names),
        *(inv_inputs.coords[name].data for name in coordinate_names),
        priors,
    )
    dense_data = dict(zip(names, computed[: len(names)], strict=True))
    dense_coordinates = dict(zip(coordinate_names, computed[len(names) :], strict=True))
    variables = dict(inv_inputs.variables)
    for name, data in {**dense_data, **dense_coordinates}.items():
        variables[name] = variables[name].copy(deep=False, data=data)
    model_inputs = inv_inputs._replace(variables=variables)
    log_timing(
        "rhime.model_inputs_materialize",
        timer_seconds(timing_start),
        variables=names,
        coordinates=coordinate_names,
    )
    return model_inputs, computed_priors


def _minimum_error_input_names(
    inputs: xr.Dataset,
    model_spec: "RhimeModelSpec",
    *,
    materialized: bool = False,
) -> tuple[str, ...]:
    """Select retained operands only when the likelihood uses a floor."""
    likelihood = model_spec.likelihood
    enabled = isinstance(likelihood, PollutionEventSettings) or (
        isinstance(likelihood, AdditiveSigmaSettings) and likelihood.use_minimum_error_floor
    )
    if not enabled:
        return ()
    if materialized:
        return ("min_error",)
    if likelihood.min_error == "residual":
        return ("mf_mod", "bc_mod") if "bc_mod" in inputs else ("mf_mod",)
    # Scalars, site mappings and percentile floors need only mf/site/time,
    # already selected by every recipe's observation component.
    return ()


def _validate_builder_boundary_inputs(
    inputs: xr.Dataset, model_spec: "RhimeModelSpec", *, materializer: str,
) -> None:
    """Require resolved BC operands at the owning eager builder boundary."""
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


def _derive_model_owned_inputs(inputs: xr.Dataset, model_spec: "RhimeModelSpec") -> xr.Dataset:
    """Resolve eager boundary states and floors from retained observations."""
    result = inputs
    if _minimum_error_input_names(inputs, model_spec, materialized=True):
        likelihood = model_spec.likelihood
        floor = MinimumError.from_observations(
            inputs, likelihood.min_error, by_site=likelihood.min_error_options["by_site"],
        )
        result = result.assign(min_error=floor.values.variable)
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
