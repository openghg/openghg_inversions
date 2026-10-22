"""Explicit eager boundaries between labelled RHIME inputs and model backends."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
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
