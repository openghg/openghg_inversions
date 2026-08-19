"""Explicit eager boundaries between labelled RHIME inputs and model backends."""

from __future__ import annotations

from collections.abc import Collection

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_data import RhimePreparedInputs

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
    computed = dask_compute(
        *(to_dense(inv_inputs[name]).data for name in names),
        *(inv_inputs.coords[name].data for name in coordinate_names),
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
    return model_inputs
