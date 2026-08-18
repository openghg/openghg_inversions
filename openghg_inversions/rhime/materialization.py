"""Explicit eager boundaries between labelled RHIME inputs and model backends."""

from __future__ import annotations

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AGGREGATION_ERROR_SD,
    DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR,
    AggregationErrorMode,
    select_aggregation_error_mode,
)

__all__ = ["materialize_pymc_inputs"]

_MODEL_INPUT_VARIABLES = (
    "H",
    "H_bc",
    "mf",
    "mf_error",
    "min_error",
    "site_indicator",
)


def materialize_pymc_inputs(
    prepared: RhimePreparedInputs,
    *,
    aggregation_error_mode: AggregationErrorMode,
) -> xr.Dataset:
    """Materialize related PyMC arrays together without mutating preparation.

    Sparse chunk payloads are converted with :func:`to_dense`; model-owned
    arrays and their lazy auxiliary coordinates are computed in one shared
    Dask operation and installed in a shallow dataset copy. Dormant error
    representations and the canonical prepared artifact remain unchanged.
    """
    timing_start = timer_start()
    inv_inputs = prepared.inv_inputs
    selected_error_mode = select_aggregation_error_mode(inv_inputs, aggregation_error_mode)
    aggregation_names: tuple[str, ...]
    if selected_error_mode == "dense":
        aggregation_names = (AGGREGATION_ERROR_COVARIANCE, AGGREGATION_ERROR_SD)
    elif selected_error_mode == "low_rank":
        aggregation_names = (LOW_RANK_FACTOR, DIAGONAL_RESIDUAL_VARIANCE, AGGREGATION_ERROR_SD)
    elif selected_error_mode == "diagonal":
        aggregation_names = (AGGREGATION_ERROR_SD,)
    else:
        aggregation_names = ()
    names = [name for name in (*_MODEL_INPUT_VARIABLES, *aggregation_names) if name in inv_inputs]
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
