"""Focused contracts for the CO2/O2 prepared-input replay boundary."""

from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
from dask import delayed
from dask.array import Array as DaskArray
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime import co2 as co2_public
from openghg_inversions.rhime.co2.co2_o2_runner import (
    _materialize_co2_o2_pymc_inputs,
    run_rhime_co2_o2_from_prepared_inputs,
)


def test_complete_recipe_name_is_reserved() -> None:
    assert not hasattr(co2_public, "run_rhime_co2_o2")
    assert hasattr(co2_public, "run_rhime_co2_o2_from_prepared_inputs")


def test_materializes_related_arrays_in_one_shared_graph_without_mutation() -> None:
    executions: list[str] = []

    @delayed
    def shared_values() -> np.ndarray:
        executions.append("shared")
        return np.arange(10, dtype=float)

    shared = da.from_delayed(shared_values(), shape=(10,), dtype=float)
    borrowed = tuple(
        xr.DataArray(shared[index : index + 2], dims=(f"axis_{index}",))
        for index in range(5)
    )
    original_data = tuple(array.data for array in borrowed)
    compute_graphs: list[object] = []

    with Callback(start=lambda graph: compute_graphs.append(graph)):
        materialized = _materialize_co2_o2_pymc_inputs(*borrowed)

    assert len(compute_graphs) == 1
    assert executions == ["shared"]
    for original, array, dense in zip(original_data, borrowed, materialized, strict=True):
        assert isinstance(original, DaskArray)
        assert array.data is original
        assert not isinstance(dense.data, DaskArray)


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_replay_rejects_nonpositive_or_nonfinite_independent_error(value: float) -> None:
    array = xr.DataArray([1.0], dims="observation")
    prepared = SimpleNamespace(
        observations=array,
        fixed_prior_contribution=array,
        co2_operator=array,
        o2_operator=array,
        aggregation_error=None,
        retained_prior=None,
    )

    with pytest.raises(ValueError, match="finite positive"):
        run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=array.copy(data=[value]),
        )
