"""Focused contracts for the CO2/O2 prepared-input replay boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import arviz as az
import dask.array as da
from dask import delayed
from dask.array import Array as DaskArray
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.rhime import co2 as co2_public
from openghg_inversions.rhime.co2 import co2_o2_runner
from openghg_inversions.rhime.co2.co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES,
    _materialize_co2_o2_pymc_inputs,
    run_rhime_co2_o2_from_prepared_inputs,
)


def test_complete_recipe_name_is_reserved() -> None:
    assert not hasattr(co2_public, "run_rhime_co2_o2")
    assert hasattr(co2_public, "run_rhime_co2_o2_from_prepared_inputs")


def test_modelled_concentration_is_not_labelled_as_pollution_only() -> None:
    assert "pollution_concentration" not in _CO2_O2_VARIABLE_ROLES
    assert _CO2_O2_VARIABLE_ROLES["modelled_concentration"] == "modelled_concentration"
    assert _CO2_O2_VARIABLE_ROLES["emissions_sensitivity"] == "co2_o2_operator"
    assert _CO2_O2_VARIABLE_ROLES["flux_contribution"] == "co2_o2_flux_contribution"


def _prepared_stub(array: xr.DataArray) -> SimpleNamespace:
    return SimpleNamespace(
        observations=array,
        fixed_prior_contribution=array,
        co2_operator=array,
        o2_operator=array,
        aggregation_error=None,
        retained_prior=None,
        provenance={},
    )


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


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, "bad", 1.0 + 0.0j])
def test_replay_rejects_invalid_independent_error(value: object) -> None:
    array = xr.DataArray(
        [1.0],
        dims="observation",
        coords={
            "observation": [0],
            "observation_units": ("observation", ["ppm"]),
        },
    )
    prepared = _prepared_stub(array)

    with pytest.raises(ValueError, match="finite positive"):
        run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=array.copy(data=[value]),
        )


def test_replay_rejects_mismatched_independent_error_labels_or_units() -> None:
    observations = xr.DataArray(
        [1.0],
        dims="observation",
        coords={
            "observation": [0],
            "observation_units": ("observation", ["ppm"]),
        },
    )
    prepared = _prepared_stub(observations)

    with pytest.raises(ValueError, match="dimension and labels"):
        run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=observations.assign_coords(observation=[1]),
        )

    with pytest.raises(ValueError, match="observation_units"):
        run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=observations.assign_coords(
                observation_units=("observation", ["per meg"])
            ),
        )


def test_replay_rejects_stale_independent_error_multiindex_level_names() -> None:
    observation_index = pd.MultiIndex.from_tuples(
        [("co2", "c1")],
        names=("species", "channel_observation"),
    )
    observations = xr.DataArray(
        [1.0],
        dims="observation",
        coords=xr.Coordinates.from_pandas_multiindex(
            observation_index,
            "observation",
        ),
    ).assign_coords(observation_units=("observation", ["ppm"]))
    stale_error = xr.DataArray(
        [1.0],
        dims="observation",
        coords=xr.Coordinates.from_pandas_multiindex(
            observation_index.set_names(("stale_species", "stale_observation")),
            "observation",
        ),
    ).assign_coords(observation_units=("observation", ["ppm"]))

    with pytest.raises(ValueError, match="dimension and labels"):
        run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=_prepared_stub(observations),
            independent_error_sd=stale_error,
        )


def test_replay_materializes_payloads_and_auxiliary_units_in_one_graph(monkeypatch) -> None:
    executions: list[str] = []

    @delayed
    def shared_values() -> np.ndarray:
        executions.append("values")
        return np.array([1.0, 2.0])

    @delayed
    def shared_units() -> np.ndarray:
        executions.append("units")
        return np.array(["ppm", "per meg"])

    values = da.from_delayed(shared_values(), shape=(2,), dtype=float)
    units = da.from_delayed(shared_units(), shape=(2,), dtype="<U7")
    array = xr.DataArray(
        values,
        dims="observation",
        coords={
            "observation": [0, 1],
            "species": ("observation", ["co2", "o2"]),
            "observation_units": ("observation", units),
        },
    )
    prepared = _prepared_stub(array)
    prepared.o2_operator.attrs["oxidation_ratio_provenance"] = json.dumps({"status": "available"})
    independent_error_sd = array.copy(deep=False, data=values)
    monkeypatch.setattr(co2_o2_runner, "build_co2_o2_model", lambda **_: pm.Model())
    monkeypatch.setattr(co2_o2_runner, "sample_rhime_model", lambda *_: az.InferenceData())
    compute_graphs: list[object] = []

    with Callback(start=lambda graph: compute_graphs.append(graph)):
        trace = run_rhime_co2_o2_from_prepared_inputs(
            prepared_inputs=prepared,
            independent_error_sd=independent_error_sd,
        )

    assert len(compute_graphs) == 1
    assert sorted(executions) == ["units", "values"]
    assert json.loads(trace.attrs["rhime_model_metadata"])["observation_units"] == {
        "co2": "ppm",
        "o2": "per meg",
    }
