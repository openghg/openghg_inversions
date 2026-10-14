"""Exercise the xarray-validate pilot at the real CO2 artifact boundary."""

from dataclasses import replace
from pathlib import Path

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.serialization import save_datatree
from test_co2_preparation import _canonical_inputs, _reduction


def _prepared() -> Co2PreparedInputs:
    canonical = _canonical_inputs()
    return prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=None)


@pytest.mark.parametrize(
    "name",
    ["H", "alpha_prior_mean", "alpha_prior_covariance", "fixed_prior_contribution", "mf", "mf_error"],
)
def test_schema_requires_each_co2_variable(name: str) -> None:
    prepared = _prepared()
    del prepared.inv_inputs[name]

    with pytest.raises(ValueError, match=f"missing keys:.*{name}"):
        replace(prepared)


def test_schema_collects_structural_errors_with_variable_paths() -> None:
    prepared = _prepared()
    inputs = prepared.inv_inputs
    inputs["H"] = inputs["H"].variable.transpose("region", "nmeasure")
    inputs["mf_error"].data = inputs["mf_error"].data.astype("int64")
    inputs["fixed_prior_contribution"] = xr.DataArray([1.0, 2.0], dims="region")

    with pytest.raises(ValueError) as error:
        replace(prepared)

    message = str(error.value)
    assert "data_vars.H.dims: dimension mismatch" in message
    assert "data_vars.mf_error.dtype: dtype mismatch" in message
    assert "data_vars.fixed_prior_contribution.dims: dimension mismatch" in message


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_schema_accepts_floating_precision_without_coercion(dtype: str) -> None:
    prepared = _prepared()
    prepared.inv_inputs["H"].data = prepared.inv_inputs["H"].data.astype(dtype)

    validated = replace(prepared)

    assert validated.inv_inputs["H"].data is prepared.inv_inputs["H"].data
    assert validated.inv_inputs["H"].dtype == np.dtype(dtype)


def test_schema_validation_keeps_dask_payloads_and_auxiliary_coordinates_lazy() -> None:
    prepared = _prepared()
    inputs = prepared.inv_inputs
    for name in tuple(inputs.data_vars):
        variable = inputs[name]
        variable.data = da.from_array(variable.data, chunks=1)

    @delayed
    def forbidden_auxiliary_execution() -> np.ndarray:
        raise AssertionError("Validation executed an auxiliary coordinate.")

    inputs.coords["lazy_auxiliary"] = (
        "nmeasure",
        da.from_delayed(forbidden_auxiliary_execution(), shape=(3,), dtype=float),
    )
    before = {
        name: (variable.data, variable.chunks)
        for name, variable in inputs.variables.items()
        if isinstance(variable.data, da.Array)
    }
    executions = []
    with Callback(pretask=lambda key, *_: executions.append(key)):
        validated = replace(prepared)

    assert executions == []
    assert validated.inv_inputs is inputs
    for name, (data, chunks) in before.items():
        assert inputs[name].data is data
        assert inputs[name].chunks == chunks


def test_scientific_covariance_shape_and_numeric_unit_scale_still_have_owners() -> None:
    prepared = _prepared()
    covariance = prepared.inv_inputs["alpha_prior_covariance"].isel(region_cov=slice(0, 1))
    rhime_inputs = replace(
        prepared.rhime_inputs,
        inv_inputs=prepared.inv_inputs.drop_dims("region_cov").assign(alpha_prior_covariance=covariance),
    )
    with pytest.raises(ValueError, match="must be square"):
        replace(prepared, rhime_inputs=rhime_inputs)

    prepared = _prepared()
    prepared.inv_inputs["H"].attrs["units"] = "ppb"
    with pytest.raises(ValueError, match="same numeric scale"):
        replace(prepared)


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_disk_loader_rejects_schema_invalid_payload(tmp_path: Path, suffix: str) -> None:
    prepared = _prepared()
    path = tmp_path / f"co2-schema{suffix}"
    prepared.save(path)
    restored = Co2PreparedInputs.load(path)
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)

    tree = prepared.to_datatree()
    node = tree["rhime_inputs/inv_inputs"]
    dataset = node.to_dataset()
    dataset["mf_error"] = dataset["mf_error"].astype("int64")
    node.ds = dataset
    save_datatree(tree, path)

    with pytest.raises(ValueError, match="data_vars.mf_error.dtype: dtype mismatch"):
        Co2PreparedInputs.load(path)
