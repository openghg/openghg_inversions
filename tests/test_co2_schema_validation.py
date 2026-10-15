"""Exercise the xarray-validate pilot at the real CO2 artifact boundary."""

from dataclasses import replace
from pathlib import Path

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr
from xarray_validate import SchemaError

from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.serialization import save_datatree
from test_co2_preparation import _canonical_inputs, _reduction


def _prepared(rank: int | None = None) -> Co2PreparedInputs:
    canonical = _canonical_inputs()
    return prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=rank)


def test_none_dimensions_match_independent_axis_names_but_preserve_rank() -> None:
    inputs = _prepared().inv_inputs.copy(deep=False)
    mean = inputs["alpha_prior_mean"]
    inputs["alpha_prior_mean"] = xr.DataArray(mean.data, dims="other_state", attrs=mean.attrs)
    portable_schema = Co2PreparedInputs.dataset_schema()

    # Independent None entries permit different state names across variables.
    portable_schema.validate(inputs)
    with pytest.raises(SchemaError, match="dimension mismatch.*other_state.*region"):
        Co2PreparedInputs.dataset_schema("region").validate(inputs)

    # None still occupies one axis; it does not mean any number of dimensions.
    inputs["H"] = xr.DataArray(np.zeros(3), dims="nmeasure")
    with pytest.raises(SchemaError, match="dimension number mismatch: got 1, expected 2"):
        portable_schema.validate(inputs)


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_product_reports_malformed_h_rank_through_its_schema(rank: int) -> None:
    prepared = _prepared()
    dimensions = ("nmeasure", "region", "unexpected")[:rank]
    shape = (3, 2, 1)[:rank]
    prepared.inv_inputs["H"] = xr.DataArray(np.zeros(shape), dims=dimensions)

    with pytest.raises(ValueError, match="data_vars.H.dims: dimension number mismatch"):
        replace(prepared)


@pytest.mark.parametrize("name", ["alpha_prior_mean", "alpha_prior_covariance"])
def test_product_schema_binds_state_relationships(name: str) -> None:
    prepared = _prepared()
    value = prepared.inv_inputs[name]
    dimensions = ("other_state",) if name == "alpha_prior_mean" else ("region", "other_covariance")
    prepared.inv_inputs[name] = xr.DataArray(value.data, dims=dimensions, attrs=value.attrs)

    with pytest.raises(ValueError, match=f"data_vars.{name}.dims: dimension mismatch"):
        replace(prepared)


@pytest.mark.parametrize("mode", ["dense", "low_rank"])
def test_bound_schema_rejects_opposing_aggregation_representation(mode: str) -> None:
    dense = _prepared()
    low_rank = _prepared(1)
    mixed = dense.inv_inputs.assign(
        low_rank_factor=low_rank.inv_inputs["low_rank_factor"].variable,
        diagonal_residual_variance=low_rank.inv_inputs["diagonal_residual_variance"].variable,
    )
    schema = Co2PreparedInputs.dataset_schema(
        "region", concentration_units="ppm", aggregation_error_mode=mode,
    )

    with pytest.raises(ValueError, match="must contain only"):
        schema.validate(mixed)
    with pytest.raises(ValueError, match="must contain only"):
        replace(dense, aggregation_error_mode=mode, rhime_inputs=replace(dense.rhime_inputs, inv_inputs=mixed))


@pytest.mark.parametrize("missing", ["low_rank_factor", "diagonal_residual_variance"])
def test_missing_aggregation_pair_reports_structure_and_representation_policy(missing: str) -> None:
    prepared = _prepared(1)
    del prepared.inv_inputs[missing]

    with pytest.raises(ValueError) as error:
        replace(prepared)

    message = str(error.value)
    assert "data_vars has missing keys" in message
    assert "together" in message
    assert missing in message


@pytest.mark.parametrize("mode", ["", "diagonal", 1])
def test_schema_construction_rejects_unsupported_bound_modes(mode: object) -> None:
    with pytest.raises(ValueError, match="aggregation_error_mode"):
        Co2PreparedInputs.dataset_schema(aggregation_error_mode=mode)


def test_product_requires_a_bound_aggregation_mode() -> None:
    with pytest.raises(ValueError, match="aggregation_error_mode"):
        replace(_prepared(), aggregation_error_mode=None)


def test_lazy_aggregation_mode_is_rejected_without_execution() -> None:
    prepared = _prepared()

    @delayed
    def forbidden() -> np.ndarray:
        raise AssertionError("Aggregation mode validation executed a graph.")

    mode = da.from_delayed(forbidden(), shape=(), dtype=str)
    executed = []
    with Callback(pretask=lambda key, *_: executed.append(key)):
        with pytest.raises(ValueError, match="aggregation_error_mode"):
            Co2PreparedInputs.dataset_schema(aggregation_error_mode=mode)
        with pytest.raises(ValueError, match="aggregation_error_mode"):
            replace(prepared, aggregation_error_mode=mode)
    assert executed == []


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


@pytest.mark.parametrize("unit_path", ["native", "expression"])
def test_schema_validation_keeps_dask_payloads_and_auxiliary_coordinates_lazy(unit_path: str) -> None:
    prepared = _prepared()
    inputs = prepared.inv_inputs
    if unit_path == "expression":
        inputs["H"].attrs["units"] = "umol/mol"
        inputs["aggregation_error_covariance"].attrs["units"] = "(1e-6)**2"
    attributes = {name: dict(variable.attrs) for name, variable in inputs.data_vars.items()}
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
    for name, attrs in attributes.items():
        assert inputs[name].attrs == attrs


@pytest.mark.parametrize(
    ("name", "units"),
    [
        ("H", "parts_per_million"),
        ("H", "1e-6"),
        ("H", "umol/mol"),
        ("aggregation_error_covariance", "(1e-6)**2"),
        ("aggregation_error_covariance", "(umol/mol)**2"),
    ],
)
def test_bound_schema_preserves_equal_scale_unit_expressions(name: str, units: str) -> None:
    prepared = _prepared()
    prepared.inv_inputs[name].attrs["units"] = units
    schema = Co2PreparedInputs.dataset_schema(
        "region", concentration_units="ppm", aggregation_error_mode="dense",
    )

    schema.validate(prepared.inv_inputs)
    validated = replace(prepared)

    assert validated.inv_inputs[name].attrs["units"] == units


def test_explicit_schema_checks_observation_units_against_its_bound_reference() -> None:
    prepared = _prepared()
    prepared.inv_inputs["mf"].attrs["units"] = "ppb"
    schema = Co2PreparedInputs.dataset_schema(concentration_units="ppm")

    with pytest.raises(ValueError, match="CO2 mf.*same numeric scale"):
        schema.validate(prepared.inv_inputs)


@pytest.mark.parametrize(
    ("name", "units", "rank"),
    [
        ("H", "ppb", None),
        ("H", "mol/mol", None),
        ("aggregation_error_covariance", "(ppb)**2", None),
        ("low_rank_factor", "ppb", 1),
        ("diagonal_residual_variance", "(ppb)**2", 1),
    ],
)
def test_bound_schema_rejects_different_unit_scales(name: str, units: str, rank: int | None) -> None:
    prepared = _prepared(rank)
    prepared.inv_inputs[name].attrs["units"] = units

    with pytest.raises(ValueError, match=f"CO2 {name}.*same numeric scale"):
        replace(prepared)


@pytest.mark.parametrize(
    ("name", "rank"),
    [
        ("H", None),
        ("fixed_prior_contribution", None),
        ("mf_error", None),
        ("aggregation_error_covariance", None),
        ("low_rank_factor", 1),
        ("diagonal_residual_variance", 1),
    ],
)
def test_bound_schema_requires_unit_metadata(name: str, rank: int | None) -> None:
    prepared = _prepared(rank)
    del prepared.inv_inputs[name].attrs["units"]

    with pytest.raises(ValueError, match=f"CO2 {name} requires non-empty units"):
        replace(prepared)


@pytest.mark.parametrize("units", [None, "", " ", 1])
def test_product_requires_observation_unit_metadata(units: object) -> None:
    prepared = _prepared()
    prepared.inv_inputs["mf"].attrs["units"] = units

    with pytest.raises(ValueError, match="CO2 observations require units"):
        replace(prepared)


@pytest.mark.parametrize("name", ["mf", "H", "alpha_prior_mean", "alpha_prior_covariance"])
def test_malformed_lazy_unit_metadata_is_rejected_without_execution(name: str) -> None:
    prepared = _prepared()

    @delayed
    def forbidden() -> np.ndarray:
        raise AssertionError("Unit metadata validation executed a graph.")

    prepared.inv_inputs[name].attrs["units"] = da.from_delayed(forbidden(), shape=(), dtype=str)
    executed = []
    with Callback(pretask=lambda key, *_: executed.append(key)):
        with pytest.raises(ValueError):
            replace(prepared)
    assert executed == []


@pytest.mark.parametrize("covariance_dim", ["region_cov", "nmeasure_cov"])
def test_different_covariance_axes_still_require_equal_label_order(covariance_dim: str) -> None:
    prepared = _prepared()
    reversed_columns = prepared.inv_inputs.isel({covariance_dim: slice(None, None, -1)})
    # Shapes, names and units are valid; only the distinct axes' label order differs.
    Co2PreparedInputs.dataset_schema(
        "region", concentration_units="ppm", aggregation_error_mode="dense",
    ).validate(reversed_columns)

    with pytest.raises(ValueError, match="same values in the same order"):
        replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=reversed_columns))


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
    # The product binds the actual H state name rather than hard-coding region.
    prepared = replace(
        prepared,
        rhime_inputs=replace(
            prepared.rhime_inputs,
            inv_inputs=prepared.inv_inputs.rename(region="retained", region_cov="retained_cov"),
        ),
    )
    before = prepared.inv_inputs.copy(deep=True)
    path = tmp_path / f"co2-schema{suffix}"
    prepared.save(path)
    xr.testing.assert_identical(prepared.inv_inputs, before)
    restored = Co2PreparedInputs.load(path)
    assert restored.inv_inputs["H"].dims == ("nmeasure", "retained")
    assert restored.inv_inputs["alpha_prior_covariance"].dims == ("retained", "retained_cov")
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)

    tree = prepared.to_datatree()
    node = tree["rhime_inputs/inv_inputs"]
    dataset = node.to_dataset()
    dataset["mf_error"] = dataset["mf_error"].astype("int64")
    node.ds = dataset
    save_datatree(tree, path)

    with pytest.raises(ValueError, match="data_vars.mf_error.dtype: dtype mismatch"):
        Co2PreparedInputs.load(path)
