"""Centered boundary transport contracts at their raw and bound handoffs."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr
from xarray_validate import DatasetSchema

from openghg_inversions.models.boundary import (
    AdditiveChoice,
    BoundAdditiveSensitivity,
    BoundCenteredSensitivity,
    MultiplicativeBoundary,
    MultiplicativeChoice,
    RawCenteredSensitivity,
    validate_centering_weights,
)


TIMES = pd.date_range("2020-01-01", periods=4, freq="15D")
CURTAINS = ["N", "E", "S"]


def _forbid_compute(*_args, **_kwargs):
    """Dask scheduler that fails if anything is computed."""
    raise AssertionError("Contract validation computed a Dask payload.")


def raw_example(*, lazy: bool = False, weights: bool = True, transposed: bool = False) -> xr.Dataset:
    """Raw centered transport on (bc_region, time), optionally Dask-backed."""
    rng = np.random.default_rng(0)
    dims = ("time", "bc_region") if transposed else ("bc_region", "time")
    shape = tuple({"time": len(TIMES), "bc_region": len(CURTAINS)}[dim] for dim in dims)
    coords = {"time": TIMES, "bc_region": CURTAINS}
    anomaly = xr.DataArray(rng.normal(size=shape), dims=dims, coords=coords, attrs={"units": "ppm"})
    mass = xr.DataArray(rng.uniform(0.1, 0.5, shape), dims=dims, coords=coords, attrs={"units": "1"})
    data = xr.Dataset(
        {"H_bc_anomaly": anomaly, "H_bc_mass": mass, "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"})},
        attrs={"boundary_representation": "centered-v1"},
    )
    if weights:
        data["bc_centering_weights"] = xr.DataArray(
            np.ones((len(CURTAINS), 1)), dims=("bc_curtain", "bc_period"),
            coords={"bc_curtain": CURTAINS, "bc_period": TIMES[:1]}, attrs={"units": "1"},
        )
    if lazy:
        data = data.chunk().assign_coords(site_altitude=("time", da.zeros(len(TIMES), chunks=1)))
    return data


def bound_example(*, lazy: bool = False, weights: bool = False) -> xr.Dataset:
    """Bound centered transport on (nmeasure, bc_region), optionally with weights."""
    states = pd.MultiIndex.from_product([CURTAINS, TIMES[:2]], names=["bc_curtain", "bc_period"])
    coords = xr.Coordinates.from_pandas_multiindex(states, "bc_region")
    rows = len(TIMES)
    data = xr.Dataset(
        {
            "H_bc_anomaly": (("nmeasure", "bc_region"), np.zeros((rows, len(states))), {"units": "ppm"}),
            "H_bc_mass": (("nmeasure", "bc_region"), np.ones((rows, len(states))), {"units": "1"}),
            "bc_reference": ((), 400.0, {"units": "ppm"}),
        },
        coords={"nmeasure": np.arange(rows), "time": ("nmeasure", TIMES), **coords},
    )
    if weights:
        data["bc_centering_weights"] = xr.DataArray(np.ones(len(states)), dims="bc_region", attrs={"units": "1"})
    if lazy:
        data = data.chunk().assign_coords(time=("nmeasure", da.from_array(TIMES.values, chunks=1)))
    return data


def _observations() -> xr.DataArray:
    """Observations on nmeasure with auxiliary timestamps."""
    return xr.DataArray(
        np.full(len(TIMES), 410.0), dims="nmeasure",
        coords={"nmeasure": np.arange(len(TIMES)), "time": ("nmeasure", TIMES)}, attrs={"units": "ppm"},
    )


def _raw_on_nmeasure(**kwargs) -> xr.Dataset:
    """Raw example on nmeasure rows, as prepared CO2 inputs store it."""
    data = raw_example(**kwargs)
    return data.rename({"time": "nmeasure"}).assign_coords(
        nmeasure=np.arange(len(TIMES)), time=("nmeasure", TIMES),
    )


# Examples for the shared conformance tests in tests/test_xr_contracts.py.
CONTRACT_EXAMPLES = {
    RawCenteredSensitivity: lambda lazy: (raw_example(lazy=lazy), {}),
    BoundCenteredSensitivity: lambda lazy: (bound_example(lazy=lazy), {}),
    BoundAdditiveSensitivity: lambda lazy: (bound_example(lazy=lazy, weights=True), {}),
}


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("weights", [False, True])
def test_raw_contract_accepts_supported_layouts_unchanged(transposed, weights):
    """Either operand order and optional weights pass, returning the same object."""
    data = raw_example(transposed=transposed, weights=weights)
    assert RawCenteredSensitivity.validate(data) is data


def test_raw_contract_permits_extra_variables_and_nmeasure_rows():
    """Prepared inputs carry other variables and store rows on nmeasure."""
    data = _raw_on_nmeasure().assign(mf=lambda ds: xr.full_like(ds.time, 410.0, float).assign_attrs(units="ppm"))
    assert RawCenteredSensitivity.validate(data) is data


def _drop(name):
    return lambda data: data.drop_vars(name)


def _set_attrs(name, **attrs):
    """Return a nonmutating metadata change, preserving shared MultiIndexes."""
    def modify(data):
        """Replace attributes on an independent dataset view."""
        result = data.copy(deep=False)
        result[name].attrs = {**data[name].attrs, **attrs}
        return result
    return modify


RAW_FAILURES = {
    "missing mass": (_drop("H_bc_mass"), "missing keys.*H_bc_mass"),
    "representation": (lambda data: data.assign_attrs(boundary_representation=None), "centered-v1"),
    "expanded marker": (
        _set_attrs("H_bc_mass", boundary_sensitivity_preparation="expanded"), "unexpanded spatial operands",
    ),
    "unmarked absolute": (
        lambda data: data.assign(H_bc=data.H_bc_anomaly.copy()), "expanded or unmarked absolute H_bc",
    ),
    "dimensions": (
        lambda data: data.assign(H_bc_mass=data.H_bc_mass.rename(bc_region="site")), "data_vars.H_bc_mass.dims",
    ),
    "duplicate states": (
        lambda data: data.assign_coords(bc_region=["N", "N", "S"]), "unique indexed bc_region",
    ),
    "reference shape": (
        lambda data: data.assign(bc_reference=data.H_bc_anomaly.isel(bc_region=0)), "data_vars.bc_reference.dims",
    ),
    "units": (_set_attrs("bc_reference", units="ppb"), "share concentration units"),
    "mass units": (_set_attrs("H_bc_mass", units="ppm"), "data_vars.H_bc_mass.attrs.*units"),
    "mf units": (
        lambda data: data.assign(mf=xr.full_like(data.time, 1.0, float).assign_attrs(units="ppb")),
        "must match mf units",
    ),
    "weight dims": (
        lambda data: data.assign(bc_centering_weights=data.bc_centering_weights.T), "dims",
    ),
    "weight periods": (
        lambda data: data.assign_coords(bc_period=["first"]), "interval start times",
    ),
    "weight units": (_set_attrs("bc_centering_weights", units="ppm"), "equivalent to '1'"),
}


@pytest.mark.parametrize("case", RAW_FAILURES, ids=list(RAW_FAILURES))
def test_raw_contract_reports_the_violated_requirement(case):
    """Malformed supplied structure is rejected with the affected element named."""
    modify, message = RAW_FAILURES[case]
    with pytest.raises(ValueError, match=message):
        RawCenteredSensitivity.validate(modify(raw_example()), label="co2 centered boundaries")


def test_weight_check_accepts_whole_window_and_equivalent_units():
    """Timestamp-free weights use the explicit whole-window label."""
    weights = xr.DataArray(
        [[1.0], [2.0]], dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": ["N", "S"], "bc_period": ["whole_window"]}, attrs={"units": "dimensionless"},
    )
    validate_centering_weights(weights)


BOUND_FAILURES = {
    "missing reference": (_drop("bc_reference"), "missing keys.*bc_reference"),
    "order": (lambda data: data.transpose("bc_region", "nmeasure"), "data_vars.H_bc_mass.dims"),
    "mass layout": (
        lambda data: data.assign(H_bc_mass=data.H_bc_mass.isel(bc_region=0)), "data_vars.H_bc_mass.dims",
    ),
    "plain states": (
        lambda data: data.drop_vars(["bc_region", "bc_curtain", "bc_period"]).assign_coords(bc_region=np.arange(6)),
        "MultiIndex",
    ),
    "units": (_set_attrs("H_bc_mass", units="ppm"), "data_vars.H_bc_mass.attrs.*units"),
}


@pytest.mark.parametrize("case", BOUND_FAILURES, ids=list(BOUND_FAILURES))
def test_bound_contract_reports_the_violated_requirement(case):
    """Bound structure that consumers rely on is checked by the bound validator."""
    modify, message = BOUND_FAILURES[case]
    with pytest.raises(ValueError, match=message):
        BoundCenteredSensitivity.validate(modify(bound_example()))


def test_additive_extension_requires_weights_on_the_bound_state_axis():
    """The extension adds its weight guarantee to the shared bound guarantees."""
    shared = bound_example()
    assert BoundCenteredSensitivity.validate(shared) is shared
    with pytest.raises(ValueError, match="missing keys.*bc_centering_weights"):
        BoundAdditiveSensitivity.validate(shared)
    with pytest.raises(ValueError, match="equivalent to '1'"):
        BoundAdditiveSensitivity.validate(shared.assign(
            bc_centering_weights=xr.DataArray(np.ones(6), dims="bc_region", attrs={"units": "ppm"}),
        ))
    extended = bound_example(weights=True)
    assert BoundAdditiveSensitivity.validate(extended) is extended
    assert BoundCenteredSensitivity.observation_dim(extended) == "nmeasure"


@pytest.fixture
def float64():
    """Centered binding requires a float64 PyTensor process."""
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.mark.usefixtures("float64")
def test_binding_constructs_the_bound_contract_lazily():
    """Binding produces validator-conforming output without computing payloads."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure(lazy=True))
    with dask.config.set(scheduler=_forbid_compute):
        bound = BoundCenteredSensitivity.bind(raw, _observations(), frequency="monthly")
        assert BoundCenteredSensitivity.validate(bound) is bound
    assert dask.is_dask_collection(bound.H_bc_mass)
    assert list(bound.indexes["bc_region"].names) == ["bc_curtain", "bc_period"]
    assert len(bound.indexes["bc_region"]) == len(CURTAINS) * 2  # Every curtain in every month.
    assert "bc_centering_weights" not in bound


@pytest.mark.usefixtures("float64")
def test_weighted_additive_binding_establishes_the_extension():
    """Weighted additive binding aligns the weights to the bound states."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure(lazy=True))
    with dask.config.set(scheduler=_forbid_compute):
        bound = AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.3).bind(_observations(), raw)
        assert BoundAdditiveSensitivity.validate(bound.data) is bound.data
    np.testing.assert_array_equal(bound.data.bc_centering_weights.compute(), np.ones(len(CURTAINS)))


@pytest.mark.usefixtures("float64")
def test_unweighted_families_share_the_centered_contract_without_weights():
    """Multiplicative and period-contrast additive bindings need no weights."""
    raw = RawCenteredSensitivity.validate(_raw_on_nmeasure(weights=False))
    contrasts = AdditiveChoice(None, 0.3, anomalies="period_contrasts").bind(_observations(), raw)
    multiplicative = MultiplicativeChoice({"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}).bind(_observations(), raw)
    for data in (contrasts.data, multiplicative.data):
        assert BoundCenteredSensitivity.validate(data) is data
        assert "bc_centering_weights" not in data
    with pytest.raises(ValueError, match="require bc_centering_weights"):
        AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.3).bind(_observations(), raw)


def test_absolute_multiplicative_route_keeps_its_own_layout():
    """Absolute transport is bound without centered assumptions."""
    absolute = _raw_on_nmeasure().H_bc_anomaly.rename("H_bc")
    bound = MultiplicativeChoice({"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}).bind(_observations(), absolute)
    assert set(bound.data.data_vars) == {"H_bc"}
    assert isinstance(bound, MultiplicativeBoundary)
    with pytest.raises(ValueError, match="missing keys.*H_bc_anomaly"):
        BoundCenteredSensitivity.validate(bound.data)


@pytest.mark.parametrize("contract", [RawCenteredSensitivity, BoundCenteredSensitivity, BoundAdditiveSensitivity])
def test_contract_schemas_expose_metadata_requirements(contract):
    """Class operations expose variable, dtype and dimensional schemas."""
    schema = contract.schema(observation_dim="row")
    assert isinstance(schema, DatasetSchema)
    assert schema.allow_extra_keys
    assert schema.data_vars["H_bc_mass"].dims.dims == ("row", "bc_region")
    assert schema.data_vars["bc_reference"].dims.dims == ()
    assert schema.data_vars["H_bc_mass"].dtype is not None
    assert schema.data_vars["H_bc_mass"].attrs.attrs["units"].value == "1"
    assert ("bc_centering_weights" in schema.data_vars) == (contract is BoundAdditiveSensitivity)


@pytest.mark.parametrize("contract,example", [
    (RawCenteredSensitivity, raw_example),
    (BoundCenteredSensitivity, bound_example),
    (BoundAdditiveSensitivity, lambda **kwargs: bound_example(weights=True, **kwargs)),
])
def test_schema_validation_preserves_lazy_payloads_and_auxiliary_coordinates(contract, example):
    """Metadata validation preserves borrowed graphs, lazy auxiliaries and identity."""
    data = example(lazy=True)
    payload = data.H_bc_mass.data
    with dask.config.set(scheduler=_forbid_compute):
        assert contract.validate(data) is data
    assert data.H_bc_mass.data is payload


@pytest.mark.parametrize("selection", [[0, 1, 2, 3, 4], [0, 3]])
def test_bound_contract_rejects_incomplete_curtain_period_grid(selection):
    """Unique pairs alone do not establish every curtain in every period."""
    data = bound_example(lazy=True).isel(bc_region=selection)
    with dask.config.set(scheduler=_forbid_compute), pytest.raises(ValueError, match="Cartesian combination"):
        BoundCenteredSensitivity.validate(data)


def test_bound_contract_accepts_reordered_complete_grid_and_unused_levels():
    """Cartesian coverage depends on represented labels rather than cached levels."""
    data = bound_example(lazy=True).isel(bc_region=[5, 4, 3, 2, 1, 0])
    assert BoundCenteredSensitivity.validate(data) is data
    one_period = data.isel(bc_region=[0, 2, 4])
    assert BoundCenteredSensitivity.validate(one_period) is one_period


def test_bound_contract_rejects_missing_multiindex_labels():
    """Missing MultiIndex codes are rejected without calling unsupported hasnans."""
    states = pd.MultiIndex.from_tuples([("N", TIMES[0]), (None, TIMES[0])], names=["bc_curtain", "bc_period"])
    data = bound_example(lazy=True).isel(bc_region=[0, 1]).drop_vars(["bc_region", "bc_curtain", "bc_period"])
    data = data.assign_coords(xr.Coordinates.from_pandas_multiindex(states, "bc_region"))
    with dask.config.set(scheduler=_forbid_compute), pytest.raises(ValueError, match="without missing labels"):
        BoundCenteredSensitivity.validate(data)


@pytest.mark.parametrize("name", ["H_bc_anomaly", "H_bc_mass", "bc_reference"])
def test_transport_contract_rejects_nonreal_dtype_without_computing(name):
    """Real-valued operands are established by their dtype metadata."""
    data = raw_example(lazy=True)
    data[name] = data[name].astype(complex)
    with dask.config.set(scheduler=_forbid_compute), pytest.raises(ValueError, match=f"data_vars.{name}.dtype"):
        RawCenteredSensitivity.validate(data)


@pytest.mark.parametrize("contract,example", [
    (RawCenteredSensitivity, raw_example),
    (BoundCenteredSensitivity, bound_example),
])
def test_lazy_stage_contract_does_not_establish_nonzero_mass(contract, example):
    """Row selection can leave zero mass; its numerical consumer owns that check."""
    data = example(lazy=True)
    data["H_bc_mass"].data = da.zeros_like(data.H_bc_mass.data)
    with dask.config.set(scheduler=_forbid_compute):
        assert contract.validate(data) is data


def test_raw_schema_failures_include_the_supplied_channel_label():
    """Library schema failures retain application context."""
    with pytest.raises(ValueError, match="co2 channel.*Validation failed"):
        RawCenteredSensitivity.validate(raw_example().drop_vars("bc_reference"), label="co2 channel")
