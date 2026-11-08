"""Centered boundary transport contracts at their raw and bound handoffs."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models.boundary import (
    AdditiveChoice,
    BoundAdditiveTransport,
    BoundCenteredTransport,
    MultiplicativeChoice,
    RawCenteredTransport,
    bind_centered_transport,
)
from openghg_inversions.models.boundary.contracts import check_centering_weights


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
    RawCenteredTransport: lambda lazy: (raw_example(lazy=lazy), {}),
    BoundCenteredTransport: lambda lazy: (bound_example(lazy=lazy), {}),
    BoundAdditiveTransport: lambda lazy: (bound_example(lazy=lazy, weights=True), {}),
}


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("weights", [False, True])
def test_raw_contract_accepts_supported_layouts_unchanged(transposed, weights):
    """Either operand order and optional weights pass, returning the same object."""
    data = raw_example(transposed=transposed, weights=weights)
    assert RawCenteredTransport.validate(data) is data


def test_raw_contract_permits_extra_variables_and_nmeasure_rows():
    """Prepared inputs carry other variables and store rows on nmeasure."""
    data = _raw_on_nmeasure().assign(mf=lambda ds: xr.full_like(ds.time, 410.0, float).assign_attrs(units="ppm"))
    assert RawCenteredTransport.validate(data) is data


def _drop(name):
    return lambda data: data.drop_vars(name)


def _set_attrs(name, **attrs):
    return lambda data: data.assign({name: data[name].assign_attrs(attrs)})


RAW_FAILURES = {
    "missing mass": (_drop("H_bc_mass"), "missing H_bc_mass"),
    "representation": (lambda data: data.assign_attrs(boundary_representation=None), "centered-v1"),
    "expanded marker": (
        _set_attrs("H_bc_mass", boundary_sensitivity_preparation="expanded"), "unexpanded spatial operands",
    ),
    "unmarked absolute": (
        lambda data: data.assign(H_bc=data.H_bc_anomaly.copy()), "expanded or unmarked absolute H_bc",
    ),
    "dimensions": (
        lambda data: data.assign(H_bc_mass=data.H_bc_mass.rename(bc_region="site")), "on bc_region and time",
    ),
    "duplicate states": (
        lambda data: data.assign_coords(bc_region=["N", "N", "S"]), "unique indexed bc_region",
    ),
    "reference shape": (
        lambda data: data.assign(bc_reference=data.H_bc_anomaly.isel(bc_region=0)), "bc_reference must be scalar",
    ),
    "units": (_set_attrs("bc_reference", units="ppb"), "share concentration units"),
    "mass units": (_set_attrs("H_bc_mass", units="ppm"), "H_bc_mass units must be '1'"),
    "mf units": (
        lambda data: data.assign(mf=xr.full_like(data.time, 1.0, float).assign_attrs(units="ppb")),
        "must match mf units",
    ),
    "weight dims": (
        lambda data: data.assign(bc_centering_weights=data.bc_centering_weights.T), "must use dimensions",
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
        RawCenteredTransport.validate(modify(raw_example()), label="co2 centered boundaries")


def test_weight_check_accepts_whole_window_and_equivalent_units():
    """Timestamp-free weights use the explicit whole-window label."""
    weights = xr.DataArray(
        [[1.0], [2.0]], dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": ["N", "S"], "bc_period": ["whole_window"]}, attrs={"units": "dimensionless"},
    )
    check_centering_weights(weights)


BOUND_FAILURES = {
    "missing reference": (_drop("bc_reference"), "lacks bc_reference"),
    "order": (lambda data: data.transpose("bc_region", "nmeasure"), r"\(observation, 'bc_region'\)"),
    "mass layout": (
        lambda data: data.assign(H_bc_mass=data.H_bc_mass.isel(bc_region=0)), r"\(observation, 'bc_region'\)",
    ),
    "plain states": (
        lambda data: data.drop_vars(["bc_region", "bc_curtain", "bc_period"]).assign_coords(bc_region=np.arange(6)),
        "MultiIndex",
    ),
    "units": (_set_attrs("H_bc_mass", units="ppm"), "H_bc_mass units must be '1'"),
}


@pytest.mark.parametrize("case", BOUND_FAILURES, ids=list(BOUND_FAILURES))
def test_bound_contract_reports_the_violated_requirement(case):
    """Bound structure that consumers rely on is checked by the bound validator."""
    modify, message = BOUND_FAILURES[case]
    with pytest.raises(ValueError, match=message):
        BoundCenteredTransport.validate(modify(bound_example()))


def test_additive_extension_requires_weights_on_the_bound_state_axis():
    """The extension adds its weight guarantee to the shared bound guarantees."""
    shared = bound_example()
    assert BoundCenteredTransport.validate(shared) is shared
    with pytest.raises(ValueError, match="bc_centering_weights on bc_region"):
        BoundAdditiveTransport.validate(shared)
    with pytest.raises(ValueError, match="equivalent to '1'"):
        BoundAdditiveTransport.validate(shared.assign(
            bc_centering_weights=xr.DataArray(np.ones(6), dims="bc_region", attrs={"units": "ppm"}),
        ))
    extended = bound_example(weights=True)
    assert BoundAdditiveTransport.validate(extended) is extended
    assert BoundCenteredTransport.observation_dim(extended) == "nmeasure"


@pytest.fixture
def float64():
    """Centered binding requires a float64 PyTensor process."""
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.mark.usefixtures("float64")
def test_binding_constructs_the_bound_contract_lazily():
    """Binding produces validator-conforming output without computing payloads."""
    raw = RawCenteredTransport.validate(_raw_on_nmeasure(lazy=True))
    with dask.config.set(scheduler=_forbid_compute):
        bound = bind_centered_transport(raw, _observations(), frequency="monthly")
        assert BoundCenteredTransport.validate(bound) is bound
    assert dask.is_dask_collection(bound.H_bc_mass)
    assert list(bound.indexes["bc_region"].names) == ["bc_curtain", "bc_period"]
    assert len(bound.indexes["bc_region"]) == len(CURTAINS) * 2  # Every curtain in every month.
    assert "bc_centering_weights" not in bound


@pytest.mark.usefixtures("float64")
def test_weighted_additive_binding_establishes_the_extension():
    """Weighted additive binding aligns the weights to the bound states."""
    raw = RawCenteredTransport.validate(_raw_on_nmeasure(lazy=True))
    with dask.config.set(scheduler=_forbid_compute):
        bound = AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.3).bind(_observations(), raw)
        assert BoundAdditiveTransport.validate(bound.data) is bound.data
    np.testing.assert_array_equal(bound.data.bc_centering_weights.compute(), np.ones(len(CURTAINS)))


@pytest.mark.usefixtures("float64")
def test_unweighted_families_share_the_centered_contract_without_weights():
    """Multiplicative and period-contrast additive bindings need no weights."""
    raw = RawCenteredTransport.validate(_raw_on_nmeasure(weights=False))
    contrasts = AdditiveChoice(None, 0.3, anomalies="period_contrasts").bind(_observations(), raw)
    multiplicative = MultiplicativeChoice({"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}).bind(_observations(), raw)
    for data in (contrasts.data, multiplicative.data):
        assert BoundCenteredTransport.validate(data) is data
        assert "bc_centering_weights" not in data
    with pytest.raises(ValueError, match="require bc_centering_weights"):
        AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 1.0}, 0.3).bind(_observations(), raw)


def test_absolute_multiplicative_route_keeps_its_own_layout():
    """Absolute transport is bound without centered assumptions."""
    absolute = _raw_on_nmeasure().H_bc_anomaly.rename("H_bc")
    bound = MultiplicativeChoice({"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}).bind(_observations(), absolute)
    assert set(bound.data.data_vars) == {"H_bc"}
    assert not bound.centered
    with pytest.raises(ValueError, match="lacks H_bc_anomaly"):
        BoundCenteredTransport.validate(bound.data)
