import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import pytest

from openghg_inversions.models.coords import (
    CoordRegistry,
    add_coords,
    attach_coord_registry,
    get_coord_registry,
    registered_model,
    restore_inferencedata_coords,
    sanitize_coords_for_pymc,
)


def test_sanitize_coords_for_pymc_returns_range_coords() -> None:
    """Check coordinate sanitization produces range-based PyMC coords."""
    coords = {"nmeasure": xr.DataArray([10, 20, 30], dims=("nmeasure",))}
    result = sanitize_coords_for_pymc(coords)
    np.testing.assert_array_equal(result["nmeasure"], np.arange(3))


def test_coord_registry_stores_original_and_sanitized_multiindex() -> None:
    """Check the coord registry preserves MultiIndex coords and derived auxiliaries."""
    multi_index = pd.MultiIndex.from_arrays(
        [["MHD", "MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"])],
        names=["site", "time"],
    )
    coords = xr.Coordinates({"nmeasure": multi_index})
    registry = CoordRegistry()

    registry.add(coords)

    assert registry.original_coords["nmeasure"].equals(multi_index)
    np.testing.assert_array_equal(registry.pymc_coords["nmeasure"], np.arange(3))
    assert "time" in registry.auxiliary_coords


def test_coord_registry_repeated_registration_and_conflict() -> None:
    """Check repeated coordinate registration is idempotent but rejects conflicts."""
    coords = {"nx": np.array([0, 1, 2])}
    registry = CoordRegistry()
    registry.add(coords)
    registry.add(coords)

    with pytest.raises(ValueError, match="values differ"):
        registry.add({"nx": np.array([0, 2, 3])})


def test_add_coords_requires_registered_model() -> None:
    """Coordinate registration is an enforced model invariant."""
    coords = {"nmeasure": xr.DataArray([1, 2], dims=("nmeasure",))}

    with pm.Model():
        with pytest.raises(RuntimeError, match="registered_model"):
            add_coords(coords)

    with registered_model() as model:
        add_coords(coords)
        registry = get_coord_registry(model)
        assert registry is not None
        assert "nmeasure" in registry.original_coords


def test_registered_model_forwards_model_arguments() -> None:
    """Constructor coordinates are sanitized and scientifically registered."""
    with registered_model(coords={"state": ["west", "east"]}) as model:
        registry = get_coord_registry(model)
        assert registry is not None
        assert model.coords["state"] == (0, 1)
        np.testing.assert_array_equal(registry.original_coords["state"], ["west", "east"])


def test_registered_model_keeps_xarray_auxiliary_coordinates_out_of_pymc_dims() -> None:
    """Constructor xarray coords seed rich registry metadata, not false dims."""
    index = pd.MultiIndex.from_arrays(
        [["MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02"])],
        names=["site", "time"],
    )
    coords = xr.Coordinates.from_pandas_multiindex(index, "nmeasure")

    with registered_model(coords=coords) as model:
        registry = get_coord_registry(model)

    assert model.coords["nmeasure"] == (0, 1)
    assert "site" not in model.coords
    assert "time" not in model.coords
    assert registry is not None
    assert registry.original_coords["nmeasure"].equals(index)
    assert {"site", "time"} <= set(registry.auxiliary_coords)


def test_add_coords_preserves_auxiliary_coords_for_model_dims() -> None:
    """Check add_coords stores auxiliary coords attached to model dimensions."""
    multi_index = pd.MultiIndex.from_arrays(
        [["MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02"])],
        names=["site", "time"],
    )
    coords = xr.Coordinates.from_pandas_multiindex(multi_index, "nmeasure")

    with pm.Model() as model:
        registry = CoordRegistry()
        attach_coord_registry(model, registry)
        add_coords(coords, model_dims=("nmeasure",))

    assert "nmeasure" in registry.original_coords
    assert "site" in registry.auxiliary_coords
    assert "time" in registry.auxiliary_coords


def test_restore_inferencedata_coords_supports_registry_and_legacy_dict() -> None:
    """Check coordinate restoration works with both registry and legacy mappings."""
    multi_index = pd.MultiIndex.from_arrays(
        [["MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02"])],
        names=["site", "time"],
    )
    posterior = xr.Dataset(
        data_vars={"x": (("chain", "draw", "nmeasure"), np.zeros((1, 1, 2)))},
        coords={"chain": [0], "draw": [0], "nmeasure": np.arange(2)},
    )
    idata = az.InferenceData(posterior=posterior)
    registry = CoordRegistry(original_coords={"nmeasure": multi_index})

    restored = restore_inferencedata_coords(idata, registry)
    assert restored.posterior.indexes["nmeasure"].equals(multi_index)

    restored2 = restore_inferencedata_coords(idata, {"nmeasure": multi_index})
    assert restored2.posterior.indexes["nmeasure"].equals(multi_index)


def test_restore_inferencedata_auxiliary_coords_owns_group_values() -> None:
    """Keep restored groups and registry auxiliary values mutation-isolated."""
    auxiliary = xr.DataArray(
        [50.0, 51.0],
        dims=("state",),
        coords={"state": np.arange(2)},
        name="latitude",
    )
    registry = CoordRegistry(
        original_coords={"state": pd.Index(["ocean", "ff"], name="state")},
        auxiliary_coords={"latitude": auxiliary},
    )
    group = xr.Dataset(
        data_vars={"x": (("chain", "draw", "state"), np.zeros((1, 1, 2)))},
        coords={"chain": [0], "draw": [0], "state": np.arange(2)},
    )
    idata = az.InferenceData(posterior=group.copy(deep=True), prior=group.copy(deep=True))

    restored = restore_inferencedata_coords(idata, registry)
    restored.posterior["latitude"].values[0] = -999.0

    np.testing.assert_array_equal(registry.auxiliary_coords["latitude"], [50.0, 51.0])
    np.testing.assert_array_equal(restored.prior["latitude"], [50.0, 51.0])


def test_restore_inferencedata_preserves_independent_auxiliary_on_multiindex() -> None:
    """Restore non-level metadata attached to a MultiIndex state dimension."""
    state_index = pd.MultiIndex.from_tuples(
        [("ocean", "atlantic"), ("ff", "north")],
        names=("source", "region"),
    )
    coords = xr.Coordinates.from_pandas_multiindex(state_index, "state")
    mean = xr.DataArray(
        [1.0, 1.0],
        dims=("state",),
        coords={**coords, "latitude": ("state", [50.0, 51.0])},
    )
    registry = CoordRegistry()
    registry.add(mean.coords, model_dims=("state",))
    posterior = xr.Dataset(
        data_vars={"x": (("chain", "draw", "state"), np.zeros((1, 1, 2)))},
        coords={"chain": [0], "draw": [0], "state": np.arange(2)},
    )

    restored = restore_inferencedata_coords(az.InferenceData(posterior=posterior), registry)

    assert restored.posterior.indexes["state"].equals(state_index)
    np.testing.assert_array_equal(restored.posterior["latitude"], [50.0, 51.0])
