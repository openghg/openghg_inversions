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


def test_add_coords_works_with_and_without_registry() -> None:
    """Check add_coords works whether or not a coord registry is attached."""
    coords = {"nmeasure": xr.DataArray([1, 2], dims=("nmeasure",))}

    with pm.Model() as model:
        add_coords(coords)
        assert "nmeasure" in model.coords
        assert get_coord_registry(model) is None

    with pm.Model() as model:
        registry = CoordRegistry()
        attach_coord_registry(model, registry)
        add_coords(coords)
        assert "nmeasure" in registry.original_coords


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
