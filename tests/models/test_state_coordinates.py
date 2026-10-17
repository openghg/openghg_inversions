"""Full prior blocks and active states have distinct scientific coordinate names."""

import numpy as np
import pandas as pd
import pytest
import pytensor
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import (
    add_state_vector,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import get_coord_registry, registered_model
from openghg_inversions.models.priors import prior_records
from openghg_inversions.models.state_activity import StateActivity, resolve_state_activity
from openghg_inversions.rhime.multisector import build_multisector_model
from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec


def _build(h, priors, *, reverse=False):
    inputs = xr.Dataset({"H": h, "mf": ("nmeasure", [1., 2.]), "mf_error": ("nmeasure", [.1, .2])})
    sectors = (SectorSpec("FF", "ff", priors["x_ff"], "ff"),
               SectorSpec("other", "other", priors["x_ff_active"], "ff_active"))
    spec = RhimeModelSpec("ch4", "EUROPE", sectors[::-1] if reverse else sectors,
                          use_bc=False, likelihood=FixedErrorSettings())
    return build_multisector_model(inputs, spec, priors=priors).model


@pytest.mark.parametrize("reverse", [False, True])
def test_sector_suffix_active_does_not_collide_with_another_active_prior_dimension(reverse):
    index = pd.MultiIndex.from_tuples([("ff", "a"), ("ff", "b"), ("other", "c")],
                                      names=["source", "region_in_source"])
    h = xr.DataArray([[1., 2., 3.], [4., 5., 6.]], dims=("nmeasure", "state"),
                     coords={**xr.Coordinates.from_pandas_multiindex(index, "state"), "nmeasure": [0, 1]})
    priors = {
        "x_ff": {"pdf": "normal", "mu": xr.DataArray([2., 3.], dims="state", coords={"state": ["a", "b"]}),
                 "sigma": .2},
        "x_ff_active": {"pdf": "normal", "mu": xr.DataArray([4.], dims="state", coords={"state": ["c"]}),
                        "sigma": .3},
    }
    model = _build(h, priors, reverse=reverse)
    evaluate = pytensor.function([model["x_ff_active"], model["x_ff_active_active"]],
                                 [model["x"], model["mu"]])
    state, mean = evaluate(np.array([2., 3.]), np.array([4.]))
    np.testing.assert_array_equal(state, [2., 3., 4.])
    np.testing.assert_array_equal(mean, h.to_numpy() @ state)
    np.testing.assert_array_equal(model["x_ff_prior_mu"].get_value(), [2., 3.])
    np.testing.assert_array_equal(model["x_ff_active_prior_mu"].get_value(), [4.])
    assert set(prior_records(model)) == {"x_ff", "x_ff_active"}
    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.original_coords["state_prior_x_ff"].tolist() == ["a", "b"]
    assert registry.original_coords["state_active_x_ff"].tolist() == ["a", "b"]
    assert registry.original_coords["state_prior_x_ff_active"].tolist() == ["c"]
    assert registry.original_coords["state_active_x_ff_active"].tolist() == ["c"]


def test_sector_suffix_active_keeps_native_multiindex_prior_and_active_levels_distinct():
    local = pd.MultiIndex.from_tuples([("north", "Jan"), ("south", "Feb")], names=["basin", "period"])
    coords = xr.Coordinates.from_pandas_multiindex(local, "state")
    h = xr.DataArray([[[1., 0.], [2., 0.]], [[3., 4.], [5., 6.]]],
                     dims=("source", "nmeasure", "state"),
                     coords={**coords, "source": ["ff", "other"], "nmeasure": [0, 1]})
    priors = {name: {"pdf": "normal", "mu": xr.DataArray(values, dims="state", coords=coords), "sigma": .2}
              for name, values in (("x_ff", [2., 3.]), ("x_ff_active", [4., 5.]))}
    model = _build(h, priors)
    evaluate = pytensor.function([model["x_ff_active"], model["x_ff_active_active"]],
                                 [model["x"], model["mu"]])
    state, mean = evaluate(np.array([2.]), np.array([4., 5.]))
    np.testing.assert_array_equal(state, [2., 1., 4., 5.])
    np.testing.assert_array_equal(mean, [34., 54.])
    registry = get_coord_registry(model)
    assert registry is not None
    for record in ("x_ff", "x_ff_active"):
        assert registry.original_coords[f"state_prior_{record}"].names == [
            f"basin_prior_{record}", f"period_prior_{record}",
        ]
        assert registry.original_coords[f"state_active_{record}"].names == [
            f"basin_active_{record}", f"period_active_{record}",
        ]
    assert priors["x_ff"]["mu"].get_index("state").equals(local)


def test_independent_and_correlated_active_coordinates_keep_existing_names():
    index = pd.MultiIndex.from_tuples([("ff", "a"), ("ff", "b"), ("other", "c")],
                                      names=["source", "region_in_source"])
    coords = xr.Coordinates.from_pandas_multiindex(index, "state")
    zero = xr.DataArray([False, False, False], dims="state", coords=coords)
    activity = resolve_state_activity(zero, StateActivity(active=np.array([True, False, True])))
    with registered_model() as model:
        add_state_vector(activity, {"pdf": "normal", "mu": 1., "sigma": .2}, "x")
    registry = get_coord_registry(model)
    assert registry is not None
    expected = registry.original_coords["state_x_active"]
    assert expected.names == ["source_x_active", "region_in_source_x_active"]
    mean = xr.DataArray([1., 1., 1.], dims="state", coords=coords)
    prior = CorrelatedLognormalPrior(mean, np.eye(3) * .04)
    # Prior preparation remains usable without an active PyMC model.
    active_prior = prepare_active_correlated_lognormal_prior(activity, prior, var_name="x")
    assert active_prior is not None
    assert active_prior.state_dim == "state_x_active"
    assert active_prior.mean.get_index("state_x_active").equals(expected)
