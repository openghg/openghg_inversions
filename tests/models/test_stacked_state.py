"""Independent source priors populate one labelled state and forward calculation."""

from dataclasses import replace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor
from scipy.stats import norm
import xarray as xr

from openghg_inversions.models._flux import _stack_multisector_sensitivity
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.priors import prior_records
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.rhime.multisector import build_multisector_model
from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec
from openghg_inversions.rhime.standard import build_standard_model


def _inputs(h: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({
        "H": h,
        "mf": xr.DataArray([1., 2.], dims="nmeasure", coords={"nmeasure": h.nmeasure}),
        "mf_error": xr.DataArray([.1, .2], dims="nmeasure", coords={"nmeasure": h.nmeasure}),
    })


def _evaluate(model: pm.Model, values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    function = pytensor.function(model.free_RVs, [model["x"], model["mu"]])
    return function(*(np.asarray(values[str(rv.name)], dtype=rv.dtype) for rv in model.free_RVs))


@pytest.mark.parametrize("local_dim", ["region", "state"])
def test_mixed_source_priors_preserve_labels_distributions_and_one_forward_model(local_dim):
    h = xr.DataArray(
        [[[1., 2.], [3., 4.]], [[-10., -20.], [-30., -40.]]],
        dims=("source", "nmeasure", local_dim),
        coords={"source": ["ff", "ocean"], "nmeasure": [0, 1], local_dim: ["north", "south"]},
    ).assign_coords(measurement_code=(("nmeasure", local_dim), [[1, 2], [3, 4]]))
    original = h.copy(deep=True)
    ff = {"pdf": "normal", "sigma": 2., "mu": xr.DataArray(
        [7., 5.], dims=local_dim, coords={local_dim: ["south", "north"]},
    )}
    ocean = {"pdf": "lognormal", "reparameterise": True,
             "mean": xr.DataArray([4., 3.], dims=local_dim, coords={local_dim: ["south", "north"]}),
             "stdev": xr.DataArray([.4, .3], dims=local_dim, coords={local_dim: ["south", "north"]})}
    spec = RhimeModelSpec(
        "ch4", "EUROPE", (SectorSpec("ocean", "ocean", ocean, "ocean"), SectorSpec("FF", "ff", ff, "ff")),
        use_bc=False, likelihood=FixedErrorSettings(),
    )
    built = build_multisector_model(_inputs(h), spec, priors={"x_ff": ff, "x_ocean": ocean})
    model = built.model
    state, mean = _evaluate(model, {
        "x_ff_active": np.array([2., -1.]), "x_ocean_active_latent": np.array([0., 0.]),
    })
    ocean_state = np.array([3., 4.]) / np.sqrt(1.01)
    np.testing.assert_allclose(state, np.r_[2., -1., ocean_state])
    np.testing.assert_allclose(mean, h.sel(source="ff").to_numpy() @ [2., -1.]
                               + h.sel(source="ocean").to_numpy() @ ocean_state)
    np.testing.assert_allclose(pm.logp(model["x_ff_active"], [2., -1.]).eval(),
                               norm.logpdf([2., -1.], loc=[5., 7.], scale=2.))
    np.testing.assert_allclose(pm.logp(model["x_ocean_active_latent"], [0., 0.]).eval(), norm.logpdf([0., 0.]))
    np.testing.assert_array_equal(model["x_ff_prior_mu"].get_value(), [5., 7.])
    np.testing.assert_array_equal(model["x_ocean_prior_mean"].get_value(), [3., 4.])
    assert set(prior_records(model)) == {"x_ff", "x_ocean"}
    assert {"x", "hx", "mu"}.issubset(model.named_vars)
    assert not {"x_ff", "x_ocean", "hx_ff", "hx_ocean", "mu_ff", "mu_ocean"}.intersection(model.named_vars)
    state_dim = model.named_vars_to_dims["x"][0]
    assert state_dim == ("flux_state" if local_dim == "state" else "state")
    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.original_coords[state_dim].tolist() == [
        ("ff", "north"), ("ff", "south"), ("ocean", "north"), ("ocean", "south"),
    ]
    np.testing.assert_array_equal(registry.auxiliary_coords["measurement_code"], [[1, 2, 1, 2], [3, 4, 3, 4]])
    xr.testing.assert_identical(h, original)


@pytest.mark.parametrize("freeze_all", [False, True])
def test_interleaved_ragged_state_preserves_source_overrides_fixed_values_and_zero_columns(freeze_all):
    index = pd.MultiIndex.from_tuples(
        [("ocean", "west"), ("ff", "north"), ("ocean", "east"), ("ff", "zero"), ("ocean", "zero")],
        names=["source", "region_in_source"],
    )
    h = xr.DataArray(
        [[4., 1., 6., 0., 0.], [2., 3., -2., 0., 0.]], dims=("nmeasure", "state"),
        coords={**xr.Coordinates.from_pandas_multiindex(index, "state"), "nmeasure": [0, 1],
                "basis_group": ("state", ["inner", "outer", "outer", "inner", "inner"])},
    )
    ff = {"pdf": "normal", "mu": xr.DataArray([20., 10.], dims="state", coords={"state": ["zero", "north"]}),
          "sigma": 1.}
    ocean = {"pdf": "normal", "mu": 1., "sigma": .2}
    ff_policy = StateActivity(
        active=xr.DataArray([True, False], dims="state", coords={"state": ["zero", "north"]}),
        fixed_value=xr.DataArray([3., 2.], dims="state", coords={"state": ["zero", "north"]}),
    )
    spec = RhimeModelSpec(
        "ch4", "EUROPE", (SectorSpec("FF", "ff", ff, "ff", state_activity=ff_policy),
                          SectorSpec("ocean", "ocean", ocean, "ocean")),
        use_bc=False, likelihood=FixedErrorSettings(),
        state_activity=StateActivity(active=not freeze_all, fixed_groups=("outer",), fixed_value=7.),
    )
    built = build_multisector_model(_inputs(h), spec, priors={"x_ff": ff, "x_ocean": ocean})
    state, mean = _evaluate(built.model, {"x_ocean_active": np.array([2.])})
    expected = np.array([7. if freeze_all else 2., 2., 7., 3., 7.])
    np.testing.assert_array_equal(state, expected)
    np.testing.assert_array_equal(mean, h.to_numpy() @ expected)
    np.testing.assert_array_equal(built.model["hx"].get_value(), h.to_numpy()[:, :3])
    np.testing.assert_array_equal(built.model["x_is_active"].get_value(), [not freeze_all, False, False, False, False])
    assert set(prior_records(built.model)) == (set() if freeze_all else {"x_ocean"})
    registry = get_coord_registry(built.model)
    assert registry is not None and registry.original_coords["state"].equals(index)


def test_standard_and_multisector_use_equivalent_independent_forward_calculations():
    h = xr.DataArray([[1., 2.], [3., 4.]], dims=("nmeasure", "region"),
                     coords={"nmeasure": [0, 1], "region": ["a", "b"]})
    prior = {"pdf": "normal", "mu": 1., "sigma": .2}
    spec = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("FF", "ff", prior, "ff"),),
                          use_bc=False, likelihood=FixedErrorSettings())
    standard = build_standard_model(_inputs(h), spec, priors={"x": prior}).model
    stacked = xr.concat([h, -2. * h], dim=xr.IndexVariable("source", ["ff", "ocean"]))
    multisector = build_multisector_model(
        _inputs(stacked), replace(spec, sectors=(*spec.sectors, SectorSpec("ocean", "ocean", prior, "ocean"))),
        priors={"x_ff": prior, "x_ocean": prior},
    ).model
    _, standard_mean = _evaluate(standard, {"x": np.array([2., 3.])})
    _, stacked_mean = _evaluate(multisector, {"x_ff_active": np.array([2., 3.]), "x_ocean_active": np.array([0., 0.])})
    np.testing.assert_array_equal(standard_mean, stacked_mean)
    assert [rv.name for rv in standard.free_RVs] == ["x"]


def test_frozen_source_prior_still_requires_exact_local_labels():
    h = xr.DataArray(np.ones((2, 2, 1)), dims=("source", "nmeasure", "region"),
                     coords={"source": ["ff", "ocean"], "nmeasure": [0, 1], "region": ["a"]})
    wrong = {"pdf": "normal", "mu": xr.DataArray([1.], dims="region", coords={"region": ["wrong"]}), "sigma": 1.}
    prior = {"pdf": "normal", "mu": 1., "sigma": 1.}
    spec = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("FF", "ff", wrong, "ff"),
                                           SectorSpec("ocean", "ocean", prior, "ocean")),
                          use_bc=False, likelihood=FixedErrorSettings(), state_activity=StateActivity(active=False))
    with pytest.raises(ValueError, match="labels must match"):
        build_multisector_model(_inputs(h), spec, priors={"x_ff": wrong, "x_ocean": prior})


def test_rectangular_stacking_borrows_lazy_data():
    h = xr.DataArray(da.ones((2, 2, 2), chunks=1), dims=("source", "nmeasure", "region"),
                     coords={"source": ["ff", "ocean"], "nmeasure": [0, 1], "region": ["a", "b"]})
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        stacked = _stack_multisector_sensitivity(h)
    assert not tasks and isinstance(stacked.data, da.Array)
    assert h.data is not stacked.data
    assert set(h.data.__dask_graph__()).issubset(stacked.data.__dask_graph__())


def test_stacked_dimension_does_not_replace_existing_auxiliary_coordinates():
    h = xr.DataArray(np.ones((2, 2, 2)), dims=("source", "nmeasure", "region"),
                     coords={"source": ["ff", "ocean"], "nmeasure": [0, 1], "region": ["a", "b"],
                             "state": ("region", [1, 2]), "flux_state": ("region", [3, 4])})
    stacked = _stack_multisector_sensitivity(h)
    assert stacked.dims == ("nmeasure", "flux_state_2")
    np.testing.assert_array_equal(stacked.coords["state"], [1, 2, 1, 2])
    np.testing.assert_array_equal(stacked.coords["flux_state"], [3, 4, 3, 4])


@pytest.mark.parametrize("boundary_coords", [("state",), ("state", "flux_state")])
@pytest.mark.parametrize("layout,metadata_owner", [
    ("rectangular", "sensitivity"), ("rectangular", "prior"), ("gathered", "prior"),
])
def test_stacked_trace_restores_flux_labels_and_boundary_auxiliary_coordinates(
    boundary_coords, layout, metadata_owner,
):
    h = xr.DataArray(
        np.arange(1., 9.).reshape(2, 2, 2), dims=("source", "nmeasure", "region"),
        coords={"source": ["ff", "ocean"], "nmeasure": [0, 1], "region": ["a", "b"]},
    )
    expected = pd.MultiIndex.from_product([["ff", "ocean"], ["a", "b"]], names=["source", "region"])
    if layout == "gathered":
        h = h.stack(state=("source", "region")).isel(state=[0, 1, 2])
        expected = expected[:3]
    inputs = _inputs(h).assign(H_bc=xr.DataArray(
        [[1., 2.], [3., 4.]], dims=("nmeasure", "bc_region"),
        coords={"nmeasure": [0, 1], "bc_region": ["north", "south"]},
    ))
    prior = {"pdf": "normal", "mu": 1., "sigma": .2}
    bc_prior = prior.copy()
    metadata = {name: ("bc_region", [10, 20]) for name in boundary_coords}
    if metadata_owner == "sensitivity":
        inputs = inputs.assign_coords(metadata)
    else:
        bc_prior["sigma"] = xr.DataArray([.2, .3], dims="bc_region", coords={
            "bc_region": ["north", "south"], **metadata,
        })
    original = inputs.copy(deep=True)
    spec = RhimeModelSpec(
        "ch4", "EUROPE", (SectorSpec("FF", "ff", prior, "ff"),
                          SectorSpec("ocean", "ocean", prior, "ocean")),
        use_bc=True, bc_prior=bc_prior, likelihood=FixedErrorSettings(),
    )
    model = build_multisector_model(inputs, spec, priors={"x_ff": prior, "x_ocean": prior, "bc": bc_prior}).model
    with model:
        trace = pm.sample_prior_predictive(draws=2, var_names=["x", "bc", "mu", "mu_bc"], random_seed=437)
    registry = get_coord_registry(model)
    assert registry is not None
    restored = restore_inferencedata_coords(trace, registry).prior
    state_dim = model.named_vars_to_dims["x"][0]
    assert restored["x"].get_index(state_dim).equals(expected)
    for source in ("ff", "ocean"):
        assert restored["x"].sel(source=source).get_index("region").equals(
            expected[expected.get_level_values("source") == source].droplevel("source"),
        )
    for name in boundary_coords:
        assert restored.coords[f"bc_{name}"].dims == ("bc_region",)
        np.testing.assert_array_equal(restored.coords[f"bc_{name}"], [10, 20])
    full_sensitivity = (h.to_numpy() if layout == "gathered"
                        else np.column_stack([h.sel(source=source) for source in h.source.values]))
    np.testing.assert_allclose(restored["mu"], restored["x"].values @ full_sensitivity.T)
    xr.testing.assert_identical(inputs, original)


def test_rectangular_sources_preserve_native_multiindex_state_and_prior_labels():
    local = pd.MultiIndex.from_tuples([("north", "Jan"), ("south", "Feb")], names=["basin", "month"])
    h = xr.DataArray(
        [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dims=("source", "nmeasure", "region"),
        coords={**xr.Coordinates.from_pandas_multiindex(local, "region"),
                "source": ["ff", "ocean"], "nmeasure": [0, 1]},
    )
    prior = {"pdf": "normal", "sigma": 1., "mu": xr.DataArray(
        [20., 10.], dims="region", coords=xr.Coordinates.from_pandas_multiindex(local[::-1], "region"),
    )}
    spec = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("FF", "ff", prior, "ff"),
                                           SectorSpec("ocean", "ocean", prior, "ocean")),
                          use_bc=False, likelihood=FixedErrorSettings())
    model = build_multisector_model(_inputs(h), spec, priors={"x_ff": prior, "x_ocean": prior}).model
    state, mean = _evaluate(model, {"x_ff_active": np.array([2., 3.]), "x_ocean_active": np.array([4., 5.])})
    np.testing.assert_array_equal(state, [2., 3., 4., 5.])
    np.testing.assert_array_equal(mean, h.sel(source="ff").to_numpy() @ [2., 3.]
                                  + h.sel(source="ocean").to_numpy() @ [4., 5.])
    np.testing.assert_array_equal(model["x_ff_prior_mu"].get_value(), [10., 20.])
    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.original_coords["state"].names == ["source", "basin", "month"]
    assert isinstance(registry.original_coords["region_active_x_ff"], pd.MultiIndex)
