"""Component-local scientific metadata must coexist in one model and trace."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import prepare_active_correlated_lognormal_prior
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.state_activity import (
    StateActivity, prepare_linear_sensitivity, resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.co2 import build_co2_model
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.multisector import build_multisector_model
from openghg_inversions.rhime.standard import build_standard_model
from openghg_inversions.rhime.nested import build_nested_rhime_model
from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec
from openghg_inversions.serialization import load_trace, save_trace


def _flux_sensitivity(layout):
    sensitivity = xr.DataArray(
        np.arange(1., 9.).reshape(2, 2, 2),
        dims=("source", "nmeasure", "region"),
        coords={
            "source": ["ff-inventory", "ocean-inventory"],
            "nmeasure": [0, 1],
            "region": ["a", "b"],
            "origin": ("region", ["flux-north", "flux-south"]),
            "site": ("nmeasure", ["MHD", "TAC"]),
        },
    )
    if layout == "gathered":
        return sensitivity.stack(state=("source", "region")).isel(state=[0, 1, 2])
    return sensitivity


def _boundary_sensitivity():
    return xr.DataArray(
        [[1., 2.], [3., 4.]], dims=("nmeasure", "bc_region"),
        coords={"nmeasure": [0, 1], "bc_region": ["north", "south"]},
    )


def _boundary_prior():
    return {
        "pdf": "normal", "mu": 1.,
        "sigma": xr.DataArray(
            [.2, .3], dims="bc_region",
            coords={
                "bc_region": ["north", "south"],
                "source": ("bc_region", ["boundary-inventory-a", "boundary-inventory-b"]),
                "origin": ("bc_region", ["boundary-north", "boundary-south"]),
            },
        ),
    }


def _multisector_model(sensitivity, boundary, bc_prior, *, partial_activity=False):
    prior = {"pdf": "normal", "mu": 1., "sigma": .2}
    inputs = xr.Dataset({
        "H": sensitivity, "H_bc": boundary,
        "mf": ("nmeasure", [1., 2.]), "mf_error": ("nmeasure", [.1, .2]),
    })
    spec = RhimeModelSpec(
        "ch4", "EUROPE",
        (SectorSpec("Fossil fuel", "ff-inventory", prior, "ff",
                    state_activity=StateActivity(active=np.array([True, False])) if partial_activity else None),
         SectorSpec("Ocean exchange", "ocean-inventory", prior, "ocean")),
        use_bc=True, bc_prior=bc_prior, likelihood=FixedErrorSettings(),
        bc_state_activity=StateActivity(active=np.array([True, False])) if partial_activity else None,
    )
    return build_multisector_model(
        inputs, spec, priors={"x_ff": prior, "x_ocean": prior, "bc": bc_prior},
    ).model


def _co2_model(sensitivity, boundary, bc_prior, *, partial_activity=False, cached=False):
    mean = xr.ones_like(sensitivity.isel(nmeasure=0, drop=True))
    retained_prior = CorrelatedLognormalPrior(mean, np.eye(mean.size) * .04)
    observations = xr.DataArray(
        [1., 2.], dims="nmeasure",
        coords={
            "nmeasure": [0, 1], "site": ("nmeasure", ["MHD", "TAC"]),
            "time": ("nmeasure", np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")),
        },
    )
    builder = build_co2_cached_sigma_model if cached else build_co2_model
    result = builder(
        sensitivity,
        retained_prior=retained_prior,
        fixed_prior_contribution=xr.zeros_like(observations).rename("fixed_prior_contribution"),
        observations=observations,
        observation_error=xr.full_like(observations, .1),
        aggregation_error=AggregationError("none", np.zeros(2)),
        boundary_sensitivity=boundary,
        bc_prior=bc_prior,
        state_activity=StateActivity(active=np.array([True, False, True])) if partial_activity else None,
        bc_state_activity=StateActivity(active=np.array([True, False])) if partial_activity else None,
        **({"tau_hours": 6., "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .5}} if cached else {}),
    )
    return result.model if cached else result


def _roundtrip_prior(model, variables, tmp_path):
    with model:
        trace = pm.sample_prior_predictive(draws=2, var_names=variables, random_seed=893)
    registry = get_coord_registry(model)
    assert registry is not None
    trace = restore_inferencedata_coords(trace, registry)
    path = tmp_path / "component-coordinates.nc"
    save_trace(trace, path)
    return load_trace(path)["prior"].to_dataset()


@pytest.mark.parametrize("layout", ["rectangular", "gathered"])
@pytest.mark.parametrize("partial_activity", [False, True])
def test_multisector_boundary_coordinates_roundtrip(layout, partial_activity, tmp_path):
    sensitivity = _flux_sensitivity(layout)
    boundary = _boundary_sensitivity()
    bc_prior = _boundary_prior()
    original_sensitivity = sensitivity.copy(deep=True)
    original_boundary = boundary.copy(deep=True)
    original_parameter = bc_prior["sigma"].copy(deep=True)
    model = _multisector_model(sensitivity, boundary, bc_prior, partial_activity=partial_activity)
    restored = _roundtrip_prior(model, ["x", "bc", "mu", "mu_bc"], tmp_path)

    expected_sources = ["ff-inventory", "ff-inventory", "ocean-inventory", "ocean-inventory"]
    expected_origins = ["flux-north", "flux-south", "flux-north", "flux-south"]
    size = restored.sizes["state"]
    assert restored["x"].dims == ("chain", "draw", "state")
    assert restored["x"].get_index("state").names == ["source", "region"]
    np.testing.assert_array_equal(restored.source, expected_sources[:size])
    np.testing.assert_array_equal(restored.origin, expected_origins[:size])
    np.testing.assert_array_equal(restored.bc_source, ["boundary-inventory-a", "boundary-inventory-b"])
    np.testing.assert_array_equal(restored.bc_origin, ["boundary-north", "boundary-south"])
    np.testing.assert_array_equal(restored.site, ["MHD", "TAC"])
    matrix = (sensitivity.to_numpy() if layout == "gathered" else
              np.column_stack([sensitivity.sel(source=source) for source in sensitivity.source.values]))
    np.testing.assert_allclose(restored.mu, restored.x.values @ matrix.T)
    np.testing.assert_allclose(restored.mu_bc, restored.bc.values @ boundary.values.T)
    xr.testing.assert_identical(sensitivity, original_sensitivity)
    xr.testing.assert_identical(boundary, original_boundary)
    xr.testing.assert_identical(bc_prior["sigma"], original_parameter)


@pytest.mark.parametrize("metadata_owner", ["prior", "sensitivity"])
@pytest.mark.parametrize("partial_activity", [False, True])
def test_co2_component_coordinates_roundtrip(metadata_owner, partial_activity, tmp_path):
    sensitivity = _flux_sensitivity("gathered")
    boundary = _boundary_sensitivity()
    bc_prior = _boundary_prior()
    if metadata_owner == "sensitivity":
        boundary = boundary.assign_coords({
            name: bc_prior["sigma"].coords[name] for name in ("source", "origin")
        })
        bc_prior["sigma"] = bc_prior["sigma"].drop_vars(["source", "origin"])
    original_sensitivity = sensitivity.copy(deep=True)
    original_boundary = boundary.copy(deep=True)
    original_parameter = bc_prior["sigma"].copy(deep=True)
    model = _co2_model(sensitivity, boundary, bc_prior, partial_activity=partial_activity)
    restored = _roundtrip_prior(
        model, ["flux_scaling", "bc", "co2_flux_contribution", "mu_bc"], tmp_path,
    )

    state_dim = model.named_vars_to_dims["flux_scaling"][0]
    assert restored.flux_scaling.get_index(state_dim).names == ["source", "region"]
    np.testing.assert_array_equal(restored.source, ["ff-inventory", "ff-inventory", "ocean-inventory"])
    np.testing.assert_array_equal(restored.origin, ["flux-north", "flux-south", "flux-north"])
    assert "flux_sector" not in restored.coords
    np.testing.assert_array_equal(restored.bc_source, ["boundary-inventory-a", "boundary-inventory-b"])
    np.testing.assert_array_equal(restored.bc_origin, ["boundary-north", "boundary-south"])
    np.testing.assert_array_equal(restored.site, ["MHD", "TAC"])
    np.testing.assert_allclose(
        restored.co2_flux_contribution, restored.flux_scaling.values @ sensitivity.values.T,
    )
    np.testing.assert_allclose(restored.mu_bc, restored.bc.values @ boundary.values.T)
    xr.testing.assert_identical(sensitivity, original_sensitivity)
    xr.testing.assert_identical(boundary, original_boundary)
    xr.testing.assert_identical(bc_prior["sigma"], original_parameter)


@pytest.mark.parametrize("recipe", ["multisector", "co2"])
def test_component_qualification_rejects_disagreeing_boundary_prior_metadata(recipe):
    sensitivity = _flux_sensitivity("gathered")
    boundary = _boundary_sensitivity()
    bc_prior = _boundary_prior()
    bc_prior["mu"] = xr.ones_like(bc_prior["sigma"]).assign_coords(
        origin=("bc_region", ["different-north", "different-south"]),
    )
    build = _multisector_model if recipe == "multisector" else _co2_model
    with pytest.raises(ValueError, match="Conflicting coord registration.*bc_origin"):
        build(sensitivity, boundary, bc_prior)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("partial_activity", [False, True])
def test_co2_flux_and_boundary_can_share_local_dimension_and_multiindex_names(cached, partial_activity, tmp_path):
    sensitivity = _flux_sensitivity("gathered")
    boundary = _boundary_sensitivity().rename(bc_region="state").assign_coords(
        source=("state", ["boundary-inventory-a", "boundary-inventory-b"]),
        region=("state", ["north", "south"]),
    ).set_index(state=["source", "region"])
    bc_prior = _boundary_prior()
    bc_prior["sigma"] = xr.DataArray(
        [.2, .3], dims="state", coords=boundary.isel(nmeasure=0, drop=True).coords,
    )
    original_sensitivity = sensitivity.copy(deep=True)
    original_boundary = boundary.copy(deep=True)
    original_parameter = bc_prior["sigma"].copy(deep=True)
    model = _co2_model(sensitivity, boundary, bc_prior, partial_activity=partial_activity, cached=cached)
    restored = _roundtrip_prior(
        model, ["flux_scaling", "bc", "co2_flux_contribution", "mu_bc"], tmp_path,
    )

    assert restored.flux_scaling.get_index("state").names == ["source", "region"]
    assert restored.bc.get_index("bc_state").names == ["bc_source", "bc_region"]
    np.testing.assert_array_equal(restored.source, ["ff-inventory", "ff-inventory", "ocean-inventory"])
    np.testing.assert_array_equal(restored.bc_source, ["boundary-inventory-a", "boundary-inventory-b"])
    np.testing.assert_allclose(
        restored.co2_flux_contribution, restored.flux_scaling.values @ sensitivity.values.T,
    )
    np.testing.assert_allclose(restored.mu_bc, restored.bc.values @ boundary.values.T)
    xr.testing.assert_identical(sensitivity, original_sensitivity)
    xr.testing.assert_identical(boundary, original_boundary)
    xr.testing.assert_identical(bc_prior["sigma"], original_parameter)


@pytest.mark.parametrize("active", [True, False, np.array([True, False, True])])
def test_correlated_flux_prior_rejects_conflicting_auxiliary_labels_before_activity_selection(active):
    sensitivity = _flux_sensitivity("gathered").reset_index("state").assign_coords(state=["a", "b", "c"])
    prior_mean = xr.ones_like(sensitivity.isel(nmeasure=0, drop=True)).assign_coords(
        source=("state", ["wrong-inventory", "ff-inventory", "ocean-inventory"]),
    )
    retained_prior = CorrelatedLognormalPrior(prior_mean, np.eye(3) * .04)
    activity = resolve_state_activity(prepare_linear_sensitivity(sensitivity).removed, StateActivity(active=active))
    with pytest.raises(ValueError, match="source"):
        prepare_active_correlated_lognormal_prior(activity, retained_prior, var_name="flux_scaling")


@pytest.mark.parametrize("nested", [False, True])
def test_standard_and_nested_boundary_source_metadata_preserves_flux_schema(nested, tmp_path):
    sensitivity = _flux_sensitivity("rectangular").sel(source="ff-inventory", drop=True).assign_coords(
        source=("region", ["inventory-a", "inventory-b"]),
    )
    boundary = _boundary_sensitivity()
    bc_prior = _boundary_prior()
    prior = {"pdf": "normal", "mu": 1., "sigma": .2}
    inputs = xr.Dataset({
        "H": sensitivity, "H_bc": boundary,
        "mf": ("nmeasure", [1., 2.]), "mf_error": ("nmeasure", [.1, .2]),
    })
    spec = RhimeModelSpec(
        "ch4", "EUROPE", (SectorSpec("Total", "inventory", prior, "total"),),
        use_bc=True, bc_prior=bc_prior, likelihood=FixedErrorSettings(),
    )
    if nested:
        inputs["H_inner"] = xr.DataArray(
            [[2.], [3.]], dims=("nmeasure", "inner_region"), coords={"inner_region": ["inner"]},
        )
        model = build_nested_rhime_model(
            inputs, spec, priors={"x_outer": prior, "x_inner": prior, "bc": bc_prior},
        ).model
        state_name = "x_outer"
    else:
        model = build_standard_model(inputs, spec, priors={"x": prior, "bc": bc_prior}).model
        state_name = "x"
    restored = _roundtrip_prior(model, [state_name, "bc", "mu_bc"], tmp_path)
    assert restored[state_name].dims == ("chain", "draw", "region")
    assert restored.bc.dims == ("chain", "draw", "bc_region")
    np.testing.assert_array_equal(restored.source, ["inventory-a", "inventory-b"])
    np.testing.assert_array_equal(restored.bc_source, ["boundary-inventory-a", "boundary-inventory-b"])
    np.testing.assert_allclose(restored.mu_bc, restored.bc.values @ boundary.values.T)


def test_scalar_metadata_is_shared_and_conflicts_are_rejected():
    from openghg_inversions.models.components import add_linear_component
    from openghg_inversions.models.coords import registered_model

    flux = xr.DataArray([[1., 2.]], dims=("nmeasure", "region"), coords={
        "region": ["a", "b"], "source": "inventory",
    })
    boundary = xr.DataArray([[3., 4.]], dims=("nmeasure", "bc_region"), coords={"bc_region": ["a", "b"]})
    parameter = xr.DataArray([.2, .3], dims="bc_region", coords={
        "bc_region": ["a", "b"], "source": "boundary-inventory",
    })
    with registered_model():
        add_linear_component(
            prepare_linear_sensitivity(flux), data_name="hx", var_name="x", output_name="mu",
            prior_args={"pdf": "normal", "mu": 1., "sigma": .2},
        )
        with pytest.raises(ValueError, match="Conflicting shared scalar coordinate 'source'.*agree across components"):
            add_linear_component(
                prepare_linear_sensitivity(boundary), data_name="hbc", var_name="bc", output_name="mu_bc",
                prior_args={"pdf": "normal", "mu": 1., "sigma": parameter}, coordinate_owner="bc",
            )


def test_correlated_coordinate_validation_materializes_shared_graph_once_without_mutation():
    import dask.array as da
    from dask import delayed
    from dataclasses import replace

    calls = []

    @delayed(pure=True)
    def labels():
        calls.append("computed")
        return np.array([1., 2.])

    lazy = da.from_delayed(labels(), shape=(2,), dtype=float)
    mean = xr.DataArray([1., 1.], dims="region", coords={
        "region": ["a", "b"], "origin": ("region", lazy),
    })
    prior = CorrelatedLognormalPrior(mean, np.eye(2) * .04)
    sensitivity = xr.DataArray([[1., 2.]], dims=("nmeasure", "region"), coords={
        "region": ["a", "b"], "origin": ("region", lazy),
    })
    activity = resolve_state_activity(prepare_linear_sensitivity(sensitivity).removed, None)
    activity = replace(activity, zero_sensitivity=activity.zero_sensitivity.assign_coords(origin=("region", lazy)))
    calls.clear()
    result = prepare_active_correlated_lognormal_prior(activity, prior, var_name="flux_scaling")
    assert result is prior
    assert calls == ["computed"]
    assert isinstance(prior.mean.origin.data, da.Array)
    assert isinstance(activity.zero_sensitivity.origin.data, da.Array)
