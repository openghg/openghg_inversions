"""Shared boundary helpers preserve imports and warn about known constant overlap."""

from dataclasses import replace
import warnings

from dask import array as da, delayed
import numpy as np
import pandas as pd
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.models import (
    AdditiveBoundary, MultiplicativeBoundary, StateActivity, add_boundary, prepare_boundary,
    registered_model,
)
from openghg_inversions.models.additive_boundary import warn_boundary_offset_confounding


@pytest.mark.parametrize("channel", [None, "co2", "o2"])
@pytest.mark.parametrize("mean_response", [np.ones(3), np.array([1.0, 1.0 + 5e-13])])
def test_unit_boundary_mean_warns_for_unrestricted_offsets(channel, mean_response):
    with pytest.warns(UserWarning, match="likelihood identifies only their combined constant") as caught:
        warn_boundary_offset_confounding(mean_response, drop_first=False, channel=channel)
    assert len(caught) == 1
    if channel is not None:
        assert str(caught[0].message).startswith(f"{channel}: ")


@pytest.mark.parametrize(
    "mean_response,drop_first",
    [
        (np.ones(3), True),
        (np.full(3, 0.8), False),
        (np.array([1.0, 0.8]), False),
        (np.array([1.0, 1.0 + 2e-12]), False),
    ],
)
def test_warning_is_limited_to_unrestricted_unit_mean_response(mean_response, drop_first):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_boundary_offset_confounding(mean_response, drop_first=drop_first)
    assert not caught


def test_warning_rejects_lazy_payload_without_computing_it():
    calls = []

    @delayed
    def response():
        calls.append("computed")
        return np.ones(3)

    lazy_response = da.from_delayed(response(), shape=(3,), dtype=float)
    with pytest.raises(TypeError, match="eager NumPy array"):
        warn_boundary_offset_confounding(lazy_response, drop_first=False)
    assert calls == []

@pytest.fixture
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _native_boundary():
    coords = {"nmeasure": [10, 20, 30], "bc_region": ["north", "south"],
              "time": ("nmeasure", pd.to_datetime(["2020-01-03", "2020-02-03", "2020-02-05"]))}
    anomaly = xr.DataArray([[1., 2.], [3., 4.], [5., 6.]], dims=("nmeasure", "bc_region"),
                           coords=coords, attrs={"units": "ppm"})
    mass = xr.ones_like(anomaly).assign_attrs(units="1") / 2
    mass.attrs["units"] = "1"
    data = xr.Dataset({"H_bc_anomaly": anomaly, "H_bc_mass": mass,
                       "bc_reference": xr.DataArray(400., attrs={"units": "ppm"})},
                      attrs={"boundary_representation": "centered-v1"})
    observations = xr.DataArray([401., 402., 403.], dims="nmeasure",
                               coords={"nmeasure": [10, 20, 30], "time": coords["time"]},
                               attrs={"units": "ppm"})
    return data, observations


def _affine_expression(result):
    return pt.as_tensor_variable(result.fixed_contribution.values) + sum(
        (pt.dot(block.design.values, block.coefficients) for block in result.blocks), start=pt.zeros((3,)),
    )


@pytest.mark.parametrize("active", [[True, False], [False, False], [True, True]])
def test_centered_affine_keeps_fixed_activity_and_physical_coefficients(double_precision, active):
    data, observations = _native_boundary()
    bound = prepare_boundary(
        observations, boundary_operands=data, bc_prior={"pdf": "normal", "mu": 1., "sigma": .1},
        bc_state_activity=StateActivity(active=np.array(active), fixed_value=2.),
    )
    assert isinstance(bound, MultiplicativeBoundary)
    with registered_model():
        result = add_boundary(bound, owner="boundary")
        difference = result.term.likelihood_contribution - _affine_expression(result)
        np.testing.assert_allclose(difference.eval(), 0., atol=1e-12)
        assert len(result.sampled_rvs) == int(any(active))
        assert len(result.blocks) == int(any(active))
        if any(active):
            assert result.blocks[0].design.attrs["units"] == "ppm"


def test_absolute_affine_keeps_inactive_effects():
    data, observations = _native_boundary()
    design = data.H_bc_anomaly.rename("H_bc")
    bound = prepare_boundary(
        observations, boundary_sensitivity=design,
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .1},
        bc_state_activity=StateActivity(active=np.array([True, False]), fixed_value=3.),
    )
    with registered_model():
        result = add_boundary(bound)
        np.testing.assert_allclose((result.term.output - _affine_expression(result)).eval(), 0., atol=1e-12)
        np.testing.assert_allclose(result.fixed_contribution, [6., 12., 18.])
        assert result.term.observation_shift is None


@pytest.mark.parametrize("scale", [0., 2.])
@pytest.mark.parametrize("prior", [
    {"pdf": "uniform", "lower": -2., "upper": 2.},
    {"pdf": "lognormal", "mu": 0., "sigma": .1, "reparameterise": True},
])
def test_additive_nonseparable_weights_and_resolved_scale(double_precision, scale, prior):
    data, observations = _native_boundary()
    data["H_bc_mass"] = data.H_bc_mass.where(data.bc_region == "north", 0.).assign_attrs(units="1")
    weights = xr.DataArray([[1., 2.], [3., 7.]], dims=("bc_curtain", "bc_period"),
                           coords={"bc_curtain": ["north", "south"],
                                   "bc_period": pd.to_datetime(["2020-01-01", "2020-02-01"])},
                           attrs={"units": "1"})
    bound = prepare_boundary(
        observations, boundary_operands=data, bc_frequency="monthly",
        bc_mean_shift_prior=prior, bc_anomaly_scale=scale, bc_centering_weights=weights,
    )
    assert isinstance(bound, AdditiveBoundary)
    with registered_model() as model:
        result = add_boundary(bound, owner="boundary")
        assert len(result.blocks) == (2 if scale else 1)
        assert len(result.sampled_rvs) == (2 if scale else 1)
        assert ("bc_anomaly_latent" in model.named_vars) == bool(scale)
        assert model["bc_anomaly_scale"].eval() == scale
        np.testing.assert_allclose(
            (result.term.likelihood_contribution - _affine_expression(result)).eval(), 0., atol=1e-12,
        )
        if scale:
            basis = model["bc_anomaly_basis"].eval()
            np.testing.assert_allclose(weights.values.ravel() @ basis, 0., atol=1e-14)
            assert result.blocks[1].design.shape == (3, 3)
        if prior["pdf"] == "lognormal":
            assert result.sampled_rvs[0] is model["bc_mean_shift_latent"]
            assert result.blocks[0].coefficients is not result.sampled_rvs[0]
        else:
            assert result.sampled_rvs[0] is model["bc_mean_shift"]


def test_additive_preparation_borrows_lazy_shared_graph(double_precision):
    calls = []
    data, observations = _native_boundary()

    original = data.H_bc_anomaly.values

    @delayed
    def payload():
        calls.append("computed")
        return original

    lazy = da.from_delayed(payload(), shape=(3, 2), dtype=float)
    data["H_bc_anomaly"] = data.H_bc_anomaly.copy(data=lazy)
    data["H_bc_mass"] = (data.H_bc_anomaly * 0 + .5).assign_attrs(units="1")
    weights = xr.DataArray([1., 2.], dims="bc_curtain", coords={"bc_curtain": ["north", "south"]},
                           attrs={"units": "1"})
    bound = prepare_boundary(observations, boundary_operands=data, bc_mean_shift_prior={"pdf": "normal", "mu": 0., "sigma": 1.},
                             bc_anomaly_scale=0., bc_centering_weights=weights)
    assert calls == []
    with registered_model(), pytest.raises(ValueError, match="Materialize"):
        add_boundary(bound)
    assert calls == []
    eager = replace(bound, data=bound.data.compute(scheduler="synchronous"))
    assert calls == ["computed"]
    with registered_model():
        add_boundary(eager)
    assert calls == ["computed"]


def test_additive_rejects_mismatched_period_labels(double_precision):
    data, observations = _native_boundary()
    weights = xr.DataArray(np.ones((2, 2)), dims=("bc_curtain", "bc_period"),
                           coords={"bc_curtain": ["north", "south"],
                                   "bc_period": pd.to_datetime(["2020-01-02", "2020-02-02"])},
                           attrs={"units": "1"})
    with pytest.raises(ValueError, match="align"):
        prepare_boundary(observations, boundary_operands=data, bc_frequency="monthly",
                         bc_mean_shift_prior={"pdf": "normal", "mu": 0., "sigma": 1.},
                         bc_anomaly_scale=2., bc_centering_weights=weights)


@pytest.mark.parametrize("scale", [0., 2.])
def test_one_state_has_only_mean_even_with_positive_scale(double_precision, scale):
    data, observations = _native_boundary()
    data = data.sel(bc_region=["north"])
    weights = xr.DataArray([100.], dims="bc_curtain", coords={"bc_curtain": ["north"]}, attrs={"units": "percent"})
    bound = prepare_boundary(
        observations, boundary_operands=data,
        bc_mean_shift_prior={"pdf": "normal", "mu": 0., "sigma": 1.},
        bc_anomaly_scale=scale, bc_centering_weights=weights,
    )
    with registered_model() as model:
        result = add_boundary(bound)
        assert len(result.blocks) == len(result.sampled_rvs) == 1
        assert "bc_anomaly_latent" not in model.named_vars
        np.testing.assert_allclose(model["bc_centering_weights"].eval(), [1.])
        np.testing.assert_allclose((result.term.likelihood_contribution - _affine_expression(result)).eval(), 0., atol=1e-12)


@pytest.mark.parametrize("options,match", [
    ({"bc_anomaly_scale": 0.}, "require"),
    ({"bc_mean_shift_prior": {"pdf": "normal", "mu": 0., "sigma": 1.}}, "require"),
    ({"bc_mean_shift_prior": {"pdf": "normal"}, "bc_prior": {"pdf": "normal"}}, "replace"),
    ({"bc_centering_weights": xr.DataArray([1.])}, "require"),
    ({"bc_state_activity": StateActivity()}, "require"),
])
def test_boundary_adapter_rejects_incompatible_choices(options, match):
    data, observations = _native_boundary()
    with pytest.raises(ValueError, match=match):
        prepare_boundary(observations, boundary_operands=data, **options)


@pytest.mark.parametrize("additive", [False, True])
def test_adapter_rejects_both_boundary_representations(double_precision, additive):
    data, observations = _native_boundary()
    choices = {"bc_prior": {"pdf": "normal", "mu": 1., "sigma": .1}}
    if additive:
        choices = {"bc_mean_shift_prior": {"pdf": "normal", "mu": 0., "sigma": 1.},
                   "bc_anomaly_scale": 0., "bc_centering_weights": xr.DataArray([1., 1.])}
    with pytest.raises(ValueError, match="not both"):
        prepare_boundary(observations, boundary_sensitivity=data.H_bc_anomaly, boundary_operands=data, **choices)


def test_absolute_boundary_registers_cli_data_name():
    data, observations = _native_boundary()
    bound = prepare_boundary(observations, boundary_sensitivity=data.H_bc_anomaly,
                             bc_prior={"pdf": "normal", "mu": 1., "sigma": .1})
    with registered_model() as model:
        add_boundary(bound, prefix="co2_")
    assert "co2_hbc" in model.named_vars
    assert "co2_H_bc" not in model.named_vars


@pytest.mark.parametrize("frequency", ["monthly", "10D"])
def test_one_temporal_period_requires_explicit_weight_periods(double_precision, frequency):
    data, observations = _native_boundary()
    observations = observations.isel(nmeasure=[0])
    weights = xr.DataArray([1., 2.], dims="bc_curtain", coords={"bc_curtain": ["north", "south"]}, attrs={"units": "1"})
    choices = {"boundary_operands": data, "bc_frequency": frequency,
               "bc_mean_shift_prior": {"pdf": "normal", "mu": 0., "sigma": 1.},
               "bc_anomaly_scale": 0., "bc_centering_weights": weights}
    with pytest.raises(ValueError, match="explicit period-start"):
        prepare_boundary(observations, **choices)
    start = "2020-01-01" if frequency == "monthly" else "2020-01-03"
    choices["bc_centering_weights"] = weights.expand_dims(bc_period=pd.to_datetime([start]))
    bound = prepare_boundary(observations, **choices)
    assert bound.data.sizes["bc_region"] == 2


def test_centered_rejects_nonfinite_reference_shift(double_precision):
    data, observations = _native_boundary()
    data["H_bc_mass"] = xr.ones_like(data.H_bc_mass).assign_attrs(units="1")
    data["bc_reference"] = xr.DataArray(1e308, attrs={"units": "ppm"})
    bound = prepare_boundary(observations, boundary_operands=data,
                             bc_prior={"pdf": "normal", "mu": 1., "sigma": .1})
    with registered_model(), np.errstate(over="ignore"), pytest.raises(ValueError, match="reference shift must be finite"):
        add_boundary(bound)


@pytest.mark.parametrize("additive", [False, True])
def test_boundary_payload_materializes_data_prior_and_activity_once(double_precision, additive):
    from dask import compute
    from openghg_inversions.models import boundary_payload, replace_boundary_payload

    calls = []
    data, observations = _native_boundary()
    original = data.H_bc_anomaly.values

    @delayed
    def payload():
        calls.append("computed")
        return original

    lazy = da.from_delayed(payload(), shape=(3, 2), dtype=float)
    data["H_bc_anomaly"] = data.H_bc_anomaly.copy(data=lazy)
    if additive:
        mean = data.H_bc_anomaly.mean()
        bound = prepare_boundary(
            observations, boundary_operands=data,
            bc_mean_shift_prior={"pdf": "normal", "mu": mean, "sigma": 1.}, bc_anomaly_scale=0.,
            bc_centering_weights=xr.DataArray([1., 2.], dims="bc_curtain",
                                             coords={"bc_curtain": ["north", "south"]}, attrs={"units": "1"}),
        )
    else:
        state = data.H_bc_anomaly.mean("nmeasure")
        bound = prepare_boundary(
            observations, boundary_sensitivity=data.H_bc_anomaly,
            bc_prior={"pdf": "normal", "mu": state, "sigma": .1},
            bc_state_activity=StateActivity(active=state > 0, fixed_value=state,
                                           fixed_groups=(), group_coord="scientific_group"),
        )
    fields = boundary_payload(bound)
    assert calls == []
    assert fields["data"] is bound.data
    eager_fields, = compute(fields, scheduler="synchronous")
    assert calls == ["computed"]
    eager = replace_boundary_payload(bound, eager_fields)
    assert type(eager) is type(bound)
    if additive:
        assert eager.anomaly_scale == bound.anomaly_scale
        assert eager.mean_prior["mu"].item() == original.mean()
    else:
        assert eager.activity.group_coord == "scientific_group"
        assert not hasattr(eager.activity.active.data, "__dask_graph__")
        np.testing.assert_allclose(eager.prior["mu"], original.mean(axis=0))
    with registered_model():
        add_boundary(eager)
    assert calls == ["computed"]


@pytest.mark.parametrize("scale,count,active", [(0., 2, False), (2., 1, False), (2., 2, True)])
def test_output_roles_and_units_declare_resolved_anomaly_state(double_precision, scale, count, active):
    from openghg_inversions.models import boundary_output_roles, boundary_output_units

    data, observations = _native_boundary()
    data = data.isel(bc_region=slice(count))
    curtains = data.bc_region.values
    weights = xr.DataArray(np.ones(count), dims="bc_curtain", coords={"bc_curtain": curtains}, attrs={"units": "1"})
    bound = prepare_boundary(
        observations, boundary_operands=data,
        bc_mean_shift_prior={"pdf": "lognormal", "mu": 0., "sigma": .1, "reparameterise": True},
        bc_anomaly_scale=scale, bc_centering_weights=weights,
    )
    roles = boundary_output_roles(bound, prefix="co2_")
    units = boundary_output_units(bound, prefix="co2_")
    assert roles["co2_boundary_concentration"] == "co2_mu_bc"
    assert roles["co2_boundary_mean_shift"] == "co2_bc_mean_shift"
    assert ("co2_boundary_correction_anomaly" in roles) == active
    assert ("co2_boundary_anomaly_concentration" in roles) == active
    assert ("co2_bc_anomaly_latent" in units) == active
    assert units["co2_bc_mean_shift"] == "ppm"
    assert units["co2_bc_mean_shift_latent"] == "1"
    assert units["co2_H_bc_mass"] == units["co2_bc_centering_weights"] == "1"
    with registered_model() as model:
        add_boundary(bound, prefix="co2_")
    assert set(roles.values()).issubset(model.named_vars)
    if active:
        assert roles["co2_boundary_correction_anomaly"] == "co2_bc_anomaly"


@pytest.mark.parametrize("centered", [False, True])
def test_multiplicative_output_declarations_match_registered_quantities(double_precision, centered):
    from openghg_inversions.models import boundary_output_roles, boundary_output_units

    data, observations = _native_boundary()
    inputs = {"boundary_operands": data} if centered else {"boundary_sensitivity": data.H_bc_anomaly}
    bound = prepare_boundary(observations, bc_prior={"pdf": "normal", "mu": 1., "sigma": .1}, **inputs)
    roles = boundary_output_roles(bound, prefix="o2_")
    units = boundary_output_units(bound, prefix="o2_")
    assert roles["o2_boundary_scale"] == "o2_bc"
    assert units["o2_bc"] == "1"
    assert units["o2_mu_bc"] == "ppm"
    if centered:
        assert roles["o2_boundary_mass"] == "o2_H_bc_mass"
        assert units["o2_H_bc_mass"] == "1"
    else:
        assert roles["o2_boundary_sensitivity"] == "o2_hbc"
        assert units["o2_hbc"] == "ppm"
    with registered_model() as model:
        add_boundary(bound, prefix="o2_")
    assert set(roles.values()).issubset(model.named_vars)


def test_direct_additive_preparation_accepts_only_actual_resolved_inputs(double_precision):
    from openghg_inversions.models.additive_boundary import prepare_additive_boundary
    from openghg_inversions.models.boundary import bind_boundary_operands

    data, observations = _native_boundary()
    operands = bind_boundary_operands(data, observations)
    weights = xr.ones_like(operands.bc_region, dtype=float).assign_attrs(units="1")
    bound = prepare_additive_boundary(operands, weights, observations,
                                      {"pdf": "normal", "mu": 0., "sigma": 1.}, 0.)
    with registered_model() as model:
        result = add_boundary(bound)
    assert len(result.blocks) == 1
    assert "bc_anomaly_latent" not in model.named_vars


@pytest.mark.parametrize("state_dim", ["bc_region", "curtain"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("owner", [None, "o2_boundary"])
def test_absolute_boundary_preserves_original_coordinate_owner(state_dim, partial, owner):
    data, observations = _native_boundary()
    design = data.H_bc_anomaly.rename(bc_region=state_dim) if state_dim != "bc_region" else data.H_bc_anomaly
    bound = prepare_boundary(
        observations, boundary_sensitivity=design,
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .1},
        bc_state_activity=StateActivity(active=np.array([True, False])) if partial else None,
    )
    with registered_model() as model:
        add_boundary(bound, owner=owner)
    expected = state_dim if owner is None and state_dim.startswith("bc_") else f"{owner or 'bc'}_{state_dim}"
    assert model.named_vars_to_dims["bc"] == (expected,)
    assert model.named_vars_to_dims["hbc"] == ("nmeasure", expected)
    if partial:
        assert model.named_vars_to_dims["bc_active"] == (f"{expected}_bc_active",)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("owner,expected", [(None, "bc_region"), ("o2_boundary", "o2_boundary_bc_region")])
def test_centered_boundary_preserves_original_coordinate_owner(double_precision, partial, owner, expected):
    data, observations = _native_boundary()
    bound = prepare_boundary(
        observations, boundary_operands=data,
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .1},
        bc_state_activity=StateActivity(active=np.array([True, False])) if partial else None,
    )
    with registered_model() as model:
        add_boundary(bound, owner=owner)
    assert model.named_vars_to_dims["bc"] == (expected,)
    assert model.named_vars_to_dims["H_bc_anomaly"] == ("nmeasure", expected)
    assert model.named_vars_to_dims["H_bc_mass"] == ("nmeasure", expected)
    if partial:
        assert model.named_vars_to_dims["bc_active"] == (f"{expected}_bc_active",)


def test_additive_boundary_keeps_unowned_default(double_precision):
    from openghg_inversions.models import get_coord_registry

    data, observations = _native_boundary()
    bound = prepare_boundary(
        observations, boundary_operands=data,
        bc_mean_shift_prior={"pdf": "normal", "mu": 0., "sigma": 1.}, bc_anomaly_scale=0.,
        bc_centering_weights=xr.DataArray([1., 2.], dims="bc_curtain",
                                         coords={"bc_curtain": ["north", "south"]}, attrs={"units": "1"}),
    )
    with registered_model() as model:
        add_boundary(bound)
    assert "rhime_component" not in get_coord_registry(model).coordinate_attrs["bc_region"]
    assert model.named_vars_to_dims["bc_correction"] == ("bc_region",)
