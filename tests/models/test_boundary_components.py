"""Bound boundary components and the constant-offset confounding warning."""

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
    AdditiveBoundary, AdditiveChoice, CenteredOperands, MultiplicativeBoundary, MultiplicativeChoice,
    StateActivity, registered_model,
)
from openghg_inversions.models.components import (
    AffineBlock, AffineConcentrationTerm, warn_constant_offset_confounding,
)


def _mean_response_term(mean_response):
    """Affine term whose only design column is the boundary mean response ``M 1``."""
    design = xr.DataArray(np.asarray(mean_response)[:, None], dims=("nmeasure", "bc_mean"))
    return AffineConcentrationTerm.absolute(
        pt.zeros(design.shape[0]), fixed_contribution=xr.zeros_like(design.isel(bc_mean=0)),
        blocks=(AffineBlock(design, pt.ones(1)),), sampled_rvs=(),
    )


@pytest.mark.parametrize("channel", [None, "co2", "o2"])
@pytest.mark.parametrize("mean_response", [np.ones(3), np.array([1.0, 1.0 + 5e-13])])
def test_unit_boundary_mean_warns_for_unrestricted_offsets(channel, mean_response):
    with pytest.warns(UserWarning, match="likelihood identifies only their combined constant") as caught:
        warn_constant_offset_confounding([_mean_response_term(mean_response)], drop_first=False, channel=channel)
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
        warn_constant_offset_confounding([_mean_response_term(mean_response)], drop_first=drop_first)
    assert not caught


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


def _whole_window(weights, observations):
    """Label curtain weights with the whole-window period starting at the first observation."""
    return weights.expand_dims(bc_period=[observations.time.values.min()], axis=-1)


def _centered(data, weights=None):
    return replace(CenteredOperands.from_dataset(data), weights=weights)


def _affine_expression(result):
    return pt.as_tensor_variable(result.fixed_contribution.values) + sum(
        (pt.dot(block.design.values, block.coefficients) for block in result.blocks), start=pt.zeros((3,)),
    )


@pytest.mark.parametrize("active", [[True, False], [False, False], [True, True]])
def test_centered_affine_keeps_fixed_activity_and_physical_coefficients(double_precision, active):
    data, observations = _native_boundary()
    bound = MultiplicativeChoice(
        {"pdf": "normal", "mu": 1., "sigma": .1}, StateActivity(active=np.array(active), fixed_value=2.),
    ).bind(observations, CenteredOperands.from_dataset(data))
    assert isinstance(bound, MultiplicativeBoundary)
    with registered_model():
        result = bound.add(owner="boundary")
        difference = result.likelihood_contribution - _affine_expression(result)
        np.testing.assert_allclose(difference.eval(), 0., atol=1e-12)
        assert len(result.sampled_rvs) == int(any(active))
        assert len(result.blocks) == int(any(active))
        if any(active):
            assert result.blocks[0].design.attrs["units"] == "ppm"


def test_boundary_families_report_whether_transport_is_centered(double_precision):
    """Every boundary family exposes the ``centered`` protocol property."""
    data, observations = _native_boundary()
    prior = {"pdf": "normal", "mu": 1., "sigma": .1}
    absolute = MultiplicativeChoice(prior).bind(observations, data.H_bc_anomaly.rename("H_bc"))
    centered = MultiplicativeChoice(prior).bind(observations, CenteredOperands.from_dataset(data))
    one_state = data.sel(bc_region=["north"])
    weights = _whole_window(
        xr.DataArray([1.], dims="bc_curtain", coords={"bc_curtain": ["north"]}, attrs={"units": "1"}), observations,
    )
    additive = AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0.).bind(
        observations, _centered(one_state, weights),
    )
    assert (absolute.centered, centered.centered, additive.centered) == (False, True, True)


def test_absolute_affine_keeps_inactive_effects():
    data, observations = _native_boundary()
    design = data.H_bc_anomaly.rename("H_bc")
    bound = MultiplicativeChoice(
        {"pdf": "normal", "mu": 1., "sigma": .1}, StateActivity(active=np.array([True, False]), fixed_value=3.),
    ).bind(observations, design)
    with registered_model():
        result = bound.add()
        np.testing.assert_allclose((result.output - _affine_expression(result)).eval(), 0., atol=1e-12)
        np.testing.assert_allclose(result.fixed_contribution, [6., 12., 18.])
        assert result.observation_shift is None


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
    bound = AdditiveChoice(prior, scale, "monthly").bind(observations, _centered(data, weights))
    assert isinstance(bound, AdditiveBoundary)
    with registered_model() as model:
        result = bound.add(owner="boundary")
        assert len(result.blocks) == (2 if scale else 1)
        assert len(result.sampled_rvs) == (2 if scale else 1)
        assert ("bc_anomaly_latent" in model.named_vars) == bool(scale)
        assert model["bc_anomaly_scale"].eval() == scale
        np.testing.assert_allclose(
            (result.likelihood_contribution - _affine_expression(result)).eval(), 0., atol=1e-12,
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
    weights = _whole_window(xr.DataArray([1., 2.], dims="bc_curtain", coords={"bc_curtain": ["north", "south"]},
                                         attrs={"units": "1"}), observations)
    bound = AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0.).bind(observations, _centered(data, weights))
    assert calls == []
    with registered_model(), pytest.raises(ValueError, match="Materialize"):
        bound.add()
    assert calls == []
    eager = replace(bound, data=bound.data.compute(scheduler="synchronous"))
    assert calls == ["computed"]
    with registered_model():
        eager.add()
    assert calls == ["computed"]


def test_additive_rejects_mismatched_period_labels(double_precision):
    data, observations = _native_boundary()
    weights = xr.DataArray(np.ones((2, 2)), dims=("bc_curtain", "bc_period"),
                           coords={"bc_curtain": ["north", "south"],
                                   "bc_period": pd.to_datetime(["2020-01-02", "2020-02-02"])},
                           attrs={"units": "1"})
    with pytest.raises(ValueError, match="align"):
        AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 2., "monthly").bind(
            observations, _centered(data, weights),
        )


@pytest.mark.parametrize("scale", [0., 2.])
def test_one_state_has_only_mean_even_with_positive_scale(double_precision, scale):
    data, observations = _native_boundary()
    data = data.sel(bc_region=["north"])
    weights = _whole_window(
        xr.DataArray([100.], dims="bc_curtain", coords={"bc_curtain": ["north"]}, attrs={"units": "1"}), observations,
    )
    bound = AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, scale).bind(observations, _centered(data, weights))
    with registered_model() as model:
        result = bound.add()
        assert len(result.blocks) == len(result.sampled_rvs) == 1
        assert "bc_anomaly_latent" not in model.named_vars
        np.testing.assert_allclose(model["bc_centering_weights"].eval(), [1.])
        np.testing.assert_allclose((result.likelihood_contribution - _affine_expression(result)).eval(), 0., atol=1e-12)


def test_additive_binding_requires_centering_weights():
    data, observations = _native_boundary()
    with pytest.raises(ValueError, match="require bc_centering_weights"):
        AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0.).bind(
            observations, CenteredOperands.from_dataset(data),
        )
def test_absolute_boundary_registers_cli_data_name():
    data, observations = _native_boundary()
    bound = MultiplicativeChoice({"pdf": "normal", "mu": 1., "sigma": .1}).bind(observations, data.H_bc_anomaly)
    with registered_model() as model:
        bound.add(prefix="co2_")
    assert "co2_hbc" in model.named_vars
    assert "co2_H_bc" not in model.named_vars


@pytest.mark.parametrize("frequency", ["monthly", "10D"])
def test_one_temporal_period_binds_explicit_weight_periods(double_precision, frequency):
    data, observations = _native_boundary()
    observations = observations.isel(nmeasure=[0])
    start = "2020-01-01" if frequency == "monthly" else "2020-01-03"
    weights = xr.DataArray([[1.], [2.]], dims=("bc_curtain", "bc_period"),
                           coords={"bc_curtain": ["north", "south"], "bc_period": pd.to_datetime([start])},
                           attrs={"units": "1"})
    bound = AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0., frequency).bind(
        observations, _centered(data, weights),
    )
    assert bound.data.sizes["bc_region"] == 2


def test_centered_rejects_nonfinite_reference_shift(double_precision):
    data, observations = _native_boundary()
    data["H_bc_mass"] = xr.ones_like(data.H_bc_mass).assign_attrs(units="1")
    data["bc_reference"] = xr.DataArray(1e308, attrs={"units": "ppm"})
    bound = MultiplicativeChoice({"pdf": "normal", "mu": 1., "sigma": .1}).bind(
        observations, CenteredOperands.from_dataset(data),
    )
    with registered_model(), np.errstate(over="ignore"), pytest.raises(ValueError, match="reference shift must be finite"):
        bound.add()


@pytest.mark.parametrize("additive", [False, True])
def test_boundary_payload_materializes_data_prior_and_activity_once(double_precision, additive):
    from dask import compute

    from openghg_inversions.array_ops import to_dense_tree

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
        weights = _whole_window(xr.DataArray([1., 2.], dims="bc_curtain",
                                             coords={"bc_curtain": ["north", "south"]}, attrs={"units": "1"}),
                                observations)
        bound = AdditiveChoice({"pdf": "normal", "mu": mean, "sigma": 1.}, 0.).bind(
            observations, _centered(data, weights),
        )
    else:
        state = data.H_bc_anomaly.mean("nmeasure")
        bound = MultiplicativeChoice(
            {"pdf": "normal", "mu": state, "sigma": .1},
            StateActivity(active=state > 0, fixed_value=state, fixed_groups=(), group_coord="scientific_group"),
        ).bind(observations, data.H_bc_anomaly)
    fields = to_dense_tree(bound)
    assert calls == []
    eager, = compute(fields, scheduler="synchronous")
    assert calls == ["computed"]
    assert type(eager) is type(bound)
    assert not any(hasattr(value.data, "__dask_graph__") for value in eager.data.data_vars.values())
    if additive:
        assert eager.choice.anomaly_scale == bound.choice.anomaly_scale
        assert eager.choice.mean_prior["mu"].item() == original.mean()
    else:
        assert eager.choice.activity.group_coord == "scientific_group"
        assert not hasattr(eager.choice.activity.active.data, "__dask_graph__")
        np.testing.assert_allclose(eager.choice.prior["mu"], original.mean(axis=0))
    with registered_model():
        eager.add()
    assert calls == ["computed"]


@pytest.mark.parametrize("scale,count,active", [(0., 2, False), (2., 1, False), (2., 2, True)])
def test_output_roles_and_units_declare_resolved_anomaly_state(double_precision, scale, count, active):
    data, observations = _native_boundary()
    data = data.isel(bc_region=slice(count))
    curtains = data.bc_region.values
    weights = _whole_window(
        xr.DataArray(np.ones(count), dims="bc_curtain", coords={"bc_curtain": curtains}, attrs={"units": "1"}),
        observations,
    )
    bound = AdditiveChoice(
        {"pdf": "lognormal", "mu": 0., "sigma": .1, "reparameterise": True}, scale,
    ).bind(observations, _centered(data, weights))
    outputs = bound.outputs(prefix="co2_")
    roles, units = outputs.roles, outputs.units
    assert roles["co2_boundary_concentration"] == "co2_mu_bc"
    assert roles["co2_boundary_mean_shift"] == "co2_bc_mean_shift"
    assert ("co2_boundary_correction_anomaly" in roles) == active
    assert ("co2_boundary_anomaly_concentration" in roles) == active
    assert ("co2_bc_anomaly_latent" in units) == active
    assert units["co2_bc_mean_shift"] == "ppm"
    assert units["co2_bc_mean_shift_latent"] == "1"
    assert units["co2_H_bc_mass"] == units["co2_bc_centering_weights"] == "1"
    with registered_model() as model:
        bound.add(prefix="co2_")
    assert set(roles.values()).issubset(model.named_vars)
    if active:
        assert roles["co2_boundary_correction_anomaly"] == "co2_bc_anomaly"


@pytest.mark.parametrize("centered", [False, True])
def test_multiplicative_output_declarations_match_registered_quantities(double_precision, centered):
    data, observations = _native_boundary()
    transport = CenteredOperands.from_dataset(data) if centered else data.H_bc_anomaly
    bound = MultiplicativeChoice({"pdf": "normal", "mu": 1., "sigma": .1}).bind(observations, transport)
    outputs = bound.outputs(prefix="o2_")
    roles, units = outputs.roles, outputs.units
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
        bound.add(prefix="o2_")
    assert set(roles.values()).issubset(model.named_vars)


def test_direct_additive_record_accepts_only_actual_resolved_inputs(double_precision):
    data, observations = _native_boundary()
    operands = CenteredOperands.from_dataset(data).bind(observations)
    weights = xr.ones_like(operands.bc_region, dtype=float).assign_attrs(units="1")
    bound = AdditiveBoundary(
        AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0.),
        operands.assign(bc_centering_weights=weights),
    )
    with registered_model() as model:
        result = bound.add()
    assert len(result.blocks) == 1
    assert "bc_anomaly_latent" not in model.named_vars


@pytest.mark.parametrize("state_dim", ["bc_region", "curtain"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("owner", [None, "o2_boundary"])
def test_absolute_boundary_preserves_original_coordinate_owner(state_dim, partial, owner):
    data, observations = _native_boundary()
    design = data.H_bc_anomaly.rename(bc_region=state_dim) if state_dim != "bc_region" else data.H_bc_anomaly
    bound = MultiplicativeChoice(
        {"pdf": "normal", "mu": 1., "sigma": .1},
        StateActivity(active=np.array([True, False])) if partial else None,
    ).bind(observations, design)
    with registered_model() as model:
        bound.add(**({} if owner is None else {"owner": owner}))  # None selects the default owner.
    expected = state_dim if owner is None and state_dim.startswith("bc_") else f"{owner or 'bc'}_{state_dim}"
    assert model.named_vars_to_dims["bc"] == (expected,)
    assert model.named_vars_to_dims["hbc"] == ("nmeasure", expected)
    if partial:
        assert model.named_vars_to_dims["bc_active"] == (f"{expected}_bc_active",)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("owner,expected", [(None, "bc_region"), ("o2_boundary", "o2_boundary_bc_region")])
def test_centered_boundary_preserves_original_coordinate_owner(double_precision, partial, owner, expected):
    data, observations = _native_boundary()
    bound = MultiplicativeChoice(
        {"pdf": "normal", "mu": 1., "sigma": .1},
        StateActivity(active=np.array([True, False])) if partial else None,
    ).bind(observations, CenteredOperands.from_dataset(data))
    with registered_model() as model:
        bound.add(**({} if owner is None else {"owner": owner}))  # None selects the default owner.
    assert model.named_vars_to_dims["bc"] == (expected,)
    assert model.named_vars_to_dims["H_bc_anomaly"] == ("nmeasure", expected)
    assert model.named_vars_to_dims["H_bc_mass"] == ("nmeasure", expected)
    if partial:
        assert model.named_vars_to_dims["bc_active"] == (f"{expected}_bc_active",)


@pytest.mark.parametrize("owner,component", [((), "bc"), ((None,), None)])
def test_additive_boundary_default_owner_keeps_local_state_names(double_precision, owner, component):
    from openghg_inversions.models import get_coord_registry

    data, observations = _native_boundary()
    weights = _whole_window(xr.DataArray([1., 2.], dims="bc_curtain",
                                         coords={"bc_curtain": ["north", "south"]}, attrs={"units": "1"}),
                            observations)
    bound = AdditiveChoice({"pdf": "normal", "mu": 0., "sigma": 1.}, 0.).bind(observations, _centered(data, weights))
    with registered_model() as model:
        bound.add(**dict(zip(("owner",), owner)))
    assert get_coord_registry(model).coordinate_attrs["bc_region"].get("rhime_component") == component
    assert model.named_vars_to_dims["bc_correction"] == ("bc_region",)


@pytest.mark.parametrize("kind", ["absolute", "centered", "additive"])
def test_boundary_metadata_uses_choice_facts_without_evaluating_lazy_inputs(double_precision, kind):
    data, observations = _native_boundary()
    prior = {"pdf": "normal", "mu": 1., "sigma": .1}
    if kind == "additive":
        weights = xr.DataArray(np.ones((2, 2)), dims=("bc_curtain", "bc_period"),
                              coords={"bc_curtain": ["north", "south"],
                                      "bc_period": pd.to_datetime(["2020-01-01", "2020-02-01"])},
                              attrs={"units": "1"})
        bound = AdditiveChoice(prior, 0., "monthly").bind(observations, _centered(data, weights))
        expected = {"boundary_formulation": "additive", "boundary_representation": "centered-v1",
                    "bc_frequency": "monthly", "bc_anchor_time": None}
    else:
        transport = data.H_bc_anomaly if kind == "absolute" else CenteredOperands.from_dataset(data)
        bound = MultiplicativeChoice(prior, frequency="10D", anchor_time="2020-01-01").bind(observations, transport)
        expected = {"boundary_formulation": "multiplicative", "bc_frequency": "10D", "bc_anchor_time": "2020-01-01"}
        if kind == "centered":
            expected["boundary_representation"] = "centered-v1"
    operator_name = "H_bc" if kind == "absolute" else "H_bc_anomaly"
    operator = bound.data[operator_name]

    @delayed
    def forbidden_payload():
        raise AssertionError("metadata must not evaluate payloads or auxiliary coordinates")

    lazy = da.from_delayed(forbidden_payload(), shape=operator.shape, dtype=float)
    auxiliary = da.from_delayed(forbidden_payload(), shape=(3,), dtype=float)
    lazy_data = bound.data.copy(deep=False)
    lazy_data[operator_name].data = lazy
    lazy_data = lazy_data.assign_coords(diagnostic=("nmeasure", auxiliary)).assign_attrs(
        bc_frequency="wrong-dataset-frequency", bc_anchor_time="wrong-dataset-anchor",
    )
    lazy_bound = replace(bound, data=lazy_data)
    assert lazy_bound.outputs().metadata == expected
    assert lazy_bound.data is lazy_data
    assert hasattr(lazy_bound.data[operator_name].data, "__dask_graph__")
    # A whole-window choice records no temporal subdivision or anchor, whatever
    # the transport's preparation sentinels say.
    unperiodized = lazy_data.copy(deep=False)
    unperiodized[operator_name].attrs = dict(operator.attrs, bc_frequency="none", bc_anchor_time="not-applicable")
    expected.update(bc_frequency=None, bc_anchor_time=None)
    whole_window = replace(bound.choice, frequency=None, anchor_time=None)
    assert replace(lazy_bound, choice=whole_window, data=unperiodized).outputs().metadata == expected
