"""Tests for labelled active/fixed state-vector model construction."""

from __future__ import annotations

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.models import (
    CoordRegistry,
    StateActivity,
    active_prior_args,
    add_state_linear_component,
    attach_coord_registry,
    build_rhime_model,
    build_rhime_multisector_model,
    resolve_state_activity,
)
from openghg_inversions.basis import project_basis_prior_stdev
from openghg_inversions.basis.basis_functions import BasisFunctions


def _sensitivity() -> xr.DataArray:
    """Return labelled sensitivity with exact-zero and near-zero columns."""
    return xr.DataArray(
        [
            [1.0, 0.0, 2.0, 1.0e-12],
            [3.0, 0.0, 4.0, -1.0e-12],
        ],
        dims=("nmeasure", "region"),
        coords={
            "nmeasure": [0, 1],
            "region": ["inner-a", "zero", "outer-a", "inner-b"],
            "basis_group": ("region", ["inner", "inner", "outer", "inner"]),
        },
        name="H",
    )


def _model_inputs(h: xr.DataArray) -> xr.Dataset:
    """Return the minimal complete dataset needed by RHIME model builders."""
    nmeasure = h.sizes["nmeasure"]
    return xr.Dataset(
        {
            "H": h,
            "mf": ("nmeasure", np.full(nmeasure, 2.0)),
            "mf_error": ("nmeasure", np.full(nmeasure, 0.1)),
            "min_error": ("nmeasure", np.zeros(nmeasure)),
            "site_indicator": ("nmeasure", np.zeros(nmeasure, dtype=int)),
            "sigma_freq_index": ("nmeasure", np.zeros(nmeasure, dtype=int)),
        }
    )


def test_resolve_state_activity_combines_labels_groups_and_exact_zero() -> None:
    """Resolve masks by labels while retaining nonzero values of any magnitude."""
    h = _sensitivity()
    explicit = xr.DataArray(
        [True, False, True, True],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    resolved = resolve_state_activity(
        h,
        StateActivity(active=explicit, fixed_groups=("outer",)),
    )

    np.testing.assert_array_equal(resolved.zero_sensitivity, [False, True, False, False])
    np.testing.assert_array_equal(resolved.active, [True, False, False, True])
    assert resolved.n_active == 2
    np.testing.assert_array_equal(resolved.active_indices, [0, 3])


def test_active_prior_args_aligns_labelled_and_array_parameters() -> None:
    """Subset labelled and positional full-state prior arrays to active order."""
    h = _sensitivity()
    resolved = resolve_state_activity(h, StateActivity(fixed_groups=("outer",)))
    labelled_mu = xr.DataArray(
        [40.0, 30.0, 20.0, 10.0],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    prior = active_prior_args(
        {
            "pdf": "normal",
            "mu": labelled_mu,
            "sigma": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        resolved,
    )

    np.testing.assert_array_equal(prior["mu"], [10.0, 40.0])
    np.testing.assert_array_equal(prior["sigma"], [1.0, 4.0])


def test_rhime_model_accepts_projected_labelled_prior_stdev() -> None:
    """The labelled basis projection passes directly through the model prior API."""
    basis = xr.DataArray(
        [[1, 1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]},
    )
    flux = xr.DataArray(
        [[1.0, 1.0, 2.0]],
        dims=("lat", "lon"),
        coords=basis.coords,
    )
    basis_functions = BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": "region"},
    )
    projected = project_basis_prior_stdev(
        basis_functions,
        area_grid=xr.ones_like(flux),
        grid_cell_prior_stdev=0.4,
    )
    sensitivity = xr.DataArray(
        [[1.0, 0.0], [0.0, 2.0]],
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": projected["region"]},
        name="H",
    )

    model = build_rhime_model(
        _model_inputs(sensitivity),
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": projected},
        use_bc=False,
        no_model_error=True,
    )

    assert model.named_vars["x_active"].eval().shape == (2,)


def test_state_linear_component_preserves_full_forward_identity() -> None:
    """Full H/state multiplication equals active plus fixed contributions."""
    h = _sensitivity()
    fixed_value = xr.DataArray(
        [4.0, 3.0, 2.0, 1.0],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_state_linear_component(
            h,
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 2.0, "sigma": 0.1},
            var_name="x",
            output_name="mu",
            state_activity=StateActivity(
                fixed_value=fixed_value,
                fixed_groups=("outer",),
            ),
        )

    x_full, mu, x_active = pm.draw(
        [result.state, result.output, model.named_vars["x_active"]],
        random_seed=42,
    )
    active = result.activity.active_indices
    fixed = result.activity.fixed_indices
    expected_split = (
        h.values[:, active] @ x_active + h.values[:, fixed] @ fixed_value.sel(region=h.region).values[fixed]
    )

    np.testing.assert_allclose(h.values @ x_full, mu)
    np.testing.assert_allclose(expected_split, mu)
    np.testing.assert_array_equal(model.named_vars["x_is_active"].eval(), [True, False, False, True])
    assert model.named_vars["x"].name == "x"
    assert model.named_vars["x_active"] in model.free_RVs


def test_state_linear_component_supports_zero_active_states() -> None:
    """An all-fixed component creates no active prior and retains its full state."""
    h = _sensitivity()
    policy = StateActivity(active=False, fixed_value=2.5)

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_state_linear_component(
            h,
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            var_name="x",
            output_name="mu",
            state_activity=policy,
        )

    x_full, mu = pm.draw([result.state, result.output], random_seed=42)
    np.testing.assert_allclose(x_full, np.full(h.sizes["region"], 2.5))
    np.testing.assert_allclose(mu, h.values @ x_full)
    assert result.latent is None
    assert "x_active" not in model.named_vars


def test_standard_rhime_uses_full_deterministic_x_and_active_prior() -> None:
    """The standard builder prunes exact-zero H but preserves full ordered x."""
    h = _sensitivity()
    model = build_rhime_model(
        _model_inputs(h),
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        use_bc=False,
        no_model_error=True,
    )

    assert {"x", "x_active", "x_is_active", "x_fixed_value", "mu"}.issubset(model.named_vars)
    assert model.named_vars["x_active"].eval().shape == (3,)
    np.testing.assert_array_equal(model.named_vars["x_is_active"].eval(), [True, False, True, True])
    assert model.named_vars["x"] not in model.free_RVs


def test_multisector_rhime_can_freeze_a_sector_and_use_array_priors() -> None:
    """Per-sector policies can freeze all states while other sectors sample arrays."""
    h = _sensitivity()
    multi_h = xr.concat(
        [h.expand_dims(source=["ff-source"]), (2.0 * h).expand_dims(source=["ocean-source"])],
        dim="source",
    )
    inputs = _model_inputs(multi_h)
    ocean_mu = xr.DataArray(
        [1.4, 1.3, 1.2, 1.1],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    model = build_rhime_multisector_model(
        inputs,
        sectors=["FF", "ocean"],
        sector_sources={"FF": "ff-source", "ocean": "ocean-source"},
        sector_priors={
            "FF": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            "ocean": {"pdf": "normal", "mu": ocean_mu, "sigma": 0.3},
        },
        sector_state_activities={"FF": StateActivity(active=False)},
        use_bc=False,
        no_model_error=True,
    )

    assert "x_ff_active" not in model.named_vars
    assert "x_ocean_active" in model.named_vars
    np.testing.assert_allclose(model.named_vars["x_ff"].eval(), np.ones(4))
    assert model.named_vars["x_ocean"].eval().shape == (4,)
    assert model.named_vars["mu"].eval().shape == (2,)
