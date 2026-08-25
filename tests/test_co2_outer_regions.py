"""Grouped inner/outer contracts for the one-state CO2 recipe."""

from __future__ import annotations

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import build_co2_model


def _grouped_inputs() -> tuple[xr.Dataset, CorrelatedLognormalPrior]:
    state = ["inner-a", "outer-a", "inner-zero", "outer-b"]
    basis_group = ["inner", "outer", "inner", "outer"]
    basis_partition = ["inner-grid", "outer-grid", "inner-grid", "outer-grid"]
    region_in_partition = [1, 1, 2, 2]
    sensitivity = xr.DataArray(
        [[1.0, 3.0, 0.0, 4.0], [2.0, 5.0, 0.0, 6.0]],
        dims=("nmeasure", "state"),
        coords={"nmeasure": [0, 1], "state": state},
        name="H",
    ).assign_coords(
        basis_group=("state", basis_group),
        basis_partition=("state", basis_partition),
        region_in_partition=("state", region_in_partition),
    )
    mean = xr.DataArray(
        [1.2, 0.8, 1.0, 1.4],
        dims="state",
        coords={"state": state},
        name="alpha_prior_mean",
    ).assign_coords(
        basis_group=("state", basis_group),
        basis_partition=("state", basis_partition),
        region_in_partition=("state", region_in_partition),
    )
    covariance_values = np.diag([0.04, 0.03, 0.02, 0.05])
    covariance_values[0, 1] = covariance_values[1, 0] = 0.005
    covariance = xr.DataArray(
        covariance_values,
        dims=("state", "state_cov"),
        coords={"state": state},
        name="alpha_prior_covariance",
    )
    observations = xr.DataArray(
        [400.0, 401.0],
        dims="nmeasure",
        coords={"nmeasure": [0, 1]},
        name="mf",
    )
    error = xr.zeros_like(observations)
    dataset = xr.Dataset(
        {
            "H": sensitivity,
            "fixed_prior_contribution": xr.zeros_like(observations).rename(
                "fixed_prior_contribution"
            ),
            "mf": observations,
            "mf_error": error.rename("mf_error"),
            "min_error": error.rename("min_error"),
        }
    )
    return dataset, CorrelatedLognormalPrior(
        mean,
        covariance,
        covariance_dim="state_cov",
    )


def _build(
    dataset: xr.Dataset,
    retained_prior: CorrelatedLognormalPrior,
    **kwargs,
) -> pm.Model:
    return build_co2_model(
        dataset["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=dataset["fixed_prior_contribution"],
        observations=dataset["mf"],
        observation_error=dataset["mf_error"],
        minimum_error=dataset["min_error"],
        aggregation_error=resolve_aggregation_error(dataset, "none"),
        fixed_model_mismatch=1.0,
        **kwargs,
    )


def test_fixed_outer_parity_uses_one_full_state_and_group_labels() -> None:
    dataset, retained_prior = _grouped_inputs()
    model = _build(
        dataset,
        retained_prior,
        state_activity=StateActivity(fixed_groups=("outer",), fixed_value=1.0),
    )
    state, flux, outer, modelled = pm.draw(
        [
            model["flux_scaling"],
            model["co2_flux_contribution"],
            model["outer_flux_contribution"],
            model["modelled_concentration"],
        ],
        random_seed=42,
    )

    inner = np.array([0, 2])
    outer_indices = np.array([1, 3])
    legacy_split_prediction = (
        dataset["H"].values[:, inner] @ state[inner]
        + dataset["H"].values[:, outer_indices] @ np.ones(2)
    )
    np.testing.assert_allclose(state[outer_indices], 1.0)
    np.testing.assert_allclose(flux, legacy_split_prediction)
    np.testing.assert_allclose(modelled, legacy_split_prediction)
    np.testing.assert_allclose(
        outer,
        dataset["H"].values[:, outer_indices] @ np.ones(2),
    )
    assert "outer_flux_scaling" not in model.named_vars

    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.auxiliary_coords["basis_group"].values.tolist() == [
        "inner",
        "outer",
        "inner",
        "outer",
    ]
    assert registry.auxiliary_coords["basis_partition"].values.tolist() == [
        "inner-grid",
        "outer-grid",
        "inner-grid",
        "outer-grid",
    ]
    assert registry.auxiliary_coords["region_in_partition"].values.tolist() == [
        1,
        1,
        2,
        2,
    ]


def test_inferred_group_prior_keeps_order_moments_and_cross_covariance() -> None:
    dataset, retained_prior = _grouped_inputs()
    mean = retained_prior.mean
    covariance = retained_prior.arithmetic_covariance
    inner = np.array([0, 2])
    outer = np.array([1, 3])

    prior_forward = dataset["H"].values @ mean.values
    split_prior_forward = (
        dataset["H"].values[:, inner] @ mean.values[inner]
        + dataset["H"].values[:, outer] @ mean.values[outer]
    )
    np.testing.assert_allclose(prior_forward, split_prior_forward)
    np.testing.assert_allclose(mean.sel(state=["inner-a", "inner-zero"]), [1.2, 1.0])
    np.testing.assert_allclose(mean.sel(state=["outer-a", "outer-b"]), [0.8, 1.4])
    assert covariance.sel(state="inner-a").isel(state_cov=1).item() == 0.005

    prior_forward_model = _build(
        dataset,
        retained_prior,
        state_activity=StateActivity(active=False, fixed_value=mean),
    )
    np.testing.assert_allclose(
        pm.draw(prior_forward_model["modelled_concentration"]),
        split_prior_forward,
    )

    model = _build(dataset, retained_prior)
    assert "outer_flux_contribution" in model.named_vars
    assert "outer_flux_scaling" not in model.named_vars
    assert "outer_observation_factor" not in model.named_vars
    assert "low_rank_factor" not in model.named_vars
    assert "aggregation_error_covariance" not in model.named_vars


def test_grouped_state_order_is_label_invariant() -> None:
    dataset, retained_prior = _grouped_inputs()
    fixed_value = xr.DataArray(
        [1.1, 0.7, 1.3, 1.5],
        dims="state",
        coords={"state": dataset["state"]},
    )
    original = _build(
        dataset,
        retained_prior,
        state_activity=StateActivity(active=False, fixed_value=fixed_value),
    )

    permutation = [3, 0, 2, 1]
    permuted_dataset = dataset.isel(state=permutation)
    permuted_mean = retained_prior.mean.isel(state=permutation)
    covariance = retained_prior.arithmetic_covariance.values[np.ix_(permutation, permutation)]
    permuted_prior = CorrelatedLognormalPrior(permuted_mean, covariance)
    permuted = _build(
        permuted_dataset,
        permuted_prior,
        state_activity=StateActivity(active=False, fixed_value=fixed_value),
    )

    np.testing.assert_allclose(
        pm.draw(original["modelled_concentration"]),
        pm.draw(permuted["modelled_concentration"]),
    )
    np.testing.assert_allclose(
        pm.draw(original["outer_flux_contribution"]),
        pm.draw(permuted["outer_flux_contribution"]),
    )


def test_fixed_outer_and_zero_sensitivity_activity_compose() -> None:
    dataset, retained_prior = _grouped_inputs()
    fixed_value = xr.DataArray(
        [9.0, 1.0, 7.0, 1.0],
        dims="state",
        coords={"state": dataset["state"]},
    )
    model = _build(
        dataset,
        retained_prior,
        state_activity=StateActivity(
            fixed_groups=("outer",),
            fixed_value=fixed_value,
        ),
    )

    np.testing.assert_array_equal(
        model["flux_scaling_is_active"].eval(),
        [True, False, False, False],
    )
    state = pm.draw(model["flux_scaling"], random_seed=42)
    np.testing.assert_allclose(state[[1, 2, 3]], [1.0, 7.0, 1.0])


def test_reporting_views_close_with_boundary_and_offset() -> None:
    dataset, retained_prior = _grouped_inputs()
    boundary_sensitivity = xr.DataArray(
        [[0.5], [0.25]],
        dims=("nmeasure", "bc_state"),
        coords={"nmeasure": dataset["nmeasure"], "bc_state": ["monthly"]},
    )
    site_indicator = xr.DataArray(
        [0, 0],
        dims="nmeasure",
        coords={"nmeasure": dataset["nmeasure"]},
    )
    model = _build(
        dataset,
        retained_prior,
        boundary_sensitivity=boundary_sensitivity,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        site_indicator=site_indicator,
        offset_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
    )
    total, flux, outer, boundary, offset = pm.draw(
        [
            model["modelled_concentration"],
            model["co2_flux_contribution"],
            model["outer_flux_contribution"],
            model["mu_bc"],
            model["offset"],
        ],
        random_seed=42,
    )

    inner = flux - outer
    composite_baseline = boundary + offset + outer
    np.testing.assert_allclose(total, inner + outer + boundary + offset)
    np.testing.assert_allclose(composite_baseline, boundary + offset + outer)
