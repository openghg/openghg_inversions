"""Focused prediction, covariance, and label tests for CO2 outer states."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.observation_error import AggregationError, resolve_aggregation_error
from openghg_inversions.models.coords import get_coord_registry, registered_model
from openghg_inversions.rhime.co2 import (
    build_co2_model,
    collapse_outer_sectors,
    prepare_outer_region_treatment,
)
from openghg_inversions.rhime.co2.outer_regions import (
    add_outer_state_component,
    add_outer_observation_covariance,
)


def _outer_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    state = pd.MultiIndex.from_tuples(
        [("ff", 8), ("gpp", 8), ("ff", 9), ("gpp", 9)],
        names=("source", "region_in_source"),
    )
    coords = xr.Coordinates.from_pandas_multiindex(state, "region")
    h = xr.DataArray(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],
        dims=("nmeasure", "region"),
        coords=coords,
        name="outer_sensitivity",
    ).assign_coords(
        nmeasure=[0, 1],
        sector=("region", ["ff", "gpp", "ff", "gpp"]),
        domain=("region", ["EUROPE"] * 4),
        basis_group=("region", ["outer"] * 4),
    )
    mean = xr.DataArray(
        [1.0, 1.5, 0.8, 1.2],
        dims="region",
        coords={"region": h["region"]},
    )
    covariance = xr.DataArray(
        np.diag([0.1, 0.2, 0.3, 0.4]),
        dims=("region", "region_cov"),
        coords={"region": h["region"]},
    )
    return h, mean, covariance


def test_outer_modes_partition_prediction_and_covariance_without_double_counting() -> None:
    h, mean, covariance = _outer_inputs()
    fixed = prepare_outer_region_treatment(h, mode="fixed")
    marginalized = prepare_outer_region_treatment(
        h,
        mode="marginalized",
        prior_mean=mean,
        prior_covariance=covariance,
    )
    inferred = prepare_outer_region_treatment(
        h,
        mode="inferred",
        prior_mean=mean,
        prior_covariance=covariance,
    )

    assert fixed.mean_contribution is None
    assert fixed.prepared_sensitivity.sensitivity is not None
    assert fixed.resolved_activity is not None
    assert not fixed.resolved_activity.active.any().item()
    np.testing.assert_allclose(fixed.resolved_activity.fixed_value, 1.0)
    np.testing.assert_allclose(marginalized.mean_contribution, h.values @ mean.values)
    assert marginalized.observation_factor is not None
    np.testing.assert_allclose(
        marginalized.observation_factor.values @ marginalized.observation_factor.values.T,
        h.values @ covariance.values @ h.values.T,
    )
    assert inferred.mean_contribution is None
    np.testing.assert_allclose(
        inferred.prepared_sensitivity.sensitivity @ inferred.prior_mean,
        h.values @ mean.values,
    )
    assert fixed.observation_factor is None
    assert inferred.observation_factor is None
    assert inferred.prepared_sensitivity.state_dim == "region"
    assert fixed.prior_covariance is None
    assert marginalized.prior_covariance is None
    assert inferred.prior_covariance is not None


def test_marginalized_outer_factor_preserves_each_aggregation_error_representation() -> None:
    h, mean, covariance = _outer_inputs()
    treatment = prepare_outer_region_treatment(
        h,
        mode="marginalized",
        prior_mean=mean,
        prior_covariance=covariance,
    )
    observation = h["nmeasure"]
    diagonal = xr.DataArray([0.4, 0.5], dims="nmeasure", coords={"nmeasure": observation})
    base_factor = xr.DataArray(
        [[0.2], [0.3]],
        dims=("nmeasure", "aggregation_error_rank"),
        coords={"nmeasure": observation, "aggregation_error_rank": [0]},
    )
    dense = xr.DataArray(
        [[0.6, 0.1], [0.1, 0.7]],
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": observation, "nmeasure_cov": observation.values},
    )
    cases = {
        "none": AggregationError(mode="none", marginal_variance=np.zeros(2)),
        "diagonal": AggregationError(
            mode="diagonal",
            marginal_variance=diagonal.values,
            diagonal_variance=diagonal,
        ),
        "low_rank": AggregationError(
            mode="low_rank",
            marginal_variance=np.sum(base_factor.values**2, axis=1) + diagonal.values,
            factor=base_factor,
            diagonal_variance=diagonal,
        ),
        "dense": AggregationError(
            mode="dense",
            marginal_variance=np.diag(dense.values),
            covariance=dense,
        ),
    }
    expected_outer = h.values @ covariance.values @ h.values.T

    for mode, base in cases.items():
        combined = add_outer_observation_covariance(base, treatment)
        if combined.mode == "dense":
            assert combined.covariance is not None
            actual = combined.covariance.values
        else:
            assert combined.mode == "low_rank"
            assert combined.factor is not None
            assert combined.diagonal_variance is not None
            actual = combined.factor.values @ combined.factor.values.T + np.diag(
                combined.diagonal_variance.values
            )
        if mode == "none":
            expected_base = np.zeros((2, 2))
        elif mode == "diagonal":
            expected_base = np.diag(diagonal.values)
        elif mode == "low_rank":
            expected_base = base_factor.values @ base_factor.values.T + np.diag(diagonal.values)
        else:
            expected_base = dense.values
        np.testing.assert_allclose(actual, expected_base + expected_outer)

    mismatched = replace(
        cases["diagonal"],
        diagonal_variance=diagonal.assign_coords(nmeasure=[1, 0]),
    )
    with pytest.raises(ValueError, match="cannot align objects with join='exact'"):
        add_outer_observation_covariance(mismatched, treatment)


def test_marginalized_outer_stays_linear_in_observations() -> None:
    nmeasure = 50_000
    sensitivity = xr.DataArray(
        np.ones((nmeasure, 2)),
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": np.arange(nmeasure), "outer_region": ["west", "east"]},
    )
    mean = xr.DataArray([1.0, 1.0], dims="outer_region", coords={"outer_region": ["west", "east"]})
    covariance = xr.DataArray(
        [[0.2, 0.2], [0.2, 0.2]],
        dims=("outer_region", "outer_region_cov"),
        coords={"outer_region": ["west", "east"]},
    )
    treatment = prepare_outer_region_treatment(
        sensitivity,
        mode="marginalized",
        prior_mean=mean,
        prior_covariance=covariance,
    )
    combined = add_outer_observation_covariance(
        AggregationError(mode="none", marginal_variance=np.zeros(nmeasure)),
        treatment,
    )

    assert treatment.observation_factor is not None
    assert not hasattr(treatment, "observation_covariance")
    assert treatment.observation_factor.shape == (nmeasure, 2)
    assert combined.mode == "low_rank"
    assert combined.factor is not None and combined.factor.shape == (nmeasure, 2)


def test_outer_treatment_preserves_source_sector_domain_activity_and_treatment_labels() -> None:
    h, mean, covariance = _outer_inputs()
    treatment = prepare_outer_region_treatment(
        h,
        mode="inferred",
        prior_mean=mean,
        prior_covariance=covariance,
    )

    assert {
        "source",
        "sector",
        "domain",
        "basis_group",
        "activity",
        "treatment",
    } <= set(treatment.state_metadata.variables)
    assert treatment.state_metadata["source"].values.tolist() == ["ff", "gpp", "ff", "gpp"]
    assert treatment.state_metadata["activity"].values.tolist() == [True] * 4
    assert treatment.state_metadata["treatment"].values.tolist() == ["inferred"] * 4


def test_collapse_outer_sectors_is_orthogonal_to_treatment_and_keeps_members() -> None:
    h, _, _ = _outer_inputs()
    groups = xr.DataArray(
        ["outer-8", "outer-8", "outer-9", "outer-9"],
        dims="region",
        coords={"region": h["region"]},
    )
    collapsed = collapse_outer_sectors(h, group_labels=groups)
    collapsed_mean = xr.DataArray(
        [1.0, 1.0],
        dims="outer_state",
        coords={"outer_state": collapsed.sensitivity["outer_state"]},
    )
    collapsed_covariance = xr.DataArray(
        np.diag([0.25, 0.5]),
        dims=("outer_state", "outer_state_cov"),
        coords={
            "outer_state": collapsed.sensitivity["outer_state"],
        },
    )

    np.testing.assert_allclose(
        collapsed.sensitivity,
        [[3.0, 7.0], [1.5, 3.5]],
    )
    assert collapsed.sensitivity["source"].values.tolist() == ["outer_total"] * 2
    assert collapsed.sensitivity["sector"].values.tolist() == ["outer_total"] * 2
    assert collapsed.members["source"].values.tolist() == ["ff", "gpp", "ff", "gpp"]
    assert collapsed.members["collapsed_state"].values.tolist() == [
        "outer-8",
        "outer-8",
        "outer-9",
        "outer-9",
    ]

    fixed = prepare_outer_region_treatment(collapsed, mode="fixed")
    inferred = prepare_outer_region_treatment(
        collapsed,
        mode="inferred",
        prior_mean=collapsed_mean,
        prior_covariance=collapsed_covariance,
    )
    with registered_model() as model:
        add_outer_state_component(fixed)
    np.testing.assert_allclose(pm.draw(model["outer_flux_contribution"]), h.values @ np.ones(4))
    assert inferred.prepared_sensitivity.removed.sizes["outer_state"] == 2


def test_collapsed_member_metadata_uses_resolved_group_activity() -> None:
    sensitivity = xr.DataArray(
        [[1.0, 2.0, 0.0], [0.5, 1.0, 0.0]],
        dims=("nmeasure", "outer_member"),
        coords={"nmeasure": [0, 1], "outer_member": ["a1", "a2", "b"]},
    )
    groups = xr.DataArray(
        ["a", "a", "b"],
        dims="outer_member",
        coords={"outer_member": sensitivity["outer_member"]},
    )
    collapsed = collapse_outer_sectors(sensitivity, group_labels=groups)
    mean = xr.DataArray(
        [1.0, 1.0],
        dims="outer_state",
        coords={"outer_state": collapsed.sensitivity["outer_state"]},
    )
    covariance = xr.DataArray(
        np.eye(2),
        dims=("outer_state", "outer_state_cov"),
        coords={"outer_state": collapsed.sensitivity["outer_state"]},
    )
    treatment = prepare_outer_region_treatment(
        collapsed,
        mode="inferred",
        prior_mean=mean,
        prior_covariance=covariance,
    )

    assert treatment.state_metadata["activity"].values.tolist() == [True, True, False]
    with registered_model() as model:
        add_outer_state_component(treatment)
    assert pm.draw(model["outer_flux_scaling"], random_seed=42)[1] == 1.0


def test_outer_treatment_carries_custom_output_dimension_into_components() -> None:
    h, mean, covariance = _outer_inputs()
    h = h.rename(nmeasure="observation").assign_coords(observation=["a", "b"])
    fixed = prepare_outer_region_treatment(h, mode="fixed", observation_dim="observation")
    marginalized = prepare_outer_region_treatment(
        h,
        mode="marginalized",
        prior_mean=mean,
        prior_covariance=covariance,
        observation_dim="observation",
    )

    with registered_model() as model:
        add_outer_state_component(fixed)
    combined = add_outer_observation_covariance(
        AggregationError(mode="none", marginal_variance=np.zeros(2)),
        marginalized,
    )

    assert fixed.prepared_sensitivity.output_dim == "observation"
    assert model.named_vars_to_dims["outer_flux_contribution"] == ("observation",)
    np.testing.assert_allclose(
        pm.draw(model["outer_flux_contribution"]),
        h.values @ np.ones(h.sizes["region"]),
    )
    assert combined.factor is not None
    assert combined.factor.dims == ("observation", "outer_covariance_rank")


def test_co2_model_composes_sampled_boundary_and_each_outer_mode() -> None:
    inner_h = xr.DataArray(
        [[1.0], [2.0]],
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": ["west"]},
    ).assign_coords(basis_group=("region", ["inner"]))
    outer_h = xr.DataArray(
        [[3.0, 4.0], [5.0, 6.0]],
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": [0, 1], "outer_region": ["west", "east"]},
    )
    outer_mean = xr.DataArray(
        [1.0, 1.0],
        dims="outer_region",
        coords={"outer_region": outer_h["outer_region"]},
    ).assign_coords(
        source=("outer_region", ["ff", "gpp"]),
        sector=("outer_region", ["ff", "gpp"]),
        domain=("outer_region", ["EUROPE", "EUROPE"]),
    )
    outer_covariance = xr.DataArray(
        np.diag([0.04, 0.09]),
        dims=("outer_region", "outer_region_cov"),
        coords={"outer_region": outer_h["outer_region"]},
    )
    boundary_h = xr.DataArray(
        [[0.5], [0.25]],
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": [0, 1], "bc_region": ["monthly"]},
    )
    data = xr.Dataset(
        {
            "mf": ("nmeasure", [400.0, 401.0]),
            "mf_error": ("nmeasure", [0.0, 0.0]),
            "min_error": ("nmeasure", [0.0, 0.0]),
        },
        coords={"nmeasure": [0, 1]},
    )
    aggregation_error = resolve_aggregation_error(data, "none")

    def build(
        mode: str,
        outer_sensitivity: xr.DataArray = outer_h,
        observations: xr.DataArray = data["mf"],
    ) -> pm.Model:
        treatment = prepare_outer_region_treatment(
            outer_sensitivity,
            mode=mode,
            prior_mean=outer_mean if mode != "fixed" else None,
            prior_covariance=outer_covariance if mode != "fixed" else None,
        )
        return build_co2_model(
            inner_h,
            prior_mean=xr.DataArray([1.0], dims="region", coords={"region": ["west"]}),
            prior_covariance=xr.DataArray(
                [[0.1]],
                dims=("region", "region_cov"),
                coords={"region": ["west"]},
            ),
            fixed_prior_contribution=xr.DataArray(
                [10.0, 20.0],
                dims="nmeasure",
                coords={"nmeasure": [0, 1]},
                name="fixed_prior_contribution",
            ),
            observations=observations,
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=aggregation_error,
            outer_treatment=treatment,
            boundary_sensitivity=boundary_h,
        )

    fixed = build("fixed")
    marginalized = build("marginalized")
    inferred = build("inferred")
    for mode in ("fixed", "marginalized", "inferred"):
        with pytest.raises(ValueError, match="Conflicting coord registration|align.*join='exact'"):
            build(mode, outer_h.sel(nmeasure=[1, 0]))

    assert "bc" in fixed.named_vars
    assert "outer_flux_scaling" in fixed.named_vars
    assert "outer_flux_scaling_active" not in fixed.named_vars
    assert "outer_flux_scaling" not in marginalized.named_vars
    assert "bc" in inferred.named_vars and "outer_flux_scaling" in inferred.named_vars
    assert "mu_baseline" not in fixed.named_vars
    assert "mu_baseline" not in marginalized.named_vars
    assert "mu_baseline" not in inferred.named_vars
    np.testing.assert_allclose(pm.draw(fixed["outer_flux_scaling"]), 1.0)
    np.testing.assert_allclose(
        pm.draw(fixed["outer_flux_contribution"]),
        outer_h.values @ np.ones(2),
    )
    np.testing.assert_allclose(
        pm.draw(marginalized["outer_flux_contribution"]),
        outer_h.values @ outer_mean.values,
    )
    expected_outer_covariance = outer_h.values @ outer_covariance.values @ outer_h.values.T
    np.testing.assert_allclose(
        marginalized["low_rank_factor"].eval() @ marginalized["low_rank_factor"].eval().T,
        expected_outer_covariance,
    )

    mu, pollution, boundary, outer, fixed_prior = pm.draw(
        [
            inferred["modelled_concentration"],
            inferred["co2_flux_contribution"],
            inferred["mu_bc"],
            inferred["outer_flux_contribution"],
            inferred["fixed_prior_contribution"],
        ],
        random_seed=42,
    )
    np.testing.assert_allclose(mu, pollution + boundary + outer + fixed_prior)


def test_inferred_zero_sensitivity_outer_state_is_fixed_at_one() -> None:
    h = xr.DataArray(
        [[2.0, 0.0], [3.0, 0.0]],
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": [0, 1], "outer_region": ["active", "zero-h"]},
    )
    mean = xr.DataArray(
        [1.0, 1.0],
        dims="outer_region",
        coords={"outer_region": h["outer_region"]},
    )
    covariance = xr.DataArray(
        np.diag([0.1, 0.2]),
        dims=("outer_region", "outer_region_cov"),
        coords={"outer_region": h["outer_region"]},
    )
    treatment = prepare_outer_region_treatment(
        h,
        mode="inferred",
        prior_mean=mean,
        prior_covariance=covariance,
    )

    with registered_model() as model:
        add_outer_state_component(treatment)
    state = pm.draw(model["outer_flux_scaling"], random_seed=42)

    assert "outer_flux_scaling_active" in model.named_vars
    assert treatment.resolved_activity is not None
    assert treatment.state_metadata["activity"].values.tolist() == [True, False]
    xr.testing.assert_equal(
        treatment.state_metadata["activity"],
        treatment.resolved_activity.active,
    )
    assert state[1] == 1.0


def test_inner_and_outer_state_auxiliary_coords_are_namespaced() -> None:
    inner_h = xr.DataArray(
        [[1.0], [2.0]],
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": ["inner"]},
    ).assign_coords(
        source=("region", ["ff"]),
        sector=("region", ["ff"]),
        domain=("region", ["PARIS"]),
        basis_group=("region", ["inner"]),
    )
    outer_h = xr.DataArray(
        [[3.0, 4.0], [5.0, 6.0]],
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": [0, 1], "outer_region": ["west", "east"]},
    ).assign_coords(
        source=("outer_region", ["ff", "gpp"]),
        sector=("outer_region", ["ff", "gpp"]),
        domain=("outer_region", ["EUROPE", "EUROPE"]),
    )
    outer_mean = xr.DataArray(
        [1.0, 1.0],
        dims="outer_region",
        coords={"outer_region": outer_h["outer_region"]},
    ).assign_coords(
        source=("outer_region", ["ff", "gpp"]),
        sector=("outer_region", ["ff", "gpp"]),
        domain=("outer_region", ["EUROPE", "EUROPE"]),
    )
    outer_covariance = xr.DataArray(
        np.diag([0.1, 0.2]),
        dims=("outer_region", "outer_region_cov"),
        coords={"outer_region": outer_h["outer_region"]},
    )
    treatment = prepare_outer_region_treatment(
        outer_h,
        mode="inferred",
        prior_mean=outer_mean,
        prior_covariance=outer_covariance,
    )
    observations = xr.DataArray([400.0, 401.0], dims="nmeasure", coords={"nmeasure": [0, 1]})
    error = xr.zeros_like(observations)
    aggregation_error = resolve_aggregation_error(
        xr.Dataset({"mf": observations, "mf_error": error, "min_error": error}),
        "none",
    )

    model = build_co2_model(
        inner_h,
        prior_mean=xr.DataArray([1.0], dims="region", coords={"region": ["inner"]}),
        prior_covariance=xr.DataArray([[0.1]], dims=("region", "region_cov"), coords={"region": ["inner"]}),
        fixed_prior_contribution=xr.zeros_like(observations).rename("fixed_prior_contribution"),
        observations=observations,
        observation_error=error,
        minimum_error=error,
        aggregation_error=aggregation_error,
        outer_treatment=treatment,
    )
    registry = get_coord_registry(model)

    assert registry is not None
    assert registry.auxiliary_coords["source"].values.tolist() == ["ff"]
    assert registry.auxiliary_coords["source_outer"].values.tolist() == ["ff", "gpp"]
    assert registry.auxiliary_coords["sector_outer"].values.tolist() == ["ff", "gpp"]
    assert registry.auxiliary_coords["domain_outer"].values.tolist() == ["EUROPE", "EUROPE"]
    assert model.named_vars_to_dims["outer_flux_scaling"] == ("outer_region_outer",)
