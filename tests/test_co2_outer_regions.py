"""Focused prediction, covariance, and label tests for CO2 outer states."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.models.coords import get_coord_registry, registered_model
from openghg_inversions.rhime.co2 import build_co2_rhime_model
from openghg_inversions.rhime.co2.outer_regions import (
    add_inferred_outer_component,
    collapse_outer_sectors,
    composite_baseline,
    prepare_outer_region_treatment,
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
    atmospheric = xr.DataArray([400.0, 401.0], dims="nmeasure")

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

    np.testing.assert_allclose(
        composite_baseline(atmospheric, fixed.fixed_contribution),
        atmospheric.values + h.values @ np.ones(h.sizes["region"]),
    )
    np.testing.assert_allclose(
        composite_baseline(atmospheric, marginalized.fixed_contribution),
        atmospheric.values + h.values @ mean.values,
    )
    np.testing.assert_allclose(
        marginalized.observation_covariance,
        h.values @ covariance.values @ h.values.T,
    )
    np.testing.assert_allclose(inferred.fixed_contribution, 0.0)
    np.testing.assert_allclose(inferred.observation_covariance, 0.0)
    np.testing.assert_allclose(
        composite_baseline(atmospheric, inferred.fixed_contribution)
        + inferred.sensitivity @ inferred.prior_mean,
        atmospheric.values + h.values @ mean.values,
    )
    assert fixed.sensitivity is None
    assert marginalized.sensitivity is None
    assert inferred.sensitivity is not None
    assert fixed.prior_covariance is None
    assert marginalized.prior_covariance is None
    assert inferred.prior_covariance is not None


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
    np.testing.assert_allclose(fixed.fixed_contribution, h.values @ np.ones(4))
    assert inferred.sensitivity is not None
    assert inferred.sensitivity.sizes["outer_state"] == 2


def test_co2_model_composes_sampled_boundary_and_each_outer_mode() -> None:
    inner_h = xr.DataArray(
        [[1.0], [2.0]],
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": ["inner"]},
    )
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

    def build(mode: str) -> pm.Model:
        treatment = prepare_outer_region_treatment(
            outer_h,
            mode=mode,
            prior_mean=outer_mean if mode != "fixed" else None,
            prior_covariance=outer_covariance if mode != "fixed" else None,
        )
        return build_co2_rhime_model(
            inner_h,
            prior_mean=xr.DataArray([1.0], dims="region", coords={"region": ["inner"]}),
            prior_covariance=xr.DataArray(
                [[0.1]],
                dims=("region", "region_cov"),
                coords={"region": ["inner"]},
            ),
            fixed_prior_contribution=xr.DataArray(
                [10.0, 20.0], dims="nmeasure", coords={"nmeasure": [0, 1]}
            ),
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=aggregation_error,
            outer_treatment=treatment,
            boundary_sensitivity=boundary_h,
            no_model_error=True,
        )

    fixed = build("fixed")
    marginalized = build("marginalized")
    inferred = build("inferred")

    assert "bc" in fixed.named_vars
    assert "x_outer" not in fixed.named_vars
    assert "x_outer" not in marginalized.named_vars
    assert "bc" in inferred.named_vars and "x_outer" in inferred.named_vars
    expected_outer_covariance = outer_h.values @ outer_covariance.values @ outer_h.values.T
    np.testing.assert_allclose(
        marginalized["y"].owner.inputs[-1].eval(),
        expected_outer_covariance,
    )

    mu, pollution, boundary, outer, fixed_prior = pm.draw(
        [
            inferred["mu"],
            inferred["mu_pollution"],
            inferred["mu_bc"],
            inferred["mu_outer"],
            inferred["fixed_prior_contribution"],
        ],
        random_seed=42,
    )
    np.testing.assert_allclose(mu, pollution + boundary + outer + fixed_prior)


def test_co2_builder_rejects_outer_states_present_in_both_designs() -> None:
    h, mean, covariance = _outer_inputs()
    treatment = prepare_outer_region_treatment(
        h,
        mode="inferred",
        prior_mean=mean,
        prior_covariance=covariance,
    )
    data = xr.Dataset(
        {
            "mf": ("nmeasure", [400.0, 401.0]),
            "mf_error": ("nmeasure", [0.0, 0.0]),
            "min_error": ("nmeasure", [0.0, 0.0]),
        }
    )
    with pytest.raises(ValueError, match="avoid double counting"):
        build_co2_rhime_model(
            h,
            prior_mean=mean,
            prior_covariance=covariance,
            fixed_prior_contribution=xr.DataArray([0.0, 0.0], dims="nmeasure"),
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, "none"),
            outer_treatment=treatment,
            no_model_error=True,
        )


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
        add_inferred_outer_component(treatment)
    state = pm.draw(model["x_outer"], random_seed=42)

    assert "x_outer_active" in model.named_vars
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
    observations = xr.DataArray(
        [400.0, 401.0], dims="nmeasure", coords={"nmeasure": [0, 1]}
    )
    error = xr.zeros_like(observations)
    aggregation_error = resolve_aggregation_error(
        xr.Dataset({"mf": observations, "mf_error": error, "min_error": error}),
        "none",
    )

    model = build_co2_rhime_model(
        inner_h,
        prior_mean=xr.DataArray([1.0], dims="region", coords={"region": ["inner"]}),
        prior_covariance=xr.DataArray(
            [[0.1]], dims=("region", "region_cov"), coords={"region": ["inner"]}
        ),
        fixed_prior_contribution=xr.zeros_like(observations),
        observations=observations,
        observation_error=error,
        minimum_error=error,
        aggregation_error=aggregation_error,
        outer_treatment=treatment,
        no_model_error=True,
    )
    registry = get_coord_registry(model)

    assert registry is not None
    assert registry.auxiliary_coords["source"].values.tolist() == ["ff"]
    assert registry.auxiliary_coords["source_outer"].values.tolist() == ["ff", "gpp"]
    assert registry.auxiliary_coords["sector_outer"].values.tolist() == ["ff", "gpp"]
    assert registry.auxiliary_coords["domain_outer"].values.tolist() == ["EUROPE", "EUROPE"]
    assert model.named_vars_to_dims["x_outer"] == ("outer_region_outer",)
