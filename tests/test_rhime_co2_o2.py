"""Focused contracts for CO2/O2 preparation, modelling, and sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    StateActivity,
    get_coord_registry,
    prepare_linear_sensitivity,
    restore_inferencedata_coords,
)
from openghg_inversions.rhime.co2 import (
    build_co2_o2_model,
    evaluate_co2_o2_prior_forward_mean,
    prepare_co2_o2_inputs,
    run_rhime_co2_o2_from_prepared_inputs,
)
from openghg_inversions.rhime.co2.co2_o2_model import (
    _gather_co2_o2_operator,
)
from openghg_inversions.rhime.co2.co2_o2_runner import _co2_o2_metadata
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import load_inferencedata, save_inferencedata


def _inputs(*, gathered_state: bool = False) -> dict[str, object]:
    labels: list[str] | pd.MultiIndex
    if gathered_state:
        labels = pd.MultiIndex.from_tuples(
            [
                ("GPP", "shared", 1),
                ("TER", "shared", 1),
                ("FF", "shared", 1),
                ("ocean", "co2", 1),
                ("ocean", "o2", 1),
            ],
            names=("source", "tracer_scope", "region_in_source"),
        )
        state_coords = xr.Coordinates.from_pandas_multiindex(labels, "state")
    else:
        labels = ["gpp:1", "ter:1", "ff:1", "co2-ocean:1", "o2-ocean:1"]
        state_coords = {
            "state": labels,
            "source": ("state", ["GPP", "TER", "FF", "ocean", "ocean"]),
            "tracer_scope": ("state", ["shared", "shared", "shared", "co2", "o2"]),
        }
    ratio_coords = (
        xr.Coordinates.from_pandas_multiindex(labels[:3], "state")
        if isinstance(labels, pd.MultiIndex)
        else {
            "state": labels[:3],
            "source": ("state", ["GPP", "TER", "FF"]),
        }
    )
    mean = xr.DataArray(
        np.ones(5),
        dims=("state",),
        coords=state_coords,
        name="retained_prior_mean",
        attrs={"units": "dimensionless flux scale"},
    )
    covariance = np.diag([0.01, 0.02, 0.03, 0.04, 0.05])
    co2 = xr.DataArray([2.0, 3.0], dims="co2_measure", coords={"co2_measure": ["c1", "c2"]})
    o2 = xr.DataArray(
        [-4.0, -5.0, -6.0],
        dims="o2_measure",
        coords={"o2_measure": ["o1", "o2", "o3"]},
    )
    return {
        "co2_observations": co2,
        "o2_observations": o2,
        "co2_prior_forward_mean": co2 - 0.25,
        "o2_prior_forward_mean": o2 + 0.5,
        "co2_operator": xr.DataArray(
            [[1, 2, 3, 4, 0], [0.5, 1, 1.5, 2, 0]],
            dims=("co2_measure", "state"),
            coords={"co2_measure": co2["co2_measure"], "state": mean["state"]},
        ),
        "o2_operator": xr.DataArray(
            [[-1, -2, -3, 0, 5], [-0.5, -1, -1.5, 0, 2.5], [-0.2, -0.4, -0.6, 0, 1]],
            dims=("o2_measure", "state"),
            coords={"o2_measure": o2["o2_measure"], "state": mean["state"]},
        ),
        "o2_co2_flux_ratio": xr.DataArray(
            [-1.1, -1.0, -1.4],
            dims="state",
            coords=ratio_coords,
            attrs={
                "direction": "O2 flux per CO2 flux",
                "sign_convention": "signed; positive CO2 flux has negative O2 loading",
                "provenance": "Verification Games source-resolved O2:CO2 ratios",
            },
        ),
        "o2_co2_flux_ratio_unavailable_reason": None,
        "co2_aggregation_covariance": xr.DataArray(
            [[1, 0.2], [0.2, 1.5]],
            dims=("co2_measure", "co2_measure_cov"),
            coords={"co2_measure": co2["co2_measure"], "co2_measure_cov": ["c1", "c2"]},
        ),
        "co2_o2_aggregation_covariance": xr.DataArray(
            [[-0.1, -0.05, -0.02], [-0.08, -0.04, -0.01]],
            dims=("co2_measure", "o2_measure"),
            coords={"co2_measure": co2["co2_measure"], "o2_measure": o2["o2_measure"]},
        ),
        "o2_aggregation_covariance": xr.DataArray(
            np.diag([2, 2.5, 3]),
            dims=("o2_measure", "o2_measure_cov"),
            coords={"o2_measure": o2["o2_measure"], "o2_measure_cov": ["o1", "o2", "o3"]},
        ),
        "retained_prior": CorrelatedLognormalPrior(mean, covariance),
        "co2_units": "ppm",
        "o2_units": "per meg",
        "provenance": {"verification_games_issue": "OPE-77", "fixture": "co2_o2"},
    }


def _independent_error(prepared, value: float = 0.1) -> xr.DataArray:
    return xr.DataArray(
        np.full(prepared.observations.size, value),
        dims=("observation",),
        coords={
            "observation": prepared.observations["observation"],
            "observation_units": prepared.observations["observation_units"],
        },
        name="fixed_independent_error_sd",
    )


def _build(prepared, *, state_activity: StateActivity | None = None):
    return build_co2_o2_model(
        observations=prepared.observations,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_operator=prepared.co2_operator,
        o2_operator=prepared.o2_operator,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=_independent_error(prepared),
        state_activity=state_activity,
    )


def test_stacks_unequal_axes_cross_covariance_and_units() -> None:
    inputs = _inputs()
    prepared = prepare_co2_o2_inputs(**inputs)

    assert prepared.observations["species"].values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert prepared.observations["observation_units"].values.tolist() == [
        "ppm",
        "ppm",
        "per meg",
        "per meg",
        "per meg",
    ]
    np.testing.assert_allclose(prepared.o2_co2_flux_ratio, [-1.1, -1.0, -1.4])
    assert prepared.o2_operator.attrs["oxidation_ratio_direction"] == "O2 flux per CO2 flux"
    assert prepared.aggregation_error.mode == "dense"
    covariance = prepared.aggregation_error.covariance
    assert covariance is not None
    np.testing.assert_allclose(covariance.values[:2, 2:], inputs["co2_o2_aggregation_covariance"])
    np.testing.assert_allclose(covariance.values[2:, :2], inputs["co2_o2_aggregation_covariance"].T)
    for operator in (prepared.co2_operator, prepared.o2_operator):
        assert operator["tracer_scope"].values.tolist() == [
            "shared",
            "shared",
            "shared",
            "co2",
            "o2",
        ]


def test_rejects_a_shared_ocean_state() -> None:
    inputs = _inputs()
    prior = inputs["retained_prior"]
    mean = prior.mean.assign_coords(tracer_scope=("state", ["shared", "shared", "shared", "shared", "o2"]))
    inputs["retained_prior"] = CorrelatedLognormalPrior(mean, prior.arithmetic_covariance)
    with pytest.raises(ValueError, match="tracer-specific"):
        prepare_co2_o2_inputs(**inputs)


def test_metadata_tracks_unavailable_native_ratio_values() -> None:
    inputs = _inputs()
    inputs["o2_co2_flux_ratio"] = None
    inputs["o2_co2_flux_ratio_unavailable_reason"] = (
        "Native paired O2 flux embeds spatial ratios before convolution."
    )

    prepared = prepare_co2_o2_inputs(**inputs)
    ratio = _co2_o2_metadata(prepared, observations=prepared.observations)["o2_operator_ratio"]

    assert ratio["status"] == "unavailable"
    assert ratio["unavailable_reason"].startswith("Native paired O2 flux")
    assert "value" not in ratio


@pytest.mark.parametrize("gathered_state", [False, True])
def test_shared_state_sensitivity_removes_only_joint_zero_columns(
    gathered_state: bool,
) -> None:
    inputs = _inputs(gathered_state=gathered_state)
    co2_operator = inputs["co2_operator"]
    o2_operator = inputs["o2_operator"]
    assert isinstance(co2_operator, xr.DataArray)
    assert isinstance(o2_operator, xr.DataArray)
    co2_operator = co2_operator.copy()
    o2_operator = o2_operator.copy()
    co2_operator[{"state": 2}] = 0.0
    o2_operator[{"state": 2}] = 0.0
    inputs["co2_operator"] = co2_operator
    inputs["o2_operator"] = o2_operator
    prepared = prepare_co2_o2_inputs(**inputs)

    joint_operator = _gather_co2_o2_operator(
        prepared.co2_operator,
        prepared.o2_operator,
    )
    joint = prepare_linear_sensitivity(joint_operator, output_dim="observation")

    assert joint.removed.isel(state=2).item() is True
    assert joint.removed.isel(state=3).item() is False
    assert joint.removed.isel(state=4).item() is False
    assert joint.sensitivity.indexes["observation"].equals(
        prepared.observations.indexes["observation"]
    )

    prior_forward = evaluate_co2_o2_prior_forward_mean(
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_operator=prepared.co2_operator,
        o2_operator=prepared.o2_operator,
        retained_prior=prepared.retained_prior,
    )
    expected_contribution = np.concatenate(
        (
            prepared.co2_operator.values @ prepared.retained_prior.mean.values,
            prepared.o2_operator.values @ prepared.retained_prior.mean.values,
        )
    )
    np.testing.assert_allclose(
        prior_forward,
        prepared.fixed_prior_contribution.values + expected_contribution,
    )


def test_model_uses_registered_explicit_arrays_and_joint_covariance() -> None:
    prepared = prepare_co2_o2_inputs(**_inputs())
    model = _build(prepared)

    assert get_coord_registry(model) is not None
    assert "fixed_prior_contribution" in model.named_vars
    assert "prior_flux_scaling" not in model.named_vars
    assert "co2_o2_operator" in model.named_vars
    assert "co2_o2_flux_contribution" in model.named_vars
    assert "co2_flux_contribution" not in model.named_vars
    assert "o2_flux_contribution" not in model.named_vars
    scaling, modelled = pm.draw(
        [model["flux_scaling"], model["modelled_concentration"]],
        draws=2,
        random_seed=4,
    )
    expected = prepared.fixed_prior_contribution.values + np.concatenate(
        (
            np.einsum("os,ds->do", prepared.co2_operator.values, scaling),
            np.einsum("os,ds->do", prepared.o2_operator.values, scaling),
        ),
        axis=1,
    )
    np.testing.assert_allclose(modelled, expected, rtol=1e-6, atol=1e-6)
    covariance = prepared.aggregation_error.covariance
    assert covariance is not None
    np.testing.assert_allclose(model["aggregation_error_covariance"].eval(), covariance)


def test_fixed_state_changes_prior_closure_and_is_not_sampled() -> None:
    prepared = prepare_co2_o2_inputs(**_inputs())
    state_dim = prepared.retained_prior.state_dim
    activity = StateActivity(
        active=xr.DataArray(
            [True, True, False, True, True],
            dims=state_dim,
            coords={state_dim: prepared.retained_prior.mean[state_dim]},
        ),
        fixed_value=xr.DataArray(
            [1.0, 1.0, 0.75, 1.0, 1.0],
            dims=state_dim,
            coords={state_dim: prepared.retained_prior.mean[state_dim]},
        ),
    )
    prior_state = prepared.retained_prior.mean.copy()
    prior_state[{state_dim: 2}] = 0.75
    prior_contribution = xr.DataArray(
        np.concatenate(
            (
                prepared.co2_operator.values @ prior_state.values,
                prepared.o2_operator.values @ prior_state.values,
            )
        ),
        dims="observation",
        coords={"observation": prepared.fixed_prior_contribution["observation"]},
    )
    expected_prior = prepared.fixed_prior_contribution + prior_contribution
    xr.testing.assert_allclose(
        evaluate_co2_o2_prior_forward_mean(
            fixed_prior_contribution=prepared.fixed_prior_contribution,
            co2_operator=prepared.co2_operator,
            o2_operator=prepared.o2_operator,
            retained_prior=prepared.retained_prior,
            state_activity=activity,
        ),
        expected_prior.rename("prior_forward_concentration"),
    )

    model = _build(prepared, state_activity=activity)
    scaling, modelled = pm.draw(
        [model["flux_scaling"], model["modelled_concentration"]],
        draws=2,
        random_seed=8,
    )
    np.testing.assert_allclose(scaling[:, 2], 0.75)
    expected_modelled = prepared.fixed_prior_contribution.values + np.concatenate(
        (
            np.einsum("os,ds->do", prepared.co2_operator.values, scaling),
            np.einsum("os,ds->do", prepared.o2_operator.values, scaling),
        ),
        axis=1,
    )
    np.testing.assert_allclose(modelled, expected_modelled, rtol=1e-6, atol=1e-6)


def test_partial_gathered_state_restores_full_multiindex() -> None:
    prepared = prepare_co2_o2_inputs(**_inputs(gathered_state=True))
    state_dim = prepared.retained_prior.state_dim
    activity = StateActivity(
        active=xr.DataArray(
            [True, True, False, True, True],
            dims=state_dim,
            coords={state_dim: prepared.retained_prior.mean[state_dim]},
        ),
        fixed_value=0.75,
    )
    model = _build(prepared, state_activity=activity)
    with model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=42)
    registry = get_coord_registry(model)
    assert registry is not None
    restored = restore_inferencedata_coords(trace, registry)

    assert restored.prior.indexes[state_dim].equals(prepared.retained_prior.mean.indexes[state_dim])
    assert restored.prior["source"].values.tolist() == ["GPP", "TER", "FF", "ocean", "ocean"]
    np.testing.assert_allclose(restored.prior["flux_scaling"].isel({state_dim: 2}), 0.75)


def _two_site_week_inputs() -> dict[str, object]:
    inputs = _inputs()
    sites = np.repeat(["TAC", "MHD"], 7)
    times = np.tile(np.arange("2021-07-12", "2021-07-19", dtype="datetime64[D]"), 2)
    nmeasure = sites.size
    co2_labels = [f"co2:{site}:{time}" for site, time in zip(sites, times, strict=True)]
    o2_labels = [f"o2:{site}:{time}" for site, time in zip(sites, times, strict=True)]
    co2 = xr.DataArray(
        np.linspace(-4.0, -3.0, nmeasure),
        dims="co2_measure",
        coords={
            "co2_measure": co2_labels,
            "site": ("co2_measure", sites),
            "time": ("co2_measure", times),
        },
    )
    o2 = xr.DataArray(
        np.linspace(5.0, 6.0, nmeasure),
        dims="o2_measure",
        coords={
            "o2_measure": o2_labels,
            "site": ("o2_measure", sites),
            "time": ("o2_measure", times),
        },
    )
    state_labels = inputs["retained_prior"].mean["state"]
    inputs.update(
        {
            "co2_observations": co2,
            "o2_observations": o2,
            "co2_prior_forward_mean": co2.copy(),
            "o2_prior_forward_mean": o2.copy(),
            "co2_operator": xr.DataArray(
                np.tile([[-0.4, 0.3, 0.2, -0.1, 0.0]], (nmeasure, 1)),
                dims=("co2_measure", "state"),
                coords={"co2_measure": co2["co2_measure"], "state": state_labels},
            ),
            "o2_operator": xr.DataArray(
                np.tile([[0.5, -0.4, -0.3, 0.0, 0.2]], (nmeasure, 1)),
                dims=("o2_measure", "state"),
                coords={"o2_measure": o2["o2_measure"], "state": state_labels},
            ),
            "co2_aggregation_covariance": xr.DataArray(
                np.eye(nmeasure) * 0.2,
                dims=("co2_measure", "co2_measure_cov"),
                coords={"co2_measure": co2["co2_measure"], "co2_measure_cov": co2_labels},
            ),
            "co2_o2_aggregation_covariance": xr.DataArray(
                np.eye(nmeasure) * 0.01,
                dims=("co2_measure", "o2_measure"),
                coords={"co2_measure": co2["co2_measure"], "o2_measure": o2["o2_measure"]},
            ),
            "o2_aggregation_covariance": xr.DataArray(
                np.eye(nmeasure) * 0.3,
                dims=("o2_measure", "o2_measure_cov"),
                coords={"o2_measure": o2["o2_measure"], "o2_measure_cov": o2_labels},
            ),
            "co2_units": "ppm",
            "o2_units": "ppm",
            "provenance": {
                "verification_games_issue": "OPE-77",
                "period": "2021-07-12/2021-07-19",
                "sites": ["TAC", "MHD"],
            },
        }
    )
    return inputs


def test_two_site_week_runner_persists_labels_roles_units_and_provenance(tmp_path: Path) -> None:
    prepared = prepare_co2_o2_inputs(**_two_site_week_inputs())
    state_dim = prepared.retained_prior.state_dim
    activity = StateActivity(
        active=xr.DataArray(
            [True, True, False, True, True],
            dims=state_dim,
            coords={state_dim: prepared.retained_prior.mean[state_dim]},
        ),
        fixed_value=xr.DataArray(
            [1.0, 1.0, 0.75, 1.0, 1.0],
            dims=state_dim,
            coords={state_dim: prepared.retained_prior.mean[state_dim]},
        ),
    )
    trace = run_rhime_co2_o2_from_prepared_inputs(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared, value=1.0),
        state_activity=activity,
        sampler=RhimeSampler(
            draws=2,
            tune=2,
            chains=1,
            nuts_sampler="pymc",
            sample_kwargs={"random_seed": 19, "compute_convergence_checks": False},
            sample_prior_predictive=False,
            sample_posterior_predictive=False,
        ),
    )

    assert trace.posterior.sizes["observation"] == 28
    assert set(trace.posterior["site"].values) == {"TAC", "MHD"}
    assert np.unique(trace.posterior["time"]).size == 7
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert roles["observation"] == "y"
    assert roles["concentration"] == "y"
    assert roles["modelled_concentration"] == "modelled_concentration"
    assert roles["emissions_sensitivity"] == "co2_o2_operator"
    assert roles["flux_contribution"] == "co2_o2_flux_contribution"
    assert "co2_emissions_sensitivity" not in roles
    assert "o2_emissions_sensitivity" not in roles
    assert "pollution_concentration" not in roles
    assert roles["coherent_prior_contribution"] == "fixed_prior_contribution"
    assert metadata["provenance"]["sites"] == ["TAC", "MHD"]
    assert metadata["o2_operator_ratio"]["convention"] == "embedded_signed_o2_per_co2"
    assert metadata["o2_operator_ratio"]["status"] == "available"
    assert metadata["o2_operator_ratio"]["direction"] == "O2 flux per CO2 flux"
    assert metadata["o2_operator_ratio"]["value"] == [-1.1, -1.0, -1.4]
    assert metadata["o2_operator_ratio"]["source"] == ["GPP", "TER", "FF"]
    assert metadata["o2_operator_ratio"]["scope"] == (
        "shared GPP/TER/FF states; O2 ocean applied directly"
    )
    assert trace.posterior["flux_scaling"].attrs["units"] == "dimensionless flux scale"
    assert json.loads(trace.posterior["flux_scaling"].attrs["rhime_scientific_roles"]) == ["flux_scale"]
    np.testing.assert_allclose(trace.posterior["flux_scaling"].sel(state="ff:1"), 0.75)
    assert trace.posterior["source"].values.tolist() == ["GPP", "TER", "FF", "ocean", "ocean"]
    observed = trace.observed_data["y"]
    np.testing.assert_allclose(observed, prepared.observations)
    assert observed["species"].values.tolist() == prepared.observations["species"].values.tolist()
    np.testing.assert_array_equal(
        observed["observation_units"],
        prepared.observations["observation_units"],
    )
    observed_units = np.unique(prepared.observations["observation_units"].values.astype(str))
    expected_units = (
        str(observed_units[0])
        if observed_units.size == 1
        else "mixed; see observation_units coordinate"
    )
    assert observed.attrs["units"] == expected_units
    assert json.loads(observed.attrs["rhime_scientific_roles"]) == [
        "concentration",
        "observation",
    ]
    assert trace.constant_data["fixed_independent_error_sd"].values.tolist() == [1.0] * 28
    assert "fixed_independent_error_sd" not in trace.observed_data
    assert trace.constant_data["fixed_prior_contribution"].attrs["units"] == expected_units
    assert json.loads(
        trace.constant_data["co2_o2_operator"].attrs["rhime_scientific_roles"]
    ) == ["emissions_sensitivity"]
    assert json.loads(
        trace.posterior["co2_o2_flux_contribution"].attrs["rhime_scientific_roles"]
    ) == ["flux_contribution"]

    path = tmp_path / "co2_o2_trace.nc"
    save_inferencedata(trace, path)
    reloaded = load_inferencedata(path)
    assert json.loads(reloaded.attrs["rhime_model_metadata"])["provenance"]["period"] == (
        "2021-07-12/2021-07-19"
    )
    assert reloaded.posterior["observation_units"].values.tolist() == ["ppm"] * 28
    assert reloaded.constant_data["fixed_independent_error_sd"].attrs["units"] == expected_units
    assert reloaded.observed_data.indexes["observation"].equals(
        prepared.observations.indexes["observation"]
    )
    assert reloaded.observed_data.indexes["observation"].names == [
        "species",
        "channel_observation",
    ]
    covariance = prepared.aggregation_error.covariance
    assert covariance is not None
    assert reloaded.constant_data.indexes["observation_cov"].equals(
        covariance.indexes["observation_cov"]
    )
    assert reloaded.constant_data.indexes["observation_cov"].names == [
        "species_cov",
        "channel_observation_cov",
    ]
