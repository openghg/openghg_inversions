"""Boundary replay uses fitted facts without a model or a period calendar."""

import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2.boundary_output import decode_boundary_facts, reconstruct_boundary


def _trace() -> xr.DataTree:
    roles = {
        "boundary_concentration": "mu_bc",
        "boundary_correction": "bc_correction",
        "boundary_mean_shift": "bc_mean_shift",
        "boundary_mean_shift_concentration": "mu_bc_mean_shift",
        "boundary_reference_concentration": "mu_bc_reference",
        "boundary_correction_anomaly": "bc_anomaly",
        "boundary_anomaly_concentration": "mu_bc_anomaly",
    }
    groups = {}
    for group, count in (("prior", 7), ("posterior", 3)):
        mean = xr.DataArray(np.arange(count, dtype=float)[None, :], dims=("chain", "draw"))
        response = xr.DataArray([0.6, 0.9], dims="nmeasure")
        correction = mean.expand_dims(bc_region=["observed", "unobserved"])
        groups[group] = xr.Dataset({
            "bc_mean_shift": mean,
            "bc_correction": correction,
            "bc_anomaly": xr.zeros_like(correction),
            "mu_bc_mean_shift": mean * response,
            "mu_bc_anomaly": xr.zeros_like(mean * response),
            "mu_bc": 400.0 + mean * response,
        })
    groups["constant_data"] = xr.Dataset({"mu_bc_reference": xr.DataArray([400.0, 400.0], dims="nmeasure")})
    trace = xr.DataTree.from_dict(groups)
    trace.attrs.update(rhime_variable_roles=json.dumps(roles), rhime_boundary_format_version=2)
    return trace


def test_mean_only_replay_reads_recorded_zero_anomalies_for_unequal_sample_sizes() -> None:
    trace = _trace()
    original = trace.copy(deep=True)
    facts = decode_boundary_facts(trace)[""]
    for group, count in (("prior", 7), ("posterior", 3)):
        result = reconstruct_boundary(facts, trace[group].to_dataset())
        assert result["anomaly"].sizes == {"bc_region": 2, "chain": 1, "draw": count}
        np.testing.assert_array_equal(result["anomaly"], 0.0)
        np.testing.assert_array_equal(result["anomaly_contribution"], 0.0)
        xr.testing.assert_allclose(result["concentration"], result["reference"] + result["mean_contribution"])
    xr.testing.assert_identical(trace, original)


@pytest.mark.parametrize("version", [None, 1])
def test_old_boundary_posterior_fails_with_prepare_again(version) -> None:
    trace = _trace()
    if version is None:
        del trace.attrs["rhime_boundary_format_version"]
    else:
        trace.attrs["rhime_boundary_format_version"] = version
    with pytest.raises(ValueError, match="prepare again"):
        decode_boundary_facts(trace)


@pytest.mark.parametrize("missing", ["bc_anomaly", "mu_bc_anomaly"])
def test_declared_anomalies_require_saved_draws(missing: str) -> None:
    trace = _trace()
    del trace["posterior"][missing]
    with pytest.raises(ValueError, match="require role 'boundary_.*' in saved draws"):
        reconstruct_boundary(decode_boundary_facts(trace)[""], trace.posterior.to_dataset())


@pytest.mark.parametrize("units", ["ppm", "1"])
@pytest.mark.parametrize("role,quantity", [("boundary_correction", "correction"),
                                           ("boundary_correction_anomaly", "anomaly")])
def test_independent_corrections_replay_through_existing_scientific_roles(
    units: str, role: str, quantity: str,
) -> None:
    from test_co2_staged_outputs import _components_fixture
    from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations

    prepared, trace = _components_fixture()
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles[role] = "independent_curtain_corrections"
    roles["boundary_scale"] = "dimensionless_scale"
    roles["boundary_reference_concentration"] = "reference"
    roles["boundary_anomaly_concentration"] = "contribution"
    trace.attrs["rhime_variable_roles"] = json.dumps(roles)
    expected = {}
    for group in ("prior", "posterior"):
        draws = trace[group].to_dataset()
        correction = xr.DataArray(
            np.arange(draws.sizes["draw"] * 2, dtype=float).reshape(1, -1, 2) / 10,
            dims=("chain", "draw", "curtain"),
            coords={"chain": draws.chain, "draw": draws.draw, "curtain": ["west", "north"]},
            attrs={"units": units},
        )
        trace[group]["independent_curtain_corrections"] = correction
        expected[group] = correction
        trace[group]["dimensionless_scale"] = correction.assign_attrs(units="ppm")
        trace[group]["state_custom"].attrs["units"] = "ppm"
        trace[group]["contribution"] = xr.zeros_like(draws.bc_custom).assign_attrs(units="ppm")
    trace["constant_data"] = xr.Dataset({
        "reference": xr.full_like(prepared.inv_inputs.mf, 2.0).assign_attrs(units="ppm"),
    })
    before = trace.copy(deep=True)
    components = reconstruct_co2_concentrations(trace, prepared)
    for group, value in expected.items():
        if group == "prior":
            value = value.rename(chain="prior_chain", draw="prior_draw")
        xr.testing.assert_identical(components[f"boundary_{quantity}_{group}"].rename(value.name), value)
        assert components[f"boundary_{quantity}_{group}"].attrs["units"] == units
        for name in ("boundary", "boundary_reference", "boundary_anomaly_contribution"):
            assert components[f"{name}_{group}"].attrs["units"] == "ppm"
        for name in ("flux_scale", "boundary_scale"):
            assert components[f"{name}_{group}"].attrs["units"] == "1"
        xr.testing.assert_allclose(
            components[f"boundary_{group}"],
            components[f"boundary_reference_{group}"] + components[f"boundary_anomaly_contribution_{group}"],
        )
    xr.testing.assert_identical(trace, before)
