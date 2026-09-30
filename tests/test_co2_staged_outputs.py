"""Scientific component and capability checks for staged CO2 output."""

from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2.co2_affine_output import (
    BoundCo2AffineFluxMap,
    produce_bucket_affine_flux_map,
)
from openghg_inversions.rhime.co2.co2_outputs import (
    _affine_products,
    _basic_product,
    reconstruct_co2_concentrations,
    validate_co2_outputs,
)
from openghg_inversions.rhime.specs import RhimeOutputSpec
from test_co2_affine_output import _prepared


def _components_fixture():
    prepared, _ = _prepared()
    roles = {
        "model_mean": "mean_custom",
        "pollution_concentration": "emission_custom",
        "flux_scale": "state_custom",
        "boundary_concentration": "bc_custom",
        "offset_concentration": "offset_custom",
        "model_error": "sd_custom",
        "concentration": "predictive_custom",
    }
    datasets = {}
    for group, draws in (("prior", 4), ("posterior", 3)):
        scale = xr.DataArray(
            np.arange(draws * 2).reshape(1, draws, 2) / 10 + 1,
            dims=("chain", "draw", "region"),
            coords={"chain": [0], "draw": np.arange(draws), "region": prepared.inv_inputs.region},
            attrs={"units": "1"},
        )
        flux = (
            xr.dot(prepared.inv_inputs.H, scale, dim="region") + prepared.inv_inputs.fixed_prior_contribution
        )
        boundary = xr.ones_like(flux) * 2
        offset = xr.ones_like(flux) * -0.5
        mean = flux + boundary + offset
        datasets[group] = xr.Dataset(
            {
                "state_custom": scale,
                "emission_custom": flux,
                "bc_custom": boundary,
                "offset_custom": offset,
                "mean_custom": mean,
                "sd_custom": xr.ones_like(flux) * 0.8,
            }
        )
        datasets[f"{group}_predictive"] = xr.Dataset(
            {
                "predictive_custom": mean
                + xr.DataArray(np.arange(draws).reshape(1, draws, 1), dims=("chain", "draw", "nmeasure"))
            }
        )
    datasets["/"] = xr.Dataset(
        attrs={
            "rhime_variable_roles": json.dumps(roles),
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
        }
    )
    return prepared, xr.DataTree.from_dict(datasets)


def test_role_selected_components_close_and_preserve_independent_predictive_draws():
    prepared, trace = _components_fixture()
    before = trace.copy(deep=True)
    components = reconstruct_co2_concentrations(trace, prepared)
    for when in ("prior", "posterior"):
        np.testing.assert_allclose(
            components[f"modelled_{when}"],
            components[f"pollution_{when}"] + components[f"boundary_{when}"] + components[f"offset_{when}"],
        )
        np.testing.assert_allclose(
            components[f"residual_{when}"], components.observed - components[f"modelled_{when}"]
        )
        assert components[f"modelled_{when}"].attrs["units"] == "ppm"
    assert components.sizes["draw"] == 3
    assert components.sizes["prior_draw"] == 4
    summary = _basic_product(components)
    np.testing.assert_allclose(
        summary.modelled_posterior_mean, trace.posterior.mean_custom.mean(("chain", "draw"))
    )
    np.testing.assert_allclose(
        summary.predictive_posterior_stdev,
        trace.posterior_predictive.predictive_custom.std(("chain", "draw")),
    )
    assert not np.allclose(summary.predictive_posterior_stdev, summary.modelled_posterior_stdev)
    xr.testing.assert_identical(trace, before)


def test_missing_declared_component_fails_instead_of_silently_inventing_zero():
    prepared, trace = _components_fixture()
    trace["posterior"] = trace.posterior.to_dataset().drop_vars("bc_custom")
    with pytest.raises(ValueError, match="boundary_concentration"):
        reconstruct_co2_concentrations(trace, prepared)


def test_unsupported_requests_are_rejected_before_destinations_exist(tmp_path):
    for output_format in ("legacy", "inv_out", "paris"):
        spec = RhimeOutputSpec(
            output_format=output_format, output_path=str(tmp_path / "unopened"), save_inversion_output=False
        )
        with pytest.raises(ValueError):
            validate_co2_outputs(spec)
    with pytest.raises(ValueError, match="save_inversion_output"):
        validate_co2_outputs(RhimeOutputSpec(output_format="basic"))
    with pytest.raises(ValueError, match="country"):
        validate_co2_outputs(
            RhimeOutputSpec(output_format="basic", country_file="countries.nc", save_inversion_output=False)
        )
    validate_co2_outputs(RhimeOutputSpec(output_format="basic", save_inversion_output=False))
    assert not (tmp_path / "unopened").exists()


def test_basic_native_product_uses_affine_nonunit_reference_and_signed_flux():
    prepared, trace = _components_fixture()
    _, native_mean = _prepared()
    bound = BoundCo2AffineFluxMap(
        produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="fixture"),
        prepared.inv_inputs.alpha_prior_mean,
    )
    native, country = _affine_products(trace, bound, None, None)
    assert country is None
    state = trace.posterior.state_custom.values
    expected = prepared.basis_functions.flux.values * (
        native_mean.values + (state - prepared.inv_inputs.alpha_prior_mean.values)
    )
    np.testing.assert_allclose(native.flux_total_posterior_mean, expected.mean((0, 1))[None, :])
    assert native.attrs["uncertainty_scope"] == "retained_state_conditional"


def test_source_to_sector_sums_signed_joint_draws_before_uncertainty():
    from postprocessing.test_co2_flux_outputs import _fixture

    bound, trace, countries = _fixture()
    spec = RhimeOutputSpec(output_format="basic", save_inversion_output=False)
    mapping = {"fossil": "net", "biosphere": "net"}
    validate_co2_outputs(spec, bound=bound, source_to_sector=mapping)
    native, country = _affine_products(trace, bound, countries, mapping)
    np.testing.assert_allclose(native.flux_net_posterior_stdev, native.flux_total_posterior_stdev)
    np.testing.assert_allclose(country.country_net_posterior_stdev, country.country_total_posterior_stdev)
    with pytest.raises(ValueError, match="every native source"):
        validate_co2_outputs(spec, bound=bound, source_to_sector={"fossil": "net"})
    with pytest.raises(ValueError, match="inversion_grid"):
        validate_co2_outputs(
            replace(spec, output_format="paris", paris_postprocessing_kwargs={"inversion_grid": True}),
            bound=bound,
        )


def test_absent_baseline_is_zero_and_basic_reduces_it_without_sample_axes():
    prepared, trace = _components_fixture()
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles.pop("boundary_concentration")
    roles.pop("offset_concentration")
    trace.attrs["rhime_variable_roles"] = json.dumps(roles)
    components = reconstruct_co2_concentrations(trace, prepared)
    basic = _basic_product(components)
    np.testing.assert_array_equal(basic.boundary_posterior_mean, [0])


def test_conditional_paris_uses_affine_draw_statistics_and_template_units():
    from types import SimpleNamespace

    from postprocessing.test_co2_flux_outputs import _fixture
    from openghg_inversions.rhime.co2.co2_outputs import _paris_products

    prepared, concentration_trace = _components_fixture()
    components = reconstruct_co2_concentrations(concentration_trace, prepared)
    bound, flux_trace, countries = _fixture()
    native, country = _affine_products(flux_trace, bound, countries, None)
    result = SimpleNamespace(
        model_spec=SimpleNamespace(domain="EUROPE"),
        output_spec=RhimeOutputSpec(
            output_format="paris",
            save_inversion_output=False,
            paris_postprocessing_kwargs={"flux_frequency": "monthly"},
        ),
        run_spec=SimpleNamespace(start_date="2020-01-01", end_date="2020-02-01", averaging_period=("1h",)),
    )
    concentration, flux = _paris_products(components, native, country, countries, result, None)
    assert flux.attrs["uncertainty_scope"] == "retained_state_conditional"
    np.testing.assert_allclose(
        flux.flux_total_posterior.isel(time=0), native.flux_total_posterior_mean, rtol=1e-6
    )
    np.testing.assert_allclose(
        flux.stdev_flux_total_posterior_country.isel(time=0),
        country.country_total_posterior_stdev / 1000,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        concentration.mf_posterior, components.modelled_posterior.mean(("chain", "draw")) * 1e-6, rtol=1e-6
    )


def test_concentration_only_basic_accepts_posterior_without_prior_predictions():
    prepared, trace = _components_fixture()
    del trace["prior"]
    del trace["prior_predictive"]
    summary = _basic_product(reconstruct_co2_concentrations(trace, prepared))
    assert "modelled_posterior_mean" in summary
    assert "modelled_prior_mean" not in summary
