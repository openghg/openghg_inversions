"""Joint-draw closure and per-species PARIS serialization."""

import json

import arviz as az
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.linked_paris_outputs import (
    make_co2_o2_paris_outputs,
    reconstruct_co2_o2_concentrations,
)
from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_model import _gather_co2_o2_sensitivity
from test_rhime_co2_o2 import _inputs


def _joint_fixture(
    *,
    baseline=True,
    o2_units="ppm",
    numeric_state=False,
    tracer_scopes=None,
    zero_ocean_sensitivities=False,
):
    inputs = _inputs()
    inputs["o2_units"] = o2_units
    if tracer_scopes is not None:
        from openghg_inversions.correlated_state import CorrelatedLognormalPrior

        prior = inputs["retained_prior"]
        inputs["retained_prior"] = CorrelatedLognormalPrior(
            prior.mean.assign_coords(tracer_scope=("state", tracer_scopes)),
            prior.arithmetic_covariance.values,
        )
    if zero_ocean_sensitivities:
        for name in ("co2_sensitivity", "o2_sensitivity"):
            inputs[name] = inputs[name] * xr.DataArray([1, 1, 1, 0, 0], dims="state")
    if numeric_state:
        from openghg_inversions.correlated_state import CorrelatedLognormalPrior

        prior = inputs["retained_prior"]
        inputs["retained_prior"] = CorrelatedLognormalPrior(
            prior.mean.assign_coords(state=np.arange(5)), prior.arithmetic_covariance.values
        )
        for name in ("co2_sensitivity", "o2_sensitivity", "o2_co2_flux_ratio"):
            array = inputs[name]
            inputs[name] = array.assign_coords(state=np.arange(array.sizes["state"]))
    for species, count in (("co2", 2), ("o2", 3)):
        observed = inputs[f"{species}_observations"]
        inputs[f"{species}_observations"] = observed.assign_coords(
            site=(observed.dims[0], ["BSD"] * count),
            time=(observed.dims[0], np.arange(count) * np.timedelta64(4, "h") + np.datetime64("2024-01-01")),
        )
    prepared = prepare_co2_o2_inputs(**inputs)
    sensitivity = _gather_co2_o2_sensitivity(prepared.co2_sensitivity, prepared.o2_sensitivity)
    state = prepared.retained_prior.mean
    scaling = xr.DataArray(
        np.arange(30).reshape(2, 3, 5) / 30 + 0.5,
        dims=("chain", "draw", "state"),
        coords={"chain": [0, 1], "draw": [0, 1, 2], **dict(state.coords)},
    )
    boundary = xr.where(prepared.observations.species == "co2", 400.0, 0.0)
    offset = xr.where(prepared.observations.species == "o2", 0.25, 0.0)
    modelled = xr.dot(sensitivity, scaling, dim="state") + prepared.fixed_prior_contribution
    posterior = xr.Dataset({"scales": scaling, "prediction": modelled, "error": xr.ones_like(modelled)})
    roles = {"flux_scale": "scales", "modelled_concentration": "prediction", "total_marginal_error": "error"}
    prior = xr.Dataset(
        {
            "scales": xr.ones_like(scaling.isel(chain=[0], draw=[0, 1])).reindex(
                draw=[0, 1, 2, 3], fill_value=1
            )
        }
    )
    if baseline:
        roles.update(
            boundary_concentration="bc",
            offset_concentration="bias",
            co2_boundary_concentration="bc",
            o2_offset_concentration="bias",
        )
        posterior["bc"] = boundary
        posterior["bias"] = offset
        posterior["prediction"] = modelled + boundary + offset
        prior["bc"] = boundary
        prior["bias"] = offset
    trace = az.InferenceData(
        posterior=posterior,
        prior=prior,
        attrs={
            "rhime_recipe": "co2_o2",
            "rhime_variable_roles": json.dumps(roles),
            "rhime_model_metadata": json.dumps({"provenance": "joint fixture"}),
        },
    )
    return prepared, trace


@pytest.mark.parametrize("tracer_scopes", [None, ["SHARED", "Shared", "sHaReD", "CO2", "O2"]])
def test_separate_products_close_and_roundtrip(tmp_path, tracer_scopes):
    prepared, trace = _joint_fixture(tracer_scopes=tracer_scopes)
    before = trace.copy()
    components = reconstruct_co2_o2_concentrations(trace, prepared)
    products = make_co2_o2_paris_outputs(trace, prepared, output_path=tmp_path)
    for species, count in (("co2", 2), ("o2", 3)):
        raw = components[species]
        total = raw.affine + raw.boundary_posterior + raw.offset_posterior
        for source in ("gpp", "ter", "ff", "ocean"):
            total = total + raw[f"{source}_posterior"]
        xr.testing.assert_allclose(total, raw.modelled_posterior.transpose(*total.dims))
        dataset = products[species]["concentration"]
        assert dataset.sizes["index"] == count
        assert dataset.attrs["species"] == species
        if tracer_scopes is not None:
            assert prepared.retained_prior.mean.tracer_scope.values.tolist() == tracer_scopes
            assert json.loads(dataset.attrs["linked_state_provenance"])["tracer_scope"] == tracer_scopes
        assert dataset.attrs["paris_concentration_template_version"] == "v04"
        assert dataset.mf_posterior.attrs["units"] == "mol mol-1"
        assert "cross-channel covariance" in dataset.attrs["linked_posterior"]
        np.testing.assert_allclose(
            dataset.mf_posterior, raw.modelled_posterior.mean(("chain", "draw")) * 1e-6
        )
        np.testing.assert_allclose(dataset.mf_bc_posterior, (400 if species == "co2" else 0.25) * 1e-6)
        expected_ocean = prepared.co2_sensitivity[:, 3] if species == "co2" else prepared.o2_sensitivity[:, 4]
        scale = trace.posterior.scales[:, :, 3 if species == "co2" else 4].mean()
        np.testing.assert_allclose(dataset.mf_ocean_posterior, expected_ocean.values * scale.values * 1e-6)
        with xr.open_dataset(tmp_path / f"{species}_concentration.nc") as restored:
            assert restored.attrs["linked_ratio_provenance"] == dataset.attrs["linked_ratio_provenance"]
            assert restored.attrs["rhime_model_metadata"] == dataset.attrs["rhime_model_metadata"]
            assert restored.sizes["index"] == count
            np.testing.assert_allclose(restored.mf_posterior, dataset.mf_posterior)
    xr.testing.assert_identical(trace.posterior, before.posterior)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["co2_concentration.nc", "o2_concentration.nc"]


def test_no_baseline_and_no_prior_draws():
    prepared, trace = _joint_fixture(baseline=False)
    products = make_co2_o2_paris_outputs(trace, prepared)
    for outputs in products.values():
        assert (outputs["concentration"].mf_bc_posterior == 0).all()
        assert "mf_bias_posterior" not in outputs["concentration"]
        assert (outputs["concentration"].stdev_mf_prior == 0).all()
    del trace.prior
    with pytest.raises(ValueError, match="flux_scale.*prior"):
        make_co2_o2_paris_outputs(trace, prepared)


def test_unsupported_units_templates_and_missing_components():
    prepared, trace = _joint_fixture()
    with pytest.raises(ValueError, match="template_version"):
        make_co2_o2_paris_outputs(trace, prepared, template_version="legacy")
    bad = trace.copy()
    bad.posterior["prediction"] = bad.posterior.prediction + 1.0
    with pytest.raises(ValueError, match="does not close"):
        make_co2_o2_paris_outputs(bad, prepared)
    del trace.prior["bc"]
    with pytest.raises(ValueError, match="boundary_concentration.*prior"):
        make_co2_o2_paris_outputs(trace, prepared)
    prepared, trace = _joint_fixture(baseline=False, o2_units="per meg")
    with pytest.raises(ValueError, match="cannot represent"):
        make_co2_o2_paris_outputs(trace, prepared)


def test_fixed_state_prior_draws_are_preserved():
    prepared, trace = _joint_fixture(baseline=False)
    trace.prior["scales"] = xr.where(trace.prior.state == "gpp:1", 2.0, trace.prior.scales)
    values = reconstruct_co2_o2_concentrations(trace, prepared)
    np.testing.assert_allclose(values["co2"].gpp_prior.mean(("prior_chain", "prior_draw")), [2, 1])
    trace.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    assert (
        make_co2_o2_paris_outputs(trace, prepared)["co2"]["concentration"].attrs["rhime_recipe"]
        == "co2_o2_cached_sigma_fixed_ou"
    )


def test_native_flux_reuses_joint_state_and_roundtrips(tmp_path, europe_country_file):
    from dataclasses import replace
    from openghg_inversions.basis.basis_functions import BasisFunctions

    prepared, trace = _joint_fixture(baseline=False, numeric_state=True)
    original_posterior = trace.posterior.copy(deep=True)
    original_prior = trace.prior.copy(deep=True)
    country = xr.load_dataset(europe_country_file).isel(
        lat=slice(0, 2), lon=slice(0, 5), ncountries=slice(0, 1)
    )
    country["name"] = ("ncountries", ["United Kingdom"])
    country["country"] = xr.zeros_like(country.country)
    country_path = tmp_path / "countries.nc"
    country.to_netcdf(country_path)
    flat = xr.DataArray(
        np.tile(np.arange(1, 6), (2, 1)), dims=("lat", "lon"), coords={"lat": country.lat, "lon": country.lon}
    )
    bases = {}
    for species, excluded, sign in (("co2", 5, 1), ("o2", 4, -1)):
        flux = xr.where(flat == excluded, 0.0, sign * 1e-6)
        flux.attrs["units"] = "mol / m^2 / s"
        bases[species] = BasisFunctions.from_flat_basis(flat, flux, metadata={"native_source": species})
    kwargs = dict(
        native_flux_bases=bases,
        start_date="2024-01-01",
        end_date="2024-02-01",
        country_file=country_path,
        country_selections=None,
    )
    products = make_co2_o2_paris_outputs(trace, prepared, output_path=tmp_path / "outputs", **kwargs)
    for species in ("co2", "o2"):
        flux = products[species]["flux"]
        assert flux.attrs["paris_flux_template_version"] == "v03"
        assert "excludes unresolved" in flux.attrs["native_uncertainty"]
        assert json.loads(flux.attrs["linked_native_flux_provenance"]) == {"native_source": species}
        expected = bases[species].flux.values * trace.posterior.scales.mean(("chain", "draw")).values[None, :]
        np.testing.assert_allclose(flux.flux_total_posterior.isel(time=0), expected, rtol=1e-6)
        assert (flux.stdev_flux_total_posterior >= 0).all()
        assert flux.stdev_flux_total_posterior.attrs["units"] == "mol m-2 s-1"
        assert flux.percentile_flux_total_posterior.attrs["units"] == "mol m-2 s-1"
        assert (flux.percentile_flux_total_posterior.diff("percentile") >= 0).all()
        for when in ("prior", "posterior"):
            draws = getattr(trace, when).scales.values[..., None, :] * bases[species].flux.values
            np.testing.assert_allclose(
                flux[f"stdev_flux_total_{when}"].isel(time=0), draws.std(axis=(0, 1)), atol=1e-12
            )
            np.testing.assert_allclose(
                flux[f"percentile_flux_total_{when}"].isel(time=0),
                np.quantile(draws, [0.159, 0.841], axis=(0, 1)),
                atol=1e-12,
            )
        with xr.open_dataset(tmp_path / "outputs" / f"{species}_flux.nc") as restored:
            assert restored.attrs["native_uncertainty"] == flux.attrs["native_uncertainty"]
    xr.testing.assert_identical(trace.posterior, original_posterior)
    xr.testing.assert_identical(trace.prior, original_prior)
    bad_bases = dict(bases)
    bad_bases["o2"] = replace(bases["o2"], flux=xr.ones_like(bases["o2"].flux))
    with pytest.raises(ValueError, match="other tracer"):
        make_co2_o2_paris_outputs(trace, prepared, **{**kwargs, "native_flux_bases": bad_bases})
    bad_bases["o2"] = replace(bases["o2"], flux=bases["o2"].flux.assign_attrs(units="kg"))
    with pytest.raises(ValueError, match="flux units"):
        make_co2_o2_paris_outputs(trace, prepared, **{**kwargs, "native_flux_bases": bad_bases})


@pytest.mark.parametrize("species", ["co2", "o2"])
def test_native_flux_rejects_uppercase_private_states_without_concentration_signal(species):
    from openghg_inversions.basis.basis_functions import BasisFunctions

    prepared, trace = _joint_fixture(
        baseline=False,
        numeric_state=True,
        tracer_scopes=["shared", "shared", "shared", "CO2", "O2"],
        zero_ocean_sensitivities=True,
    )
    # Both private ocean states are invisible to concentration closure. They
    # must still be excluded from the other tracer's native flux product.
    make_co2_o2_paris_outputs(trace, prepared)
    flat = xr.DataArray(
        np.tile(np.arange(1, 6), (2, 1)),
        dims=("lat", "lon"),
        coords={"lat": [50.0, 51.0], "lon": np.arange(5.0)},
    )
    flux = xr.ones_like(flat, dtype=float).assign_attrs(units="mol / m^2 / s")
    basis = BasisFunctions.from_flat_basis(flat, flux)
    with pytest.raises(ValueError, match=f"Native {species} flux basis includes the other tracer"):
        make_co2_o2_paris_outputs(
            trace,
            prepared,
            native_flux_bases={species: basis},
            start_date="2024-01-01",
            end_date="2024-02-01",
        )


@pytest.mark.parametrize("native_multiindex", [False, True])
def test_cached_joint_posterior_with_baseline_emits_paris(tmp_path, native_multiindex):
    from openghg_inversions.rhime.co2 import run_rhime_co2_o2_cached_sigma_from_prepared_inputs
    from openghg_inversions.rhime.sampling import RhimeSampler
    from openghg_inversions.serialization import load_inferencedata, save_inferencedata
    from test_rhime_co2_o2 import _independent_error
    from test_rhime_co2_o2_baselines import _native_multiindex_inputs, _prepared

    if native_multiindex:
        inputs = _native_multiindex_inputs(extra_level=True, co2_dim="observation")
        observed = inputs["co2_observations"]
        inputs["boundary_sensitivity"] = {
            "co2": xr.DataArray(
                [[1.0, 2.0], [3.0, 4.0]],
                dims=(observed.dims[0], "boundary"),
                coords={**observed.coords, "boundary": ["north", "south"]},
            )
        }
        prepared = prepare_co2_o2_inputs(**inputs)
    else:
        prepared = _prepared(("co2",))
    trace = run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared),
        tau_hours=12.0,
        site_amplitude_prior_scale=0.5,
        bc_prior={"co2": {"pdf": "normal", "mu": 1.0, "sigma": 0.1}},
        offset_prior={"o2": {"pdf": "normal", "mu": 0.0, "sigma": 0.1}},
        offset_args={"o2": {"per_site": False}},
        sampler=RhimeSampler(
            nuts_sampler="pymc",
            draws=3,
            tune=3,
            burn=0,
            chains=1,
            sample_kwargs={
                "cores": 1,
                "random_seed": 185,
                "progressbar": False,
                "compute_convergence_checks": False,
            },
            sample_prior_predictive=5,
        ),
    )
    save_inferencedata(trace, tmp_path / "joint.nc")
    restored = load_inferencedata(tmp_path / "joint.nc")
    products = make_co2_o2_paris_outputs(restored, prepared, output_path=tmp_path / "paris")
    roles = json.loads(restored.attrs["rhime_variable_roles"])
    for species, rows in (("co2", slice(0, 2)), ("o2", slice(2, 5))):
        dataset = products[species]["concentration"]
        expected = (
            restored.posterior[roles["modelled_concentration"]].isel(observation=rows).mean(("chain", "draw"))
        )
        np.testing.assert_allclose(dataset.mf_posterior, expected * 1e-6, rtol=1e-6)
        assert dataset.attrs["rhime_recipe"] == "co2_o2_cached_sigma_fixed_ou"
    assert "mf_bias_posterior" not in products["co2"]["concentration"]
    assert "mf_bias_posterior" in products["o2"]["concentration"]
    assert restored.log_likelihood.y.dims == ("chain", "draw")
