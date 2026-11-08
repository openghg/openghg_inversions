"""Joint-draw closure and per-species PARIS serialization."""

import json
from dataclasses import replace

from dask.callbacks import Callback
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
    trace = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={
                    "rhime_recipe": "co2_o2",
                    "rhime_boundary_format_version": 2,
                    "rhime_variable_roles": json.dumps(roles),
                    "rhime_model_metadata": json.dumps({"provenance": "joint fixture"}),
                }
            ),
            "posterior": posterior,
            "prior": prior,
        }
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
        scale = trace["posterior"].scales[:, :, 3 if species == "co2" else 4].mean()
        np.testing.assert_allclose(dataset.mf_ocean_posterior, expected_ocean.values * scale.values * 1e-6)
        with xr.open_dataset(tmp_path / f"{species}_concentration.nc") as restored:
            assert restored.attrs["linked_ratio_provenance"] == dataset.attrs["linked_ratio_provenance"]
            assert restored.attrs["rhime_model_metadata"] == dataset.attrs["rhime_model_metadata"]
            assert restored.sizes["index"] == count
            np.testing.assert_allclose(restored.mf_posterior, dataset.mf_posterior)
    xr.testing.assert_identical(trace["posterior"], before["posterior"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["co2_concentration.nc", "o2_concentration.nc"]


def test_no_baseline_and_no_prior_draws():
    prepared, trace = _joint_fixture(baseline=False)
    products = make_co2_o2_paris_outputs(trace, prepared)
    for outputs in products.values():
        assert (outputs["concentration"].mf_bc_posterior == 0).all()
        assert "mf_bias_posterior" not in outputs["concentration"]
        assert (outputs["concentration"].stdev_mf_prior == 0).all()
    del trace["prior"]
    with pytest.raises(ValueError, match="flux_scale.*prior"):
        make_co2_o2_paris_outputs(trace, prepared)


def test_unsupported_units_templates_and_missing_components():
    prepared, trace = _joint_fixture()
    with pytest.raises(ValueError, match="template_version"):
        make_co2_o2_paris_outputs(trace, prepared, template_version="legacy")
    bad = trace.copy()
    bad["posterior"]["prediction"] = bad["posterior"].prediction + 1.0
    with pytest.raises(ValueError, match="does not close"):
        make_co2_o2_paris_outputs(bad, prepared)
    del trace["prior"]["bc"]
    with pytest.raises(ValueError, match="boundary_concentration"):
        make_co2_o2_paris_outputs(trace, prepared)
    prepared, trace = _joint_fixture(baseline=False, o2_units="per meg")
    with pytest.raises(ValueError, match="cannot represent"):
        make_co2_o2_paris_outputs(trace, prepared)


def test_fixed_state_prior_draws_are_preserved():
    prepared, trace = _joint_fixture(baseline=False)
    trace["prior"]["scales"] = xr.where(trace["prior"].state == "gpp:1", 2.0, trace["prior"].scales)
    values = reconstruct_co2_o2_concentrations(trace, prepared)
    np.testing.assert_allclose(values["co2"].gpp_prior.mean(("prior_chain", "prior_draw")), [2, 1])
    trace.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    assert (
        make_co2_o2_paris_outputs(trace, prepared)["co2"]["concentration"].attrs["rhime_recipe"]
        == "co2_o2_cached_sigma_fixed_ou"
    )


@pytest.mark.parametrize("anomaly_units,zero_anomaly", [("1", False), ("ppm", False), ("ppm", True)])
def test_declared_boundary_coefficients_survive_linked_products(monkeypatch, anomaly_units, zero_anomaly):
    from openghg_inversions.postprocessing import linked_paris_outputs
    from openghg_inversions.rhime.co2.co2_o2_outputs import (
        Co2O2Output, Co2O2OutputOptions, calculate_co2_o2_products,
    )

    prepared, trace = _joint_fixture(baseline=False)
    # Exercise native-row selection independently of preparation's channel concatenation.
    order = [2, 0, 3, 1, 4]
    sensitivity = _gather_co2_o2_sensitivity(prepared.co2_sensitivity, prepared.o2_sensitivity).isel(
        observation=order
    )
    monkeypatch.setattr(linked_paris_outputs, "_gather_co2_o2_sensitivity", lambda *_: sensitivity)
    prepared = replace(
        prepared,
        observations=prepared.observations.isel(observation=order),
        fixed_prior_contribution=prepared.fixed_prior_contribution.isel(observation=order),
    )
    trace["posterior"] = trace.posterior.to_dataset().isel(observation=order)
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    roles["boundary_concentration"] = "all_boundary"
    constants = xr.Dataset()
    for species in ("co2", "o2"):
        prefix = f"{species}_"
        declared = {
            "boundary_concentration": "contribution",
            "boundary_reference_concentration": "reference",
            "boundary_mean_shift": "level",
            "boundary_mean_shift_concentration": "level_response",
        }
        declared.update(boundary_correction_anomaly="pattern", boundary_anomaly_concentration="pattern_response")
        roles.update({prefix + role: prefix + name for role, name in declared.items()})
        for group in ("posterior", "prior"):
            draws = trace[group].to_dataset()
            samples = draws.scales.isel(state=0, drop=True).drop_vars(["source", "tracer_scope"], errors="ignore")
            level = (samples * xr.DataArray(
                [0.2, -0.1], dims=prefix + "level_period",
                coords={prefix + "level_period": np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")},
            )).assign_attrs(units="ppm", long_name="physical level coefficient").chunk()
            pattern = (samples * xr.DataArray(
                [0.1, -0.3, 0.2], dims=prefix + "pattern_term",
                coords={prefix + "pattern_term": ["east", "west", "top"]},
            )).assign_attrs(units=anomaly_units, long_name="physical pattern coefficient").chunk()
            native = prepared.observations.species == species
            reference = xr.where(native, 400.0 if species == "co2" else 209300.0, 0.0)
            level_response = xr.where(native, level.sum(prefix + "level_period"), 0.0)
            pattern_response = xr.where(native, (pattern * [1.0, 2.0, 4.0]).sum(prefix + "pattern_term"), 0.0)
            if zero_anomaly:
                # Unsampled anomalies are recorded as zeros.
                pattern, pattern_response = xr.zeros_like(pattern), xr.zeros_like(level_response)
            draws[prefix + "pattern"] = pattern
            draws[prefix + "pattern_response"] = pattern_response.assign_attrs(units="ppm")
            draws[prefix + "level"] = level
            draws[prefix + "reference"] = reference.assign_attrs(units="ppm")
            draws[prefix + "level_response"] = level_response.assign_attrs(units="ppm")
            contribution = reference + level_response + pattern_response
            draws[prefix + "contribution"] = contribution.assign_attrs(units="ppm")
            draws["all_boundary"] = draws.get("all_boundary", 0.0) + contribution
            if group == "posterior":
                draws["prediction"] = draws.prediction + contribution
            trace[group] = draws
    trace["constant_data"] = constants
    trace.attrs.update(rhime_boundary_format_version=2, rhime_variable_roles=json.dumps(roles))
    before = trace.copy(deep=True)
    observed_data = prepared.observations.data
    coefficient_data = trace.posterior.co2_level.data
    tasks = []
    output = Co2O2Output(prepared, trace, "2024-01-01", "2024-02-01", "EUROPE")
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        components = reconstruct_co2_o2_concentrations(trace, prepared)
    assert tasks == []
    calculate_co2_o2_products(result=output, output=Co2O2OutputOptions())
    products = make_co2_o2_paris_outputs(trace, prepared)
    for species, raw in components.items():
        assert raw.sizes["draw"] == 3 and raw.sizes["prior_draw"] == 4
        xr.testing.assert_equal(
            raw.observed,
            prepared.observations.where(prepared.observations.species == species, drop=True).rename("observed"),
        )
        for group in ("posterior", "prior"):
            for component, source in (
                ("mean_shift", "level"), ("anomaly", "pattern"),
            ):
                name = f"boundary_{component}_coefficient_{group}"
                expected = trace[group][f"{species}_{source}"]
                if group == "prior":
                    expected = expected.rename(chain="prior_chain", draw="prior_draw")
                xr.testing.assert_identical(raw[name], expected.rename(name))
                assert "observation" not in raw[name].dims
                assert hasattr(raw[name].data, "__dask_graph__")
                assert name in output.outputs[f"{species}_concentration_components"]
                sample_dims = ("prior_chain", "prior_draw") if group == "prior" else ("chain", "draw")
                xr.testing.assert_allclose(
                    output.outputs[f"{species}_basic"][f"{name}_mean"],
                    expected.mean(sample_dims).rename(f"{name}_mean"),
                )
            xr.testing.assert_allclose(
                raw[f"boundary_{group}"],
                (raw[f"boundary_reference_{group}"] + raw[f"boundary_mean_shift_{group}"]
                 + raw[f"boundary_anomaly_{group}"]).transpose(*raw[f"boundary_{group}"].dims),
            )
            total = raw[f"pollution_{group}"] + raw[f"boundary_{group}"] + raw[f"offset_{group}"]
            xr.testing.assert_allclose(total, raw[f"modelled_{group}"].transpose(*total.dims))
        if zero_anomaly:
            assert (raw.boundary_anomaly_coefficient_prior == 0).all()
            assert (raw.boundary_anomaly_posterior == 0).all()
        paris = products[species]["concentration"]
        assert not any("coefficient" in str(name) for name in paris.variables)
        assert set(paris.dims) <= {"index", "platform", "sector", "percentile", "nbnds"}
        np.testing.assert_allclose(paris.mf_bc_posterior, raw.boundary_posterior.mean(("chain", "draw")) * 1e-6)
    xr.testing.assert_identical(trace, before)
    assert trace.posterior.co2_level.data is coefficient_data
    assert prepared.observations.data is observed_data


def test_native_flux_reuses_joint_state_and_roundtrips(tmp_path):
    from dataclasses import replace
    from openghg_inversions.basis.basis_functions import BasisFunctions

    prepared, trace = _joint_fixture(baseline=False, numeric_state=True)
    original_posterior = trace["posterior"].copy(deep=True)
    original_prior = trace["prior"].copy(deep=True)
    country = xr.Dataset(
        {"country": (("lat", "lon"), np.zeros((2, 5), dtype=int)),
         "name": ("ncountries", ["United Kingdom"])},
        coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0, 0.0, 1.0, 2.0]},
    )
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
        expected = bases[species].flux.values * trace["posterior"].scales.mean(("chain", "draw")).values[None, :]
        np.testing.assert_allclose(flux.flux_total_posterior.isel(time=0), expected, rtol=1e-6)
        assert (flux.stdev_flux_total_posterior >= 0).all()
        assert flux.stdev_flux_total_posterior.attrs["units"] == "mol m-2 s-1"
        assert flux.percentile_flux_total_posterior.attrs["units"] == "mol m-2 s-1"
        assert (flux.percentile_flux_total_posterior.diff("percentile") >= 0).all()
        for when in ("prior", "posterior"):
            draws = trace[when].scales.values[..., None, :] * bases[species].flux.values
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
    xr.testing.assert_identical(trace["posterior"], original_posterior)
    xr.testing.assert_identical(trace["prior"], original_prior)
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
def test_cached_joint_draws_with_baseline_emit_paris(tmp_path, native_multiindex, monkeypatch):
    import pymc as pm

    from openghg_inversions.models.boundary import MultiplicativeChoice
    from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2 import run_rhime_co2_o2_cached_sigma_from_prepared_inputs
    from openghg_inversions.rhime.co2 import co2_o2_cached_sigma_runner
    from openghg_inversions.rhime.sampling import RhimeSampler
    from openghg_inversions.serialization import load_trace, save_trace
    from test_rhime_co2_o2 import _independent_error
    from test_rhime_co2_o2_baselines import _native_multiindex_inputs, _prepared

    def graph_draws(built, sampler, *, recompute_deterministics=False):
        assert recompute_deterministics is True
        # PARIS closure needs draws from the real graph; linked sampler
        # integration is covered in test_co2_o2_fixed_ou.py.
        with built.model:
            trace = pm.sample_prior_predictive(draws=5, random_seed=185)
        trace["posterior"] = trace["prior"].to_dataset().isel(draw=slice(0, 3))
        return restore_inferencedata_coords(trace, get_coord_registry(built.model))

    monkeypatch.setattr(co2_o2_cached_sigma_runner, "sample_rhime_model", graph_draws)
    monkeypatch.setattr(co2_o2_cached_sigma_runner, "_sampler_for_cached_graph", lambda sampler, **_: sampler)

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
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5},
        boundary={"co2": MultiplicativeChoice({"pdf": "normal", "mu": 1.0, "sigma": 0.1})},
        offset_prior={"o2": {"pdf": "normal", "mu": 0.0, "sigma": 0.1}},
        offset_args={"o2": {"per_site": False}},
        sampler=RhimeSampler(
            nuts_sampler="pymc",
            sample_prior_predictive=5,
        ),
    )
    save_trace(trace, tmp_path / "joint.nc")
    restored = load_trace(tmp_path / "joint.nc")
    products = make_co2_o2_paris_outputs(restored, prepared, output_path=tmp_path / "paris")
    roles = json.loads(restored.attrs["rhime_variable_roles"])
    for species, rows in (("co2", slice(0, 2)), ("o2", slice(2, 5))):
        dataset = products[species]["concentration"]
        expected = (
            restored["posterior"][roles["modelled_concentration"]].isel(observation=rows).mean(("chain", "draw"))
        )
        np.testing.assert_allclose(dataset.mf_posterior, expected * 1e-6, rtol=1e-6)
        assert dataset.attrs["rhime_recipe"] == "co2_o2_cached_sigma_fixed_ou"
    assert "mf_bias_posterior" not in products["co2"]["concentration"]
    assert "mf_bias_posterior" in products["o2"]["concentration"]
