"""Shared CO2 scientific equations and independent full/saved adapters."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from examples.co2_handoffs import END_DATE, START_DATE, configuration, synthetic_inputs
from openghg_inversions.basis.affine_flux_map_io import save as save_affine
from openghg_inversions.rhime.co2 import inference as co2_inference
from openghg_inversions.rhime.co2.co2_affine_output import prepared_inputs_content_id
from openghg_inversions.rhime.co2.co2_affine_output import _bind_affine_flux_map
from openghg_inversions.rhime.co2.co2_outputs import (
    Co2OutputOptions, calculate_co2_products, make_co2_output, validate_co2_product_request,
)
from openghg_inversions.rhime.co2.configuration import resolve_co2_model, _resolve_co2_sampling


def test_scientific_view_and_products_preserve_complete_trace(monkeypatch):
    """Sample groups keep separate axes and remain the caller's original tree."""
    from test_co2_staged_outputs import _components_fixture

    prepared, trace = _components_fixture()
    for name in tuple(trace.children):
        dataset = trace[name].to_dataset()
        if name.startswith("prior"):
            dataset = dataset.assign_coords(chain=[10], draw=[100, 101, 102, 103])
        else:
            reversed_draws = dataset.isel(draw=slice(None, None, -1)).assign_coords(draw=dataset.draw)
            dataset = xr.concat([dataset, reversed_draws], dim="chain").assign_coords(
                chain=[20, 30], draw=[7, 8, 9],
            )
        for variable in dataset.data_vars:
            dataset[variable].attrs["units"] = "1" if variable == "state_custom" else "ppm"
        trace[name] = dataset
    before = trace.copy(deep=True)
    view = make_co2_output(
        prepared=prepared, trace=trace, start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
    )
    assert view.prepared is prepared
    assert view.idata is trace
    assert view.idata.prior.sizes["draw"] == 4
    assert view.idata.posterior.sizes["draw"] == 3
    assert set(view.idata.children) == set(trace.children)
    assert not hasattr(view, "output_spec")
    assert not hasattr(view, "source_to_sector")
    to_dataset = xr.DataTree.to_dataset

    def child_dataset_only(tree, *args, **kwargs):
        if tree is trace:
            raise AssertionError("Product calculation must not flatten the complete trace tree")
        return to_dataset(tree, *args, **kwargs)

    with monkeypatch.context() as product_checks:
        product_checks.setattr(xr.DataTree, "to_dataset", child_dataset_only)
        output = Co2OutputOptions()
        countries = validate_co2_product_request(output, prepared=prepared)
        calculate_co2_products(result=view, prepared=prepared, output=output, countries=countries)
    assert view.idata is trace
    assert view.outputs["concentration_components"].sizes["prior_draw"] == 4
    assert view.outputs["concentration_components"].sizes["draw"] == 3
    assert view.outputs["concentration_components"].sizes["prior_chain"] == 1
    assert view.outputs["concentration_components"].sizes["chain"] == 2
    np.testing.assert_allclose(
        view.outputs["basic"].modelled_prior_mean, trace.prior.mean_custom.mean(("chain", "draw")),
    )
    np.testing.assert_allclose(
        view.outputs["basic"].modelled_posterior_mean, trace.posterior.mean_custom.mean(("chain", "draw")),
    )
    xr.testing.assert_identical(trace, before)


@pytest.mark.parametrize("option", ["save_inversion_output", "output_filename_convention", "save_basis"])
def test_ordinary_output_options_reject_unrelated_standard_flags(option):
    with pytest.raises(ValueError, match="Unsupported CO2 output"):
        Co2OutputOptions.from_params({option: False})


def test_ordinary_output_options_normalize_only_their_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = Co2OutputOptions.from_params({"output_path": "products", "save_trace": "samples.nc"})
    assert options.output_path == tmp_path / "products"
    assert options.save_trace == tmp_path / "samples.nc"
    assert options.output_name == "rhime"


@pytest.mark.parametrize("likelihood_kind", ["additive_sigma", "fixed_ou"])
def test_shared_inference_preserves_coherent_equations_and_products(
    tmp_path, monkeypatch, likelihood_kind,
):
    """Each inference algorithm preserves the independent coherent equations."""
    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    prepared_before = prepared.inv_inputs.copy(deep=True)
    params = configuration(tmp_path / "full-products", countries)
    if likelihood_kind == "fixed_ou":
        params["likelihood"] = {
            "kind": "fixed_ou", "tau_hours": 24.0,
            "site_amplitude_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
        }
    choices = {name: params[name] for name in ("model", "likelihood")}
    model = resolve_co2_model(choices)
    sampling = _resolve_co2_sampling(params, model=model)
    built_models = []
    build = co2_inference.build_co2_inference

    def capture_build(*args, **kwargs):
        built = build(*args, **kwargs)
        built_models.append(built)
        return built

    monkeypatch.setattr(co2_inference, "build_co2_inference", capture_build)
    trace, actual = co2_inference.sample_co2_inference(
        prepared, model, sampling.sampler,
        sigma_target_accept=sampling.sigma_target_accept,
        state_target_accept=sampling.state_target_accept,
    )
    assert len(built_models) == 1
    bound = _bind_affine_flux_map(affine, prepared, prepared_inputs_id=affine.prepared_inputs_id)
    result = make_co2_output(
        prepared=prepared, trace=trace, bound=bound, output_metadata={"sampler": actual},
        start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
    )
    output = Co2OutputOptions(
        output_format="paris", country_file=str(countries), output_path=tmp_path / "calculated-only",
    )
    selected_countries = validate_co2_product_request(output, prepared=prepared, bound=bound)
    calculate_co2_products(
        result=result, prepared=prepared, bound=bound, countries=selected_countries, output=output,
    )
    assert {"basic", "concentration_components", "native_flux", "country_flux",
            "paris_concentration", "paris_flux"} == set(result.outputs)
    assert actual["draws"] == 12
    if likelihood_kind == "fixed_ou":
        assert actual["algorithm"] == "cached_ou_sigma_then_state_nuts"
        assert actual["nuts_sampler"] == "pymc"
        assert actual["sigma_target_accept"] == 0.8
        assert actual["state_target_accept"] == 0.9
    else:
        assert actual["algorithm"] == "nuts"
        assert actual["nuts_sampler"] == "numpyro"
        assert actual["target_accept_source"] == "pm.sample_default"
        assert "target_accept" not in actual
    assert json.loads(trace.attrs["rhime_actual_inference"]) == actual
    assert not hasattr(result, "output_spec")
    assert not hasattr(result, "source_to_sector")
    assert not hasattr(result, "model")
    assert not hasattr(result, "run_spec")
    assert not hasattr(result, "model_spec")
    assert result.start_date == START_DATE
    assert result.end_date == END_DATE
    assert str(prepared.inv_inputs.time.min().values)[:10] == "2020-01-02"
    assert str(prepared.inv_inputs.time.max().values)[:10] == "2020-01-04"
    assert not (tmp_path / "calculated-only").exists()
    xr.testing.assert_identical(prepared.inv_inputs, prepared_before)

    roles = json.loads(trace.attrs["rhime_variable_roles"])
    for group, count in (("prior", 8), ("posterior", 12)):
        draws = trace[group]
        assert draws[roles["model_mean"]].attrs["units"] == "ppm"
        assert draws[roles["flux_scale"]].attrs["units"] == "1"
        assert draws.indexes["nmeasure"].equals(prepared.inv_inputs.indexes["nmeasure"])
        assert draws.indexes["region"].equals(prepared.inv_inputs.indexes["region"])
        predictive = trace[f"{group}_predictive"][roles["concentration"]]
        assert predictive.sizes["draw"] == count
        assert predictive.indexes["nmeasure"].equals(prepared.inv_inputs.indexes["nmeasure"])
        assert np.isfinite(predictive).all()
    posterior = trace.posterior.to_dataset()
    state = posterior[roles["flux_scale"]]
    pollution = xr.dot(prepared.inv_inputs.H, state, dim="region") + prepared.inv_inputs.fixed_prior_contribution
    np.testing.assert_allclose(
        posterior[roles["pollution_concentration"]],
        pollution.transpose(*posterior[roles["pollution_concentration"]].dims), rtol=2e-6,
    )
    mean = pollution + posterior[roles["boundary_concentration"]] + posterior[roles["offset_concentration"]]
    np.testing.assert_allclose(
        posterior[roles["model_mean"]], mean.transpose(*posterior[roles["model_mean"]].dims), rtol=2e-6,
    )
    # Independent bucket lift: the first column belongs to state 0, the second to state 1.
    shift = (state - prepared.inv_inputs.alpha_prior_mean).transpose("chain", "draw", "region").values
    native_draws = affine.affine_map.flux.values * (
        affine.affine_map.native_mean.values + shift[:, :, None, :]
    )
    native = result.outputs["native_flux"]
    np.testing.assert_allclose(native.flux_total_posterior_mean.squeeze(), native_draws.mean((0, 1)), rtol=2e-6)
    np.testing.assert_allclose(native.flux_total_posterior_stdev.squeeze(), native_draws.std((0, 1)), rtol=2e-6)
    assert native.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert result.outputs["concentration_components"].sizes["prior_draw"] == 8
    assert result.outputs["concentration_components"].sizes["draw"] == 12
    assert result.outputs["concentration_components"].modelled_posterior.attrs["units"] == "ppm"
    assert "active_flux_scale_posterior" in result.outputs["concentration_components"]

    built, cached = built_models[0]
    point = built.model.initial_point()
    mean_function = built.model.compile_fn(
        built.model.replace_rvs_by_values([built.model[roles["model_mean"]]])[0],
        inputs=built.model.value_vars, on_unused_input="ignore",
    )
    factor = prepared.inv_inputs.low_rank_factor.values
    covariance = factor @ factor.T + np.diag(
        prepared.inv_inputs.diagonal_residual_variance.values + prepared.inv_inputs.mf_error.values**2
    )
    if likelihood_kind == "fixed_ou":
        assert cached is not None
        amplitude_function = built.model.compile_fn(
            built.model.replace_rvs_by_values([built.model["ou_site_amplitude"]])[0],
            inputs=built.model.value_vars, on_unused_input="ignore",
        )
        cached.shared_cache.update(cached.target.refresh(amplitude_function(point)))
        times = prepared.inv_inputs.time.values
        hours = (times[:, None] - times[None, :]) / np.timedelta64(1, "h")
        correlation = np.exp(-np.abs(hours) / 24.0)
        draw_amplitude = posterior.ou_site_amplitude.isel(chain=0, draw=0).item()
        draw_mean = posterior[roles["model_mean"]].isel(chain=0, draw=0).values
        draw_reference = multivariate_normal.logpdf(
            prepared.inv_inputs.mf.values, mean=draw_mean,
            cov=covariance + draw_amplitude**2 * correlation,
        )
        assert trace.log_likelihood.y.isel(chain=0, draw=0).item() == pytest.approx(draw_reference, abs=1e-4)
        covariance += amplitude_function(point).item()**2 * correlation
    likelihood = float(built.model.compile_logp(vars=built.model.observed_RVs)(point))
    reference = multivariate_normal.logpdf(prepared.inv_inputs.mf.values, mean=mean_function(point), cov=covariance)
    assert likelihood == pytest.approx(reference, rel=1e-5, abs=1e-4)

    assert result.outputs["paris_flux"].attrs["uncertainty_scope"] == "retained_state_conditional"
    requested_bounds = (
        np.asarray([[START_DATE, END_DATE]], dtype="datetime64[D]") - np.datetime64("1970-01-01")
    ) / np.timedelta64(1, "D")
    np.testing.assert_allclose(result.outputs["paris_flux"].time_bnds, requested_bounds)


@pytest.mark.parametrize("route", ["full", "saved"])
@pytest.mark.parametrize("likelihood_kind", ["additive_sigma", "fixed_ou"])
def test_co2_execution_adapters_forward_inference_and_preserve_trace(tmp_path, monkeypatch, route, likelihood_kind):
    """Adapters forward resolved inputs; saved output retains the supplied trace exactly."""
    from openghg_inversions.rhime.co2 import execution, handoffs
    from test_co2_handoffs import _components_fixture

    prepared, trace = _components_fixture()
    for name in tuple(trace.children):
        dataset = trace[name].to_dataset()
        if name.startswith("prior"):
            dataset = dataset.assign_coords(chain=[10], draw=[100, 101, 102, 103])
        else:
            reversed_draws = dataset.isel(draw=slice(None, None, -1)).assign_coords(draw=dataset.draw)
            dataset = xr.concat([dataset, reversed_draws], dim="chain").assign_coords(
                chain=[20, 30], draw=[7, 8, 9],
            )
        trace[name] = dataset
    params = {
        "prepared_inputs": {"start_date": START_DATE, "end_date": END_DATE, "domain": "EUROPE"},
        "likelihood": {"kind": likelihood_kind},
        "sampling": {"draws": 3, "tune": 2, "chains": 2, "random_seed": 17},
        "outputs": {"output_format": "none", "output_path": str(tmp_path / "unopened")},
    }
    if likelihood_kind == "fixed_ou":
        params["likelihood"].update(
            tau_hours=24.0, site_amplitude_prior={"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
        )
        params["sampling"].update(sigma_target_accept=0.85, state_target_accept=0.92)
        targets = {"sigma_target_accept": 0.85, "state_target_accept": 0.92}
        sample_kwargs = {"random_seed": 17}
    else:
        params["likelihood"]["no_model_error"] = True
        params["sampling"]["target_accept"] = 0.93
        targets = {"sigma_target_accept": 0.8, "state_target_accept": 0.9}
        sample_kwargs = {"random_seed": 17, "target_accept": 0.93}
    model = resolve_co2_model(params)
    actual = {"draws": 3, "chains": 2, "algorithm": "adapter-test"}
    trace.attrs["rhime_actual_inference"] = json.dumps(actual)
    calls = []

    def sample(incoming, choices, sampler, **kwargs):
        calls.append(incoming)
        xr.testing.assert_identical(incoming.inv_inputs, prepared.inv_inputs)
        assert choices == model
        assert sampler.draws == 3 and sampler.tune == 2 and sampler.chains == 2
        assert sampler.sample_kwargs == sample_kwargs
        assert sampler.nuts_sampler == ("pymc" if likelihood_kind == "fixed_ou" else "numpyro")
        assert kwargs == targets
        return trace, actual

    def forbidden(*args, **kwargs):
        pytest.fail("Execution adapters must use their declared shared inference or saved trace")

    monkeypatch.setattr(co2_inference, "build_co2_inference", forbidden)
    if route == "full":
        monkeypatch.setattr(execution, "sample_co2_inference", sample)
        for name in ("load_preparation", "save_preparation", "sample", "save_samples"):
            monkeypatch.setattr(handoffs, name, forbidden)
        result = execution.run_co2(params=params, prepared=prepared)
        assert len(calls) == 1 and calls[0] is prepared
        assert result.idata is trace
        assert result.output_metadata["sampler"] == actual
        assert not list(tmp_path.glob("**/*manifest*"))
    else:
        preparation = handoffs.save_preparation(
            prepared, output_dir=tmp_path / "prepared", **params["prepared_inputs"],
        )
        before = preparation.read_bytes()
        monkeypatch.setattr(handoffs, "sample_co2_inference", sample)
        saved = handoffs.sample(
            preparation, model=params, sampling=params["sampling"], output_dir=tmp_path / "samples",
        )
        record = json.loads(saved.read_text())
        assert record["actual_inference"] == actual
        assert record["preparation_generation"] == json.loads(before)["generation"]
        assert not {"artifact_identities", "input_identities"} & record.keys()
        result = handoffs.load_samples(saved)
        assert len(calls) == 1
        xr.testing.assert_identical(result.idata, trace)
        assert result.output_metadata["sampler"] == actual
        assert preparation.read_bytes() == before
    assert result.start_date == START_DATE
    assert result.end_date == END_DATE
    assert not (tmp_path / "unopened").exists()


def test_configured_full_uses_output_policy_before_sampling(tmp_path, monkeypatch):
    """Unsupported output choices fail before expensive scientific execution."""
    from openghg_inversions.rhime.co2 import run_co2

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "unsupported", countries)
    params["outputs"]["output_format"] = "inv_out"

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported output must fail before model construction")

    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(co2_inference, "build_co2_from_materialized", forbidden)
    with pytest.raises(ValueError, match="support"):
        run_co2(params=params, prepared=prepared, affine_artifact=affine)
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("country_output", [False, True], ids=["native", "native-country"])
def test_affine_basic_requires_prior_draws_before_materialization(tmp_path, monkeypatch, country_output):
    """Conditional native summaries require both prior and posterior states."""
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "rejected", countries)
    params["sampling"]["sample_prior_predictive"] = False
    if not country_output:
        params["outputs"].pop("country_file")

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing affine prior draws must fail before materialization or sampling")

    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(co2_inference, "sample_rhime_model", forbidden)
    with pytest.raises(ValueError, match="prior draws|prior.*flux_scale|sample_prior_predictive"):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)
    assert not (tmp_path / "rejected").exists()


def test_configured_concentration_only_basic_accepts_no_prior_draws(tmp_path, monkeypatch):
    """The affine preflight does not require prior draws for concentration summaries."""
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, _, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "concentrations", countries)
    params["model"] = {}
    params["outputs"].pop("country_file")
    params["sampling"].update(sample_prior_predictive=False, sample_posterior_predictive=False)
    state = xr.DataArray(
        prepared.inv_inputs.alpha_prior_mean.values[None, None, :] * np.asarray([.9, 1.1])[None, :, None],
        dims=("chain", "draw", "region"),
        coords={"chain": [0], "draw": [0, 1], "region": prepared.inv_inputs.region},
    )
    mean = (xr.dot(prepared.inv_inputs.H, state, dim="region") + prepared.inv_inputs.fixed_prior_contribution).transpose(
        "chain", "draw", "nmeasure",
    )
    marginal_error = np.sqrt(
        prepared.inv_inputs.mf_error**2 + prepared.inv_inputs.diagonal_residual_variance
        + (prepared.inv_inputs.low_rank_factor**2).sum("agg_rank")
    )
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset({
        "flux_scaling": state, "co2_flux_contribution": mean,
        "modelled_concentration": mean, "epsilon": xr.ones_like(mean) * marginal_error,
    })})
    sampled = []

    def reference_draws(built, sampler):
        assert not sampler.sample_prior_predictive
        sampled.append(built)
        return trace

    monkeypatch.setattr(co2_inference, "sample_rhime_model", reference_draws)
    result = execution.run_co2(params=params, prepared=prepared)
    assert len(sampled) == 1
    assert result.idata is trace
    assert "prior" not in result.idata.children
    assert "native_flux" not in result.outputs
    xr.testing.assert_allclose(result.outputs["basic"].modelled_posterior_mean, mean.mean(("chain", "draw")))
    assert Path(result.output_metadata["basic_path"]).is_file()


def test_full_rejects_supplied_preparation_with_an_unrelated_saved_affine_pair(tmp_path, monkeypatch):
    """The sampled preparation must own the reconstruction, even with output off."""
    from openghg_inversions.rhime.co2 import execution
    from openghg_inversions.rhime.co2.co2_preparation import Co2PreparedInputs
    from openghg_inversions.inversion_data import RhimePreparedInputs

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    source = tmp_path / "source.nc"
    companion = tmp_path / "source-affine.nc"
    prepared.save(source)
    save_affine(replace(affine, prepared_inputs_id=prepared_inputs_content_id(source)), companion)
    changed = prepared.inv_inputs.assign(alpha_prior_mean=prepared.inv_inputs.alpha_prior_mean * 1.1)
    changed.alpha_prior_mean.attrs = dict(prepared.inv_inputs.alpha_prior_mean.attrs)
    supplied = Co2PreparedInputs(
        RhimePreparedInputs(changed, prepared.basis_functions, prepared.site_metadata),
        prepared.aggregation_error_mode, prepared.provenance,
    )
    params = configuration(tmp_path / "rejected", countries)
    params["prepared_inputs"]["path"] = str(source)
    params["outputs"]["output_format"] = "none"
    params["prepared_inputs"]["reconstruction_path"] = str(companion)

    def forbidden(*args, **kwargs):
        raise AssertionError("A conflicting prepared/affine pair must fail before inference")

    monkeypatch.setattr(co2_inference, "sample_rhime_model", forbidden)
    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="prepar|affine|source"):
        execution.run_co2(params=params, prepared=supplied)
    assert not (tmp_path / "rejected").exists()


def test_full_rejects_bucket_affine_flux_mismatch_before_inference(tmp_path, monkeypatch):
    """In-memory full execution applies the same bucket flux check as saved pairs."""
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    changed_flux = (affine.affine_map.flux * 2).assign_attrs(affine.affine_map.flux.attrs)
    altered = replace(affine, affine_map=replace(affine.affine_map, flux=changed_flux))
    params = configuration(tmp_path / "rejected", countries)
    params["outputs"]["output_format"] = "none"

    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible bucket flux must fail before inference")

    monkeypatch.setattr(co2_inference, "sample_rhime_model", forbidden)
    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="flux"):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=altered)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("invalid", ["observation-dimension", "state-dimension", "state-units"])
def test_no_output_replay_rejects_inconsistent_scientific_role_meaning(tmp_path, invalid):
    """Disabled products cannot bypass saved scientific dimensions or units."""
    from openghg_inversions.rhime.co2 import handoffs
    from test_co2_staged_outputs import _components_fixture

    prepared, trace = _components_fixture()
    for group in ("prior", "posterior"):
        for name in ("mean_custom", "emission_custom", "bc_custom", "offset_custom", "sd_custom"):
            trace[group][name].attrs["units"] = "ppm"
    if invalid == "observation-dimension":
        trace.posterior["mean_custom"] = trace.posterior.mean_custom.isel(nmeasure=0, drop=True)
    elif invalid == "state-dimension":
        trace.posterior["state_custom"] = trace.posterior.state_custom.isel(region=0, drop=True)
    else:
        trace.posterior.state_custom.attrs["units"] = "ppm"
    prep = handoffs.save_preparation(
        prepared, output_dir=tmp_path / "prep", start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
    )
    # A reader owns external trace validation; live PyMC traces need no repeated check.
    with pytest.raises(ValueError, match="role|dimension|units|flux_scale"):
        sample = handoffs.save_samples(
            prep, trace, actual_inference={"draws": 3},
            preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample",
        )
        handoffs.load_samples(sample)
