"""Real coherent CO2 full/staged/replay products and independent equations."""
from openghg_inversions.rhime.co2 import inference as co2_inference

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from examples.co2_handoffs import END_DATE, START_DATE, configuration, synthetic_inputs
from openghg_inversions.basis.affine_flux_map_io import save as save_affine
from openghg_inversions.rhime.co2 import co2_runner
from openghg_inversions.rhime.co2.co2_affine_output import prepared_inputs_content_id
from openghg_inversions.rhime.co2.co2_outputs import (
    Co2OutputOptions, calculate_co2_products, make_co2_output, validate_co2_product_request,
)
from openghg_inversions.rhime.co2.configuration import resolve_co2_model


def test_scientific_view_and_products_preserve_complete_trace(monkeypatch):
    """Sample groups keep separate axes and remain the caller's original tree."""
    from test_co2_staged_outputs import _components_fixture

    prepared, trace = _components_fixture()
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


@pytest.mark.parametrize("target_accept", [None, 0.93])
@pytest.mark.parametrize("imported", [False, True], ids=["new-pair", "imported-pair"])
def test_configured_full_saved_and_replay_preserve_coherent_products(tmp_path, monkeypatch, imported, target_accept):
    """Every product route uses retained scientific meaning, including signed flux."""
    from openghg_inversions.rhime.co2 import handoffs
    from openghg_inversions.rhime.co2 import execution
    from openghg_inversions.rhime import run

    canonical, reduction, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "full-products", countries)
    if target_accept is not None:
        params["sampling"]["target_accept"] = target_accept
    choices = {name: params[name] for name in ("model", "likelihood")}
    model = resolve_co2_model(choices)
    if imported:
        source = tmp_path / "source.nc"
        companion = tmp_path / "source-affine.nc"
        prepared.save(source)
        save_affine(replace(affine, prepared_inputs_id=prepared_inputs_content_id(source)), companion)
        manifest = handoffs.prepare(
            output_dir=tmp_path / "prepared", prepared_path=source, affine_path=companion,
            start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
        )
    else:
        manifest = handoffs.save_preparation(
            prepared, output_dir=tmp_path / "prepared", affine_artifact=affine,
            start_date=START_DATE, end_date=END_DATE, domain="EUROPE",
        )
    before = manifest.read_bytes()
    samples = handoffs.sample(manifest, model=choices, sampling=params["sampling"], output_dir=tmp_path / "samples")
    staged = handoffs.postprocess(
        samples, output=Co2OutputOptions(
            output_format="basic", output_name="co2",
            country_file=str(countries), output_path=str(tmp_path / "staged-products"),
        ),
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("Full execution and replay must not use checkpoint preparation or sampling")

    with monkeypatch.context() as full_checks:
        full_checks.setattr(handoffs, "load_preparation", forbidden)
        full_checks.setattr(handoffs, "save_preparation", forbidden)
        full_checks.setattr(handoffs, "sample", forbidden)
        full = run(
            model="co2", params=params, canonical_inputs=canonical, reduction=reduction,
            aggregation_error_rank=1, affine_artifact=affine,
        )
    for name in ("basic", "concentration_components", "native_flux", "country_flux"):
        xr.testing.assert_allclose(full.outputs[name], staged.outputs[name])
        if name != "concentration_components":
            assert Path(full.output_metadata[f"{name}_path"]).is_file()
            assert Path(staged.output_metadata[f"{name}_path"]).is_file()
    assert full.output_metadata["sampler"]["draws"] == 12
    assert full.output_metadata["sampler"] == staged.output_metadata["sampler"]
    assert full.output_metadata["sampler"]["algorithm"] == "nuts"
    assert full.output_metadata["sampler"]["nuts_sampler"] == "numpyro"
    if target_accept is None:
        assert full.output_metadata["sampler"]["target_accept_source"] == "pm.sample_default"
        assert "target_accept" not in full.output_metadata["sampler"]
    else:
        assert full.output_metadata["sampler"]["target_accept"] == target_accept
        assert "target_accept_source" not in full.output_metadata["sampler"]
    assert json.loads(full.idata.attrs["rhime_actual_inference"]) == full.output_metadata["sampler"]
    assert not hasattr(full, "output_spec")
    assert not hasattr(full, "source_to_sector")
    assert not hasattr(staged, "model")
    assert not hasattr(full, "run_spec")
    assert not hasattr(full, "model_spec")
    assert staged.start_date == full.start_date == START_DATE
    assert staged.end_date == full.end_date == END_DATE
    assert str(prepared.inv_inputs.time.min().values)[:10] == "2020-01-02"
    assert str(prepared.inv_inputs.time.max().values)[:10] == "2020-01-04"
    assert manifest.read_bytes() == before

    trace = staged.idata
    roles = json.loads(trace.attrs["rhime_variable_roles"])
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
    native = staged.outputs["native_flux"]
    np.testing.assert_allclose(native.flux_total_posterior_mean.squeeze(), native_draws.mean((0, 1)), rtol=2e-6)
    np.testing.assert_allclose(native.flux_total_posterior_stdev.squeeze(), native_draws.std((0, 1)), rtol=2e-6)
    assert native.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert staged.outputs["concentration_components"].sizes["prior_draw"] == 8
    assert staged.outputs["concentration_components"].sizes["draw"] == 12
    assert staged.outputs["concentration_components"].modelled_posterior.attrs["units"] == "ppm"

    model_inputs, priors = co2_runner.materialize_co2_inputs(prepared_inputs=prepared, **model)
    built = co2_runner.build_co2_from_materialized(model_inputs, prepared_inputs=prepared, priors=priors, **model)
    point = built.model.initial_point()
    mean_function = built.model.compile_fn(
        built.model.replace_rvs_by_values([built.model[roles["model_mean"]]])[0],
        inputs=built.model.value_vars, on_unused_input="ignore",
    )
    likelihood = float(built.model.compile_logp(vars=built.model.observed_RVs)(point))
    factor = prepared.inv_inputs.low_rank_factor.values
    covariance = factor @ factor.T + np.diag(
        prepared.inv_inputs.diagonal_residual_variance.values + prepared.inv_inputs.mf_error.values**2
    )
    reference = multivariate_normal.logpdf(prepared.inv_inputs.mf.values, mean=mean_function(point), cov=covariance)
    assert likelihood == pytest.approx(reference, rel=1e-5, abs=1e-4)

    monkeypatch.setattr(co2_runner, "build_co2_from_materialized", forbidden)
    monkeypatch.setattr(co2_runner, "build_co2_model", forbidden)
    monkeypatch.setattr(co2_runner, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", forbidden)
    monkeypatch.setattr(handoffs, "prepare_co2_inputs", forbidden)
    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(co2_inference, "build_co2_from_materialized", forbidden)
    monkeypatch.setattr(co2_inference, "sample_rhime_model", forbidden)
    monkeypatch.setattr(execution, "run_co2", forbidden)
    replay = handoffs.postprocess(
        samples, output=Co2OutputOptions(
            output_format="paris", output_name="replay",
            country_file=str(countries), output_path=str(tmp_path / "replay-products"),
        ),
    )
    assert replay.output_metadata["sampler"] == staged.output_metadata["sampler"]
    for name in ("basic", "concentration_components", "native_flux", "country_flux"):
        xr.testing.assert_identical(replay.outputs[name], staged.outputs[name])
    for name in ("paris_concentration", "paris_flux"):
        assert Path(replay.output_metadata[f"{name}_path"]).is_file()
    assert replay.outputs["paris_flux"].attrs["uncertainty_scope"] == "retained_state_conditional"
    requested_bounds = (
        np.asarray([[START_DATE, END_DATE]], dtype="datetime64[D]") - np.datetime64("1970-01-01")
    ) / np.timedelta64(1, "D")
    np.testing.assert_allclose(replay.outputs["paris_flux"].time_bnds, requested_bounds)


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
