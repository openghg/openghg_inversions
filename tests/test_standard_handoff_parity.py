"""Executable, non-mocked scientific acceptance of the standard saved route."""

from examples.standard_handoffs import run_example


def test_one_preparation_two_models_two_outputs_match_full_execution(tmp_path):
    run_example(tmp_path)


def test_cookiecutter_project_runs_real_student_t_recipe(tmp_path):
    import numpy as np
    from scipy.stats import norm, t

    from examples.rhime_cookiecutter.my_inversion.runner import run
    from examples.standard_handoffs import synthetic_inputs

    merged, options, _, _, h, y = synthetic_inputs(tmp_path)
    result = run(**options, merged_data=merged, output_name="project", output_format="none",
                 x_prior={"pdf": "normal", "mu": 1., "sigma": .2},
                 draws=10, tune=10, chains=1, sample_kwargs={"random_seed": 42})
    logp = result.model.compile_logp()
    # The production graph uses float32; scipy evaluates the oracle in float64.
    for state in (.7, 1., 1.4):
        reference = norm.logpdf(state, 1., .2) + t.logpdf(y, df=4., loc=h * state, scale=.1).sum()
        np.testing.assert_allclose(logp({"x": np.array([state])}), reference, rtol=1e-6, atol=1e-5)
    assert result.idata["posterior"].sizes["draw"] == 10

    # Low-level saves also preserve a directly customized scientific recipe.
    from openghg_inversions.rhime import stages, PreparationConfig, RhimeOutputSpec
    from openghg_inversions.rhime.standard import prepare_standard
    preparation = PreparationConfig.from_params(options)
    prepared = prepare_standard(preparation, merged_data=merged)
    manifest = stages.save_preparation(prepared, preparation=preparation, output_dir=tmp_path / "custom-prepared")
    samples = stages.save_samples(
        manifest, result.idata, output_contract=result.output_contract, model=result.model_spec,
        actual_inference=result.output_metadata["sampler"], build_and_sample_seconds=0.,
        output_dir=tmp_path / "custom-samples",
    )
    replay = stages.postprocess(samples, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    assert replay.model is None and replay.sampler is None
    np.testing.assert_array_equal(replay.idata["posterior"]["x"], result.idata["posterior"]["x"])


def test_saved_preparation_retains_subset_and_borrows_lazy_inputs(tmp_path):
    from dataclasses import replace
    import dask.array as da
    import xarray as xr

    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.inversion_data import SiteOptions
    from openghg_inversions.rhime.params import PreparationConfig
    from openghg_inversions.rhime.stages import prepare, load_preparation

    merged, options, *_ = synthetic_inputs(tmp_path)
    borrowed = merged.site_data["TAC"]
    graph = borrowed.fp_x_flux.data
    selectors = SiteOptions.from_inputs(sites=["MHD", "TAC"], averaging_period=["3h", "1h"])
    merged = replace(merged, site_options=selectors,
                     site_data={"MHD": borrowed.isel(time=slice(0, 0)), "TAC": borrowed})
    configuration = PreparationConfig.from_params({**options, "sites": ["MHD", "TAC"],
                                                   "averaging_period": ["3h", "1h"]})
    manifest = prepare(configuration, merged_data=merged, output_dir=tmp_path / "prepared")
    reopened, context = load_preparation(manifest)
    assert reopened.sites == ("TAC",)
    assert reopened.averaging_period == ("1h",)
    assert context["start_date"] == "2019-01-01" and context["end_date"] == "2019-02-01"
    assert merged.sites == ("MHD", "TAC")
    assert borrowed.fp_x_flux.data is graph and isinstance(graph, da.Array)
    assert "mf_error" not in borrowed
    xr.testing.assert_equal(reopened.inv_inputs.mf.unstack("nmeasure").sel(site="TAC", drop=True), borrowed.mf)


def test_saved_model_missing_likelihood_input_fails_before_inference(tmp_path, monkeypatch):
    from dataclasses import replace
    import pytest
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import _standard_stages as stages
    from openghg_inversions.rhime.params import PreparationConfig
    from openghg_inversions.rhime.standard import prepare_standard

    merged, options, *_ = synthetic_inputs(tmp_path)
    preparation = PreparationConfig.from_params(options)
    prepared = prepare_standard(preparation, merged_data=merged)
    prepared = replace(prepared, inv_inputs=prepared.inv_inputs.drop_vars("min_error"))
    manifest = stages.save_preparation(prepared, preparation=preparation, output_dir=tmp_path / "prepared")
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *args, **kwargs: pytest.fail("inference"))
    with pytest.raises(ValueError, match="Selected built-in likelihood.*min_error"):
        stages.sample(manifest, model={"mismatch_model": "pollution_event"}, output_dir=tmp_path / "samples")
    assert not (tmp_path / "samples").exists()
