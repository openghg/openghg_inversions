"""Executable, non-mocked scientific acceptance of the standard saved route."""

from examples.standard_handoffs import run_example


def test_one_preparation_two_models_two_outputs_match_full_execution(tmp_path):
    run_example(tmp_path)


def test_lazy_array_prior_uses_same_full_prepared_and_saved_construction(tmp_path):
    from dataclasses import replace
    import json
    import dask
    import dask.array as da
    import numpy as np
    import xarray as xr

    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.inference.sampling import RhimeSampler
    from openghg_inversions.models.priors import trace_prior_records
    from openghg_inversions.rhime import DataConfig, RhimeConfig, RhimeOutputSpec, run_standard, stages
    from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs
    from openghg_inversions.rhime.params import resolve_model_options

    merged, options, *_ = synthetic_inputs(tmp_path)
    preparation = stages.prepare(DataConfig.from_params(options), merged_data=merged,
                                 output_dir=tmp_path / "prepared")
    calls = []

    @dask.delayed
    def mean():
        calls.append(1)
        return np.array([1.2])

    prepared, data, _ = stages.load_preparation(preparation)
    state = prepared.inv_inputs["H"]["state"]
    mu = xr.DataArray(da.from_delayed(mean(), shape=(1,), dtype=float), dims="state", coords=state.coords)
    prior = {"pdf": "normal", "mu": mu, "sigma": .3}
    sampler = RhimeSampler(draws=4, tune=4, chains=1, sample_prior_predictive=False,
                           sample_posterior_predictive=False, sample_kwargs={"cores": 1, "random_seed": 42})
    config = RhimeConfig.from_params({**options, "x_prior": prior, "mismatch_model": "fixed_error", "output_format": "none",
                                     "output_name": "array-prior"}, multisector=False)
    full = run_standard(config=replace(config, sampler=sampler), merged_data=merged)
    assert calls == [1]
    manifest = stages.sample(preparation, model={"x_prior": prior, "mismatch_model": "fixed_error"},
                             sampler=sampler, output_dir=tmp_path / "samples")
    assert calls == [1, 1]
    metadata = json.loads(manifest.read_text())["model"]["sectors"][0]["x_prior"]
    assert metadata == {"pdf": "normal", "mu": {"data": "x_prior_mu"}, "sigma": .3}
    replay = stages.postprocess(manifest, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    assert calls == [1, 1]
    np.testing.assert_allclose(replay.trace["constant_data"]["x_prior_mu"], [1.2])
    assert trace_prior_records(full.trace) == trace_prior_records(replay.trace)

    prepared, data, _ = stages.load_preparation(preparation)
    model = resolve_model_options({"x_prior": prior, "mismatch_model": "fixed_error"}, data_config=data)
    model_inputs, priors = materialize_standard_inputs(prepared, model)
    built = build_standard_model(model_inputs, model, priors=priors)
    assert calls == [1, 1, 1]
    from scipy.stats import norm
    h = prepared.inv_inputs.H.values.squeeze()
    y = prepared.inv_inputs.mf.values
    np.testing.assert_allclose(full.trace["posterior"]["x"], replay.trace["posterior"]["x"])
    for x in (.7, 1., 1.4):
        reference = norm.logpdf(x, 1.2, .3) + norm.logpdf(y, h * x, .1).sum()
        np.testing.assert_allclose(built.model.compile_logp()({"x": np.array([x])}), reference, rtol=1e-6)


def test_cookiecutter_project_runs_real_student_t_recipe(tmp_path):
    import numpy as np
    from scipy.stats import norm, t

    from examples.rhime_cookiecutter.my_inversion.runner import run
    from examples.standard_handoffs import synthetic_inputs

    merged, options, _, _, h, y = synthetic_inputs(tmp_path)
    result = run(**options, merged_data=merged, output_name="project", output_format="none",
                 x_prior={"pdf": "normal", "mu": 1., "sigma": .2},
                 draws=10, tune=10, chains=1, sample_kwargs={"random_seed": 42})
    from examples.rhime_cookiecutter.my_inversion.likelihoods import likelihood_builder
    from openghg_inversions.rhime import stages, DataConfig, RhimeConfig, RhimeOutputSpec
    from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs

    manifest = stages.prepare(DataConfig.from_params(options), merged_data=merged,
                              output_dir=tmp_path / "custom-prepared")
    prepared, _, _ = stages.load_preparation(manifest)
    config = RhimeConfig.from_params({**options, "output_name": "project", "output_format": "none",
                                     "mismatch_model": None, "x_prior": {"pdf": "normal", "mu": 1., "sigma": .2}},
                                    multisector=False)
    model_inputs, priors = materialize_standard_inputs(prepared, config.model)
    built = build_standard_model(model_inputs, config.model, priors=priors, likelihood_builder=likelihood_builder)
    logp = built.model.compile_logp()
    for state in (.7, 1., 1.4):
        reference = norm.logpdf(state, 1., .2) + t.logpdf(y, df=4., loc=h * state, scale=.1).sum()
        np.testing.assert_allclose(logp({"x": np.array([state])}), reference, rtol=1e-6, atol=1e-5)
    assert result.trace["posterior"].sizes["draw"] == 10
    samples = stages.save_samples(
        manifest, result.trace, output_contract=built.output_contract, model=config.model,
        actual_inference=result.output_metadata["sampler"],
        output_dir=tmp_path / "custom-samples",
        preparation_generation=stages.load_preparation(manifest)[2],
    )
    replay = stages.postprocess(samples, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    np.testing.assert_array_equal(replay.trace["posterior"]["x"], result.trace["posterior"]["x"])


def test_saved_preparation_retains_subset_and_borrows_lazy_inputs(tmp_path):
    from dataclasses import replace
    import dask.array as da
    import xarray as xr

    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.inversion_data import SiteOptions
    from openghg_inversions.rhime.params import DataConfig
    from openghg_inversions.rhime.stages import prepare, load_preparation

    merged, options, *_ = synthetic_inputs(tmp_path)
    borrowed = merged.site_data["TAC"]
    graph = borrowed.fp_x_flux.data
    selectors = SiteOptions.from_inputs(sites=["MHD", "TAC"], averaging_period=["3h", "1h"])
    merged = replace(merged, site_options=selectors,
                     site_data={"MHD": borrowed.isel(time=slice(0, 0)), "TAC": borrowed})
    configuration = DataConfig.from_params({**options, "sites": ["MHD", "TAC"],
                                                   "averaging_period": ["3h", "1h"]})
    manifest = prepare(configuration, merged_data=merged, output_dir=tmp_path / "prepared")
    reopened, data, _ = load_preparation(manifest)
    assert reopened.sites == ("TAC",)
    assert reopened.averaging_period == ("1h",)
    assert data.start_date == "2019-01-01" and data.end_date == "2019-02-01"
    assert data.site_options.sites == ("MHD", "TAC")  # the request, not the retained sites
    assert merged.sites == ("MHD", "TAC")
    assert borrowed.fp_x_flux.data is graph and isinstance(graph, da.Array)
    assert "mf_error" not in borrowed
    xr.testing.assert_equal(reopened.inv_inputs.mf.unstack("nmeasure").sel(site="TAC", drop=True), borrowed.mf)


def test_saved_model_missing_likelihood_input_fails_before_inference(tmp_path, monkeypatch):
    from dataclasses import replace
    import pytest
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import _standard_stages as stages
    from openghg_inversions.rhime.params import DataConfig

    merged, options, *_ = synthetic_inputs(tmp_path)
    complete = stages.prepare(DataConfig.from_params(options), merged_data=merged, output_dir=tmp_path / "complete")
    prepared, data, _ = stages.load_preparation(complete)
    prepared = replace(prepared, inv_inputs=prepared.inv_inputs.drop_vars("min_error"))
    manifest = stages.save_preparation(prepared, data_config=data, output_dir=tmp_path / "prepared")
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *args, **kwargs: pytest.fail("inference"))
    with pytest.raises(ValueError, match="Selected built-in likelihood.*min_error"):
        stages.sample(manifest, model={"mismatch_model": "pollution_event"}, output_dir=tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


def test_default_log_normal_state_is_retained_and_explicit_omission_is_rejected(tmp_path):
    import pytest
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import stages, DataConfig, RhimeSampler

    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    # Default reparameterised prior exposes scientific x separately from x_latent.
    sampler = RhimeSampler(draws=10, tune=10, chains=1, sample_kwargs={"random_seed": 42})
    sampled = stages.sample(manifest, model={"mismatch_model": "fixed_error"},
                            sampler=sampler, output_dir=tmp_path / "samples")
    from openghg_inversions.rhime import RhimeOutputSpec
    result = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    assert "x" in result.trace["posterior"]
    assert "x_latent" in result.trace["posterior"]
    with pytest.raises(ValueError, match="var_names"):
        RhimeSampler(sample_kwargs={"var_names": ["x_latent", "mu"]})


def test_multiple_native_sources_are_retained_once_and_replayed_without_original_operator(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr
    from openghg_inversions.rhime import standard, stages, RhimeOutputSpec

    resolve = standard.resolve_runtime_flux
    prepare_basis = standard.prepare_basis
    resolutions = []
    weights = []
    def resolve_once(sources, **kwargs):
        resolutions.append(tuple(sources))
        return resolve(sources, **kwargs)
    def check_first_weight(**kwargs):
        weights.append(kwargs["weighting_flux"])
        return prepare_basis(**kwargs)
    monkeypatch.setattr(standard, "resolve_runtime_flux", resolve_once)
    monkeypatch.setattr(standard, "prepare_basis", check_first_weight)
    run_example(tmp_path, multiple_sources=True)
    # One shared preparation plus the two independent full-execution references.
    assert resolutions == [("inventory", "supplemental")] * 3
    assert len(weights) == 3
    for weight in weights:
        np.testing.assert_array_equal(weight, [[1., 2.], [3., 4.]])
    # Scientific callers may change source order without changing the sum;
    # only the first requested source supplies basis-design weighting.
    from dataclasses import replace
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import DataConfig
    from openghg_inversions.rhime.preparation import resolve_basis_inputs
    merged, options, *_ = synthetic_inputs(tmp_path / "reverse", multiple_sources=True)
    merged = replace(merged, flux_data=dict(reversed(tuple(merged.flux_data.items()))))
    data = DataConfig.from_params(options)
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific preparation performed acquisition or export")
    from openghg_inversions.basis import BasisFunctions
    monkeypatch.setattr(standard.RhimeMergedData, "from_options", forbidden)
    monkeypatch.setattr(BasisFunctions, "save", forbidden)
    prepared = standard.prepare_standard(
        merged, basis_inputs=resolve_basis_inputs(data),
        **{**data.select("domain", "start_date", "flux_sources", "averaging_error", "filters",
                        "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                        "min_error", "min_error_options"),
           "flux_sources": ("supplemental", "inventory")},
    )
    assert resolutions[-1] == ("supplemental", "inventory")
    np.testing.assert_array_equal(weights[-1], [[.3, 1.2], [2.1, .4]])
    np.testing.assert_allclose(prepared.basis_functions.flux, [[1.3, 3.2], [5.1, 4.4]])
    (tmp_path / "operator.nc").unlink()
    result = stages.postprocess(tmp_path / "sample-1/sample-manifest.json",
                                output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    expected = xr.DataArray([[1.3, 3.2], [5.1, 4.4]], dims=("lat", "lon"),
                            coords={"lat": [50., 51.], "lon": [-2., -1.]})
    xr.testing.assert_allclose(result.basis_functions.flux, expected)
    assert len(resolutions) == 4  # replay did not resolve native flux again


def test_scientific_offset_is_required_but_predictive_samples_are_optional(tmp_path):
    import json
    import pytest
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import stages, DataConfig, RhimeOutputSpec, RhimeSampler
    from openghg_inversions.serialization import save_trace

    merged, options, *_ = synthetic_inputs(tmp_path)
    prepared = stages.prepare(DataConfig.from_params(options), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    model = {"mismatch_model": "fixed_error", "add_offset": True,
             "x_prior": {"pdf": "normal", "mu": 1., "sigma": .2}}
    sampler = RhimeSampler(draws=10, tune=10, chains=1,
                           sample_kwargs={"random_seed": 42},
                           sample_prior_predictive=False, sample_posterior_predictive=False)
    sampled = stages.sample(prepared, model=model, sampler=sampler, output_dir=tmp_path / "complete")
    output = RhimeOutputSpec(output_format="inv_out", save_inversion_output=False)
    replay = stages.postprocess(sampled, output=output)
    assert "offset" in replay.trace["posterior"]
    assert "posterior_predictive" not in replay.trace.children
    assert "prior_predictive" not in replay.trace.children
    # A missing offset must also fail when loading a previously saved handoff.
    record = json.loads(sampled.read_text())
    replay.trace["posterior"] = replay.trace["posterior"].to_dataset().drop_vars("offset")
    save_trace(replay.trace, sampled.parent / record["posterior"])
    with pytest.raises(ValueError, match="reconstruction state.*offset"):
        stages.postprocess(sampled, output=output)


def test_custom_likelihood_saved_sampling_matches_full_route(tmp_path):
    import numpy as np
    from examples.rhime_cookiecutter.my_inversion.likelihoods import likelihood_builder
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import DataConfig, RhimeConfig, RhimeOutputSpec, RhimeSampler, run_standard, stages

    merged, options, *_ = synthetic_inputs(tmp_path)
    prior = {"x_prior": {"pdf": "normal", "mu": 1., "sigma": .2}}
    sampler = RhimeSampler(draws=10, tune=10, chains=1, sample_kwargs={"random_seed": 42})
    manifest = stages.prepare(DataConfig.from_params(options), merged_data=merged, output_dir=tmp_path / "prepared")
    sampled = stages.sample(manifest, model=prior, sampler=sampler, output_dir=tmp_path / "samples",
                            likelihood_builder=likelihood_builder)
    replay = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))
    config = RhimeConfig.from_params({**options, **prior, "mismatch_model": None, "draws": 10, "tune": 10,
                                      "chains": 1, "sample_kwargs": {"random_seed": 42},
                                      "output_name": "rhime", "output_format": "none"}, multisector=False)
    full = run_standard(config=config, merged_data=merged, likelihood_builder=likelihood_builder)
    np.testing.assert_allclose(replay.trace["posterior"]["x"], full.trace["posterior"]["x"])
    assert "rhime_likelihood_builder" in replay.trace.attrs
    assert replay.model_metadata["builder"] == full.model_metadata["builder"]


def test_standard_model_and_enabled_priors_share_computation_before_zero_pruning(tmp_path):
    from dataclasses import replace
    import dask
    import dask.array as da
    import numpy as np
    import xarray as xr
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.models.priors import prior_records
    from openghg_inversions.rhime import DataConfig, stages
    from openghg_inversions.rhime.params import resolve_model_options
    from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs

    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    prepared, data, _ = stages.load_preparation(manifest)
    sensitivity = prepared.inv_inputs["H"]
    for all_zero in (False, True):
        calls = []

        @dask.delayed
        def shared():
            calls.append(1)
            return np.zeros(sensitivity.shape) if all_zero else sensitivity.values

        raw = da.from_delayed(shared(), shape=sensitivity.shape, dtype=float)
        h = sensitivity.copy(data=raw)
        mean = h.sum("nmeasure").assign_coords(prior_group=("state", raw.sum(axis=1).astype(int)))
        prior = {"pdf": "normal", "mu": mean, "sigma": xr.DataArray(raw.sum() + 1.)}

        @dask.delayed
        def disabled():
            raise AssertionError("disabled component prior executed")

        unused = {"pdf": "normal", "mu": da.from_delayed(disabled(), shape=(1,), dtype=float), "sigma": .1}
        model = resolve_model_options({"x_prior": prior, "bc_prior": unused, "offset_prior": unused,
                                       "mismatch_model": "fixed_error"}, data_config=data)
        borrowed = replace(prepared, inv_inputs=prepared.inv_inputs.assign(H=h))
        model_inputs, priors = materialize_standard_inputs(borrowed, model)
        built = build_standard_model(model_inputs, model, priors=priors)
        assert calls == [1]
        assert h.data is raw and dask.is_dask_collection(mean)
        assert prior["mu"] is mean
        records = prior_records(built.model)
        assert ("x" in records) is not all_zero
        assert "bc" not in records and "offset_latent" not in records
        if not all_zero:
            np.testing.assert_allclose(built.model["x_prior_mu"].get_value(), sensitivity.sum("nmeasure"))
            assert records["x"]["sigma"] == float(sensitivity.sum() + 1.)
