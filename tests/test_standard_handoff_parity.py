"""Scientific acceptance and graph-free replay of standard preparations."""

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
    logp = built.model.compile_logp()
    for x in (.7, 1., 1.4):
        reference = norm.logpdf(x, 1.2, .3) + norm.logpdf(y, h * x, .1).sum()
        np.testing.assert_allclose(logp({"x": np.array([x])}), reference, rtol=1e-6)


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
    from dataclasses import replace
    import json
    import dask
    import numpy as np
    import xarray as xr
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.basis import BasisFunctions
    from openghg_inversions.postprocessing.contracts import OutputContract
    from openghg_inversions.rhime import DataConfig, RhimeOutputSpec, RhimeSampler, _standard_stages as stages, standard
    from openghg_inversions.rhime.params import resolve_model_options
    from openghg_inversions.rhime.preparation import resolve_basis_inputs
    from tests.helpers import make_trace

    merged, options, _, _, h, y = synthetic_inputs(tmp_path, multiple_sources=True)
    site = merged.site_data["TAC"]
    original = site.copy(deep=True)
    graph = site.fp_x_flux.data
    original_flux = {source: field.copy(deep=True) for source, field in merged.flux_data.items()}
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

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation or replay performed acquisition, basis export or inference")

    monkeypatch.setattr(standard, "resolve_runtime_flux", resolve_once)
    monkeypatch.setattr(standard, "prepare_basis", check_first_weight)
    monkeypatch.setattr(standard.RhimeMergedData, "from_options", forbidden)
    monkeypatch.setattr(BasisFunctions, "save", forbidden)
    monkeypatch.setattr(RhimeSampler, "sample", forbidden)
    manifest = stages.prepare(DataConfig.from_params(options), merged_data=merged, output_dir=tmp_path / "prepared")
    prepared, data, generation = stages.load_preparation(manifest)
    np.testing.assert_allclose(prepared.inv_inputs.H, h[None, :])
    np.testing.assert_allclose(prepared.inv_inputs.mf, y)
    np.testing.assert_allclose(prepared.inv_inputs.mf_error, 0.1)
    expected = xr.DataArray([[1.3, 3.2], [5.1, 4.4]], dims=("lat", "lon"),
                            coords={"lat": [50.0, 51.0], "lon": [-2.0, -1.0]})
    xr.testing.assert_allclose(prepared.basis_functions.flux, expected)

    # Scientific callers may change source order without changing the sum;
    # only the first requested source supplies basis-design weighting.
    reversed_merged = replace(merged, flux_data=dict(reversed(tuple(merged.flux_data.items()))))
    reversed_prepared = standard.prepare_standard(
        reversed_merged, basis_inputs=resolve_basis_inputs(data),
        **{**data.select("domain", "start_date", "flux_sources", "averaging_error", "filters",
                        "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                        "min_error", "min_error_options"),
           "flux_sources": ("supplemental", "inventory")},
    )
    assert resolutions == [("inventory", "supplemental"), ("supplemental", "inventory")]
    assert len(weights) == 2
    np.testing.assert_array_equal(weights[0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(weights[1], [[0.3, 1.2], [2.1, 0.4]])
    np.testing.assert_allclose(reversed_prepared.inv_inputs.H, h[None, :])
    np.testing.assert_allclose(reversed_prepared.inv_inputs.mf_error, 0.1)
    xr.testing.assert_allclose(reversed_prepared.basis_functions.flux, expected)
    xr.testing.assert_equal(site, original)
    assert site.fp_x_flux.data is graph and dask.is_dask_collection(graph)
    assert "mf_error" not in site
    for source, field in merged.flux_data.items():
        xr.testing.assert_equal(field, original_flux[source])

    scales = np.array([0.5, 1.25, 2.0])
    trace = make_trace(posterior=xr.Dataset(
        {"x": (("chain", "draw", "state"), scales[None, :, None])},
        coords={"chain": [3], "draw": [2, 4, 8], "state": prepared.inv_inputs.state},
    ))
    model = resolve_model_options({"mismatch_model": "fixed_error",
                                   "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}}, data_config=data)
    trace.attrs["rhime_priors"] = json.dumps({"x": model.sectors[0].x_prior})
    sampled = stages.save_samples(
        manifest, trace, model=model, output_contract=OutputContract(
            variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "inv_out"),
        ), actual_inference={"draws": 3, "chains": 1}, preparation_generation=generation,
        output_dir=tmp_path / "samples",
    )
    (tmp_path / "operator.nc").unlink()
    monkeypatch.setattr(standard, "build_standard_model", forbidden)
    monkeypatch.setattr(standard, "materialize_standard_inputs", forbidden)
    monkeypatch.setattr(stages, "build_standard_model", forbidden)
    monkeypatch.setattr(stages, "materialize_standard_inputs", forbidden)
    monkeypatch.setattr(stages, "sample_rhime_model", forbidden)
    monkeypatch.setattr(stages, "resolve_sampler", forbidden)
    result = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    assert result.run_metadata["start_date"] == "2019-01-01"
    assert result.run_metadata["end_date"] == "2019-02-01"
    xr.testing.assert_identical(result.trace["posterior"], trace["posterior"])
    xr.testing.assert_allclose(result.basis_functions.flux, expected)
    reconstructed = result.basis_functions.state_to_flux(result.trace["posterior"]["x"]).squeeze("chain", drop=True)
    expected_flux = xr.DataArray(
        scales[:, None, None] * expected.values,
        dims=("draw", "lat", "lon"), coords={"draw": [2, 4, 8], **expected.coords},
    )
    xr.testing.assert_allclose(reconstructed, expected_flux.transpose(*reconstructed.dims))
    assert len(resolutions) == 2  # replay did not resolve native flux again


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
    from dataclasses import replace
    import numpy as np
    from scipy.stats import norm, t
    import xarray as xr

    from examples.rhime_customisation.likelihoods import likelihood_builder
    from examples.standard_handoffs import synthetic_inputs
    from openghg_inversions.rhime import DataConfig, RhimeConfig, RhimeOutputSpec, RhimeSampler, run_standard, stages
    from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs

    merged, options, _, _, h, y = synthetic_inputs(tmp_path)
    model = {"mismatch_model": None, "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.2}}
    sampler = RhimeSampler(draws=10, tune=10, chains=1, sample_kwargs={"cores": 1, "random_seed": 42})
    manifest = stages.prepare(
        DataConfig.from_params(options), merged_data=merged, output_dir=tmp_path / "prepared",
    )
    prepared, _, generation = stages.load_preparation(manifest)
    config = RhimeConfig.from_params(
        {**options, **model, "output_name": "rhime", "output_format": "none"}, multisector=False,
    )
    config = replace(config, sampler=sampler)
    model_inputs, priors = materialize_standard_inputs(prepared, config.model)
    built = build_standard_model(
        model_inputs, config.model, priors=priors, likelihood_builder=likelihood_builder,
    )
    logp = built.model.compile_logp()
    for state in (0.7, 1.0, 1.4):
        reference = norm.logpdf(state, 1.0, 0.2) + t.logpdf(y, df=4.0, loc=h * state, scale=0.1).sum()
        np.testing.assert_allclose(logp({"x": np.array([state])}), reference, rtol=1e-6, atol=1e-5)

    sampled = stages.sample(
        manifest, model=model, sampler=sampler, output_dir=tmp_path / "samples",
        likelihood_builder=likelihood_builder,
    )
    replay = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))
    full = run_standard(config=config, merged_data=merged, likelihood_builder=likelihood_builder)
    xr.testing.assert_allclose(
        replay.trace["posterior"]["x"], full.trace["posterior"]["x"], rtol=1e-7, atol=0,
    )
    np.testing.assert_array_equal(replay.trace["posterior"]["state"], prepared.inv_inputs["state"])
    assert full.trace["posterior"].sizes["draw"] == 10
    assert replay.trace.attrs["rhime_likelihood_builder"] == full.trace.attrs["rhime_likelihood_builder"]
    assert replay.model_metadata["builder"] == full.model_metadata["builder"]

    samples = stages.save_samples(
        manifest, full.trace, output_contract=built.output_contract, model=config.model,
        actual_inference=full.output_metadata["sampler"], output_dir=tmp_path / "external-samples",
        preparation_generation=generation,
    )
    external = stages.postprocess(
        samples, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False),
    )
    xr.testing.assert_equal(external.trace["posterior"]["x"], full.trace["posterior"]["x"])
    assert external.trace.attrs["rhime_likelihood_builder"] == full.trace.attrs["rhime_likelihood_builder"]


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
