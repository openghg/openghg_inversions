"""Scientific acceptance of reusable multisector preparation and sample replay."""

from dataclasses import replace
import json
import subprocess
import sys

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from examples.multisector_handoffs import model_options, run_example, synthetic_inputs
from openghg_inversions.inversion_data import PreparedInputs, SiteOptions
from openghg_inversions.models.priors import prior_records
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import make_multisector_flux_trace_outputs
from openghg_inversions.rhime import multisector, multisector_stages as stages
from openghg_inversions.rhime.params import DataConfig, RhimeConfig, resolve_model_options
from openghg_inversions.rhime.preparation import resolve_basis_inputs
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeModelSpec, RhimeOutputSpec
from tests.helpers import make_trace


def test_sector_mapping_order_does_not_change_preparation_weighting(tmp_path, monkeypatch):
    """Full and phase-only requests use the declared first source for basis weights."""
    merged, options, flux, *_ = synthetic_inputs(tmp_path)
    options.pop("basis_path")
    options.update(flux_sources=["inventory", "supplemental"],
                   basis_algorithm="quadtree", nbasis=2, fix_basis_outer_regions=False)
    flux = {name: field.expand_dims(time=merged.site_data["TAC"].time) for name, field in flux.items()}
    merged = replace(merged,
        flux_data={name: field.to_dataset(name="flux") for name, field in flux.items()},
        site_data={
            name: site.assign(fp=xr.ones_like(site.fp_x_flux_sectoral.isel(source=0, drop=True)))
            for name, site in merged.site_data.items()
        },
    )
    config = RhimeConfig.from_params(
        {**options, **model_options(.2), "output_name": "weights", "output_format": "none"},
        multisector=True,
    )
    phase_data = DataConfig.from_params(options, multisector=True)
    assert config.model.sectors[0].flux_source == "supplemental"
    assert config.data.flux_sources == phase_data.flux_sources == ("inventory", "supplemental")
    weights = []
    prepare_basis = multisector.prepare_basis

    def capture_weights(**kwargs):
        weights.append(kwargs["weighting_flux"])
        return prepare_basis(**kwargs)

    monkeypatch.setattr(multisector, "prepare_basis", capture_weights)
    preparations = []
    for data in (phase_data, config.data):
        preparations.append(multisector.prepare_multisector(
            merged, basis_inputs=resolve_basis_inputs(data),
            **data.select("domain", "start_date", "flux_sources", "averaging_error", "filters",
                          "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                          "min_error", "min_error_options"),
        ))
    xr.testing.assert_equal(preparations[0].basis_functions.flat_basis(),
                            preparations[1].basis_functions.flat_basis())
    xr.testing.assert_equal(preparations[0].basis_functions.operator.basis_matrix,
                            preparations[1].basis_functions.operator.basis_matrix)
    xr.testing.assert_equal(preparations[0].inv_inputs.H, preparations[1].inv_inputs.H)
    assert len(weights) == 2
    for weighting_flux in weights:
        xr.testing.assert_equal(weighting_flux, flux["inventory"])


def test_one_preparation_two_priors_and_likelihoods_matches_full_prepared_and_replay(tmp_path, monkeypatch):
    resolutions = []
    resolve = multisector.resolve_runtime_flux

    def count(sources, **kwargs):
        resolutions.append(tuple(sources))
        return resolve(sources, **kwargs)

    monkeypatch.setattr(multisector, "resolve_runtime_flux", count)
    run_example(tmp_path, draws=8)
    # One shared preparation, and one independent acquired/full reference per model.
    assert resolutions == [("inventory", "supplemental")] * 3


def test_preparation_retains_site_subset_requested_window_sources_and_borrowed_arrays(tmp_path):
    merged, options, flux, h, _ = synthetic_inputs(tmp_path)
    site = merged.site_data["TAC"]
    graph = site.fp_x_flux_sectoral.data
    original = site.copy(deep=True)
    selectors = SiteOptions.from_inputs(sites=["MHD", "TAC"], averaging_period=["3h", "1h"])
    merged = replace(merged, site_options=selectors,
                     site_data={"MHD": site.isel(time=slice(0, 0)), "TAC": site})
    data = DataConfig.from_params({**options, "sites": ["MHD", "TAC"], "averaging_period": ["3h", "1h"]},
                                  multisector=True)
    manifest = stages.prepare(data, merged_data=merged, output_dir=tmp_path / "prepared")
    prepared, recorded, _ = stages.load_preparation(manifest)
    assert prepared.sites == ("TAC",) and prepared.averaging_period == ("1h",)
    assert recorded.site_options.sites == ("MHD", "TAC")
    assert recorded.start_date == "2019-01-01" and recorded.end_date == "2019-02-01"
    assert tuple(prepared.inv_inputs.H.source.values) == tuple(merged.flux_data)
    assert tuple(prepared.basis_functions.flux.source.values) == tuple(merged.flux_data)
    for source in h:
        np.testing.assert_allclose(prepared.inv_inputs.H.sel(source=source).squeeze(), h[source])
        xr.testing.assert_allclose(prepared.basis_functions.flux.sel(source=source, drop=True), flux[source])
    assert merged.sites == ("MHD", "TAC")
    assert site.fp_x_flux_sectoral.data is graph and isinstance(graph, da.Array)
    assert "mf_error" not in site
    xr.testing.assert_equal(site, original)


def test_full_and_prepared_execution_need_no_checkpoint_machinery_and_retain_scientific_states(tmp_path, monkeypatch):
    merged, options, *_ = synthetic_inputs(tmp_path)
    config = RhimeConfig.from_params({**options, "sector_sources": model_options(.2)["sector_sources"],
                                     "mismatch_model": "fixed_error", "output_name": "in-memory",
                                     "output_format": "none"}, multisector=True)
    config = replace(config, sampler=RhimeSampler(
        draws=4, tune=4, chains=1, sample_kwargs={"cores": 1, "random_seed": 42},
        sample_prior_predictive=False, sample_posterior_predictive=False,
    ))
    prepared = multisector.prepare_multisector(
        merged, basis_inputs=resolve_basis_inputs(config.data),
        **config.data.select("domain", "start_date", "flux_sources", "averaging_error", "filters",
                             "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                             "min_error", "min_error_options"),
    )
    diagnostic = da.from_array(np.arange(prepared.inv_inputs.sizes["nmeasure"]), chunks=2)
    prepared = replace(prepared, inv_inputs=prepared.inv_inputs.assign(diagnostic=("nmeasure", diagnostic)))
    borrowed = merged.site_data["TAC"].fp_x_flux_sectoral.data

    def forbidden(*args, **kwargs):
        pytest.fail("in-memory execution entered checkpoint machinery")

    monkeypatch.setattr(PreparedInputs, "save", forbidden)
    monkeypatch.setattr(RhimeModelSpec, "to_record", forbidden)
    monkeypatch.setattr(stages, "start_checkpoint", forbidden)
    monkeypatch.setattr(stages, "prepare", forbidden)
    monkeypatch.setattr(stages, "save_preparation", forbidden)
    monkeypatch.setattr(stages, "save_samples", forbidden)
    full = multisector.run_multisector(config=config, merged_data=merged)
    resumed = multisector.run_multisector(config=config, prepared=prepared)
    assert isinstance(full, InversionOutput) and isinstance(resumed, InversionOutput)
    assert resumed.basis_functions.operator is prepared.basis_functions.operator
    assert resumed.basis_functions.flux.data is prepared.basis_functions.flux.data
    assert resumed.inv_inputs.mf.data is prepared.inv_inputs.mf.data
    assert resumed.inv_inputs.diagnostic.data is diagnostic and dask.is_dask_collection(diagnostic)
    assert merged.site_data["TAC"].fp_x_flux_sectoral.data is borrowed
    assert "mf_error" not in merged.site_data["TAC"]
    xr.testing.assert_allclose(full.trace["posterior"].to_dataset(), resumed.trace["posterior"].to_dataset())
    posterior = resumed.trace["posterior"]
    for suffix, source in (("biosphere", "supplemental"), ("fossil", "inventory")):
        np.testing.assert_allclose(
            posterior["x"].sel(source=source).values,
            np.exp(-np.log(2) / 2 + np.sqrt(np.log(2)) * posterior[f"x_{suffix}_active_latent"].values),
            rtol=1e-6,
        )
    assert not list(tmp_path.glob("**/*manifest.json"))


def test_equivalent_preparation_configuration_representation_reuses_unchanged_numerical_inputs(tmp_path, monkeypatch):
    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    numerical = (manifest.parent / "prepared-inputs.nc").read_bytes()
    record = json.loads(manifest.read_text())
    generation = record["generation"]
    assert record["data"].pop("averaging_error") is True
    assert record["data"]["averaging_period"] == ["1h"]
    record["data"]["averaging_period"] = "1h"
    manifest.write_text(json.dumps(record))

    def forbidden(*args, **kwargs):
        pytest.fail("equivalent configuration repeated acquisition or preparation")

    monkeypatch.setattr(stages, "acquire_merged_data", forbidden)
    monkeypatch.setattr(stages, "prepare_multisector", forbidden)
    sampler = RhimeSampler(draws=4, tune=4, chains=1, sample_kwargs={"cores": 1, "random_seed": 42},
                           sample_prior_predictive=False, sample_posterior_predictive=False)
    sampled = stages.sample(manifest, model=model_options(.6), sampler=sampler, output_dir=tmp_path / "samples")
    replay = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))
    assert isinstance(replay, InversionOutput)
    assert replay.trace["posterior"].sizes["draw"] == 4
    assert replay.run_metadata["averaging_period"] == ["1h"]
    assert json.loads(manifest.read_text())["generation"] == generation
    assert (manifest.parent / "prepared-inputs.nc").read_bytes() == numerical
    assert "configuration_identity" not in json.loads(sampled.read_text())


def test_coherent_extra_prepared_source_cannot_silently_disappear_from_model_or_total_flux(tmp_path, monkeypatch):
    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "original")
    prepared, data, _ = stages.load_preparation(manifest)
    source_variables = {
        name: xr.concat([variable, variable.sel(source="inventory", drop=True).expand_dims(source=["unmodelled"])],
                        dim="source")
        for name, variable in prepared.inv_inputs.data_vars.items() if "source" in variable.dims
    }
    flux = prepared.basis_functions.flux
    extra_flux = flux.sel(source="inventory", drop=True).expand_dims(source=["unmodelled"])
    prepared = replace(prepared, inv_inputs=prepared.inv_inputs.drop_dims("source").assign(source_variables),
                       basis_functions=prepared.basis_functions.with_flux(xr.concat([flux, extra_flux], dim="source")))
    assert prepared.basis_functions.source_labels == tuple(prepared.inv_inputs.H.source.values)
    assert prepared.basis_functions.source_labels == ("inventory", "supplemental", "unmodelled")
    manifest = stages.save_preparation(prepared, data_config=data, output_dir=tmp_path / "extra-source")

    def forbidden(*args, **kwargs):
        pytest.fail("an unmatched scientific source reached materialization or inference")

    monkeypatch.setattr(multisector, "_materialize_pymc_inputs_and_priors", forbidden)
    monkeypatch.setattr(stages, "sample_rhime_model", forbidden)
    with pytest.raises(ValueError, match="source|Source"):
        stages.sample(manifest, model=model_options(.2), output_dir=tmp_path / "samples")
    config = RhimeConfig.from_params({**options, **model_options(.2), "output_name": "incompatible",
                                     "output_format": "none"}, multisector=True)
    with pytest.raises(ValueError, match="source|Source"):
        multisector.run_multisector(config=config, prepared=prepared)
    with pytest.raises(ValueError, match="sources must cover the retained source layout exactly"):
        multisector.build_multisector_model(
            prepared.inv_inputs,
            config.model,
            priors={f"x_{sector.variable_suffix}": sector.x_prior for sector in config.model.sectors},
        )
    assert not (tmp_path / "samples").exists()


def test_model_construction_consumes_supplied_eager_sensitivities_without_replacing_them(tmp_path):
    merged, options, _, h, _ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    prepared, _, _ = stages.load_preparation(manifest)
    original = prepared.inv_inputs.H.copy(deep=True)
    config = RhimeConfig.from_params({**options, **model_options(.2), "output_name": "modified-sensitivity",
                                     "output_format": "none"}, multisector=True)
    inputs, priors = multisector.materialize_multisector_inputs(prepared, config.model)
    modified = inputs.assign(H=inputs.H * 10)
    built = multisector.build_multisector_model(modified, config.model, priors=priors)
    expected = np.column_stack([h[source] for source in inputs.H.source.values]) * 10
    np.testing.assert_allclose(built.model["hx"].get_value(), expected)
    xr.testing.assert_identical(prepared.inv_inputs.H, original)


def test_signed_source_preparation_and_saved_replay_preserve_native_sector_and_total_flux(tmp_path):
    merged, options, flux, h, _ = synthetic_inputs(tmp_path)
    original_site = merged.site_data["TAC"]
    sign = xr.DataArray([1., -1.], dims="source", coords={"source": list(flux)})
    signed_site = original_site.assign(
        fp_x_flux_sectoral=original_site.fp_x_flux_sectoral * sign,
        mf=(original_site.mf - 1.8 * xr.DataArray(h["supplemental"], dims="time",
                                                 coords={"time": original_site.time})).assign_attrs(original_site.mf.attrs),
    )
    signed_flux = -flux["supplemental"]
    merged = replace(merged, site_data={"TAC": signed_site},
                     flux_data={**merged.flux_data, "supplemental": signed_flux.to_dataset(name="flux")})
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "signed-preparation")
    prepared, data, generation = stages.load_preparation(manifest)
    np.testing.assert_allclose(prepared.inv_inputs.H.sel(source="supplemental").squeeze(), -h["supplemental"])
    xr.testing.assert_allclose(prepared.basis_functions.flux.sel(source="supplemental", drop=True), signed_flux)
    biosphere = np.array([.5, 2.])
    fossil = np.array([1.2, .8])
    posterior = make_trace(posterior=xr.Dataset(
        {"x_biosphere": (("chain", "draw", "state"), biosphere[None, :, None]),
         "x_fossil": (("chain", "draw", "state"), fossil[None, :, None])},
        coords={"chain": [0], "draw": [0, 1], "state": prepared.inv_inputs.state},
    ))
    choices = model_options(.2)
    posterior.attrs["rhime_priors"] = json.dumps({
        f"x_{name}": prior for name, prior in choices["sector_priors"].items()
    })
    sampled = stages.save_samples(
        manifest, posterior, model=resolve_model_options(choices, data_config=data),
        output_contract=OutputContract(
            variable_roles={"flux_scale:biosphere": "x_biosphere", "flux_scale:fossil": "x_fossil"},
            supported_output_formats=("none", "inv_out"),
        ), actual_inference={"method": "controlled numerical reference", "draws": 2, "chains": 1},
        preparation_generation=generation, output_dir=tmp_path / "signed-samples",
    )
    replay = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))
    products = make_multisector_flux_trace_outputs(replay, report_flux_on_inversion_grid=False)
    expected_biosphere = biosphere[:, None, None] * signed_flux.values
    expected_fossil = fossil[:, None, None] * flux["inventory"].values
    for sector, expected in (("biosphere", expected_biosphere), ("fossil", expected_fossil),
                             ("total", expected_biosphere + expected_fossil)):
        actual = products[f"flux_{sector}_posterior"].squeeze(drop=True).transpose("draw", "lat", "lon")
        np.testing.assert_allclose(actual, expected)
    assert bool((replay.basis_functions.flux.sel(source="supplemental") < 0).all())
    assert bool((products.flux_biosphere_posterior < 0).all())


@pytest.mark.parametrize("incompatibility", ["minimum-error", "boundary-policy", "source-layout", "source-claim"])
def test_genuine_preparation_incompatibilities_fail_without_inference(tmp_path, monkeypatch, incompatibility):
    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "original")
    prepared, data, _ = stages.load_preparation(manifest)
    model = model_options(.2)
    if incompatibility == "minimum-error":
        prepared = replace(prepared, inv_inputs=prepared.inv_inputs.drop_vars("min_error"))
        model["mismatch_model"] = "pollution_event"
        match = "min_error"
    elif incompatibility == "boundary-policy":
        model["bc_freq"] = "monthly"
        match = "conflicts with data"
    elif incompatibility == "source-layout":
        match = "source|Source"
    else:
        model["flux_sources"] = ["supplemental", "unavailable"]
        match = "conflict|source|Source"
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *a, **k: pytest.fail("inference executed"))
    with pytest.raises(ValueError, match=match):
        if incompatibility == "source-layout":
            prepared = replace(prepared, inv_inputs=prepared.inv_inputs.sel(source=["supplemental"]))
        manifest = stages.save_preparation(prepared, data_config=data, output_dir=tmp_path / "changed")
        stages.sample(manifest, model=model, output_dir=tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


def test_fresh_process_replay_uses_original_science_and_current_output_without_graph(tmp_path):
    from tests.test_multisector_handoffs import _trace_and_contract

    merged, options, *_ = synthetic_inputs(tmp_path)
    preparation = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                                 output_dir=tmp_path / "prepared")
    prepared, data, generation = stages.load_preparation(preparation)
    trace, contract = _trace_and_contract(prepared)
    sampled = stages.save_samples(
        preparation, trace, output_contract=contract,
        model=resolve_model_options(model_options(.2), data_config=data),
        actual_inference={"draws": 3, "chains": 2}, output_dir=tmp_path / "samples",
        preparation_generation=generation,
    )
    # The retained scientific basis is sufficient; replay needs no original operator file.
    (tmp_path / "operator.nc").unlink()
    script = '''
import sys
from pathlib import Path
import numpy as np
from openghg_inversions.rhime import multisector, multisector_stages as stages
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeOutputSpec
root = Path(sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError("replay acquired, materialized, constructed or sampled a model")
multisector.build_multisector_model = forbidden
multisector.materialize_multisector_inputs = forbidden
stages.build_multisector_model = forbidden
stages.materialize_multisector_inputs = forbidden
stages.MergedData.from_options = forbidden
stages.sample_rhime_model = forbidden
stages.resolve_sampler = forbidden
RhimeSampler.sample = forbidden
result = stages.postprocess(root / "samples/sample-manifest.json", output=RhimeOutputSpec(
    output_format="inv_out", output_name="current-report", output_path=str(root / "current-products")))
assert not hasattr(result, "model") and not hasattr(result, "sampler")
assert result.run_metadata["start_date"] == "2019-01-01" and result.run_metadata["end_date"] == "2019-02-01"
assert result.output_metadata["sampler"] == {"draws": 3, "chains": 2}
np.testing.assert_array_equal(result.trace["posterior"]["x_biosphere"], np.ones((2, 3, 1)))
np.testing.assert_array_equal(result.trace["posterior"]["x_fossil"], np.full((2, 3, 1), 1.2))
assert [sector["name"] for sector in result.model_metadata["sectors"]] == ["biosphere", "fossil"]
assert [sector["flux_source"] for sector in result.model_metadata["sectors"]] == ["supplemental", "inventory"]
np.testing.assert_allclose(result.basis_functions.flux.sel(source="inventory"), [[1.,2.],[3.,4.]])
assert "build_and_sample_seconds" not in result.output_metadata
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
    files = list((tmp_path / "current-products").glob("*inversion_output.nc"))
    assert len(files) == 1 and files[0].name.startswith("current-report2019-01-01")
    assert sampled.is_file()


@pytest.mark.parametrize("all_zero", [False, True])
def test_sector_priors_and_model_arrays_share_materialization_and_skip_disabled_priors(tmp_path, all_zero):
    merged, options, *_ = synthetic_inputs(tmp_path)
    manifest = stages.prepare(DataConfig.from_params(options, multisector=True), merged_data=merged,
                              output_dir=tmp_path / "prepared")
    prepared, data, _ = stages.load_preparation(manifest)
    sensitivity = prepared.inv_inputs.H
    calls = []

    @dask.delayed
    def shared():
        calls.append(1)
        return np.zeros(sensitivity.shape) if all_zero else sensitivity.values

    raw = da.from_delayed(shared(), shape=sensitivity.shape, dtype=float)
    h = sensitivity.copy(data=raw)
    mean = h.sum(("source", "nmeasure"))
    state_dim = next(dim for dim in mean.dims)
    mean = mean.assign_coords(prior_group=(state_dim, raw.sum(axis=tuple(h.get_axis_num(dim) for dim in ("source", "nmeasure"))).astype(int)))
    prior = {"pdf": "normal", "mu": mean, "sigma": xr.DataArray(raw.sum() + 1.)}

    @dask.delayed
    def disabled():
        raise AssertionError("disabled baseline prior executed")

    unused = {"pdf": "normal", "mu": da.from_delayed(disabled(), shape=(1,), dtype=float), "sigma": .1}
    model = resolve_model_options({**model_options(.2), "sector_priors": {"biosphere": prior, "fossil": prior},
                                   "bc_prior": unused, "offset_prior": unused}, data_config=data)
    borrowed = replace(prepared, inv_inputs=prepared.inv_inputs.assign(H=h))
    model_inputs, priors = multisector.materialize_multisector_inputs(borrowed, model)
    built = multisector.build_multisector_model(model_inputs, model, priors=priors)
    assert calls == [1]
    assert h.data is raw and dask.is_dask_collection(mean)
    assert prior["mu"] is mean
    records = prior_records(built.model)
    assert ("x_biosphere" in records) is not all_zero
    assert ("x_fossil" in records) is not all_zero
    assert "bc" not in records and "offset_latent" not in records
    if not all_zero:
        for suffix in ("biosphere", "fossil"):
            np.testing.assert_allclose(built.model[f"x_{suffix}_prior_mu"].get_value(),
                                       sensitivity.sum(("source", "nmeasure")))
            assert records[f"x_{suffix}"]["sigma"] == float(sensitivity.sum() + 1.)
