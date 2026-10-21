"""Linked checkpoints preserve joint science and permit graph-free replay."""

from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from examples.co2_o2_handoffs import FACTS, synthetic_inputs
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs
from openghg_inversions.rhime.co2.co2_o2_outputs import Co2O2OutputOptions
from openghg_inversions.rhime.co2.co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs


MODEL = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}}


@pytest.fixture(scope="module")
def joint_trace():
    return run_rhime_co2_o2_from_prepared_inputs(
        prepared_inputs=synthetic_inputs(), sampler=RhimeSampler(
            draws=4, tune=4, chains=1, nuts_sampler="pymc", sample_prior_predictive=2,
            sample_posterior_predictive=True,
            sample_kwargs={"random_seed": 19, "compute_convergence_checks": False},
        ),
    )


@pytest.mark.parametrize("extension", [".nc", ".zarr"])
def test_import_keeps_unequal_axis_artifact_and_original_request(tmp_path, extension):
    prepared = synthetic_inputs()
    original = prepared.observations.copy(deep=True)
    source = tmp_path / f"prepared{extension}"
    prepared.save(source)
    manifest = handoffs.prepare(prepared_path=source, output_dir=tmp_path / "import")
    loaded = handoffs.load_preparation(manifest)
    assert loaded.payload == source.resolve()
    assert list(manifest.parent.iterdir()) == [manifest]
    assert (loaded.start_date, loaded.end_date, loaded.domain) == tuple(FACTS.values())
    assert prepared.observations.time.min() > np.datetime64(loaded.start_date)
    assert prepared.observations.time.max() < np.datetime64(loaded.end_date)
    xr.testing.assert_identical(prepared.observations, original)
    xr.testing.assert_identical(loaded.prepared.observations, prepared.observations)
    xr.testing.assert_identical(loaded.prepared.aggregation_error.covariance,
                                prepared.aggregation_error.covariance)
    xr.testing.assert_identical(loaded.prepared.o2_co2_flux_ratio, prepared.o2_co2_flux_ratio)
    xr.testing.assert_identical(loaded.prepared.independent_error_sd, prepared.independent_error_sd)
    assert loaded.prepared.retained_prior.mean.source.values.tolist() == ["GPP", "TER", "FF", "ocean", "ocean"]
    np.testing.assert_allclose(loaded.prepared.aggregation_error.covariance[:2, 2:],
                               [[-.1, -.05, -.02], [-.08, -.04, -.01]])


@pytest.mark.parametrize("missing", ["start_date", "end_date", "domain"])
def test_missing_imported_request_fails_before_destination(tmp_path, missing):
    prepared = replace(synthetic_inputs(), provenance={})
    source = tmp_path / "prepared.nc"
    prepared.save(source)
    facts = {name: value for name, value in FACTS.items() if name != missing}
    with pytest.raises(ValueError, match=missing):
        handoffs.prepare(prepared_path=source, output_dir=tmp_path / "missing", **facts)
    assert not (tmp_path / "missing").exists()


def test_preparation_replacement_rejects_old_samples_before_writing(tmp_path, joint_trace):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    generation = handoffs.load_preparation(prep).generation
    sample = handoffs.save_samples(prep, joint_trace, actual_inference={"draws": 4},
        preparation_generation=generation, output_dir=tmp_path / "existing-sample")
    handoffs.save_preparation(synthetic_inputs(), output_dir=prep.parent)
    with pytest.raises(ValueError, match="replaced|generation"):
        handoffs.load_samples(sample)
    with pytest.raises(ValueError, match="replaced|generation"):
        handoffs.save_samples(prep, joint_trace, actual_inference={"draws": 4},
                              preparation_generation=generation, output_dir=tmp_path / "sample")
    assert not (tmp_path / "sample").exists()


def test_sample_cannot_overwrite_referenced_preparation(tmp_path, joint_trace):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    source = prep.parent / "prepared-inputs.nc"
    original = source.read_bytes()
    (prep.parent / "posterior.nc").symlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        handoffs.save_samples(prep, joint_trace, actual_inference={"draws": 4},
                              preparation_generation=handoffs.load_preparation(prep).generation,
                              output_dir=prep.parent)
    assert source.read_bytes() == original


def test_failed_sample_write_invalidates_completion(tmp_path, monkeypatch, joint_trace):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    options = {"actual_inference": {"draws": 4},
               "preparation_generation": handoffs.load_preparation(prep).generation,
               "output_dir": tmp_path / "sample"}
    sample = handoffs.save_samples(prep, joint_trace, **options)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(handoffs, "save_trace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        handoffs.save_samples(prep, joint_trace, **options)
    assert not sample.exists()


def test_graph_free_replay_retains_roles_labels_and_inference(tmp_path, monkeypatch, joint_trace):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    sample = handoffs.save_samples(prep, joint_trace, actual_inference={"draws": 4, "chains": 1},
        preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample")

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not construct a model or sampler")

    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    monkeypatch.setattr(handoffs, "sample", forbidden)
    monkeypatch.setattr(handoffs, "prior_predictive", forbidden)
    monkeypatch.setattr(handoffs, "build_rhime_co2_o2", forbidden)
    monkeypatch.setattr(handoffs, "sample_rhime_model", forbidden)
    result = handoffs.postprocess(sample, output=Co2O2OutputOptions(
        output_format="basic", output_path=tmp_path / "products", output_name="replayed",
    ))
    assert result.start_date == FACTS["start_date"]
    assert result.end_date == FACTS["end_date"]
    assert result.output_metadata["sampler"] == {"draws": 4, "chains": 1}
    assert json.loads(result.idata.attrs["rhime_variable_roles"]) == json.loads(joint_trace.attrs["rhime_variable_roles"])
    assert result.idata.observed_data.indexes["observation"].equals(synthetic_inputs().observations.indexes["observation"])
    scales = json.loads(result.idata.attrs["rhime_variable_roles"])["flux_scale"]
    assert result.idata.posterior[scales].source.values.tolist() == ["GPP", "TER", "FF", "ocean", "ocean"]
    assert result.outputs


def test_heterogeneous_import_fails_before_checkpoint(tmp_path):
    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from test_rhime_co2_o2 import _inputs

    prepared = prepare_co2_o2_inputs(**_inputs())
    source = tmp_path / "mixed.nc"
    prepared.save(source)
    with pytest.raises(ValueError, match="unit|OPE-86"):
        handoffs.prepare(prepared_path=source, output_dir=tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("supported_units", [True, False])
def test_saving_lazy_units_runs_shared_payload_once_and_publishes_only_supported_units(tmp_path, supported_units):
    import dask.array as da
    from dask import delayed

    prepared = synthetic_inputs()
    values = prepared.observations.to_numpy()
    units = np.array(["ppm"] * 5 if supported_units else ["ppm"] * 2 + ["per meg"] * 3)
    executions = []

    @delayed
    def shared_payload():
        executions.append("shared")
        return values, units

    shared = shared_payload()
    lazy_values = da.from_delayed(shared[0], shape=values.shape, dtype=values.dtype)
    lazy_units = da.from_delayed(shared[1], shape=units.shape, dtype=units.dtype)
    prepared = replace(
        prepared,
        observations=prepared.observations.copy(data=lazy_values).assign_coords(
            observation_units=("observation", lazy_units)),
        fixed_prior_contribution=prepared.fixed_prior_contribution.assign_coords(
            observation_units=("observation", lazy_units)),
        independent_error_sd=prepared.independent_error_sd.assign_coords(
            observation_units=("observation", lazy_units)),
        aggregation_error=replace(prepared.aggregation_error,
            covariance=prepared.aggregation_error.covariance.assign_coords(
                observation_units=("observation", lazy_units))),
    )
    destination = tmp_path / "prepare"
    if supported_units:
        manifest = handoffs.save_preparation(prepared, output_dir=destination)
        assert manifest.is_file()
    else:
        with pytest.raises(ValueError, match="identical|OPE-86"):
            handoffs.save_preparation(prepared, output_dir=destination)
        assert not (destination / "prepare-manifest.json").exists()
    assert executions == ["shared"]
    assert prepared.observations.data is lazy_values
    assert prepared.observations.observation_units.data is lazy_units


@pytest.mark.parametrize("ou_amplitude", [None, .3])
def test_loaded_joint_likelihood_matches_dense_covariance_once(tmp_path, ou_amplitude):
    from scipy.stats import multivariate_normal

    from openghg_inversions.rhime.co2.co2_o2_configuration import resolve_co2_o2_model

    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    loaded = handoffs.load_preparation(prep).prepared
    choices = dict(MODEL)
    covariance = loaded.aggregation_error.covariance.values + np.diag(loaded.independent_error_sd.values ** 2)
    if ou_amplitude is not None:
        choices["likelihood"] = {"kind": "fixed_ou", "tau_hours": 12., "fixed_site_amplitudes": ou_amplitude}
        observations = loaded.observations
        time_difference = np.abs(observations.time.values[:, None] - observations.time.values[None, :]) / np.timedelta64(1, "h")
        same_block = ((observations.species.values[:, None] == observations.species.values[None, :])
                      & (observations.site.values[:, None] == observations.site.values[None, :]))
        covariance = covariance + ou_amplitude ** 2 * np.exp(-time_difference / 12.) * same_block
    built = handoffs.build_rhime_co2_o2(prepared_inputs=loaded, **resolve_co2_o2_model(choices, loaded))
    model = built.model
    point = model.initial_point()
    mean = model.compile_fn(model.replace_rvs_by_values([model["modelled_concentration"]]))(point)[0]
    actual = model.compile_logp(vars=[model["y"]])(point)
    expected = multivariate_normal.logpdf(loaded.observations.values, mean=mean, cov=covariance)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_fixed_ou_prior_stage_keeps_joint_prediction_labels(tmp_path):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    result = handoffs.prior_predictive(prep, model={
        **MODEL, "likelihood": {"kind": "fixed_ou", "tau_hours": 12., "fixed_site_amplitudes": .3},
    }, output_dir=tmp_path / "prior", draws=2, random_seed=19)
    assert result["status"] == "pass"
    from openghg_inversions.serialization import load_trace

    prior = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    assert prior.prior_predictive.indexes["observation"].equals(synthetic_inputs().observations.indexes["observation"])
    assert json.loads(prior.attrs["rhime_model_metadata"])["likelihood"] == "joint Gaussian with fixed OU blocks by (species, site)"
