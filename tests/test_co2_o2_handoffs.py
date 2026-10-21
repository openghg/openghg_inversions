"""Linked checkpoints preserve joint science and permit graph-free replay."""

from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from examples.co2_o2_handoffs import FACTS, synthetic_inputs
from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs, co2_o2_runner
from openghg_inversions.rhime.co2.co2_o2_model import _gather_co2_o2_sensitivity
from openghg_inversions.rhime.co2.co2_o2_outputs import Co2O2OutputOptions


MODEL = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}}


@pytest.fixture
def no_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Linked persistence must not construct a model or sampler")

    monkeypatch.setattr(handoffs.pm.Model, "__init__", forbidden)
    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    monkeypatch.setattr(handoffs, "sample", forbidden)
    monkeypatch.setattr(handoffs, "prior_predictive", forbidden)
    monkeypatch.setattr(handoffs, "build_rhime_co2_o2", forbidden)
    monkeypatch.setattr(handoffs, "sample_rhime_model", forbidden)


@pytest.fixture
def joint_trace(no_inference):
    """Use labelled, nonconstant draws to test persistence without fitting a model."""
    prepared = synthetic_inputs()
    sensitivity = _gather_co2_o2_sensitivity(prepared.co2_sensitivity, prepared.o2_sensitivity)
    scaling = xr.DataArray(
        np.arange(20).reshape(1, 4, 5) / 20 + 0.5,
        dims=("chain", "draw", "state"),
        coords={"chain": [0], "draw": np.arange(4), **dict(prepared.retained_prior.mean.coords)},
        attrs={"units": "dimensionless flux scale"},
    )
    contribution = xr.dot(sensitivity, scaling, dim="state").transpose("chain", "draw", "observation")
    mean = contribution + prepared.fixed_prior_contribution
    error = xr.DataArray(
        np.sqrt(np.diag(prepared.aggregation_error.covariance) + prepared.independent_error_sd.values**2),
        dims="observation", coords=prepared.observations.coords,
    )
    posterior = xr.Dataset({
        "flux_scaling": scaling, "co2_o2_flux_contribution": contribution,
        "modelled_concentration": mean, "epsilon": error.broadcast_like(mean),
    })
    for name in ("co2_o2_flux_contribution", "modelled_concentration", "epsilon"):
        posterior[name].attrs["units"] = "ppm"
    prior = posterior.isel(draw=[3, 0]).assign_coords(draw=[0, 1])
    return xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={
            "rhime_recipe": "co2_o2",
            "rhime_variable_roles": json.dumps(co2_o2_runner._CO2_O2_VARIABLE_ROLES),
            "rhime_model_metadata": json.dumps(co2_o2_runner._co2_o2_metadata(
                prepared, observations=prepared.observations)),
        }),
        "posterior": posterior, "prior": prior,
        "posterior_predictive": xr.Dataset({"y": posterior.modelled_concentration - 0.25}),
        "prior_predictive": xr.Dataset({"y": prior.modelled_concentration + 0.125}),
        "observed_data": xr.Dataset({"y": prepared.observations}),
        "constant_data": xr.Dataset({
            "co2_o2_sensitivity": sensitivity,
            "fixed_prior_contribution": prepared.fixed_prior_contribution,
            "fixed_independent_error_sd": prepared.independent_error_sd,
            "aggregation_error_covariance": prepared.aggregation_error.covariance,
        }),
    })


@pytest.mark.parametrize("extension", [".nc", ".zarr"])
def test_import_keeps_unequal_axis_artifact_and_original_request(tmp_path, extension, no_inference):
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
def test_missing_imported_request_fails_before_destination(tmp_path, missing, no_inference):
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


def test_graph_free_replay_retains_roles_labels_and_inference(tmp_path, joint_trace):
    prep = handoffs.save_preparation(synthetic_inputs(), output_dir=tmp_path / "prepare")
    sample = handoffs.save_samples(prep, joint_trace, actual_inference={"draws": 4, "chains": 1},
        preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample")

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
    for name in joint_trace.children:
        xr.testing.assert_identical(result.idata[name], joint_trace[name])
    for species, count in (("co2", 2), ("o2", 3)):
        components = result.outputs[f"{species}_concentration_components"]
        assert components.sizes["observation"] == count
        assert components.sizes["draw"] == 4
        assert components.sizes["prior_draw"] == 2
        xr.testing.assert_allclose(
            components.modelled_posterior,
            (components.pollution_posterior + components.boundary_posterior + components.offset_posterior)
            .transpose(*components.modelled_posterior.dims),
        )


def test_heterogeneous_import_fails_before_checkpoint(tmp_path, no_inference):
    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from test_rhime_co2_o2 import _inputs

    prepared = prepare_co2_o2_inputs(**_inputs())
    source = tmp_path / "mixed.nc"
    prepared.save(source)
    with pytest.raises(ValueError, match="unit|OPE-86"):
        handoffs.prepare(prepared_path=source, output_dir=tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("supported_units", [True, False])
def test_saving_lazy_units_runs_shared_payload_once_and_publishes_only_supported_units(tmp_path, supported_units, no_inference):
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
