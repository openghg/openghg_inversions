"""CO2 stage boundary checks without costly posterior sampling."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import Co2PreparedInputs
from openghg_inversions.rhime.co2.stages import (
    co2_configuration_identity,
    prepare_co2_stage,
    resolve_co2_stage_setup,
)
from test_co2_preparation import _canonical_inputs, _reduction
from test_co2_staged_workflow import _handoff


def _config(path):
    return {
        "format_version": 1,
        "recipe": "co2",
        "variant": "ordinary",
        "prepared_inputs": {"path": str(path)},
        "likelihood": {"kind": "additive_sigma", "no_model_error": True},
        "sampling": {"draws": 4, "tune": 4, "chains": 1, "nuts_sampler": "pymc"},
    }


def test_direct_source_neutral_coherent_preparation(tmp_path):
    canonical = _canonical_inputs()
    setup = resolve_co2_stage_setup(_config(tmp_path / "not-needed.nc"))
    manifest = prepare_co2_stage(
        setup=setup,
        output_dir=tmp_path / "stage",
        canonical_inputs=canonical,
        reduction=_reduction(canonical),
        aggregation_error_rank=None,
    )
    prepared = Co2PreparedInputs.load(tmp_path / "stage" / "prepared-inputs.nc")
    assert prepared.aggregation_error_mode == "dense"
    xr.testing.assert_identical(prepared.inv_inputs.mf, canonical.inv_inputs.mf)
    assert manifest["stage"] == "prepare"
    assert "artifact_identities" not in manifest
    persisted = json.loads((tmp_path / "stage" / "prepare-manifest.json").read_text())
    assert persisted["ogi"] == manifest["ogi"]
    assert not (tmp_path / "not-needed.nc").exists()


def test_identity_ignores_transport_output_and_sampling_but_binds_science(tmp_path):
    params = _config(tmp_path / "original.nc")
    first = co2_configuration_identity(resolve_co2_stage_setup(params))
    relocated = deepcopy(params)
    relocated["prepared_inputs"]["path"] = str(tmp_path / "relocated.nc")
    relocated["sampling"]["draws"] = 10
    relocated["outputs"] = {"output_name": "renamed"}
    assert co2_configuration_identity(resolve_co2_stage_setup(relocated)) == first
    relocated["likelihood"]["fixed_model_mismatch"] = 1.0
    assert co2_configuration_identity(resolve_co2_stage_setup(relocated)) != first


def test_invalid_cached_site_mapping_has_no_destination(tmp_path):
    source, _ = _handoff(tmp_path)
    params = _config(source)
    params.update(
        variant="cached_fixed_ou",
        likelihood={
            "kind": "fixed_ou",
            "tau_hours": {"wrong": 24.0},
            "site_amplitude_prior_scale": 1.0,
        },
    )
    setup = resolve_co2_stage_setup(params)
    with pytest.raises(ValueError, match="prepared site labels"):
        prepare_co2_stage(setup=setup, output_dir=tmp_path / "unopened")
    assert not (tmp_path / "unopened").exists()


def test_unsupported_sector_request_fails_at_config_boundary(tmp_path):
    params = _config(tmp_path / "absent.nc")
    params["outputs"] = {"source_to_sector": {"fossil": "total"}}
    with pytest.raises(ValueError, match="source_to_sector"):
        resolve_co2_stage_setup(params)


@pytest.mark.parametrize("variant", ["ordinary", "cached_fixed_ou"])
@pytest.mark.parametrize("has_prior", [True, False])
def test_affine_replay_checks_saved_draws_instead_of_current_sampler(
    tmp_path, monkeypatch, variant, has_prior
):
    from openghg_inversions.basis.affine_flux_map_io import save
    from openghg_inversions.rhime.co2 import stages
    from openghg_inversions.rhime.co2.co2_affine_output import (
        prepared_inputs_content_id,
        produce_bucket_affine_flux_map,
    )
    from openghg_inversions.rhime.sampling import RhimeSampler
    from openghg_inversions.serialization import save_trace
    from test_co2_affine_output import _prepared
    from test_co2_staged_outputs import _components_fixture

    prepared, trace = _components_fixture()
    _, native_mean = _prepared()
    source = tmp_path / "source.nc"
    prepared.save(source)
    companion = tmp_path / "affine.nc"
    save(
        produce_bucket_affine_flux_map(
            prepared, native_mean, prepared_inputs_id=prepared_inputs_content_id(source)
        ),
        companion,
    )
    params = _config(source)
    if variant == "cached_fixed_ou":
        params.update(
            variant=variant,
            likelihood={"kind": "fixed_ou", "tau_hours": 24.0, "site_amplitude_prior_scale": 1.0},
        )
        trace.attrs["rhime_model_metadata"] = json.dumps({"recipe": "co2_cached_sigma_fixed_ou"})
    params["outputs"] = {"output_format": "basic", "reconstruction_path": str(companion)}
    params["sampling"]["sample_prior_predictive"] = 4
    sampled_setup = resolve_co2_stage_setup(params)
    preparation = prepare_co2_stage(setup=sampled_setup, output_dir=tmp_path / "prepare")
    sample = stages._manifest(sampled_setup, "sample")
    sampling = tmp_path / "sample-manifest.json"
    sampling.write_text(json.dumps(sample))
    if not has_prior:
        del trace["prior"]
    posterior = tmp_path / "posterior.nc"
    save_trace(trace, posterior)

    # The current sampling choice cannot remove draws from the saved trace.
    params["sampling"]["sample_prior_predictive"] = False
    replay_setup = resolve_co2_stage_setup(params)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved CO2 replay must not build a model or execute inference.")

    monkeypatch.setattr(stages, "build_rhime_co2", forbidden)
    monkeypatch.setattr(stages, "build_rhime_co2_cached_sigma", forbidden)
    monkeypatch.setattr(RhimeSampler, "sample", forbidden)
    arguments = {
        "setup": replay_setup,
        "prepared_inputs": tmp_path / "prepare" / "prepared-inputs.nc",
        "preparation_manifest": preparation["manifest_path"],
        "posterior": posterior,
        "sample_manifest": sampling,
        "output_dir": tmp_path / "replay",
    }
    if not has_prior:
        with pytest.raises(ValueError, match="Affine CO2 flux outputs require prior flux_scale draws"):
            stages.postprocess_co2_stage(**arguments)
        assert not (tmp_path / "replay").exists()
        return
    result = stages.postprocess_co2_stage(**arguments)
    assert result.model is None
    assert result.sampler is None
    assert (tmp_path / "replay" / "native_flux.nc").is_file()
    assert "flux_total_prior_mean" in result.outputs["native_flux"]
    manifest = json.loads(Path(result.output_metadata["postprocess_manifest_path"]).read_text())
    assert manifest["sampling_configuration"]["sample_prior_predictive"] == 4


def test_cached_tuning_is_sampling_configuration(tmp_path):
    params = _config(tmp_path / "source.nc")
    params.update(
        variant="cached_fixed_ou",
        likelihood={
            "kind": "fixed_ou",
            "tau_hours": 24.0,
            "site_amplitude_prior_scale": 1.0,
        },
    )
    first = co2_configuration_identity(resolve_co2_stage_setup(params))
    params["likelihood"].update(sigma_target_accept=0.95, state_target_accept=0.95)
    assert co2_configuration_identity(resolve_co2_stage_setup(params)) == first


def test_co2_preparation_allows_unknown_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openghg_inversions import _provenance
    from openghg_inversions.rhime.co2 import stages

    monkeypatch.setattr(_provenance, "__file__", str(tmp_path / "openghg_inversions" / "_provenance.py"))
    monkeypatch.setattr(
        _provenance.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="0.7.3", read_text=lambda name: None),
    )
    assert _provenance.installed_ogi_provenance() == {"version": "0.7.3", "revision": None, "dirty": None}
    setup = resolve_co2_stage_setup(_config(tmp_path / "not-needed.nc"))
    canonical = _canonical_inputs()
    manifest = stages.prepare_co2_stage(
        setup=setup,
        output_dir=tmp_path / "prepare",
        canonical_inputs=canonical,
        reduction=_reduction(canonical),
    )
    assert manifest["ogi"] == {"version": "0.7.3", "revision": None, "dirty": None}
    assert (tmp_path / "prepare" / "prepared-inputs.nc").is_file()


@pytest.mark.parametrize(
    "binding,recorded",
    [(None, "complete"), (None, None), (None, {"draws": "invalid"}),
     (None, {"draws": 17}), ("absent", "complete"), ("malformed", "complete")],
)
def test_co2_replay_requires_version_one_and_stays_graph_free(tmp_path, monkeypatch, binding, recorded):
    from openghg_inversions.rhime import _standard_stages, stages
    from openghg_inversions.rhime._stage_artifacts import write_json
    from openghg_inversions.rhime.co2 import stages as co2_stages
    from openghg_inversions.serialization import save_trace
    from test_co2_staged_outputs import _components_fixture

    monkeypatch.setattr(
        co2_stages, "installed_ogi_provenance",
        lambda: {"version": "0.7.4", "revision": None, "dirty": None},
    )
    prepared, trace = _components_fixture()
    source = tmp_path / "source.nc"
    prepared.save(source)
    params = _config(source)
    params["outputs"] = {"output_format": "basic"}
    setup = stages.resolve_stage_setup(params, model="co2")
    preparation = stages.prepare_rhime_stage(setup=setup, model="co2", output_dir=tmp_path / "prepare")
    prepared_path = tmp_path / "prepare" / "prepared-inputs.nc"
    posterior = tmp_path / "posterior.nc"
    save_trace(trace, posterior)
    sample = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "sample",
        "configuration_identity": stages.configuration_identity(setup, model="co2"),
        "effective_configuration": stages.effective_configuration(setup, model="co2"),
    }
    sampled_settings = sample["effective_configuration"]["sampler"]
    sampled_settings.update(draws=17, burn=2, tune=13, chains=3)
    if recorded != "complete":
        sampled_settings = recorded
        sample["effective_configuration"]["sampler"] = recorded
    if binding is not None:
        sample["schema_version"] = 2
        if binding != "absent":
            binding_path = tmp_path / "output-binding.json"
            binding_path.write_text("{" if binding == "malformed" else "{}")
            sample["artifacts"] = {"output_binding": binding_path.name}
    sampling = write_json(tmp_path / "sample-manifest.json", sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("CO2 version-1 replay must not build a model.")

    from openghg_inversions.rhime.sampling import RhimeSampler

    monkeypatch.setattr(RhimeSampler, "sample", forbidden)
    monkeypatch.setattr(co2_stages, "build_rhime_co2", forbidden)
    monkeypatch.setattr(co2_stages, "build_rhime_co2_cached_sigma", forbidden)
    monkeypatch.setattr(_standard_stages, "_build_prepared_model", forbidden)
    if binding is not None:
        monkeypatch.setattr(co2_stages, "load_trace", forbidden)
        with pytest.raises(ValueError, match="CO2 postprocessing requires schema_version=1"):
            stages.postprocess_rhime_stage(
                setup=setup,
                model="co2",
                prepared_inputs=prepared_path,
                preparation_manifest=preparation["manifest_path"],
                posterior=posterior,
                sample_manifest=sampling,
                output_dir=tmp_path / "replay",
            )
        assert not (tmp_path / "replay").exists()
        return
    result = stages.postprocess_rhime_stage(
        setup=setup,
        model="co2",
        prepared_inputs=prepared_path,
        preparation_manifest=preparation["manifest_path"],
        posterior=posterior,
        sample_manifest=sampling,
        output_dir=tmp_path / "replay",
    )
    assert result.model is None
    assert result.sampler is None
    if recorded == "complete":
        expected = {"draws": 17, "burn": 2, "tune": 13, "chains": 3, "nuts_sampler": "pymc"}
    else:
        expected = {"draws": 17} if recorded == {"draws": 17} else {}
    assert result.output_metadata["sampler"] == expected
    replay = json.loads(Path(result.output_metadata["postprocess_manifest_path"]).read_text())
    assert replay["effective_configuration"]["sampler"] == sampled_settings
    assert replay["sampling_configuration"] == sampled_settings
    assert replay["ogi"]["revision"] is None
    assert "artifact_identities" not in replay
    assert "input_identities" not in replay
    assert (tmp_path / "replay" / "basic.nc").is_file()


def test_staged_co2_rejects_incompatible_affine_projection(tmp_path):
    from dataclasses import replace

    from openghg_inversions.basis.affine_flux_map_io import load, save
    from openghg_inversions.rhime.co2.stages import prior_predictive_co2_stage

    source, _ = _handoff(tmp_path)
    params = _config(source)
    companion = tmp_path / "affine.nc"
    params["outputs"] = {"output_format": "basic", "reconstruction_path": str(companion)}
    setup = resolve_co2_stage_setup(params)
    preparation = prepare_co2_stage(setup=setup, output_dir=tmp_path / "prepare")
    artifact = load(companion)
    incompatible_path = tmp_path / "incompatible-affine.nc"
    save(replace(artifact, projection_provenance={"projection_strategy": "different"}), incompatible_path)
    setup = replace(setup, reconstruction_path=incompatible_path)
    with pytest.raises(ValueError, match="projection provenance differs"):
        prior_predictive_co2_stage(
            setup=setup,
            prepared_inputs=tmp_path / "prepare" / "prepared-inputs.nc",
            preparation_manifest=preparation["manifest_path"],
            output_dir=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()
