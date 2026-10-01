"""CO2 stage boundary checks without costly posterior sampling."""

from copy import deepcopy
import json

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
    assert manifest["artifact_identities"]["prepared_inputs"].startswith("sha256:")
    assert json.loads((tmp_path / "stage" / "prepare-manifest.json").read_text())["ogi"]["revision"]
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


def test_unsupported_sector_request_and_missing_prior_fail_at_config_boundary(tmp_path):
    params = _config(tmp_path / "absent.nc")
    params["outputs"] = {"source_to_sector": {"fossil": "total"}}
    with pytest.raises(ValueError, match="source_to_sector"):
        resolve_co2_stage_setup(params)
    params["outputs"] = {"reconstruction_path": str(tmp_path / "absent-affine.nc")}
    params["sampling"]["sample_prior_predictive"] = False
    with pytest.raises(ValueError, match="prior_predictive"):
        resolve_co2_stage_setup(params)


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


def test_shared_provenance_allows_unknown_revision_but_co2_requires_it(tmp_path, monkeypatch):
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
    with pytest.raises(ValueError, match="identifiable installed Git revision"):
        stages._manifest(setup, "prepare")


def test_co2_version_one_replay_stays_graph_free_through_public_facade(tmp_path, monkeypatch):
    from openghg_inversions.rhime import _standard_stages, stages
    from openghg_inversions.rhime._stage_artifacts import file_identity, write_json
    from openghg_inversions.rhime.co2 import stages as co2_stages
    from openghg_inversions.serialization import save_trace
    from test_co2_staged_outputs import _components_fixture

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
        "artifact_identities": {
            "prepared_inputs": file_identity(prepared_path),
            "posterior": file_identity(posterior),
        },
    }
    sampling = write_json(tmp_path / "sample-manifest.json", sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("CO2 version-1 replay must not build a model.")

    monkeypatch.setattr(co2_stages, "build_rhime_co2", forbidden)
    monkeypatch.setattr(co2_stages, "build_rhime_co2_cached_sigma", forbidden)
    monkeypatch.setattr(_standard_stages, "_build_prepared_model", forbidden)
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
    assert (tmp_path / "replay" / "basic.nc").is_file()


def test_staged_co2_authenticates_affine_content_before_replay(tmp_path):
    from openghg_inversions.rhime.co2.stages import prior_predictive_co2_stage

    source, _ = _handoff(tmp_path)
    params = _config(source)
    companion = tmp_path / "affine.nc"
    params["outputs"] = {"output_format": "basic", "reconstruction_path": str(companion)}
    setup = resolve_co2_stage_setup(params)
    preparation = prepare_co2_stage(setup=setup, output_dir=tmp_path / "prepare")
    companion.write_bytes(companion.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="Affine-reconstruction content does not match"):
        prior_predictive_co2_stage(
            setup=setup,
            prepared_inputs=tmp_path / "prepare" / "prepared-inputs.nc",
            preparation_manifest=preparation["manifest_path"],
            output_dir=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()
