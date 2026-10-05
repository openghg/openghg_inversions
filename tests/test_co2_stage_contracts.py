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


@pytest.mark.parametrize("binding", [None, "absent", "malformed", "incorrect_digest"])
def test_co2_replay_requires_family_version_and_stays_graph_free(tmp_path, monkeypatch, binding):
    from openghg_inversions.rhime import stages
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
        "schema_version": 3,
        "identity_version": 1,
        "recipe": "co2",
        "producer": "openghg_inversions",
        "stage": "sample",
        "configuration_identity": stages.configuration_identity(setup, model="co2"),
        "effective_configuration": stages.effective_configuration(setup, model="co2"),
        "artifact_identities": {
            "prepared_inputs": file_identity(prepared_path),
            "posterior": file_identity(posterior),
        },
    }
    if binding is not None:
        sample["schema_version"] = 2
        if binding != "absent":
            binding_path = tmp_path / "output-binding.json"
            binding_path.write_text("{" if binding == "malformed" else "{}")
            sample["artifacts"] = {"output_binding": binding_path.name}
            sample["artifact_identities"]["output_binding"] = (
                file_identity(binding_path) if binding == "malformed" else "sha256:" + "0" * 64
            )
    sampling = write_json(tmp_path / "sample-manifest.json", sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("CO2 authenticated replay must not build a model.")

    monkeypatch.setattr(co2_stages, "build_rhime_co2", forbidden)
    monkeypatch.setattr(co2_stages, "build_rhime_co2_cached_sigma", forbidden)
    if binding is not None:
        monkeypatch.setattr(co2_stages, "load_trace", forbidden)
        with pytest.raises(ValueError, match="unsupported schema_version"):
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


@pytest.mark.parametrize("variant", ["ordinary", "cached_fixed_ou"])
def test_direct_and_staged_co2_sampling_and_products_agree(tmp_path, variant):
    from openghg_inversions.rhime.co2 import resolve_co2_family_config
    from openghg_inversions.rhime.co2 import stages
    from openghg_inversions.rhime._stage_artifacts import file_identity, write_json
    from openghg_inversions.serialization import load_trace, save_trace

    source, prepared = _handoff(tmp_path)
    params = _config(source)
    params["variant"] = variant
    params["sampling"].update(random_seed=42, sample_prior_predictive=3, sample_posterior_predictive=True)
    params["outputs"] = {"output_format": "basic"}
    if variant == "cached_fixed_ou":
        params["likelihood"] = {
            "kind": "fixed_ou",
            "tau_hours": {"AAA": 24.0, "BBB": 12.0},
            "site_amplitude_prior_scale": 0.3,
        }
        # Use canonical retained site labels with distinct scientific options.
        params["likelihood"]["tau_hours"] = dict(zip(prepared.sites, (24.0, 12.0), strict=True))
    config = resolve_co2_family_config(params)
    direct = config.runner(**config.runner_arguments(prepared))
    preparation = stages.prepare(setup=config, output_dir=tmp_path / "prepare")
    prepared_path = tmp_path / "prepare" / "prepared-inputs.nc"
    sampled = stages.sample(
        setup=config,
        prepared_inputs=prepared_path,
        preparation_manifest=preparation["manifest_path"],
        output_dir=tmp_path / "sample",
    )
    staged = load_trace(tmp_path / "sample" / "posterior.nc")
    for group in ("posterior", "posterior_predictive", "log_likelihood"):
        xr.testing.assert_allclose(direct[group].to_dataset(), staged[group].to_dataset())
    result = stages.postprocess(
        setup=config,
        prepared_inputs=prepared_path,
        preparation_manifest=preparation["manifest_path"],
        posterior=tmp_path / "sample" / "posterior.nc",
        sample_manifest=sampled["manifest_path"],
        output_dir=tmp_path / "products",
    )
    # The ordinary sampler does not seed prior draws. Product parity uses the
    # same saved prior and posterior, rather than comparing independent priors.
    xr.testing.assert_identical(
        direct["prior_predictive"]["y"].coords, staged["prior_predictive"]["y"].coords
    )
    assert bool(direct["prior_predictive"]["y"].notnull().all())
    direct["prior"] = staged["prior"].to_dataset()
    direct["prior_predictive"] = staged["prior_predictive"].to_dataset()
    direct_path = tmp_path / "direct.nc"
    save_trace(direct, direct_path)
    sampled["artifact_identities"]["posterior"] = file_identity(direct_path)
    direct_manifest = write_json(tmp_path / "direct-manifest.json", sampled)
    replay = stages.postprocess(
        setup=config,
        prepared_inputs=prepared_path,
        preparation_manifest=preparation["manifest_path"],
        posterior=direct_path,
        sample_manifest=direct_manifest,
        output_dir=tmp_path / "direct-products",
    )
    xr.testing.assert_allclose(result.outputs["basic"], replay.outputs["basic"])
    assert config.sampler.sample_kwargs == {"random_seed": 42}


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", 2),
        ("identity_version", 2),
        ("identity_version", None),
        ("recipe", "standard"),
    ],
)
def test_co2_sample_contract_rejects_unsupported_family_before_loading_posterior(
    tmp_path, monkeypatch, field, value
):
    from openghg_inversions.rhime.co2 import stages
    from openghg_inversions.rhime._stage_artifacts import write_json

    source, _ = _handoff(tmp_path)
    config = stages.resolve_config(_config(source))
    preparation = stages.prepare(setup=config, output_dir=tmp_path / "prepare")
    sample = stages._manifest(config, "sample")
    sample[field] = value
    sample_path = write_json(tmp_path / "sample.json", sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported CO2 contract must fail before posterior loading")

    monkeypatch.setattr(stages, "load_trace", forbidden)
    with pytest.raises(ValueError):
        stages.postprocess(
            setup=config,
            prepared_inputs=tmp_path / "prepare" / "prepared-inputs.nc",
            preparation_manifest=preparation["manifest_path"],
            posterior=tmp_path / "missing.nc",
            sample_manifest=sample_path,
            output_dir=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("empty_variable", [False, True])
def test_co2_empty_prior_evidence_fails_but_preserves_predictive_artifacts(
    tmp_path, monkeypatch, empty_variable
):
    from openghg_inversions.rhime.co2 import stages

    source, _ = _handoff(tmp_path)
    config = stages.resolve_config(_config(source))
    preparation = stages.prepare(setup=config, output_dir=tmp_path / "prepare")
    values = xr.Dataset({"empty": ("draw", [])}) if empty_variable else xr.Dataset()
    monkeypatch.setattr(
        stages.pm, "sample_prior_predictive", lambda *args, **kwargs: xr.DataTree.from_dict({"prior": values})
    )
    result = stages.prior_predictive(
        setup=config,
        prepared_inputs=tmp_path / "prepare" / "prepared-inputs.nc",
        preparation_manifest=preparation["manifest_path"],
        output_dir=tmp_path / "prior",
    )
    assert result["status"] == "fail"
    assert (tmp_path / "prior" / "prior-predictive.nc").exists()
    assert (tmp_path / "prior" / "prior-predictive-manifest.json").exists()
