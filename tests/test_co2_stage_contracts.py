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
