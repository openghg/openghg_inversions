"""CO2 handoff boundary checks without costly posterior sampling."""

from dataclasses import replace
import json

import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import Co2PreparedInputs, handoffs
from openghg_inversions.rhime.co2.co2_outputs import Co2OutputOptions
from openghg_inversions.rhime.co2.execution import load_co2_params, run_co2
from openghg_inversions.rhime.sampling import RhimeSampler
from test_co2_preparation import _canonical_inputs, _reduction
from test_co2_staged_workflow import _handoff
from test_co2_handoffs import FACTS, _components_fixture


def test_direct_source_neutral_coherent_preparation(tmp_path):
    from openghg_inversions.rhime.co2.handoffs import prepare, load_preparation

    canonical = _canonical_inputs()
    manifest = prepare(
        output_dir=tmp_path / "stage",
        canonical_inputs=canonical,
        reduction=_reduction(canonical),
        aggregation_error_rank=None,
        start_date="2020-01-01", end_date="2020-02-01", domain="EUROPE",
    )
    prepared = Co2PreparedInputs.load(tmp_path / "stage" / "prepared-inputs.nc")
    assert prepared.aggregation_error_mode == "dense"
    xr.testing.assert_identical(prepared.inv_inputs.mf, canonical.inv_inputs.mf)
    loaded = load_preparation(manifest)
    assert loaded.start_date == "2020-01-01"
    assert loaded.end_date == "2020-02-01"
    persisted = json.loads((tmp_path / "stage" / "prepare-manifest.json").read_text())
    assert set(persisted["provenance"]) == {"version", "revision", "dirty"}
    assert not (tmp_path / "not-needed.nc").exists()


@pytest.mark.parametrize("variant", ["ordinary", "cached_fixed_ou"])
def test_co2_rejects_retired_variant_without_loading_inputs(variant):
    with pytest.raises(ValueError, match="config.variant"):
        load_co2_params(params={"variant": variant})


def test_invalid_inferred_ou_site_mapping_has_no_sample_destination(tmp_path):
    source, _ = _handoff(tmp_path)
    prep = handoffs.prepare(prepared_path=source, output_dir=tmp_path / "prepare", **FACTS)
    model = {"likelihood": {"kind": "fixed_ou", "tau_hours": {"wrong": 24.0},
                            "site_amplitude_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 3.0}}}
    with pytest.raises(ValueError, match="every coordinate label"):
        handoffs.sample(prep, model=model, output_dir=tmp_path / "unopened")
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("invalid", ["country_grid", "averaging_period"])
def test_full_co2_validates_reporting_inputs_before_inference(tmp_path, monkeypatch, invalid):
    from openghg_inversions.rhime.co2 import execution
    from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map

    _, prepared = _handoff(tmp_path)
    native = xr.ones_like(prepared.basis_functions.flux).assign_attrs(units="1")
    affine = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="memory")
    outputs = {"output_format": "basic", "country_file": str(tmp_path / "countries.nc")}
    if invalid == "country_grid":
        with xr.open_dataset(tmp_path / "countries.nc") as countries:
            mismatched = countries.load().assign_coords(lat=countries.lat + 10)
        mismatched.to_netcdf(tmp_path / "countries.nc")
        message = "align"
    else:
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs,
            site_metadata=prepared.site_metadata.assign(averaging_period=("site", ["1h", "2h"]))))
        outputs["output_format"] = "paris"
        message = "one common observation averaging period"

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid reporting request must fail before sampling.")

    monkeypatch.setattr(execution, "sample_co2_inference", forbidden)
    with pytest.raises(ValueError, match=message):
        run_co2(prepared=prepared, affine_artifact=affine, params={
            "prepared_inputs": FACTS, "likelihood": {"kind": "additive_sigma", "no_model_error": True},
            "sampling": {"sample_prior_predictive": 4}, "outputs": outputs,
        }, output_path=tmp_path / "unopened")
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize("has_prior", [True, False])
def test_affine_replay_checks_saved_draws_and_stays_graph_free(tmp_path, monkeypatch, has_prior):
    from openghg_inversions.rhime.co2 import inference
    from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map
    from test_co2_affine_output import _prepared

    prepared, trace = _components_fixture()
    _, native = _prepared()
    affine = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="memory")
    prep = handoffs.save_preparation(prepared, affine_artifact=affine, output_dir=tmp_path / "prepare", **FACTS)
    if not has_prior:
        del trace["prior"]
    samples = handoffs.save_samples(prep, trace, actual_inference={"sample_prior_predictive": 4},
        preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample")

    def forbidden(*args, **kwargs):
        pytest.fail("Replay must not construct a graph or sampler.")

    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    monkeypatch.setattr(inference, "build_co2_inference", forbidden)
    monkeypatch.setattr(inference, "sample_co2_inference", forbidden)
    output = Co2OutputOptions(output_format="basic", output_path=tmp_path / "replay")
    if not has_prior:
        with pytest.raises(ValueError, match="require prior flux_scale draws"):
            handoffs.postprocess(samples, output=output)
        assert not (tmp_path / "replay").exists()
    else:
        result = handoffs.postprocess(samples, output=output)
        assert "flux_total_prior_mean" in result.outputs["native_flux"]
        assert result.start_date == FACTS["start_date"]
        assert result.end_date == FACTS["end_date"]
        assert result.domain == FACTS["domain"]
    none = handoffs.postprocess(samples, output=Co2OutputOptions(output_format="none", output_path=tmp_path / "none"))
    assert none.bound is not None
    assert none.outputs == {}
    assert not (tmp_path / "none").exists()


def test_co2_preparation_allows_unknown_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from openghg_inversions import _provenance

    monkeypatch.setattr(_provenance, "__file__", str(tmp_path / "openghg_inversions" / "_provenance.py"))
    monkeypatch.setattr(_provenance.metadata, "distribution",
        lambda name: SimpleNamespace(version="0.7.3", read_text=lambda name: None))
    canonical = _canonical_inputs()
    manifest = handoffs.prepare(canonical_inputs=canonical, reduction=_reduction(canonical),
        output_dir=tmp_path / "prepare", **FACTS)
    record = json.loads(manifest.read_text())
    assert record["provenance"] == {"version": "0.7.3", "revision": None, "dirty": None}
