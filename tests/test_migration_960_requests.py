"""Issue #960: independently recorded requests survive compatibility removal.

The plain original files and expected.json are the durable migration evidence.
The retired route is recorded in the pinned archive; modern checks use
canonical settings while preserving plain cases for later #949 converter checks.
These checks stop before acquisition, materialization or inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import xarray as xr

from openghg_inversions import cli
from openghg_inversions.inference import sampling
from openghg_inversions.rhime import RhimeConfig
from openghg_inversions.rhime import standard
from openghg_inversions.rhime.ini import read_rhime_ini
from openghg_inversions.rhime.outputs import _define_output_filename, _resolve_output_path
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.rhime.specs import RhimeOutputSpec


FIXTURES = Path(__file__).parent / "fixtures" / "migration_960" / "requests"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


def _assert_fields(actual, expected):
    assert {name: getattr(actual, name) for name in expected} == expected


def test_recorded_modern_defaults_before_any_scientific_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RhimeConfig.from_params(load_params(config_file=FIXTURES / "modern.ini"), multisector=False)
    _assert_fields(config.sampler, EXPECTED["default_sampler"])
    _assert_fields(config.data, EXPECTED["default_data"])
    _assert_fields(config.output, EXPECTED["modern_output"])
    science = EXPECTED["default_science"]
    assert config.model.sectors[0].x_prior == science["x_prior"]
    assert config.model.bc_prior == science["bc_prior"]
    _assert_fields(config.model, {key: science[key] for key in ("add_offset", "aggregation_error_mode")})
    _assert_fields(config.model.likelihood, {key: science[key] for key in (
        "sigma_prior", "sigma_freq", "sigma_per_site", "sigma_freq_anchor", "pollution_events_from_obs", "power",
    )})


def test_explicit_archived_scientific_settings_on_canonical_inputs():
    """Preserve settings coverage without executing a retired translator."""
    expected = EXPECTED["explicit_wrapper"]
    params = read_rhime_ini(FIXTURES / "modern.ini")
    params.update(expected["sampler"])
    params.update({name: expected[name] for name in (
        "x_prior", "bc_prior", "sigma_prior", "sigma_freq", "sigma_per_site",
        "sigma_freq_anchor", "pollution_events_from_obs", "power", "min_error",
    )})
    config = RhimeConfig.from_params(params, multisector=False)
    _assert_fields(config.sampler, expected["sampler"])
    assert config.model.sectors[0].x_prior == expected["x_prior"]
    assert config.model.bc_prior == expected["bc_prior"]
    _assert_fields(config.model.likelihood, {name: expected[name] for name in (
        "sigma_prior", "sigma_freq", "sigma_per_site", "sigma_freq_anchor", "pollution_events_from_obs", "power",
    )})
    assert config.data.min_error == expected["min_error"]


def test_draws_are_post_tuning_then_burn_is_removed_per_chain(monkeypatch):
    """A synthetic trace checks indexing; pm.sample never performs inference."""
    import pymc as pm

    called = {}

    def sample(**kwargs):
        called.update(kwargs)
        return xr.DataTree.from_dict({"posterior": xr.Dataset(
            {"x": (("chain", "draw"), [[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16]])},
            coords={"chain": [0, 1], "draw": list(range(7))},
        )})

    monkeypatch.setattr(pm, "sample", sample)
    sampler = sampling.RhimeSampler(
        draws=7, burn=2, tune=3, chains=2, sample_prior_predictive=False, sample_posterior_predictive=False,
    )
    trace = sampler.sample(pm.Model())
    assert {name: called[name] for name in ("draws", "tune", "chains")} == {"draws": 7, "tune": 3, "chains": 2}
    assert trace["posterior"]["x"].values.tolist() == [[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]
    assert trace["posterior"].coords["draw"].values.tolist() == [0, 1, 2, 3, 4]
    assert trace["posterior"].attrs["burn"] == 2


def test_recorded_ini_decoding_is_case_sensitive_and_first_section_wins():
    assert read_rhime_ini(FIXTURES / "decoding.ini") == EXPECTED["decoded"]


@pytest.mark.parametrize("source_kind", ["ini", "json"])
def test_canonical_overrides_preserve_winning_values(tmp_path, source_kind):
    path = tmp_path / f"request.{source_kind}"
    path.write_text('{"output_name": "file"}' if source_kind == "json" else "[RUN]\noutput_name = 'file'\n")
    source = {"params_file" if source_kind == "json" else "config_file": path}
    assert load_params(**source, overrides={"output_name": "override"})["output"] == {"output_name": "override"}


@pytest.mark.parametrize("old", [
    "outputpath", "outputname", "xprior", "bcprior", "sigprior", "offsetprior",
    "emissions_name", "outer_region_definition_file", "emissions_store", "emissions_domain",
    "inner_emissions_store", "inner_emissions_domain",
])
def test_recorded_runtime_aliases_are_now_unsupported(old):
    with pytest.raises(ValueError, match="Unsupported RHIME.*" + old):
        load_params(overrides={old: "historical"})


@pytest.mark.parametrize("value", [False, True, None, 0, "false"])
def test_tracer_occurrence_is_unsupported_regardless_of_value(value):
    with pytest.raises(ValueError, match="Unsupported RHIME.*use_tracer"):
        load_params(overrides={"use_tracer": value})


def test_request_source_exclusivity_unknown_case_and_no_environment_fallback(monkeypatch):
    monkeypatch.setenv("CONFIG_FILE", str(FIXTURES / "modern.ini"))
    assert load_params() == {"data": {}, "model": {}, "sampler": {}, "output": {}}
    with pytest.raises(ValueError, match="at most one"):
        load_params(config_file=FIXTURES / "modern.ini", params_file=FIXTURES / "explicit-wrapper.json")
    with pytest.raises(ValueError, match="Unsupported RHIME.*OutputName"):
        load_params(overrides={"OutputName": "case-sensitive"})


def test_cli_json_overrides_dates_and_output_flags_before_file_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []

    def inspect_request(*, config_file, **kwargs):
        captured.append(RhimeConfig.from_params(load_params(config_file=config_file, overrides=kwargs), multisector=False))

    monkeypatch.setattr(standard, "run_standard", inspect_request)
    cli.main([
        "run", "2020-01-01", "2022-02-01", "-c", str(FIXTURES / "modern.ini"),
        "--output-path", "flag", "--kwargs", '{"start_date":"2021-01-01","output_path":"json"}',
    ])
    config, = captured
    assert (config.data.start_date, config.data.end_date) == ("2021-01-01", "2022-02-01")
    assert config.output.output_path == tmp_path / "json"


@pytest.mark.parametrize("output_format, save", [("none", False), ("inv_out", True), ("legacy", False), ("basic", False), ("paris", False)])
def test_omitted_save_policy_and_explicit_save_paths(tmp_path, output_format, save):
    spec = RhimeOutputSpec.from_params({"output_format": output_format, "output_path": tmp_path}, multisector=False)
    assert spec.save_inversion_output is save
    explicit = tmp_path / "exact.nc"
    assert _resolve_output_path(explicit, tmp_path / "ignored", "ignored.nc") == explicit
    assert _resolve_output_path(False, tmp_path, "unused.nc") is None
    assert _resolve_output_path(True, tmp_path, "trace.nc") == tmp_path / "trace.nc"


@pytest.mark.parametrize("convention", ["rhime", "legacy"])
def test_filename_policy_is_independently_recorded(tmp_path, convention):
    assert _define_output_filename(
        tmp_path, "ch4", "EUROPE", "migration", "2019-01-01", filename_convention=convention,
    ) == tmp_path / EXPECTED["filenames"][convention]


def test_current_python_name_removal_and_explicit_likelihood_requirement(monkeypatch):
    import openghg_inversions.rhime as rhime

    with pytest.raises(ImportError, match="removed in 0.8.*run_standard.*InversionOutput"):
        getattr(rhime, "run_rhime")
    monkeypatch.setattr(standard, "acquire_merged_data", lambda *args, **kwargs: pytest.fail("must not acquire"))
    params = read_rhime_ini(FIXTURES / "modern.ini")
    params.pop("mismatch_model")
    with pytest.raises(ValueError, match="requires a built-in or custom likelihood"):
        standard.run_standard(**params)


def test_installed_run_rhime_rejects_nit_before_acquisition(tmp_path):
    executable = shutil.which("rhime", path=str(Path(sys.executable).parent))
    assert executable is not None, "Run from the installed project environment (uv sync --locked)."
    path = tmp_path / "nit.ini"
    path.write_text((FIXTURES / "modern.ini").read_text() + "\nnit = 7\n")
    result = subprocess.run([executable, "run", "-c", str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Unsupported RHIME parameter(s): ['nit']" in result.stderr


def test_modern_summary_pools_the_archived_chain_witness():
    """This scientific expectation needs no HBMCMC import after removal."""
    from openghg_inversions.postprocessing.inversion_output import merge_trace_groups
    from openghg_inversions.postprocessing.stats import calculate_stats

    historical = json.loads((FIXTURES.parent / "historical" / "expected.json").read_text())
    witness = historical["scientific_cases"]["chain_summary"]
    trace = xr.DataTree.from_dict({
        "posterior": xr.Dataset({"x": (("chain", "draw"), witness["input_chains"])})
    })
    merged = merge_trace_groups(trace)
    assert merged.sizes["chain"] == 2
    actual = calculate_stats(merged, stats=["mean"])
    assert actual.x_posterior_mean.item() == witness["modern_all_chain_mean"]
    assert actual.x_posterior_mean.item() != witness["released_chain0_mean"]
