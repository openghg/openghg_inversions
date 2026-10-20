"""Installed CO2 stage acceptance using package-built synthetic scientific inputs."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.affine_flux_map_io import save as save_affine
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.cli import main
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs
from openghg_inversions.rhime.co2.co2_affine_output import (
    prepared_inputs_content_id,
    produce_bucket_affine_flux_map,
)
from openghg_inversions.serialization import load_trace
from test_co2_preparation import _canonical_inputs, _reduction


def _handoff(directory: Path) -> tuple[Path, Co2PreparedInputs]:
    """Prepare a signed-flux case with a fixed outer state and explicit baselines."""
    original = _canonical_inputs()
    grid = {"lat": [50.0, 51.0], "lon": [-2.0, -1.0]}
    basis = xr.DataArray([[1, 2], [1, 2]], dims=("lat", "lon"), coords=grid)
    flux = xr.DataArray(
        [[-2.0, 3.0], [-1.0, 4.0]], dims=basis.dims, coords=grid, attrs={"units": "mol m-2 s-1"}
    )
    inputs = original.inv_inputs.assign_coords(basis_group=("region", ["inner", "outer"]))
    inputs["H_bc"] = xr.DataArray(
        [[0.1], [0.2], [0.15]],
        dims=("nmeasure", "bc_region"),
        coords={"bc_region": ["background"]},
        attrs={"units": "ppm"},
    )
    canonical = RhimePreparedInputs(
        inputs,
        BasisFunctions.from_flat_basis(basis, flux, operator_kwargs={"state_dim": "region"}),
        original.site_metadata.assign(averaging_period=("site", ["1h", "1h"])),
    )
    state = canonical.basis_functions.operator.basis_matrix.region.values
    canonical = RhimePreparedInputs(
        canonical.inv_inputs.assign_coords(region=state), canonical.basis_functions, canonical.site_metadata
    )
    reduction = _reduction(canonical)
    reduction = replace(
        reduction,
        retained_mean=reduction.retained_mean.assign_coords(state=state),
        retained_covariance=reduction.retained_covariance.assign_coords(state=state, state_cov=state),
        effective_observation_operator=reduction.effective_observation_operator.assign_coords(state=state),
    )
    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=1)
    path = directory / "source-prepared.nc"
    prepared.save(path)
    native_mean = xr.ones_like(flux).assign_attrs(units="1")
    save_affine(
        produce_bucket_affine_flux_map(
            prepared, native_mean, prepared_inputs_id=prepared_inputs_content_id(path)
        ),
        directory / "affine.nc",
    )
    xr.Dataset(
        {
            "country": (("lat", "lon"), [[0, 1], [1, 0]]),
            "name": ("ncountries", ["FRANCE", "GERMANY"]),
        },
        coords=grid,
    ).to_netcdf(directory / "countries.nc")
    return path, prepared


def _configuration(directory: Path, variant: str) -> Path:
    preparation_reconstruction = 'reconstruction_path = "affine.nc"' if variant == "ordinary" else ""
    output_reconstruction = 'reconstruction_path = "affine.nc"' if variant != "ordinary" else ""
    likelihood = (
        'kind = "additive_sigma"\nno_model_error = true\nfixed_model_mismatch = 0.0'
        if variant == "ordinary"
        else 'kind = "fixed_ou"\ntau_hours = 24.0\nsite_amplitude_prior_scale = 0.3'
    )
    path = directory / f"{variant}.toml"
    path.write_text(
        f'''format_version = 1
recipe = "co2"
variant = "{variant}"
[prepared_inputs]
path = "source-prepared.nc"
start_date = "2020-01-01"
end_date = "2020-02-01"
domain = "EUROPE"
{preparation_reconstruction}
[likelihood]
{likelihood}
[model.boundary]
enabled = true
prior = {{ pdf = "normal", mu = 1.0, sigma = 0.1 }}
[model.offset]
prior = {{ pdf = "normal", mu = 0.0, sigma = 0.1 }}
per_site = true
drop_first = false
[sampling]
draws = 12
tune = 12
chains = 2
nuts_sampler = "pymc"
progressbar = false
random_seed = 126
sample_prior_predictive = 8
sample_posterior_predictive = true
[outputs]
output_format = "basic"
output_name = "co2-stage-test"
{output_reconstruction}
country_file = "countries.nc"
''',
        encoding="utf-8",
    )
    return path


def _cli(*arguments: str | Path, success: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute the installed console script without replacing model or sampler code."""
    executable = Path(sys.executable).with_name("openghg-inversions")
    environment = os.environ.copy()
    environment.pop("RUN_ROOT", None)
    started = perf_counter()
    result = subprocess.run(
        [str(executable), *map(str, arguments)],
        text=True,
        capture_output=True,
        env=environment,
        timeout=240,
    )
    print(f"Installed CLI {arguments[0]}: {perf_counter() - started:.2f}s")
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


def _in_process_cli(*arguments: str | Path) -> None:
    """Reuse the CLI parser and handlers for checks on the already sampled handoff."""
    started = perf_counter()
    main(list(map(str, arguments)))
    print(f"In-process CLI {arguments[0]}: {perf_counter() - started:.2f}s")


@pytest.mark.parametrize("variant", ["ordinary", "cached_fixed_ou"])
def test_installed_co2_stages_preserve_scientific_and_file_contracts(
    tmp_path: Path, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUN_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    source, original = _handoff(tmp_path)
    config = _configuration(tmp_path, variant)
    common = ["--model", "co2", "--config", str(config)]
    prepared = source if variant == "ordinary" else tmp_path / "prepare" / "prepared-inputs.nc"
    preparation = tmp_path / "prepare" / "prepare-manifest.json"
    posterior = tmp_path / "sample" / "posterior.nc"
    sampling = tmp_path / "sample" / "sample-manifest.json"
    _cli("prepare", *common, "--output-dir", tmp_path / "prepare")
    if variant == "ordinary":
        from openghg_inversions.rhime.co2.handoffs import load_preparation

        loaded = load_preparation(preparation)
        assert loaded.payload == source
        assert loaded.affine_payload == tmp_path / "affine.nc"
        assert not (tmp_path / "prepare" / "prepared-inputs.nc").exists()
        assert not (tmp_path / "prepare" / "affine-flux-map.nc").exists()
    else:
        assert prepared.read_bytes() == source.read_bytes()
    restored = Co2PreparedInputs.load(prepared)
    xr.testing.assert_identical(restored.inv_inputs, original.inv_inputs)
    handoff = ["--preparation-manifest", str(preparation)]
    if variant == "cached_fixed_ou":
        handoff = ["--prepared-inputs", str(prepared), *handoff]
    _cli(
        "prior-predictive", *common, *handoff, "--draws", "8", "--strict", "--output-dir", tmp_path / "prior"
    )
    prior = json.loads((tmp_path / "prior" / "prior-predictive-readiness.json").read_text())
    assert prior["status"] == "pass"
    prior_trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    assert prior_trace["prior_predictive"].indexes["nmeasure"].equals(original.inv_inputs.indexes["nmeasure"])
    assert prior_trace["prior"].indexes["region"].equals(original.inv_inputs.indexes["region"])
    _cli("sample", *common, *handoff, "--output-dir", tmp_path / "sample")
    trace = load_trace(posterior)
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    saved_prior = trace["prior_predictive"].to_dataset()[roles["concentration"]]
    assert saved_prior.sizes["draw"] == 8
    assert saved_prior.indexes["nmeasure"].equals(original.inv_inputs.indexes["nmeasure"])
    assert np.isfinite(saved_prior).all()
    draws = trace["posterior"].to_dataset()
    state = draws[roles["flux_scale"]]
    np.testing.assert_allclose(state.sel(region=1), 0.9)
    expected_flux = (
        xr.dot(original.inv_inputs.H, state, dim="region") + original.inv_inputs.fixed_prior_contribution
    )
    pollution = draws[roles["pollution_concentration"]]
    np.testing.assert_allclose(pollution, expected_flux.transpose(*pollution.dims), rtol=2e-6)
    expected_total = (
        expected_flux + draws[roles["boundary_concentration"]] + draws[roles["offset_concentration"]]
    )
    np.testing.assert_allclose(
        draws[roles["model_mean"]], expected_total.transpose(*draws[roles["model_mean"]].dims), rtol=2e-6
    )
    assert draws[roles["model_mean"]].attrs["units"] == "ppm"
    assert state.attrs["units"] == "1"
    assert draws.indexes["nmeasure"].equals(original.inv_inputs.indexes["nmeasure"])
    assert "posterior_predictive" in trace.children
    sample_manifest = json.loads(sampling.read_text())
    prepare_manifest = json.loads(preparation.read_text())
    provenance_key = "provenance" if variant == "ordinary" else "ogi"
    assert set(prepare_manifest[provenance_key]) == {"version", "revision", "dirty"}
    assert prepare_manifest[provenance_key]["version"]
    revision = prepare_manifest[provenance_key]["revision"]
    assert len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)
    assert sample_manifest[provenance_key] == prepare_manifest[provenance_key]
    if variant == "ordinary":
        assert sample_manifest["preparation_generation"] == prepare_manifest["generation"]
        assert prepare_manifest["start_date"] == "2020-01-01"
        assert prepare_manifest["end_date"] == "2020-02-01"
    else:
        assert sample_manifest["configuration_identity"] == prepare_manifest["configuration_identity"]
    assert "artifact_identities" not in prepare_manifest
    assert "artifact_identities" not in sample_manifest
    assert "input_identities" not in sample_manifest
    diagnostic_inputs = [] if variant == "ordinary" else ["--posterior", posterior]
    _in_process_cli(
        "diagnose", *diagnostic_inputs,
        "--sample-manifest",
        sampling,
        "--output-dir",
        tmp_path / "diagnose",
    )
    diagnostics = json.loads((tmp_path / "diagnose" / "sampler-convergence.json").read_text())
    assert diagnostics["measured_values"]["chains"] == 2
    assert diagnostics["measured_values"]["draws_per_chain"] == 12
    if variant == "cached_fixed_ou":
        diagnose_manifest = json.loads((tmp_path / "diagnose" / "diagnose-manifest.json").read_text())
        assert diagnose_manifest["ogi"]["revision"] == revision
        assert diagnose_manifest["configuration_identity"] == sample_manifest["configuration_identity"]
        assert "input_identities" not in diagnose_manifest
        assert "artifact_identities" not in diagnose_manifest
    postprocess_inputs = [] if variant == "ordinary" else [*handoff, "--posterior", posterior]
    def product_path(directory, name):
        filename = f"co2-stage-test_{name}_co2_EUROPE_2020-01-01.nc" if variant == "ordinary" else f"{name}.nc"
        return tmp_path / directory / filename
    _in_process_cli(
        "postprocess",
        *common,
        *postprocess_inputs,
        "--sample-manifest",
        sampling,
        "--output-dir",
        tmp_path / "postprocess",
    )
    if variant == "cached_fixed_ou":
        post = json.loads((tmp_path / "postprocess" / "postprocess-manifest.json").read_text())
        assert "input_identities" not in post
        assert "artifact_identities" not in post
    assert product_path("postprocess", "basic").is_file()
    for name in ("native_flux", "country_flux"):
        with xr.open_dataset(product_path("postprocess", name)) as product:
            assert product.attrs["uncertainty_scope"] == "retained_state_conditional"
            assert product.attrs["prepared_inputs_id"] == prepared_inputs_content_id(source)
            assert all(
                value.attrs["uncertainty_scope"] == "retained_state_conditional"
                for value in product.data_vars.values()
            )
    if variant == "ordinary":
        from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations

        components = reconstruct_co2_concentrations(trace, restored)
    else:
        with xr.open_dataset(tmp_path / "postprocess" / "concentration_components.nc") as saved_components:
            components = saved_components.load()
    assert "predictive_posterior" in components
    assert components.predictive_prior.sizes["prior_draw"] == 8
    assert "active_flux_scale_posterior" in components
    assert "residual_posterior" in components
    assert components["predictive_posterior"].attrs["units"] == "ppm"
    # Installed stages above cover process boundaries; replay uses the same CLI in-process.
    # Replaying saved inputs and posterior must produce identical scientific products.
    _in_process_cli(
        "postprocess",
        *common,
        *postprocess_inputs,
        "--sample-manifest",
        sampling,
        "--output-dir",
        tmp_path / "replay",
    )
    replay_products = ["basic", "native_flux", "country_flux"]
    if variant == "cached_fixed_ou":
        replay_products.append("concentration_components")
    for name in replay_products:
        with (
            xr.open_dataset(product_path("postprocess", name)) as result,
            xr.open_dataset(product_path("replay", name)) as replay,
        ):
            xr.testing.assert_identical(result, replay)
    # PARIS adaptation reuses the exact sampled handoff, with no second sampler run.
    outputs = {
        "outputs": {
            "output_format": "paris",
            "output_name": "co2-stage-test",
            "country_file": str(tmp_path / "countries.nc"),
        }
    }
    if variant == "cached_fixed_ou":
        outputs["outputs"]["reconstruction_path"] = str(tmp_path / "affine.nc")
    _in_process_cli(
        "postprocess",
        *common,
        *postprocess_inputs,
        "--sample-manifest",
        sampling,
        "--kwargs",
        json.dumps(outputs),
        "--output-dir",
        tmp_path / "paris",
    )
    for name in ("paris_concentration", "paris_flux"):
        path = product_path("paris", name)
        assert path.is_file()
        with NetCDFDataset(path) as product:
            if name == "paris_flux":
                assert product.getncattr("uncertainty_scope") == "retained_state_conditional"
            time_variable = product.variables["time"]
            assert time_variable.getncattr("units") == "days since 1970-01-01 00:00:00"
            assert time_variable.getncattr("calendar") == "proleptic_gregorian"
            assert time_variable.getncattr("bounds") == "time_bnds"
            # Latest templates leave bounds attributes empty; CF inherits time metadata.
            assert "units" not in product.variables["time_bnds"].ncattrs()
            assert "calendar" not in product.variables["time_bnds"].ncattrs()
            time = time_variable[:]
            bounds = product.variables["time_bnds"][:]
            np.testing.assert_allclose(time, bounds.mean(axis=1))
    # A different preparation requires its associated affine reconstruction.
    changed = restored.inv_inputs.copy(deep=True)
    changed["mf"].data = changed["mf"].data + 1.0
    unrelated = Co2PreparedInputs(
        RhimePreparedInputs(changed, restored.basis_functions, restored.site_metadata),
        restored.aggregation_error_mode,
        restored.provenance,
    )
    unrelated.save(tmp_path / "unrelated.nc")
    rejected_inputs = ["--prepared-inputs", tmp_path / "unrelated.nc"]
    rejection = (
        "Ordinary CO2 phases take one manifest" if variant == "ordinary"
        else "Affine flux prepared-input content identity does not match"
    )
    with pytest.raises(ValueError, match=rejection):
        _in_process_cli(
            "sample",
            *common,
            *rejected_inputs,
            "--preparation-manifest",
            preparation,
            "--output-dir",
            tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected" / "posterior.nc").exists()


@pytest.mark.parametrize("variant", ["ordinary", "cached_fixed_ou"])
def test_installed_co2_rejects_unsupported_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _handoff(tmp_path)
    config = _configuration(tmp_path, variant)
    config.write_text(config.read_text().replace('output_format = "basic"', 'output_format = "inv_out"'))
    destination = tmp_path / "unsupported"
    arguments = (
        ["run", "--model", "co2", "--config", config, "--output-path", destination]
        if variant == "ordinary"
        else ["prepare", "--model", "co2", "--config", config, "--output-dir", destination]
    )
    failure = _cli(*arguments, success=False)
    assert "unsupported" in failure.stderr or "support" in failure.stderr
    assert not destination.exists()


def test_installed_configured_co2_full_execution_writes_selected_products(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _handoff(tmp_path)
    config = _configuration(tmp_path, "ordinary")
    config.write_text(config.read_text().replace('[outputs]\n', '[outputs]\nsave_trace = true\n'))
    destination = tmp_path / "full-products"
    _cli("run", "--model", "co2", "--config", config, "--output-path", destination)
    for name in ("basic", "native_flux", "country_flux"):
        path = destination / f"co2-stage-test_{name}_co2_EUROPE_2020-01-01.nc"
        assert path.is_file()
        with xr.open_dataset(path) as saved:
            assert saved.data_vars
            if name != "basic":
                assert saved.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert not list(destination.glob("*-manifest.json"))
    trace = load_trace(destination / "co2-stage-test2020-01-01_trace.nc")
    assert trace.posterior.sizes["draw"] == 12
    assert trace.posterior.sizes["chain"] == 2
    assert trace.prior.sizes["draw"] == 8


def test_co2_overrides_replace_sections_before_file_path_conversion(tmp_path, monkeypatch):
    from openghg_inversions.rhime.stages import load_stage_params

    file_dir = tmp_path / "config"
    file_dir.mkdir()
    source = file_dir / "run.json"
    source.write_text(json.dumps({
        "prepared_inputs": {"path": "prepared.nc"},
        "outputs": {"country_file": None, "reconstruction_path": None},
    }))
    monkeypatch.chdir(tmp_path)
    overrides = {"outputs": {"country_file": "countries.nc"}}

    params = load_stage_params(params_file=source, overrides=overrides, model="co2")

    assert params["prepared_inputs"] == {"path": str(file_dir / "prepared.nc")}
    assert params["outputs"] == {"country_file": str(tmp_path / "countries.nc")}
    assert overrides == {"outputs": {"country_file": "countries.nc"}}
