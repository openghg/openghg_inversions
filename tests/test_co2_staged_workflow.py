"""CO2 checkpoint/output adapters and bounded installed acceptance."""

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


def _configuration(directory: Path, likelihood_kind: str) -> Path:
    preparation_reconstruction = 'reconstruction_path = "affine.nc"'
    likelihood = (
        'kind = "additive_sigma"\nno_model_error = true\nfixed_model_mismatch = 0.0'
        if likelihood_kind == "additive_sigma"
        else 'kind = "fixed_ou"\ntau_hours = 24.0\nsite_amplitude_prior = {pdf = "gamma", alpha = 2.0, beta = 6.0}'
    )
    path = directory / f"{likelihood_kind}.toml"
    path.write_text(
        f'''format_version = 1
recipe = "co2"
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
progressbar = false
random_seed = 126
sample_prior_predictive = 8
sample_posterior_predictive = true
[outputs]
output_format = "basic"
output_name = "co2-stage-test"
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


def test_installed_co2_stages_preserve_scientific_and_file_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep installed cached inference, scientific handoff and two-chain diagnosis."""
    monkeypatch.delenv("RUN_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    source, _ = _handoff(tmp_path)
    config = _configuration(tmp_path, "fixed_ou")
    common = ["--model", "co2", "--config", str(config)]
    preparation = tmp_path / "prepare" / "prepare-manifest.json"
    sampling = tmp_path / "sample" / "sample-manifest.json"
    _cli("prepare", *common, "--output-dir", tmp_path / "prepare")
    prepare_manifest = json.loads(preparation.read_text())
    assert (preparation.parent / prepare_manifest["prepared_inputs"]).resolve() == source
    assert (preparation.parent / prepare_manifest["affine_reconstruction"]).resolve() == tmp_path / "affine.nc"
    assert not (tmp_path / "prepare" / "prepared-inputs.nc").exists()
    assert not (tmp_path / "prepare" / "affine-flux-map.nc").exists()
    handoff = ["--preparation-manifest", str(preparation)]
    _cli(
        "prior-predictive", *common, *handoff, "--draws", "8", "--strict", "--output-dir", tmp_path / "prior"
    )
    prior = json.loads((tmp_path / "prior" / "prior-predictive-readiness.json").read_text())
    assert prior["status"] == "pass"
    assert prior["measured_values"]["draws"] == 8
    assert (tmp_path / "prior" / "prior-predictive.nc").is_file()
    _cli("sample", *common, *handoff, "--output-dir", tmp_path / "sample")
    trace = load_trace(tmp_path / "sample" / "posterior.nc")
    assert trace.posterior.sizes["chain"] == 2
    assert trace.posterior.sizes["draw"] == 12
    assert trace.prior.sizes["draw"] == 8
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    np.testing.assert_allclose(trace.posterior[roles["flux_scale"]].sel(region=1), 0.9)
    saved_prior = trace.prior_predictive[roles["concentration"]]
    assert saved_prior.sizes["draw"] == 8
    assert np.isfinite(saved_prior).all()
    assert "posterior_predictive" in trace.children
    sample_manifest = json.loads(sampling.read_text())
    provenance = prepare_manifest["provenance"]
    assert set(provenance) == {"version", "revision", "dirty"}
    assert provenance["version"]
    revision = provenance["revision"]
    assert len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)
    assert sample_manifest["provenance"] == provenance
    assert sample_manifest["preparation_generation"] == prepare_manifest["generation"]
    assert prepare_manifest["start_date"] == "2020-01-01"
    assert prepare_manifest["end_date"] == "2020-02-01"
    assert "artifact_identities" not in prepare_manifest
    assert "artifact_identities" not in sample_manifest
    assert "input_identities" not in sample_manifest
    _in_process_cli(
        "diagnose", "--sample-manifest", sampling, "--output-dir", tmp_path / "diagnose",
    )
    diagnostics = json.loads((tmp_path / "diagnose" / "sampler-convergence.json").read_text())
    assert diagnostics["measured_values"]["chains"] == 2
    assert diagnostics["measured_values"]["draws_per_chain"] == 12
    _in_process_cli(
        "postprocess", *common, "--sample-manifest", sampling, "--output-dir", tmp_path / "postprocess",
    )
    for name in ("basic", "native_flux", "country_flux"):
        assert (tmp_path / "postprocess" / f"co2-stage-test_{name}_co2_EUROPE_2020-01-01.nc").is_file()


def test_installed_co2_rejects_unsupported_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed command reports the shared output-policy failure."""
    monkeypatch.chdir(tmp_path)
    config = _configuration(tmp_path, "additive_sigma")
    config.write_text(config.read_text().replace('output_format = "basic"', 'output_format = "inv_out"'))
    destination = tmp_path / "unsupported"
    arguments = ["run", "--model", "co2", "--config", config, "--output-path", destination]
    failure = _cli(*arguments, success=False)
    assert "unsupported" in failure.stderr or "support" in failure.stderr
    assert not destination.exists()


def test_installed_configured_co2_full_execution_writes_selected_products(tmp_path, monkeypatch):
    """Keep one installed full-run acceptance; shared inference owns likelihood choices."""
    monkeypatch.chdir(tmp_path)
    _handoff(tmp_path)
    config = _configuration(tmp_path, "additive_sigma")
    config.write_text(config.read_text().replace('[outputs]\n', '[outputs]\nsave_trace = true\n'))
    destination = tmp_path / "full-products"
    _cli("run", "--model", "co2", "--config", config, "--output-path", destination)
    for name in ("basic", "native_flux", "country_flux"):
        path = destination / f"co2-stage-test_{name}_co2_EUROPE_2020-01-01.nc"
        assert path.is_file()
    with xr.open_dataset(destination / "co2-stage-test_basic_co2_EUROPE_2020-01-01.nc") as saved:
        assert saved.data_vars
    assert not list(destination.glob("*-manifest.json"))
    trace = load_trace(destination / "co2-stage-test2020-01-01_trace.nc")
    assert trace.posterior.sizes["draw"] == 12
    assert trace.posterior.sizes["chain"] == 2
    assert trace.prior.sizes["draw"] == 8


def test_saved_co2_product_adapter_preserves_in_memory_products_and_time_codecs(tmp_path, monkeypatch):
    """Small valid traces own replay, product files and exact CF time encoding."""
    from openghg_inversions.inference.sampling import RhimeSampler
    from openghg_inversions.postprocessing import make_paris_outputs
    from openghg_inversions.rhime.co2 import handoffs, inference
    from openghg_inversions.rhime.co2.co2_affine_output import load_and_bind_affine_flux_map
    from openghg_inversions.rhime.co2.co2_outputs import (
        Co2OutputOptions, calculate_co2_products, make_co2_output, validate_co2_product_request,
    )
    from test_co2_affine_output import _prepared
    from test_co2_handoffs import _components_fixture

    monkeypatch.setattr(make_paris_outputs, "timestamp_now", lambda: "2020-02-01T00:00:00+00:00")
    prepared, trace = _components_fixture()
    _, native_mean = _prepared()
    # Country-cell areas require latitude spacing, so retain two grid rows.
    basis = BasisFunctions.from_flat_basis(
        prepared.basis_functions.flat_basis().reindex(lat=[50.0, 51.0], method="nearest"),
        prepared.basis_functions.flux.reindex(lat=[50.0, 51.0], method="nearest"),
        operator_kwargs={"state_dim": "region"},
    )
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, basis_functions=basis))
    native_mean = native_mean.reindex(lat=basis.flux.lat, method="nearest")
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="in-memory")
    preparation = handoffs.save_preparation(
        prepared, affine_artifact=affine, output_dir=tmp_path / "preparation",
        start_date="2020-01-01", end_date="2020-02-01", domain="EUROPE",
    )
    loaded = handoffs.load_preparation(preparation)
    bound = load_and_bind_affine_flux_map(loaded.affine_payload, loaded.payload)
    country_path = tmp_path / "countries.nc"
    xr.Dataset(
        {"country": (("lat", "lon"), [[0, 1], [1, 0]]), "name": ("ncountries", ["FRANCE", "GERMANY"])},
        coords={"lat": prepared.basis_functions.flux.lat, "lon": prepared.basis_functions.flux.lon},
    ).to_netcdf(country_path)
    output = Co2OutputOptions(
        output_format="paris", output_name="co2-adapter", country_file=str(country_path),
        output_path=tmp_path / "products",
    )
    expected = make_co2_output(
        prepared=prepared, trace=trace, bound=bound,
        start_date="2020-01-01", end_date="2020-02-01", domain="EUROPE",
    )
    countries = validate_co2_product_request(output, prepared=prepared, bound=bound)
    calculate_co2_products(result=expected, prepared=prepared, bound=bound, countries=countries, output=output)
    samples = handoffs.save_samples(
        preparation, trace, actual_inference={"draws": 3, "chains": 1},
        preparation_generation=loaded.generation, output_dir=tmp_path / "samples",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Saved-product adaptation must not construct a graph or sampler")

    monkeypatch.setattr(handoffs.pm.Model, "__init__", forbidden)
    monkeypatch.setattr(handoffs, "build_co2_inference", forbidden)
    monkeypatch.setattr(inference, "build_co2_inference", forbidden)
    monkeypatch.setattr(inference, "sample_rhime_model", forbidden)
    monkeypatch.setattr(handoffs, "sample_co2_inference", forbidden)
    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    result = handoffs.postprocess(samples, output=output)
    xr.testing.assert_identical(result.idata, trace)
    assert result.outputs.keys() == expected.outputs.keys()
    for name in result.outputs:
        xr.testing.assert_identical(result.outputs[name], expected.outputs[name])
    components = result.outputs["concentration_components"]
    assert {"predictive_posterior", "residual_posterior"} <= set(components)
    assert components.predictive_prior.sizes["prior_draw"] == 4
    assert components.predictive_posterior.sizes["draw"] == 3
    assert components.predictive_posterior.attrs["units"] == "ppm"
    for name in ("basic", "native_flux", "country_flux"):
        path = Path(result.output_metadata[f"{name}_path"])
        assert path.is_file()
        if name != "basic":
            with xr.open_dataset(path) as product:
                assert product.attrs["uncertainty_scope"] == "retained_state_conditional"
                assert product.attrs["prepared_inputs_id"] == prepared_inputs_content_id(loaded.payload)
                assert all(value.attrs["uncertainty_scope"] == "retained_state_conditional"
                           for value in product.data_vars.values())
    for name in ("paris_concentration", "paris_flux"):
        with NetCDFDataset(result.output_metadata[f"{name}_path"]) as product:
            if name == "paris_flux":
                assert product.getncattr("uncertainty_scope") == "retained_state_conditional"
            time = product.variables["time"]
            bounds = product.variables["time_bnds"]
            assert time.getncattr("units") == "days since 1970-01-01 00:00:00"
            assert time.getncattr("calendar") == "proleptic_gregorian"
            assert time.getncattr("bounds") == "time_bnds"
            assert "units" not in bounds.ncattrs()
            assert "calendar" not in bounds.ncattrs()
            np.testing.assert_allclose(time[:], bounds[:].mean(axis=1))


def test_co2_sample_cli_adapter_rejects_prepared_input_override(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([
            "sample", "--model", "co2", "--preparation-manifest", str(tmp_path / "prepare.json"),
            "--prepared-inputs", str(tmp_path / "unrelated.nc"), "--output-dir", str(tmp_path / "rejected"),
        ])
    assert error.value.code == 2
    assert "--prepared-inputs" in capsys.readouterr().err
    assert not (tmp_path / "rejected").exists()


def test_co2_overrides_replace_sections_before_file_path_conversion(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2.execution import load_co2_params
    from openghg_inversions.rhime.co2.configuration import resolve_co2_preparation

    file_dir = tmp_path / "config"
    file_dir.mkdir()
    source = file_dir / "run.json"
    source.write_text(json.dumps({
        "prepared_inputs": {"path": "prepared.nc"},
        "outputs": {"country_file": None},
    }))
    monkeypatch.chdir(tmp_path)
    overrides = {"outputs": {"country_file": "countries.nc"}}

    params = load_co2_params(params_file=source, overrides=overrides)

    assert resolve_co2_preparation(params)["prepared_path"] == tmp_path / "prepared.nc"
    assert params["outputs"] == {"country_file": "countries.nc"}
    assert overrides == {"outputs": {"country_file": "countries.nc"}}
