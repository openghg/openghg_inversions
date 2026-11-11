"""Contract tests for file-backed RHIME workflow stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.cli import build_parser, main
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.rhime.stages import (
    CONVERGENCE_CHECK_NAME,
    diagnose_rhime_stage,
)
from openghg_inversions.serialization import save_trace
from tests.helpers import make_trace


def _params(**overrides: Any) -> dict[str, Any]:
    params = {
        "species": "ch4",
        "sites": ["TAC"],
        "averaging_period": ["1h"],
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "end_date": "2019-02-01",
        "output_name": "staged-test",
        "flux_sources": ["synthetic-total"],
        "use_bc": False,
        "mismatch_model": "fixed_error",
        "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
        "draws": 10,
        "tune": 10,
        "chains": 2,
        "output_format": "inv_out",
        "output_path": "/explicitly-replaced-by-stage",
        "save_inversion_output": True,
    }
    params.update(overrides)
    return params


def _prepared() -> PreparedInputs:
    nmeasure = pd.MultiIndex.from_arrays(
        [["TAC", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02"])],
        names=["site", "time"],
    )
    coords = xr.Coordinates.from_pandas_multiindex(nmeasure, "nmeasure")
    inputs = xr.Dataset(
        {
            "H": (("region", "nmeasure"), [[1.0, 0.5]]),
            "mf": ("nmeasure", [1.1, 0.6]),
            "mf_error": ("nmeasure", [0.1, 0.1]),
            "site_indicator": ("nmeasure", [0, 0]),
        },
        coords={"region": [0], **coords},
    )
    inputs["mf"].attrs["units"] = "ppm"
    basis_flat = xr.DataArray(
        [[0]],
        dims=("lat", "lon"),
        coords={"lat": [51.0], "lon": [-2.0]},
    )
    flux = xr.ones_like(basis_flat, dtype=float)
    flux.attrs["units"] = "mol m-2 s-1"
    basis = BasisFunctions.from_flat_basis(
        basis_flat=basis_flat,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
    )
    metadata = xr.Dataset(
        {"averaging_period": ("site", ["1h"])},
        coords={"site": ["TAC"]},
    )
    return PreparedInputs(inv_inputs=inputs, basis_functions=basis, site_metadata=metadata)



def test_stage_parser_requires_explicit_config_source_and_model() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "--params-file",
            "/run/effective.json",
            "--model",
            "standard",
            "--output-dir",
            "/run/outputs/prepare",
        ]
    )

    assert args.params_file == "/run/effective.json"
    assert args.config is None
    assert args.model == "standard"
    assert args.output_dir == "/run/outputs/prepare"


def test_downstream_stage_parser_requires_handoff_manifests() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "sample",
                "--params-file",
                "/run/effective.json",
                "--model",
                "standard",
                "--output-dir",
                "/run/sample",
            ]
        )


def test_stage_config_paths_are_relative_to_the_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "config" / "params.json"
    params_path.parent.mkdir()
    params_path.write_text(
        json.dumps(_params(bc_basis={"paths": ["relative/bc.nc"]}, basis_partition_path="relative/partition.nc",
                           country_file="relative/country.nc")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)

    loaded = load_params(params_file=params_path)

    assert loaded["data"]["bc_basis"] == {"paths": (tmp_path.parent / "relative/bc.nc",)}
    assert loaded["output"]["country_file"] == tmp_path.parent / "relative/country.nc"
    assert loaded["data"]["basis_partition_path"] == tmp_path.parent / "relative/partition.nc"


def test_diagnostics_emit_issue_667_convergence_signals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    posterior = xr.Dataset(
        {
            "x": (("chain", "draw", "region"), rng.normal(size=(4, 500, 2))),
            "epsilon": (("chain", "draw", "nmeasure"), np.ones((4, 500, 20))),
        },
        coords={"chain": range(4), "draw": range(500), "region": ["north", "south"]},
    )
    diverging = np.zeros((4, 500), dtype=bool)
    diverging[2, 10] = True
    idata = make_trace(
        posterior=posterior,
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), diverging)},
            coords={"chain": range(4), "draw": range(500)},
        ),
    )
    idata.attrs["sampler_convergence_variables"] = json.dumps(["x"])
    posterior_path = tmp_path / "posterior.nc"
    save_trace(idata, posterior_path)
    engines: list[str | None] = []
    to_netcdf = xr.Dataset.to_netcdf

    def record_engine(dataset: xr.Dataset, *args: Any, **kwargs: Any):
        engines.append(kwargs.get("engine"))
        return to_netcdf(dataset, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", record_engine)

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["name"] == CONVERGENCE_CHECK_NAME
    assert result["status"] == "fail"
    assert result["measured_values"]["chains"] == 4
    assert result["measured_values"]["assessed_variables"] == ["x"]
    assert result["measured_values"]["draws_per_chain"] == 500
    assert result["measured_values"]["divergences"] == 1
    assert result["measured_values"]["arviz_has_errors"] is True
    assert result["thresholds"] == {"policy": "arviz.diagnose defaults"}
    assert Path(result["artifact_paths"][0]).is_file()
    assert engines == [None]


@pytest.mark.parametrize(("strict", "raises"), [(False, False), (True, True)])
def test_diagnose_cli_prints_json_and_strict_failure_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    strict: bool,
    raises: bool,
) -> None:
    """The CLI always prints CheckResult JSON and opts into failure exits."""
    result = {
        "schema_version": 1,
        "name": "sampler-convergence",
        "status": "fail",
        "producer": "openghg_inversions",
        "measured_values": {},
        "thresholds": {},
        "message": "failed",
        "artifact_paths": [],
        "stage": "posterior",
    }
    monkeypatch.setattr(
        "openghg_inversions.rhime.stages.diagnose_rhime_stage",
        lambda **kwargs: result,
    )
    args = ["diagnose", "--posterior", "posterior.nc", "--output-dir", "diagnostics"]
    if strict:
        args.append("--strict")

    if raises:
        with pytest.raises(SystemExit, match="1"):
            main(args)
    else:
        main(args)

    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize(
    "thresholds",
    [
        {"rhat_max": np.nan},
        {"rhat_max": np.inf},
        {"rhat_max": 0.99},
        {"ess_min_ratio": np.nan},
        {"bfmi_threshold": -1.0},
        {"ess_threshold": -1},
        {"ess_threshold": 2.5},
    ],
)
def test_diagnostics_reject_invalid_thresholds_before_loading(
    tmp_path: Path,
    thresholds: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError, match="Diagnostic threshold"):
        diagnose_rhime_stage(
            posterior=tmp_path / "missing.nc",
            output_dir=tmp_path / "diagnose",
            **thresholds,
        )


def test_diagnostics_reject_empty_check_stage_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        diagnose_rhime_stage(
            posterior=tmp_path / "missing.nc",
            output_dir=tmp_path / "diagnose",
            stage=" ",
        )


@pytest.mark.parametrize("model", ["standard", "multisector", "co2"])
def test_diagnostics_accept_digest_free_sample_manifest(tmp_path: Path, model: str) -> None:
    from openghg_inversions.rhime._stage_artifacts import write_json

    posterior = make_trace(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))})
    )
    supplied_path = tmp_path / "supplied.nc"
    save_trace(posterior, supplied_path)
    manifest_path = tmp_path / "sample-manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1 if model == "co2" else 3,
            "producer": "openghg_inversions",
            "stage": "sample",
            "schema": f"openghg_inversions.{model}.sample",
        },
    )

    result = diagnose_rhime_stage(
        posterior=supplied_path,
        sample_manifest=manifest_path,
        output_dir=tmp_path / "diagnose",
    )
    assert result["measured_values"]["chains"] == 2
    assert result["measured_values"]["draws_per_chain"] == 4
    assert not (tmp_path / "diagnose" / "diagnose-manifest.json").exists()

@pytest.mark.parametrize("use_override", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_resource_origins_match_full_requests(tmp_path, monkeypatch, use_override, absolute):
    from openghg_inversions.rhime.params import RhimeConfig
    from openghg_inversions.rhime.requests import load_params

    config_dir = tmp_path / "configs"
    invocation = tmp_path / "invocation"
    config_dir.mkdir()
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    resources = {
        "basis_path": "basis.nc", "basis_partition_path": "partition.nc",
        "landsea_path": "landsea.nc", "outer_regions_path": "outer.nc",
        "basis_output_path": "runs/prepared/basis.zarr", "country_file": "country.nc",
    }
    if absolute:
        resources = {name: str(tmp_path / value) for name, value in resources.items()}
    source = config_dir / "run.json"
    source.write_text(json.dumps(_params(**resources)))
    overrides = resources if use_override else None
    # The same file has the same resource meaning in both routes.
    staged = load_params(params_file=source, overrides=overrides)
    full = load_params(params_file=source, overrides=overrides)
    def paths(loaded):
        return {name: {**loaded["data"], **loaded["output"]}[name] for name in resources}
    assert paths(staged) == {name: invocation / value
                             for name, value in resources.items()}
    assert paths(full) == {name: invocation / value for name, value in resources.items()}
    config = RhimeConfig.from_params(staged, multisector=False)
    assert config.output.country_file == paths(staged)["country_file"]
    assert config.data.basis_path == paths(staged)["basis_path"]


def test_co2_stage_override_paths_use_invocation_directory(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2.execution import load_co2_params, resolve_co2_output
    from openghg_inversions.rhime.co2.configuration import Co2PreparationConfig

    source = tmp_path / "configs" / "co2.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"prepared_inputs": {"path": "prepared.nc"},
                                 "outputs": {"country_file": "country.nc"}}))
    monkeypatch.chdir(tmp_path)
    overrides = {"prepared_inputs": {"path": "selected.nc"}}
    loaded = load_co2_params(params_file=source, overrides=overrides)
    assert Co2PreparationConfig.from_params(loaded).preparation_arguments()["prepared_path"] == tmp_path / "selected.nc"
    assert resolve_co2_output(loaded)["output"].country_file == tmp_path / "country.nc"
    assert overrides["prepared_inputs"]["path"] == "selected.nc"


@pytest.mark.parametrize("model", ["standard", "multisector", "co2"])
def test_diagnosis_rejects_old_sample_manifest_before_numerical_io(tmp_path, monkeypatch, model):
    from openghg_inversions.rhime import _stage_checks
    manifest = tmp_path / "sample-manifest.json"
    manifest.write_text(json.dumps({
        "schema": f"openghg_inversions.{model}.sample", "schema_version": 2,
        "producer": "openghg_inversions", "stage": "sample",
        "model": {"aggregation_error_mode": "none"}, "posterior": "missing.nc",
    }))
    monkeypatch.setattr(_stage_checks, "load_trace", lambda *a, **k: pytest.fail("loaded obsolete numerical payload"))
    with pytest.raises(ValueError, match="unsupported schema_version.*Use the software version that wrote it or prepare and sample again"):
        diagnose_rhime_stage(sample_manifest=manifest, output_dir=tmp_path / "diagnose")
    assert not (tmp_path / "diagnose").exists()
