"""Contract tests for file-backed RHIME workflow stages."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.cli import build_parser
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.stages import (
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    configuration_identity,
    diagnose_rhime_stage,
    postprocess_rhime_stage,
    prepare_rhime_stage,
    prior_predictive_stage,
    resolve_stage_setup,
    sample_rhime_stage,
)
from openghg_inversions.serialization import save_inferencedata


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


def _prepared() -> RhimePreparedInputs:
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
    return RhimePreparedInputs(inv_inputs=inputs, basis_functions=basis, site_metadata=metadata)


def _fake_save_merged(_data: object, directory: str | Path, **_kwargs: object) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True)
    (destination / "merged-data.nc").write_bytes(b"synthetic merged data")


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


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("species", "sf6"),
        ("start_date", "2020-01-01"),
        ("x_prior", {"pdf": "normal", "mu": 0.5, "sigma": 0.2}),
    ],
)
def test_configuration_identity_covers_gas_period_and_prior(change: str, value: Any) -> None:
    original = resolve_stage_setup(_params(), model="standard")
    changed = resolve_stage_setup(_params(**{change: value}), model="standard")

    assert configuration_identity(original, model="standard") != configuration_identity(
        changed, model="standard"
    )


def test_configuration_identity_allows_sampling_and_output_changes() -> None:
    original = resolve_stage_setup(_params(), model="standard")
    changed = resolve_stage_setup(
        _params(draws=100, output_path="/another-explicit-output"),
        model="standard",
    )

    assert configuration_identity(original, model="standard") == configuration_identity(
        changed, model="standard"
    )


def test_prepare_is_independent_and_writes_inspectable_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    prepared = _prepared()
    sentinel = SimpleNamespace(sites=("TAC",), fp_all={})
    monkeypatch.setattr(stages, "retrieve_or_reload_rhime_data", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(stages, "filter_rhime_observations", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(stages, "build_rhime_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_rhime_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(stages, "_save_merged_data", _fake_save_merged)
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *args, **kwargs: pytest.fail("sampled"))

    manifest = prepare_rhime_stage(
        setup=resolve_stage_setup(_params(), model="standard"),
        model="standard",
        output_dir=tmp_path / "prepare",
    )

    prepared_path = Path(manifest["artifacts"]["prepared_inputs"])
    assert prepared_path.is_file()
    assert Path(manifest["artifacts"]["merged_data"]).is_file()
    assert manifest["artifact_identities"]["prepared_inputs"].startswith("sha256:")
    assert Path(manifest["manifest_path"]).is_file()
    xr.testing.assert_identical(RhimePreparedInputs.load(prepared_path).inv_inputs, prepared.inv_inputs)


def test_prepare_fails_when_a_requested_site_was_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    prepared = _prepared()
    merged = SimpleNamespace(sites=("TAC",), fp_all={})
    monkeypatch.setattr(stages, "retrieve_or_reload_rhime_data", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "filter_rhime_observations", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "build_rhime_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_rhime_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(stages, "_save_merged_data", _fake_save_merged)

    with pytest.raises(ValueError, match="could not produce required site.*MHD"):
        prepare_rhime_stage(
            setup=resolve_stage_setup(
                _params(sites=["TAC", "MHD"], averaging_period=["1h", "1h"]),
                model="standard",
            ),
            model="standard",
            output_dir=tmp_path,
        )


def test_prior_predictive_failure_writes_gate_compatible_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    monkeypatch.setattr(
        stages,
        "_load_prepared",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing configured prior flux")),
    )
    check_path = tmp_path / "prior-ready.json"
    result = prior_predictive_stage(
        setup=resolve_stage_setup(_params(), model="standard"),
        model="standard",
        prepared_inputs=tmp_path / "missing.nc",
        output_dir=tmp_path,
        check_output=check_path,
        draws=5,
    )

    assert result["name"] == PREPARATION_CHECK_NAME
    assert result["status"] == "fail"
    assert result["schema_version"] == 1
    assert "missing configured prior flux" in result["message"]
    assert check_path.is_file()


def test_diagnostics_emit_issue_667_convergence_signals(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    posterior = xr.Dataset(
        {"x": (("chain", "draw", "region"), rng.normal(size=(4, 500, 2)))},
        coords={"chain": range(4), "draw": range(500), "region": ["north", "south"]},
    )
    diverging = np.zeros((4, 500), dtype=bool)
    diverging[2, 10] = True
    idata = az.InferenceData(
        posterior=posterior,
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), diverging)},
            coords={"chain": range(4), "draw": range(500)},
        ),
    )
    posterior_path = tmp_path / "posterior.nc"
    save_inferencedata(idata, posterior_path)

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["name"] == CONVERGENCE_CHECK_NAME
    assert result["status"] == "fail"
    assert result["measured_values"]["chains"] == 4
    assert result["measured_values"]["draws_per_chain"] == 500
    assert result["measured_values"]["divergences"] == 1
    assert result["measured_values"]["divergences_by_chain"] == [0, 0, 1, 0]
    assert result["measured_values"]["max_rhat_variable"] is not None
    assert result["measured_values"]["min_bulk_ess_variable"] is not None
    assert result["measured_values"]["min_tail_ess_variable"] is not None
    assert Path(result["artifact_paths"][0]).is_file()


def test_synthetic_staged_tracer_bullet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise prepare -> prior -> sample -> diagnose -> postprocess without SLURM."""
    from openghg_inversions.rhime import stages

    prepared = _prepared()
    sentinel = SimpleNamespace(sites=("TAC",), fp_all={})
    monkeypatch.setattr(stages, "retrieve_or_reload_rhime_data", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(stages, "filter_rhime_observations", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(stages, "build_rhime_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_rhime_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(stages, "_save_merged_data", _fake_save_merged)
    setup = resolve_stage_setup(
        _params(sample_kwargs={"random_seed": 42}),
        model="standard",
    )

    preparation = prepare_rhime_stage(setup=setup, model="standard", output_dir=tmp_path / "prepare")
    prepared_path = Path(preparation["artifacts"]["prepared_inputs"])
    manifest_path = Path(preparation["manifest_path"])
    prior = prior_predictive_stage(
        setup=setup,
        model="standard",
        prepared_inputs=prepared_path,
        preparation_manifest=manifest_path,
        output_dir=tmp_path / "prior",
        draws=5,
    )
    sampled = sample_rhime_stage(
        setup=setup,
        model="standard",
        prepared_inputs=prepared_path,
        preparation_manifest=manifest_path,
        output_dir=tmp_path / "sample",
    )
    posterior_path = Path(sampled["artifacts"]["posterior"])
    convergence = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")
    result = postprocess_rhime_stage(
        setup=setup,
        model="standard",
        prepared_inputs=prepared_path,
        preparation_manifest=manifest_path,
        posterior=posterior_path,
        output_dir=tmp_path / "postprocess",
    )

    assert prior["status"] == "pass"
    assert posterior_path.is_file()
    assert convergence["status"] in {"fail", "unknown", "pass"}
    assert result.output_metadata["inversion_output_path"].startswith(str(tmp_path / "postprocess"))
    assert Path(result.output_metadata["inversion_output_path"]).is_file()
    assert Path(result.output_metadata["postprocess_manifest_path"]).is_file()
