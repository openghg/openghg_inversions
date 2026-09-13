"""Contract tests for file-backed RHIME workflow stages."""

from __future__ import annotations

from contextlib import nullcontext
import json
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
    load_stage_params,
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
                "--prepared-inputs",
                "/run/prepared.nc",
                "--output-dir",
                "/run/sample",
            ]
        )


def test_stage_config_paths_are_relative_to_the_config_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "config" / "params.json"
    params_path.parent.mkdir()
    params_path.write_text(
        json.dumps(_params(basis_directory="relative/basis", country_file="relative/country.nc")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)

    loaded = load_stage_params(params_file=params_path)

    assert loaded["basis_directory"] == str(params_path.parent / "relative/basis")
    assert loaded["country_file"] == str(params_path.parent / "relative/country.nc")


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
        _params(
            draws=100,
            output_path="/another-explicit-output",
            output_name="another-name",
            save_merged_data=True,
            merged_data_dir="/another/cache",
            basis_output_path="/another/basis-output",
        ),
        model="standard",
    )

    assert configuration_identity(original, model="standard") == configuration_identity(
        changed, model="standard"
    )


def test_configuration_identity_serialises_slice_inlet_selectors() -> None:
    setup = resolve_stage_setup(_params(inlet=[slice(3, 10)]), model="standard")

    assert configuration_identity(setup, model="standard").startswith("sha256:")


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

    setup = resolve_stage_setup(
        _params(
            save_merged_data=True,
            basis_output_path="/outside/basis",
            inlet=[slice(3, 10)],
        ),
        model="standard",
    )
    manifest = prepare_rhime_stage(
        setup=setup,
        model="standard",
        output_dir=tmp_path / "prepare",
    )

    prepared_path = Path(manifest["artifacts"]["prepared_inputs"])
    assert prepared_path.is_file()
    assert Path(manifest["artifacts"]["merged_data"]).is_file()
    assert manifest["artifact_identities"]["prepared_inputs"].startswith("sha256:")
    assert manifest["requested_configuration"]["preparation"]["save_merged_data"] is True
    assert manifest["effective_configuration"]["preparation"]["save_merged_data"] is False
    assert manifest["effective_configuration"]["preparation"]["basis_output_path"] == str(
        tmp_path / "prepare" / "basis"
    )
    assert Path(manifest["manifest_path"]).is_file()
    persisted_manifest = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert persisted_manifest["effective_configuration"]["preparation"]["inlet"] == [
        {"type": "slice", "start": 3, "stop": 10, "step": None}
    ]
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


def test_prepare_accepts_canonicalised_site_labels(
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

    manifest = prepare_rhime_stage(
        setup=resolve_stage_setup(_params(sites=["tac"]), model="standard"),
        model="standard",
        output_dir=tmp_path,
    )

    assert Path(manifest["artifacts"]["prepared_inputs"]).is_file()


def test_stage_rejects_preexisting_symlink_output_targets(tmp_path: Path) -> None:
    destination = tmp_path / "prepare"
    destination.mkdir()
    outside = tmp_path / "outside.nc"
    outside.write_bytes(b"unchanged")
    (destination / "prepared-inputs.nc").symlink_to(outside)

    with pytest.raises(ValueError, match="contains symlink"):
        prepare_rhime_stage(
            setup=resolve_stage_setup(_params(), model="standard"),
            model="standard",
            output_dir=destination,
        )

    assert outside.read_bytes() == b"unchanged"


def test_prior_predictive_failure_writes_gate_compatible_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    setup = resolve_stage_setup(_params(), model="standard")
    monkeypatch.setattr(stages, "_load_prepared", lambda *args, **kwargs: (_prepared(), setup))
    monkeypatch.setattr(
        stages,
        "_build_prepared_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing configured prior flux")),
    )
    check_path = tmp_path / "prior-ready.json"
    result = prior_predictive_stage(
        setup=setup,
        model="standard",
        prepared_inputs=tmp_path / "missing.nc",
        preparation_manifest=tmp_path / "prepare-manifest.json",
        output_dir=tmp_path,
        check_output=check_path,
        draws=5,
    )

    assert result["name"] == PREPARATION_CHECK_NAME
    assert result["status"] == "fail"
    assert result["schema_version"] == 1
    assert "missing configured prior flux" in result["message"]
    assert check_path.is_file()


def test_prior_predictive_does_not_hide_serialisation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    setup = resolve_stage_setup(_params(), model="standard")
    prior = az.InferenceData(prior=xr.Dataset({"x": (("chain", "draw"), [[1.0]])}))
    monkeypatch.setattr(stages, "_load_prepared", lambda *args, **kwargs: (_prepared(), setup))
    monkeypatch.setattr(
        stages,
        "_build_prepared_model",
        lambda *args, **kwargs: SimpleNamespace(model=nullcontext()),
    )
    monkeypatch.setattr(stages.pm, "sample_prior_predictive", lambda *args, **kwargs: prior)
    monkeypatch.setattr(
        stages,
        "save_inferencedata",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        prior_predictive_stage(
            setup=setup,
            model="standard",
            prepared_inputs=tmp_path / "prepared.nc",
            preparation_manifest=tmp_path / "prepare-manifest.json",
            output_dir=tmp_path / "prior",
        )

    assert not (tmp_path / "prior" / "prior-predictive-readiness.json").exists()


@pytest.mark.parametrize("draws", [0, -1, True])
def test_prior_predictive_rejects_invalid_draw_counts(tmp_path: Path, draws: int) -> None:
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        prior_predictive_stage(
            setup=resolve_stage_setup(_params(), model="standard"),
            model="standard",
            prepared_inputs=tmp_path / "missing.nc",
            preparation_manifest=tmp_path / "missing-manifest.json",
            output_dir=tmp_path / "prior",
            draws=draws,
        )


def test_preparation_manifest_authenticates_supplied_prepared_inputs(
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
    setup = resolve_stage_setup(_params(), model="standard")
    preparation = prepare_rhime_stage(setup=setup, model="standard", output_dir=tmp_path / "prepare")

    unrelated_inputs = prepared.inv_inputs.copy(deep=True)
    unrelated_inputs["mf"].data[0] = 99.0
    unrelated = RhimePreparedInputs(
        inv_inputs=unrelated_inputs,
        basis_functions=prepared.basis_functions,
        site_metadata=prepared.site_metadata,
    )
    unrelated_path = tmp_path / "unrelated.nc"
    unrelated.save(unrelated_path)

    with pytest.raises(ValueError, match="content does not match"):
        stages._load_prepared(
            unrelated_path,
            setup=setup,
            model="standard",
            preparation_manifest=preparation["manifest_path"],
        )


def test_diagnostics_emit_issue_667_convergence_signals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    assert result["measured_values"]["draws_per_chain"] == 500
    assert result["measured_values"]["divergences"] == 1
    assert result["measured_values"]["divergences_by_chain"] == [0, 0, 1, 0]
    assert result["measured_values"]["max_rhat_variable"] is not None
    assert result["measured_values"]["min_bulk_ess_variable"] is not None
    assert result["measured_values"]["min_tail_ess_variable"] is not None
    assert Path(result["artifact_paths"][0]).is_file()
    assert engines == ["h5netcdf"]


@pytest.mark.parametrize(("finite_rhat", "expected_status"), [(1.0, "unknown"), (1.2, "fail")])
def test_diagnostics_preserve_finite_failures_when_one_metric_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    finite_rhat: float,
    expected_status: str,
) -> None:
    from openghg_inversions.rhime import stages

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw", "region"), np.ones((2, 4, 2)))},
            coords={"chain": range(2), "draw": range(4), "region": ["known", "undefined"]},
        ),
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), np.zeros((2, 4), dtype=bool))},
            coords={"chain": range(2), "draw": range(4)},
        ),
    )
    posterior_path = tmp_path / "posterior.nc"
    save_inferencedata(idata, posterior_path)
    summary = xr.Dataset(
        {
            "x": (
                ("metric", "region"),
                [[finite_rhat, np.nan], [800.0, 800.0], [700.0, 700.0]],
            )
        },
        coords={"metric": ["r_hat", "ess_bulk", "ess_tail"], "region": ["known", "undefined"]},
    )
    monkeypatch.setattr(stages.az, "summary", lambda *args, **kwargs: summary)

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["status"] == expected_status
    assert result["measured_values"]["max_rhat"] == finite_rhat
    assert result["measured_values"]["unassessable_rhat"] == ["x,region=undefined"]
    assert "max_rhat" in result["message"]


def test_diagnostics_handle_unassessable_scalar_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import stages

    idata = az.InferenceData(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))}),
        sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((2, 4), dtype=bool))}),
    )
    posterior_path = tmp_path / "posterior.nc"
    save_inferencedata(idata, posterior_path)
    monkeypatch.setattr(
        stages.az,
        "summary",
        lambda *args, **kwargs: xr.Dataset(
            {"x": ("metric", [np.nan, 800.0, 700.0])},
            coords={"metric": ["r_hat", "ess_bulk", "ess_tail"]},
        ),
    )

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["status"] == "unknown"
    assert result["measured_values"]["unassessable_rhat"] == ["x"]


@pytest.mark.parametrize(
    "thresholds",
    [
        {"max_rhat": np.nan},
        {"max_rhat": np.inf},
        {"max_rhat": 0.99},
        {"min_bulk_ess": np.nan},
        {"min_tail_ess": -1.0},
        {"max_divergences": -1},
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


def test_diagnostics_authenticate_posterior_with_sample_manifest(tmp_path: Path) -> None:
    from openghg_inversions.rhime import stages

    posterior = az.InferenceData(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))})
    )
    recorded_path = tmp_path / "recorded.nc"
    supplied_path = tmp_path / "supplied.nc"
    save_inferencedata(posterior, recorded_path)
    changed = posterior.copy()
    changed.posterior["x"].data[0, 0] = 2.0
    save_inferencedata(changed, supplied_path)
    manifest_path = tmp_path / "sample-manifest.json"
    stages._write_json(
        manifest_path,
        {
            "schema_version": 1,
            "producer": "openghg_inversions",
            "stage": "sample",
            "artifact_identities": {"posterior": stages._file_identity(recorded_path)},
        },
    )

    with pytest.raises(ValueError, match="Posterior content does not match"):
        diagnose_rhime_stage(
            posterior=supplied_path,
            sample_manifest=manifest_path,
            output_dir=tmp_path / "diagnose",
        )


@pytest.mark.parametrize("output_name", ["/tmp/outside-", "../outside-"])
def test_postprocess_rejects_output_name_that_can_escape_output_dir(
    tmp_path: Path,
    output_name: str,
) -> None:
    prepared_path = tmp_path / "prepared.nc"
    _prepared().save(prepared_path)
    setup = resolve_stage_setup(_params(output_name=output_name), model="standard")

    with pytest.raises(ValueError, match="must be a non-empty filename component without directories"):
        postprocess_rhime_stage(
            setup=setup,
            model="standard",
            prepared_inputs=prepared_path,
            preparation_manifest=tmp_path / "unused-prepare-manifest.json",
            sample_manifest=tmp_path / "unused-sample-manifest.json",
            posterior=tmp_path / "unused.nc",
            output_dir=tmp_path / "postprocess",
        )


def test_postprocess_rejects_other_unsafe_filename_components(tmp_path: Path) -> None:
    setup = resolve_stage_setup(
        _params(species="../../outside", output_filename_convention="legacy"),
        model="standard",
    )

    with pytest.raises(ValueError, match="species.*filename component"):
        postprocess_rhime_stage(
            setup=setup,
            model="standard",
            prepared_inputs=tmp_path / "unused.nc",
            preparation_manifest=tmp_path / "unused-prepare-manifest.json",
            sample_manifest=tmp_path / "unused-sample-manifest.json",
            posterior=tmp_path / "unused.nc",
            output_dir=tmp_path / "postprocess",
        )


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
    sample_manifest = Path(sampled["manifest_path"])
    convergence = diagnose_rhime_stage(
        posterior=posterior_path,
        sample_manifest=sample_manifest,
        output_dir=tmp_path / "diagnose",
    )
    result = postprocess_rhime_stage(
        setup=setup,
        model="standard",
        prepared_inputs=prepared_path,
        preparation_manifest=manifest_path,
        sample_manifest=sample_manifest,
        posterior=posterior_path,
        output_dir=tmp_path / "postprocess",
    )

    assert prior["status"] == "pass"
    assert posterior_path.is_file()
    assert convergence["status"] in {"fail", "unknown", "pass"}
    assert result.output_metadata["inversion_output_path"].startswith(str(tmp_path / "postprocess"))
    assert Path(result.output_metadata["inversion_output_path"]).is_file()
    postprocess_manifest = Path(result.output_metadata["postprocess_manifest_path"])
    assert postprocess_manifest.is_file()
    postprocess_contract = json.loads(postprocess_manifest.read_text(encoding="utf-8"))
    assert postprocess_contract["input_identities"] == sampled["artifact_identities"]
    assert postprocess_contract["effective_configuration"]["run_spec"]["output"]["output_path"] == str(
        tmp_path / "postprocess"
    )
