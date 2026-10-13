"""Contract tests for file-backed RHIME workflow stages."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.cli import build_parser
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs, SiteOptions
from openghg_inversions.inference import diagnostics as inference_diagnostics
from openghg_inversions.rhime.stages import (
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    configuration_identity,
    diagnose_rhime_stage,
    load_stage_params,
    prepare_rhime_stage,
    prior_predictive_stage,
    resolve_stage_setup,
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


def _multisector_params(**overrides):
    return _params(**{"flux_sources": ["synthetic-total", "second"], **overrides})


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
        json.dumps(_multisector_params(basis_directory="relative/basis", basis_partition_path="relative/partition.nc",
                           country_file="relative/country.nc")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)

    loaded = load_stage_params(params_file=params_path)

    assert loaded["basis_directory"] == str(params_path.parent / "relative/basis")
    assert loaded["country_file"] == str(params_path.parent / "relative/country.nc")
    assert loaded["basis_partition_path"] == str(params_path.parent / "relative/partition.nc")


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("species", "sf6"),
        ("start_date", "2020-01-01"),
        ("x_prior", {"pdf": "normal", "mu": 0.5, "sigma": 0.2}),
    ],
)
def test_configuration_identity_covers_gas_period_and_prior(change: str, value: Any) -> None:
    original = resolve_stage_setup(_multisector_params(), model="multisector")
    changed = resolve_stage_setup(_multisector_params(**{change: value}), model="multisector")

    assert configuration_identity(original, model="multisector") != configuration_identity(
        changed, model="multisector"
    )


def test_configuration_identity_allows_sampling_and_output_changes() -> None:
    original = resolve_stage_setup(_multisector_params(), model="multisector")
    changed = resolve_stage_setup(
        _multisector_params(
            draws=100,
            output_path="/another-explicit-output",
            output_name="another-name",
            basis_output_path="/another/basis-output",
        ),
        model="multisector",
    )

    assert configuration_identity(original, model="multisector") == configuration_identity(
        changed, model="multisector"
    )


def test_configuration_identity_serialises_slice_inlet_selectors() -> None:
    setup = resolve_stage_setup(_multisector_params(inlet=[slice(3, 10)]), model="multisector")

    assert configuration_identity(setup, model="multisector").startswith("sha256:")


@pytest.mark.parametrize("output_format", ["inv_out", "paris"])
def test_prepare_is_independent_and_writes_inspectable_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    output_format: str,
) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    prepared = _prepared()
    sentinel = RhimeMergedData(
        site_data={"TAC": xr.Dataset({
            "mf_error": ("time", [1.0]),
            "mf_repeatability": ("time", [0.0]),
            "mf_variability": ("time", [0.0]),
        })},
        flux_data={"synthetic-total": prepared.basis_functions.flux.to_dataset(name="flux")},
        site_options=SiteOptions.from_inputs(sites=("TAC",), averaging_period="1h"),
    )
    monkeypatch.setattr(stages.RhimeMergedData, "from_options", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(stages, "filter_observations", lambda *args, **kwargs: sentinel)
    def prepare_selected_basis(**kwargs):
        assert kwargs["output_path"] == "/outside/basis"
        return prepared.basis_functions

    monkeypatch.setattr(stages, "prepare_basis", prepare_selected_basis)
    monkeypatch.setattr(stages, "build_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *args, **kwargs: pytest.fail("sampled"))

    setup = resolve_stage_setup(
        _multisector_params(
            basis_output_path="/outside/basis",
            output_format=output_format,
            inlet=[slice(3, 10)],
        ),
        model="multisector",
    )
    manifest = prepare_rhime_stage(
        setup=setup,
        model="multisector",
        output_dir=tmp_path / "prepare",
    )

    prepared_path = Path(manifest["artifacts"]["prepared_inputs"])
    assert prepared_path.is_file()
    assert set(manifest["artifacts"]) == {"prepared_inputs"}
    assert "artifact_identities" not in manifest
    assert not (tmp_path / "prepare" / "merged-data").exists()
    assert manifest["effective_configuration"]["preparation"]["basis_output_path"] == "/outside/basis"
    assert Path(manifest["manifest_path"]).is_file()
    persisted_manifest = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert persisted_manifest["effective_configuration"]["preparation"]["inlet"] == [
        {"type": "slice", "start": 3, "stop": 10, "step": None}
    ]
    xr.testing.assert_identical(RhimePreparedInputs.load(prepared_path).inv_inputs, prepared.inv_inputs)

    assert capsys.readouterr().out.count("TIMING rhime.prepare_inputs.merged_data ") == 1


def test_prepare_fails_when_a_requested_site_was_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    prepared = _prepared()
    merged = RhimeMergedData(
        site_data={"TAC": xr.Dataset({
            "mf_error": ("time", [1.0]),
            "mf_repeatability": ("time", [0.0]),
            "mf_variability": ("time", [0.0]),
        })},
        flux_data={"synthetic-total": prepared.basis_functions.flux.to_dataset(name="flux")},
        site_options=SiteOptions.from_inputs(sites=("TAC",), averaging_period="1h"),
    )
    monkeypatch.setattr(stages.RhimeMergedData, "from_options", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "filter_observations", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "prepare_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)

    with pytest.raises(ValueError, match="could not produce required site.*MHD"):
        prepare_rhime_stage(
            setup=resolve_stage_setup(
                _multisector_params(sites=["TAC", "MHD"], averaging_period=["1h", "1h"]),
                model="multisector",
            ),
            model="multisector",
            output_dir=tmp_path,
        )


def test_prepare_accepts_canonicalised_site_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    prepared = _prepared()
    merged = RhimeMergedData(
        site_data={"TAC": xr.Dataset({
            "mf_error": ("time", [1.0]),
            "mf_repeatability": ("time", [0.0]),
            "mf_variability": ("time", [0.0]),
        })},
        flux_data={"synthetic-total": prepared.basis_functions.flux.to_dataset(name="flux")},
        site_options=SiteOptions.from_inputs(sites=("TAC",), averaging_period="1h"),
    )
    monkeypatch.setattr(stages.RhimeMergedData, "from_options", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "filter_observations", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "prepare_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)

    manifest = prepare_rhime_stage(
        setup=resolve_stage_setup(_multisector_params(sites=["tac"]), model="multisector"),
        model="multisector",
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
            setup=resolve_stage_setup(_multisector_params(), model="multisector"),
            model="multisector",
            output_dir=destination,
        )

    assert outside.read_bytes() == b"unchanged"


def test_prior_predictive_failure_writes_gate_compatible_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    setup = resolve_stage_setup(_multisector_params(), model="multisector")
    monkeypatch.setattr(stages, "_load_prepared", lambda *args, **kwargs: (_prepared(), setup))
    monkeypatch.setattr(
        stages,
        "_build_prepared_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing configured prior flux")),
    )
    check_path = tmp_path / "prior-ready.json"
    result = prior_predictive_stage(
        setup=setup,
        model="multisector",
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
    from openghg_inversions.rhime import _standard_stages as stages

    setup = resolve_stage_setup(_multisector_params(), model="multisector")
    prior = make_trace(prior=xr.Dataset({"x": (("chain", "draw"), [[1.0]])}))
    monkeypatch.setattr(stages, "_load_prepared", lambda *args, **kwargs: (_prepared(), setup))
    monkeypatch.setattr(
        stages,
        "_build_prepared_model",
        lambda *args, **kwargs: SimpleNamespace(model=nullcontext()),
    )
    monkeypatch.setattr(stages.pm, "sample_prior_predictive", lambda *args, **kwargs: prior)
    monkeypatch.setattr(
        stages,
        "save_trace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        prior_predictive_stage(
            setup=setup,
            model="multisector",
            prepared_inputs=tmp_path / "prepared.nc",
            preparation_manifest=tmp_path / "prepare-manifest.json",
            output_dir=tmp_path / "prior",
        )

    assert not (tmp_path / "prior" / "prior-predictive-readiness.json").exists()


@pytest.mark.parametrize("draws", [0, -1, True])
def test_prior_predictive_rejects_invalid_draw_counts(tmp_path: Path, draws: int) -> None:
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        prior_predictive_stage(
            setup=resolve_stage_setup(_multisector_params(), model="multisector"),
            model="multisector",
            prepared_inputs=tmp_path / "missing.nc",
            preparation_manifest=tmp_path / "missing-manifest.json",
            output_dir=tmp_path / "prior",
            draws=draws,
        )


@pytest.mark.parametrize("contents", [None, "not a prepared NetCDF file"])
def test_preparation_uses_ordinary_prepared_input_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str | None,
) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    prepared = _prepared()
    merged = RhimeMergedData(
        site_data={"TAC": xr.Dataset({
            "mf_error": ("time", [1.0]),
            "mf_repeatability": ("time", [0.0]),
            "mf_variability": ("time", [0.0]),
        })},
        flux_data={"synthetic-total": prepared.basis_functions.flux.to_dataset(name="flux")},
        site_options=SiteOptions.from_inputs(sites=("TAC",), averaging_period="1h"),
    )
    monkeypatch.setattr(stages.RhimeMergedData, "from_options", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "filter_observations", lambda *args, **kwargs: merged)
    monkeypatch.setattr(stages, "prepare_basis", lambda *args, **kwargs: prepared.basis_functions)
    monkeypatch.setattr(stages, "build_sensitivities", lambda *args, **kwargs: {})
    monkeypatch.setattr(stages, "assemble_rhime_inputs", lambda *args, **kwargs: prepared)
    setup = resolve_stage_setup(_multisector_params(), model="multisector")
    preparation = prepare_rhime_stage(setup=setup, model="multisector", output_dir=tmp_path / "prepare")

    supplied_path = tmp_path / "supplied.nc"
    if contents is not None:
        supplied_path.write_text(contents)
    with pytest.raises((OSError, ValueError)):
        stages._load_prepared(
            supplied_path,
            setup=setup,
            model="multisector",
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
    idata = make_trace(
        posterior=posterior,
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), diverging)},
            coords={"chain": range(4), "draw": range(500)},
        ),
    )
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
    assert result["measured_values"]["draws_per_chain"] == 500
    assert result["measured_values"]["divergences"] == 1
    assert result["measured_values"]["divergences_by_chain"] == [0, 0, 1, 0]
    assert result["measured_values"]["max_rhat_variable"] is not None
    assert result["measured_values"]["min_bulk_ess_variable"] is not None
    assert result["measured_values"]["min_tail_ess_variable"] is not None
    assert Path(result["artifact_paths"][0]).is_file()
    assert engines == [None]


def test_diagnostics_preserve_threshold_edge_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unrounded diagnostics just beyond convergence thresholds must fail."""

    idata = make_trace(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))}),
        sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((2, 4), dtype=bool))}),
    )
    posterior_path = tmp_path / "posterior.nc"
    save_trace(idata, posterior_path)

    def fake_summary(*args: object, **kwargs: object) -> xr.Dataset:
        """Return diagnostic values that default ArviZ rounding can hide."""
        assert kwargs["round_to"] == "none"
        return xr.Dataset(
            {"x": ("metric", [1.0101, 399.99, 399.99])},
            coords={"metric": ["r_hat", "ess_bulk", "ess_tail"]},
        )

    monkeypatch.setattr(inference_diagnostics.az, "summary", fake_summary)

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["status"] == "fail"
    assert result["measured_values"]["max_rhat"] == 1.0101
    assert result["measured_values"]["min_bulk_ess"] == 399.99
    assert result["measured_values"]["min_tail_ess"] == 399.99


@pytest.mark.parametrize(("finite_rhat", "expected_status"), [(1.0, "unknown"), (1.2, "fail")])
def test_diagnostics_preserve_finite_failures_when_one_metric_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    finite_rhat: float,
    expected_status: str,
) -> None:

    idata = make_trace(
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
    save_trace(idata, posterior_path)
    summary = xr.Dataset(
        {
            "x": (
                ("metric", "region"),
                [[finite_rhat, np.nan], [800.0, 800.0], [700.0, 700.0]],
            )
        },
        coords={"metric": ["r_hat", "ess_bulk", "ess_tail"], "region": ["known", "undefined"]},
    )
    monkeypatch.setattr(inference_diagnostics.az, "summary", lambda *args, **kwargs: summary)

    result = diagnose_rhime_stage(posterior=posterior_path, output_dir=tmp_path / "diagnose")

    assert result["status"] == expected_status
    assert result["measured_values"]["max_rhat"] == finite_rhat
    assert result["measured_values"]["unassessable_rhat"] == ["x,region=undefined"]
    assert "max_rhat" in result["message"]


def test_diagnostics_handle_unassessable_scalar_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:

    idata = make_trace(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))}),
        sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((2, 4), dtype=bool))}),
    )
    posterior_path = tmp_path / "posterior.nc"
    save_trace(idata, posterior_path)
    monkeypatch.setattr(
        inference_diagnostics.az,
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


@pytest.mark.parametrize("model", ["standard", "co2"])
def test_diagnostics_accept_digest_free_sample_manifest(tmp_path: Path, model: str) -> None:
    from openghg_inversions.rhime import _standard_stages as stages

    posterior = make_trace(
        posterior=xr.Dataset({"x": (("chain", "draw"), np.ones((2, 4)))})
    )
    supplied_path = tmp_path / "supplied.nc"
    save_trace(posterior, supplied_path)
    manifest_path = tmp_path / "sample-manifest.json"
    stages._write_json(
        manifest_path,
        {
            "schema_version": 1,
            "producer": "openghg_inversions",
            "stage": "sample",
            "effective_configuration": {"model": model},
        },
    )

    result = diagnose_rhime_stage(
        posterior=supplied_path,
        sample_manifest=manifest_path,
        output_dir=tmp_path / "diagnose",
    )
    assert result["measured_values"]["chains"] == 2
    assert result["measured_values"]["draws_per_chain"] == 4
    if model == "co2":
        manifest = json.loads((tmp_path / "diagnose" / "diagnose-manifest.json").read_text())
        assert manifest["configuration_identity"] is None


def test_configuration_identity_does_not_visit_opaque_output_options():
    """Output iterables cannot affect or break the preparation identity."""
    original = resolve_stage_setup(_multisector_params(), model="multisector")
    selections = (name for name in ["UNITED KINGDOM"])
    with_opaque_output = resolve_stage_setup(
        _multisector_params(paris_postprocessing_kwargs={"country_selections": selections}), model="multisector",
    )
    assert configuration_identity(original, model="multisector") == configuration_identity(
        with_opaque_output, model="multisector"
    )
    assert next(selections) == "UNITED KINGDOM"
