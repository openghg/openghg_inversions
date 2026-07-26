"""End-to-end tests for strict conditional-reference certification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any, Literal, Mapping, cast

import arviz as az
import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import (
    mh_local_search_conditional_reference as conditional,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_conditional_reference import (
    CONDITIONAL_REFERENCE_KEYS,
    certify_conditional_reference,
    validate_conditional_reference_record,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_nuts_reference import (
    PROJECTION_NAMES,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    canonical_json,
    file_sha256,
    materialize_replicate,
    write_envelope,
)

_EXAMPLES = Path(__file__).parents[3] / "examples" / "rjmcmc"
Scenario = Literal["aligned", "edge-one", "relocation-one"]
TopologyRole = Literal["p0", "pstar"]


def _load_driver(name: str) -> ModuleType:
    path = _EXAMPLES / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _create_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _create_npz(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(cast(Any, handle), **cast(Any, arrays))


def _fake_inference_data(data: Any, draws: int) -> az.InferenceData:
    generator = np.random.Generator(np.random.PCG64(7821))
    root = generator.lognormal(-0.02, 0.12, size=(4, draws))
    shares = generator.dirichlet(data.dirichlet_alpha, size=(4, draws))
    masses = root[..., None] * shares
    scaling = masses / data.nominal_leaf_share
    mean = data.fixed_offset[None, None, :] + masses @ data.dynamic_design.T
    residual = (data.observations[None, None, :] - mean) / data.observation_sd
    pointwise = -0.5 * residual * residual - np.log(data.observation_sd) - 0.5 * np.log(2.0 * np.pi)
    return az.from_dict(
        posterior={
            "root_total": root.astype(np.float64),
            "leaf_share": shares.astype(np.float64),
            "leaf_mass": masses.astype(np.float64),
            "leaf_scaling": scaling.astype(np.float64),
            "fixed_coefficient": np.empty((4, draws, 0), dtype=np.float64),
            "mean_observation": mean.astype(np.float64),
        },
        sample_stats={
            "diverging": np.zeros((4, draws), dtype=bool),
            "n_steps": np.ones((4, draws), dtype=np.int64),
            "tree_depth": np.ones((4, draws), dtype=np.int64),
            "acceptance_rate": np.full((4, draws), 0.9, dtype=np.float64),
            "energy": np.ones((4, draws), dtype=np.float64),
            "lp": -np.ones((4, draws), dtype=np.float64),
            "step_size": np.full((4, draws), 0.1, dtype=np.float64),
        },
        observed_data={"observed": np.asarray(data.observations, dtype=np.float64)},
        log_likelihood={"observed": pointwise.astype(np.float64)},
        coords={
            "leaf": np.asarray(data.leaf_labels),
            "fixed": np.asarray([], dtype=np.str_),
            "observation": np.arange(data.observations.size),
        },
        dims={
            "leaf_share": ["leaf"],
            "leaf_mass": ["leaf"],
            "leaf_scaling": ["leaf"],
            "fixed_coefficient": ["fixed"],
            "mean_observation": ["observation"],
            "observed": ["observation"],
        },
    )


def _nuts_bundle(
    *,
    directory: Path,
    training_path: Path,
    evaluation_path: Path,
    training: Any,
    evaluation: Any,
    topology_role: TopologyRole,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    driver = _load_driver("mh_local_search_nuts_reference")
    fake_model = object()
    backend = {
        "pymc_version": "test-pymc",
        "pytensor_version": "test-pytensor",
        "jax_version": "test-jax",
        "numpyro_version": "test-numpyro",
        "arviz_version": "test-arviz",
        "pytensor_floatX": "float64",
        "jax_enable_x64": True,
        "jax_backend": "cpu",
    }

    def fake_preflight(setup: Any) -> tuple[object, tuple[dict[str, object], ...]]:
        return fake_model, tuple(
            {
                **backend,
                "model_value_variables_float64": True,
                "model_value_variable_count": 2,
                "constrained_log_target": start.state.log_target,
                "expected_log_target": start.state.log_target,
                "log_target_difference": 0.0,
                "log_target_absolute_tolerance": 5.0e-10 * max(1.0, abs(start.state.log_target)),
            }
            for start in setup.starts
        )

    def fake_sampler(model: object, data: Any, **settings: Any) -> az.InferenceData:
        assert model is fake_model
        return _fake_inference_data(data, int(settings["draws"]))

    monkeypatch.setattr(
        driver,
        "require_fixed_basis_nuts_float64",
        lambda: backend,
    )
    monkeypatch.setattr(driver, "preflight_s0_nuts_reference", fake_preflight)
    monkeypatch.setattr(driver, "sample_fixed_basis_nuts", fake_sampler)
    arguments = driver.parser().parse_args(
        [
            "--training",
            str(training_path),
            "--evaluation",
            str(evaluation_path),
            "--topology",
            topology_role,
            "--profile",
            "primary",
            "--output-directory",
            str(directory),
            "--source-revision",
            subprocess.run(
                ("git", "-C", str(Path(__file__).parents[3]), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "--cell-id",
            training.cell_id,
            "--definition-sha256",
            training.definition_sha256,
        ]
    )
    driver.run(arguments, enforce_clean_revision=False)
    return directory


def _local_bundle(
    *,
    directory: Path,
    training_path: Path,
    evaluation_path: Path,
    topology_role: TopologyRole,
) -> Path:
    driver = _load_driver("mh_local_search_synthetic")
    branch = directory.parent / "branch"
    if topology_role == "p0":
        driver._run_short_pair_for_test(
            training_path=training_path,
            output_directory=branch,
            conditioning_cycles=1,
            production_cycles=100,
        )
    else:
        driver._run_oracle_short_for_test(
            training_path=training_path,
            evaluation_path=evaluation_path,
            output_directory=branch,
            conditioning_cycles=1,
            production_cycles=100,
        )
    driver._run_local_reference_short_for_test(
        training_path=training_path,
        evaluation_path=evaluation_path,
        topology=topology_role,
        branch_run_directory=branch,
        output_directory=directory,
        production_cycles=100,
    )
    return directory


def _complete_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: Scenario,
    topology_role: TopologyRole,
) -> tuple[Path, Path, Path, Path]:
    training, evaluation = materialize_replicate(
        build_stage_definition("s0"),
        scenario=scenario,
        replicate=0,
    )
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    nuts = _nuts_bundle(
        directory=tmp_path / "nuts",
        training_path=training_path,
        evaluation_path=evaluation_path,
        training=training,
        evaluation=evaluation,
        topology_role=topology_role,
        monkeypatch=monkeypatch,
    )
    local = _local_bundle(
        directory=tmp_path / "local",
        training_path=training_path,
        evaluation_path=evaluation_path,
        topology_role=topology_role,
    )
    return training_path, evaluation_path, nuts, local


@pytest.mark.parametrize(
    ("nuts_profile", "local_profile"),
    [
        ("primary", "primary"),
        ("primary", "factor4"),
        ("retry1", "primary"),
        ("retry1", "factor4"),
    ],
)
def test_nuts_and_local_profiles_form_a_cartesian_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nuts_profile: Literal["primary", "retry1"],
    local_profile: Literal["primary", "factor4"],
) -> None:
    training, evaluation = materialize_replicate(
        build_stage_definition("s0"),
        scenario="edge-one",
        replicate=0,
    )
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    projections = {name: {"mean": 1.0, "sd": 1.0, "mcse_mean": 0.01} for name in PROJECTION_NAMES}
    nuts = {
        "projections": projections,
        "divergences": 0,
        "worst_rhat_variable": "root_total",
        "worst_rhat_value": 1.0,
        "minimum_bulk_ess_variable": "root_total",
        "minimum_bulk_ess_value": 300.0,
        "minimum_tail_ess_variable": "root_total",
        "minimum_tail_ess_value": 300.0,
    }
    nuts_audit = {
        "profile": nuts_profile,
        "source_revision": "a" * 40,
        "completion_sha256": "1" * 64,
    }
    local_audit = {
        "profile": local_profile,
        "completion_sha256": "2" * 64,
        "projection_first_half_mean": {name: 1.0 for name in PROJECTION_NAMES},
        "projection_second_half_mean": {name: 1.0 for name in PROJECTION_NAMES},
    }
    monkeypatch.setattr(
        conditional,
        "_validated_nuts",
        lambda **kwargs: (
            nuts,
            nuts_audit,
            "3" * 64,
            "p0",
        ),
    )
    monkeypatch.setattr(
        conditional,
        "_validated_local",
        lambda **kwargs: (
            np.full(len(PROJECTION_NAMES), 0.01, dtype=np.float64),
            np.full(len(PROJECTION_NAMES), 0.01, dtype=np.float64),
            local_audit,
        ),
    )

    certificate = certify_conditional_reference(
        training_path=training_path,
        evaluation_path=evaluation_path,
        nuts_directory=tmp_path / "unused-nuts",
        local_directory=tmp_path / "unused-local",
    )

    assert certificate.record["profile"] == nuts_profile
    assert cast(Mapping[str, object], certificate.audit["nuts"])["profile"] == nuts_profile
    assert cast(Mapping[str, object], certificate.audit["local"])["profile"] == local_profile


@pytest.mark.parametrize(
    ("scenario", "topology_role"),
    [
        ("aligned", "p0"),
        ("edge-one", "p0"),
        ("edge-one", "pstar"),
        ("relocation-one", "p0"),
        ("relocation-one", "pstar"),
    ],
)
def test_recomputes_exact_certificate_from_real_local_producer_for_all_five_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    topology_role: TopologyRole,
) -> None:
    training, evaluation, nuts, local = _complete_bundles(
        tmp_path,
        monkeypatch,
        scenario=scenario,
        topology_role=topology_role,
    )

    certificate = certify_conditional_reference(
        training_path=training,
        evaluation_path=evaluation,
        nuts_directory=nuts,
        local_directory=local,
        _test_short_budget=(1, 100),
    )

    assert frozenset(certificate.record) == CONDITIONAL_REFERENCE_KEYS
    assert certificate.record["nuts_artifact_sha256"] == file_sha256(nuts / "complete.json")
    nuts_audit = cast(Mapping[str, object], certificate.audit["nuts"])
    topology_audit = cast(Mapping[str, object], certificate.audit["topology"])
    local_audit = cast(Mapping[str, object], certificate.audit["local"])
    assert nuts_audit["trace_sha256"] == file_sha256(nuts / "trace.nc")
    assert certificate.record["nuts_artifact_sha256"] != nuts_audit["trace_sha256"]
    assert topology_audit["role"] == topology_role
    assert local_audit["profile"] == "test-short"
    assert len(cast(list[object], local_audit["chains_audit"])) == 4
    projection = cast(Mapping[str, Mapping[str, float]], certificate.audit["per_projection"])["whole_domain"]
    assert projection["combined_mcse"] == pytest.approx(
        np.sqrt(projection["local_late_window_mcse"] ** 2 + projection["nuts_mcse"] ** 2),
        rel=0.0,
        abs=0.0,
    )
    replayed = validate_conditional_reference_record(
        certificate.record,
        training_path=training,
        evaluation_path=evaluation,
        nuts_directory=nuts,
        local_directory=local,
        _test_short_budget=(1, 100),
    )
    assert replayed.record == certificate.record


def test_corrupt_derived_local_diagnostics_rejected_despite_refreshed_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation, nuts, local = _complete_bundles(
        tmp_path,
        monkeypatch,
        scenario="aligned",
        topology_role="p0",
    )
    path = local / "diagnostics_input.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["common_totals"][0, 0, 0] += 1.0
    path.unlink()
    _create_npz(path, arrays)
    completion = json.loads((local / "complete.json").read_text(encoding="utf-8"))
    completion["files"]["diagnostics_input.npz"] = file_sha256(path)
    _create_json(local / "complete.json", completion)

    with pytest.raises(ValueError, match="does not rebuild from raw traces"):
        certify_conditional_reference(
            training_path=training,
            evaluation_path=evaluation,
            nuts_directory=nuts,
            local_directory=local,
            _test_short_budget=(1, 100),
        )


def test_corrupt_raw_local_trace_rejected_after_rehashing_both_catalogues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation, nuts, local = _complete_bundles(
        tmp_path,
        monkeypatch,
        scenario="aligned",
        topology_role="p0",
    )
    child = local / "chain-0"
    trace_path = child / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["log_target"][0] = np.nextafter(arrays["log_target"][0], math.inf)
    trace_path.unlink()
    _create_npz(trace_path, arrays)

    child_completion_path = child / "complete.json"
    child_completion = json.loads(child_completion_path.read_text(encoding="utf-8"))
    child_completion["files"]["trace.npz"] = file_sha256(trace_path)
    _create_json(child_completion_path, child_completion)
    top_completion_path = local / "complete.json"
    top_completion = json.loads(top_completion_path.read_text(encoding="utf-8"))
    top_completion["files"]["chain-0/complete.json"] = file_sha256(child_completion_path)
    _create_json(top_completion_path, top_completion)

    with pytest.raises(ValueError, match="does not rebuild exactly"):
        certify_conditional_reference(
            training_path=training,
            evaluation_path=evaluation,
            nuts_directory=nuts,
            local_directory=local,
            _test_short_budget=(1, 100),
        )


def test_internally_inconsistent_nuts_preflight_rejected_after_full_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation, nuts, local = _complete_bundles(
        tmp_path,
        monkeypatch,
        scenario="aligned",
        topology_role="p0",
    )
    manifest_path = nuts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preflight"][0]["log_target_difference"] = np.nextafter(0.0, 1.0)
    _create_json(manifest_path, manifest)

    checksums_path = nuts / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["files"]["manifest.json"] = file_sha256(manifest_path)
    _create_json(checksums_path, checksums)
    completion_path = nuts / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"]["manifest.json"] = file_sha256(manifest_path)
    completion["files"]["checksums.json"] = file_sha256(checksums_path)
    completion["checksums_sha256"] = file_sha256(checksums_path)
    _create_json(completion_path, completion)

    with pytest.raises(ValueError, match="preflight target parity failed"):
        certify_conditional_reference(
            training_path=training,
            evaluation_path=evaluation,
            nuts_directory=nuts,
            local_directory=local,
            _test_short_budget=(1, 100),
        )


def test_cli_publishes_certificate_audit_and_completion_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation, nuts, local = _complete_bundles(
        tmp_path,
        monkeypatch,
        scenario="aligned",
        topology_role="p0",
    )
    driver = _load_driver("mh_local_search_conditional_reference")
    output = tmp_path / "certificate"
    publication_audits: list[dict[str, str]] = []
    original_audit = driver._audit_staged_outputs

    def recording_audit(directory: Path, **kwargs: Any) -> dict[str, str]:
        assert not (directory / "complete.json").exists()
        result = cast(dict[str, str], original_audit(directory, **kwargs))
        publication_audits.append(result)
        return result

    monkeypatch.setattr(driver, "_audit_staged_outputs", recording_audit)
    arguments = argparse.Namespace(
        training=training,
        evaluation=evaluation,
        nuts_directory=nuts,
        local_directory=local,
        output_directory=output,
        source_revision=subprocess.run(
            ("git", "-C", str(Path(__file__).parents[3]), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        _test_short_budget=(1, 100),
    )

    driver.run(arguments, enforce_clean_revision=False)

    assert len(publication_audits) == 1
    assert set(publication_audits[0]) == {
        "conditional_reference.json",
        "audit.json",
    }
    assert {path.name for path in output.iterdir()} == {
        "conditional_reference.json",
        "audit.json",
        "complete.json",
    }
    completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    assert isinstance(completion["pass"], bool)
    for name, digest in completion["files"].items():
        assert file_sha256(output / name) == digest
    indexed = json.loads((output / "conditional_reference.json").read_text(encoding="utf-8"))
    indexed["worst_rhat_value"] = 9.0
    with pytest.raises(ValueError, match="differs from the recomputed"):
        validate_conditional_reference_record(
            indexed,
            training_path=training,
            evaluation_path=evaluation,
            nuts_directory=nuts,
            local_directory=local,
            _test_short_budget=(1, 100),
        )


def test_conditional_final_publication_audit_rejects_semantic_drift(
    tmp_path: Path,
) -> None:
    driver = _load_driver("mh_local_search_conditional_reference")
    output = tmp_path / "staged"
    output.mkdir()
    expected_record = {"schema": "test.record.v1", "pass": True}
    expected_audit = {"schema": "test.audit.v1", "certificate": expected_record}
    for name, payload in (
        ("conditional_reference.json", expected_record),
        ("audit.json", expected_audit),
    ):
        (output / name).write_text(driver.canonical_json(payload) + "\n", encoding="utf-8")
    first_pass = {
        name: driver.file_sha256(output / name) for name in ("conditional_reference.json", "audit.json")
    }
    changed_audit = {**expected_audit, "certificate": {**expected_record, "pass": False}}
    (output / "audit.json").write_text(
        driver.canonical_json(changed_audit) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="changed semantically"):
        driver._audit_staged_outputs(
            output,
            expected_record=expected_record,
            expected_audit=expected_audit,
            first_pass_hashes=first_pass,
        )
    assert not (output / "complete.json").exists()
