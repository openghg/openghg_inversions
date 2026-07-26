"""Focused tests for the bounded S0 NumPyro NUTS reference."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Literal, Mapping, cast

import arviz as az
import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.mh_local_search_nuts_reference import (
    PRIMARY_PROFILE,
    PROJECTION_NAMES,
    REFERENCE_CELLS,
    RETRY_PROFILE,
    START_SEEDS,
    prepare_s0_nuts_reference,
    reference_profile,
    summarize_reference_trace,
    validate_reference_trace,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    PrimaryNUTSFailure,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    materialize_replicate,
    topology_sha256,
    write_envelope,
)

_DRIVER_PATH = Path(__file__).parents[3] / "examples" / "rjmcmc" / "mh_local_search_nuts_reference.py"
ScenarioName = Literal["aligned", "edge-one", "relocation-one"]
TopologyRole = Literal["p0", "pstar"]


@pytest.fixture(scope="module")
def nuts_driver() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "mh_local_search_nuts_reference_driver",
        _DRIVER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the synthetic NUTS reference driver")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _artifacts(scenario: ScenarioName, replicate: int = 0) -> tuple[Any, Any]:
    definition = build_stage_definition("s0")
    return materialize_replicate(
        definition,
        scenario=scenario,
        replicate=replicate,
    )


@pytest.mark.parametrize(
    ("scenario", "role", "cell_name"),
    [
        ("aligned", "p0", "aligned-p0"),
        ("edge-one", "p0", "edge-one-p0"),
        ("edge-one", "pstar", "edge-one-pstar"),
        ("relocation-one", "p0", "relocation-one-p0"),
        ("relocation-one", "pstar", "relocation-one-pstar"),
    ],
)
def test_exact_five_cells_have_frozen_deterministic_starts(
    scenario: ScenarioName,
    role: TopologyRole,
    cell_name: str,
) -> None:
    training, evaluation = _artifacts(scenario)
    first = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role=role,
    )
    second = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role=role,
    )

    assert tuple(REFERENCE_CELLS) == (
        "aligned-p0",
        "edge-one-p0",
        "edge-one-pstar",
        "relocation-one-p0",
        "relocation-one-pstar",
    )
    assert first.cell_name == cell_name
    assert [start.seed for start in first.starts] == list(START_SEEDS)
    assert [start.profile for start in first.starts] == [
        "prior-mean",
        "prior-draw",
        "prior-draw",
        "prior-draw",
    ]
    for actual, replayed in zip(first.starts, second.starts, strict=True):
        assert actual.state.allocation.tiling == replayed.state.allocation.tiling
        assert actual.state.root_total == replayed.state.root_total
        assert np.array_equal(actual.state.leaf_masses, replayed.state.leaf_masses)
        assert actual.state.log_target == replayed.state.log_target
        assert np.array_equal(
            cast(Any, actual.initvals["leaf_share"]),
            cast(Any, replayed.initvals["leaf_share"]),
        )
    assert first.starts[0].state.root_total == pytest.approx(1.0)
    assert len({start.state.root_total for start in first.starts}) == 4


def test_unsupported_duplicate_and_nonrepresentative_cells_fail_closed() -> None:
    training, evaluation = _artifacts("aligned")
    with pytest.raises(ValueError, match="duplicates"):
        prepare_s0_nuts_reference(training, evaluation, topology_role="pstar")

    later_training, later_evaluation = _artifacts("edge-one", replicate=1)
    with pytest.raises(ValueError, match="replicate zero"):
        prepare_s0_nuts_reference(
            later_training,
            later_evaluation,
            topology_role="p0",
        )


def test_only_primary_and_retry_profiles_exist() -> None:
    assert reference_profile("primary") == PRIMARY_PROFILE
    assert reference_profile("retry1") == RETRY_PROFILE
    assert (
        PRIMARY_PROFILE.tune,
        PRIMARY_PROFILE.draws,
        PRIMARY_PROFILE.target_accept,
        PRIMARY_PROFILE.max_tree_depth,
        PRIMARY_PROFILE.dense_mass,
    ) == (1_000, 1_000, 0.90, 10, False)
    assert (
        RETRY_PROFILE.tune,
        RETRY_PROFILE.draws,
        RETRY_PROFILE.target_accept,
        RETRY_PROFILE.max_tree_depth,
        RETRY_PROFILE.dense_mass,
    ) == (2_000, 2_000, 0.95, 12, False)
    with pytest.raises(ValueError, match="primary"):
        reference_profile("unplanned")  # type: ignore[arg-type]


def _valid_inference_data(data: Any, draws: int) -> az.InferenceData:
    generator = np.random.Generator(np.random.PCG64(919))
    root = generator.lognormal(mean=-0.02, sigma=0.12, size=(4, draws))
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


def test_trace_validation_diagnostics_and_common_projections() -> None:
    training, evaluation = _artifacts("edge-one")
    setup = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role="pstar",
    )
    trace = _valid_inference_data(setup.data, 300)

    audit = validate_reference_trace(
        trace,
        data=setup.data,
        expected_draws=300,
    )
    summary = summarize_reference_trace(
        trace,
        data=setup.data,
        nominal_weight=training.nominal_weight,
    )

    assert audit["chains"] == 4
    assert audit["draws"] == 300
    assert summary["divergences"] == 0
    assert summary["first_failed_gate"] is None
    diagnostics = cast(Mapping[str, object], summary["root_leaf_diagnostics"])
    assert cast(str, summary["worst_rhat_variable"]) in diagnostics
    assert cast(str, summary["minimum_bulk_ess_variable"]) in diagnostics
    assert cast(str, summary["minimum_tail_ess_variable"]) in diagnostics
    projections = cast(
        Mapping[str, Mapping[str, float]],
        summary["projections"],
    )
    assert tuple(projections) == PROJECTION_NAMES
    posterior = cast(Any, getattr(trace, "posterior"))
    assert projections["whole_domain"]["mean"] == pytest.approx(
        float(np.mean(posterior["root_total"].values))
    )


def test_corrupt_trace_fails_before_it_can_be_certified() -> None:
    training, evaluation = _artifacts("aligned")
    setup = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role="p0",
    )
    trace = _valid_inference_data(setup.data, 20)
    cast(Any, getattr(trace, "posterior"))["leaf_mass"].values[0, 0, 0] *= 2.0
    with pytest.raises(RuntimeError, match="deterministic identity"):
        validate_reference_trace(
            trace,
            data=setup.data,
            expected_draws=20,
        )


def _write_artifact_pair(
    directory: Path,
    *,
    scenario: ScenarioName,
) -> tuple[Path, Path, Any, Any]:
    training, evaluation = _artifacts(scenario)
    training_path = directory / "training.json"
    evaluation_path = directory / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    return training_path, evaluation_path, training, evaluation


def test_driver_publishes_create_only_hash_certified_bundle(
    tmp_path: Path,
    nuts_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path, evaluation_path, training, evaluation = _write_artifact_pair(
        tmp_path,
        scenario="relocation-one",
    )
    output = tmp_path / "output"
    setup = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role="pstar",
    )
    writes: list[str] = []
    publication_audits: list[dict[str, str]] = []
    original_create = nuts_driver._create_json
    original_audit = nuts_driver._audit_staged_outputs

    def recording_create(path: Path, payload: Any) -> None:
        writes.append(path.name)
        original_create(path, payload)

    def recording_audit(directory: Path, **kwargs: Any) -> dict[str, str]:
        assert not (directory / nuts_driver.COMPLETION_FILENAME).exists()
        result = cast(dict[str, str], original_audit(directory, **kwargs))
        publication_audits.append(result)
        return result

    def fake_sampler(model: Any, data: Any, **kwargs: Any) -> az.InferenceData:
        assert model is fake_model
        assert data is setup.data or topology_sha256(data.tiling) == topology_sha256(setup.data.tiling)
        assert kwargs["chains"] == 4
        assert kwargs["cores"] == 1
        assert kwargs["chain_method"] == "vectorized"
        assert kwargs["draws"] == 1_000
        assert kwargs["tune"] == 1_000
        assert kwargs["target_accept"] == 0.90
        assert kwargs["max_tree_depth"] == 10
        assert kwargs["dense_mass"] is False
        assert len(kwargs["initvals"]) == 4
        return _valid_inference_data(data, kwargs["draws"])

    fake_model = object()
    monkeypatch.setattr(nuts_driver, "_create_json", recording_create)
    monkeypatch.setattr(nuts_driver, "_audit_staged_outputs", recording_audit)
    monkeypatch.setattr(
        nuts_driver,
        "require_fixed_basis_nuts_float64",
        lambda: {
            "pytensor_floatX": "float64",
            "jax_enable_x64": True,
            "jax_backend": "cpu",
        },
    )
    monkeypatch.setattr(
        nuts_driver,
        "preflight_s0_nuts_reference",
        lambda actual: (
            fake_model,
            tuple(
                {
                    "constrained_log_target": start.state.log_target,
                    "expected_log_target": start.state.log_target,
                    "log_target_difference": 0.0,
                }
                for start in actual.starts
            ),
        ),
    )
    monkeypatch.setattr(nuts_driver, "sample_fixed_basis_nuts", fake_sampler)
    arguments = nuts_driver.parser().parse_args(
        [
            "--training",
            str(training_path),
            "--evaluation",
            str(evaluation_path),
            "--topology",
            "pstar",
            "--profile",
            "primary",
            "--output-directory",
            str(output),
            "--source-revision",
            "a" * 40,
            "--cell-id",
            training.cell_id,
            "--definition-sha256",
            training.definition_sha256,
        ]
    )

    summary = nuts_driver.run(arguments, enforce_clean_revision=False)

    assert summary["status"] == "complete"
    assert summary["profile"] == "primary"
    assert summary["cell_id"] == training.cell_id
    assert summary["definition_sha256"] == training.definition_sha256
    assert summary["topology_sha256"] == topology_sha256(setup.data.tiling)
    assert summary["nuts_artifact_sha256"] == nuts_driver.file_sha256(output / nuts_driver.TRACE_FILENAME)
    assert len(publication_audits) == 1
    assert set(publication_audits[0]) == {
        nuts_driver.TRACE_FILENAME,
        nuts_driver.MANIFEST_FILENAME,
        nuts_driver.SUMMARY_FILENAME,
        nuts_driver.CHECKSUM_FILENAME,
    }
    assert writes[-1] == nuts_driver.COMPLETION_FILENAME
    assert {path.name for path in output.iterdir()} == {
        nuts_driver.TRACE_FILENAME,
        nuts_driver.MANIFEST_FILENAME,
        nuts_driver.SUMMARY_FILENAME,
        nuts_driver.CHECKSUM_FILENAME,
        nuts_driver.COMPLETION_FILENAME,
    }
    completion = json.loads((output / nuts_driver.COMPLETION_FILENAME).read_text(encoding="utf-8"))
    for name, digest in completion["files"].items():
        assert nuts_driver.file_sha256(output / name) == digest
    with pytest.raises(FileExistsError, match="already exists"):
        nuts_driver.run(arguments, enforce_clean_revision=False)


def test_nuts_final_publication_audit_rejects_semantic_drift(
    tmp_path: Path,
    nuts_driver: ModuleType,
) -> None:
    training, evaluation = _artifacts("aligned")
    setup = prepare_s0_nuts_reference(training, evaluation, topology_role="p0")
    trace = _valid_inference_data(setup.data, 20)
    output = tmp_path / "staged"
    output.mkdir()
    trace_path = output / nuts_driver.TRACE_FILENAME
    trace.to_netcdf(trace_path)
    trace_audit = validate_reference_trace(trace, data=setup.data, expected_draws=20)
    scientific_summary = summarize_reference_trace(
        trace,
        data=setup.data,
        nominal_weight=training.nominal_weight,
    )
    manifest = {"schema": "test.nuts.manifest.v1"}
    summary = {
        "schema": "test.nuts.summary.v1",
        "nuts_artifact_sha256": nuts_driver.file_sha256(trace_path),
        **scientific_summary,
    }
    nuts_driver._create_json(output / nuts_driver.MANIFEST_FILENAME, manifest)
    nuts_driver._create_json(output / nuts_driver.SUMMARY_FILENAME, summary)
    first_pass = {
        name: nuts_driver.file_sha256(output / name)
        for name in (
            nuts_driver.TRACE_FILENAME,
            nuts_driver.MANIFEST_FILENAME,
            nuts_driver.SUMMARY_FILENAME,
        )
    }
    checksums = {
        "schema": "openghg_inversions.mh_local_search_nuts_checksums.v1",
        "files": first_pass,
    }
    nuts_driver._create_json(output / nuts_driver.CHECKSUM_FILENAME, checksums)
    changed_summary = {**summary, "divergences": int(summary["divergences"]) + 1}
    (output / nuts_driver.SUMMARY_FILENAME).write_text(
        nuts_driver.canonical_json(changed_summary) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="changed semantically"):
        nuts_driver._audit_staged_outputs(
            output,
            expected_manifest=manifest,
            expected_summary=summary,
            expected_checksums=checksums,
            first_pass_hashes=first_pass,
            data=setup.data,
            nominal_weight=training.nominal_weight,
            expected_draws=20,
            expected_trace_audit=trace_audit,
        )
    assert not (output / nuts_driver.COMPLETION_FILENAME).exists()


def test_retry1_driver_requires_and_cites_failed_primary_completion(
    tmp_path: Path,
    nuts_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path, evaluation_path, training, _ = _write_artifact_pair(
        tmp_path,
        scenario="aligned",
    )
    primary = tmp_path / "primary-nuts"
    primary.mkdir()
    (primary / "complete.json").write_text("{}\n", encoding="utf-8")
    source_digest = "e" * 64
    calls: list[Path] = []

    def validate_source(**kwargs: Any) -> PrimaryNUTSFailure:
        calls.append(cast(Path, kwargs["primary_nuts_directory"]))
        return PrimaryNUTSFailure(
            completion_sha256=source_digest,
            first_failed_gate="bulk_ess",
        )

    monkeypatch.setattr(
        nuts_driver,
        "validate_primary_nuts_retry_source",
        validate_source,
    )
    monkeypatch.setattr(
        nuts_driver,
        "require_fixed_basis_nuts_float64",
        lambda: {
            "pytensor_floatX": "float64",
            "jax_enable_x64": True,
            "jax_backend": "cpu",
        },
    )
    monkeypatch.setattr(
        nuts_driver,
        "preflight_s0_nuts_reference",
        lambda setup: (object(), ()),
    )
    arguments = nuts_driver.parser().parse_args(
        [
            "--training",
            str(training_path),
            "--evaluation",
            str(evaluation_path),
            "--topology",
            "p0",
            "--profile",
            "retry1",
            "--primary-nuts-directory",
            str(primary),
            "--output-directory",
            str(tmp_path / "retry"),
            "--source-revision",
            "a" * 40,
            "--cell-id",
            training.cell_id,
            "--definition-sha256",
            training.definition_sha256,
            "--dry-run",
        ]
    )
    result = nuts_driver.run(arguments, enforce_clean_revision=False)
    manifest = cast(Mapping[str, object], result["manifest"])
    assert calls == [primary]
    assert manifest["schema"] == "openghg_inversions.mh_local_search_nuts_manifest.v2"
    assert manifest["retry_source_nuts_completion_sha256"] == source_digest
    assert manifest["retry_source_first_failed_gate"] == "bulk_ess"

    arguments.primary_nuts_directory = None
    with pytest.raises(ValueError, match="requires --primary-nuts-directory"):
        nuts_driver.run(arguments, enforce_clean_revision=False)
    arguments.profile = "primary"
    arguments.primary_nuts_directory = primary
    with pytest.raises(ValueError, match="cannot cite retry lineage"):
        nuts_driver.run(arguments, enforce_clean_revision=False)


def test_real_two_draw_vectorized_numpyro_and_all_start_parity(
    tmp_path: Path,
) -> None:
    """Exercise the real backend in an isolated float64 CPU subprocess."""
    code = """
from openghg_inversions.experimental.rjmcmc.fixed_basis_nuts import sample_fixed_basis_nuts
from openghg_inversions.experimental.rjmcmc.mh_local_search_nuts_reference import (
    preflight_s0_nuts_reference,
    prepare_s0_nuts_reference,
    validate_reference_trace,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    build_stage_definition,
    materialize_replicate,
)
training, evaluation = materialize_replicate(
    build_stage_definition("s0"),
    scenario="aligned",
    replicate=0,
)
setup = prepare_s0_nuts_reference(training, evaluation, topology_role="p0")
model, audits = preflight_s0_nuts_reference(setup)
assert len(audits) == 4
assert all(abs(float(item["log_target_difference"])) <= float(item["log_target_absolute_tolerance"]) for item in audits)
trace = sample_fixed_basis_nuts(
    model,
    setup.data,
    draws=2,
    tune=2,
    seed=64100,
    target_accept=0.90,
    chains=4,
    cores=1,
    chain_method="vectorized",
    progressbar=False,
    max_tree_depth=10,
    dense_mass=False,
    initvals=tuple(start.initvals for start in setup.starts),
)
audit = validate_reference_trace(trace, data=setup.data, expected_draws=2)
assert audit["chains"] == 4 and audit["draws"] == 2
print("ok")
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTENSOR_FLAGS"] = (
        f"floatX=float64,warn_float64=ignore,base_compiledir={tmp_path / 'pytensor'}"
    )
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_DRIVER_PATH.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, (
        f"real vectorized NumPyro probe failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip().endswith("ok")
