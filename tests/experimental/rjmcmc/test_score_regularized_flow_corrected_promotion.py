"""Focused tests for corrected all-six promotion selection and publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from examples.rjmcmc import score_regularized_flow_corrected_merge as merger
from examples.rjmcmc import (
    score_regularized_flow_corrected_promotion_merge as promotion,
)


SOURCE_GIT_REVISION = "a" * 40
EVALUATION_GIT_REVISION = "b" * 40


def _selected_scientific_evaluation() -> dict[str, Any]:
    return {
        "scientific_metrics_interpretable": True,
        "normalization": {
            "normalized_density_by_flow_and_jacobian_construction": True,
            "finite_on_complete_metric_grid": True,
        },
        "vectorized_public_likelihood_parity": {"pass": True},
        "pointwise": {
            "prior_weighted_median_absolute_log_likelihood_error_nat": 0.01,
            "exact_posterior_weighted_p99_absolute_log_likelihood_error_nat": 0.02,
        },
        "gradient": {"scaled_error": 0.01},
        "evidence": {
            "absolute_learned_error_from_adaptive_reference_nat": 0.01,
        },
        "posterior": {
            "mean_error_reference_sd": 0.01,
            "sd_relative_error": 0.01,
            "interval_endpoint_error_reference_sd": 0.01,
        },
    }


def _attempt(
    init_index: int,
    model_selection_nll: float,
    *,
    selected_quality: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "attempt_id": f"attempt-{init_index}",
        "initialization_index": init_index,
        "target": {
            "case_id": "near_gaussian__two_cell__root",
            "sample_count": 4_096,
        },
        "candidate": {
            "config_id": promotion.PRIMARY_CONFIG_ID,
            "architecture_rank": 3,
        },
        "model_selection": {
            "nll_nat_per_dimension": model_selection_nll,
        },
    }
    if selected_quality:
        report["model_selection"].update(
            {
                "nll_nat_per_draw": 1.0,
                "nll_iid_mcse_nat_per_draw": 0.01,
                "fisher_scaled_partial_score_risk": 1.0,
                "fisher_scaled_observation_score_risk": 1.0,
            }
        )
        report["reporting_test"] = {
            "nll_nat_per_draw": 1.01,
            "nll_iid_mcse_nat_per_draw": 0.01,
            "fisher_scaled_partial_score_risk": 1.0,
            "fisher_scaled_observation_score_risk": 1.0,
        }
        report["scientific_evaluation"] = _selected_scientific_evaluation()
    else:
        # Deliberately nonsensical held-out/oracle fields. They must not affect
        # selection because this start has a worse independent selection NLL.
        report["reporting_test"] = {"poison": object()}
        report["scientific_evaluation"] = {"poison": object()}
        report["approximate_evidence"] = -1.0e300
    return report


def test_start_selection_uses_only_independent_model_selection_nll() -> None:
    reports = [
        _attempt(0, 1.4, selected_quality=False),
        _attempt(1, 0.9, selected_quality=True),
        _attempt(2, 1.2, selected_quality=False),
        _attempt(3, 1.1, selected_quality=False),
    ]

    selected = promotion._selected_row(reports)

    assert selected["selected_initialization_index"] == 1
    assert selected["selected_attempt_id"] == "attempt-1"
    assert selected["model_selection_nll_by_initialization"] == {
        "0": 1.4,
        "1": 0.9,
        "2": 1.2,
        "3": 1.1,
    }


def test_start_selection_tie_breaks_lowest_init_and_rejects_nonfinite_nll() -> None:
    reports = [
        _attempt(0, 1.1, selected_quality=False),
        _attempt(1, 0.9, selected_quality=True),
        _attempt(2, 0.9, selected_quality=False),
        _attempt(3, 1.2, selected_quality=False),
    ]
    assert promotion._selected_row(reports)["selected_initialization_index"] == 1

    reports[3]["model_selection"]["nll_nat_per_dimension"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        promotion._selected_row(reports)


def _authenticated(
    report: dict[str, Any],
    *,
    report_sha: str,
    artifact_sha: str,
) -> merger._AuthenticatedAttempt:
    report["sha256"] = report_sha
    return merger._AuthenticatedAttempt(
        report=report,
        report_file_sha256=f"file-{report_sha}",
        serialized_artifact_file_sha256=artifact_sha,
    )


def test_only_model_selection_winner_is_evaluated_after_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "near_gaussian__two_cell__root"
    public_evidence = {
        domain: {
            "base_seed": 1731,
            "domain": domain,
        }
        for domain in promotion.experiment.domains.PUBLIC_DOMAINS
    }
    authenticated: list[merger._AuthenticatedAttempt] = []
    for init_index, nll in enumerate((1.4, 0.9, 1.2, 1.1)):
        report = _attempt(
            init_index,
            nll,
            selected_quality=False,
        )
        report["target"] = {"case_id": case_id, "sample_count": 4_096}
        report["domain_evidence"] = public_evidence
        report["execution"] = {
            "output_root": str(tmp_path / f"attempt-{init_index}"),
        }
        report["training_loss_scale_diagnostics"] = {
            "partial_score_rms": 2.0,
            "observation_score_rms_by_coordinate": [3.0, 4.0],
        }
        report["runtime_identity_sha256"] = f"runtime-{init_index}"
        report["execution_identity_sha256"] = f"execution-{init_index}"
        report["reporting_test"] = {
            "status": "deferred_until_model_selection_only_start_selection",
            "selection_uses_reporting_data": False,
        }
        report["scientific_evaluation"] = {
            "status": "deferred_until_model_selection_only_start_selection",
            "selection_uses_oracle_evidence": False,
        }
        attempt_root = Path(report["execution"]["output_root"])
        attempt_root.mkdir()
        artifact_bytes = f"artifact-{init_index}".encode("ascii")
        artifact_path = attempt_root / "artifact.bin"
        artifact_path.write_bytes(artifact_bytes)
        authenticated.append(
            _authenticated(
                report,
                report_sha=f"payload-{init_index}",
                artifact_sha=hashlib.sha256(artifact_bytes).hexdigest(),
            )
        )

    selected = promotion._select_attempt(authenticated)
    evaluated_artifacts: list[bytes] = []

    def deserialize(
        artifact_bytes: bytes,
        *,
        expected_sha256: str,
    ) -> SimpleNamespace:
        assert hashlib.sha256(artifact_bytes).hexdigest() == expected_sha256
        evaluated_artifacts.append(artifact_bytes)
        return SimpleNamespace(flow="selected-flow")

    expected_reporting_evidence = public_evidence[
        promotion.experiment.domains.DEVELOPMENT_REPORTING_TEST_DOMAIN
    ]
    monkeypatch.setattr(
        promotion.experiment.ScoreRegularizedRootFlow,
        "from_bytes",
        deserialize,
    )
    monkeypatch.setattr(
        promotion.experiment.domains,
        "simulate_tiny_score_domain",
        lambda *args, **kwargs: SimpleNamespace(
            evidence=SimpleNamespace(
                payload=lambda: expected_reporting_evidence,
            )
        ),
    )
    monkeypatch.setattr(
        promotion.experiment,
        "_score_domain",
        lambda *args, **kwargs: {"reporting": "selected-only"},
    )
    scientific_calls: list[tuple[str, bool, dict[int, object] | None, dict[str, object]]] = []

    def scientific(
        _artifact: object,
        observed_case_id: str,
        _reference: object,
        *,
        interpolate_posterior_quantiles: bool,
        exact_grid_cache: dict[int, object] | None = None,
        promotion_oracle_case: dict[str, object],
    ) -> dict[str, object]:
        assert exact_grid_cache is None or isinstance(
            exact_grid_cache,
            dict,
        )
        scientific_calls.append(
            (
                observed_case_id,
                interpolate_posterior_quantiles,
                exact_grid_cache,
                promotion_oracle_case,
            )
        )
        return {"scientific": "selected-only"}

    monkeypatch.setattr(
        promotion.experiment,
        "_scientific_grid_evaluation",
        scientific,
    )
    oracle_case = {
        "reference": {"sha256": "oracle-reference"},
    }
    oracle_bundle = {
        "selected_cases": {
            case_id: oracle_case,
        }
    }

    evaluated = promotion._evaluate_selected(selected, oracle_bundle)

    assert selected.report["initialization_index"] == 1
    assert evaluated_artifacts == [b"artifact-1"]
    assert scientific_calls == [(case_id, True, None, oracle_case)]
    assert evaluated["reporting_test"] == {"reporting": "selected-only"}
    assert evaluated["scientific_evaluation"] == {
        "scientific": "selected-only",
    }
    assert evaluated["post_selection_evaluation"] == {
        "reporting_and_oracle_evaluated_after_selection": True,
        "selection_used_reporting_data": False,
        "selection_used_oracle_evidence": False,
        "artifact_path": str(tmp_path / "attempt-1" / "artifact.bin"),
        "artifact_file_sha256": hashlib.sha256(b"artifact-1").hexdigest(),
        "report_file_sha256": "file-payload-1",
    }
    assert len(selected.all_start_manifest) == 4
    assert {row["serialized_artifact_file_sha256"] for row in selected.all_start_manifest} == {
        hashlib.sha256(f"artifact-{index}".encode("ascii")).hexdigest() for index in range(4)
    }
    assert {row["runtime_identity_sha256"] for row in selected.all_start_manifest} == {
        f"runtime-{index}" for index in range(4)
    }
    assert {row["execution_identity_sha256"] for row in selected.all_start_manifest} == {
        f"execution-{index}" for index in range(4)
    }


def test_post_selection_evaluation_rejects_precomputed_reporting_or_oracle(
    tmp_path: Path,
) -> None:
    report = _attempt(0, 0.9, selected_quality=True)
    report["execution"] = {"output_root": str(tmp_path)}
    selected = promotion.SelectedAttempt(
        report=report,
        report_file_sha256="a" * 64,
        artifact_file_sha256="b" * 64,
        all_start_manifest=(),
    )
    with pytest.raises(
        ValueError,
        match="evaluated reporting data before selection",
    ):
        promotion._evaluate_selected(selected, {"selected_cases": {}})

    report["reporting_test"] = {
        "status": "deferred_until_model_selection_only_start_selection",
        "selection_uses_reporting_data": False,
    }
    with pytest.raises(
        ValueError,
        match="evaluated oracle evidence before selection",
    ):
        promotion._evaluate_selected(selected, {"selected_cases": {}})


def test_case_evaluation_uses_one_disposable_spawned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {
        promotion.PRIMARY_CONFIG_ID: promotion.SelectedAttempt(
            report={},
            report_file_sha256="a" * 64,
            artifact_file_sha256="b" * 64,
            all_start_manifest=(),
        )
    }
    oracle = {"selected_cases": {}}
    expected = ([{"case_id": "case"}], None)
    calls: list[tuple[object, object, bool]] = []
    context = object()

    class FakeFuture:
        def result(self) -> tuple[list[dict[str, str]], None]:
            return expected

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 1
            assert mp_context is context

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self,
            function: object,
            observed_selected: object,
            observed_oracle: object,
            *,
            development: bool,
        ) -> FakeFuture:
            assert function is promotion._evaluate_case_group
            calls.append((observed_selected, observed_oracle, development))
            return FakeFuture()

    def get_context(method: str) -> object:
        assert method == "spawn"
        return context

    monkeypatch.setattr(promotion.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(promotion, "ProcessPoolExecutor", FakeExecutor)

    observed = promotion._evaluate_case_isolated(
        selected,
        oracle,
        development=False,
    )

    assert observed == expected
    assert calls == [(selected, oracle, False)]


def test_case_group_reuses_one_cache_and_preserves_frozen_config_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "near_gaussian__two_cell__root"

    def selected(config_id: str) -> promotion.SelectedAttempt:
        return promotion.SelectedAttempt(
            report={
                "target": {"case_id": case_id, "sample_count": 4_096},
                "candidate": {"config_id": config_id},
            },
            report_file_sha256=f"report-{config_id}",
            artifact_file_sha256=f"artifact-{config_id}",
            all_start_manifest=(),
        )

    selected_by_config = {
        promotion.PRIMARY_CONFIG_ID: selected(promotion.PRIMARY_CONFIG_ID),
        promotion.COMPARATOR_CONFIG_ID: selected(promotion.COMPARATOR_CONFIG_ID),
    }
    evaluation_calls: list[tuple[str, dict[int, object]]] = []

    def evaluate(
        item: promotion.SelectedAttempt,
        _oracle: dict[str, object],
        *,
        exact_grid_cache: dict[int, object] | None,
    ) -> dict[str, object]:
        assert exact_grid_cache is not None
        config_id = str(item.report["candidate"]["config_id"])
        evaluation_calls.append((config_id, exact_grid_cache))
        return {
            "target": item.report["target"],
            "candidate": item.report["candidate"],
        }

    monkeypatch.setattr(promotion, "_evaluate_selected", evaluate)
    monkeypatch.setattr(
        promotion,
        "_post_selection_row",
        lambda item, evaluated: {
            "case_id": evaluated["target"]["case_id"],
            "sample_count": evaluated["target"]["sample_count"],
            "config_id": item.report["candidate"]["config_id"],
        },
    )
    monkeypatch.setattr(
        promotion,
        "_primary_comparator_checks",
        lambda primary, comparator: (
            {
                "ordered_pair": (
                    primary["candidate"]["config_id"] == promotion.PRIMARY_CONFIG_ID
                    and comparator["candidate"]["config_id"] == promotion.COMPARATOR_CONFIG_ID
                )
            },
            {"diagnostic": 1.0},
        ),
    )

    rows, comparison = promotion._evaluate_case_group(
        selected_by_config,
        {"selected_cases": {}},
        development=True,
    )

    assert [config_id for config_id, _ in evaluation_calls] == sorted(selected_by_config)
    assert evaluation_calls[0][1] is evaluation_calls[1][1]
    assert [row["config_id"] for row in rows] == sorted(selected_by_config)
    assert comparison == {
        "case_id": case_id,
        "sample_count": 4_096,
        "checks": {"ordered_pair": True},
        "diagnostics": {"diagnostic": 1.0},
        "pass": True,
    }


def test_case_evaluation_spawn_propagates_worker_failure() -> None:
    with pytest.raises(
        ValueError,
        match="development case does not contain",
    ):
        promotion._evaluate_case_isolated(
            {},
            {"selected_cases": {}},
            development=True,
        )


def test_isolated_case_result_must_rebind_selected_hashes() -> None:
    selected = promotion.SelectedAttempt(
        report={
            "attempt_id": "attempt-2",
            "initialization_index": 2,
            "sha256": "d" * 64,
            "runtime_identity_sha256": "e" * 64,
            "execution_identity_sha256": "f" * 64,
            "target": {
                "case_id": "near_gaussian__two_cell__root",
                "sample_count": 4_096,
            },
            "candidate": {"config_id": promotion.PRIMARY_CONFIG_ID},
        },
        report_file_sha256="a" * 64,
        artifact_file_sha256="b" * 64,
        all_start_manifest=(),
    )
    row = {
        "case_id": "near_gaussian__two_cell__root",
        "sample_count": 4_096,
        "config_id": promotion.PRIMARY_CONFIG_ID,
        "selected_initialization_index": 2,
        "selected_attempt_id": "attempt-2",
        "selected_report_payload_sha256": "d" * 64,
        "selected_report_file_sha256": "a" * 64,
        "selected_artifact_file_sha256": "c" * 64,
        "selected_runtime_identity_sha256": "e" * 64,
        "selected_execution_identity_sha256": "f" * 64,
        "all_start_manifest": [],
        "model_selection_nll_by_initialization": {},
    }

    with pytest.raises(
        ValueError,
        match="does not bind its selected input",
    ):
        promotion._validate_case_result(
            [row],
            None,
            {promotion.PRIMARY_CONFIG_ID: selected},
            case_id="near_gaussian__two_cell__root",
            sample_count=4_096,
            development=False,
        )
    row["selected_artifact_file_sha256"] = "b" * 64
    row["selected_attempt_id"] = "tampered-attempt"
    with pytest.raises(
        ValueError,
        match="does not bind its selected input",
    ):
        promotion._validate_case_result(
            [row],
            None,
            {promotion.PRIMARY_CONFIG_ID: selected},
            case_id="near_gaussian__two_cell__root",
            sample_count=4_096,
            development=False,
        )


def test_merge_selects_every_group_before_oracle_and_streams_by_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_id = "promotion_confirmation_s16384_seed2731"
    attempt_tag = "streaming-contract"
    cases = (
        "near_gaussian__two_cell__root",
        "skewed__four_cell__root",
    )
    matrix = tuple(
        ("promotion", case_id, 16_384, promotion.PRIMARY_CONFIG_ID, init_index)
        for case_id in cases
        for init_index in promotion.FIXED_INITIALIZATION_INDICES
    )
    matrices = dict(promotion.arrays.MATRICES)
    matrices[matrix_id] = matrix
    counts = dict(promotion.arrays.EXPECTED_MATRIX_ATTEMPT_COUNTS)
    counts[matrix_id] = len(matrix)
    monkeypatch.setattr(promotion.arrays, "MATRICES", matrices)
    monkeypatch.setattr(
        promotion.arrays,
        "EXPECTED_MATRIX_ATTEMPT_COUNTS",
        counts,
    )
    spec = promotion.arrays.PROMOTION_MATRIX_SPECS[matrix_id]
    events: list[str] = []

    def load_attempt(
        _attempt_root: Path,
        *,
        source_git_revision: str,
        matrix_id: str,
        array_task_id: int,
        matrix_row: tuple[str, str, int, str, int],
        attempt_tag: str,
    ) -> merger._AuthenticatedAttempt:
        assert source_git_revision == SOURCE_GIT_REVISION
        assert matrix_id == "promotion_confirmation_s16384_seed2731"
        assert attempt_tag == "streaming-contract"
        assert matrix_row == matrix[array_task_id]
        events.append(f"load-{array_task_id}")
        _, case_id, sample_count, config_id, init_index = matrix_row
        report = {
            "attempt_id": f"{case_id}-{init_index}",
            "initialization_index": init_index,
            "sha256": f"{array_task_id:064x}",
            "runtime_identity_sha256": f"{array_task_id + 100:064x}",
            "execution_identity_sha256": f"{array_task_id + 200:064x}",
            "target": {
                "case_id": case_id,
                "sample_count": sample_count,
            },
            "candidate": {
                "config_id": config_id,
                **{
                    key: spec[key]
                    for key in (
                        "learning_rate",
                        "batch_size",
                        "microbatch_size",
                        "maximum_total_epochs",
                        "patience",
                    )
                },
            },
            "model_selection": {
                "nll_nat_per_dimension": float(10 - init_index),
            },
            "domain_evidence": {
                domain: {"base_seed": 2731} for domain in promotion.experiment.domains.PUBLIC_DOMAINS
            },
        }
        return merger._AuthenticatedAttempt(
            report=report,
            report_file_sha256=f"{array_task_id + 300:064x}",
            serialized_artifact_file_sha256=f"{array_task_id + 400:064x}",
        )

    original_select = promotion._select_attempt

    def select_attempt(
        attempts: list[merger._AuthenticatedAttempt],
    ) -> promotion.SelectedAttempt:
        events.append(f"select-{attempts[0].report['target']['case_id']}")
        return original_select(attempts)

    def load_oracle(
        _path: Path,
        source_git_revision: str,
        *,
        promotion: bool,
    ) -> dict[str, object]:
        assert source_git_revision == SOURCE_GIT_REVISION
        assert promotion is True
        assert sum(event.startswith("select-") for event in events) == len(cases)
        assert not any(event.startswith("evaluate-") for event in events)
        events.append("oracle")
        return {}

    def evaluate_case(
        selected_by_config: dict[str, promotion.SelectedAttempt],
        _oracle: dict[str, object],
        *,
        development: bool,
    ) -> tuple[list[dict[str, object]], None]:
        assert development is False
        assert "oracle" in events
        assert set(selected_by_config) == {promotion.PRIMARY_CONFIG_ID}
        selected = selected_by_config[promotion.PRIMARY_CONFIG_ID]
        report = selected.report
        events.append(f"evaluate-{report['target']['case_id']}")
        return [
            {
                "case_id": report["target"]["case_id"],
                "sample_count": report["target"]["sample_count"],
                "config_id": report["candidate"]["config_id"],
                "selected_initialization_index": report["initialization_index"],
                "selected_attempt_id": report["attempt_id"],
                "selected_report_payload_sha256": report["sha256"],
                "selected_report_file_sha256": selected.report_file_sha256,
                "selected_artifact_file_sha256": selected.artifact_file_sha256,
                "selected_runtime_identity_sha256": report["runtime_identity_sha256"],
                "selected_execution_identity_sha256": report["execution_identity_sha256"],
                "all_start_manifest": list(selected.all_start_manifest),
                "model_selection_nll_by_initialization": {
                    str(item["initialization_index"]): item["model_selection_nll_nat_per_dimension"]
                    for item in selected.all_start_manifest
                },
                "pass": True,
            }
        ], None

    monkeypatch.setattr(promotion.exploratory_merge, "_load_attempt", load_attempt)
    monkeypatch.setattr(promotion, "_select_attempt", select_attempt)
    monkeypatch.setattr(promotion.experiment, "_load_oracle_bundle", load_oracle)
    monkeypatch.setattr(promotion, "_evaluate_case_isolated", evaluate_case)

    result = promotion.merge(
        matrix_id,
        attempt_tag,
        SOURCE_GIT_REVISION,
        EVALUATION_GIT_REVISION,
        tmp_path,
    )

    assert result["complete"] is True
    assert result["artifact_source_git_revision"] == SOURCE_GIT_REVISION
    assert result["evaluation_source_git_revision"] == EVALUATION_GIT_REVISION
    assert [row["case_id"] for row in result["selected_rows"]] == sorted(cases)
    oracle_index = events.index("oracle")
    assert all(
        events.index(f"select-{case_id}") < oracle_index < events.index(f"evaluate-{case_id}")
        for case_id in cases
    )


def test_promotion_completion_binds_exact_summary_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_id = "promotion_confirmation_s16384_seed2731"
    attempt_tag = "promotion-test"
    without_sha = {
        "schema": promotion.SCHEMA,
        "matrix_id": matrix_id,
        "attempt_tag": attempt_tag,
        "artifact_source_git_revision": SOURCE_GIT_REVISION,
        "evaluation_source_git_revision": EVALUATION_GIT_REVISION,
        "complete": True,
        "promotion_pass": True,
    }
    result = {
        **without_sha,
        "sha256": promotion._sha256_json(without_sha),
    }
    monkeypatch.setattr(promotion, "merge", lambda *_args: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_regularized_flow_corrected_promotion_merge.py",
            "--matrix-id",
            matrix_id,
            "--attempt-tag",
            attempt_tag,
            "--artifact-source-git-revision",
            SOURCE_GIT_REVISION,
            "--evaluation-source-git-revision",
            EVALUATION_GIT_REVISION,
            "--output-root",
            str(tmp_path),
        ],
    )

    promotion.main()

    summary_root = tmp_path / "promotion_summaries" / f"{matrix_id}__{attempt_tag}"
    summary_path = summary_root / "summary.json"
    completion_path = summary_root / "COMPLETE.json"
    summary_bytes = summary_path.read_bytes()
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    assert completion["artifact_source_git_revision"] == SOURCE_GIT_REVISION
    assert completion["evaluation_source_git_revision"] == EVALUATION_GIT_REVISION
    assert completion["matrix_id"] == matrix_id
    assert completion["attempt_tag"] == attempt_tag
    assert completion["summary_path"] == str(summary_path)
    assert completion["summary_payload_sha256"] == result["sha256"]
    assert completion["summary_file_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert completion["completion_marker_published_last"] is True

    with pytest.raises(FileExistsError, match="refusing to replace"):
        promotion.main()


def test_incomplete_marker_binds_both_source_revisions(tmp_path: Path) -> None:
    without_sha = {
        "schema": promotion.SCHEMA,
        "matrix_id": "promotion_development_s4096",
        "attempt_tag": "incomplete",
        "artifact_source_git_revision": SOURCE_GIT_REVISION,
        "evaluation_source_git_revision": EVALUATION_GIT_REVISION,
        "complete": False,
        "promotion_pass": False,
    }
    result = {
        **without_sha,
        "sha256": promotion._sha256_json(without_sha),
    }
    summary_root = tmp_path / "incomplete"

    with pytest.raises(RuntimeError, match="summary is incomplete"):
        promotion._publish(result, summary_root=summary_root)

    marker = json.loads((summary_root / "INCOMPLETE.json").read_text(encoding="ascii"))
    assert marker["artifact_source_git_revision"] == SOURCE_GIT_REVISION
    assert marker["evaluation_source_git_revision"] == EVALUATION_GIT_REVISION
    assert marker["completion_marker_published"] is False
    assert not (summary_root / "COMPLETE.json").exists()
