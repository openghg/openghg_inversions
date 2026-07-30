"""Focused tests for corrected NLE cross-matrix promotion certification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.rjmcmc import (
    score_regularized_flow_corrected_promotion_certify as certify,
)
from examples.rjmcmc import (
    score_regularized_flow_corrected_promotion_merge as promotion,
)


SOURCE_GIT_REVISION = "a" * 40


def _summary(
    matrix_id: str,
    *,
    sample_count: int,
    base_seed: int,
    omit_case: str | None = None,
    promotion_pass: bool = True,
) -> dict[str, object]:
    rows = [
        {
            "case_id": case_id,
            "sample_count": sample_count,
            "config_id": promotion.PRIMARY_CONFIG_ID,
        }
        for case_id in promotion.experiment.PROMOTION_CASES
        if case_id != omit_case
    ]
    without_sha: dict[str, object] = {
        "schema": promotion.SCHEMA,
        "matrix_id": matrix_id,
        "attempt_tag": f"{matrix_id}-tag",
        "source_git_revision": SOURCE_GIT_REVISION,
        "expected_base_seed": base_seed,
        "complete": True,
        "selected_rows": rows,
        "promotion_pass": promotion_pass,
    }
    return {
        **without_sha,
        "sha256": promotion._sha256_json(without_sha),
    }


def _publish_summary(
    tmp_path: Path,
    matrix_id: str,
    *,
    base_seed: int,
    omit_case: str | None = None,
    promotion_pass: bool = True,
) -> Path:
    sample_count = 4_096 if matrix_id.endswith("s4096") else 16_384
    root = tmp_path / matrix_id
    promotion._publish(
        _summary(
            matrix_id,
            sample_count=sample_count,
            base_seed=base_seed,
            omit_case=omit_case,
            promotion_pass=promotion_pass,
        ),
        summary_root=root,
    )
    return root


def _passing_cross_size_rows() -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "pass": True,
        }
        for case_id in promotion.experiment.PROMOTION_CASES
    ]


def _disable_numerical_cross_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        certify.experiment,
        "_load_oracle_bundle",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        certify,
        "_cross_size_rows",
        lambda *args, **kwargs: _passing_cross_size_rows(),
    )


def test_certifier_requires_exact_development_pair_or_all_five_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_numerical_cross_size(monkeypatch)
    development = [
        _publish_summary(
            tmp_path,
            "promotion_development_s4096",
            base_seed=1731,
        ),
        _publish_summary(
            tmp_path,
            "promotion_development_s16384",
            base_seed=1731,
        ),
    ]

    development_certificate = certify.certify(
        development,
        source_git_revision=SOURCE_GIT_REVISION,
        output_root=tmp_path,
    )
    assert development_certificate["phase"] == "development"
    assert development_certificate["input_matrix_ids"] == sorted(certify.DEVELOPMENT_MATRIX_IDS)
    assert development_certificate["certificate_pass"] is True
    assert development_certificate["eligible_for_confirmation"] is True
    assert development_certificate["promotion_pass"] is False

    with pytest.raises(ValueError, match="not the frozen development or final set"):
        certify.certify(
            development[:1],
            source_git_revision=SOURCE_GIT_REVISION,
            output_root=tmp_path,
        )
    with pytest.raises(ValueError, match="duplicate matrix IDs"):
        certify.certify(
            [development[0], development[0]],
            source_git_revision=SOURCE_GIT_REVISION,
            output_root=tmp_path,
        )

    confirmations = [
        _publish_summary(
            tmp_path,
            f"promotion_confirmation_s16384_seed{seed}",
            base_seed=seed,
        )
        for seed in (2731, 3731, 4731)
    ]
    final_certificate = certify.certify(
        [*development, *confirmations],
        source_git_revision=SOURCE_GIT_REVISION,
        output_root=tmp_path,
    )
    assert final_certificate["phase"] == "final_confirmation"
    assert final_certificate["input_matrix_ids"] == sorted(certify.FINAL_MATRIX_IDS)
    assert final_certificate["checks"]["confirmation_seed_set_is_frozen"]
    assert final_certificate["certificate_pass"] is True
    assert final_certificate["eligible_for_confirmation"] is False
    assert final_certificate["promotion_pass"] is True


def test_certifier_fails_closed_on_missing_case_bad_seed_or_failed_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_numerical_cross_size(monkeypatch)
    missing_case = promotion.experiment.PROMOTION_CASES[-1]
    development = [
        _publish_summary(
            tmp_path,
            "promotion_development_s4096",
            base_seed=1731,
            omit_case=missing_case,
        ),
        _publish_summary(
            tmp_path,
            "promotion_development_s16384",
            base_seed=1731,
        ),
    ]
    with pytest.raises(ValueError, match="lacks one all-six primary row"):
        certify.certify(
            development,
            source_git_revision=SOURCE_GIT_REVISION,
            output_root=tmp_path,
        )

    other_root = tmp_path / "other"
    passing_development = [
        _publish_summary(
            other_root,
            "promotion_development_s4096",
            base_seed=1731,
        ),
        _publish_summary(
            other_root,
            "promotion_development_s16384",
            base_seed=1731,
            promotion_pass=False,
        ),
    ]
    confirmations = [
        _publish_summary(
            other_root,
            f"promotion_confirmation_s16384_seed{matrix_seed}",
            base_seed=observed_seed,
        )
        for matrix_seed, observed_seed in (
            (2731, 2731),
            (3731, 3732),
            (4731, 4731),
        )
    ]
    certificate = certify.certify(
        [*passing_development, *confirmations],
        source_git_revision=SOURCE_GIT_REVISION,
        output_root=other_root,
    )
    assert certificate["checks"]["all_input_matrix_summaries_pass"] is False
    assert certificate["checks"]["confirmation_seed_set_is_frozen"] is False
    assert certificate["certificate_pass"] is False
    assert certificate["promotion_pass"] is False


def test_summary_loader_and_certificate_publication_bind_exact_bytes(
    tmp_path: Path,
) -> None:
    summary_root = _publish_summary(
        tmp_path,
        "promotion_development_s4096",
        base_seed=1731,
    )
    summary, identity = certify._load_summary(
        summary_root,
        source_git_revision=SOURCE_GIT_REVISION,
    )
    assert identity["summary_payload_sha256"] == summary["sha256"]
    assert (
        identity["summary_file_sha256"]
        == hashlib.sha256((summary_root / "summary.json").read_bytes()).hexdigest()
    )
    with (summary_root / "summary.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="identity does not replay"):
        certify._load_summary(
            summary_root,
            source_git_revision=SOURCE_GIT_REVISION,
        )

    without_sha = {
        "schema": certify.SCHEMA,
        "source_git_revision": SOURCE_GIT_REVISION,
        "phase": "development",
        "certificate_pass": True,
    }
    certificate = {
        **without_sha,
        "sha256": certify._sha256_json(without_sha),
    }
    certificate_root = tmp_path / "certificate"
    certify._publish(certificate, certificate_root=certificate_root)
    report_path = certificate_root / "certificate.json"
    completion = json.loads((certificate_root / "COMPLETE.json").read_text(encoding="ascii"))
    assert completion == {
        "schema": certify.SCHEMA,
        "source_git_revision": SOURCE_GIT_REVISION,
        "phase": "development",
        "certificate_path": str(report_path),
        "certificate_payload_sha256": certificate["sha256"],
        "certificate_file_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "certificate_pass": True,
        "completion_marker_published_last": True,
    }
    with pytest.raises(FileExistsError, match="refusing to replace"):
        certify._publish(certificate, certificate_root=certificate_root)


def test_cross_size_certifier_rejects_exact_grid_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "near_gaussian__two_cell__root"
    count = 16
    monkeypatch.setattr(
        certify.experiment,
        "PROMOTION_CASES",
        (case_id,),
    )
    monkeypatch.setattr(certify.experiment, "GRID_COUNTS", (count,))
    monkeypatch.setattr(certify, "_artifact", lambda _row: object())
    monkeypatch.setattr(
        certify.experiment,
        "_vectorized_artifact_log_likelihood",
        lambda _artifact, _observation, totals: np.zeros_like(
            totals,
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        certify.aggregation_error_tiny_oracle,
        "root_conditional_log_likelihood",
        lambda _case_id, totals, fraction_order: np.zeros_like(
            totals,
            dtype=np.float64,
        ),
    )
    selected_row = {
        "case_id": case_id,
        "config_id": promotion.PRIMARY_CONFIG_ID,
        "selected_artifact_path": "unused",
        "selected_artifact_file_sha256": "a" * 64,
    }
    summary = {"selected_rows": [selected_row]}
    oracle_bundle = {
        "selected_cases": {
            case_id: {
                "reference": {"fraction_order": 32},
                "metric_grid_preflight": {
                    "rows": [
                        {
                            "count": count,
                            "total_grid_sha256": "0" * 64,
                            "exact_log_likelihood_sha256": "1" * 64,
                        }
                    ],
                },
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="exact grid differs from the oracle preflight",
    ):
        certify._cross_size_rows(
            summary,
            summary,
            oracle_bundle,
        )
