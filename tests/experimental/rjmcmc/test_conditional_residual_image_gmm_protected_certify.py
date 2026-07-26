"""Tests for the one-shot residual-image GMM protected certifier."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_gmm_protected_certify as protected
from examples.rjmcmc import conditional_residual_image_gmm_tiny_screen as gmm


def _catalogue() -> dict[str, Any]:
    """Return one structurally valid protected catalogue."""
    return {
        "schema": protected.CATALOGUE_SCHEMA,
        "case_ids": ["__".join(case) for case in gmm.DEVELOPMENT_MATRIX],
        "protocol": gmm.PROTOCOL,
        "construction_method": gmm.CONSTRUCTION_METHOD,
        "sample_count_per_case": gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT,
        "split_domain": protected.PROTECTED_SPLIT_DOMAIN,
        "master_seed_hex": "9a" * 32,
    }


def _write_canonical(path: Path, payload: object) -> str:
    """Write canonical JSON and return its raw SHA-256 digest."""
    raw = f"{c1._canonical_json(payload)}\n".encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _evaluation(
    *,
    base_seed: int,
    sample_count: int,
    artifact_sha256: str = "a" * 64,
    envelope_sha256: str = "b" * 64,
    log_evidence: float = 1.0,
) -> dict[str, Any]:
    """Return the certificate fields needed for one passing evaluation."""
    return {
        "sample_count": sample_count,
        "base_seed": base_seed,
        "scientific_pass": True,
        "scientific_model_gates_pass": True,
        "fit_development_pass": True,
        "posterior_summary": {"log_evidence": log_evidence},
        "training": {
            "training_sample_count": sample_count,
            "artifact_sha256": artifact_sha256,
            "fitted_bundle_envelope": {
                "payload": {},
                "sha256": envelope_sha256,
            },
        },
    }


def _certificate_payload() -> dict[str, Any]:
    """Return a completed certificate payload for structural validation tests."""
    locked = gmm.DEVELOPMENT_SAMPLE_COUNTS[1]
    cases = []
    for index, expected_case in enumerate(gmm.DEVELOPMENT_MATRIX):
        case_id = "__".join(expected_case)
        artifact_sha256 = f"{index + 1:064x}"
        envelope_sha256 = f"{index + 20:064x}"
        nominated = _evaluation(
            base_seed=gmm.DEVELOPMENT_SELECTION_SEED,
            sample_count=locked,
            artifact_sha256=artifact_sha256,
            envelope_sha256=envelope_sha256,
            log_evidence=2.0,
        )
        confirmations = [
            _evaluation(
                base_seed=seed,
                sample_count=locked,
                artifact_sha256=f"{index + seed:064x}",
                envelope_sha256=f"{index + seed + 1:064x}",
                log_evidence=2.0,
            )
            for seed in gmm.CONFIRMATION_SEEDS
        ]
        cases.append(
            {
                "case_id": case_id,
                "input_sha256": f"{index + 40:064x}",
                "context_sha256": f"{index + 50:064x}",
                "locked_sample_count": locked,
                "nominated_development_raw_sha256": f"{index + 60:064x}",
                "nominated_development_evaluation": nominated,
                "nominated_fitted_bundle_sha256": envelope_sha256,
                "nominated_artifact_sha256": artifact_sha256,
                "confirmation_evaluations": confirmations,
                "confirmation_individual_passes": [
                    {"base_seed": seed, "pass": True} for seed in gmm.CONFIRMATION_SEEDS
                ],
                "four_bank_log_evidence_range_nat": 0.0,
                "four_bank_log_evidence_range_pass": True,
                "confirmation_pass": True,
                "development_pass": True,
            }
        )
    return {
        "schema": protected.development_certifier.CERTIFICATE_SCHEMA,
        "certification_protocol": protected.development_certifier.CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": (protected.development_certifier._certification_protocol_sha256()),
        "certifier_source_sha256": (protected.development_certifier._certifier_source_sha256()),
        "source_git_revision": "3" * 40,
        "scientific_driver_sha256": (protected.development_certifier._driver_source_sha256()),
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix_catalogue": gmm.matrix_catalogue(),
        "common_lock_raw_sha256": "5" * 64,
        "common_lock_sha256": "6" * 64,
        "locked_sample_count": locked,
        "confirmation_seeds": list(gmm.CONFIRMATION_SEEDS),
        "execution_certified": True,
        "decision": "pass",
        "development_pass": True,
        "eligible_for_protected_holdout": True,
        "protected_holdout_pass": None,
        "scientific_pass": False,
        "scientific_pass_available": False,
        "scientific_pass_reason": "protected holdout remains sealed",
        "structural_inference_licensed": False,
        "held_out_information_read": False,
        "cases": cases,
    }


def _shard_payload(
    *,
    case_id: str,
    source_revision: str,
    driver_sha256: str,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a minimal canonical development shard."""
    return {
        "schema": gmm.SCHEMA,
        "protocol": gmm.PROTOCOL,
        "profile": "development",
        "execution_mode": "development_size_shard",
        "selected_case_id": case_id,
        "source_git_revision": source_revision,
        "driver_sha256": driver_sha256,
        "protocol_sha256": gmm._protocol_sha256(),
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "structural_inference_licensed": False,
        "cases": [
            {
                "case_id": case_id,
                "development_evaluations": evaluations,
            }
        ],
    }


class _UnitGaussianArtifact:
    """Small zero-input unit Gaussian used to isolate the density gate."""

    residual_rank = 1
    region_count = 1

    def _components(
        self,
        masses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one normalized standard-normal component."""
        assert masses.tolist() == [1.0]
        return (
            np.array([0.0], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
            np.array([[[1.0]]], dtype=np.float64),
        )


def test_wrong_catalogue_hash_stops_before_json_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalogue reader authenticates raw bytes before JSON decoding."""
    catalogue_path = tmp_path / "sealed.json"
    catalogue_path.write_bytes(b"not even JSON\n")
    monkeypatch.setattr(
        protected.json,
        "loads",
        lambda *_args, **_kwargs: pytest.fail("JSON parser must not run"),
    )

    with pytest.raises(ValueError, match="raw SHA-256"):
        protected._read_protected_catalogue(catalogue_path)


@pytest.mark.parametrize(
    ("version_name", "message"),
    [("numpy_version", "NumPy"), ("scipy_version", "SciPy")],
)
def test_protected_certification_rejects_runtime_drift_before_artifact_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_name: str,
    message: str,
) -> None:
    """Protected numerical replay must use the frozen NumPy/SciPy runtime."""
    monkeypatch.setattr(gmm, version_name, "unexpected")
    touched: list[str] = []

    def forbidden_source_access(*_: object, **__: object) -> None:
        touched.append("source")
        raise AssertionError("runtime drift must stop before development access")

    monkeypatch.setattr(
        protected.development_certifier,
        "_validate_live_source",
        forbidden_source_access,
    )
    monkeypatch.setattr(
        protected,
        "_read_protected_catalogue",
        lambda *_args, **_kwargs: pytest.fail("protected catalogue must remain sealed"),
    )

    with pytest.raises(RuntimeError, match=message):
        protected.certify(
            source_directory=tmp_path,
            expected_source_revision="1" * 40,
            development_certificate=tmp_path / "development.json",
            expected_development_certificate_sha256="2" * 64,
            development_shards_directory=tmp_path / "shards",
            protected_catalogue=tmp_path / "catalogue.json",
            output=tmp_path / "result.json",
        )
    assert touched == []


def test_ineligible_certificate_stops_before_protected_catalogue_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed development gate must leave the protected input untouched."""
    monkeypatch.setattr(gmm, "_validate_development_protocol", lambda: None)
    monkeypatch.setattr(
        protected.development_certifier,
        "_validate_live_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        protected,
        "_read_certificate",
        lambda *_args, **_kwargs: ({}, "1" * 64, "2" * 64),
    )
    monkeypatch.setattr(
        protected,
        "_validate_certificate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("development certificate is not eligible")
        ),
    )
    monkeypatch.setattr(
        protected,
        "_read_protected_catalogue",
        lambda *_args, **_kwargs: pytest.fail("protected catalogue must remain sealed"),
    )

    with pytest.raises(ValueError, match="not eligible"):
        protected.certify(
            source_directory=tmp_path,
            expected_source_revision="1" * 40,
            development_certificate=tmp_path / "development.json",
            expected_development_certificate_sha256="2" * 64,
            development_shards_directory=tmp_path / "shards",
            protected_catalogue=tmp_path / "catalogue.json",
            output=tmp_path / "result.json",
        )


@pytest.mark.parametrize("failure_mode", ["missing", "tampered"])
def test_invalid_nominated_shard_stops_before_protected_catalogue_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """Every nominated shard must authenticate before the seal is opened."""
    certificate_payload = _certificate_payload()
    case_map = {case["case_id"]: case for case in certificate_payload["cases"]}
    monkeypatch.setattr(gmm, "_validate_development_protocol", lambda: None)
    monkeypatch.setattr(
        protected.development_certifier,
        "_validate_live_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        protected,
        "_read_certificate",
        lambda *_args, **_kwargs: (certificate_payload, "1" * 64, "2" * 64),
    )
    monkeypatch.setattr(
        protected,
        "_validate_certificate",
        lambda *_args, **_kwargs: (gmm.DEVELOPMENT_SAMPLE_COUNTS[1], case_map),
    )
    if failure_mode == "missing":
        monkeypatch.setattr(protected, "_regular_file_digest_map", lambda _path: {})
        expected_error = "nominated development shard is absent"
    else:
        digest_map = {
            case["nominated_development_raw_sha256"]: tmp_path / f"{index}.json"
            for index, case in enumerate(certificate_payload["cases"])
        }
        monkeypatch.setattr(protected, "_regular_file_digest_map", lambda _path: digest_map)
        monkeypatch.setattr(
            protected,
            "_prepare_nominated_case",
            lambda **_kwargs: (_ for _ in ()).throw(ValueError("nominated shard replay changed")),
        )
        expected_error = "nominated shard replay changed"
    monkeypatch.setattr(
        protected,
        "_read_protected_catalogue",
        lambda *_args, **_kwargs: pytest.fail("protected catalogue must remain sealed"),
    )

    with pytest.raises(ValueError, match=expected_error):
        protected.certify(
            source_directory=tmp_path,
            expected_source_revision="3" * 40,
            development_certificate=tmp_path / "development.json",
            expected_development_certificate_sha256="a" * 64,
            development_shards_directory=tmp_path / "shards",
            protected_catalogue=tmp_path / "catalogue.json",
            output=tmp_path / "result.json",
        )


def test_catalogue_schema_is_exact_and_seed_derivation_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the frozen catalogue fields may influence deterministic seeds."""
    catalogue = _catalogue()
    path = tmp_path / "catalogue.json"
    raw_sha256 = _write_canonical(path, catalogue)
    monkeypatch.setattr(gmm, "PROTECTED_HOLDOUT_CATALOGUE_SHA256", raw_sha256)

    observed, observed_raw = protected._read_protected_catalogue(path)

    assert observed == catalogue
    assert observed_raw == raw_sha256
    assert (
        protected._protected_seed(
            "9a" * 32,
            case_id="near_gaussian__two_cell__root",
        )
        == 5_550_216_977_343_770_791
    )
    tampered = copy.deepcopy(catalogue)
    tampered["retune"] = True
    tampered_sha256 = _write_canonical(path, tampered)
    monkeypatch.setattr(gmm, "PROTECTED_HOLDOUT_CATALOGUE_SHA256", tampered_sha256)
    with pytest.raises(ValueError, match="unexpected schema"):
        protected._read_protected_catalogue(path)


def test_certificate_rejects_tampered_nomination_and_confirmation() -> None:
    """Seed, locked-size, and four-bank evidence tampering must fail closed."""
    payload = _certificate_payload()
    locked, cases = protected._validate_certificate(
        payload,
        expected_source_revision="3" * 40,
    )
    assert locked == gmm.DEVELOPMENT_SAMPLE_COUNTS[1]
    assert len(cases) == 6

    wrong_seed = copy.deepcopy(payload)
    wrong_seed["cases"][0]["nominated_development_evaluation"]["base_seed"] = 1877
    with pytest.raises(ValueError, match="seed-731"):
        protected._validate_certificate(
            wrong_seed,
            expected_source_revision="3" * 40,
        )

    wrong_evidence = copy.deepcopy(payload)
    wrong_evidence["cases"][0]["confirmation_evaluations"][0]["posterior_summary"]["log_evidence"] = 2.1
    with pytest.raises(ValueError, match="four-bank evidence"):
        protected._validate_certificate(
            wrong_evidence,
            expected_source_revision="3" * 40,
        )


def test_shard_promotion_selects_exactly_seed_731_at_locked_size(
    tmp_path: Path,
) -> None:
    """No confirmation seed or duplicate nomination can be promoted."""
    case_id = "near_gaussian__two_cell__root"
    locked = gmm.DEVELOPMENT_SAMPLE_COUNTS[1]
    source_revision = "1" * 40
    driver_sha256 = "2" * 64
    path = tmp_path / "shard.json"
    nominated = _evaluation(
        base_seed=gmm.DEVELOPMENT_SELECTION_SEED,
        sample_count=locked,
    )
    raw_sha256 = _write_canonical(
        path,
        _shard_payload(
            case_id=case_id,
            source_revision=source_revision,
            driver_sha256=driver_sha256,
            evaluations=[nominated],
        ),
    )
    _, observed = protected._read_nominated_evaluation(
        path,
        raw_sha256=raw_sha256,
        case_id=case_id,
        locked_sample_count=locked,
        source_revision=source_revision,
        driver_sha256=driver_sha256,
    )
    assert observed["base_seed"] == 731

    confirmation_only = _shard_payload(
        case_id=case_id,
        source_revision=source_revision,
        driver_sha256=driver_sha256,
        evaluations=[
            _evaluation(
                base_seed=gmm.CONFIRMATION_SEEDS[0],
                sample_count=locked,
            )
        ],
    )
    raw_sha256 = _write_canonical(path, confirmation_only)
    with pytest.raises(ValueError, match="exactly one seed-731"):
        protected._read_nominated_evaluation(
            path,
            raw_sha256=raw_sha256,
            case_id=case_id,
            locked_sample_count=locked,
            source_revision=source_revision,
            driver_sha256=driver_sha256,
        )


def test_density_holdout_gate_can_pass_and_fail() -> None:
    """The protected NLL comparison must implement the frozen combined-MCSE rule."""
    artifact = _UnitGaussianArtifact()
    draws = np.zeros((16, 1), dtype=np.float64)
    standard_normal_nll_at_zero = 0.5 * np.log(2.0 * np.pi)

    passing = protected._protected_density_gate(
        draws,
        artifact=artifact,  # type: ignore[arg-type]
        development_generalization={
            "validation_nll_nat_per_draw": standard_normal_nll_at_zero,
            "validation_nll_mcse_nat_per_draw": 0.0,
        },
    )
    failing = protected._protected_density_gate(
        draws,
        artifact=artifact,  # type: ignore[arg-type]
        development_generalization={
            "validation_nll_nat_per_draw": standard_normal_nll_at_zero + 0.021,
            "validation_nll_mcse_nat_per_draw": 0.0,
        },
    )

    assert passing["pass"] is True
    assert passing["threshold_nat_per_draw"] == pytest.approx(0.02)
    assert failing["absolute_nll_gap_nat_per_draw"] == pytest.approx(0.021)
    assert failing["pass"] is False


@pytest.mark.parametrize("failed_index", [None, 4])
def test_final_claim_requires_all_six_and_output_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int | None,
) -> None:
    """A completed run is available, but passes only when all six cases pass."""
    catalogue_path = tmp_path / "catalogue.json"
    raw_sha256 = _write_canonical(catalogue_path, _catalogue())
    monkeypatch.setattr(gmm, "PROTECTED_HOLDOUT_CATALOGUE_SHA256", raw_sha256)
    monkeypatch.setattr(gmm, "_validate_development_protocol", lambda: None)
    certificate_payload = _certificate_payload()
    case_map = {case["case_id"]: case for case in certificate_payload["cases"]}
    monkeypatch.setattr(
        protected,
        "_read_certificate",
        lambda _path, **_kwargs: (certificate_payload, "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        protected,
        "_validate_certificate",
        lambda _payload, **_kwargs: (gmm.DEVELOPMENT_SAMPLE_COUNTS[1], case_map),
    )
    monkeypatch.setattr(
        protected.development_certifier,
        "_validate_live_source",
        lambda *_args, **_kwargs: None,
    )
    digest_map = {
        case["nominated_development_raw_sha256"]: tmp_path / f"{index}.json"
        for index, case in enumerate(certificate_payload["cases"])
    }
    monkeypatch.setattr(protected, "_regular_file_digest_map", lambda _path: digest_map)

    monkeypatch.setattr(
        protected,
        "_prepare_nominated_case",
        lambda **kwargs: kwargs["expected_case"],
    )

    def fake_case(prepared: tuple[str, str, str], **_kwargs: Any) -> dict[str, Any]:
        case_id = "__".join(prepared)
        index = ["__".join(case) for case in gmm.DEVELOPMENT_MATRIX].index(case_id)
        return {
            "case_id": case_id,
            "pass": index != failed_index,
            "structural_inference_licensed": False,
        }

    monkeypatch.setattr(protected, "_certify_prepared_case", fake_case)
    output = tmp_path / f"result-{failed_index}.json"
    result = protected.certify(
        source_directory=tmp_path,
        expected_source_revision="3" * 40,
        development_certificate=tmp_path / "development.json",
        expected_development_certificate_sha256="a" * 64,
        development_shards_directory=tmp_path / "shards",
        protected_catalogue=catalogue_path,
        output=output,
    )

    expected = failed_index is None
    assert result["scientific_pass_available"] is True
    assert result["scientific_pass"] is expected
    assert result["protected_holdout_pass"] is expected
    assert result["structural_inference_licensed"] is False
    assert all("retune" not in key and "retrain" not in key for key in result)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        protected.certify(
            source_directory=tmp_path,
            expected_source_revision="3" * 40,
            development_certificate=tmp_path / "development.json",
            expected_development_certificate_sha256="a" * 64,
            development_shards_directory=tmp_path / "shards",
            protected_catalogue=catalogue_path,
            output=output,
        )


def test_certifier_exposes_no_training_or_retuning_path() -> None:
    """The protected executable must contain no fitting or tuning entry point."""
    source = Path(protected.__file__).read_text(encoding="utf-8")
    parser = protected._parser()

    assert not hasattr(protected, "fit_gaussian_mixture")
    assert "fit_gaussian_mixture(" not in source
    assert "fit_gaussian_mixture" not in {action.dest for action in parser._actions}
    assert "retune" not in {action.dest for action in parser._actions}
    assert "retrain" not in {action.dest for action in parser._actions}
