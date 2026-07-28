"""Focused tests for the BP1 chunked projected-bank HPC driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.rjmcmc import (
    conditional_residual_image_chunked_projected_bank_hpc as hpc,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
    build_chunked_projected_root_bank,
)

REVISION = "9" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_two_aggregate_calibration_reproduces_both_target_cvs() -> None:
    """The analytic calibration should satisfy both user-specified totals."""
    root_variance, concentration = hpc._solve_two_aggregate_calibration(
        broad_factor=8.874250601269965,
        local_factor=117.60829531564202,
        broad_target_cv=hpc.MODELED_EUROPEAN_DOMAIN_TARGET_CV,
        local_target_cv=hpc.GBR_TARGET_CV,
    )

    assert root_variance == pytest.approx(hpc.SCIENTIFIC_ROOT_VARIANCE, abs=1e-15)
    assert concentration == pytest.approx(hpc.SCIENTIFIC_CONCENTRATION, abs=1e-11)
    for factor, target in (
        (8.874250601269965, hpc.MODELED_EUROPEAN_DOMAIN_TARGET_CV),
        (117.60829531564202, hpc.GBR_TARGET_CV),
    ):
        achieved = np.sqrt(root_variance + (1.0 + root_variance) * factor / (concentration + 1.0))
        assert achieved == pytest.approx(target, abs=1e-15)


def test_aggregate_variance_factor_uses_physical_total_weights() -> None:
    """The Dirichlet factor must be based on the modeled physical aggregate."""
    nominal = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    physical = np.array([2.0, 1.0, 1.0], dtype=np.float64)

    factor, total = hpc._aggregate_variance_factor(nominal, physical)

    assert total == 4.0
    assert factor == pytest.approx(0.0, abs=1e-15)


def test_paris_builder_rejects_an_unlocked_concentration(tmp_path: Path) -> None:
    """Historical engineering concentrations must not reach the science stages."""
    with pytest.raises(ValueError, match="scientific lock"):
        hpc._build_paris_aggregation(
            tmp_path / "unused.nc",
            expected_input_sha256="a" * 64,
            concentration=100.0,
        )


def test_tiny_v3_json_and_binary_roundtrip(tmp_path: Path) -> None:
    """The G0 helper should publish replayable create-only artifacts."""
    report = hpc.run_tiny(tmp_path, source_revision=REVISION)

    assert report["json_replay_exact"] is True
    assert report["binary_roundtrip_exact"] is True
    array = np.load(tmp_path / "tiny_bank.npy", allow_pickle=False)
    assert array.shape == (64, 2)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        hpc.run_tiny(tmp_path, source_revision=REVISION)


def test_spectrum_bundle_roundtrips_exactly(tmp_path: Path) -> None:
    """Every spectrum array and scalar should survive binary publication."""
    aggregation = hpc._synthetic_aggregation(
        cells=9,
        observations=4,
        alpha_mode="heterogeneous",
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    hpc._write_spectrum_bundle(
        tmp_path,
        spectrum,
        source_revision=REVISION,
        input_path=tmp_path / "frozen.nc",
        input_sha256="a" * 64,
        concentration=hpc.SCIENTIFIC_CONCENTRATION,
        elapsed_seconds=1.25,
    )

    replay = hpc._load_spectrum_bundle(tmp_path / "spectrum_manifest.json")

    np.testing.assert_array_equal(replay.observation_mean_design, spectrum.observation_mean_design)
    np.testing.assert_array_equal(replay.noise_sd, spectrum.noise_sd)
    np.testing.assert_array_equal(replay.basis, spectrum.basis)
    np.testing.assert_array_equal(replay.eigenvalues, spectrum.eigenvalues)
    assert hpc._spectrum_scalars(replay) == hpc._spectrum_scalars(spectrum)


def test_spectrum_bundle_rejects_binary_mutation(tmp_path: Path) -> None:
    """Consumption should fail before replay when a binary file changes."""
    aggregation = hpc._synthetic_aggregation(
        cells=5,
        observations=3,
        alpha_mode="heterogeneous",
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    hpc._write_spectrum_bundle(
        tmp_path,
        spectrum,
        source_revision=REVISION,
        input_path=tmp_path / "frozen.nc",
        input_sha256="b" * 64,
        concentration=hpc.SCIENTIFIC_CONCENTRATION,
        elapsed_seconds=1.0,
    )
    basis_path = tmp_path / "basis.npy"
    raw = bytearray(basis_path.read_bytes())
    raw[-1] ^= 1
    basis_path.write_bytes(raw)

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        hpc._load_spectrum_bundle(tmp_path / "spectrum_manifest.json")


def test_second_spectrum_is_a_covariance_audit_only(tmp_path: Path) -> None:
    """An exact second construction should pass without becoming authoritative."""
    aggregation = hpc._synthetic_aggregation(
        cells=9,
        observations=4,
        alpha_mode="heterogeneous",
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    authoritative = tmp_path / "authoritative"
    audit = tmp_path / "audit"
    authoritative.mkdir()
    audit.mkdir()
    for directory in (authoritative, audit):
        hpc._write_spectrum_bundle(
            directory,
            spectrum,
            source_revision=REVISION,
            input_path=tmp_path / "frozen.nc",
            input_sha256="c" * 64,
            concentration=hpc.SCIENTIFIC_CONCENTRATION,
            elapsed_seconds=1.0,
        )

    report = hpc.run_g2_audit(
        tmp_path / "audit_report.json",
        authoritative_manifest=authoritative / "spectrum_manifest.json",
        audit_manifest=audit / "spectrum_manifest.json",
        source_revision=REVISION,
    )

    assert report["passed"] is True
    assert report["audit_is_authoritative"] is False
    assert report["exact_context_identity"] is True


def test_parity_record_reports_absolute_and_ulp_differences() -> None:
    """The frozen parity gate should report both requested metrics."""
    reference = np.array([[-1.0, 0.5], [0.0, 2.0]], dtype=np.float64)
    candidate = reference.copy()
    candidate[0, 1] = np.nextafter(candidate[0, 1], np.inf)

    record = hpc._parity_record(reference, candidate, native_cells=4)

    assert record["passed"] is True
    assert record["maximum_ulp_difference"] == 1
    maximum_absolute_difference = record["maximum_absolute_difference"]
    assert isinstance(maximum_absolute_difference, float)
    assert maximum_absolute_difference > 0.0


def test_projection_microbatches_replay_and_meet_frozen_parity() -> None:
    """Each fixed P should replay exactly while different BLAS shapes meet parity."""
    aggregation = hpc._synthetic_aggregation(
        cells=129,
        observations=4,
        alpha_mode="heterogeneous",
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)

    def build(projection_chunk_size: int) -> np.ndarray:
        bank = build_chunked_projected_root_bank(
            aggregation,
            spectrum,
            mixture_rank=3,
            sample_count=256,
            sample_chunk_size=256,
            projection_chunk_size=projection_chunk_size,
            source_seed=731,
            source_provenance="focused P-parity test",
        )
        return np.asarray(bank.projected_unit_mass_residual_factors[:, :, 0])

    reference = build(64)
    np.testing.assert_array_equal(reference, build(64))
    candidate = build(128)
    record = hpc._parity_record(
        reference,
        candidate,
        native_cells=aggregation.cell_alphas.size,
    )
    assert record["passed"] is True


def test_locked_p_requires_a_frozen_parity_g1_manifest(tmp_path: Path) -> None:
    """A malformed or failed parity/throughput lock must not reach G3."""
    manifest = {
        "schema": hpc.SCHEMA,
        "stage": "G1",
        "projection_microbatch_selection": {
            "all_candidate_outputs_within_frozen_parity_tolerance": True,
            "locked_projection_chunk_size": 128,
        },
    }
    path = tmp_path / "g1.json"
    path.write_text(hpc._canonical_json(manifest) + "\n", encoding="ascii")
    assert hpc._locked_p(path) == 128

    manifest["projection_microbatch_selection"]["all_candidate_outputs_within_frozen_parity_tolerance"] = (
        False
    )
    path.write_text(hpc._canonical_json(manifest) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="no valid frozen-parity"):
        hpc._locked_p(path)


def test_output_policy_rejects_paris_inversions(tmp_path: Path) -> None:
    """The experimental driver must not publish into production output."""
    parent = tmp_path / "PARIS_inversions"
    parent.mkdir()

    with pytest.raises(ValueError, match="PARIS_inversions"):
        hpc._atomic_write_json(parent / "report.json", {"passed": True})


def test_json_reader_rejects_noncanonical_text(tmp_path: Path) -> None:
    """Manifests must have one exact canonical representation."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"b": 1, "a": 2}) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="not canonical"):
        hpc._read_json(path)


def test_json_writer_roundtrips_stringified_control_keys(tmp_path: Path) -> None:
    """Published control maps must retain their canonical key ordering."""
    payload = {
        "cross_candidate_parity": {
            "64": {"passed": True},
            "128": {"passed": True},
            "256": {"passed": True},
        }
    }
    path = tmp_path / "manifest.json"

    hpc._atomic_write_json(path, payload)

    assert hpc._read_json(path) == payload


def test_sacct_records_preserve_logical_array_task_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array accounting must join on JobID, not physical JobIDRaw values."""
    stdout = "\n".join(
        (
            "9000_0|COMPLETED|17|",
            "9000_0.batch|COMPLETED|17|2048K",
            "9000_0.extern|COMPLETED|17|",
            "9000_10|COMPLETED|19|",
            "9000_10.batch|COMPLETED|19|3G",
            "9000_10.extern|COMPLETED|19|",
        )
    )

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        assert "--format=JobID,State,ElapsedRaw,MaxRSS" in command
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(hpc.subprocess, "run", fake_run)

    assert hpc._sacct_records(["9000_0", "9000_10"]) == {
        "9000_0": {
            "state": "COMPLETED",
            "elapsed_seconds": 17,
            "max_rss_bytes": 2 * (1 << 20),
        },
        "9000_10": {
            "state": "COMPLETED",
            "elapsed_seconds": 19,
            "max_rss_bytes": 3 * (1 << 30),
        },
    }


@pytest.mark.parametrize("launcher", ("run_g1.sbatch", "run_g3_bank.sbatch"))
def test_timing_selected_launchers_request_exclusive_nodes(launcher: str) -> None:
    """Unrelated node workloads must not determine computational timing locks."""
    path = REPOSITORY_ROOT / "docs" / "plans" / "rjmcmc_chunked_projected_bank_assets" / launcher

    assert "#SBATCH --exclusive" in path.read_text(encoding="utf-8").splitlines()


def test_g3_certifier_selects_lowest_passing_median(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The G3 merger should authenticate all repeats and publish its marker last."""
    prefix_path = tmp_path / "g3a.json"
    prefix = {
        "schema": hpc.SCHEMA,
        "stage": "G3a",
        "source_revision": REVISION,
        "input_sha256": "a" * 64,
        "spectrum_manifest_sha256": "b" * 64,
        "native_concentration": hpc.SCIENTIFIC_CONCENTRATION,
        "root_variance": hpc.SCIENTIFIC_ROOT_VARIANCE,
        "science_calibration_schema": hpc.SCIENCE_CALIBRATION_SCHEMA,
        "records": [{"projection_chunk_size": 256}],
        "passed": True,
    }
    prefix_path.write_text(hpc._canonical_json(prefix) + "\n", encoding="ascii")
    candidates: list[Path] = []
    job_records: dict[str, hpc.SlurmRecord] = {}
    job_number = 10_000
    array_job_id = "9000"
    array_task_id = 0
    for chunk_index, chunk in enumerate(hpc.RESOURCE_C_LADDER):
        for repeat in (0, 1, 2):
            directory = tmp_path / f"C{chunk}" / f"repeat{repeat}"
            directory.mkdir(parents=True)
            job_id = str(job_number)
            job_number += 1
            elapsed = float(10 + chunk_index * 5 + repeat)
            payload = {
                "schema": hpc.SCHEMA,
                "stage": "G3b-candidate",
                "source_revision": REVISION,
                "input_sha256": "a" * 64,
                "spectrum_manifest_sha256": "b" * 64,
                "g1_manifest_sha256": "c" * 64,
                "native_concentration": hpc.SCIENTIFIC_CONCENTRATION,
                "root_variance": hpc.SCIENTIFIC_ROOT_VARIANCE,
                "science_calibration_schema": hpc.SCIENCE_CALIBRATION_SCHEMA,
                "sample_chunk_size": chunk,
                "projection_chunk_size": 256,
                "repeat": repeat,
                "slurm_job_id": job_id,
                "slurm_array_job_id": array_job_id,
                "slurm_array_task_id": str(array_task_id),
                "constructor_seconds": elapsed,
                "passed_internal_checks": True,
                "projected_array": {
                    "array_sha256": "d" * 64,
                    "file_sha256": "e" * 64,
                },
            }
            manifest = directory / "bank_manifest.json"
            manifest.write_text(
                hpc._canonical_json(payload) + "\n",
                encoding="ascii",
            )
            (directory / "resource.time").write_text(
                "\tSwaps: 0\n",
                encoding="utf-8",
            )
            candidates.append(manifest)
            job_records[f"{array_job_id}_{array_task_id}"] = {
                "state": "COMPLETED",
                "elapsed_seconds": int(elapsed),
                "max_rss_bytes": 2 * (1 << 30),
            }
            array_task_id += 1
    monkeypatch.setattr(hpc, "_sacct_records", lambda job_ids: job_records)

    report = hpc.run_g3_certify(
        tmp_path / "g3_decision.json",
        tmp_path / "G3_COMPLETE.txt",
        prefix_manifest=prefix_path,
        candidate_manifests=candidates,
        source_revision=REVISION,
    )

    assert report["passed"] is True
    assert report["selected_sample_chunk_size"] == 1_024
    assert report["selected_projection_microbatch"] == 256
    assert (tmp_path / "G3_COMPLETE.txt").is_file()
