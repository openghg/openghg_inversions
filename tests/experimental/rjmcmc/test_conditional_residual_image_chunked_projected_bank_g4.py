"""Focused synthetic tests for the BP1 G4 projected-source driver."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.special import logsumexp

from examples.rjmcmc import conditional_residual_image_chunked_projected_bank_g4 as g4
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

REVISION = "1" * 40


def _spectrum() -> RootResidualSpectrum:
    aggregation = AdditiveDirichletAggregation(
        np.array([0.7, 1.1, 1.6, 0.9]),
        np.array(
            [
                [1.8, -0.5, 0.3, 0.9],
                [0.2, 1.4, -0.7, 0.1],
                [0.5, -0.2, 1.1, 0.8],
            ]
        ),
        np.array([0.35, 0.8, 0.6]),
        np.eye(3),
    )
    return RootResidualSpectrum.from_aggregation(aggregation)


def _canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )


def test_frozen_grid_controls_replay_and_are_positive() -> None:
    first_mass = g4._grid_masses()
    second_mass = g4._grid_masses()
    first_outer = g4._outer_coefficients(6)
    second_outer = g4._outer_coefficients(6)
    first_noise = g4._held_out_noise(7)
    second_noise = g4._held_out_noise(7)

    np.testing.assert_array_equal(first_mass, second_mass)
    np.testing.assert_array_equal(first_outer, second_outer)
    np.testing.assert_array_equal(first_noise, second_noise)
    assert first_mass.shape == (256,)
    assert first_outer.shape == (256, 6)
    assert first_noise.shape == (256, 7)
    assert np.all(first_mass > 0.0)
    assert np.all(first_outer > 0.0)
    assert np.all(np.isfinite(first_noise))


def test_moment_metrics_use_active_and_absolute_tiny_gates() -> None:
    root_two = math.sqrt(2.0)
    active = np.array(
        [
            [root_two, 0.0],
            [-root_two, 0.0],
            [0.0, root_two],
            [0.0, -root_two],
        ]
        * 2
    )
    tiny = np.zeros((8, 1))
    locations = np.column_stack((active, tiny))
    record = g4._moment_metrics(
        locations,
        np.array([1.0, 1.0, 1.0e-20]),
        sample_count=8,
        rank=3,
        eigenvalue_threshold=1.0e-12,
        maximum_sample_count=8,
    )

    assert record["active_coordinate_count"] == 2
    assert record["tiny_coordinate_count"] == 1
    assert record["passed"] is True


def test_tail_metrics_check_identical_marginal_joint_and_radial_samples() -> None:
    generator = np.random.default_rng(7102)
    values = generator.normal(size=(2_048, 2))
    record = g4._tail_metrics(
        values,
        values.copy(),
        np.ones(2),
        rank=2,
        eigenvalue_threshold=1.0e-12,
    )

    assert record["coordinatewise_ks_maximum"] == 0.0
    assert record["joint_max_abs_at_least_2_maximum_probability_difference"] == 0.0
    assert record["radial_ks_maximum"] == 0.0
    assert record["passed"] is True


def test_direct_hybrid_likelihood_matches_manual_finite_source_formula() -> None:
    spectrum = _spectrum()
    locations = np.array(
        [
            [-0.4, 0.2],
            [0.1, -0.3],
            [0.5, 0.7],
            [-0.2, -0.6],
        ]
    )
    centered = np.array([[0.7, -0.2, 1.1], [-0.4, 0.9, 0.3]])
    masses = np.array([0.8, 1.4])
    actual = g4._direct_hybrid_log_likelihoods(
        centered,
        masses,
        locations,
        spectrum,
        sample_counts=(2, 4),
        ranks=(1, 2),
    )

    expected = np.empty_like(actual)
    for grid_index, mass in enumerate(masses):
        whitened = (centered[grid_index] - mass * spectrum.observation_mean_design) / spectrum.noise_sd
        coordinates = spectrum.basis.T @ whitened
        orthogonal = whitened - spectrum.basis @ coordinates
        common = -float(np.sum(np.log(spectrum.noise_sd)))
        common -= 0.5 * (
            (spectrum.observation_mean_design.size - spectrum.retained_rank) * math.log(2.0 * math.pi)
            + float(orthogonal @ orthogonal)
        )
        for rank_index, rank in enumerate((1, 2)):
            tail_variance = 1.0 + mass * mass * spectrum.eigenvalues[rank:]
            tail = -0.5 * float(
                np.sum(np.log(2.0 * math.pi * tail_variance) + np.square(coordinates[rank:]) / tail_variance)
            )
            kernels = -0.5 * (
                rank * math.log(2.0 * math.pi)
                + np.sum(
                    np.square(coordinates[np.newaxis, :rank] - mass * locations[:, :rank]),
                    axis=1,
                )
            )
            for count_index, count in enumerate((2, 4)):
                expected[rank_index, count_index, grid_index] = (
                    common + tail + float(cast(Real, logsumexp(kernels[:count]))) - math.log(count)
                )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)


def test_common_suffix_requires_two_all_larger_passing_ranks() -> None:
    assert g4._common_suffix({16: False, 32: True, 64: True, 128: True}) == (32, 64, 128)
    assert g4._common_suffix({16: True, 32: False, 64: True, 128: True}) == (64, 128)
    assert g4._common_suffix({16: True, 32: True, 64: False, 128: True}) == ()


def test_g3_controls_require_the_scientific_selected_c_and_p(tmp_path: Path) -> None:
    decision = tmp_path / "g3.json"
    _canonical(
        decision,
        {
            "schema": g4.hpc.SCHEMA,
            "stage": "G3",
            "source_revision": REVISION,
            "native_concentration": g4.hpc.SCIENTIFIC_CONCENTRATION,
            "root_variance": g4.hpc.SCIENTIFIC_ROOT_VARIANCE,
            "science_calibration_schema": g4.hpc.SCIENCE_CALIBRATION_SCHEMA,
            "selected_sample_chunk_size": 4_096,
            "selected_projection_microbatch": 256,
            "passed": True,
        },
    )

    chunk, microbatch, _ = g4._strict_g3_controls(
        decision,
        source_revision=REVISION,
    )

    assert (chunk, microbatch) == (4_096, 256)


def test_development_certifier_publishes_marker_only_for_common_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_report = tmp_path / "seed.json"
    spectrum = tmp_path / "spectrum.json"
    grid = tmp_path / "grid.json"
    for path in (seed_report, spectrum, grid):
        path.write_bytes(b"x")
    payload = {
        "source_seed": 731,
        "development_rebuild": {"passed": True},
        "rank_decisions": {
            "16": {"development_passed": False},
            "32": {"development_passed": True},
            "64": {"development_passed": True},
            "128": {"development_passed": True},
        },
    }
    monkeypatch.setattr(g4, "_strict_seed_report", lambda *args, **kwargs: (payload, None, None))
    output = tmp_path / "development.json"
    marker = tmp_path / "DEVELOPMENT_COMPLETE.txt"
    report = g4.run_development_certify(
        output,
        marker,
        seed_report=seed_report,
        source_revision=REVISION,
        spectrum_manifest=spectrum,
        grid_manifest=grid,
    )

    assert report["passing_suffix"] == [32, 64, 128]
    assert report["selected_rank"] == 32
    assert marker.is_file()


def test_all_seed_certifier_requires_every_seed_and_pairwise_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development.json"
    spectrum_manifest = tmp_path / "spectrum.json"
    grid_manifest = tmp_path / "grid.json"
    _canonical(
        development,
        {
            "schema": g4.SCHEMA,
            "stage": "G4-development",
            "source_revision": REVISION,
            "passed": True,
            "passing_suffix": [64, 128],
        },
    )
    spectrum_manifest.write_bytes(b"spectrum")
    grid_manifest.write_bytes(b"grid")
    paths = []
    source = np.zeros((g4.S_LADDER[-1], g4.Q_LADDER[-1]))
    likelihood = np.zeros((len(g4.Q_LADDER), len(g4.S_LADDER), g4.GRID_SIZE))
    reports = {}
    for seed in g4.SOURCE_SEEDS:
        path = tmp_path / f"seed-{seed}.json"
        path.write_bytes(str(seed).encode())
        paths.append(path)
        reports[path] = {
            "source_seed": seed,
            "g3_decision": {"sha256": "a" * 64},
            "rank_decisions": {str(rank): {"within_seed_passed": True} for rank in g4.Q_LADDER},
        }

    monkeypatch.setattr(
        g4,
        "_strict_seed_report",
        lambda path, **kwargs: (reports[path], source, likelihood),
    )
    monkeypatch.setattr(g4, "_strict_spectrum", lambda *args, **kwargs: (_spectrum(), {}))
    monkeypatch.setattr(g4, "_eigenvalue_threshold", lambda spectrum: 1.0e-12)
    monkeypatch.setattr(g4, "_tail_metrics", lambda *args, **kwargs: {"passed": True})
    output = tmp_path / "all.json"
    marker = tmp_path / "G4_SOURCE_LOCK.txt"
    report = g4.run_all_seed_certify(
        output,
        marker,
        development_report=development,
        seed_reports=paths,
        source_revision=REVISION,
        spectrum_manifest=spectrum_manifest,
        grid_manifest=grid_manifest,
    )

    assert report["passed"] is True
    assert report["selected_rank"] == 64
    assert marker.is_file()
