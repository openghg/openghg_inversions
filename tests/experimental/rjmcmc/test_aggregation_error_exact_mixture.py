"""Regression tests for finite-bank Gamma--Dirichlet mixture compression."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_conditional_mixture as conditional_mixture_module,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    CompressedRootMixture,
    RootResidualSpectrum,
    build_chunked_projected_root_bank,
    compressed_root_mixture_log_likelihood,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    low_rank_gaussian_log_likelihood,
)


def _native_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small heterogeneous native aggregation problem."""
    alphas = np.array([0.7, 1.1, 1.6, 0.9])
    design = np.array(
        [
            [1.8, -0.5, 0.3, 0.9],
            [0.2, 1.4, -0.7, 0.1],
            [0.5, -0.2, 1.1, 0.8],
        ]
    )
    noise_sd = np.array([0.35, 0.8, 0.6])
    return alphas, design, noise_sd


def _aggregation(
    *,
    summary_basis: np.ndarray | None = None,
) -> AdditiveDirichletAggregation:
    """Build the heterogeneous aggregation with a caller-selected basis."""
    alphas, design, noise_sd = _native_inputs()
    if summary_basis is None:
        summary_basis = np.eye(design.shape[0])
    return AdditiveDirichletAggregation(alphas, design, noise_sd, summary_basis)


def _spectrum(
    *,
    retained_variance_fraction: float = 1.0,
    maximum_rank: int | None = None,
) -> RootResidualSpectrum:
    """Build the analytic root spectrum for the heterogeneous fixture."""
    return RootResidualSpectrum.from_aggregation(
        _aggregation(),
        retained_variance_fraction=retained_variance_fraction,
        maximum_rank=maximum_rank,
    )


def _source(
    spectrum: RootResidualSpectrum,
    *,
    sample_count: int = 64,
    source_seed: int = 731,
    labels: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
) -> ConditionalAllocationMixture:
    """Build a root Sobol source bank in the spectrum's exact basis."""
    if labels is None:
        labels = np.zeros(4, dtype=np.int64)
    if cell_ids is None:
        cell_ids = np.array([101, 102, 201, 202], dtype=np.int64)
    return ConditionalAllocationMixture.from_aggregation(
        _aggregation(summary_basis=spectrum.basis),
        labels,
        sample_count=sample_count,
        source_seed=source_seed,
        source_provenance="unit-test exact-mixture source",
        cell_ids=cell_ids,
        construction_method="scrambled_sobol_balanced_dirichlet",
    )


def _compressed(
    spectrum: RootResidualSpectrum,
    *,
    mixture_rank: int,
    component_count: int,
    sample_count: int = 64,
    random_seed: int = 4242,
) -> tuple[ConditionalAllocationMixture, CompressedRootMixture]:
    """Build one deterministic compressed root mixture and its source."""
    source = _source(spectrum, sample_count=sample_count)
    artifact = CompressedRootMixture.from_source(
        source,
        spectrum,
        mixture_rank=mixture_rank,
        component_count=component_count,
        restart_count=3,
        random_seed=random_seed,
    )
    return source, artifact


def _chunked_source(
    spectrum: RootResidualSpectrum,
    *,
    mixture_rank: int,
    sample_count: int = 64,
    sample_chunk_size: int = 8,
    projection_chunk_size: int = 4,
) -> ConditionalAllocationMixture:
    """Build a directly projected chunked root bank."""
    return build_chunked_projected_root_bank(
        _aggregation(),
        spectrum,
        mixture_rank=mixture_rank,
        sample_count=sample_count,
        sample_chunk_size=sample_chunk_size,
        projection_chunk_size=projection_chunk_size,
        source_seed=731,
        source_provenance="unit-test chunked projected source",
        cell_ids=np.array([101, 102, 201, 202], dtype=np.int64),
    )


def _direct_root_covariance() -> np.ndarray:
    """Return the analytic unit-mass covariance in whitened observation space."""
    alphas, design, noise_sd = _native_inputs()
    proportions = alphas / math.fsum(float(value) for value in alphas)
    mean_design = design @ proportions
    centered = (design - mean_design[:, np.newaxis]) / noise_sd[:, np.newaxis]
    simplex_covariance = (np.diag(proportions) - np.outer(proportions, proportions)) / (
        float(np.sum(alphas)) + 1.0
    )
    covariance = centered @ simplex_covariance @ centered.T
    return 0.5 * (covariance + covariance.T)


def _mixture_moments(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the global mean and covariance of a finite Gaussian mixture."""
    mean = np.einsum("m,mq->q", weights, means)
    covariance = np.zeros((means.shape[1], means.shape[1]), dtype=np.float64)
    for weight, component_mean, component_covariance in zip(
        weights,
        means,
        covariances,
        strict=True,
    ):
        displacement = component_mean - mean
        covariance += float(weight) * (component_covariance + np.outer(displacement, displacement))
    return mean, covariance


def _manual_compressed_log_likelihood(
    artifact: CompressedRootMixture,
    observation: np.ndarray,
    root_mass: float,
    offset: np.ndarray,
) -> float:
    """Evaluate the normalized hybrid mixture directly in orthogonal coordinates."""
    spectrum = artifact.spectrum
    residual = (observation - offset - root_mass * spectrum.observation_mean_design) / spectrum.noise_sd
    coordinates = spectrum.basis.T @ residual
    projected = spectrum.basis @ coordinates
    orthogonal = residual - projected
    logp = -float(np.sum(np.log(spectrum.noise_sd)))
    logp -= 0.5 * (
        (residual.size - spectrum.retained_rank) * math.log(2.0 * math.pi) + float(orthogonal @ orthogonal)
    )

    q = artifact.mixture_rank
    r = spectrum.retained_rank
    if q:
        component_logp = np.empty(artifact.component_count, dtype=np.float64)
        for component in range(artifact.component_count):
            covariance = np.eye(q) + root_mass**2 * artifact.covariances[component]
            component_logp[component] = math.log(float(artifact.weights[component]))
            component_logp[component] += float(
                multivariate_normal.logpdf(
                    coordinates[:q],
                    mean=root_mass * artifact.means[component],
                    cov=covariance,
                )
            )
        logp += float(logsumexp(component_logp))  # pyright: ignore[reportArgumentType]
    for index in range(q, r):
        variance = 1.0 + root_mass**2 * float(spectrum.eigenvalues[index])
        logp -= 0.5 * (math.log(2.0 * math.pi * variance) + float(coordinates[index]) ** 2 / variance)
    return logp


def test_root_spectrum_matches_analytic_covariance_and_tail_trace() -> None:
    """Eigenpairs should reconstruct the analytic covariance and report its tail."""
    full = _spectrum()
    expected = _direct_root_covariance()

    assert np.all(np.diff(full.eigenvalues) <= 0.0)
    assert np.all(full.eigenvalues >= 0.0)
    np.testing.assert_allclose(
        full.basis.T @ full.basis,
        np.eye(full.retained_rank),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        (full.basis * full.eigenvalues[np.newaxis, :]) @ full.basis.T,
        expected,
        rtol=3.0e-15,
        atol=5.0e-15,
    )
    assert full.total_variance == pytest.approx(float(np.trace(expected)), abs=5.0e-15)
    assert full.discarded_variance == pytest.approx(0.0, abs=5.0e-15)

    truncated = _spectrum(
        retained_variance_fraction=0.5,
        maximum_rank=1,
    )
    assert truncated.retained_rank == 1
    assert truncated.discarded_variance == pytest.approx(
        full.total_variance - float(truncated.eigenvalues.sum()),
        abs=5.0e-15,
    )


def test_root_spectrum_selection_ignores_summary_basis_and_has_no_state_input() -> None:
    """Spectrum selection should depend on native prior/operator inputs, not state."""
    raw = np.array([[1.0, 0.4], [-0.3, 1.0], [0.8, -0.2]])
    rotated, _ = np.linalg.qr(raw)
    original = RootResidualSpectrum.from_aggregation(
        _aggregation(),
        retained_variance_fraction=0.75,
    )
    alternative = RootResidualSpectrum.from_aggregation(
        _aggregation(summary_basis=rotated),
        retained_variance_fraction=0.75,
    )

    assert original.retained_rank == alternative.retained_rank
    np.testing.assert_allclose(original.eigenvalues, alternative.eigenvalues, atol=5.0e-15)
    np.testing.assert_allclose(
        original.basis @ original.basis.T,
        alternative.basis @ alternative.basis.T,
        atol=5.0e-15,
    )


@pytest.mark.parametrize(
    ("alphas", "design"),
    [
        (np.array([1.2]), np.array([[0.4], [-0.7]])),
        (np.array([0.8, 1.3]), np.array([[0.4, 0.4], [-0.7, -0.7]])),
    ],
)
def test_zero_aggregation_error_has_literal_rank_zero(
    alphas: np.ndarray,
    design: np.ndarray,
) -> None:
    """One cell and equal footprints should produce no residual directions."""
    aggregation = AdditiveDirichletAggregation(
        alphas,
        design,
        np.array([0.5, 0.8]),
        np.eye(2),
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)

    assert spectrum.retained_rank == 0
    assert spectrum.basis.shape == (2, 0)
    assert spectrum.eigenvalues.shape == (0,)
    assert spectrum.total_variance == 0.0
    assert spectrum.discarded_variance == 0.0


def test_source_must_be_root_and_use_the_spectrum_basis() -> None:
    """Compression should reject non-root banks and incompatible projection bases."""
    spectrum = _spectrum()
    non_root = _source(
        spectrum,
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="root|one region"):
        CompressedRootMixture.from_source(
            non_root,
            spectrum,
            mixture_rank=1,
            component_count=2,
            restart_count=2,
            random_seed=1,
        )

    incompatible_aggregation = _aggregation(summary_basis=-spectrum.basis)
    incompatible = ConditionalAllocationMixture.from_aggregation(
        incompatible_aggregation,
        np.zeros(4, dtype=np.int64),
        sample_count=64,
        source_seed=731,
        source_provenance="incompatible basis",
        cell_ids=np.array([101, 102, 201, 202]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    with pytest.raises(ValueError, match="basis|operator"):
        CompressedRootMixture.from_source(
            incompatible,
            spectrum,
            mixture_rank=1,
            component_count=2,
            restart_count=2,
            random_seed=1,
        )


def test_compression_replays_and_preserves_source_bank_moments() -> None:
    """Fixed clustering should replay and preserve all finite-bank first two moments."""
    spectrum = _spectrum()
    source, original = _compressed(
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=7,
    )
    _, replay = _compressed(
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=7,
    )

    assert np.array_equal(original.weights, replay.weights)
    assert np.array_equal(original.means, replay.means)
    assert np.array_equal(original.covariances, replay.covariances)
    assert np.array_equal(
        original.covariance_eigenvalues,
        replay.covariance_eigenvalues,
    )
    assert np.array_equal(
        original.covariance_eigenvectors,
        replay.covariance_eigenvectors,
    )
    assert np.array_equal(original.cluster_counts, replay.cluster_counts)
    assert original.selected_restart == replay.selected_restart
    assert np.array_equal(original.restart_inertias, replay.restart_inertias)
    assert np.all(original.weights > 0.0)
    assert math.fsum(float(weight) for weight in original.weights) == pytest.approx(
        1.0,
        abs=2.0e-16,
    )
    assert np.all(np.linalg.eigvalsh(original.covariances) >= -2.0e-15)
    assert math.isfinite(original.kl_upper_bound)
    assert original.kl_upper_bound >= 0.0
    assert all(
        not values.flags.writeable
        for values in (
            original.weights,
            original.means,
            original.covariances,
            original.covariance_eigenvalues,
            original.covariance_eigenvectors,
            original.cluster_counts,
            original.restart_inertias,
        )
    )
    reconstructed_covariances = np.einsum(
        "mij,mj,mkj->mik",
        original.covariance_eigenvectors,
        original.covariance_eigenvalues,
        original.covariance_eigenvectors,
        optimize=False,
    )
    np.testing.assert_allclose(
        reconstructed_covariances,
        original.covariances,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    source_locations = source.projected_unit_mass_residual_factors[
        :,
        : original.mixture_rank,
        0,
    ]
    source_mean = np.mean(source_locations, axis=0)
    centered = source_locations - source_mean
    source_covariance = centered.T @ centered / source.sample_count
    compressed_mean, compressed_covariance = _mixture_moments(
        original.weights,
        original.means,
        original.covariances,
    )
    np.testing.assert_allclose(compressed_mean, source_mean, rtol=0.0, atol=5.0e-16)
    np.testing.assert_allclose(
        compressed_covariance,
        source_covariance,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_stable_cell_permutation_preserves_source_compression_and_density() -> None:
    """Reordering native storage with stable IDs should not alter the approximation."""
    spectrum = _spectrum()
    source = _source(spectrum)
    original = CompressedRootMixture.from_source(
        source,
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=7,
        restart_count=3,
        random_seed=4242,
    )

    alphas, design, noise_sd = _native_inputs()
    permutation = np.array([2, 0, 3, 1])
    permuted_native = AdditiveDirichletAggregation(
        alphas[permutation],
        design[:, permutation],
        noise_sd,
        np.eye(3),
    )
    permuted_spectrum = RootResidualSpectrum.from_aggregation(permuted_native)
    permuted_source = ConditionalAllocationMixture.from_aggregation(
        AdditiveDirichletAggregation(
            alphas[permutation],
            design[:, permutation],
            noise_sd,
            permuted_spectrum.basis,
        ),
        np.zeros(4, dtype=np.int64),
        sample_count=64,
        source_seed=731,
        source_provenance="permuted unit-test exact-mixture source",
        cell_ids=np.array([101, 102, 201, 202], dtype=np.int64)[permutation],
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    permuted = CompressedRootMixture.from_source(
        permuted_source,
        permuted_spectrum,
        mixture_rank=permuted_spectrum.retained_rank,
        component_count=7,
        restart_count=3,
        random_seed=4242,
    )

    np.testing.assert_allclose(spectrum.eigenvalues, permuted_spectrum.eigenvalues, atol=5.0e-15)
    np.testing.assert_allclose(spectrum.basis, permuted_spectrum.basis, atol=5.0e-15)
    np.testing.assert_allclose(
        source.projected_unit_mass_residual_factors,
        permuted_source.projected_unit_mass_residual_factors,
        atol=5.0e-15,
    )
    np.testing.assert_allclose(original.weights, permuted.weights, atol=2.0e-16)
    np.testing.assert_allclose(original.means, permuted.means, atol=5.0e-15)
    np.testing.assert_allclose(original.covariances, permuted.covariances, atol=5.0e-15)
    observation = np.array([0.8, -0.3, 1.4])
    assert original.log_likelihood(observation, 1.7) == pytest.approx(
        permuted.log_likelihood(observation, 1.7),
        abs=2.0e-14,
    )


def test_singleton_compression_has_zero_covariance_and_zero_kl_bound() -> None:
    """One component per source location should be the unchanged finite mixture."""
    spectrum = _spectrum()
    source, artifact = _compressed(
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=16,
        sample_count=16,
    )

    assert artifact.component_count == source.sample_count
    np.testing.assert_array_equal(
        artifact.covariances,
        np.zeros_like(artifact.covariances),
    )
    assert artifact.kl_upper_bound == 0.0


@pytest.mark.parametrize("mixture_rank", [0, 1, 3])
def test_hybrid_likelihood_matches_manual_gaussian_mixture(
    mixture_rank: int,
) -> None:
    """The q=0, q<r, and q=r likelihoods should match a direct decomposition."""
    spectrum = _spectrum()
    _, artifact = _compressed(
        spectrum,
        mixture_rank=mixture_rank,
        component_count=5,
    )
    observation = np.array([0.8, -0.3, 1.4])
    offset = np.array([0.1, -0.2, 0.05])
    root_mass = 1.7
    expected = _manual_compressed_log_likelihood(
        artifact,
        observation,
        root_mass,
        offset,
    )

    assert artifact.log_likelihood(
        observation,
        root_mass,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)
    assert compressed_root_mixture_log_likelihood(
        observation,
        root_mass,
        artifact,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)


def test_q_zero_matches_the_existing_low_rank_gaussian_closure() -> None:
    """The no-mixture limit should exactly recover the Gaussian moment closure."""
    spectrum = _spectrum()
    _, artifact = _compressed(
        spectrum,
        mixture_rank=0,
        component_count=1,
    )
    observation = np.array([0.8, -0.3, 1.4])
    offset = np.array([0.1, -0.2, 0.05])
    root_mass = 1.7
    mean = offset + root_mass * spectrum.observation_mean_design
    expected = low_rank_gaussian_log_likelihood(
        observation,
        mean,
        spectrum.noise_sd,
        spectrum.basis,
        root_mass**2 * np.diag(spectrum.eigenvalues),
    )

    assert artifact.log_likelihood(
        observation,
        root_mass,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)


def test_no_compression_matches_the_direct_finite_source_mixture() -> None:
    """Full-rank singleton clusters should replay the direct source density."""
    spectrum = _spectrum()
    source, artifact = _compressed(
        spectrum,
        mixture_rank=spectrum.retained_rank,
        component_count=16,
        sample_count=16,
    )
    observation = np.array([0.8, -0.3, 1.4])
    offset = np.array([0.1, -0.2, 0.05])
    root_mass = 1.7
    proportions = _native_inputs()[0] / float(np.sum(_native_inputs()[0]))
    mean_design = _native_inputs()[1] @ proportions
    locations = source.projected_unit_mass_residual_factors[:, :, 0]
    component_logp = []
    for location in locations:
        component_mean = (
            offset + root_mass * mean_design + root_mass * spectrum.noise_sd * (spectrum.basis @ location)
        )
        component_logp.append(
            multivariate_normal.logpdf(
                observation,
                mean=component_mean,
                cov=np.diag(np.square(spectrum.noise_sd)),  # pyright: ignore[reportArgumentType]
            )
        )
    expected = float(
        logsumexp(component_logp)  # pyright: ignore[reportOperatorIssue]
        - math.log(source.sample_count)
    )

    assert artifact.log_likelihood(
        observation,
        root_mass,
        offset=offset,
    ) == pytest.approx(expected, abs=3.0e-13)


def test_rank_zero_likelihood_is_the_normalized_diagonal_gaussian() -> None:
    """No aggregation directions should leave only measurement-error density."""
    aggregation = AdditiveDirichletAggregation(
        np.array([0.8, 1.3]),
        np.array([[0.4, 0.4], [-0.7, -0.7]]),
        np.array([0.5, 0.8]),
        np.eye(2),
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    source = ConditionalAllocationMixture.from_aggregation(
        AdditiveDirichletAggregation(
            aggregation.cell_alphas,
            aggregation.design,
            aggregation.noise_sd,
            spectrum.basis,
        ),
        np.zeros(2, dtype=np.int64),
        sample_count=8,
        source_seed=31,
        source_provenance="rank-zero source",
        cell_ids=np.array([10, 20]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    artifact = CompressedRootMixture.from_source(
        source,
        spectrum,
        mixture_rank=0,
        component_count=1,
        restart_count=2,
        random_seed=17,
    )
    observation = np.array([0.2, -0.4])
    offset = np.array([0.1, 0.05])
    root_mass = 1.6
    expected = multivariate_normal.logpdf(
        observation,
        mean=offset + root_mass * spectrum.observation_mean_design,
        cov=np.diag(np.square(spectrum.noise_sd)),  # pyright: ignore[reportArgumentType]
    )

    assert artifact.log_likelihood(
        observation,
        root_mass,
        offset=offset,
    ) == pytest.approx(expected, abs=2.0e-15)


def test_chunked_projected_bank_matches_all_at_once_leading_coordinates() -> None:
    """Chunking should preserve the frozen Sobol bank before compression."""
    spectrum = _spectrum()
    full_source = _source(spectrum, sample_count=64)
    chunked = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
    )
    larger_memory_chunk = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=16,
    )

    assert chunked.summary_rank == 2
    assert chunked.construction_sample_chunk_size == 8
    assert chunked.construction_projection_chunk_size == 4
    np.testing.assert_allclose(
        chunked.projected_unit_mass_residual_factors,
        full_source.projected_unit_mass_residual_factors[:, :2],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_array_equal(
        chunked.projected_unit_mass_residual_factors,
        larger_memory_chunk.projected_unit_mass_residual_factors,
    )


def test_chunked_projected_bank_replays_and_roundtrips_v3() -> None:
    """A fixed chunk traversal should replay its values and v3 identity."""
    spectrum = _spectrum()
    first = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
    )
    replayed_construction = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
    )
    serialized = first.to_json()
    replayed_artifact = ConditionalAllocationMixture.from_json(
        serialized,
        expected_sha256=first.sha256,
    )

    np.testing.assert_array_equal(
        first.projected_unit_mass_residual_factors,
        replayed_construction.projected_unit_mass_residual_factors,
    )
    assert first.sha256 == replayed_construction.sha256
    assert replayed_artifact.to_json() == serialized
    assert replayed_artifact.sha256 == first.sha256
    assert json.loads(serialized)["schema"] == ("aggregation-conditional-allocation-mixture-v3")


def test_chunked_projected_bank_preserves_nested_same_seed_prefix() -> None:
    """Increasing a Sobol bank should retain every earlier projected row."""
    spectrum = _spectrum()
    short = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=32,
        sample_chunk_size=8,
        projection_chunk_size=4,
    )
    long = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=16,
        projection_chunk_size=4,
    )

    np.testing.assert_array_equal(
        short.projected_unit_mass_residual_factors,
        long.projected_unit_mass_residual_factors[:32],
    )


def test_chunked_projected_bank_preserves_stable_cell_permutation() -> None:
    """Stable IDs should make v3 independent of native storage order."""
    alphas, design, noise_sd = _native_inputs()
    cell_ids = np.array([101, 102, 201, 202], dtype=np.int64)
    original_native = AdditiveDirichletAggregation(
        alphas,
        design,
        noise_sd,
        np.eye(3),
    )
    original_spectrum = RootResidualSpectrum.from_aggregation(original_native)
    original = build_chunked_projected_root_bank(
        original_native,
        original_spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
        projection_chunk_size=4,
        source_seed=731,
        source_provenance="original v3 stable-ID source",
        cell_ids=cell_ids,
    )
    permutation = np.array([2, 0, 3, 1])
    permuted_native = AdditiveDirichletAggregation(
        alphas[permutation],
        design[:, permutation],
        noise_sd,
        np.eye(3),
    )
    permuted_spectrum = RootResidualSpectrum.from_aggregation(permuted_native)
    permuted = build_chunked_projected_root_bank(
        permuted_native,
        permuted_spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
        projection_chunk_size=4,
        source_seed=731,
        source_provenance="permuted v3 stable-ID source",
        cell_ids=cell_ids[permutation],
    )

    np.testing.assert_allclose(
        original.projected_unit_mass_residual_factors,
        permuted.projected_unit_mass_residual_factors,
        rtol=0.0,
        atol=5.0e-15,
    )


def test_projected_source_compresses_like_legacy_full_rank_source() -> None:
    """A q-rank bank should preserve leading-q compression and likelihood."""
    spectrum = _spectrum()
    full_source = _source(spectrum, sample_count=64)
    projected_source = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
    )
    legacy = CompressedRootMixture.from_source(
        full_source,
        spectrum,
        mixture_rank=2,
        component_count=1,
        random_seed=919,
    )
    projected = CompressedRootMixture.from_source(
        projected_source,
        spectrum,
        mixture_rank=2,
        component_count=1,
        random_seed=919,
    )

    np.testing.assert_allclose(projected.means, legacy.means, atol=2.0e-16)
    np.testing.assert_allclose(
        projected.covariances,
        legacy.covariances,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert projected.log_likelihood(
        np.array([0.4, -0.3, 0.8]),
        1.7,
    ) == pytest.approx(
        legacy.log_likelihood(np.array([0.4, -0.3, 0.8]), 1.7),
        abs=2.0e-14,
    )


def test_rank_zero_chunked_source_uses_full_gaussian_complement() -> None:
    """A zero-column bank should leave every spectrum direction in closure."""
    spectrum = _spectrum()
    source = _chunked_source(
        spectrum,
        mixture_rank=0,
        sample_count=16,
        sample_chunk_size=8,
        projection_chunk_size=4,
    )
    artifact = CompressedRootMixture.from_source(
        source,
        spectrum,
        mixture_rank=0,
        component_count=1,
    )
    observation = np.array([0.4, -0.3, 0.8])
    root_mass = 1.7
    expected = low_rank_gaussian_log_likelihood(
        observation,
        root_mass * spectrum.observation_mean_design,
        spectrum.noise_sd,
        spectrum.basis,
        np.diag(root_mass**2 * spectrum.eigenvalues),
    )

    assert source.projected_unit_mass_residual_factors.shape == (16, 0, 1)
    assert artifact.log_likelihood(observation, root_mass) == pytest.approx(
        expected,
        abs=2.0e-14,
    )


def test_compression_rejects_nearly_rotated_source_basis() -> None:
    """Mixture coordinates require an exact leading spectrum-basis identity."""
    spectrum = _spectrum()
    source = _source(spectrum)
    angle = 1.0e-12
    rotation = np.eye(source.summary_rank)
    rotation[:2, :2] = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    rotated_basis = source.summary_basis @ rotation
    rotated_source = ConditionalAllocationMixture(
        source.projected_unit_mass_residual_factors,
        source.observation_mean_design,
        source.noise_sd,
        rotated_basis,
        source.labels,
        source.cell_ids,
        source.alpha_totals,
        source.source_seed,
        "nearly rotated source",
        source.cell_alphas_sha256,
        source.design_sha256,
        source.noise_sd_sha256,
        conditional_mixture_module._array_sha256(rotated_basis),
        source.construction_method,
        source.construction_scipy_version,
    )

    with pytest.raises(ValueError, match="leading prefix"):
        CompressedRootMixture.from_source(
            rotated_source,
            spectrum,
            mixture_rank=2,
            component_count=1,
        )


def test_chunked_projected_bank_rejects_invalid_chunk_and_native_identity() -> None:
    """Invalid chunk geometry and mismatched spectra should fail closed."""
    spectrum = _spectrum()
    with pytest.raises(ValueError, match="power of two"):
        _chunked_source(
            spectrum,
            mixture_rank=2,
            sample_count=64,
            sample_chunk_size=7,
            projection_chunk_size=4,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _chunked_source(
            spectrum,
            mixture_rank=2,
            sample_count=64,
            sample_chunk_size=128,
            projection_chunk_size=4,
        )
    incompatible = AdditiveDirichletAggregation(
        np.array([0.8, 1.1, 1.6, 0.9]),
        _native_inputs()[1],
        _native_inputs()[2],
        np.eye(3),
    )
    with pytest.raises(ValueError, match="native identities"):
        build_chunked_projected_root_bank(
            incompatible,
            spectrum,
            mixture_rank=2,
            sample_count=64,
            sample_chunk_size=8,
            projection_chunk_size=4,
            source_seed=731,
            source_provenance="incompatible source",
        )


def test_chunked_constructor_never_requests_more_than_one_sample_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sobol temporaries should remain bounded by the declared chunk size."""
    from scipy.stats import qmc

    observed_counts: list[int] = []
    original = qmc.Sobol.random

    def recording_random(
        self: qmc.Sobol,
        n: int = 1,
        *,
        workers: int = 1,
    ) -> np.ndarray:
        """Record each allocation request before delegating to SciPy."""
        observed_counts.append(n)
        return original(self, n=n, workers=workers)

    monkeypatch.setattr(qmc.Sobol, "random", recording_random)
    _chunked_source(
        _spectrum(),
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
        projection_chunk_size=4,
    )

    assert observed_counts
    assert max(observed_counts) == 8


def test_chunked_projected_bank_preserves_forced_multiblock_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking should preserve coordinates across Sobol dimension blocks."""
    monkeypatch.setattr(
        conditional_mixture_module,
        "_SOBOL_MAX_DIMENSION",
        2,
    )
    spectrum = _spectrum()
    full_source = _source(spectrum, sample_count=64)
    chunked = _chunked_source(
        spectrum,
        mixture_rank=2,
        sample_count=64,
        sample_chunk_size=8,
        projection_chunk_size=4,
    )

    np.testing.assert_allclose(
        chunked.projected_unit_mass_residual_factors,
        full_source.projected_unit_mass_residual_factors[:, :2],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_chunked_centring_matches_stored_mean_for_heterogeneous_large_alphas() -> None:
    """Projected residuals and means should share one represented alpha total."""
    aggregation = AdditiveDirichletAggregation(
        np.array([1.0, 8.0e-17, 8.0e-17, 8.0e-17]),
        np.array(
            [
                [0.2, 1.1e9, -0.7e9, 0.9e9],
                [1.4, -0.3e9, 0.8e9, 0.1e9],
            ]
        ),
        np.array([0.6, 0.9]),
        np.eye(2),
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    source_aggregation = AdditiveDirichletAggregation(
        aggregation.cell_alphas,
        aggregation.design,
        aggregation.noise_sd,
        spectrum.basis,
    )
    full_source = ConditionalAllocationMixture.from_aggregation(
        source_aggregation,
        np.zeros(4, dtype=np.int64),
        sample_count=16,
        source_seed=731,
        source_provenance="large-alpha all-at-once source",
        cell_ids=np.array([101, 102, 201, 202]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    chunked = build_chunked_projected_root_bank(
        aggregation,
        spectrum,
        mixture_rank=spectrum.retained_rank,
        sample_count=16,
        sample_chunk_size=8,
        projection_chunk_size=4,
        source_seed=731,
        source_provenance="large-alpha chunked source",
        cell_ids=np.array([101, 102, 201, 202]),
    )

    np.testing.assert_array_equal(
        chunked.observation_mean_design,
        full_source.observation_mean_design,
    )
    np.testing.assert_allclose(
        chunked.projected_unit_mass_residual_factors,
        full_source.projected_unit_mass_residual_factors,
        rtol=0.0,
        atol=2.0e-15,
    )
