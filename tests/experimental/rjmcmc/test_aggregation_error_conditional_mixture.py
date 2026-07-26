"""Tests for the frozen conditional-allocation aggregation mixture."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_conditional_mixture as conditional_mixture_module,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    TwoCellAggregationOracle,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
    conditional_allocation_mixture_log_likelihood,
    conditional_allocation_mixture_log_likelihood_and_gradient,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _model(*, rank: int = 2) -> AdditiveDirichletAggregation:
    """Return one small heterogeneous fixed native model."""
    alphas = np.asarray([0.7, 1.1, 1.6, 0.9])
    design = np.asarray(
        [
            [1.8, -0.5, 0.3, 0.9],
            [0.2, 1.4, -0.7, 0.1],
            [0.5, -0.2, 1.1, 0.8],
        ]
    )
    noise_sd = np.asarray([0.35, 0.8, 0.6])
    if rank == 0:
        basis = np.empty((3, 0))
    else:
        raw_basis = np.asarray([[1.0, 0.3], [0.4, 1.0], [-0.2, 0.5]])
        basis, _ = np.linalg.qr(raw_basis)
        basis = basis[:, :rank]
    return AdditiveDirichletAggregation(alphas, design, noise_sd, basis)


def _artifact(
    *,
    sample_count: int = 2_000,
    rank: int = 2,
) -> ConditionalAllocationMixture:
    """Return a replayable two-region conditional mixture."""
    return ConditionalAllocationMixture.from_aggregation(
        _model(rank=rank),
        np.asarray([0, 0, 1, 1]),
        sample_count=sample_count,
        source_seed=731,
        source_provenance="unit-test fixed allocation bank",
        cell_ids=np.asarray([101, 102, 201, 202]),
    )


def _sobol_artifact(
    *,
    sample_count: int = 64,
    rank: int = 2,
) -> ConditionalAllocationMixture:
    """Return one balanced-tree scrambled-Sobol allocation bank."""
    return ConditionalAllocationMixture.from_aggregation(
        _model(rank=rank),
        np.asarray([0, 0, 1, 1]),
        sample_count=sample_count,
        source_seed=731,
        source_provenance="unit-test Sobol allocation bank",
        cell_ids=np.asarray([101, 102, 201, 202]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )


def test_default_pcg64_construction_preserves_v1_golden_artifact() -> None:
    """The new opt-in construction must not change the legacy v1 contract."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cell_ids = np.asarray([101, 102, 201, 202])
    artifact = _artifact(sample_count=17)

    expected_factors = np.empty_like(artifact.projected_unit_mass_residual_factors)
    flat_alphas = model.cell_alphas.reshape(-1)
    flat_ids = cell_ids.reshape(-1)
    summary_design = model.summary_design
    for region in range(2):
        selected = np.flatnonzero(labels == region)
        sorted_indices = selected[np.argsort(flat_ids[selected], kind="stable")]
        sorted_ids = np.asarray(flat_ids[sorted_indices], dtype=np.int64)
        digest = hashlib.sha256(b"aggregation-conditional-allocation-mixture-v1")
        digest.update((731).to_bytes(8, byteorder="little", signed=False))
        digest.update(np.ascontiguousarray(sorted_ids, dtype="<i8").tobytes())
        region_seed = int.from_bytes(
            digest.digest(),
            byteorder="little",
            signed=False,
        )
        region_alphas = flat_alphas[sorted_indices]
        shares = np.random.Generator(np.random.PCG64(region_seed)).dirichlet(region_alphas, size=17)
        region_columns = summary_design[:, sorted_indices]
        expected = region_columns @ (region_alphas / float(np.sum(region_alphas)))
        expected_factors[:, :, region] = shares @ region_columns.T - expected[np.newaxis, :]

    np.testing.assert_array_equal(
        artifact.projected_unit_mass_residual_factors,
        expected_factors,
    )
    assert artifact.construction_method == "keyed_pcg64_dirichlet"
    assert artifact.payload["schema"] == ("aggregation-conditional-allocation-mixture-v1")
    assert "construction_method" not in artifact.payload
    # NumPy's Dirichlet implementation has produced both authenticated byte
    # streams across version/build/architecture combinations. The direct
    # construction above is the platform-local compatibility oracle; this set
    # additionally rejects any result outside the observed exact legacy
    # streams.
    authenticated_goldens = {
        (
            "b3b229bfc247d65834582c41d99b1807be44506dcf754f0411ae8a4bc3a5e242",
            "347a74cfa0e84ab1ee7adc5b4ead73f0c1c2c6853713398517e9e2879872fce2",
        ),
        (
            "021c3d9a11e1dd1896e1643847151142ee6d4e96273d3a4023b5423d79d7bd19",
            "7544a0bcf048ebfe2d4449311407db3cb2a7a381bdd52525a2473666a0b2895e",
        ),
    }
    observed_golden = (
        artifact.sha256,
        hashlib.sha256(artifact.projected_unit_mass_residual_factors.tobytes()).hexdigest(),
    )
    assert observed_golden in authenticated_goldens


def test_scrambled_sobol_bank_has_exact_nested_prefixes_and_private_rng() -> None:
    """Larger same-seed Sobol banks must extend rather than replace draws."""
    np.random.seed(5129)
    expected_global_draws = np.random.random(8)
    np.random.seed(5129)

    small = _sobol_artifact(sample_count=64)
    replay = _sobol_artifact(sample_count=64)
    large = _sobol_artifact(sample_count=256)
    different_seed = ConditionalAllocationMixture.from_aggregation(
        _model(),
        np.asarray([0, 0, 1, 1]),
        sample_count=64,
        source_seed=732,
        source_provenance="unit-test Sobol allocation bank",
        cell_ids=np.asarray([101, 102, 201, 202]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    observed_global_draws = np.random.random(8)

    np.testing.assert_array_equal(observed_global_draws, expected_global_draws)
    np.testing.assert_array_equal(
        small.projected_unit_mass_residual_factors,
        replay.projected_unit_mass_residual_factors,
    )
    np.testing.assert_array_equal(
        small.projected_unit_mass_residual_factors,
        large.projected_unit_mass_residual_factors[:64],
    )
    assert not np.array_equal(
        small.projected_unit_mass_residual_factors,
        different_seed.projected_unit_mass_residual_factors,
    )
    assert small.payload["schema"] == ("aggregation-conditional-allocation-mixture-v2")
    assert small.payload["sobol_block_dimensions"] == [2]


def test_scrambled_sobol_bank_is_cell_and_region_label_invariant() -> None:
    """Stable-ID catalogues should remove cell-order and label-order effects."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cell_ids = np.asarray([101, 102, 201, 202])
    original = _sobol_artifact(sample_count=64)
    relabelled = ConditionalAllocationMixture.from_aggregation(
        model,
        1 - labels,
        sample_count=64,
        source_seed=731,
        source_provenance="unit-test Sobol allocation bank",
        cell_ids=cell_ids,
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    order = np.asarray([2, 0, 3, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas[order],
        model.design[:, order],
        model.noise_sd,
        model.summary_basis,
    )
    permuted = ConditionalAllocationMixture.from_aggregation(
        permuted_model,
        labels[order],
        sample_count=64,
        source_seed=731,
        source_provenance="unit-test Sobol allocation bank",
        cell_ids=cell_ids[order],
        construction_method="scrambled_sobol_balanced_dirichlet",
    )

    np.testing.assert_array_equal(
        original.projected_unit_mass_residual_factors,
        relabelled.projected_unit_mass_residual_factors[:, :, ::-1],
    )
    np.testing.assert_array_equal(
        original.projected_unit_mass_residual_factors,
        permuted.projected_unit_mass_residual_factors,
    )
    assert original.payload["sobol_catalogue_sha256"] == (relabelled.payload["sobol_catalogue_sha256"])
    assert original.payload["sobol_catalogue_sha256"] == (permuted.payload["sobol_catalogue_sha256"])


@pytest.mark.parametrize(
    ("labels", "masses"),
    [
        pytest.param(
            np.asarray([0, 0, 0, 0]),
            np.asarray([2.3]),
            id="depth-two-root",
        ),
        pytest.param(
            np.asarray([0, 0, 1, 1]),
            np.asarray([2.3, 1.7]),
            id="two-region-product",
        ),
    ],
)
def test_scrambled_sobol_bank_reproduces_analytic_dirichlet_moments(
    labels: np.ndarray,
    masses: np.ndarray,
) -> None:
    """RQMC root and product banks should recover exact projected moments."""
    model = _model()
    cached = model.partition_factors(labels)
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        cached,
        sample_count=16_384,
        source_seed=731,
        source_provenance="Sobol moment regression",
        cell_ids=np.asarray([101, 102, 201, 202]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    residual_draws = np.einsum(
        "sqr,r->sq",
        artifact.projected_unit_mass_residual_factors,
        masses,
        optimize=False,
    )

    # These tolerances are substantially tighter than ordinary Monte Carlo
    # error at this bank size, but deliberately avoid pinning tests to the
    # last digits of one SciPy inverse-transform implementation.
    np.testing.assert_allclose(
        residual_draws.mean(axis=0),
        np.zeros(artifact.summary_rank),
        rtol=0.0,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        np.cov(residual_draws, rowvar=False, bias=True),
        cached.summary_residual_covariance(masses),
        rtol=5.0e-4,
        atol=5.0e-4,
    )
    if artifact.region_count == 2:
        first = artifact.projected_unit_mass_residual_factors[:, :, 0]
        second = artifact.projected_unit_mass_residual_factors[:, :, 1]
        centered_first = first - np.mean(first, axis=0)
        centered_second = second - np.mean(second, axis=0)
        cross_covariance = centered_first.T @ centered_second / artifact.sample_count
        np.testing.assert_allclose(
            cross_covariance,
            np.zeros_like(cross_covariance),
            rtol=0.0,
            atol=5.0e-4,
        )


def test_scrambled_sobol_multiblock_replays_and_is_permutation_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple Sobol blocks must retain nesting, metadata, and stable-ID symmetry."""
    monkeypatch.setattr(
        conditional_mixture_module,
        "_SOBOL_MAX_DIMENSION",
        2,
    )
    alphas = np.asarray([0.4, 0.7, 1.1, 0.8, 1.3, 1.8])
    design = np.asarray(
        [
            [1.2, -0.5, 0.4, 0.8, -0.2, 0.1],
            [0.3, 0.9, -0.7, 0.2, 1.0, -0.4],
        ]
    )
    model = AdditiveDirichletAggregation(
        alphas,
        design,
        np.asarray([0.5, 0.8]),
        np.eye(2),
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    cell_ids = np.asarray([101, 102, 103, 201, 202, 203])

    def build(
        candidate_model: AdditiveDirichletAggregation,
        candidate_labels: np.ndarray,
        candidate_ids: np.ndarray,
        *,
        sample_count: int,
    ) -> ConditionalAllocationMixture:
        """Build one forced-multiblock artifact."""
        return ConditionalAllocationMixture.from_aggregation(
            candidate_model,
            candidate_labels,
            sample_count=sample_count,
            source_seed=9281,
            source_provenance="forced Sobol multiblock regression",
            cell_ids=candidate_ids,
            construction_method="scrambled_sobol_balanced_dirichlet",
        )

    small = build(model, labels, cell_ids, sample_count=8)
    large = build(model, labels, cell_ids, sample_count=32)
    relabelled = build(model, 1 - labels, cell_ids, sample_count=8)
    order = np.asarray([4, 0, 5, 2, 3, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas[order],
        model.design[:, order],
        model.noise_sd,
        model.summary_basis,
    )
    permuted = build(
        permuted_model,
        labels[order],
        cell_ids[order],
        sample_count=8,
    )

    assert small.payload["schema"] == "aggregation-conditional-allocation-mixture-v2"
    assert small.payload["sobol_block_dimensions"] == [2, 2]
    np.testing.assert_array_equal(
        small.projected_unit_mass_residual_factors,
        large.projected_unit_mass_residual_factors[:8],
    )
    np.testing.assert_array_equal(
        small.projected_unit_mass_residual_factors,
        relabelled.projected_unit_mass_residual_factors[:, :, ::-1],
    )
    np.testing.assert_array_equal(
        small.projected_unit_mass_residual_factors,
        permuted.projected_unit_mass_residual_factors,
    )
    serialized = small.to_json()
    replay = ConditionalAllocationMixture.from_json(
        serialized,
        expected_sha256=small.sha256,
    )
    assert replay.to_json() == serialized
    assert replay.payload["sobol_block_dimensions"] == [2, 2]


def test_scrambled_sobol_singletons_are_exact_and_tiny_alphas_are_safe() -> None:
    """Singletons stay exact and late tiny alpha subtrees must not cancel."""
    fine = ConditionalAllocationMixture.from_aggregation(
        _model(),
        np.arange(4, dtype=np.int64),
        sample_count=8,
        source_seed=41,
        source_provenance="Sobol singleton regression",
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    np.testing.assert_array_equal(
        fine.projected_unit_mass_residual_factors,
        np.zeros((8, 2, 4)),
    )
    assert fine.payload["sobol_block_dimensions"] == []

    heterogeneous = AdditiveDirichletAggregation(
        np.asarray([1.0e16, 1.0e-8, 2.0e-8]),
        np.asarray([[0.2, 1.0, -0.5]]),
        np.asarray([0.7]),
        np.ones((1, 1)),
    )
    heterogeneous_artifact = ConditionalAllocationMixture.from_aggregation(
        heterogeneous,
        np.zeros(3, dtype=np.int64),
        sample_count=16,
        source_seed=938,
        source_provenance="Sobol heterogeneous alpha regression",
        cell_ids=np.asarray([10, 20, 30]),
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    assert np.all(np.isfinite(heterogeneous_artifact.projected_unit_mass_residual_factors))


def test_scrambled_sobol_tiny_alpha_endpoint_draws_have_sound_first_moments() -> None:
    """Endpoint-heavy inverse draws should conserve mass and recover tiny means."""
    alphas = np.asarray([1.0e-4, 3.0e-3, 0.2, 2.0])
    expected_shares = alphas / float(np.sum(alphas))
    model = AdditiveDirichletAggregation(
        alphas,
        np.eye(4),
        np.ones(4),
        np.eye(4),
    )
    scramble_means = []
    for source_seed in (11, 29, 47, 83):
        artifact = ConditionalAllocationMixture.from_aggregation(
            model,
            np.zeros(4, dtype=np.int64),
            sample_count=4_096,
            source_seed=source_seed,
            source_provenance="Sobol tiny-alpha endpoint regression",
            construction_method="scrambled_sobol_balanced_dirichlet",
        )
        # Identity design/basis makes each residual coordinate equal to its
        # allocation share minus the exact Dirichlet mean.
        shares = artifact.projected_unit_mass_residual_factors[:, :, 0] + expected_shares[np.newaxis, :]
        assert np.all(np.isfinite(shares))
        assert np.all((shares >= 0.0) & (shares <= 1.0))
        np.testing.assert_allclose(
            np.sum(shares, axis=1),
            np.ones(artifact.sample_count),
            rtol=0.0,
            atol=4.0e-16,
        )
        scramble_means.append(np.mean(shares, axis=0))

    # Pooling independent scrambles is the relevant RQMC first-moment check.
    # The absolute tolerance deliberately covers the extremely rare first
    # component without imposing an unstable relative-error requirement.
    np.testing.assert_allclose(
        np.mean(scramble_means, axis=0),
        expected_shares,
        rtol=0.0,
        atol=3.0e-4,
    )


def test_scrambled_sobol_v2_roundtrip_and_metadata_tamper_rejection() -> None:
    """V2 construction metadata and its whole artifact must replay strictly."""
    artifact = _sobol_artifact(sample_count=64)
    serialized = artifact.to_json()
    replay = ConditionalAllocationMixture.from_json(
        serialized,
        expected_sha256=artifact.sha256,
    )

    assert replay.to_json() == serialized
    assert replay.sha256 == artifact.sha256
    assert replay.construction_method == ("scrambled_sobol_balanced_dirichlet")

    for key, value, match in (
        ("construction_method", "keyed_pcg64_dirichlet", "construction method"),
        ("inverse_transform", "scipy.special.gammaincinv", "inverse transform"),
        ("sobol_catalogue_sha256", "0" * 64, "catalogue identity"),
        ("sobol_block_dimensions", [1, 1], "block dimensions"),
    ):
        payload = json.loads(serialized)
        payload[key] = value
        tampered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        with pytest.raises(ValueError, match=match):
            ConditionalAllocationMixture.from_json(
                tampered,
                expected_sha256=hashlib.sha256(tampered.encode()).hexdigest(),
            )


def test_scrambled_sobol_rejects_non_power_of_two_and_unknown_method() -> None:
    """Construction method selection and Sobol balance must fail closed."""
    with pytest.raises(ValueError, match="power of two"):
        _sobol_artifact(sample_count=63)
    with pytest.raises(ValueError, match="construction_method"):
        ConditionalAllocationMixture.from_aggregation(
            _model(),
            np.asarray([0, 0, 1, 1]),
            sample_count=64,
            source_seed=731,
            source_provenance="unknown construction",
            construction_method="sobol-ish",
        )


def test_frozen_bank_replays_dirichlet_moments_and_cached_factors() -> None:
    """Frozen projected draws should reproduce exact block-Dirichlet moments."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cached = model.partition_factors(labels)
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        cached,
        sample_count=80_000,
        source_seed=9281,
        source_provenance="moment regression",
    )
    masses = np.asarray([2.3, 1.7])
    residual_draws = np.einsum(
        "sqr,r->sq",
        artifact.projected_unit_mass_residual_factors,
        masses,
        optimize=False,
    )
    empirical_mean = residual_draws.mean(axis=0)
    empirical_covariance = np.cov(residual_draws, rowvar=False, bias=True)
    exact_covariance = cached.summary_residual_covariance(masses)

    np.testing.assert_allclose(empirical_mean, np.zeros(2), atol=4.0e-3)
    np.testing.assert_allclose(
        empirical_covariance,
        exact_covariance,
        rtol=2.5e-2,
        atol=2.0e-3,
    )
    np.testing.assert_array_equal(
        artifact.observation_mean_design,
        cached.observation_mean_design,
    )
    assert artifact.sample_count == 80_000
    assert artifact.summary_rank == 2
    assert artifact.region_count == 2
    assert artifact.storage_nbytes == sum(
        array.nbytes
        for array in (
            artifact.projected_unit_mass_residual_factors,
            artifact.observation_mean_design,
            artifact.noise_sd,
            artifact.summary_basis,
            artifact.labels,
            artifact.cell_ids,
            artifact.alpha_totals,
        )
    )


def test_one_dimensional_density_is_normalized() -> None:
    """The equal conditional Gaussian mixture must integrate to one."""
    model = AdditiveDirichletAggregation(
        np.asarray([0.8, 1.3]),
        np.asarray([[1.7, -0.4]]),
        0.65,
        np.ones((1, 1)),
    )
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        np.zeros(2, dtype=np.int64),
        sample_count=257,
        source_seed=902,
        source_provenance="one-dimensional normalization",
    )

    integral, error = quad(
        lambda value: math.exp(artifact.log_likelihood([value], [2.1])),
        -np.inf,
        np.inf,
        epsabs=2.0e-10,
        epsrel=2.0e-10,
        limit=160,
    )

    assert integral == pytest.approx(1.0, abs=3.0e-10)
    assert error < 3.0e-9


def test_large_bank_converges_to_exact_two_cell_conditional_oracle() -> None:
    """A large frozen bank should approach the exact Beta-mixture likelihood."""
    alphas = np.asarray([0.7, 1.3])
    design = np.asarray([[1.5, -0.4]])
    noise_sd = np.asarray([0.8])
    model = AdditiveDirichletAggregation(
        alphas,
        design,
        noise_sd,
        np.ones((1, 1)),
    )
    artifact = ConditionalAllocationMixture.from_aggregation(
        model,
        np.zeros(2, dtype=np.int64),
        sample_count=100_000,
        source_seed=7741,
        source_provenance="two-cell exact-oracle convergence",
    )
    total = 2.2
    observation = np.asarray([0.7])
    oracle = TwoCellAggregationOracle(
        gamma_shape=2.0,
        gamma_rate=1.0,
        beta_first_shape=alphas[0],
        beta_second_shape=alphas[1],
        fraction_order=128,
    )

    expected = oracle.coarse_conditional_log_likelihood(
        total,
        observation,
        design,
        noise_sd,
    )
    observed = artifact.log_likelihood(observation, [total])

    assert observed == pytest.approx(expected, abs=4.0e-3)


def test_analytic_mass_gradient_matches_centered_finite_difference() -> None:
    """The gradient should include both retained mean and component transport."""
    artifact = _artifact(sample_count=1_500)
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])
    offset = np.asarray([0.1, -0.2, 0.05])
    logp, gradient = conditional_allocation_mixture_log_likelihood_and_gradient(
        observation,
        masses,
        artifact,
        mean_offset=offset,
    )
    step = 2.0e-6
    finite_difference = np.empty(2)
    for index in range(2):
        high = masses.copy()
        low = masses.copy()
        high[index] += step
        low[index] -= step
        finite_difference[index] = (
            conditional_allocation_mixture_log_likelihood(
                observation,
                high,
                artifact,
                mean_offset=offset,
            )
            - conditional_allocation_mixture_log_likelihood(
                observation,
                low,
                artifact,
                mean_offset=offset,
            )
        ) / (2.0 * step)

    assert logp == artifact.log_likelihood(
        observation,
        masses,
        mean_offset=offset,
    )
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    assert not gradient.flags.writeable


def test_region_relabeling_and_cell_permutation_preserve_the_density() -> None:
    """Stable cell IDs should make replay independent of incidental ordering."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    cell_ids = np.asarray([101, 102, 201, 202])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids,
    )
    relabelled = ConditionalAllocationMixture.from_aggregation(
        model,
        1 - labels,
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids,
    )
    np.testing.assert_array_equal(
        original.projected_unit_mass_residual_factors,
        relabelled.projected_unit_mass_residual_factors[:, :, ::-1],
    )

    cell_order = np.asarray([2, 0, 3, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas[cell_order],
        model.design[:, cell_order],
        model.noise_sd,
        model.summary_basis,
    )
    cell_permuted = ConditionalAllocationMixture.from_aggregation(
        permuted_model,
        labels[cell_order],
        sample_count=512,
        source_seed=831,
        source_provenance="permutation baseline",
        cell_ids=cell_ids[cell_order],
    )
    np.testing.assert_allclose(
        original.projected_unit_mass_residual_factors,
        cell_permuted.projected_unit_mass_residual_factors,
        rtol=0.0,
        atol=3.0e-16,
    )

    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])
    assert original.log_likelihood(observation, masses) == pytest.approx(
        relabelled.log_likelihood(observation, masses[::-1]),
        abs=2.0e-15,
    )
    assert original.log_likelihood(observation, masses) == pytest.approx(
        cell_permuted.log_likelihood(observation, masses),
        abs=2.0e-15,
    )


def test_seeded_construction_replays_without_using_global_numpy_rng() -> None:
    """Bank construction should use only its private keyed PCG64 streams."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    np.random.seed(1928)
    expected_global_draws = np.random.random(8)
    np.random.seed(1928)

    first = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=99,
        source_provenance="private RNG replay",
    )
    replay = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=99,
        source_provenance="private RNG replay",
    )
    different = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=128,
        source_seed=100,
        source_provenance="private RNG replay",
    )
    observed_global_draws = np.random.random(8)

    np.testing.assert_array_equal(observed_global_draws, expected_global_draws)
    np.testing.assert_array_equal(
        first.projected_unit_mass_residual_factors,
        replay.projected_unit_mass_residual_factors,
    )
    assert not np.array_equal(
        first.projected_unit_mass_residual_factors,
        different.projected_unit_mass_residual_factors,
    )
    assert first.sha256 == replay.sha256
    assert first.sha256 != different.sha256


def test_observation_permutation_preserves_logp_and_gradient() -> None:
    """A simultaneous row permutation is only a change of observation order."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=600,
        source_seed=991,
        source_provenance="observation permutation",
    )
    order = np.asarray([2, 0, 1])
    permuted_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design[order],
        model.noise_sd[order],
        model.summary_basis[order],
    )
    permuted = ConditionalAllocationMixture.from_aggregation(
        permuted_model,
        labels,
        sample_count=600,
        source_seed=991,
        source_provenance="observation permutation",
    )
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])

    original_result = original.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    permuted_result = permuted.log_likelihood_and_mass_gradient(
        observation[order],
        masses,
    )

    assert original_result[0] == pytest.approx(permuted_result[0], abs=3.0e-14)
    np.testing.assert_allclose(
        original_result[1],
        permuted_result[1],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_signed_summary_coordinate_permutation_preserves_evaluation() -> None:
    """Signed orthogonal summary coordinates should not change the density."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    original = ConditionalAllocationMixture.from_aggregation(
        model,
        labels,
        sample_count=600,
        source_seed=225,
        source_provenance="summary coordinate baseline",
    )
    coordinate_map = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    transformed_model = AdditiveDirichletAggregation(
        model.cell_alphas,
        model.design,
        model.noise_sd,
        model.summary_basis @ coordinate_map,
    )
    transformed = ConditionalAllocationMixture.from_aggregation(
        transformed_model,
        labels,
        sample_count=600,
        source_seed=225,
        source_provenance="summary coordinate baseline",
    )
    observation = np.asarray([0.4, -0.7, 1.2])
    masses = np.asarray([1.8, 2.4])

    original_result = original.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    transformed_result = transformed.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )

    assert original_result[0] == pytest.approx(
        transformed_result[0],
        abs=3.0e-14,
    )
    np.testing.assert_allclose(
        original_result[1],
        transformed_result[1],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_rank_zero_and_fine_partition_reduce_to_diagonal_gaussian() -> None:
    """Rank zero is the declared diagonal closure; a fine partition is exact."""
    observation = np.asarray([0.4, -0.7, 1.2])
    offset = np.asarray([0.1, -0.2, 0.05])

    rank_zero_model = _model(rank=0)
    rank_zero = ConditionalAllocationMixture.from_aggregation(
        rank_zero_model,
        np.asarray([0, 0, 1, 1]),
        sample_count=19,
        source_seed=52,
        source_provenance="rank-zero limit",
    )
    masses = np.asarray([1.8, 2.4])
    mean = offset + rank_zero.observation_mean_design @ masses
    residual = (observation - mean) / rank_zero.noise_sd
    expected = -float(np.sum(np.log(rank_zero.noise_sd))) - 0.5 * (
        3 * math.log(2.0 * math.pi) + float(residual @ residual)
    )
    assert rank_zero.summary_rank == 0
    assert rank_zero.projected_unit_mass_residual_factors.shape == (19, 0, 2)
    assert rank_zero.log_likelihood(
        observation,
        masses,
        mean_offset=offset,
    ) == pytest.approx(expected, abs=2.0e-15)

    fine_model = _model()
    fine = ConditionalAllocationMixture.from_aggregation(
        fine_model,
        np.arange(4, dtype=np.int64),
        sample_count=23,
        source_seed=52,
        source_provenance="fine-cell limit",
    )
    fine_masses = np.asarray([0.5, 1.2, 0.8, 1.5])
    np.testing.assert_array_equal(
        fine.projected_unit_mass_residual_factors,
        np.zeros((23, 2, 4)),
    )
    fine_mean = offset + fine.observation_mean_design @ fine_masses
    fine_residual = (observation - fine_mean) / fine.noise_sd
    fine_expected = -float(np.sum(np.log(fine.noise_sd))) - 0.5 * (
        3 * math.log(2.0 * math.pi) + float(fine_residual @ fine_residual)
    )
    assert fine.log_likelihood(
        observation,
        fine_masses,
        mean_offset=offset,
    ) == pytest.approx(fine_expected, abs=8.0e-15)


def test_artifact_json_roundtrip_fingerprint_and_immutability() -> None:
    """Canonical serialization should replay rank-zero shapes and identities."""
    artifact = _artifact(sample_count=17, rank=0)
    serialized = artifact.to_json()
    replay = ConditionalAllocationMixture.from_json(
        serialized,
        expected_sha256=artifact.sha256,
    )

    assert replay.to_json() == serialized
    assert replay.sha256 == artifact.sha256
    assert replay.partition_sha256 == artifact.partition_sha256
    assert replay.source_operator_sha256 == artifact.source_operator_sha256
    assert replay.projected_unit_mass_residual_factors.shape == (17, 0, 2)
    assert all(
        not values.flags.writeable
        for values in (
            replay.projected_unit_mass_residual_factors,
            replay.observation_mean_design,
            replay.noise_sd,
            replay.summary_basis,
            replay.labels,
            replay.cell_ids,
            replay.alpha_totals,
        )
    )
    for values in (
        replay.projected_unit_mass_residual_factors,
        replay.observation_mean_design,
        replay.noise_sd,
        replay.summary_basis,
        replay.labels,
        replay.cell_ids,
        replay.alpha_totals,
    ):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(ValueError, match="SHA-256"):
        ConditionalAllocationMixture.from_json(
            serialized,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        noncanonical = json.dumps(json.loads(serialized))
        ConditionalAllocationMixture.from_json(
            noncanonical,
            expected_sha256=hashlib.sha256(noncanonical.encode()).hexdigest(),
        )
    axes_payload = json.loads(serialized)
    axes_payload["factor_axes"] = ["region", "summary", "sample"]
    wrong_axes = json.dumps(
        axes_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="factor axes"):
        ConditionalAllocationMixture.from_json(
            wrong_axes,
            expected_sha256=hashlib.sha256(wrong_axes.encode()).hexdigest(),
        )


def test_replay_rejects_tampered_stored_arrays_and_external_identity() -> None:
    """Internal hashes and the required whole-artifact pin should fail closed."""
    artifact = _artifact(sample_count=17)
    serialized = artifact.to_json()

    noise_payload = json.loads(serialized)
    noise_payload["noise_sd"]["values"][0] *= 2.0
    tampered_noise = json.dumps(
        noise_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="noise_sd_sha256"):
        ConditionalAllocationMixture.from_json(
            tampered_noise,
            expected_sha256=hashlib.sha256(tampered_noise.encode()).hexdigest(),
        )

    basis_payload = json.loads(serialized)
    basis_values = basis_payload["summary_basis"]["values"]
    for index in range(0, len(basis_values), artifact.summary_rank):
        basis_values[index] *= -1.0
    tampered_basis = json.dumps(
        basis_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="summary_basis_sha256"):
        ConditionalAllocationMixture.from_json(
            tampered_basis,
            expected_sha256=hashlib.sha256(tampered_basis.encode()).hexdigest(),
        )

    external_payload = json.loads(serialized)
    external_payload["design_sha256"] = "0" * 64
    tampered_external_identity = json.dumps(
        external_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ConditionalAllocationMixture.from_json(
            tampered_external_identity,
            expected_sha256=artifact.sha256,
        )


def test_likelihood_only_path_survives_unused_gradient_overflow() -> None:
    """Finite logp must not depend on representability of an unused gradient."""
    baseline = _artifact(sample_count=3, rank=0)
    extreme = ConditionalAllocationMixture(
        baseline.projected_unit_mass_residual_factors,
        np.full_like(baseline.observation_mean_design, 1.0e308),
        baseline.noise_sd,
        baseline.summary_basis,
        baseline.labels,
        baseline.cell_ids,
        baseline.alpha_totals,
        baseline.source_seed,
        "finite likelihood with unrepresentable derivative",
        baseline.cell_alphas_sha256,
        baseline.design_sha256,
        baseline.noise_sd_sha256,
        baseline.summary_basis_sha256,
    )
    observation = np.full(extreme.observation_count, 10.0)
    masses = np.zeros(extreme.region_count)

    assert math.isfinite(extreme.log_likelihood(observation, masses))
    with pytest.raises(ValueError, match="gradient"):
        extreme.log_likelihood_and_mass_gradient(observation, masses)


def test_construction_and_evaluation_reject_invalid_inputs() -> None:
    """Malformed scientific arrays and seeds should fail before evaluation."""
    model = _model()
    labels = np.asarray([0, 0, 1, 1])
    with pytest.raises(TypeError, match="integer array"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels.astype(float),
            sample_count=10,
            source_seed=1,
            source_provenance="invalid labels",
        )
    with pytest.raises(TypeError, match="sample_count"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=True,
            source_seed=1,
            source_provenance="invalid count",
        )
    with pytest.raises(ValueError, match="source_seed"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=-1,
            source_provenance="invalid seed",
        )
    with pytest.raises(ValueError, match="unique"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=1,
            source_provenance="invalid cell ids",
            cell_ids=np.zeros(4, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="signed int64"):
        ConditionalAllocationMixture.from_aggregation(
            model,
            labels,
            sample_count=10,
            source_seed=1,
            source_provenance="out-of-range cell ids",
            cell_ids=np.asarray([0, 1, 2, 2**63], dtype=np.uint64),
        )

    artifact = _artifact(sample_count=20)
    with pytest.raises(ValueError, match="observation"):
        artifact.log_likelihood([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="masses"):
        artifact.log_likelihood(np.ones(3), [1.0, -2.0])
    with pytest.raises(ValueError, match="mean_offset"):
        artifact.log_likelihood(
            np.ones(3),
            [1.0, 2.0],
            mean_offset=np.ones(2),
        )
