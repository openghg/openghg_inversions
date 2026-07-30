"""Tests for observation-blind score-regularized tiny simulator domains."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from examples.rjmcmc import score_regularized_flow_tiny_domains as domains
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_regularized_flow import (
    component_observation_score,
    component_partial_log_mass_score,
    standardize_simulator_draw,
)


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_all_six_tiny_cases_have_complete_finite_score_domains(case_id: str) -> None:
    result = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=64,
        base_seed=731,
    )
    sample_count = result.total_mass.size
    rank = result.spectrum.retained_rank
    assert sample_count == 64
    assert rank >= 1
    assert result.root_total_uniform.shape == (sample_count,)
    assert result.gaussian_noise_uniform.shape == (sample_count, rank)
    assert result.total_mass.shape == (sample_count,)
    assert result.raw_log_mass.shape == (sample_count,)
    assert result.allocation_residual.shape == (sample_count, rank)
    assert result.gaussian_noise.shape == (sample_count, rank)
    assert result.standardized_draw.shape == (sample_count, rank)
    assert result.mass_score_target.shape == (sample_count,)
    assert result.observation_score_target.shape == (sample_count, rank)
    assert result.spectrum.discarded_variance <= result.spectrum.eigenvalue_tolerance * max(
        1, result.spectrum.noise_sd.size
    )
    for values in (
        result.root_total_uniform,
        result.gaussian_noise_uniform,
        result.total_mass,
        result.raw_log_mass,
        result.allocation_residual,
        result.gaussian_noise,
        result.standardized_draw,
        result.mass_score_target,
        result.observation_score_target,
    ):
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert not values.flags.writeable
    assert np.all((result.root_total_uniform > 0.0) & (result.root_total_uniform < 1.0))
    assert np.all(
        (result.gaussian_noise_uniform > 0.0)
        & (result.gaussian_noise_uniform < 1.0)
    )
    np.testing.assert_array_equal(result.raw_log_mass, np.log(result.total_mass))
    np.testing.assert_array_equal(
        result.standardized_draw,
        standardize_simulator_draw(
            result.total_mass,
            result.spectrum.eigenvalues,
            result.allocation_residual,
            result.gaussian_noise,
        ),
    )
    np.testing.assert_array_equal(
        result.mass_score_target,
        component_partial_log_mass_score(
            result.total_mass,
            result.spectrum.eigenvalues,
            result.allocation_residual,
            result.gaussian_noise,
            result.standardized_draw,
        ),
    )
    np.testing.assert_array_equal(
        result.observation_score_target,
        component_observation_score(
            result.total_mass,
            result.spectrum.eigenvalues,
            result.gaussian_noise,
        ),
    )
    assert result.T is result.total_mass
    assert result.raw_tau is result.raw_log_mass
    assert result.xi is result.allocation_residual
    assert result.epsilon is result.gaussian_noise
    assert result.x is result.standardized_draw
    result.verify()


def test_stream_seed_follows_the_frozen_byte_contract() -> None:
    case_id = domains.CASE_IDS[0]
    seed = domains.domain_stream_seed(
        731,
        case_id=case_id,
        domain=domains.TRAINING_DOMAIN,
        stream_name=domains.ROOT_TOTAL_STREAM,
    )
    digest = hashlib.sha256(domains.PROTOCOL.encode("ascii"))
    digest.update((731).to_bytes(8, byteorder="little", signed=False))
    digest.update(case_id.encode("ascii"))
    digest.update(domains.TRAINING_DOMAIN.encode("ascii"))
    digest.update(domains.ROOT_TOTAL_STREAM.encode("ascii"))
    assert seed == int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)
    assert 0 <= seed < 2**64


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_nested_sample_sizes_are_exact_prefixes_for_every_returned_array(
    case_id: str,
) -> None:
    small = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=731,
    )
    replay = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=731,
    )
    large = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=64,
        base_seed=731,
    )
    for name in (
        "root_total_uniform",
        "gaussian_noise_uniform",
        "total_mass",
        "raw_log_mass",
        "allocation_residual",
        "gaussian_noise",
        "standardized_draw",
        "mass_score_target",
        "observation_score_target",
    ):
        small_values = getattr(small, name)
        np.testing.assert_array_equal(small_values, getattr(replay, name))
        np.testing.assert_array_equal(small_values, getattr(large, name)[:16])
    assert small.evidence.sha256 == replay.evidence.sha256
    assert small.evidence.sha256 != large.evidence.sha256


def test_public_domains_and_base_seeds_are_replayable_and_disjoint() -> None:
    case_id = domains.CASE_IDS[0]
    catalogues = {
        domain: domains.simulate_tiny_score_domain(
            case_id,
            domain=domain,
            sample_count=16,
            base_seed=731,
        )
        for domain in domains.PUBLIC_DOMAINS
    }
    replay = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=731,
    )
    np.testing.assert_array_equal(
        replay.standardized_draw,
        catalogues[domains.TRAINING_DOMAIN].standardized_draw,
    )
    assert len({result.evidence.sha256 for result in catalogues.values()}) == 3
    assert len(
        {
            result.evidence.stream_seeds
            for result in catalogues.values()
        }
    ) == 3
    catalogue_values = tuple(catalogues.values())
    for first, second in zip(catalogue_values, catalogue_values[1:], strict=False):
        for name in (
            "root_total_uniform",
            "allocation_residual",
            "gaussian_noise_uniform",
        ):
            assert not np.array_equal(getattr(first, name), getattr(second, name))
    other_seed = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=1_877,
    )
    assert other_seed.evidence.stream_seeds != replay.evidence.stream_seeds
    for name in (
        "root_total_uniform",
        "allocation_residual",
        "gaussian_noise_uniform",
    ):
        assert not np.array_equal(getattr(other_seed, name), getattr(replay, name))


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_iid_pcg64_sources_recover_analytic_moments_roughly(case_id: str) -> None:
    result = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.MODEL_SELECTION_VALIDATION_DOMAIN,
        sample_count=4_096,
        base_seed=731,
    )
    shapes, rate, _, _, _ = aggregation_error_tiny_oracle.tiny_root_case(
        case_id
    ).arrays()
    gamma_shape = float(np.sum(shapes))
    assert float(np.mean(result.total_mass)) == pytest.approx(
        gamma_shape / rate,
        rel=8.0e-2,
        abs=8.0e-3,
    )
    assert float(np.var(result.total_mass)) == pytest.approx(
        gamma_shape / rate**2,
        rel=1.5e-1,
        abs=8.0e-3,
    )
    standardized_allocation_mean = np.mean(
        result.allocation_residual,
        axis=0,
    ) / np.sqrt(result.spectrum.eigenvalues)
    np.testing.assert_allclose(
        standardized_allocation_mean,
        0.0,
        rtol=0.0,
        atol=5.0e-2,
    )
    allocation_covariance = np.cov(
        result.allocation_residual,
        rowvar=False,
        bias=True,
    )
    allocation_covariance = np.atleast_2d(allocation_covariance)
    np.testing.assert_allclose(
        np.diag(allocation_covariance),
        result.spectrum.eigenvalues,
        rtol=1.5e-1,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.mean(result.gaussian_noise, axis=0),
        0.0,
        rtol=0.0,
        atol=5.0e-2,
    )
    np.testing.assert_allclose(
        np.var(result.gaussian_noise, axis=0),
        1.0,
        rtol=0.0,
        atol=8.0e-2,
    )
    center, scale = result.conditioning
    assert center == pytest.approx(
        float(np.asarray(np.log(result.total_mass)).mean()),
        abs=8.0e-2,
    )
    assert scale**2 == pytest.approx(
        float(np.asarray(np.log(result.total_mass)).var()),
        rel=1.5e-1,
        abs=2.0e-2,
    )


def _midpoint_empirical_uniform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return ranks


def _legendre_audit(left: np.ndarray, right: np.ndarray) -> float:
    left_one = math.sqrt(3.0) * (2.0 * left - 1.0)
    right_one = math.sqrt(3.0) * (2.0 * right - 1.0)
    left_two = math.sqrt(5.0) * (6.0 * np.square(left) - 6.0 * left + 1.0)
    right_two = math.sqrt(5.0) * (6.0 * np.square(right) - 6.0 * right + 1.0)
    return max(
        abs(float(np.mean(left_order * right_order)))
        for left_order in (left_one, left_two)
        for right_order in (right_one, right_two)
    )


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
@pytest.mark.parametrize("domain", domains.PUBLIC_DOMAINS)
def test_every_public_case_domain_has_no_detectable_cross_block_copula(
    case_id: str,
    domain: str,
) -> None:
    result = domains.simulate_tiny_score_domain(
        case_id,
        domain=domain,
        sample_count=8_192,
        base_seed=731,
    )
    blocks = {
        "total": (
            result.total_mass,
            result.root_total_uniform,
        ),
    }
    blocks.update(
        {
            f"allocation_{index}": (
                result.allocation_residual[:, index],
                _midpoint_empirical_uniform(result.allocation_residual[:, index]),
            )
            for index in range(result.allocation_residual.shape[1])
        }
    )
    blocks.update(
        {
            f"noise_{index}": (
                result.gaussian_noise[:, index],
                result.gaussian_noise_uniform[:, index],
            )
            for index in range(result.gaussian_noise.shape[1])
        }
    )
    pairs: list[tuple[str, str]] = []
    pairs.extend(
        ("total", name)
        for name in blocks
        if name.startswith(("allocation_", "noise_"))
    )
    pairs.extend(
        (allocation, noise)
        for allocation in blocks
        if allocation.startswith("allocation_")
        for noise in blocks
        if noise.startswith("noise_")
    )
    pairs.extend(
        (left, right)
        for index, left in enumerate(blocks)
        if left.startswith("noise_")
        for right in tuple(blocks)[index + 1 :]
        if right.startswith("noise_")
    )
    for left_name, right_name in pairs:
        left_raw, left_uniform = blocks[left_name]
        right_raw, right_uniform = blocks[right_name]
        correlation = float(np.corrcoef(left_raw, right_raw)[0, 1])
        quadrant = float(
            np.mean((left_uniform < 0.5) & (right_uniform < 0.5))
        )
        assert abs(correlation) <= 0.10, (left_name, right_name, correlation)
        assert abs(quadrant - 0.25) <= 0.04, (
            left_name,
            right_name,
            quadrant,
        )
        nonlinear_cross_moment = _legendre_audit(
            left_uniform,
            right_uniform,
        )
        assert nonlinear_cross_moment <= 0.10, (
            left_name,
            right_name,
            nonlinear_cross_moment,
        )
    total_phi = math.sqrt(3.0) * (2.0 * blocks["total"][1] - 1.0)
    for allocation_name in blocks:
        if not allocation_name.startswith("allocation_"):
            continue
        allocation_phi = math.sqrt(3.0) * (
            2.0 * blocks[allocation_name][1] - 1.0
        )
        for noise_name in blocks:
            if not noise_name.startswith("noise_"):
                continue
            noise_phi = math.sqrt(3.0) * (
                2.0 * blocks[noise_name][1] - 1.0
            )
            three_way = abs(
                float(np.mean(total_phi * allocation_phi * noise_phi))
            )
            assert three_way <= 0.10, (
                allocation_name,
                noise_name,
                three_way,
            )


def test_all_public_case_domain_stream_seeds_are_unique() -> None:
    seeds = [
        domains.domain_stream_seed(
            731,
            case_id=case_id,
            domain=domain,
            stream_name=stream,
        )
        for case_id in domains.CASE_IDS
        for domain in domains.PUBLIC_DOMAINS
        for stream in domains.SIMULATOR_STREAMS
    ]
    assert len(seeds) == 54
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_cell_and_observation_permutations_preserve_scientific_arrays(
    case_id: str,
) -> None:
    shapes, _, design, _, _ = aggregation_error_tiny_oracle.tiny_root_case(
        case_id
    ).arrays()
    cell_permutation = np.arange(shapes.size - 1, -1, -1, dtype=np.int64)
    observation_permutation = np.arange(design.shape[0] - 1, -1, -1, dtype=np.int64)
    canonical = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.DEVELOPMENT_REPORTING_TEST_DOMAIN,
        sample_count=64,
        base_seed=731,
    )
    permuted = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.DEVELOPMENT_REPORTING_TEST_DOMAIN,
        sample_count=64,
        base_seed=731,
        cell_permutation=cell_permutation,
        observation_permutation=observation_permutation,
    )
    np.testing.assert_allclose(
        permuted.spectrum.eigenvalues,
        canonical.spectrum.eigenvalues,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    np.testing.assert_array_equal(permuted.total_mass, canonical.total_mass)
    np.testing.assert_array_equal(permuted.raw_log_mass, canonical.raw_log_mass)
    np.testing.assert_array_equal(
        permuted.root_total_uniform,
        canonical.root_total_uniform,
    )
    np.testing.assert_array_equal(
        permuted.gaussian_noise_uniform,
        canonical.gaussian_noise_uniform,
    )
    np.testing.assert_array_equal(permuted.gaussian_noise, canonical.gaussian_noise)
    for name in (
        "allocation_residual",
        "standardized_draw",
        "mass_score_target",
        "observation_score_target",
    ):
        np.testing.assert_allclose(
            getattr(permuted, name),
            getattr(canonical, name),
            rtol=2.0e-12,
            atol=6.0e-14,
        )
    assert permuted.evidence.stream_seeds == canonical.evidence.stream_seeds
    assert permuted.evidence.scientific_input_sha256 != (
        canonical.evidence.scientific_input_sha256
    )


@pytest.mark.parametrize(
    ("sample_count", "error"),
    [
        (0, ValueError),
        (3, ValueError),
        (12, ValueError),
        (True, TypeError),
    ],
)
def test_non_power_of_two_sample_counts_are_rejected(
    sample_count: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="power of two|integer"):
        domains.simulate_tiny_score_domain(
            domains.CASE_IDS[0],
            domain=domains.TRAINING_DOMAIN,
            sample_count=sample_count,
            base_seed=731,
        )


@pytest.mark.parametrize(
    "domain",
    [
        "protected",
        "protected-holdout",
        "internal-validation",
        "",
    ],
)
def test_protected_and_unknown_domains_are_rejected_before_seed_derivation(
    domain: str,
) -> None:
    with pytest.raises(ValueError, match="protected or unknown"):
        domains.domain_stream_seed(
            731,
            case_id=domains.CASE_IDS[0],
            domain=domain,
            stream_name=domains.ROOT_TOTAL_STREAM,
        )
    with pytest.raises(ValueError, match="protected or unknown"):
        domains.simulate_tiny_score_domain(
            domains.CASE_IDS[0],
            domain=domain,
            sample_count=16,
            base_seed=731,
        )


def test_invalid_cases_streams_seeds_and_permutations_fail_closed() -> None:
    with pytest.raises(ValueError, match="six frozen"):
        domains.simulate_tiny_score_domain(
            "equal_footprint__two_cell__root",
            domain=domains.TRAINING_DOMAIN,
            sample_count=16,
            base_seed=731,
        )
    with pytest.raises(ValueError, match="unknown simulator stream"):
        domains.domain_stream_seed(
            731,
            case_id=domains.CASE_IDS[0],
            domain=domains.TRAINING_DOMAIN,
            stream_name="optimizer-0",
        )
    with pytest.raises(ValueError, match=r"\[0, 2\*\*64\)"):
        domains.simulate_tiny_score_domain(
            domains.CASE_IDS[0],
            domain=domains.TRAINING_DOMAIN,
            sample_count=16,
            base_seed=-1,
        )
    with pytest.raises(ValueError, match="cell_permutation"):
        domains.simulate_tiny_score_domain(
            domains.CASE_IDS[0],
            domain=domains.TRAINING_DOMAIN,
            sample_count=16,
            base_seed=731,
            cell_permutation=[0, 0],
        )


def test_conditioning_is_raw_log_mass_not_empirical_and_hashes_are_strict() -> None:
    result = domains.simulate_tiny_score_domain(
        domains.CASE_IDS[0],
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=731,
    )
    assert math.isfinite(result.evidence.conditioning_center)
    assert result.evidence.conditioning_scale > 0.0
    assert result.hashes["raw_log_mass"] == dict(result.evidence.array_sha256)[
        "raw_log_mass"
    ]
    assert result.evidence.construction_method == "keyed_pcg64_dirichlet"
    assert result.evidence.numpy_version == np.__version__
    result.verify()
    with pytest.raises(ValueError, match="read-only"):
        result.total_mass[0] = 1.0
