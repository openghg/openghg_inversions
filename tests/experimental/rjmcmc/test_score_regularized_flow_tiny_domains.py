"""Tests for observation-blind score-regularized tiny simulator domains."""

from __future__ import annotations

import hashlib
import math
from typing import cast

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import score_regularized_flow_tiny_domains as domains
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
    assert seed == int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)
    assert 0 <= seed < 2**32


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
    for first, second in zip(
        catalogues.values(),
        tuple(catalogues.values())[1:],
        strict=False,
    ):
        assert not np.array_equal(first.standardized_draw, second.standardized_draw)
    other_seed = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.TRAINING_DOMAIN,
        sample_count=16,
        base_seed=1_877,
    )
    assert other_seed.evidence.stream_seeds != replay.evidence.stream_seeds
    assert not np.array_equal(other_seed.standardized_draw, replay.standardized_draw)


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_scrambled_sobol_sources_recover_analytic_moments_roughly(case_id: str) -> None:
    result = domains.simulate_tiny_score_domain(
        case_id,
        domain=domains.MODEL_SELECTION_VALIDATION_DOMAIN,
        sample_count=4_096,
        base_seed=731,
    )
    regime_name, family_name, _ = case_id.split("__")
    regime = c1._regime(regime_name)
    shapes, rate, _, _, _ = c1._case_arrays(
        regime,
        cast(c1.Family, family_name),
    )
    gamma_shape = float(np.sum(shapes))
    assert float(np.mean(result.total_mass)) == pytest.approx(
        gamma_shape / rate,
        rel=2.0e-3,
        abs=2.0e-3,
    )
    assert float(np.var(result.total_mass)) == pytest.approx(
        gamma_shape / rate**2,
        rel=4.0e-2,
        abs=2.0e-3,
    )
    np.testing.assert_allclose(
        np.mean(result.allocation_residual, axis=0),
        0.0,
        rtol=0.0,
        atol=2.0e-3,
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
        rtol=4.0e-2,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        np.mean(result.gaussian_noise, axis=0),
        0.0,
        rtol=0.0,
        atol=3.0e-3,
    )
    np.testing.assert_allclose(
        np.var(result.gaussian_noise, axis=0),
        1.0,
        rtol=0.0,
        atol=8.0e-3,
    )
    center, scale = result.conditioning
    assert center == pytest.approx(
        float(np.asarray(np.log(result.total_mass)).mean()),
        abs=3.0e-3,
    )
    assert scale**2 == pytest.approx(
        float(np.asarray(np.log(result.total_mass)).var()),
        rel=2.0e-2,
        abs=2.0e-3,
    )


@pytest.mark.parametrize("case_id", domains.CASE_IDS)
def test_cell_and_observation_permutations_preserve_scientific_arrays(
    case_id: str,
) -> None:
    regime_name, family_name, _ = case_id.split("__")
    regime = c1._regime(regime_name)
    shapes, _, design, _, _ = c1._case_arrays(
        regime,
        cast(c1.Family, family_name),
    )
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
            rtol=3.0e-13,
            atol=3.0e-14,
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
    result.verify()
    with pytest.raises(ValueError, match="read-only"):
        result.total_mass[0] = 1.0
