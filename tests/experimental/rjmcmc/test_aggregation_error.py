"""Tests for the exact two-cell aggregation-error quadrature oracle."""

from __future__ import annotations

from math import exp, gamma, log, pi

import numpy as np
import pytest
from scipy.integrate import quad

import openghg_inversions.experimental.rjmcmc.aggregation_error as aggregation_error
from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
    beta_quadrature,
    gamma_quadrature,
    log_posterior_partition_probabilities,
    posterior_partition_probabilities,
)


def _four_cell_case(
    *,
    total_order: int = 20,
    fraction_order: int = 14,
    chunk_size: int = 257,
) -> tuple[FourCellAggregationOracle, np.ndarray, np.ndarray, np.ndarray]:
    """Return a heterogeneous deterministic 2x2 oracle test case."""
    oracle = FourCellAggregationOracle(
        [0.7, 1.1, 1.6, 0.9],
        1.4,
        total_order=total_order,
        fraction_order=fraction_order,
        chunk_size=chunk_size,
    )
    observation = np.array([0.7, -0.2, 0.4])
    design = np.array(
        [
            [1.8, -0.5, 0.3, 0.9],
            [0.2, 1.4, -0.7, 0.1],
            [0.5, -0.2, 1.1, 0.8],
        ]
    )
    noise_sd = np.array([0.35, 0.8, 0.6])
    return oracle, observation, design, noise_sd


def test_probability_rules_are_normalized_and_recover_distribution_moments() -> None:
    """Endpoint-aware rules should integrate low-order Beta and Gamma moments."""
    first_shape = 0.25
    second_shape = 0.6
    beta_rule = beta_quadrature(first_shape, second_shape, order=4)
    beta_total = first_shape + second_shape

    gamma_shape = 0.4
    gamma_rate = 1.7
    gamma_rule = gamma_quadrature(gamma_shape, gamma_rate, order=4)

    assert beta_rule.weights.sum() == pytest.approx(1.0, abs=2.0e-14)
    assert np.dot(beta_rule.weights, beta_rule.nodes) == pytest.approx(
        first_shape / beta_total,
        abs=2.0e-14,
    )
    assert np.dot(beta_rule.weights, np.square(beta_rule.nodes)) == pytest.approx(
        first_shape * (first_shape + 1.0) / (beta_total * (beta_total + 1.0)),
        abs=2.0e-14,
    )
    assert gamma_rule.weights.sum() == pytest.approx(1.0, abs=2.0e-14)
    assert np.dot(gamma_rule.weights, gamma_rule.nodes) == pytest.approx(
        gamma_shape / gamma_rate,
        abs=2.0e-14,
    )
    assert np.dot(gamma_rule.weights, np.square(gamma_rule.nodes)) == pytest.approx(
        gamma_shape * (gamma_shape + 1.0) / gamma_rate**2,
        abs=2.0e-14,
    )


def test_coarse_conditional_likelihood_is_normalized_in_observation() -> None:
    """The finite hidden-fraction mixture should integrate to one in data space."""
    oracle = TwoCellAggregationOracle(
        gamma_shape=2.3,
        gamma_rate=1.4,
        beta_first_shape=0.35,
        beta_second_shape=0.65,
        fraction_order=48,
    )

    integral, _ = quad(
        lambda observation: oracle.coarse_conditional_likelihood(
            1.7,
            observation,
            np.array([1.8, -0.5]),
            0.35,
        ),
        -np.inf,
        np.inf,
        epsabs=1.0e-10,
        limit=200,
    )

    assert integral == pytest.approx(1.0, abs=2.0e-8)


def test_log_evidence_remains_finite_for_1382_observations() -> None:
    """Log mixtures should remain usable when every linear likelihood underflows."""
    observation_count = 1_382
    observation = np.zeros(observation_count)
    design = np.zeros((observation_count, 2))
    noise_sd = np.ones(observation_count)
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=12, fraction_order=10)
    expected = -0.5 * observation_count * log(2.0 * pi)

    conditional = oracle.coarse_conditional_log_likelihood(
        1.7,
        observation,
        design,
        noise_sd,
    )
    coarse = oracle.coarse_log_evidence(observation, design, noise_sd)
    fine = oracle.fine_log_evidence(observation, design, noise_sd)
    structural_prior = np.array([0.83, 0.17])
    log_posterior = log_posterior_partition_probabilities(
        np.log(structural_prior),
        np.array([coarse, fine]),
    )

    assert conditional == pytest.approx(expected, abs=5.0e-13)
    assert coarse == pytest.approx(expected, abs=5.0e-13)
    assert fine == pytest.approx(expected, abs=5.0e-13)
    np.testing.assert_allclose(np.exp(log_posterior), structural_prior, atol=2.0e-14)
    assert oracle.coarse_evidence(observation, design, noise_sd) == 0.0
    assert oracle.fine_evidence(observation, design, noise_sd) == 0.0


def test_evidence_matches_adaptive_nested_quadrature_and_posterior_normalizes() -> None:
    """Independent normalized densities should reproduce evidence and posterior mass."""
    shape = 2.3
    rate = 1.4
    first_shape = 2.5
    second_shape = 4.0
    observation = 0.7
    design = np.array([1.8, -0.5])
    noise_sd = 0.35
    oracle = TwoCellAggregationOracle(
        shape,
        rate,
        first_shape,
        second_shape,
        total_order=80,
        fraction_order=80,
    )
    gamma_normalizer = rate**shape / gamma(shape)
    beta_normalizer = gamma(first_shape + second_shape) / (gamma(first_shape) * gamma(second_shape))
    gaussian_normalizer = 1.0 / (noise_sd * (2.0 * pi) ** 0.5)

    def conditional_reference(total: float) -> float:
        """Adaptively integrate the explicitly normalized hidden Beta density."""

        def beta_integrand(fraction: float) -> float:
            """Return normalized Beta density times one normalized Gaussian."""
            mean = total * (design[0] * fraction + design[1] * (1.0 - fraction))
            beta_density = (
                beta_normalizer * fraction ** (first_shape - 1.0) * (1.0 - fraction) ** (second_shape - 1.0)
            )
            gaussian_density = gaussian_normalizer * exp(-0.5 * ((observation - mean) / noise_sd) ** 2)
            return beta_density * gaussian_density

        return quad(beta_integrand, 0.0, 1.0, epsabs=2.0e-11, epsrel=2.0e-11)[0]

    def evidence_integrand(total: float) -> float:
        """Return normalized Gamma density times adaptive conditional likelihood."""
        gamma_density = gamma_normalizer * total ** (shape - 1.0) * exp(-rate * total)
        return gamma_density * conditional_reference(total)

    reference = quad(
        evidence_integrand,
        0.0,
        np.inf,
        epsabs=2.0e-9,
        epsrel=2.0e-9,
        limit=200,
    )[0]
    posterior_mass = quad(
        lambda total: oracle.total_posterior_density(
            total,
            observation,
            design,
            noise_sd,
        ),
        0.0,
        np.inf,
        epsabs=2.0e-8,
        epsrel=2.0e-8,
        limit=200,
    )[0]

    assert np.exp(oracle.coarse_log_evidence(observation, design, noise_sd)) == pytest.approx(
        reference,
        rel=2.0e-7,
        abs=2.0e-9,
    )
    assert np.exp(oracle.fine_log_evidence(observation, design, noise_sd)) == pytest.approx(
        reference,
        rel=2.0e-7,
        abs=2.0e-9,
    )
    assert posterior_mass == pytest.approx(1.0, rel=2.0e-7, abs=2.0e-8)


def test_product_quadrature_converges_for_endpoint_singular_beta_prior() -> None:
    """Each rule order should independently converge to a high-order evidence."""
    parameters = {
        "gamma_shape": 2.3,
        "gamma_rate": 1.4,
        "beta_first_shape": 0.7,
        "beta_second_shape": 2.1,
    }
    reference = TwoCellAggregationOracle(**parameters, total_order=120, fraction_order=120)
    reference_evidence = reference.fine_evidence(0.7, [1.8, -0.5], 0.35)
    orders = {
        "low_total": (12, 120),
        "high_total": (48, 120),
        "low_fraction": (120, 12),
        "high_fraction": (120, 48),
    }
    errors = {
        name: abs(
            TwoCellAggregationOracle(
                **parameters,
                total_order=total_order,
                fraction_order=fraction_order,
            ).fine_evidence(0.7, [1.8, -0.5], 0.35)
            - reference_evidence
        )
        for name, (total_order, fraction_order) in orders.items()
    }

    assert errors["high_total"] < errors["low_total"] / 500.0
    assert errors["high_fraction"] < errors["low_fraction"] / 100_000.0
    assert errors["high_total"] < 1.0e-6
    assert errors["high_fraction"] < 1.0e-8


def test_coarse_and_fine_evidence_obey_the_tower_identity_for_vector_data() -> None:
    """Marginalizing W before or after T should give the same vector-data evidence."""
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=48, fraction_order=44)
    observation = np.array([0.7, -0.2])
    design = np.array([[1.8, -0.5], [0.2, 1.4]])
    noise_sd = np.array([0.35, 0.8])

    coarse = oracle.coarse_evidence(observation, design, noise_sd)
    fine = oracle.fine_evidence(observation, design, noise_sd)

    assert coarse == pytest.approx(fine, rel=2.0e-15, abs=1.0e-15)


def test_common_total_posterior_agrees_between_representations() -> None:
    """Coarse and fine paths should induce the same posterior masses for T."""
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=48, fraction_order=44)

    coarse = oracle.total_posterior_quadrature(
        0.7,
        [1.8, -0.5],
        0.35,
        representation="coarse",
    )
    fine = oracle.total_posterior_quadrature(
        0.7,
        [1.8, -0.5],
        0.35,
        representation="fine",
    )

    np.testing.assert_array_equal(coarse.nodes, fine.nodes)
    np.testing.assert_allclose(coarse.weights, fine.weights, rtol=2.0e-15, atol=2.0e-16)
    assert coarse.weights.sum() == pytest.approx(1.0, abs=2.0e-14)


def test_equal_exact_evidences_preserve_a_nonuniform_structural_prior() -> None:
    """Data should not update partition weights in the exact-representation limit."""
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=48, fraction_order=44)
    evidences = np.array(
        [
            oracle.coarse_evidence(0.7, [1.8, -0.5], 0.35),
            oracle.fine_evidence(0.7, [1.8, -0.5], 0.35),
        ]
    )
    structural_prior = np.array([0.83, 0.17])

    posterior = posterior_partition_probabilities(structural_prior, evidences)

    np.testing.assert_allclose(posterior, structural_prior, rtol=2.0e-15, atol=2.0e-16)


def test_evidence_is_the_power_one_normalized_gaussian_without_structural_mass() -> None:
    """A zero design should reduce pure evidence to one known vector Gaussian."""
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=32, fraction_order=28)
    observation = np.array([0.7, -0.2])
    noise_sd = np.array([0.35, 0.8])
    design = np.zeros((2, 2))
    expected = np.exp(-0.5 * np.square(observation / noise_sd).sum()) / (
        np.prod(noise_sd) * (2.0 * np.pi) ** (observation.size / 2.0)
    )

    coarse = oracle.coarse_evidence(observation, design, noise_sd)
    fine = oracle.fine_evidence(observation, design, noise_sd)

    assert coarse == pytest.approx(expected, rel=3.0e-14, abs=1.0e-15)
    assert fine == pytest.approx(expected, rel=3.0e-14, abs=1.0e-15)
    assert coarse != pytest.approx(expected**2, rel=1.0e-3)


def test_partition_combiner_applies_unequal_evidence_once() -> None:
    """Posterior partition weights should normalize prior times one evidence."""
    structural_prior = np.array([0.8, 0.2])
    evidences = np.array([0.25, 2.0])

    posterior = posterior_partition_probabilities(structural_prior, evidences)

    np.testing.assert_allclose(posterior, np.array([1.0 / 3.0, 2.0 / 3.0]), atol=1.0e-15)


def test_total_posterior_density_rejects_nonpositive_support_for_small_shape() -> None:
    """Posterior density should reject the Gamma boundary when shape is below one."""
    oracle = TwoCellAggregationOracle(0.4, 1.7, 2.5, 4.0)

    with pytest.raises(ValueError, match="strictly positive"):
        oracle.total_posterior_density(0.0, 0.7, [1.8, -0.5], 0.35)
    with pytest.raises(ValueError, match="strictly positive"):
        oracle.total_posterior_density(-1.0, 0.7, [1.8, -0.5], 0.35)

    assert np.isfinite(oracle.total_posterior_density(1.0e-12, 0.7, [1.8, -0.5], 0.35))


def test_quadrature_reports_nonfinite_scipy_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distribution-specific errors should identify non-finite SciPy rules."""
    monkeypatch.setattr(
        aggregation_error,
        "roots_jacobi",
        lambda *_: (np.array([np.nan]), np.array([np.nan])),
    )

    with pytest.raises(ValueError, match="SciPy Gauss--Jacobi"):
        beta_quadrature(2.5, 4.0, order=1)

    monkeypatch.setattr(
        aggregation_error,
        "roots_genlaguerre",
        lambda *_: (np.array([np.nan]), np.array([np.nan])),
    )

    with pytest.raises(ValueError, match="SciPy generalized Gauss--Laguerre"):
        gamma_quadrature(2.3, 1.4, order=1)


def test_identical_footprints_remove_fraction_dependence() -> None:
    """Identical design columns should make exact and nominal fill agree."""
    observation = np.array([0.7, -0.2])
    identical_footprints = np.array([[1.2, 1.2], [-0.4, -0.4]])
    noise_sd = np.array([0.35, 0.8])
    first = TwoCellAggregationOracle(2.3, 1.4, 0.7, 2.1, total_order=52, fraction_order=40)
    second = TwoCellAggregationOracle(2.3, 1.4, 30.0, 2.0, total_order=52, fraction_order=40)

    first_exact = first.fine_evidence(observation, identical_footprints, noise_sd)
    second_exact = second.fine_evidence(observation, identical_footprints, noise_sd)

    assert first_exact == pytest.approx(
        first.nominal_fill_evidence(
            observation,
            identical_footprints,
            noise_sd,
        ),
        rel=3.0e-14,
        abs=1.0e-15,
    )
    assert second_exact == pytest.approx(
        second.nominal_fill_evidence(
            observation,
            identical_footprints,
            noise_sd,
        ),
        rel=3.0e-14,
        abs=1.0e-15,
    )
    assert first_exact == pytest.approx(second_exact, rel=3.0e-14, abs=1.0e-15)


def test_high_beta_concentration_recovers_nominal_fill_limit() -> None:
    """A concentrated hidden fraction should approach deterministic prior-mean fill."""
    low = TwoCellAggregationOracle(
        2.3,
        1.4,
        3.0,
        7.0,
        total_order=100,
        fraction_order=48,
    )
    high = TwoCellAggregationOracle(
        2.3,
        1.4,
        30_000.0,
        70_000.0,
        total_order=100,
        fraction_order=48,
    )
    low_error = abs(
        low.fine_evidence(0.7, [2.0, 0.0], 0.35) - low.nominal_fill_evidence(0.7, [2.0, 0.0], 0.35)
    )
    high_error = abs(
        high.fine_evidence(0.7, [2.0, 0.0], 0.35) - high.nominal_fill_evidence(0.7, [2.0, 0.0], 0.35)
    )

    assert high_error < 1.0e-5
    assert high_error < low_error / 1_000.0


def test_nominal_fill_sentinel_has_unequal_evidence_for_contrasting_footprints() -> None:
    """Deterministic fill should expose aggregation error when cell designs differ."""
    oracle = TwoCellAggregationOracle(2.3, 1.4, 2.5, 4.0, total_order=80, fraction_order=80)

    coarse = oracle.coarse_evidence(0.2, [2.0, 0.0], 0.1)
    fine = oracle.fine_evidence(0.2, [2.0, 0.0], 0.1)
    nominal = oracle.nominal_fill_evidence(0.2, [2.0, 0.0], 0.1)
    structural_prior = np.array([0.35, 0.65])
    log_posterior = log_posterior_partition_probabilities(
        np.log(structural_prior),
        np.array(
            [
                oracle.coarse_log_evidence(0.2, [2.0, 0.0], 0.1),
                oracle.nominal_fill_log_evidence(0.2, [2.0, 0.0], 0.1),
            ]
        ),
    )

    assert coarse == pytest.approx(fine, rel=2.0e-15, abs=1.0e-15)
    assert abs(nominal - fine) > 0.1
    assert np.exp(log_posterior[0]) > structural_prior[0]


def test_four_cell_projection_frontiers_are_history_independent() -> None:
    """The 2x2 frontiers should be explicit projections, not tree histories."""
    oracle, _, _, _ = _four_cell_case(total_order=4, fraction_order=4)

    np.testing.assert_array_equal(oracle.projection("root"), [[1.0, 1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(
        oracle.projection("row"),
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
    )
    np.testing.assert_array_equal(
        oracle.projection("column"),
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(oracle.projection("fine"), np.eye(4))
    assert not oracle.projection("row").flags.writeable
    with pytest.raises(ValueError, match="tiling"):
        oracle.projection("diagonal")  # type: ignore[arg-type]


def test_four_cell_conditional_towers_hold_in_each_beta_chart() -> None:
    """Root-to-pair-to-native marginalization should obey both towers."""
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=8,
        fraction_order=14,
        chunk_size=97,
    )
    total = 1.7
    a0, a1, a2, a3 = oracle.native_shapes

    row_aggregate = beta_quadrature(a0 + a1, a2 + a3, oracle.fraction_order)
    row_masses = np.stack(
        (total * row_aggregate.nodes, total * (1.0 - row_aggregate.nodes)),
        axis=-1,
    )
    row_conditional = oracle.conditional_log_likelihood(
        row_masses,
        observation,
        design,
        noise_sd,
        tiling="row",
    )
    row_tower = float(aggregation_error._stable_logsumexp(np.log(row_aggregate.weights) + row_conditional))
    row_root = oracle.conditional_log_likelihood(
        total,
        observation,
        design,
        noise_sd,
        tiling="root",
        root_chart="row-first",
    )

    column_aggregate = beta_quadrature(a0 + a2, a1 + a3, oracle.fraction_order)
    column_masses = np.stack(
        (
            total * column_aggregate.nodes,
            total * (1.0 - column_aggregate.nodes),
        ),
        axis=-1,
    )
    column_conditional = oracle.conditional_log_likelihood(
        column_masses,
        observation,
        design,
        noise_sd,
        tiling="column",
    )
    column_tower = float(
        aggregation_error._stable_logsumexp(np.log(column_aggregate.weights) + column_conditional)
    )
    column_root = oracle.conditional_log_likelihood(
        total,
        observation,
        design,
        noise_sd,
        tiling="root",
        root_chart="column-first",
    )

    assert row_tower == pytest.approx(row_root, abs=2.0e-15)
    assert column_tower == pytest.approx(column_root, abs=2.0e-15)


def test_four_cell_root_conditional_is_normalized_in_observation() -> None:
    """The finite Dirichlet Gaussian mixture should integrate to one in data."""
    oracle = FourCellAggregationOracle(
        [0.7, 1.1, 1.6, 0.9],
        1.4,
        total_order=4,
        fraction_order=6,
        chunk_size=47,
    )

    integral, _ = quad(
        lambda observation: np.exp(
            oracle.conditional_log_likelihood(
                1.7,
                observation,
                [1.8, -0.5, 0.3, 0.9],
                0.35,
                tiling="root",
            )
        ),
        -np.inf,
        np.inf,
        epsabs=2.0e-9,
        limit=200,
    )

    assert integral == pytest.approx(1.0, abs=2.0e-8)


def test_four_cell_pair_conditional_is_native_likelihood_mixture() -> None:
    """A row conditional should equal explicit normalized native integration."""
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=6,
        fraction_order=12,
        chunk_size=31,
    )
    row_masses = np.array([0.8, 1.3])
    a0, a1, a2, a3 = oracle.native_shapes
    first = beta_quadrature(a0, a1, oracle.fraction_order)
    second = beta_quadrature(a2, a3, oracle.fraction_order)
    first_grid, second_grid = np.meshgrid(first.nodes, second.nodes, indexing="ij")
    native = np.stack(
        (
            row_masses[0] * first_grid,
            row_masses[0] * (1.0 - first_grid),
            row_masses[1] * second_grid,
            row_masses[1] * (1.0 - second_grid),
        ),
        axis=-1,
    )
    native_log_likelihood = oracle.conditional_log_likelihood(
        native,
        observation,
        design,
        noise_sd,
        tiling="fine",
    )
    expected = float(
        aggregation_error._stable_logsumexp(
            native_log_likelihood
            + np.log(first.weights)[:, np.newaxis]
            + np.log(second.weights)[np.newaxis, :]
        )
    )

    actual = oracle.conditional_log_likelihood(
        row_masses,
        observation,
        design,
        noise_sd,
        tiling="row",
    )

    assert actual == pytest.approx(expected, abs=2.0e-15)


def test_four_cell_evidence_and_total_posterior_agree_across_frontiers() -> None:
    """All exact projections should recover common evidence and total posterior."""
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=22,
        fraction_order=18,
        chunk_size=401,
    )
    configurations = (
        ("root", "row-first"),
        ("root", "column-first"),
        ("row", "row-first"),
        ("column", "row-first"),
        ("fine", "row-first"),
        ("fine", "column-first"),
    )
    evidences = np.array(
        [
            oracle.log_evidence(
                observation,
                design,
                noise_sd,
                tiling=tiling,
                root_chart=chart,
            )
            for tiling, chart in configurations
        ]
    )
    posterior_rules = [
        oracle.total_posterior_quadrature(
            observation,
            design,
            noise_sd,
            tiling=tiling,
            root_chart=chart,
        )
        for tiling, chart in configurations
    ]

    np.testing.assert_allclose(evidences, evidences[0], rtol=0.0, atol=8.0e-7)
    for rule in posterior_rules:
        np.testing.assert_array_equal(rule.nodes, posterior_rules[0].nodes)
        np.testing.assert_allclose(
            rule.weights,
            posterior_rules[0].weights,
            rtol=0.0,
            atol=2.0e-7,
        )


def test_four_cell_chart_discrepancy_converges_with_quadrature_order() -> None:
    """Row-first and column-first chart residuals should converge to zero."""
    _, observation, design, noise_sd = _four_cell_case()
    discrepancies = []
    for order in (8, 12, 18):
        oracle, _, _, _ = _four_cell_case(
            total_order=22,
            fraction_order=order,
            chunk_size=503,
        )
        discrepancies.append(
            abs(
                oracle.root_log_evidence(
                    observation,
                    design,
                    noise_sd,
                    chart="row-first",
                )
                - oracle.root_log_evidence(
                    observation,
                    design,
                    noise_sd,
                    chart="column-first",
                )
            )
        )

    assert discrepancies[2] < discrepancies[1] < discrepancies[0]
    assert discrepancies[2] < 8.0e-7


def test_four_cell_exact_evidence_preserves_nonuniform_structural_prior() -> None:
    """Informative data should preserve prior weights over four exact tilings."""
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=30,
        fraction_order=30,
        chunk_size=1_009,
    )
    log_evidences = np.array(
        [
            oracle.root_log_evidence(observation, design, noise_sd),
            oracle.row_log_evidence(observation, design, noise_sd),
            oracle.column_log_evidence(observation, design, noise_sd),
            oracle.fine_log_evidence(observation, design, noise_sd),
        ]
    )
    prior = np.array([0.41, 0.19, 0.11, 0.29])

    posterior = np.exp(log_posterior_partition_probabilities(np.log(prior), log_evidences))

    assert np.ptp(log_evidences) < 7.0e-13
    np.testing.assert_allclose(posterior, prior, rtol=0.0, atol=2.0e-13)


def test_four_cell_matches_independent_native_gamma_reference() -> None:
    """All projections should match factorized adaptive native-Gamma integrals."""
    native_shapes = np.array([0.7, 1.1, 1.6, 0.9])
    gamma_rate = 1.4
    observation = np.array([0.3, 0.8, -0.1, 1.2])
    diagonal = np.array([1.2, 0.7, -0.4, 1.5])
    design = np.diag(diagonal)
    noise_sd = np.array([0.3, 0.5, 0.8, 0.4])
    component_evidences = []
    component_means = []
    component_variances = []
    for shape, observed, coefficient, scale in zip(
        native_shapes,
        observation,
        diagonal,
        noise_sd,
        strict=True,
    ):
        normalizer = gamma_rate**shape / (gamma(shape) * scale * np.sqrt(2.0 * pi))

        def integrand(mass: float) -> float:
            """Return one normalized Gamma-prior Gaussian likelihood product."""
            return (
                normalizer
                * mass ** (shape - 1.0)
                * exp(-gamma_rate * mass - 0.5 * ((observed - coefficient * mass) / scale) ** 2)
            )

        evidence = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=300,
        )[0]
        first_moment = (
            quad(
                lambda mass: mass * integrand(mass),
                0.0,
                np.inf,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=300,
            )[0]
            / evidence
        )
        second_moment = (
            quad(
                lambda mass: mass * mass * integrand(mass),
                0.0,
                np.inf,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=300,
            )[0]
            / evidence
        )
        component_evidences.append(evidence)
        component_means.append(first_moment)
        component_variances.append(second_moment - first_moment**2)

    reference_log_evidence = float(np.log(component_evidences).sum())
    reference_total_mean = float(np.sum(component_means))
    reference_total_variance = float(np.sum(component_variances))
    oracle = FourCellAggregationOracle(
        native_shapes,
        gamma_rate,
        total_order=64,
        fraction_order=26,
        chunk_size=3_001,
    )
    evidence_routes = [
        oracle.root_log_evidence(
            observation,
            design,
            noise_sd,
            chart="row-first",
        ),
        oracle.root_log_evidence(
            observation,
            design,
            noise_sd,
            chart="column-first",
        ),
        *[
            oracle.log_evidence(
                observation,
                design,
                noise_sd,
                tiling=tiling,
            )
            for tiling in ("root", "row", "column", "fine")
        ],
    ]

    np.testing.assert_allclose(
        evidence_routes,
        reference_log_evidence,
        rtol=0.0,
        atol=7.0e-10,
    )
    for tiling in ("root", "row", "column", "fine"):
        posterior = oracle.total_posterior_quadrature(
            observation,
            design,
            noise_sd,
            tiling=tiling,
        )
        total_mean = float(posterior.weights @ posterior.nodes)
        total_variance = float(posterior.weights @ np.square(posterior.nodes - total_mean))
        assert total_mean == pytest.approx(reference_total_mean, abs=7.0e-9)
        assert total_variance == pytest.approx(
            reference_total_variance,
            abs=2.0e-8,
        )


def test_four_cell_nominal_fill_exposes_aggregation_error() -> None:
    """Deterministic native fill should differ under heterogeneous footprints."""
    oracle = FourCellAggregationOracle(
        [0.7, 1.1, 1.6, 0.9],
        1.4,
        total_order=42,
        fraction_order=22,
        chunk_size=809,
    )
    observation = 0.2
    design = np.array([2.0, 0.0, -0.5, 1.2])
    noise_sd = 0.1

    exact = oracle.fine_log_evidence(observation, design, noise_sd)
    root_nominal = oracle.nominal_fill_log_evidence(
        observation,
        design,
        noise_sd,
        tiling="root",
    )
    row_nominal = oracle.nominal_fill_log_evidence(
        observation,
        design,
        noise_sd,
        tiling="row",
    )

    assert abs(root_nominal - exact) > 0.1
    assert abs(row_nominal - exact) > 0.01
    assert oracle.nominal_fill_log_evidence(
        observation,
        design,
        noise_sd,
        tiling="fine",
    ) == pytest.approx(exact, abs=2.0e-15)


def test_four_cell_identical_columns_remove_allocation_dependence() -> None:
    """Identical native footprints should make exact and nominal routes agree."""
    oracle, observation, _, noise_sd = _four_cell_case(
        total_order=22,
        fraction_order=12,
        chunk_size=113,
    )
    common_column = np.array([1.2, -0.4, 0.6])
    design = np.repeat(common_column[:, np.newaxis], 4, axis=1)
    exact = oracle.root_log_evidence(
        observation,
        design,
        noise_sd,
        chart="column-first",
    )

    for tiling in ("root", "row", "column", "fine"):
        assert oracle.log_evidence(
            observation,
            design,
            noise_sd,
            tiling=tiling,
        ) == pytest.approx(exact, abs=2.0e-13)
        assert oracle.nominal_fill_log_evidence(
            observation,
            design,
            noise_sd,
            tiling=tiling,
        ) == pytest.approx(exact, abs=2.0e-13)


def test_four_cell_log_evidence_is_stable_for_1382_observations() -> None:
    """Sufficient-statistic log evaluation should survive linear underflow."""
    observation_count = 1_382
    observation = np.zeros(observation_count)
    design = np.zeros((observation_count, 4))
    noise_sd = np.ones(observation_count)
    oracle = FourCellAggregationOracle(
        [0.7, 1.1, 1.6, 0.9],
        1.4,
        total_order=8,
        fraction_order=7,
        chunk_size=53,
    )
    expected = -0.5 * observation_count * log(2.0 * pi)

    for tiling in ("root", "row", "column", "fine"):
        log_evidence = oracle.log_evidence(
            observation,
            design,
            noise_sd,
            tiling=tiling,
        )
        assert log_evidence == pytest.approx(expected, abs=2.0e-12)
        assert (
            oracle.evidence(
                observation,
                design,
                noise_sd,
                tiling=tiling,
            )
            == 0.0
        )


def test_four_cell_sufficient_statistics_recover_exact_fit_normalizer() -> None:
    """Direct fallback should preserve normalization under huge cancellation."""
    oracle = FourCellAggregationOracle(
        [1.0, 1.0, 1.0, 1.0],
        1.0,
        total_order=2,
        fraction_order=2,
    )

    log_likelihood = oracle.conditional_log_likelihood(
        [1.0, 0.0, 0.0, 0.0],
        1.0e150,
        [1.0e150, 0.0, 0.0, 0.0],
        1.0,
        tiling="fine",
    )

    assert log_likelihood == pytest.approx(-0.5 * log(2.0 * pi), abs=1.0e-15)


def test_four_cell_sufficient_statistics_recover_large_signal_residual() -> None:
    """A finite near-fit should match direct residual arithmetic at large scale."""
    oracle = FourCellAggregationOracle(
        [1.0, 1.0, 1.0, 1.0],
        1.0,
        total_order=2,
        fraction_order=2,
    )
    observation = 1.0e10 + 6_897.0
    direct_residual = observation - 1.0e10
    expected = -0.5 * direct_residual**2 - 0.5 * log(2.0 * pi)

    log_likelihood = oracle.conditional_log_likelihood(
        [1.0, 0.0, 0.0, 0.0],
        observation,
        [1.0e10, 0.0, 0.0, 0.0],
        1.0,
        tiling="fine",
    )

    assert log_likelihood == expected


def test_four_cell_likelihood_batches_respect_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root integration should never evaluate more than the configured chunk."""
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=3,
        fraction_order=7,
        chunk_size=23,
    )
    observed_batch_sizes: list[int] = []
    original = aggregation_error._GaussianSufficientStatistics.log_likelihood

    def recording_log_likelihood(
        self: aggregation_error._GaussianSufficientStatistics,
        mass: np.ndarray,
    ) -> np.ndarray:
        """Record each quadrature likelihood batch before delegating."""
        observed_batch_sizes.append(mass.reshape(-1, 4).shape[0])
        return original(self, mass)

    monkeypatch.setattr(
        aggregation_error._GaussianSufficientStatistics,
        "log_likelihood",
        recording_log_likelihood,
    )

    oracle.root_log_evidence(observation, design, noise_sd)

    assert observed_batch_sizes
    assert max(observed_batch_sizes) <= oracle.chunk_size


def test_four_cell_rejects_invalid_shapes_and_frontier_masses() -> None:
    """Validation should reject Boolean shapes and wrong projected dimensions."""
    with pytest.raises(TypeError, match="Boolean"):
        FourCellAggregationOracle([1.0, True, 1.0, 1.0], 1.0)
    with pytest.raises(ValueError, match="four"):
        FourCellAggregationOracle([1.0, 1.0, 1.0], 1.0)
    oracle, observation, design, noise_sd = _four_cell_case(
        total_order=3,
        fraction_order=3,
    )
    with pytest.raises(ValueError, match="final axis length 2"):
        oracle.conditional_log_likelihood(
            [1.0, 2.0, 3.0],
            observation,
            design,
            noise_sd,
            tiling="row",
        )
    with pytest.raises(ValueError, match="chart"):
        oracle.root_log_evidence(
            observation,
            design,
            noise_sd,
            chart="diagonal",  # type: ignore[arg-type]
        )
