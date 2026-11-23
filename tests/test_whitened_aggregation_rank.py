"""Tests for choosing the LRPD aggregation-error rank from the whitened spectrum."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import eigh

from openghg_inversions.observation_error import (
    WhitenedRankRule,
    choose_whitened_aggregation_rank,
    select_whitened_rank,
    whitened_aggregation_spectrum,
    whitened_eigenvalue_kl,
)
from openghg_inversions.rhime.co2 import Co2PreparedInputs, prepare_co2_inputs

from test_co2_preparation import _canonical_inputs, _reduction


def _smooth_case(n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth non-diagonal A, two interleaved sites, hours and errors."""
    rng = np.random.default_rng(7)
    hours = np.repeat(np.arange(n // 2, dtype=float), 2)
    sites = np.tile(["MHD", "TAC"], n // 2)
    position = np.arange(n, dtype=float)
    kernel = np.exp(-np.abs(position[:, None] - position[None, :]) / 4.0)
    loading = rng.normal(size=(n, 3))
    covariance = 2.0 * kernel + loading @ loading.T
    error = rng.uniform(0.1, 0.3, size=n)
    return covariance, sites, hours, error, loading


def _ou_reference(
    sites: np.ndarray, hours: np.ndarray, error: np.ndarray, amplitude: dict[str, float], tau: float
) -> np.ndarray:
    """Build the dense reference covariance R0 directly."""
    same_site = sites[:, None] == sites[None, :]
    sd = np.asarray([amplitude[site] for site in sites])
    correlation = np.where(same_site, np.exp(-np.abs(hours[:, None] - hours[None, :]) / tau), 0.0)
    return np.diag(error**2) + sd[:, None] * correlation * sd[None, :]


def _gaussian_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Return KL(N(0, p) || N(0, q)) from the textbook trace/log-det formula."""
    n = p.shape[0]
    _, logdet_p = np.linalg.slogdet(p)
    _, logdet_q = np.linalg.slogdet(q)
    return 0.5 * float(np.trace(np.linalg.solve(q, p)) - n + logdet_q - logdet_p)


def test_dropped_mode_kl_matches_dense_gaussian_kl_for_ou_reference() -> None:
    """The spectral dropped-mode KL equals the dense KL of the whitened truncation."""
    covariance, sites, hours, error, _ = _smooth_case()
    amplitude = {"MHD": 0.8, "TAC": 1.3}
    reference = _ou_reference(sites, hours, error, amplitude, tau=5.0)

    mu = whitened_aggregation_spectrum(
        covariance,
        observation_error_sd=error,
        reference_sd=np.asarray([amplitude[site] for site in sites]),
        site_index=(sites == "TAC").astype(int),
        observation_hours=hours,
        tau_hours=np.full(sites.size, 5.0),
    )
    values, vectors = eigh(covariance, reference)
    np.testing.assert_allclose(mu, values[::-1], rtol=1e-9, atol=1e-12)

    k = 4
    leading = vectors[:, ::-1][:, :k]
    truncated = reference @ leading @ np.diag(values[::-1][:k]) @ leading.T @ reference
    expected = _gaussian_kl(reference + covariance, reference + truncated)
    rank, record = select_whitened_rank(mu, WhitenedRankRule(rank=k, mu_threshold=None, kl_budget_per_observation=1.0))

    assert rank == k
    np.testing.assert_allclose(record["dropped_mode_kl"], expected, rtol=1e-8)
    np.testing.assert_allclose(whitened_eigenvalue_kl(mu[k:]), expected, rtol=1e-8)


def test_independent_reference_spectrum_matches_scaled_covariance() -> None:
    """Without OU the reference is diagonal and whitening is a row/column scaling."""
    covariance, _, _, error, _ = _smooth_case()
    reference_sd = np.full(error.size, 1.0)

    mu = whitened_aggregation_spectrum(covariance, observation_error_sd=error, reference_sd=reference_sd)

    expected = eigh(covariance, np.diag(error**2 + 1.0), eigvals_only=True)[::-1]
    np.testing.assert_allclose(mu, expected, rtol=1e-9, atol=1e-12)


def test_rank_selection_on_synthetic_spectrum() -> None:
    """Count, KL-budget and combined criteria select the documented ranks."""
    mu = np.asarray([5.0, 1.0, 0.5, 0.2, 0.11, 0.1, 0.05, 0.01])
    terms = 0.5 * (mu - np.log1p(mu))

    count_rank, count_record = select_whitened_rank(mu, WhitenedRankRule())
    assert count_rank == 5
    assert count_record["count_mu_above_threshold"] == 5
    np.testing.assert_allclose(count_record["dropped_mode_kl"], terms[5:].sum())
    assert count_record["first_dropped_mu"] == 0.1

    budget = (terms[3:].sum() + 1e-12) / mu.size
    kl_rank, kl_record = select_whitened_rank(
        mu, WhitenedRankRule(mu_threshold=None, kl_budget_per_observation=budget)
    )
    assert kl_rank == 3
    assert kl_record["kl_budget_rank"] == 3
    assert kl_record["dropped_mode_kl_per_observation"] <= budget

    both_rank, _ = select_whitened_rank(mu, WhitenedRankRule(kl_budget_per_observation=budget))
    assert both_rank == 5

    none_rank, none_record = select_whitened_rank(np.asarray([0.05, 0.0]), WhitenedRankRule())
    assert none_rank == 1
    assert none_record["count_mu_above_threshold"] == 0


def test_explicit_rank_below_rule_rank_warns_and_is_kept() -> None:
    """A benchmarking override is honoured, with a warning only below the rule rank."""
    mu = np.asarray([5.0, 1.0, 0.5, 0.2, 0.01])

    with pytest.warns(UserWarning, match="below the whitened-spectrum rule rank 4"):
        rank, record = select_whitened_rank(mu, WhitenedRankRule(rank=2))
    assert (rank, record["rule_rank"], record["override_rank"]) == (2, 4, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, _ = select_whitened_rank(mu, WhitenedRankRule(rank=50))
    assert rank == mu.size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_sd": 0.0},
        {"reference_sd": {"MHD": -1.0}},
        {"tau_hours": 0.0},
        {"mu_threshold": None, "kl_budget_per_observation": None},
        {"rank": 0},
        {"units": ""},
    ],
)
def test_rule_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    """Zero amplitudes, empty criteria and invalid ranks are rejected."""
    with pytest.raises(ValueError, match="WhitenedRankRule"):
        WhitenedRankRule(**kwargs)  # type: ignore[arg-type]


def test_reference_site_mapping_and_units_are_resolved_per_row() -> None:
    """Site mappings expand by site label and reference units convert to observation units."""
    covariance, sites, hours, error, _ = _smooth_case()
    ppm = WhitenedRankRule(reference_sd={"MHD": 0.8, "TAC": 1.3}, tau_hours=5.0)
    ppb = WhitenedRankRule(reference_sd={"MHD": 800.0, "TAC": 1300.0}, units="ppb", tau_hours=5.0)

    rank_ppm, record_ppm = choose_whitened_aggregation_rank(
        covariance, ppm, observation_error_sd=error, sites=sites, observation_times=hours
    )
    rank_ppb, record_ppb = choose_whitened_aggregation_rank(
        covariance, ppb, observation_error_sd=error, sites=sites, observation_times=hours,
        reference_unit_scale=1e-3,
    )

    assert rank_ppm == rank_ppb
    np.testing.assert_allclose(record_ppm["dropped_mode_kl"], record_ppb["dropped_mode_kl"])
    assert record_ppb["reference_covariance"]["reference_sd_observation_units_by_site"] == pytest.approx(
        {"MHD": 0.8, "TAC": 1.3}
    )
    with pytest.raises(ValueError, match="missing observation site"):
        choose_whitened_aggregation_rank(
            covariance, WhitenedRankRule(reference_sd={"MHD": 1.0}), observation_error_sd=error, sites=sites
        )


def test_default_preparation_records_rank_selection_through_serialization() -> None:
    """The default CO2 preparation chooses the rank from the rule and records it."""
    canonical = _canonical_inputs()
    reduction = _reduction(canonical)
    rule = WhitenedRankRule(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)

    prepared = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=rule)
    restored = Co2PreparedInputs.from_datatree(prepared.to_datatree())

    selection = restored.provenance["aggregation_error_rank_selection"]
    assert restored.provenance["aggregation_error_rank"] == "whitened_spectrum"
    assert selection["chosen_rank"] == prepared.inv_inputs.sizes["agg_rank"]
    assert selection["chosen_rank"] == restored.provenance["aggregation_error"]["requested_rank"]
    assert selection["count_mu_above_threshold"] is not None
    assert selection["dropped_mode_kl"] >= 0.0
    reference = selection["reference_covariance"]
    assert reference["reference_sd"] == {"MHD": 1.5, "TAC": 2.0}
    assert reference["tau_hours"] == 24.0
    assert reference["correlation"] == "within-site OU"
    assert reference["observation_units"] == "ppm"


def test_explicit_preparation_rank_below_count_warns() -> None:
    """An integer rank is an override checked against the default reference."""
    canonical = _canonical_inputs()

    with pytest.warns(UserWarning, match="below the whitened-spectrum rule rank"):
        prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=1)

    selection = prepared.provenance["aggregation_error_rank_selection"]
    assert prepared.provenance["aggregation_error_rank"] == 1
    assert (selection["override_rank"], selection["chosen_rank"]) == (1, 1)
    assert selection["rule_rank"] == 3
    assert prepared.inv_inputs.sizes["agg_rank"] == 1
