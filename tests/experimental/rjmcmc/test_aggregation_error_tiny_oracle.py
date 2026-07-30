"""Tests for support-aware corrected tiny root-NLE references."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_tiny_oracle as oracle,
)


def test_all_frozen_cases_have_finite_conditional_root_likelihoods() -> None:
    totals = np.asarray((0.1, 0.5, 1.0, 2.0), dtype=np.float64)
    for case_id in oracle.CASE_IDS:
        values = np.asarray(
            oracle.root_conditional_log_likelihood(
                case_id,
                totals,
                fraction_order=4,
            ),
            dtype=np.float64,
        )
        assert values.shape == totals.shape
        assert np.all(np.isfinite(values))


def test_boundary_adaptive_log_total_reference_matches_frozen_values() -> None:
    summary = oracle.adaptive_log_total_summary(
        oracle.BOUNDARY_CASE_ID,
        fraction_order=32,
    )
    assert summary.log_evidence == pytest.approx(
        -1.7949049875906,
        abs=2.0e-10,
    )
    assert summary.posterior_mean_total == pytest.approx(
        0.90254144350,
        abs=2.0e-10,
    )
    assert summary.posterior_sd_total == pytest.approx(
        0.06508269572,
        abs=2.0e-10,
    )
    assert summary.posterior_lower_0_025 == pytest.approx(
        0.78466927377,
        abs=2.0e-10,
    )
    assert summary.posterior_median == pytest.approx(
        0.89901102296,
        abs=2.0e-10,
    )
    assert summary.posterior_upper_0_975 == pytest.approx(
        1.04346682078,
        abs=2.0e-10,
    )
    assert summary.represented_prior_mass >= 1.0 - 3.0e-15
    assert summary.represented_posterior_mass >= 1.0 - 1.0e-10
    assert summary.mode_included
    summary.verify()


def test_independent_native_log_mass_tail_ladder_converges() -> None:
    summaries = [
        oracle.native_log_mass_summary(
            lower_log_mass=lower,
            epsabs=2.0e-7,
            epsrel=2.0e-7,
        )
        for lower in (-40.0, -80.0, -120.0)
    ]
    assert summaries[0].log_evidence == pytest.approx(
        -1.79602794090,
        abs=2.0e-8,
    )
    assert summaries[-1].log_evidence == pytest.approx(
        -1.79490498818,
        abs=2.0e-8,
    )
    assert abs(summaries[-1].log_evidence + 1.79490498759) <= 2.5e-8
    assert summaries[-1].posterior_mean_total == pytest.approx(
        0.90254144350,
        abs=2.0e-8,
    )
    assert summaries[-1].posterior_sd_total == pytest.approx(
        0.06508269572,
        abs=2.0e-8,
    )
    for summary in summaries:
        summary.verify()


def test_support_audit_rejects_a_mode_excluding_negligible_subset() -> None:
    log_prior = np.log(np.asarray((0.4999998, 1.0e-7, 0.5000000, 1.0e-7)))
    log_likelihood = np.asarray((-40.0, -40.0, 0.0, -40.0))
    checkerboard = np.asarray((False, True, False, True))
    audit = oracle.audit_evaluation_support(
        log_prior,
        log_likelihood,
        checkerboard,
    )
    assert audit.retained_prior_mass == pytest.approx(2.0e-7)
    assert audit.retained_posterior_mass < 1.0e-12
    assert not audit.posterior_mode_included
    assert not audit.posterior_weighted_metric_valid
    assert not audit.conditional_renormalization_permitted


def test_full_support_is_valid_and_reports_no_omission() -> None:
    log_prior = np.log(np.asarray((0.2, 0.3, 0.5)))
    log_likelihood = np.asarray((-1.0, 2.0, 0.5))
    audit = oracle.audit_evaluation_support(
        log_prior,
        log_likelihood,
        np.ones(3, dtype=np.bool_),
    )
    assert audit.retained_prior_mass == pytest.approx(1.0)
    assert audit.retained_posterior_mass == pytest.approx(1.0)
    assert audit.posterior_mode_included
    assert audit.posterior_weighted_metric_valid
