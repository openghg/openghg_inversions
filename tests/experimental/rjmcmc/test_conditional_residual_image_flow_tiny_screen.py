"""Tests for the direct noisy-residual FlowJAX BP1 driver."""

from __future__ import annotations

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as flow_screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_flow import (
    conditional_residual_unit_covariances,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _source_case() -> tuple[
    AdditiveDirichletAggregation,
    c1.IntArray,
    ResidualImageContext,
    np.ndarray,
    float,
    float,
]:
    """Return the source-pinned near-Gaussian two-cell root case."""
    regime = c1._regime("near_gaussian")
    shapes, rate, design, observation, noise = c1._case_arrays(
        regime,
        "two_cell",
    )
    labels = c1.labels_for_tiling("two_cell", "root")
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size),
    )
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        np.arange(shapes.size, dtype=np.int64),
        source_provenance="flow-screen unit-test context",
    )
    masses, _ = c1._mass_grid(
        shapes=shapes,
        rate=rate,
        family="two_cell",
        tiling="root",
        total_order=8,
        fraction_order=6,
    )
    log_totals = np.log(masses.sum(axis=1))
    return (
        aggregation,
        labels,
        context,
        conditional_residual_unit_covariances(aggregation, context),
        float(log_totals.min() - flow_screen.LOG_TOTAL_MARGIN),
        float(log_totals.max() + flow_screen.LOG_TOTAL_MARGIN),
    )


def test_public_simulation_domains_replay_and_are_domain_separated() -> None:
    """Domain keys must replay exactly without sharing simulated arrays."""
    aggregation, labels, context, covariances, log_minimum, log_maximum = (
        _source_case()
    )
    arguments = {
        "case_id": "near_gaussian__two_cell__root",
        "sample_count": 1_024,
        "base_seed": 731,
        "log_total_minimum": log_minimum,
        "log_total_maximum": log_maximum,
    }
    first = flow_screen._simulated_domain(
        aggregation,
        labels,
        context,
        covariances,
        domain=flow_screen.TRAINING_DOMAIN,
        **arguments,
    )
    replay = flow_screen._simulated_domain(
        aggregation,
        labels,
        context,
        covariances,
        domain=flow_screen.TRAINING_DOMAIN,
        **arguments,
    )
    independent = flow_screen._simulated_domain(
        aggregation,
        labels,
        context,
        covariances,
        domain=flow_screen.VALIDATION_DOMAIN,
        **arguments,
    )

    np.testing.assert_array_equal(first.targets, replay.targets)
    np.testing.assert_array_equal(first.conditions, replay.conditions)
    assert first.evidence == replay.evidence
    assert first.targets.shape == (1_024, context.residual_rank)
    assert first.conditions.shape == (1_024, context.region_count)
    assert np.max(np.abs(first.conditions)) <= 1.0
    assert first.evidence["targets_sha256"] != independent.evidence["targets_sha256"]
    assert (
        first.evidence["conditions_sha256"]
        != independent.evidence["conditions_sha256"]
    )


def test_protected_or_unknown_domains_cannot_be_derived() -> None:
    """No development code path may derive a protected-domain seed."""
    with pytest.raises(ValueError, match="protected or unknown"):
        flow_screen._domain_seed(
            731,
            case_id="near_gaussian__two_cell__root",
            domain="protected-holdout",
            stream="anything",
        )


def test_protocol_identity_is_complete_and_source_inputs_are_strict() -> None:
    """The protocol must hash canonically and source identities fail closed."""
    assert len(flow_screen._protocol_sha256()) == 64
    assert (
        flow_screen._protocol_sha256()
        == flow_screen.DEVELOPMENT_PROTOCOL_SHA256
    )
    assert flow_screen._protocol_payload()["protected_holdout_catalogue_sha256"] == (
        flow_screen.PROTECTED_HOLDOUT_CATALOGUE_SHA256
    )
    with pytest.raises(ValueError, match="full lower-case Git SHA"):
        flow_screen.run_case(
            regime_name="near_gaussian",
            family="two_cell",
            training_sample_count=4_096,
            base_seed=731,
            profile="smoke",
            source_git_revision="short",
            driver_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        flow_screen.run_case(
            regime_name="near_gaussian",
            family="two_cell",
            training_sample_count=4_096,
            base_seed=731,
            profile="smoke",
            source_git_revision="0" * 40,
            driver_sha256="short",
        )
