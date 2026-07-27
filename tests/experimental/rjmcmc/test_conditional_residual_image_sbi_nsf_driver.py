"""Protocol and simulator tests for the BP1 sbi-NSF driver."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special

from examples.rjmcmc import conditional_residual_image_sbi_nsf_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_sbi_nsf import (
    conditional_residual_unit_covariances,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _fixture() -> tuple[
    AdditiveDirichletAggregation,
    ResidualImageContext,
    np.ndarray,
]:
    """Return one root aggregation fixture."""
    aggregation = AdditiveDirichletAggregation(
        np.asarray([1.2, 2.1], dtype=np.float64),
        np.asarray(
            [
                [1.0, 0.2],
                [0.4, 1.3],
            ],
            dtype=np.float64,
        ),
        np.asarray([0.3, 0.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )
    labels = np.zeros(2, dtype=np.int64)
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        np.arange(2, dtype=np.int64),
        source_provenance="sbi-NSF driver unit fixture",
    )
    return aggregation, context, labels


def test_protocol_has_no_protected_action_and_uses_ancestral_gamma() -> None:
    """The declaration must keep G3 closed and identify the joint simulator."""
    payload = screen._protocol_payload()
    assert payload["protected"]["protected_action_authorized"] is False
    assert payload["simulator"]["total_mass"] == "exact root Gamma inverse CDF"
    assert screen.PROTECTED_HOLDOUT_CATALOGUE_ID not in screen.PUBLIC_DOMAINS
    assert len(screen._protocol_sha256()) == 64


def test_unknown_or_protected_domain_fails_before_seed_derivation() -> None:
    """No unlisted domain can reach the sample generator."""
    with pytest.raises(ValueError, match="protected or unknown"):
        screen._domain_seed(
            731,
            case_id="near_gaussian__two_cell__root",
            domain=screen.PROTECTED_HOLDOUT_CATALOGUE_ID,
            stream="attempted-protected-read",
        )


def test_prior_predictive_domain_reconstructs_gamma_log_moments() -> None:
    """Ancestral total draws must follow the declared root pushforward prior."""
    aggregation, context, labels = _fixture()
    total_shape = float(np.sum(aggregation.cell_alphas))
    rate = 2.7
    center = np.asarray(
        [special.digamma(total_shape) - math.log(rate)],
        dtype=np.float64,
    )
    scale = np.asarray(
        [math.sqrt(float(special.polygamma(1, total_shape)))],
        dtype=np.float64,
    )
    domain = screen._simulated_domain(
        aggregation,
        labels,
        context,
        conditional_residual_unit_covariances(aggregation, context),
        case_id="unit__two_cell__root",
        domain=screen.TRAINING_DOMAIN,
        sample_count=4_096,
        base_seed=731,
        total_shape=total_shape,
        rate=rate,
        conditioner_center=center,
        conditioner_scale=scale,
    )
    reconstructed_log_totals = (
        domain.conditions[:, 0] * scale[0] + center[0]
    )
    assert float(np.mean(reconstructed_log_totals)) == pytest.approx(
        center[0],
        abs=5.0e-4,
    )
    assert float(np.std(reconstructed_log_totals)) == pytest.approx(
        scale[0],
        abs=8.0e-4,
    )
    assert domain.evidence["domain"] == screen.TRAINING_DOMAIN
    assert domain.targets.shape == (4_096, context.residual_rank)
