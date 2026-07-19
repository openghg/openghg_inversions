"""Tests for exact Gamma--Beta aggregate calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.calibration import (
    aggregate_prior_moments,
    calibrate_group_root_variance,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
)


def _two_region_samples(*, root_variance: float = 0.0):
    """Return a balanced one-split Gamma--Beta prior for calibration tests."""
    group = GammaBetaGroupSpec(
        "target",
        np.ones((1, 2), dtype=bool),
        root_variance=root_variance,
        max_depth=1,
    )
    forest = GammaBetaForest.from_groups(
        np.ones((1, 2)),
        [group],
        require_full_coverage=True,
    )
    return forest.sample(
        2,
        kappa_strategy=DepthKappaStrategy(base_kappa=4.0, depth_multiplier=1.0),
        rng=1,
    )


def test_aggregate_moments_use_terminal_covariance_exactly() -> None:
    """A full-group total has only root uncertainty after conservation."""
    samples = _two_region_samples(root_variance=0.25)

    moments = aggregate_prior_moments(samples, np.ones((1, 2)))

    assert moments.expected_total == pytest.approx(2.0)
    assert moments.variance == pytest.approx(1.0)
    assert moments.relative_standard_deviation == pytest.approx(0.5)
    np.testing.assert_allclose(moments.terminal_weights, np.ones(2))
    assert not moments.terminal_weights.flags.writeable


def test_aggregate_moments_reject_mass_outside_forest_support() -> None:
    """Aggregate mass cannot silently disappear outside declared groups."""
    group = GammaBetaGroupSpec("partial", np.array([[True, False]]), max_depth=0)
    forest = GammaBetaForest.from_groups(np.ones((1, 2)), [group])
    samples = forest.sample(
        2,
        kappa_strategy=DepthKappaStrategy(base_kappa=2.0),
        rng=1,
    )

    with pytest.raises(ValueError, match="zero outside the forest support"):
        aggregate_prior_moments(samples, np.ones((1, 2)))


def test_root_variance_calibration_hits_feasible_target_exactly() -> None:
    """Affine root calibration reaches a requested aggregate relative SD."""
    samples = _two_region_samples()
    included_mass = np.array([[1.0, 0.0]])

    result = calibrate_group_root_variance(
        samples,
        included_mass,
        group_name="target",
        target_relative_standard_deviation=0.75,
    )

    assert result.feasible
    assert result.minimum_relative_standard_deviation == pytest.approx(np.sqrt(0.2))
    assert result.calibrated_root_variance is not None
    assert result.achieved_relative_standard_deviation == pytest.approx(0.75)


def test_root_variance_calibration_reports_contrast_floor() -> None:
    """A root variance cannot reduce uncertainty already created by a split."""
    samples = _two_region_samples()

    result = calibrate_group_root_variance(
        samples,
        np.array([[1.0, 0.0]]),
        group_name="target",
        target_relative_standard_deviation=0.2,
    )

    assert not result.feasible
    assert result.calibrated_root_variance is None
    assert result.achieved_relative_standard_deviation == pytest.approx(np.sqrt(0.2))


@pytest.mark.parametrize("invalid_target", [0.0, -1.0, np.nan, np.inf])
def test_root_variance_calibration_rejects_invalid_target(invalid_target: float) -> None:
    """Aggregate calibration requires a positive finite relative SD."""
    with pytest.raises(ValueError, match="must be finite and positive"):
        calibrate_group_root_variance(
            _two_region_samples(),
            np.ones((1, 2)),
            group_name="target",
            target_relative_standard_deviation=invalid_target,
        )
