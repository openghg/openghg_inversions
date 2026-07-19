"""Exact aggregate diagnostics and root-variance calibration for Gamma--Beta priors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from .gamma_beta import GammaBetaSamples


@dataclass(frozen=True, slots=True, eq=False)
class AggregatePriorMoments:
    """Exact moments of one additive aggregate of terminal regional scalings.

    Attributes:
        expected_total: Prior mean of the aggregate.
        variance: Exact prior variance of the aggregate.
        standard_deviation: Square root of ``variance``.
        relative_standard_deviation: Standard deviation divided by the prior
            mean.
        terminal_weights: Additive expected mass assigned to each terminal
            regional scaling, in ``forest.leaf_ids`` order.
    """

    expected_total: float
    variance: float
    standard_deviation: float
    relative_standard_deviation: float
    terminal_weights: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RootVarianceCalibration:
    """Exact one-group root-variance calibration for an additive aggregate.

    Attributes:
        group_name: Group root whose variance is calibrated.
        target_relative_standard_deviation: Requested aggregate relative SD.
        minimum_relative_standard_deviation: Aggregate relative SD when the
            selected group root is fixed at one. Split contrasts and all other
            groups remain active.
        calibrated_root_variance: Solved non-negative root variance, or
            ``None`` when the target lies below the contrast-only minimum.
        achieved_relative_standard_deviation: Exact relative SD under the
            solved variance, or the minimum when calibration is infeasible.
        feasible: Whether a non-negative root variance can reach the target.
    """

    group_name: str
    target_relative_standard_deviation: float
    minimum_relative_standard_deviation: float
    calibrated_root_variance: float | None
    achieved_relative_standard_deviation: float
    feasible: bool


def aggregate_prior_moments(
    samples: GammaBetaSamples,
    grid_expected_mass: npt.ArrayLike,
    *,
    root_variances: dict[str, float] | None = None,
) -> AggregatePriorMoments:
    """Return exact moments for a grid-weighted terminal-state aggregate.

    Args:
        samples: Compiled Gamma--Beta split concentrations and fixed topology.
            Finite Monte Carlo draws are not used by this calculation.
        grid_expected_mass: Non-negative additive expected mass included in the
            aggregate on the native grid. For a country total this is normally
            ``expected_mass * country_mask``.
        root_variances: Optional group root-variance overrides passed to the
            analytic terminal covariance calculation.

    Returns:
        Exact prior aggregate moments and terminal-state weights.

    Raises:
        ValueError: If weights are invalid, outside the forest support, or have
            zero total.
    """
    forest = samples.forest
    grid_weights = np.asarray(grid_expected_mass, dtype=np.float64)
    if grid_weights.shape != forest.shape:
        raise ValueError("grid_expected_mass must match the forest grid shape.")
    if not np.isfinite(grid_weights).all() or (grid_weights < 0.0).any():
        raise ValueError("grid_expected_mass must be finite and non-negative.")

    flattened = grid_weights.reshape(-1)
    supported = np.zeros(flattened.shape, dtype=bool)
    for node_id in forest.leaf_ids:
        supported[forest.nodes[node_id].flat_indices] = True
    if np.any(flattened[~supported] > 0.0):
        raise ValueError("grid_expected_mass must be zero outside the forest support.")
    terminal_weights = np.asarray(
        [float(flattened[forest.nodes[node_id].flat_indices].sum()) for node_id in forest.leaf_ids],
        dtype=np.float64,
    )
    expected_total = float(terminal_weights.sum())
    if expected_total <= 0.0:
        raise ValueError("grid_expected_mass must have positive mass inside the forest support.")

    covariance = samples.analytic_leaf_covariance(root_variances=root_variances)
    variance = float(terminal_weights @ covariance @ terminal_weights)
    numerical_tolerance = np.finfo(np.float64).eps * expected_total**2 * len(terminal_weights)
    if variance < -numerical_tolerance:
        raise ValueError("Analytic aggregate variance is unexpectedly negative.")
    variance = max(variance, 0.0)
    standard_deviation = math.sqrt(variance)
    terminal_weights.setflags(write=False)
    return AggregatePriorMoments(
        expected_total=expected_total,
        variance=variance,
        standard_deviation=standard_deviation,
        relative_standard_deviation=standard_deviation / expected_total,
        terminal_weights=terminal_weights,
    )


def calibrate_group_root_variance(
    samples: GammaBetaSamples,
    grid_expected_mass: npt.ArrayLike,
    *,
    group_name: str,
    target_relative_standard_deviation: float,
) -> RootVarianceCalibration:
    """Solve one group root variance for a target aggregate relative SD.

    For fixed topology and split concentrations, every within-group terminal
    second moment is linear in ``1 + root_variance``. The aggregate variance is
    therefore affine in the selected root variance, so two exact covariance
    evaluations determine the solution without numerical optimization.

    Args:
        samples: Compiled Gamma--Beta split concentrations and fixed topology.
        grid_expected_mass: Non-negative additive expected mass included in the
            target aggregate on the native grid.
        group_name: Existing hard-group name whose root variance is adjustable.
        target_relative_standard_deviation: Positive requested aggregate SD
            divided by its prior mean.

    Returns:
        Exact calibration result. A target below the contrast-only minimum is
        marked infeasible and has no calibrated root variance.

    Raises:
        ValueError: If the group or target is invalid, or aggregate variance is
            not increasing in the selected root variance.
    """
    if group_name not in {group.name for group in samples.forest.groups}:
        raise ValueError(f"Unknown Gamma-Beta group {group_name!r}.")
    target = float(target_relative_standard_deviation)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_relative_standard_deviation must be finite and positive.")

    at_zero = aggregate_prior_moments(
        samples,
        grid_expected_mass,
        root_variances={group_name: 0.0},
    )
    desired_variance = (target * at_zero.expected_total) ** 2
    tolerance = np.finfo(np.float64).eps * at_zero.expected_total**2 * 32.0
    if desired_variance < at_zero.variance - tolerance:
        return RootVarianceCalibration(
            group_name=group_name,
            target_relative_standard_deviation=target,
            minimum_relative_standard_deviation=at_zero.relative_standard_deviation,
            calibrated_root_variance=None,
            achieved_relative_standard_deviation=at_zero.relative_standard_deviation,
            feasible=False,
        )

    at_one = aggregate_prior_moments(
        samples,
        grid_expected_mass,
        root_variances={group_name: 1.0},
    )
    variance_slope = at_one.variance - at_zero.variance
    if variance_slope <= tolerance:
        raise ValueError("Aggregate variance does not increase with the selected group root variance.")

    calibrated_variance = max((desired_variance - at_zero.variance) / variance_slope, 0.0)
    achieved = aggregate_prior_moments(
        samples,
        grid_expected_mass,
        root_variances={group_name: calibrated_variance},
    )
    return RootVarianceCalibration(
        group_name=group_name,
        target_relative_standard_deviation=target,
        minimum_relative_standard_deviation=at_zero.relative_standard_deviation,
        calibrated_root_variance=calibrated_variance,
        achieved_relative_standard_deviation=achieved.relative_standard_deviation,
        feasible=True,
    )


__all__ = [
    "AggregatePriorMoments",
    "RootVarianceCalibration",
    "aggregate_prior_moments",
    "calibrate_group_root_variance",
]
