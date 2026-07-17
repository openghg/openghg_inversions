"""Scientific orchestration for the experimental fixed-count dyadic SLS demo.

The historical quadratic tile score is used only to construct a useful greedy
initializer. The stochastic local search itself maximizes full Gaussian DFS
under an explicit isotropic region-covariance benchmark. This module contains
no atmospheric file loading or plotting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log

import numpy as np
import numpy.typing as npt

from .initializers import greedy_partition
from .multiscale import MultiscaleDesign
from .objectives import GaussianDFSObjective, IsotropicRegionCovariance, prototype_quadratic_tile_scores
from .proposals import PairedMove, apply_move, enumerate_merge_moves, enumerate_split_moves
from .search import PiecewiseGeometricSchedule, SearchProposal, SearchResult, stochastic_local_search
from .state import PartitionState
from .tree import DyadicTree


@dataclass(frozen=True, slots=True)
class DemoSearchConfig:
    """Configuration for one reproducible fixed-count Gaussian DFS search.

    Args:
        target_regions: Fixed number of active dyadic regions.
        iterations: Number of stochastic local-search proposals.
        pilot_proposals: Number of local losses used to calibrate temperature.
        tau: Region-multiplier prior standard deviation for the explicitly
            benchmark-only isotropic covariance.
        seed: Random seed used for pilot proposals and the search.
        record_every: Retain every Nth evaluated search proposal, in addition
            to accepted, best, and final proposals.
    """

    target_regions: int = 32
    iterations: int = 300
    pilot_proposals: int = 100
    tau: float = 1.0
    seed: int = 20260717
    record_every: int = 5

    def __post_init__(self) -> None:
        """Validate search configuration."""
        if self.target_regions < 2:
            raise ValueError("target_regions must be at least 2 for paired moves.")
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.pilot_proposals < 1:
            raise ValueError("pilot_proposals must be positive.")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be positive and finite.")
        if self.record_every < 1:
            raise ValueError("record_every must be positive.")


@dataclass(frozen=True, slots=True)
class DemoSearchRun:
    """Reusable state and diagnostics from one fixed-count search run.

    Attributes:
        config: Frozen run configuration.
        tree: Canonical tree over the supplied design grid.
        design: Precomputed candidate observation columns.
        initial_state: Proxy-informed greedy initializer.
        result: Full stochastic local-search result.
        schedule: Calibrated temperature schedule.
        pilot_losses: Positive full-DFS losses observed during calibration.
    """

    config: DemoSearchConfig
    tree: DyadicTree
    design: MultiscaleDesign
    initial_state: PartitionState
    result: SearchResult[PartitionState, PairedMove]
    schedule: PiecewiseGeometricSchedule
    pilot_losses: tuple[float, ...]


def run_fixed_count_dfs_search(
    contribution_grid: npt.ArrayLike,
    r_diag: npt.ArrayLike,
    config: DemoSearchConfig,
    *,
    support_grid: npt.ArrayLike | None = None,
) -> DemoSearchRun:
    """Run proxy-initialized fixed-count SLS using full Gaussian DFS.

    Args:
        contribution_grid: Observation contributions with shape
            ``(observation, row, column)``.
        r_diag: Positive diagonal of observation covariance ``R``.
        config: Reproducible search configuration.
        support_grid: Optional positive physical support for each spatial cell.
            Defaults to one per input cell. Partial coarsening blocks should
            pass their fine-cell support counts here.

    Returns:
        Search state, calibrated schedule, pilot losses, and full trace.

    Raises:
        ValueError: If inputs are invalid or the requested partition has no
            valid fixed-count paired proposal.

    Notes:
        The isotropic covariance used here does not implement the
        Bocquet-consistent transformation ``B_P = P B P.T``. It is an explicit
        proof-of-concept benchmark behind the same covariance-builder boundary.
    """
    grid = _validated_grid(contribution_grid)
    variances = _validated_variances(r_diag, observations=grid.shape[0])
    spatial_shape = (grid.shape[1], grid.shape[2])
    cell_support = _validated_support(support_grid, shape=spatial_shape)
    tree = DyadicTree.from_shape(spatial_shape)
    if config.target_regions > len(tree.leaf_ids):
        raise ValueError("target_regions exceeds the number of grid cells.")

    design = MultiscaleDesign.from_grid(grid, tree)
    support = MultiscaleDesign.from_grid(cell_support[np.newaxis, :, :], tree).H[0]
    proxy_scores = prototype_quadratic_tile_scores(design.H, 1.0 / variances, support)

    def split_gain(node_id: int) -> float:
        """Return the historical proxy improvement from splitting one node."""
        children = tree.children(node_id)
        return float(proxy_scores[list(children)].sum() - proxy_scores[node_id])

    initial_state = greedy_partition(tree, config.target_regions, split_gain).state
    objective = GaussianDFSObjective(variances, IsotropicRegionCovariance(config.tau))

    def score(state: PartitionState) -> float:
        """Evaluate full Gaussian DFS for one active frontier."""
        return objective(state, design)

    rng = np.random.default_rng(config.seed)
    initial_score = score(initial_state)
    pilot_losses = _pilot_local_losses(
        tree,
        initial_state,
        initial_score,
        score,
        config.pilot_proposals,
        rng,
    )
    schedule = _schedule_from_losses(pilot_losses, initial_score)
    result = stochastic_local_search(
        initial_state,
        objective=score,
        propose=lambda state, generator: _sample_paired_proposal(tree, state, generator),
        schedule=schedule,
        iterations=config.iterations,
        rng=rng,
        record_every=config.record_every,
    )
    return DemoSearchRun(
        config=config,
        tree=tree,
        design=design,
        initial_state=initial_state,
        result=result,
        schedule=schedule,
        pilot_losses=pilot_losses,
    )


def _sample_paired_proposal(
    tree: DyadicTree,
    state: PartitionState,
    rng: np.random.Generator,
) -> SearchProposal[PartitionState, PairedMove] | None:
    """Sample a valid merge-then-split move without enumerating all pairs.

    Args:
        tree: Canonical tree defining local moves.
        state: Current fixed-count partition.
        rng: Caller-owned random generator.

    Returns:
        One paired proposal, or ``None`` if the state has no nontrivial paired
        move. The proposal is intended for optimization and is not claimed to
        be uniform over unique neighboring partitions.
    """
    merge_moves = enumerate_merge_moves(tree, state)
    for merge_index in rng.permutation(len(merge_moves)):
        merge_move = merge_moves[int(merge_index)]
        merged_state = apply_move(tree, state, merge_move)
        split_moves = tuple(
            move for move in enumerate_split_moves(tree, merged_state) if move.node_id != merge_move.parent_id
        )
        if not split_moves:
            continue
        split_move = split_moves[int(rng.integers(len(split_moves)))]
        move = PairedMove(merge_move.parent_id, split_move.node_id)
        return SearchProposal(apply_move(tree, state, move), move)
    return None


def _pilot_local_losses(
    tree: DyadicTree,
    state: PartitionState,
    state_score: float,
    score: Callable[[PartitionState], float],
    proposals: int,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    """Collect positive full-score losses from local fixed-count proposals.

    Args:
        tree: Canonical tree defining proposals.
        state: Initial partition around which proposals are sampled.
        state_score: Full objective value for ``state``.
        score: Callable returning full Gaussian DFS.
        proposals: Maximum number of candidates to evaluate.
        rng: Caller-owned random generator.

    Returns:
        Positive score losses in evaluation order.
    """
    losses: list[float] = []
    for _ in range(proposals):
        proposal = _sample_paired_proposal(tree, state, rng)
        if proposal is None:
            break
        loss = state_score - float(score(proposal.state))
        if loss > 0.0:
            losses.append(loss)
    return tuple(losses)


def _schedule_from_losses(losses: tuple[float, ...], reference_score: float) -> PiecewiseGeometricSchedule:
    """Calibrate 0.8-to-0.01 median-loss acceptance temperatures.

    Args:
        losses: Positive local objective losses from a discarded pilot.
        reference_score: Initial full objective, used only for a deterministic
            fallback when no sampled proposal loses score.

    Returns:
        Ten-percent hold, eighty-percent cooling, ten-percent polish schedule.
    """
    if losses:
        representative_loss = float(np.median(losses))
    else:
        representative_loss = max(abs(reference_score) * 1e-3, 1e-6)
    return PiecewiseGeometricSchedule(
        initial_temperature=-representative_loss / log(0.8),
        final_temperature=-representative_loss / log(0.01),
        hold_fraction=0.1,
        polish_fraction=0.1,
    )


def _validated_grid(values: npt.ArrayLike) -> np.ndarray:
    """Return a finite non-empty observation-by-grid floating array."""
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 3 or any(extent == 0 for extent in grid.shape):
        raise ValueError("contribution_grid must have non-empty (observation, row, column) shape.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("contribution_grid must contain only finite values.")
    return grid


def _validated_variances(values: npt.ArrayLike, *, observations: int) -> np.ndarray:
    """Return positive finite variances matching the observation count."""
    variances = np.asarray(values, dtype=float)
    if variances.shape != (observations,):
        raise ValueError("r_diag must have one value per observation.")
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("r_diag must contain only positive finite values.")
    return variances


def _validated_support(values: npt.ArrayLike | None, *, shape: tuple[int, int]) -> np.ndarray:
    """Return positive finite physical support matching the search grid."""
    if values is None:
        return np.ones(shape, dtype=float)
    support = np.asarray(values, dtype=float)
    if support.shape != shape:
        raise ValueError("support_grid must match the spatial contribution_grid shape.")
    if not np.all(np.isfinite(support)) or np.any(support <= 0.0):
        raise ValueError("support_grid must contain only positive finite values.")
    return support


__all__ = ["DemoSearchConfig", "DemoSearchRun", "run_fixed_count_dfs_search"]
