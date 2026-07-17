"""Scientific orchestration for experimental fixed- and variable-count SLS.

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
from .objectives import (
    GaussianDFSObjective,
    IsotropicRegionCovariance,
    isotropic_observation_space_dfs,
    prototype_quadratic_tile_scores,
)
from .proposals import (
    Move,
    PairedMove,
    apply_move,
    enumerate_merge_moves,
    enumerate_split_moves,
)
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
class VariableKSearchConfig:
    """Configuration for a variable-count Gaussian DFS search.

    Args:
        initial_regions: Number of regions in the proxy-greedy initializer.
        free_regions: Region count below which no complexity penalty applies.
        min_regions: Hard lower bound on active regions.
        max_regions: Hard upper bound on active regions.
        penalty_per_extra_region: Utility penalty for every region above
            ``free_regions``.
        paired_move_probability: Probability assigned to a fixed-count paired
            move when split, merge, and paired move types are all available.
        initial_loss_acceptance: Target acceptance probability for a
            representative pilot loss at the initial temperature.
        final_loss_acceptance: Target acceptance probability for the same loss
            at the end of geometric cooling.
        hold_fraction: Fraction of iterations at the initial temperature.
        polish_fraction: Fraction of iterations at zero temperature.
        iterations: Number of stochastic local-search proposals.
        pilot_proposals: Number of local utility losses used to calibrate the
            temperature schedule.
        tau: Region-multiplier prior standard deviation for the benchmark-only
            isotropic covariance.
        seed: Random seed used for pilot proposals and search.
        record_every: Retain every Nth proposal in addition to accepted, best,
            and final proposals.
    """

    initial_regions: int = 24
    free_regions: int = 32
    min_regions: int = 2
    max_regions: int = 96
    penalty_per_extra_region: float = 0.02
    paired_move_probability: float = 0.2
    iterations: int = 600
    pilot_proposals: int = 150
    tau: float = 1.0
    seed: int = 20260717
    record_every: int = 5
    initial_loss_acceptance: float = 0.8
    final_loss_acceptance: float = 0.01
    hold_fraction: float = 0.1
    polish_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Validate variable-count search configuration."""
        if self.min_regions < 1:
            raise ValueError("min_regions must be positive.")
        if not self.min_regions <= self.initial_regions <= self.max_regions:
            raise ValueError("initial_regions must lie between min_regions and max_regions.")
        if self.free_regions < 0:
            raise ValueError("free_regions must be non-negative.")
        if not np.isfinite(self.penalty_per_extra_region) or self.penalty_per_extra_region < 0.0:
            raise ValueError("penalty_per_extra_region must be finite and non-negative.")
        if not 0.0 <= self.paired_move_probability <= 1.0:
            raise ValueError("paired_move_probability must lie in [0, 1].")
        if not 0.0 < self.final_loss_acceptance <= self.initial_loss_acceptance < 1.0:
            raise ValueError(
                "loss acceptance probabilities must satisfy "
                "0 < final_loss_acceptance <= initial_loss_acceptance < 1."
            )
        if not 0.0 <= self.hold_fraction < 1.0:
            raise ValueError("hold_fraction must lie in [0, 1).")
        if not 0.0 <= self.polish_fraction < 1.0:
            raise ValueError("polish_fraction must lie in [0, 1).")
        if self.hold_fraction + self.polish_fraction >= 1.0:
            raise ValueError("hold_fraction and polish_fraction must sum to less than 1.")
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


@dataclass(frozen=True, slots=True)
class VariableKSearchRun:
    """State and diagnostics from one variable-count search.

    Attributes:
        config: Frozen variable-count configuration.
        tree: Canonical tree over the supplied design grid.
        design: Precomputed candidate observation columns.
        initial_state: Proxy-informed greedy initializer.
        result: Search result whose scores are penalized utilities.
        schedule: Calibrated temperature schedule.
        pilot_losses: Positive utility losses observed during calibration.
        initial_dfs: Unpenalized Gaussian DFS at the initializer.
        final_dfs: Unpenalized Gaussian DFS at the final state.
        best_dfs: Unpenalized Gaussian DFS at the best-utility state.
        cellwise_isotropic_dfs: DFS for all coarsened cells under an independent
            isotropic cell prior. This is not an upper bound for partition DFS
            because the current regional prior is not projected from it.
    """

    config: VariableKSearchConfig
    tree: DyadicTree
    design: MultiscaleDesign
    initial_state: PartitionState
    result: SearchResult[PartitionState, Move]
    schedule: PiecewiseGeometricSchedule
    pilot_losses: tuple[float, ...]
    initial_dfs: float
    final_dfs: float
    best_dfs: float
    cellwise_isotropic_dfs: float


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


def run_variable_k_dfs_search(
    contribution_grid: npt.ArrayLike,
    r_diag: npt.ArrayLike,
    config: VariableKSearchConfig,
    *,
    support_grid: npt.ArrayLike | None = None,
) -> VariableKSearchRun:
    """Run split/merge SLS with an explicit soft penalty above a free K.

    Args:
        contribution_grid: Observation contributions with shape
            ``(observation, row, column)``.
        r_diag: Positive diagonal of observation covariance ``R``.
        config: Variable-count search configuration.
        support_grid: Optional positive physical support for each spatial cell.
            Defaults to one per input cell.

    Returns:
        Variable-count search state and separate DFS/utility diagnostics.

    Raises:
        ValueError: If inputs or configured region bounds are invalid.

    Notes:
        Search utility is Gaussian DFS minus a soft complexity penalty. It is
        an optimization criterion, not a log posterior or a prior on ``K``.
    """
    grid = _validated_grid(contribution_grid)
    variances = _validated_variances(r_diag, observations=grid.shape[0])
    spatial_shape = (grid.shape[1], grid.shape[2])
    cell_support = _validated_support(support_grid, shape=spatial_shape)
    tree = DyadicTree.from_shape(spatial_shape)
    if config.max_regions > len(tree.leaf_ids):
        raise ValueError("max_regions exceeds the number of grid cells.")

    design = MultiscaleDesign.from_grid(grid, tree)
    cellwise_isotropic_dfs = isotropic_observation_space_dfs(
        design.H[:, tree.leaf_ids],
        variances,
        config.tau,
    )
    support = MultiscaleDesign.from_grid(cell_support[np.newaxis, :, :], tree).H[0]
    proxy_scores = prototype_quadratic_tile_scores(design.H, 1.0 / variances, support)

    def split_gain(node_id: int) -> float:
        """Return the historical proxy improvement from splitting one node."""
        children = tree.children(node_id)
        return float(proxy_scores[list(children)].sum() - proxy_scores[node_id])

    initial_state = greedy_partition(tree, config.initial_regions, split_gain).state
    objective = GaussianDFSObjective(variances, IsotropicRegionCovariance(config.tau))

    def dfs(state: PartitionState) -> float:
        """Evaluate unpenalized Gaussian DFS for one partition."""
        return objective(state, design)

    def utility(state: PartitionState) -> float:
        """Evaluate DFS minus the configured excess-region penalty."""
        return dfs(state) - excess_region_penalty(len(state.active), config)

    def propose(
        state: PartitionState, rng: np.random.Generator
    ) -> SearchProposal[PartitionState, Move] | None:
        """Sample one valid split, merge, or paired optimization move."""
        return _sample_variable_k_proposal(tree, state, config, rng)

    rng = np.random.default_rng(config.seed)
    initial_utility = utility(initial_state)
    pilot_losses = _pilot_variable_k_losses(
        initial_state,
        initial_utility,
        utility,
        propose,
        config.pilot_proposals,
        rng,
    )
    schedule = _schedule_from_losses(
        pilot_losses,
        initial_utility,
        initial_loss_acceptance=config.initial_loss_acceptance,
        final_loss_acceptance=config.final_loss_acceptance,
        hold_fraction=config.hold_fraction,
        polish_fraction=config.polish_fraction,
    )
    result = stochastic_local_search(
        initial_state,
        objective=utility,
        propose=propose,
        schedule=schedule,
        iterations=config.iterations,
        rng=rng,
        record_every=config.record_every,
    )
    return VariableKSearchRun(
        config=config,
        tree=tree,
        design=design,
        initial_state=initial_state,
        result=result,
        schedule=schedule,
        pilot_losses=pilot_losses,
        initial_dfs=dfs(initial_state),
        final_dfs=dfs(result.final_state),
        best_dfs=dfs(result.best_state),
        cellwise_isotropic_dfs=cellwise_isotropic_dfs,
    )


def excess_region_penalty(region_count: int, config: VariableKSearchConfig) -> float:
    """Return the soft utility penalty for regions above ``free_regions``.

    Args:
        region_count: Number of active regions in a candidate partition.
        config: Variable-count configuration defining the free count and slope.

    Returns:
        Non-negative linear complexity penalty.

    Raises:
        ValueError: If ``region_count`` is negative.
    """
    if region_count < 0:
        raise ValueError("region_count must be non-negative.")
    return config.penalty_per_extra_region * max(0, region_count - config.free_regions)


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


def _sample_variable_k_proposal(
    tree: DyadicTree,
    state: PartitionState,
    config: VariableKSearchConfig,
    rng: np.random.Generator,
) -> SearchProposal[PartitionState, Move] | None:
    """Sample a split, merge, or paired move within configured K bounds.

    Args:
        tree: Canonical tree defining local moves.
        state: Current variable-count partition.
        config: Bounds and paired-move probability.
        rng: Caller-owned random generator.

    Returns:
        One valid proposal, or ``None`` if no configured move is available.

    Notes:
        Move-type weights control optimizer exploration only. They are not
        Hastings probabilities for posterior MCMC.
    """
    region_count = len(state.active)
    split_moves = enumerate_split_moves(tree, state) if region_count < config.max_regions else ()
    merge_moves = enumerate_merge_moves(tree, state) if region_count > config.min_regions else ()
    paired_proposal = _sample_paired_proposal(tree, state, rng)

    single_weight = (1.0 - config.paired_move_probability) / 2.0
    choices: list[tuple[str, float]] = []
    if split_moves:
        choices.append(("split", single_weight))
    if merge_moves:
        choices.append(("merge", single_weight))
    if paired_proposal is not None:
        choices.append(("paired", config.paired_move_probability))
    if not choices:
        return None

    weights = np.asarray([weight for _, weight in choices], dtype=float)
    if float(weights.sum()) == 0.0:
        weights = np.ones(len(choices), dtype=float)
    weights /= weights.sum()
    choice = choices[int(rng.choice(len(choices), p=weights))][0]

    if choice == "split":
        move: Move = split_moves[int(rng.integers(len(split_moves)))]
        return SearchProposal(apply_move(tree, state, move), move)
    if choice == "merge":
        move = merge_moves[int(rng.integers(len(merge_moves)))]
        return SearchProposal(apply_move(tree, state, move), move)

    if paired_proposal is None:  # pragma: no cover - excluded from choices above.
        return None
    move = paired_proposal.move
    return SearchProposal(paired_proposal.state, move)


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


def _pilot_variable_k_losses(
    state: PartitionState,
    state_utility: float,
    utility: Callable[[PartitionState], float],
    propose: Callable[
        [PartitionState, np.random.Generator],
        SearchProposal[PartitionState, Move] | None,
    ],
    proposals: int,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    """Collect positive utility losses from variable-count proposals.

    Args:
        state: Initial partition around which proposals are sampled.
        state_utility: Penalized utility of ``state``.
        utility: Callable returning penalized search utility.
        propose: Variable-count proposal sampler.
        proposals: Maximum number of candidates to evaluate.
        rng: Caller-owned random generator.

    Returns:
        Positive utility losses in evaluation order.
    """
    losses: list[float] = []
    for _ in range(proposals):
        proposal = propose(state, rng)
        if proposal is None:
            break
        loss = state_utility - float(utility(proposal.state))
        if loss > 0.0:
            losses.append(loss)
    return tuple(losses)


def _schedule_from_losses(
    losses: tuple[float, ...],
    reference_score: float,
    *,
    initial_loss_acceptance: float = 0.8,
    final_loss_acceptance: float = 0.01,
    hold_fraction: float = 0.1,
    polish_fraction: float = 0.1,
) -> PiecewiseGeometricSchedule:
    """Calibrate temperatures from representative-loss acceptance targets.

    Args:
        losses: Positive local objective losses from a discarded pilot.
        reference_score: Initial full objective, used only for a deterministic
            fallback when no sampled proposal loses score.
        initial_loss_acceptance: Target initial acceptance probability for the
            representative loss.
        final_loss_acceptance: Target acceptance probability at the end of
            geometric cooling.
        hold_fraction: Fraction of iterations held at initial temperature.
        polish_fraction: Fraction of iterations run at zero temperature.

    Returns:
        Calibrated hold, cooling, and zero-temperature polish schedule.
    """
    if losses:
        representative_loss = float(np.median(losses))
    else:
        representative_loss = max(abs(reference_score) * 1e-3, 1e-6)
    return PiecewiseGeometricSchedule(
        initial_temperature=-representative_loss / log(initial_loss_acceptance),
        final_temperature=-representative_loss / log(final_loss_acceptance),
        hold_fraction=hold_fraction,
        polish_fraction=polish_fraction,
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


__all__ = [
    "DemoSearchConfig",
    "DemoSearchRun",
    "VariableKSearchConfig",
    "VariableKSearchRun",
    "excess_region_penalty",
    "run_fixed_count_dfs_search",
    "run_variable_k_dfs_search",
]
