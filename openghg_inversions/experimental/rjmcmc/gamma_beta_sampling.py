"""Seeded topology-mobility sampling for the Gamma--Beta baseline.

This module deliberately orchestrates only the mixed local split/merge
reversible-jump kernel. Root-total and persistent-fraction rejuvenation are
separate kernels so prior-only topology mobility can be measured without
conflating it with continuous tuning.

The root total therefore remains fixed, and a fraction already active in both
the source and candidate is not refreshed independently. This is a diagnostic
chain conditional on its supplied continuous coordinates, not an ergodic
sampler for the full joint Gamma--Beta posterior. Use the separate refresh
proposals in :mod:`openghg_inversions.experimental.rjmcmc.gamma_beta_proposals`
when constructing a compound full-posterior schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.dyadic_tree import DyadicFrontier
from openghg_inversions.experimental.rjmcmc.gamma_beta_proposals import (
    GammaBetaTransitionTerms,
    accept_or_reject,
    propose_merge,
    propose_split,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreeProblem,
    GammaBetaTreeState,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import PCG64State

GAMMA_BETA_STRUCTURAL_SCHEDULE_ID = "gamma_beta_mixed_split_merge_v1"


def _readonly_vector(
    values: object,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    name: str,
) -> np.ndarray:
    """Return an owned read-only one-dimensional array."""
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GammaBetaSamplerConfig:
    """Settings for one fixed-tree structural sampling segment.

    Args:
        iterations: Positive number of atomic mixed split/merge opportunities.
        seed: Non-negative integer seed used to initialize NumPy PCG64, or
            ``None`` for NumPy's nondeterministic initialization.
        split_direction_probability: Probability of selecting split rather
            than merge at each opportunity. It must be strictly between zero
            and one. Unavailable selected directions remain self-transitions.

    Raises:
        TypeError: If ``iterations`` or a non-null ``seed`` is not
            integer-like, or the direction probability is not float-like.
        ValueError: If an integer setting or direction probability lies
            outside its supported range.
    """

    iterations: int
    seed: int | None = None
    split_direction_probability: float = 0.5

    def __post_init__(self) -> None:
        """Normalize and validate kernel settings."""
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, Integral):
            raise TypeError("iterations must be an integer.")
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise TypeError("seed must be a non-negative integer or None.")
            if self.seed < 0:
                raise ValueError("seed must be non-negative.")
            object.__setattr__(self, "seed", int(self.seed))
        split_probability = float(self.split_direction_probability)
        if not isfinite(split_probability) or not 0.0 < split_probability < 1.0:
            raise ValueError("split_direction_probability must lie strictly between zero and one.")
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "split_direction_probability", split_probability)


@dataclass(frozen=True, slots=True)
class GammaBetaTrace:
    """Retained active-only states and every structural transition diagnostic.

    Variable-dimensional fraction vectors are stored as an immutable tuple,
    one canonical read-only vector per retained state. Attempt diagnostics
    cover every atomic transition and therefore need not align with retained
    states when warmup or thinning is active.

    Attributes:
        frontiers: Retained canonical active frontiers.
        split_fractions: Retained read-only fraction vectors in each
            frontier's canonical active-split-node order.
        root_total: Read-only retained root totals. They are constant under
            this structural-only schedule.
        k: Read-only retained frontier sizes.
        log_target: Read-only retained complete log targets.
        state_transition: Read-only global transition coordinates of retained
            states; coordinate zero denotes the supplied initial state.
        moves: Read-only move names for every attempted structural
            opportunity.
        valid: Read-only validity flag for every attempt.
        accepted: Read-only acceptance flag for every attempt. Invalid
            boundary opportunities are included in the denominator.
        node_id: Read-only selected node IDs, or ``-1`` when the selected
            direction was unavailable.
        log_acceptance_ratio: Read-only raw, untruncated Metropolis--Hastings
            log ratios; invalid opportunities store negative infinity.
    """

    frontiers: tuple[DyadicFrontier, ...]
    split_fractions: tuple[NDArray[np.float64], ...]
    root_total: NDArray[np.float64]
    k: NDArray[np.int64]
    log_target: NDArray[np.float64]
    state_transition: NDArray[np.int64]
    moves: NDArray[np.str_]
    valid: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    node_id: NDArray[np.int64]
    log_acceptance_ratio: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own arrays and validate retained/attempted axis contracts."""
        frontiers = tuple(self.frontiers)
        if any(not isinstance(frontier, DyadicFrontier) for frontier in frontiers):
            raise TypeError("frontiers must contain DyadicFrontier values.")
        object.__setattr__(self, "frontiers", frontiers)
        frozen_fractions: list[NDArray[np.float64]] = []
        for values in self.split_fractions:
            fraction = _readonly_vector(
                values,
                dtype=np.float64,
                name="split_fractions entry",
            )
            if np.any(~np.isfinite(fraction)) or np.any((fraction <= 0.0) | (fraction >= 1.0)):
                raise ValueError("retained split fractions must lie strictly between zero and one.")
            frozen_fractions.append(fraction)
        object.__setattr__(self, "split_fractions", tuple(frozen_fractions))

        retained_fields = {
            "root_total": np.float64,
            "k": np.int64,
            "log_target": np.float64,
            "state_transition": np.int64,
        }
        for name, dtype in retained_fields.items():
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=dtype, name=name),
            )
        attempted_fields = {
            "moves": np.dtype("U5"),
            "valid": np.bool_,
            "accepted": np.bool_,
            "node_id": np.int64,
            "log_acceptance_ratio": np.float64,
        }
        for name, dtype in attempted_fields.items():
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=dtype, name=name),
            )

        retained = len(self.frontiers)
        if len(self.split_fractions) != retained:
            raise ValueError("split_fractions must contain one vector per retained frontier.")
        for name in retained_fields:
            if getattr(self, name).shape != (retained,):
                raise ValueError(f"{name} must have one entry per retained frontier.")
        for position, (frontier, fractions) in enumerate(
            zip(self.frontiers, self.split_fractions, strict=True)
        ):
            if fractions.shape != (len(frontier.node_ids) - 1,):
                raise ValueError(f"split_fractions[{position}] must have K-1 entries for its frontier.")
        if retained and (
            np.any(self.root_total <= 0.0)
            or np.any(~np.isfinite(self.root_total))
            or np.any(~np.isfinite(self.log_target))
            or np.any(self.k < 1)
            or np.any(self.state_transition < 0)
            or np.any(np.diff(self.state_transition) <= 0)
        ):
            raise ValueError("retained state summaries contain invalid values.")
        if any(int(k) != len(frontier) for k, frontier in zip(self.k, frontiers, strict=True)):
            raise ValueError("each retained k must equal its frontier size.")

        attempted = self.moves.size
        for name in attempted_fields:
            if getattr(self, name).shape != (attempted,):
                raise ValueError(f"{name} must have one entry per attempted transition.")
        if np.any(~np.isin(self.moves, ("split", "merge"))):
            raise ValueError("moves must contain only 'split' or 'merge'.")
        if np.any(self.accepted & ~self.valid):
            raise ValueError("accepted transitions must be valid.")
        if np.any(np.isnan(self.log_acceptance_ratio)) or np.any(self.log_acceptance_ratio == np.inf):
            raise ValueError("log_acceptance_ratio cannot contain NaN or positive infinity.")

    @property
    def acceptance_rate(self) -> float:
        """Return the accepted fraction of all structural opportunities.

        Invalid boundary opportunities remain in the denominator.

        Returns:
            Mean of the every-attempt ``accepted`` flags.
        """
        return float(np.mean(self.accepted))


@dataclass(frozen=True, slots=True)
class GammaBetaCheckpoint:
    """Exact in-memory continuation boundary for the structural baseline.

    The checkpoint deliberately retains the exact problem object and is not a
    durable serialization contract. Continuation requires that same object by
    identity.

    Attributes:
        problem: Exact immutable problem object used by the chain.
        state: Final active-only state at the segment boundary.
        rng_state: Exact NumPy PCG64 state after the segment.
        transitions_completed: Number of global atomic opportunities already
            completed.
        split_direction_probability: Structural direction probability fixed
            for continuation.
        retention: Global warmup/thinning coordinates fixed for continuation.
        schedule_id: Identifier for the structural-only mixed schedule.
    """

    problem: GammaBetaTreeProblem
    state: GammaBetaTreeState
    rng_state: PCG64State
    transitions_completed: int
    split_direction_probability: float
    retention: RetentionSettings
    schedule_id: str = GAMMA_BETA_STRUCTURAL_SCHEDULE_ID

    def __post_init__(self) -> None:
        """Validate the exact continuation boundary."""
        if not isinstance(self.problem, GammaBetaTreeProblem):
            raise TypeError("problem must be a GammaBetaTreeProblem.")
        if not isinstance(self.state, GammaBetaTreeState):
            raise TypeError("state must be a GammaBetaTreeState.")
        if self.state.problem is not self.problem:
            raise ValueError("checkpoint state must belong to checkpoint problem.")
        if not isfinite(self.state.log_target):
            raise ValueError("checkpoint state must have finite target support.")
        if not isinstance(self.rng_state, PCG64State):
            raise TypeError("rng_state must be a PCG64State.")
        if isinstance(self.transitions_completed, bool) or not isinstance(
            self.transitions_completed,
            Integral,
        ):
            raise TypeError("transitions_completed must be an integer.")
        if self.transitions_completed < 0:
            raise ValueError("transitions_completed must be non-negative.")
        split_probability = float(self.split_direction_probability)
        if not isfinite(split_probability) or not 0.0 < split_probability < 1.0:
            raise ValueError("split_direction_probability must lie strictly between zero and one.")
        if not isinstance(self.retention, RetentionSettings):
            raise TypeError("retention must be a RetentionSettings.")
        if not isinstance(self.schedule_id, str):
            raise TypeError("schedule_id must be a string.")
        object.__setattr__(self, "transitions_completed", int(self.transitions_completed))
        object.__setattr__(self, "split_direction_probability", split_probability)


@dataclass(frozen=True, slots=True)
class GammaBetaSamplingResult:
    """One sampling segment, its final state, and continuation boundary.

    Attributes:
        trace: Retained states and every-attempt structural diagnostics for
            this segment.
        final_state: State visited after the segment's final opportunity.
        checkpoint: Exact in-memory continuation boundary.
    """

    trace: GammaBetaTrace
    final_state: GammaBetaTreeState
    checkpoint: GammaBetaCheckpoint


def _retained_transition_numbers(
    *,
    transitions_completed: int,
    iterations: int,
    retention: RetentionSettings,
    include_initial: bool,
) -> NDArray[np.int64]:
    """Return global transition coordinates retained in one segment."""
    lower = transitions_completed if include_initial else transitions_completed + 1
    upper = transitions_completed + iterations
    first = max(lower, retention.warmup_transitions)
    remainder = (first - retention.warmup_transitions) % retention.thin
    if remainder:
        first += retention.thin - remainder
    if first > upper:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, upper + 1, retention.thin, dtype=np.int64)


def _draw_structural_transition(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    split_direction_probability: float,
    rng: np.random.Generator,
) -> GammaBetaTransitionTerms:
    """Draw one mixed split/merge candidate with explicit boundary attempts."""
    select_split = float(rng.random()) < split_direction_probability
    if select_split:
        splittable = problem.tree.splittable_nodes(state.frontier)
        if not splittable:
            return propose_split(
                problem,
                state,
                leaf_node_id=-1,
                new_fraction=0.5,
                split_direction_probability=split_direction_probability,
            )
        node_id = splittable[int(rng.integers(len(splittable)))]
        alpha, beta = problem.prior.beta_parameters(node_id)
        fraction = float(rng.beta(alpha, beta))
        return propose_split(
            problem,
            state,
            leaf_node_id=node_id,
            new_fraction=fraction,
            split_direction_probability=split_direction_probability,
        )

    mergeable = problem.tree.mergeable_parents(state.frontier)
    if not mergeable:
        return propose_merge(
            problem,
            state,
            parent_node_id=-1,
            split_direction_probability=split_direction_probability,
        )
    parent_id = mergeable[int(rng.integers(len(mergeable)))]
    return propose_merge(
        problem,
        state,
        parent_node_id=parent_id,
        split_direction_probability=split_direction_probability,
    )


def _run_segment(
    problem: GammaBetaTreeProblem,
    initial_state: GammaBetaTreeState,
    *,
    iterations: int,
    rng: np.random.Generator,
    transitions_completed: int,
    split_direction_probability: float,
    retention: RetentionSettings,
    include_initial: bool,
) -> GammaBetaSamplingResult:
    """Run an exact structural segment using global retention coordinates."""
    if initial_state.problem is not problem:
        raise ValueError("initial_state must have been built for the supplied problem.")
    if not isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target support.")
    retained_transitions = _retained_transition_numbers(
        transitions_completed=transitions_completed,
        iterations=iterations,
        retention=retention,
        include_initial=include_initial,
    )
    retained_frontiers: list[DyadicFrontier] = []
    retained_fractions: list[NDArray[np.float64]] = []
    retained_root: list[float] = []
    retained_k: list[int] = []
    retained_target: list[float] = []
    moves = np.empty(iterations, dtype="U5")
    valid = np.empty(iterations, dtype=np.bool_)
    accepted = np.empty(iterations, dtype=np.bool_)
    node_id = np.full(iterations, -1, dtype=np.int64)
    log_acceptance_ratio = np.empty(iterations, dtype=np.float64)

    state = initial_state
    retained_position = 0

    def retain(current: GammaBetaTreeState) -> None:
        """Copy one active-only state into variable-dimensional trace storage."""
        nonlocal retained_position
        retained_frontiers.append(current.frontier)
        fractions = np.array(current.split_fractions, copy=True)
        fractions.setflags(write=False)
        retained_fractions.append(fractions)
        retained_root.append(current.root_total)
        retained_k.append(current.k)
        retained_target.append(current.log_target)
        retained_position += 1

    if retained_transitions.size and retained_transitions[0] == transitions_completed:
        retain(state)

    next_retained = (
        int(retained_transitions[retained_position])
        if retained_position < retained_transitions.size
        else None
    )
    for iteration in range(iterations):
        transition = _draw_structural_transition(
            problem,
            state,
            split_direction_probability=split_direction_probability,
            rng=rng,
        )
        uniform = float(rng.random())
        log_uniform = -np.inf if uniform == 0.0 else log(uniform)
        next_state = accept_or_reject(state, transition, log_uniform=log_uniform)
        proposal_accepted = transition.valid and next_state is transition.candidate
        moves[iteration] = transition.move
        valid[iteration] = transition.valid
        accepted[iteration] = proposal_accepted
        if transition.node_id is not None:
            node_id[iteration] = transition.node_id
        log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
        state = next_state
        completed = transitions_completed + iteration + 1
        if next_retained == completed:
            retain(state)
            next_retained = (
                int(retained_transitions[retained_position])
                if retained_position < retained_transitions.size
                else None
            )

    if retained_position != retained_transitions.size:
        raise RuntimeError("retained Gamma-Beta state count did not match planned coordinates.")
    total_transitions = transitions_completed + iterations
    checkpoint = GammaBetaCheckpoint(
        problem=problem,
        state=state,
        rng_state=PCG64State.from_generator(rng),
        transitions_completed=total_transitions,
        split_direction_probability=split_direction_probability,
        retention=retention,
    )
    return GammaBetaSamplingResult(
        trace=GammaBetaTrace(
            frontiers=tuple(retained_frontiers),
            split_fractions=tuple(retained_fractions),
            root_total=np.asarray(retained_root, dtype=np.float64),
            k=np.asarray(retained_k, dtype=np.int64),
            log_target=np.asarray(retained_target, dtype=np.float64),
            state_transition=retained_transitions,
            moves=moves,
            valid=valid,
            accepted=accepted,
            node_id=node_id,
            log_acceptance_ratio=log_acceptance_ratio,
        ),
        final_state=state,
        checkpoint=checkpoint,
    )


def sample_gamma_beta_tree(
    problem: GammaBetaTreeProblem,
    initial_state: GammaBetaTreeState,
    config: GammaBetaSamplerConfig,
    *,
    retention: RetentionSettings | None = None,
) -> GammaBetaSamplingResult:
    """Run a fresh seeded topology-mobility Gamma--Beta chain.

    The root total is fixed and persistent active fractions are not
    independently refreshed. This function therefore samples a diagnostic
    conditional target, not the full joint Gamma--Beta posterior.

    Args:
        problem: Fixed-tree observation model and normalized priors.
        initial_state: State built for the exact supplied problem object.
        config: Seed, transition count, and mixed-direction probability.
        retention: Optional global warmup/thinning coordinates.

    Returns:
        Segment trace, final state, and exact in-memory checkpoint.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the initial state belongs to a different problem.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(initial_state, GammaBetaTreeState):
        raise TypeError("initial_state must be a GammaBetaTreeState.")
    if not isinstance(config, GammaBetaSamplerConfig):
        raise TypeError("config must be a GammaBetaSamplerConfig.")
    retention_settings = RetentionSettings() if retention is None else retention
    if not isinstance(retention_settings, RetentionSettings):
        raise TypeError("retention must be a RetentionSettings instance or None.")
    rng = np.random.Generator(np.random.PCG64(config.seed))
    return _run_segment(
        problem,
        initial_state,
        iterations=config.iterations,
        rng=rng,
        transitions_completed=0,
        split_direction_probability=config.split_direction_probability,
        retention=retention_settings,
        include_initial=True,
    )


def continue_gamma_beta_tree(
    problem: GammaBetaTreeProblem,
    checkpoint: GammaBetaCheckpoint,
    *,
    iterations: int,
) -> GammaBetaSamplingResult:
    """Continue the topology-mobility baseline from an in-memory checkpoint.

    As in :func:`sample_gamma_beta_tree`, root total and persistent active
    fractions are not independently refreshed.

    Args:
        problem: The exact problem object retained by ``checkpoint``.
        checkpoint: In-memory checkpoint from this structural schedule.
        iterations: Positive number of further atomic opportunities.

    Returns:
        Continued segment trace, final state, and next exact checkpoint.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If iterations are non-positive, the problem object differs,
            or the checkpoint schedule is incompatible.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(checkpoint, GammaBetaCheckpoint):
        raise TypeError("checkpoint must be a GammaBetaCheckpoint.")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral):
        raise TypeError("iterations must be an integer.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if checkpoint.problem is not problem:
        raise ValueError("continuation requires the exact in-memory problem object.")
    if checkpoint.schedule_id != GAMMA_BETA_STRUCTURAL_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible with this sampler.")
    return _run_segment(
        problem,
        checkpoint.state,
        iterations=int(iterations),
        rng=checkpoint.rng_state.generator(),
        transitions_completed=checkpoint.transitions_completed,
        split_direction_probability=checkpoint.split_direction_probability,
        retention=checkpoint.retention,
        include_initial=False,
    )


__all__ = [
    "GAMMA_BETA_STRUCTURAL_SCHEDULE_ID",
    "GammaBetaCheckpoint",
    "GammaBetaSamplerConfig",
    "GammaBetaSamplingResult",
    "GammaBetaTrace",
    "continue_gamma_beta_tree",
    "sample_gamma_beta_tree",
]
