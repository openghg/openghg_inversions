"""Seeded four-slot reference sampler for spatial trans-dimensional MCMC.

The single-chain driver repeats a coefficient slot, two identical mixed
birth/death slots, and a configurable nucleus-location slot using a seeded
NumPy generator. Each dimension slot selects birth or death independently with
equal probability. State traces include the initial state at row zero, while
transition diagnostics describe each attempted move from row ``i`` to row
``i + 1``. This reference driver does not yet provide burn-in management,
parallel chains, or parallel tempering.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.tdmcmc.core import (
    Backend,
    TransDimensionalProblem,
    TransDimensionalState,
)
from openghg_inversions.tdmcmc.proposals import (
    TransitionTerms,
    accept_or_reject,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_global_move,
    propose_local_move,
)


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for the first single-chain reference sampler.

    Every four transitions contain one coefficient proposal, two independent
    equal-probability birth/death proposals, and one nucleus-location proposal.

    Args:
        iterations: Number of attempted transitions.
        coefficient_proposal_sd: Gaussian random-walk scale for coefficients.
        birth_proposal_sd: Gaussian auxiliary-coefficient scale for birth/death.
        seed: Seed passed to :func:`numpy.random.default_rng`.
        backend: State-recomputation backend used by every candidate.
        nucleus_move: Whether the fourth schedule step proposes a globally
            uniform or local discrete-Gaussian nucleus destination.
        local_move_scale: Gaussian distance scale in grid-coordinate units.
            Required when ``nucleus_move`` is ``"local"``.

    Raises:
        ValueError: If iterations, proposal scales, move mode, or backend are
            malformed.
    """

    iterations: int
    coefficient_proposal_sd: float
    birth_proposal_sd: float
    seed: int | None = None
    backend: Backend = "numpy"
    nucleus_move: Literal["global", "local"] = "global"
    local_move_scale: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed sampler settings before allocating a trace."""
        if isinstance(self.iterations, bool) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer.")
        for name in ("coefficient_proposal_sd", "birth_proposal_sd"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.backend not in ("numpy", "numba"):
            raise ValueError("backend must be 'numpy' or 'numba'.")
        if self.nucleus_move not in ("global", "local"):
            raise ValueError("nucleus_move must be 'global' or 'local'.")
        if self.local_move_scale is not None:
            local_move_scale = float(self.local_move_scale)
            if not isfinite(local_move_scale) or local_move_scale <= 0.0:
                raise ValueError("local_move_scale must be finite and positive.")
            object.__setattr__(self, "local_move_scale", local_move_scale)
        if self.nucleus_move == "local" and self.local_move_scale is None:
            raise ValueError("local_move_scale is required for local nucleus moves.")


@dataclass(frozen=True)
class SamplingTrace:
    """Fixed-capacity trace and transition diagnostics from one chain.

    Attributes:
        k: Active-region count at every saved state, with shape
            ``(iterations + 1,)``. Row zero is the initial state.
        nuclei: Padded nucleus indices with shape
            ``(iterations + 1, k_max)``.
        coefficients: Padded region coefficients with shape
            ``(iterations + 1, k_max)``.
        log_target: Normalized log target at every saved state, with shape
            ``(iterations + 1,)``.
        moves: Proposal name for each attempted transition, with shape
            ``(iterations,)``. Entry ``i`` describes the move from state row
            ``i`` to row ``i + 1``. Mixed dimension slots record the selected
            proposal label, ``"birth"`` or ``"death"``, rather than a generic
            dimension-slot label.
        accepted: Whether each attempted transition changed the chain state,
            with shape ``(iterations,)``.
        log_acceptance_ratio: Untruncated log Metropolis-Hastings ratio for
            each attempted transition, with shape ``(iterations,)``.
    """

    k: NDArray[np.int64]
    nuclei: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    log_target: NDArray[np.float64]
    moves: NDArray[np.str_]
    accepted: NDArray[np.bool_]
    log_acceptance_ratio: NDArray[np.float64]

    @property
    def acceptance_rate(self) -> float:
        """Fraction of attempted transitions that changed the chain state."""
        return float(np.mean(self.accepted))


@dataclass(frozen=True)
class SamplingResult:
    """Reference sampler output containing the trace and final cached state.

    Attributes:
        trace: Fixed-capacity states and transition diagnostics for the chain.
        final_state: Fully cached final state corresponding to the last trace
            row.
    """

    trace: SamplingTrace
    final_state: TransDimensionalState


def _empty_cells(problem: TransDimensionalProblem, state: TransDimensionalState) -> NDArray[np.int64]:
    """Return sorted cells that do not currently contain a nucleus."""
    return np.setdiff1d(
        np.arange(problem.n_grid_cells, dtype=np.int64),
        state.active_nuclei,
        assume_unique=True,
    )


def _nearest_parent_coefficient(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    cell: int,
) -> float:
    """Return the coefficient attached to the closest active nucleus."""
    differences = problem.grid_coordinates[state.active_nuclei] - problem.grid_coordinates[int(cell)]
    squared_distance = np.sum(np.square(differences), axis=1)
    return float(state.active_coefficients[int(np.argmin(squared_distance))])


def _draw_transition(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    config: SamplerConfig,
    rng: np.random.Generator,
    move: str,
) -> TransitionTerms:
    """Draw explicit proposal values and construct one seeded transition."""
    if move == "dimension":
        move = "birth" if rng.random() < 0.5 else "death"

    if move == "coefficient":
        position = int(rng.integers(state.k))
        value = float(state.active_coefficients[position] + rng.normal(scale=config.coefficient_proposal_sd))
        return propose_coefficient(
            problem,
            state,
            coefficient_position=position,
            proposed_coefficient=value,
            proposal_stdev=config.coefficient_proposal_sd,
            backend=config.backend,
        )

    if move == "birth":
        empty = _empty_cells(problem, state)
        cell = int(rng.choice(empty)) if empty.size else int(state.active_nuclei[0])
        parent = _nearest_parent_coefficient(problem, state, cell) if empty.size else 1.0
        value = float(parent + rng.normal(scale=config.birth_proposal_sd))
        return propose_birth(
            problem,
            state,
            new_nucleus=cell,
            proposed_coefficient=value,
            proposal_stdev=config.birth_proposal_sd,
            backend=config.backend,
        )

    if move == "death":
        return propose_death(
            problem,
            state,
            remove_position=int(rng.integers(state.k)),
            proposal_stdev=config.birth_proposal_sd,
            backend=config.backend,
        )

    if move == "global_move":
        empty = _empty_cells(problem, state)
        cell = int(rng.choice(empty)) if empty.size else int(state.active_nuclei[0])
        return propose_global_move(
            problem,
            state,
            move_position=int(rng.integers(state.k)),
            new_nucleus=cell,
            backend=config.backend,
        )

    if move == "local_move":
        scale = config.local_move_scale
        if scale is None:  # guarded by SamplerConfig; keeps type narrowing local
            raise ValueError("local_move_scale is required for local nucleus moves.")
        position = int(rng.integers(state.k))
        empty = _empty_cells(problem, state)
        if empty.size:
            origin = int(state.active_nuclei[position])
            differences = (problem.grid_coordinates[empty] - problem.grid_coordinates[origin]) / scale
            log_weights = -0.5 * np.einsum("ij,ij->i", differences, differences)
            weights = np.exp(log_weights - np.max(log_weights))
            cell = int(rng.choice(empty, p=weights / weights.sum()))
        else:
            cell = int(state.active_nuclei[position])
        return propose_local_move(
            problem,
            state,
            move_position=position,
            new_nucleus=cell,
            proposal_scale=scale,
            backend=config.backend,
        )

    raise ValueError(f"Unknown proposal move {move!r}.")


def sample(
    problem: TransDimensionalProblem,
    initial_state: TransDimensionalState,
    config: SamplerConfig,
) -> SamplingResult:
    """Run the first auditable single-chain spatial RJMCMC implementation.

    The four-slot schedule repeats a coefficient move, two identical dimension
    moves, and a nucleus move. Each dimension slot independently chooses birth
    or death with probability one half, making it an invariant paired RJ
    kernel. The equal move-type probabilities cancel from the reported
    Metropolis-Hastings ratio. Nucleus moves are globally uniform by default;
    setting ``nucleus_move="local"`` uses a normalized discrete-Gaussian
    destination kernel instead. Impossible boundary proposals remain explicit
    self-transitions rather than renormalizing the birth/death selection.

    Args:
        problem: Immutable target and fine-grid numerical inputs.
        initial_state: Complete initial state compatible with ``problem``.
        config: Transition count, scales, seed, and numerical backend.

    Returns:
        Fixed-capacity trace and the final fully cached sampler state.

    Raises:
        ValueError: If the initial state capacity is incompatible with the
            problem or its target density is not finite.
    """
    if initial_state.capacity != problem.k_max:
        raise ValueError("initial_state capacity must equal problem.k_max.")
    if not np.isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target density.")

    rng = np.random.default_rng(config.seed)
    capacity = problem.k_max
    k_trace = np.empty(config.iterations + 1, dtype=np.int64)
    nuclei_trace = np.empty((config.iterations + 1, capacity), dtype=np.int64)
    coefficient_trace = np.empty((config.iterations + 1, capacity), dtype=np.float64)
    log_target_trace = np.empty(config.iterations + 1, dtype=np.float64)
    moves = np.empty(config.iterations, dtype="U16")
    accepted = np.zeros(config.iterations, dtype=np.bool_)
    log_acceptance_ratio = np.empty(config.iterations, dtype=np.float64)

    state = initial_state
    k_trace[0] = state.k
    nuclei_trace[0] = state.nuclei
    coefficient_trace[0] = state.coefficients
    log_target_trace[0] = state.log_target
    nucleus_move = "global_move" if config.nucleus_move == "global" else "local_move"
    schedule = ("coefficient", "dimension", "dimension", nucleus_move)

    for iteration in range(config.iterations):
        move = schedule[iteration % len(schedule)]
        transition = _draw_transition(problem, state, config, rng, move)
        uniform = float(rng.random())
        log_uniform = log(uniform) if uniform > 0.0 else -np.inf
        next_state = accept_or_reject(state, transition, log_uniform=log_uniform)
        accepted[iteration] = transition.valid and next_state is transition.candidate
        moves[iteration] = transition.move
        log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
        state = next_state
        k_trace[iteration + 1] = state.k
        nuclei_trace[iteration + 1] = state.nuclei
        coefficient_trace[iteration + 1] = state.coefficients
        log_target_trace[iteration + 1] = state.log_target

    return SamplingResult(
        trace=SamplingTrace(
            k=k_trace,
            nuclei=nuclei_trace,
            coefficients=coefficient_trace,
            log_target=log_target_trace,
            moves=moves,
            accepted=accepted,
            log_acceptance_ratio=log_acceptance_ratio,
        ),
        final_state=state,
    )


__all__ = ["SamplerConfig", "SamplingResult", "SamplingTrace", "sample"]
