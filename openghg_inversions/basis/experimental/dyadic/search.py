"""Generic stochastic local search for experimental dyadic partitions.

The runner in this module is an optimizer. Its temperature-based acceptance
rule is not a Metropolis-Hastings posterior transition because it does not
include a target density or forward/reverse proposal probabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp
from typing import Generic, Protocol, TypeVar

import numpy as np


StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT")


class TemperatureSchedule(Protocol):
    """Return the optimizer temperature for one zero-based iteration."""

    def __call__(self, iteration: int, total_iterations: int, /) -> float:
        """Return a finite non-negative temperature."""
        ...


@dataclass(frozen=True)
class PiecewiseGeometricSchedule:
    """Hold, geometrically cool, and finish with zero-temperature polishing.

    Args:
        initial_temperature: Temperature during the initial hold period.
        final_temperature: Positive temperature at the end of geometric cooling.
        hold_fraction: Fraction of iterations spent at the initial temperature.
        polish_fraction: Fraction of iterations spent at zero temperature.
    """

    initial_temperature: float
    final_temperature: float
    hold_fraction: float = 0.1
    polish_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Validate schedule parameters."""
        if not np.isfinite(self.initial_temperature) or self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive and finite.")
        if not np.isfinite(self.final_temperature) or self.final_temperature <= 0.0:
            raise ValueError("final_temperature must be positive and finite.")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must not exceed initial_temperature.")
        if not 0.0 <= self.hold_fraction < 1.0:
            raise ValueError("hold_fraction must be in [0, 1).")
        if not 0.0 <= self.polish_fraction < 1.0:
            raise ValueError("polish_fraction must be in [0, 1).")
        if self.hold_fraction + self.polish_fraction >= 1.0:
            raise ValueError("hold_fraction and polish_fraction must sum to less than 1.")

    def __call__(self, iteration: int, total_iterations: int) -> float:
        """Return the temperature for one iteration.

        Args:
            iteration: Zero-based iteration index.
            total_iterations: Total number of configured search iterations.

        Returns:
            Initial, geometrically cooled, or zero polishing temperature.

        Raises:
            ValueError: If the iteration bounds are invalid.
        """
        if total_iterations < 1:
            raise ValueError("total_iterations must be positive.")
        if iteration < 0 or iteration >= total_iterations:
            raise ValueError("iteration must be within total_iterations.")

        hold_steps = int(total_iterations * self.hold_fraction)
        polish_steps = int(total_iterations * self.polish_fraction)
        cool_stop = total_iterations - polish_steps
        if iteration < hold_steps:
            return self.initial_temperature
        if iteration >= cool_stop:
            return 0.0

        cool_steps = max(cool_stop - hold_steps, 1)
        progress = (iteration - hold_steps) / max(cool_steps - 1, 1)
        ratio = self.final_temperature / self.initial_temperature
        return float(self.initial_temperature * ratio**progress)


@dataclass(frozen=True)
class SearchProposal(Generic[StateT, MoveT]):
    """One candidate state and the move metadata that produced it."""

    state: StateT
    move: MoveT


@dataclass(frozen=True)
class SearchStep(Generic[StateT, MoveT]):
    """Diagnostics for one evaluated search proposal."""

    iteration: int
    temperature: float
    candidate_score: float
    current_score: float
    best_score: float
    accepted: bool
    new_best: bool
    current_state: StateT
    move: MoveT


@dataclass(frozen=True)
class SearchResult(Generic[StateT, MoveT]):
    """Initial, final, and best states from stochastic local search."""

    initial_state: StateT
    final_state: StateT
    best_state: StateT
    initial_score: float
    final_score: float
    best_score: float
    accepted_moves: int
    evaluated_moves: int
    trace: tuple[SearchStep[StateT, MoveT], ...]
    stop_reason: str


ProposalFunction = Callable[[StateT, np.random.Generator], SearchProposal[StateT, MoveT] | None]
ObjectiveFunction = Callable[[StateT], float]


def stochastic_local_search(
    initial_state: StateT,
    *,
    objective: ObjectiveFunction[StateT],
    propose: ProposalFunction[StateT, MoveT],
    schedule: TemperatureSchedule,
    iterations: int,
    rng: np.random.Generator,
    record_every: int = 1,
) -> SearchResult[StateT, MoveT]:
    """Maximize an objective with temperature-controlled local proposals.

    Args:
        initial_state: Immutable state from which the search starts.
        objective: Callable returning a finite score to maximize.
        propose: Callable returning a candidate state and move metadata. Returning
            None stops the search because no proposal is available.
        schedule: Temperature schedule used for accepting score decreases.
        iterations: Maximum number of proposal evaluations.
        rng: Random generator owned by the caller.
        record_every: Record every Nth evaluated proposal. Accepted moves, new
            best states, and the final iteration are always recorded.

    Returns:
        Search result containing final/best states and a bounded trace.

    Raises:
        ValueError: If configuration or objective values are invalid.

    Notes:
        The acceptance probability for a score decrease loss is
        exp(-loss / temperature). Proposal probabilities are deliberately
        absent; this function performs optimization, not posterior sampling.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if record_every < 1:
        raise ValueError("record_every must be positive.")

    current_state = initial_state
    current_score = _finite_score(objective(current_state))
    initial_score = current_score
    best_state = current_state
    best_score = current_score
    accepted_moves = 0
    evaluated_moves = 0
    trace: list[SearchStep[StateT, MoveT]] = []
    stop_reason = "iteration_limit"

    for iteration in range(iterations):
        proposal = propose(current_state, rng)
        if proposal is None:
            stop_reason = "no_proposal"
            break

        temperature = float(schedule(iteration, iterations))
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("schedule returned an invalid temperature.")

        candidate_score = _finite_score(objective(proposal.state))
        score_change = candidate_score - current_score
        accepted = score_change >= 0.0
        if not accepted and temperature > 0.0:
            accepted = bool(rng.random() < exp(score_change / temperature))

        evaluated_moves += 1
        if accepted:
            current_state = proposal.state
            current_score = candidate_score
            accepted_moves += 1

        new_best = current_score > best_score
        if new_best:
            best_state = current_state
            best_score = current_score

        should_record = (
            evaluated_moves % record_every == 0 or accepted or new_best or iteration == iterations - 1
        )
        if should_record:
            trace.append(
                SearchStep(
                    iteration=iteration,
                    temperature=temperature,
                    candidate_score=candidate_score,
                    current_score=current_score,
                    best_score=best_score,
                    accepted=accepted,
                    new_best=new_best,
                    current_state=current_state,
                    move=proposal.move,
                )
            )

    return SearchResult(
        initial_state=initial_state,
        final_state=current_state,
        best_state=best_state,
        initial_score=initial_score,
        final_score=current_score,
        best_score=best_score,
        accepted_moves=accepted_moves,
        evaluated_moves=evaluated_moves,
        trace=tuple(trace),
        stop_reason=stop_reason,
    )


def _finite_score(score: float) -> float:
    """Return a finite floating score."""
    value = float(score)
    if not np.isfinite(value):
        raise ValueError("objective must return a finite score.")
    return value


__all__ = [
    "PiecewiseGeometricSchedule",
    "SearchProposal",
    "SearchResult",
    "SearchStep",
    "TemperatureSchedule",
    "stochastic_local_search",
]
