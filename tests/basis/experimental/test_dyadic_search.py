"""Tests for the experimental stochastic local-search runner."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.search import (
    PiecewiseGeometricSchedule,
    SearchProposal,
    SearchResult,
    stochastic_local_search,
)
from openghg_inversions.basis.experimental.dyadic.initializers import random_partition
from openghg_inversions.basis.experimental.dyadic.proposals import PairedMove, enumerate_paired_neighbors
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_piecewise_schedule_holds_cools_and_polishes() -> None:
    """The schedule should expose its three configured temperature phases."""
    schedule = PiecewiseGeometricSchedule(
        initial_temperature=10.0,
        final_temperature=0.1,
        hold_fraction=0.2,
        polish_fraction=0.2,
    )

    temperatures = [schedule(index, 10) for index in range(10)]

    assert temperatures[:2] == [10.0, 10.0]
    assert temperatures[-2:] == [0.0, 0.0]
    assert all(left >= right for left, right in zip(temperatures, temperatures[1:]))
    assert temperatures[2] == pytest.approx(10.0)
    assert temperatures[7] == pytest.approx(0.1)


def test_zero_temperature_search_tracks_best_without_accepting_losses() -> None:
    """A zero-temperature search should keep improvements and reject losses."""
    candidates = iter([1, 0, 3, 2])

    def propose(state: int, rng: np.random.Generator) -> SearchProposal[int, str] | None:
        """Return the next deterministic integer candidate."""
        del state, rng
        try:
            candidate = next(candidates)
        except StopIteration:
            return None
        return SearchProposal(candidate, f"to-{candidate}")

    result = stochastic_local_search(
        0,
        objective=float,
        propose=propose,
        schedule=lambda iteration, total: 0.0,
        iterations=4,
        rng=np.random.default_rng(1),
    )

    assert result.final_state == 3
    assert result.best_state == 3
    assert result.best_score == 3.0
    assert result.accepted_moves == 2
    assert [step.accepted for step in result.trace] == [True, False, True, False]


def test_positive_temperature_search_is_seeded_and_reproducible() -> None:
    """The same random seed should reproduce every accepted state."""

    def run(seed: int) -> SearchResult[int, int]:
        """Run a random-walk optimizer for one seed."""

        def propose(state: int, rng: np.random.Generator) -> SearchProposal[int, int]:
            """Propose a unit random-walk move."""
            move = int(rng.choice([-1, 1]))
            return SearchProposal(state + move, move)

        return stochastic_local_search(
            0,
            objective=lambda state: -abs(state - 2),
            propose=propose,
            schedule=lambda iteration, total: 2.0,
            iterations=20,
            rng=np.random.default_rng(seed),
            record_every=3,
        )

    first = run(42)
    second = run(42)

    assert first == second
    assert first.best_score >= first.initial_score
    assert len(first.trace) <= first.evaluated_moves


def test_fixed_count_dyadic_moves_compose_with_search_runner() -> None:
    """Paired partition proposals should preserve a valid fixed-size frontier."""
    tree = DyadicTree.from_shape((4, 4))
    initial = random_partition(tree, target_regions=6, rng=np.random.default_rng(8)).state

    def propose(
        state: PartitionState,
        rng: np.random.Generator,
    ) -> SearchProposal[PartitionState, PairedMove] | None:
        """Sample uniformly from the state's unique paired neighbors."""
        neighbors = enumerate_paired_neighbors(tree, state)
        if not neighbors:
            return None
        neighbor = neighbors[int(rng.integers(len(neighbors)))]
        return SearchProposal(neighbor.state, neighbor.move)

    result = stochastic_local_search(
        initial,
        objective=lambda state: -float(sum(state.active)),
        propose=propose,
        schedule=lambda iteration, total: 0.5,
        iterations=30,
        rng=np.random.default_rng(9),
    )

    result.final_state.validate(tree)
    result.best_state.validate(tree)
    assert len(result.final_state.active) == len(initial.active) == 6
    assert result.best_score >= result.initial_score


def test_search_stops_when_no_proposal_is_available() -> None:
    """A missing proposal should stop cleanly without evaluating a move."""
    result = stochastic_local_search(
        "root",
        objective=lambda state: 1.0,
        propose=lambda state, rng: None,
        schedule=lambda iteration, total: 0.0,
        iterations=5,
        rng=np.random.default_rng(2),
    )

    assert result.stop_reason == "no_proposal"
    assert result.evaluated_moves == 0
    assert result.final_state == "root"
    assert result.trace == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": -1}, "iterations"),
        ({"record_every": 0}, "record_every"),
    ],
)
def test_search_rejects_invalid_configuration(kwargs: dict[str, int], message: str) -> None:
    """Invalid bounds should fail before any proposal is evaluated."""
    options = {"iterations": 1, "record_every": 1}
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        stochastic_local_search(
            0,
            objective=float,
            propose=lambda state, rng: SearchProposal(state, None),
            schedule=lambda iteration, total: 0.0,
            rng=np.random.default_rng(3),
            **options,
        )


def test_search_rejects_non_finite_scores_and_temperatures() -> None:
    """Non-finite numerical inputs should not enter the optimizer trace."""
    with pytest.raises(ValueError, match="objective"):
        stochastic_local_search(
            0,
            objective=lambda state: np.nan,
            propose=lambda state, rng: SearchProposal(state, None),
            schedule=lambda iteration, total: 0.0,
            iterations=1,
            rng=np.random.default_rng(4),
        )

    with pytest.raises(ValueError, match="temperature"):
        stochastic_local_search(
            0,
            objective=float,
            propose=lambda state, rng: SearchProposal(state, None),
            schedule=lambda iteration, total: np.inf,
            iterations=1,
            rng=np.random.default_rng(5),
        )
