"""Deterministic proposal accounting for active-only Gamma--Beta trees.

This module implements local split and sibling-merge reversible-jump moves for
the root-plus-active-fraction coordinates defined in
:mod:`openghg_inversions.experimental.rjmcmc.gamma_beta_tree`. Every random
choice is supplied explicitly, so proposal construction is deterministic and
pointwise reverse terms can be tested without depending on a random-number
generator. Structural moves record normalized direction, eligible-node
selection, and Beta auxiliary densities separately. Coordinate insertion and
deletion have a unit Jacobian in this parameterization.

Independent-prior refreshes of the root total and one selected active fraction
are provided as separate fixed-dimensional kernels.  Their proposal terms
retain the normalized prior densities even though those terms cancel the
matching target-prior changes in the Metropolis--Hastings ratio.

The main entry points are :func:`propose_split`, :func:`propose_merge`,
:func:`propose_root_refresh`, and :func:`propose_fraction_refresh`; each
returns immutable :class:`GammaBetaTransitionTerms`. :func:`accept_or_reject`
applies an explicitly supplied log-uniform draw. Kernels never mutate their
source state, and invalid proposals retain that exact source object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Literal

import numpy as np

from .dyadic_tree import DyadicFrontier
from .gamma_beta_tree import (
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    build_gamma_beta_tree_state,
)


GammaBetaMove = Literal["split", "merge", "root_refresh", "fraction_refresh"]
"""Literal name of a supported Gamma--Beta proposal move."""


@dataclass(frozen=True, slots=True)
class GammaBetaTransitionTerms:
    """A Gamma--Beta candidate and its decomposed Metropolis--Hastings terms.

    Attributes:
        candidate: Candidate tree state. Invalid proposals retain the source
            state object.
        delta_log_likelihood: Candidate-minus-source likelihood target
            component, including any configured likelihood power.
        delta_log_root_prior: Candidate-minus-source normalized root Gamma log
            density.
        delta_log_fraction_prior: Candidate-minus-source sum of normalized
            active-fraction Beta log densities.
        delta_log_partition_prior: Candidate-minus-source normalized frontier
            log probability.
        log_q_forward_direction: Log probability of selecting the forward move
            direction. Fixed-dimensional refreshes use zero.
        log_q_forward_selection: Log probability of selecting the forward
            structural node or active fraction. Root refreshes use zero.
        log_q_forward_auxiliary: Forward normalized independent-prior density.
            A merge has no forward auxiliary and uses zero.
        log_q_reverse_direction: Log probability of selecting the pointwise
            reverse direction.
        log_q_reverse_selection: Log probability of selecting the pointwise
            reverse structural node or active fraction.
        log_q_reverse_auxiliary: Reverse normalized independent-prior density.
            A split has no reverse auxiliary and uses zero.
        log_jacobian: Log absolute Jacobian determinant. Structural
            insertion/deletion in root-plus-fraction coordinates uses zero.
        move: Stable kernel name.
        node_id: Selected structural or active-split node, a negative sentinel
            for an unavailable structural direction, or ``None`` for a root
            refresh.
        valid: Whether the proposal is eligible for Metropolis--Hastings
            acceptance.
        reason: Explanation for an invalid self-transition, otherwise
            ``None``.
        log_target_delta: Sum of the four target-component changes.
        log_q_forward: Sum of the three forward proposal components.
        log_q_reverse: Sum of the three reverse proposal components.
        log_acceptance_ratio: Complete untruncated log
            Metropolis--Hastings ratio. Invalid transitions use negative
            infinity.

    Raises:
        TypeError: If candidate or transition metadata have the wrong type.
        ValueError: If move metadata or any log-ratio component is malformed.
    """

    candidate: GammaBetaTreeState
    delta_log_likelihood: float
    delta_log_root_prior: float
    delta_log_fraction_prior: float
    delta_log_partition_prior: float
    log_q_forward_direction: float
    log_q_forward_selection: float
    log_q_forward_auxiliary: float
    log_q_reverse_direction: float
    log_q_reverse_selection: float
    log_q_reverse_auxiliary: float
    log_jacobian: float
    move: GammaBetaMove
    node_id: int | None
    valid: bool = True
    reason: str | None = None
    log_target_delta: float = field(init=False)
    log_q_forward: float = field(init=False)
    log_q_reverse: float = field(init=False)
    log_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate metadata and calculate all aggregate log terms.

        Scalar components are normalized to Python floats and ``node_id`` is
        normalized to an integer. The aggregate target, forward proposal,
        reverse proposal, and acceptance-ratio fields are then assigned on the
        otherwise frozen instance.

        Raises:
            TypeError: If candidate, node, or validity metadata have the wrong
                type.
            ValueError: If move/validity metadata are inconsistent or any
                supplied or calculated log term is NaN.
        """
        if not isinstance(self.candidate, GammaBetaTreeState):
            raise TypeError("candidate must be a GammaBetaTreeState.")
        if self.move not in ("split", "merge", "root_refresh", "fraction_refresh"):
            raise ValueError("move must name a Gamma--Beta proposal kernel.")
        if self.node_id is not None:
            if isinstance(self.node_id, bool) or not isinstance(self.node_id, Integral):
                raise TypeError("node_id must be an integer or None.")
            object.__setattr__(self, "node_id", int(self.node_id))
        if self.move == "root_refresh" and self.node_id is not None:
            raise ValueError("a root refresh cannot select a node.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a Boolean.")
        if self.valid and self.reason is not None:
            raise ValueError("a valid transition cannot have an invalidity reason.")
        if not self.valid and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("an invalid transition must provide a non-empty reason.")

        target_components = (
            float(self.delta_log_likelihood),
            float(self.delta_log_root_prior),
            float(self.delta_log_fraction_prior),
            float(self.delta_log_partition_prior),
        )
        forward_components = (
            float(self.log_q_forward_direction),
            float(self.log_q_forward_selection),
            float(self.log_q_forward_auxiliary),
        )
        reverse_components = (
            float(self.log_q_reverse_direction),
            float(self.log_q_reverse_selection),
            float(self.log_q_reverse_auxiliary),
        )
        log_jacobian = float(self.log_jacobian)
        components = target_components + forward_components + reverse_components + (log_jacobian,)
        if any(math.isnan(value) for value in components):
            raise ValueError("transition log terms cannot be NaN.")

        field_names = (
            "delta_log_likelihood",
            "delta_log_root_prior",
            "delta_log_fraction_prior",
            "delta_log_partition_prior",
            "log_q_forward_direction",
            "log_q_forward_selection",
            "log_q_forward_auxiliary",
            "log_q_reverse_direction",
            "log_q_reverse_selection",
            "log_q_reverse_auxiliary",
        )
        for name, value in zip(field_names, target_components + forward_components + reverse_components):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "log_jacobian", log_jacobian)

        log_target_delta = sum(target_components)
        log_q_forward = sum(forward_components)
        log_q_reverse = sum(reverse_components)
        log_acceptance_ratio = (
            log_target_delta + log_q_reverse - log_q_forward + log_jacobian if self.valid else -math.inf
        )
        aggregates = (
            log_target_delta,
            log_q_forward,
            log_q_reverse,
            log_acceptance_ratio,
        )
        if any(math.isnan(value) for value in aggregates):
            raise ValueError("calculated transition log terms cannot be NaN.")
        object.__setattr__(self, "log_target_delta", log_target_delta)
        object.__setattr__(self, "log_q_forward", log_q_forward)
        object.__setattr__(self, "log_q_reverse", log_q_reverse)
        object.__setattr__(self, "log_acceptance_ratio", log_acceptance_ratio)

    @property
    def log_q_forward_fraction(self) -> float:
        """Return the forward auxiliary-density term.

        Returns:
            Alias of ``log_q_forward_auxiliary``; zero when the forward
            direction introduces no auxiliary coordinate.
        """
        return self.log_q_forward_auxiliary

    @property
    def log_q_reverse_fraction(self) -> float:
        """Return the reverse auxiliary-density term.

        Returns:
            Alias of ``log_q_reverse_auxiliary``; zero when the reverse
            direction introduces no auxiliary coordinate.
        """
        return self.log_q_reverse_auxiliary


def _validate_problem_state(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
) -> None:
    """Validate the exact problem identity and active-fraction alignment.

    Args:
        problem: Candidate fixed-tree problem.
        state: Candidate active-frontier state.

    Raises:
        TypeError: If either argument has the wrong type.
        ValueError: If the state belongs to another problem instance or its
            fraction count does not match its active split ancestors.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(state, GammaBetaTreeState):
        raise TypeError("state must be a GammaBetaTreeState.")
    if state.problem is not problem:
        raise ValueError("state must have been built for problem.")
    if not math.isfinite(state.log_target):
        raise ValueError("source state must have finite target support.")
    node_ids = state.frontier.active_split_nodes(problem.tree)
    if len(node_ids) != len(state.active_fractions):
        raise ValueError("state active fractions do not align with its frontier.")


def _validate_direction_probability(value: float) -> float:
    """Normalize one finite split-direction probability.

    Args:
        value: Candidate probability.

    Returns:
        Python float strictly between zero and one.

    Raises:
        TypeError: If ``value`` is Boolean.
        ValueError: If ``value`` cannot be converted to float or lies outside
            the open unit interval.
    """
    if isinstance(value, bool):
        raise TypeError("split_direction_probability must be a real number.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("split_direction_probability must lie strictly between zero and one.")
    return probability


def _node_id_or_none(value: object) -> int | None:
    """Return an integer node ID or ``None`` for a non-integer choice."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _open_unit_value(value: float) -> float | None:
    """Return a finite value strictly inside the unit interval or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0.0 < result < 1.0 else None


def _positive_value(value: float) -> float | None:
    """Return a finite positive value or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _active_fraction_mapping(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
) -> dict[int, float]:
    """Associate canonical active split-node IDs with their fractions.

    Args:
        problem: Problem whose fixed tree defines canonical node order.
        state: Previously validated state whose fractions align with active
            split ancestors.

    Returns:
        Mutable mapping from stable node ID to Python float.
    """
    node_ids = state.frontier.active_split_nodes(problem.tree)
    return {
        node_id: float(fraction) for node_id, fraction in zip(node_ids, state.active_fractions, strict=True)
    }


def _ordered_fractions(
    problem: GammaBetaTreeProblem,
    frontier: DyadicFrontier,
    values_by_node: dict[int, float],
) -> np.ndarray:
    """Order a node-keyed fraction mapping for one candidate frontier.

    Args:
        problem: Problem whose fixed tree defines canonical node order.
        frontier: Candidate frontier.
        values_by_node: Fraction values keyed by active split-node ID.

    Returns:
        One-dimensional ``float64`` array in canonical active-split order.

    Raises:
        KeyError: If ``values_by_node`` omits an active split ancestor.
    """
    return np.array(
        [values_by_node[node_id] for node_id in frontier.active_split_nodes(problem.tree)],
        dtype=np.float64,
    )


def _invalid_transition(
    state: GammaBetaTreeState,
    *,
    move: GammaBetaMove,
    node_id: int | None,
    reason: str,
) -> GammaBetaTransitionTerms:
    """Construct a rejected-by-definition self-transition.

    Args:
        state: Source state retained by identity as the candidate.
        move: Attempted kernel name.
        node_id: Attempted node or sentinel.
        reason: Non-empty invalidity explanation.

    Returns:
        Immutable transition with zero target, proposal, and Jacobian terms
        and negative-infinite acceptance ratio.
    """
    return GammaBetaTransitionTerms(
        candidate=state,
        delta_log_likelihood=0.0,
        delta_log_root_prior=0.0,
        delta_log_fraction_prior=0.0,
        delta_log_partition_prior=0.0,
        log_q_forward_direction=0.0,
        log_q_forward_selection=0.0,
        log_q_forward_auxiliary=0.0,
        log_q_reverse_direction=0.0,
        log_q_reverse_selection=0.0,
        log_q_reverse_auxiliary=0.0,
        log_jacobian=0.0,
        move=move,
        node_id=node_id,
        valid=False,
        reason=reason,
    )


def _valid_transition(
    source: GammaBetaTreeState,
    candidate: GammaBetaTreeState,
    *,
    move: GammaBetaMove,
    node_id: int | None,
    log_q_forward_direction: float = 0.0,
    log_q_forward_selection: float = 0.0,
    log_q_forward_auxiliary: float = 0.0,
    log_q_reverse_direction: float = 0.0,
    log_q_reverse_selection: float = 0.0,
    log_q_reverse_auxiliary: float = 0.0,
) -> GammaBetaTransitionTerms:
    """Construct complete valid accounting from source and candidate caches.

    Args:
        source: Source state.
        candidate: Fully rebuilt candidate state.
        move: Proposal kernel name.
        node_id: Selected structural or active-fraction node.
        log_q_forward_direction: Forward direction log probability.
        log_q_forward_selection: Forward node-selection log probability.
        log_q_forward_auxiliary: Forward independent-prior log density.
        log_q_reverse_direction: Reverse direction log probability.
        log_q_reverse_selection: Reverse node-selection log probability.
        log_q_reverse_auxiliary: Reverse independent-prior log density.

    Returns:
        Immutable transition containing cached target-component differences
        and the supplied proposal decomposition. The log Jacobian is exactly
        zero in root-plus-fraction coordinates.
    """
    return GammaBetaTransitionTerms(
        candidate=candidate,
        delta_log_likelihood=candidate.log_likelihood - source.log_likelihood,
        delta_log_root_prior=candidate.log_root_prior - source.log_root_prior,
        delta_log_fraction_prior=candidate.log_fraction_prior - source.log_fraction_prior,
        delta_log_partition_prior=candidate.log_partition_prior - source.log_partition_prior,
        log_q_forward_direction=log_q_forward_direction,
        log_q_forward_selection=log_q_forward_selection,
        log_q_forward_auxiliary=log_q_forward_auxiliary,
        log_q_reverse_direction=log_q_reverse_direction,
        log_q_reverse_selection=log_q_reverse_selection,
        log_q_reverse_auxiliary=log_q_reverse_auxiliary,
        log_jacobian=0.0,
        move=move,
        node_id=node_id,
    )


def propose_split(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    leaf_node_id: int,
    new_fraction: float,
    split_direction_probability: float = 0.5,
) -> GammaBetaTransitionTerms:
    """Propose splitting one selected active leaf with an explicit fraction.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        leaf_node_id: Stable node ID selected uniformly from the source
            splittable active leaves.
        new_fraction: Explicit proposed allocation fraction in ``(0, 1)``.
        split_direction_probability: Probability of choosing the split
            direction at each structural opportunity.

    Returns:
        Immutable proposal accounting. An unavailable direction, ineligible
        node, or out-of-support fraction produces an invalid self-transition.

    Raises:
        TypeError: If problem, state, or direction probability has the wrong
            type.
        ValueError: If the problem/state pairing or direction probability is
            invalid.
    """
    _validate_problem_state(problem, state)
    probability = _validate_direction_probability(split_direction_probability)
    selected_node_id = _node_id_or_none(leaf_node_id)
    splittable = problem.tree.splittable_nodes(state.frontier)
    if not splittable:
        return _invalid_transition(
            state,
            move="split",
            node_id=selected_node_id,
            reason="the source frontier has no splittable active leaves",
        )
    if selected_node_id is None or selected_node_id not in splittable:
        return _invalid_transition(
            state,
            move="split",
            node_id=selected_node_id,
            reason="leaf_node_id is not a splittable active leaf",
        )
    fraction = _open_unit_value(new_fraction)
    if fraction is None:
        return _invalid_transition(
            state,
            move="split",
            node_id=selected_node_id,
            reason="new_fraction must lie strictly between zero and one",
        )

    candidate_frontier = state.frontier.split(problem.tree, selected_node_id)
    fractions_by_node = _active_fraction_mapping(problem, state)
    fractions_by_node[selected_node_id] = fraction
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=candidate_frontier,
        root_total=state.root_total,
        active_fractions=_ordered_fractions(
            problem,
            candidate_frontier,
            fractions_by_node,
        ),
    )
    reverse_candidates = problem.tree.mergeable_parents(candidate.frontier)
    log_beta_density = problem.prior.log_fraction_density(selected_node_id, fraction)
    return _valid_transition(
        state,
        candidate,
        move="split",
        node_id=selected_node_id,
        log_q_forward_direction=math.log(probability),
        log_q_forward_selection=-math.log(len(splittable)),
        log_q_forward_auxiliary=log_beta_density,
        log_q_reverse_direction=math.log1p(-probability),
        log_q_reverse_selection=-math.log(len(reverse_candidates)),
    )


def propose_merge(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    parent_node_id: int,
    split_direction_probability: float = 0.5,
) -> GammaBetaTransitionTerms:
    """Propose merging one selected active cherry and remove its fraction.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        parent_node_id: Stable parent ID selected uniformly from the source
            mergeable cherries.
        split_direction_probability: Probability of choosing the split
            direction at each structural opportunity.

    Returns:
        Immutable proposal accounting. An unavailable direction or non-cherry
        node produces an invalid self-transition.

    Raises:
        TypeError: If problem, state, or direction probability has the wrong
            type.
        ValueError: If the problem/state pairing or direction probability is
            invalid.
    """
    _validate_problem_state(problem, state)
    probability = _validate_direction_probability(split_direction_probability)
    selected_node_id = _node_id_or_none(parent_node_id)
    mergeable = problem.tree.mergeable_parents(state.frontier)
    if not mergeable:
        return _invalid_transition(
            state,
            move="merge",
            node_id=selected_node_id,
            reason="the source frontier has no mergeable cherry parents",
        )
    if selected_node_id is None or selected_node_id not in mergeable:
        return _invalid_transition(
            state,
            move="merge",
            node_id=selected_node_id,
            reason="parent_node_id is not a mergeable cherry parent",
        )

    fractions_by_node = _active_fraction_mapping(problem, state)
    removed_fraction = fractions_by_node.pop(selected_node_id)
    candidate_frontier = state.frontier.merge(problem.tree, selected_node_id)
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=candidate_frontier,
        root_total=state.root_total,
        active_fractions=_ordered_fractions(
            problem,
            candidate_frontier,
            fractions_by_node,
        ),
    )
    reverse_candidates = problem.tree.splittable_nodes(candidate.frontier)
    log_beta_density = problem.prior.log_fraction_density(
        selected_node_id,
        removed_fraction,
    )
    return _valid_transition(
        state,
        candidate,
        move="merge",
        node_id=selected_node_id,
        log_q_forward_direction=math.log1p(-probability),
        log_q_forward_selection=-math.log(len(mergeable)),
        log_q_reverse_direction=math.log(probability),
        log_q_reverse_selection=-math.log(len(reverse_candidates)),
        log_q_reverse_auxiliary=log_beta_density,
    )


def propose_root_refresh(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    new_root_total: float,
) -> GammaBetaTransitionTerms:
    """Propose an explicit root total from its independent Gamma prior.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        new_root_total: Explicit positive root-total proposal.

    Returns:
        Immutable fixed-dimensional transition. Values outside the root prior
        support produce invalid self-transitions.

    Raises:
        TypeError: If problem or state has the wrong type.
        ValueError: If the problem/state pairing is invalid.
    """
    _validate_problem_state(problem, state)
    root_total = _positive_value(new_root_total)
    if root_total is None:
        return _invalid_transition(
            state,
            move="root_refresh",
            node_id=None,
            reason="new_root_total must be finite and positive",
        )
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=state.frontier,
        root_total=root_total,
        active_fractions=state.active_fractions,
    )
    return _valid_transition(
        state,
        candidate,
        move="root_refresh",
        node_id=None,
        log_q_forward_auxiliary=problem.prior.log_root_density(root_total),
        log_q_reverse_auxiliary=problem.prior.log_root_density(state.root_total),
    )


def propose_fraction_refresh(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    split_node_id: int,
    new_fraction: float,
) -> GammaBetaTransitionTerms:
    """Refresh one uniformly selected active fraction from its Beta prior.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        split_node_id: Stable split-node ID selected uniformly from all active
            fractions.
        new_fraction: Explicit independent Beta-prior proposal in ``(0, 1)``.

    Returns:
        Immutable fixed-dimensional transition. A state without active
        fractions, an inactive node, or an out-of-support fraction produces an
        invalid self-transition.

    Raises:
        TypeError: If problem or state has the wrong type.
        ValueError: If the problem/state pairing is invalid.
    """
    _validate_problem_state(problem, state)
    selected_node_id = _node_id_or_none(split_node_id)
    active_node_ids = state.frontier.active_split_nodes(problem.tree)
    if not active_node_ids:
        return _invalid_transition(
            state,
            move="fraction_refresh",
            node_id=selected_node_id,
            reason="the source frontier has no active fractions",
        )
    if selected_node_id is None or selected_node_id not in active_node_ids:
        return _invalid_transition(
            state,
            move="fraction_refresh",
            node_id=selected_node_id,
            reason="split_node_id does not identify an active fraction",
        )
    fraction = _open_unit_value(new_fraction)
    if fraction is None:
        return _invalid_transition(
            state,
            move="fraction_refresh",
            node_id=selected_node_id,
            reason="new_fraction must lie strictly between zero and one",
        )

    fractions_by_node = _active_fraction_mapping(problem, state)
    old_fraction = fractions_by_node[selected_node_id]
    fractions_by_node[selected_node_id] = fraction
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=state.frontier,
        root_total=state.root_total,
        active_fractions=_ordered_fractions(
            problem,
            state.frontier,
            fractions_by_node,
        ),
    )
    log_selection = -math.log(len(active_node_ids))
    return _valid_transition(
        state,
        candidate,
        move="fraction_refresh",
        node_id=selected_node_id,
        log_q_forward_selection=log_selection,
        log_q_forward_auxiliary=problem.prior.log_fraction_density(
            selected_node_id,
            fraction,
        ),
        log_q_reverse_selection=log_selection,
        log_q_reverse_auxiliary=problem.prior.log_fraction_density(
            selected_node_id,
            old_fraction,
        ),
    )


def accept_or_reject(
    state: GammaBetaTreeState,
    transition: GammaBetaTransitionTerms,
    *,
    log_uniform: float,
) -> GammaBetaTreeState:
    """Select a candidate using an explicitly supplied log-uniform draw.

    Args:
        state: Source state to retain on rejection.
        transition: Gamma--Beta proposal accounting and candidate state.
        log_uniform: Natural logarithm of a draw in ``(0, 1]``. Values must be
            non-positive; negative infinity is accepted as the limiting value.

    Returns:
        ``transition.candidate`` when ``log_uniform`` is strictly below the
        truncated log acceptance ratio, otherwise the unchanged source state.

    Raises:
        TypeError: If state or transition has the wrong type, or
            ``log_uniform`` cannot be converted to float.
        ValueError: If ``log_uniform`` cannot be parsed as a float, is NaN, or
            is positive.
    """
    if not isinstance(state, GammaBetaTreeState):
        raise TypeError("state must be a GammaBetaTreeState.")
    if not isinstance(transition, GammaBetaTransitionTerms):
        raise TypeError("transition must be a GammaBetaTransitionTerms instance.")
    log_uniform = float(log_uniform)
    if math.isnan(log_uniform) or log_uniform > 0.0:
        raise ValueError("log_uniform must be non-positive and cannot be NaN.")
    threshold = min(0.0, transition.log_acceptance_ratio)
    if transition.valid and log_uniform < threshold:
        return transition.candidate
    return state


__all__ = [
    "GammaBetaMove",
    "GammaBetaTransitionTerms",
    "accept_or_reject",
    "propose_fraction_refresh",
    "propose_merge",
    "propose_root_refresh",
    "propose_split",
]
