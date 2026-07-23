"""Deterministic proposal accounting for active-only Gamma--Beta trees.

This module implements local split and sibling-merge reversible-jump moves,
plus fixed-dimensional relocation and bounded subtree-retile moves, for the
root-plus-active-fraction coordinates defined in
:mod:`openghg_inversions.experimental.rjmcmc.gamma_beta_tree`. Every random
choice is supplied explicitly, so proposal construction is deterministic and
pointwise reverse terms can be tested without depending on a random-number
generator. Structural moves record normalized direction, eligible-node
selection, and Beta auxiliary densities separately. Coordinate insertion,
deletion, and replacement have a unit Jacobian in this parameterization.

Independent-prior refreshes of the root total and one selected active fraction
are provided as separate fixed-dimensional kernels.  Their proposal terms
retain the normalized prior densities even though those terms cancel the
matching target-prior changes in the Metropolis--Hastings ratio.

The main entry points are :func:`propose_split`, :func:`propose_merge`,
:func:`propose_relocate`, :func:`propose_subtree_retile`,
:func:`propose_root_refresh`, :func:`propose_fraction_refresh`, and
:func:`propose_fixed_coefficient`; each returns immutable
:class:`GammaBetaTransitionTerms`. :func:`accept_or_reject` applies an
explicitly supplied log-uniform draw. Kernels never mutate their source state,
and invalid proposals retain that exact source object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Literal

import numpy as np

from .dyadic_tree import DyadicFrontier, SubtreePartitionIndex
from .gamma_beta_tree import (
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    build_gamma_beta_tree_state,
)


GammaBetaMove = Literal[
    "split",
    "merge",
    "relocate",
    "subtree_retile",
    "root_refresh",
    "fraction_refresh",
    "fixed_coefficient",
]
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
        delta_log_fixed_coefficient_prior: Candidate-minus-source normalized
            fixed-block coefficient log-prior density.
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
            or fixed-coefficient refresh.
        secondary_node_id: Destination split node for a relocation, otherwise
            ``None``.
        block_leaf_count: Number of active regions inside a selected subtree
            retile block, otherwise ``None``.
        coefficient_id: Selected fixed-block coefficient position, or
            ``None`` for every other kernel.
        valid: Whether the proposal is eligible for Metropolis--Hastings
            acceptance.
        reason: Explanation for an invalid self-transition, otherwise
            ``None``.
        log_target_delta: Sum of the five target-component changes.
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
    delta_log_fixed_coefficient_prior: float = 0.0
    coefficient_id: int | None = None
    secondary_node_id: int | None = None
    block_leaf_count: int | None = None
    log_target_delta: float = field(init=False)
    log_q_forward: float = field(init=False)
    log_q_reverse: float = field(init=False)
    log_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate metadata and calculate all aggregate log terms.

        Scalar components are normalized to Python floats. ``node_id`` and
        ``coefficient_id`` are normalized to integers when present. The
        aggregate target, forward proposal, reverse proposal, and
        acceptance-ratio fields are then assigned on the otherwise frozen
        instance.

        Raises:
            TypeError: If candidate, node, or validity metadata have the wrong
                type.
            ValueError: If move/validity metadata are inconsistent or any
                supplied or calculated log term is NaN.
        """
        if not isinstance(self.candidate, GammaBetaTreeState):
            raise TypeError("candidate must be a GammaBetaTreeState.")
        if self.move not in (
            "split",
            "merge",
            "relocate",
            "subtree_retile",
            "root_refresh",
            "fraction_refresh",
            "fixed_coefficient",
        ):
            raise ValueError("move must name a Gamma--Beta proposal kernel.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a Boolean.")
        if self.node_id is not None:
            if isinstance(self.node_id, bool) or not isinstance(self.node_id, Integral):
                raise TypeError("node_id must be an integer or None.")
            object.__setattr__(self, "node_id", int(self.node_id))
        if self.secondary_node_id is not None:
            if isinstance(self.secondary_node_id, bool) or not isinstance(
                self.secondary_node_id,
                Integral,
            ):
                raise TypeError("secondary_node_id must be an integer or None.")
            object.__setattr__(self, "secondary_node_id", int(self.secondary_node_id))
        if self.block_leaf_count is not None:
            if isinstance(self.block_leaf_count, bool) or not isinstance(
                self.block_leaf_count,
                Integral,
            ):
                raise TypeError("block_leaf_count must be an integer or None.")
            if self.block_leaf_count < 1:
                raise ValueError("block_leaf_count must be positive when present.")
            object.__setattr__(self, "block_leaf_count", int(self.block_leaf_count))
        if self.move == "root_refresh" and self.node_id is not None:
            raise ValueError("a root refresh cannot select a node.")
        if self.coefficient_id is not None:
            if isinstance(self.coefficient_id, bool) or not isinstance(
                self.coefficient_id,
                Integral,
            ):
                raise TypeError("coefficient_id must be an integer or None.")
            object.__setattr__(self, "coefficient_id", int(self.coefficient_id))
        if self.move == "fixed_coefficient":
            if self.node_id is not None or self.coefficient_id is None:
                raise ValueError("a fixed-coefficient refresh must select only a coefficient.")
        elif self.coefficient_id is not None:
            raise ValueError("only a fixed-coefficient refresh can select a coefficient.")
        if self.move == "relocate":
            if self.valid and (
                self.node_id is None
                or self.secondary_node_id is None
                or self.node_id == self.secondary_node_id
            ):
                raise ValueError("a valid relocation requires distinct source and destination nodes.")
            if self.block_leaf_count is not None:
                raise ValueError("a relocation cannot select a subtree block size.")
        elif self.secondary_node_id is not None:
            raise ValueError("only a relocation can select a secondary node.")
        if self.move == "subtree_retile":
            if self.valid and (self.node_id is None or self.block_leaf_count is None):
                raise ValueError("a valid subtree retile requires a block node and leaf count.")
        elif self.block_leaf_count is not None:
            raise ValueError("only a subtree retile can record a block leaf count.")
        if self.valid and self.reason is not None:
            raise ValueError("a valid transition cannot have an invalidity reason.")
        if not self.valid and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("an invalid transition must provide a non-empty reason.")

        target_components = (
            float(self.delta_log_likelihood),
            float(self.delta_log_root_prior),
            float(self.delta_log_fraction_prior),
            float(self.delta_log_partition_prior),
            float(self.delta_log_fixed_coefficient_prior),
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
            "delta_log_fixed_coefficient_prior",
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


def _positive_scale(value: float, *, name: str) -> float:
    """Return a finite positive proposal scale.

    Args:
        value: Candidate scalar.
        name: Public argument name used in validation errors.

    Returns:
        Normalized Python float.

    Raises:
        TypeError: If ``value`` is Boolean.
        ValueError: If conversion fails or the value is not finite and
            strictly positive.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _normal_log_density(value: float, *, mean: float, stdev: float) -> float:
    """Return a normalized univariate Gaussian log density.

    Args:
        value: Point at which to evaluate the density.
        mean: Gaussian mean.
        stdev: Positive Gaussian standard deviation, validated by the caller.

    Returns:
        Natural logarithm of the normalized Gaussian density.
    """
    standardized = (value - mean) / stdev
    return float(-0.5 * standardized * standardized - math.log(stdev) - 0.5 * math.log(2.0 * math.pi))


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
    coefficient_id: int | None = None,
    secondary_node_id: int | None = None,
    block_leaf_count: int | None = None,
    reason: str,
) -> GammaBetaTransitionTerms:
    """Construct a rejected-by-definition self-transition.

    Args:
        state: Source state retained by identity as the candidate.
        move: Attempted kernel name.
        node_id: Attempted node or sentinel.
        coefficient_id: Attempted fixed-block coefficient position.
        secondary_node_id: Attempted relocation destination node.
        block_leaf_count: Selected subtree block region count.
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
        delta_log_fixed_coefficient_prior=0.0,
        log_q_forward_direction=0.0,
        log_q_forward_selection=0.0,
        log_q_forward_auxiliary=0.0,
        log_q_reverse_direction=0.0,
        log_q_reverse_selection=0.0,
        log_q_reverse_auxiliary=0.0,
        log_jacobian=0.0,
        move=move,
        node_id=node_id,
        coefficient_id=coefficient_id,
        secondary_node_id=secondary_node_id,
        block_leaf_count=block_leaf_count,
        valid=False,
        reason=reason,
    )


def _valid_transition(
    source: GammaBetaTreeState,
    candidate: GammaBetaTreeState,
    *,
    move: GammaBetaMove,
    node_id: int | None,
    coefficient_id: int | None = None,
    secondary_node_id: int | None = None,
    block_leaf_count: int | None = None,
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
        coefficient_id: Selected fixed-block coefficient position.
        secondary_node_id: Selected relocation destination node.
        block_leaf_count: Selected subtree block region count.
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
        delta_log_fixed_coefficient_prior=(
            candidate.log_fixed_coefficient_prior - source.log_fixed_coefficient_prior
        ),
        log_q_forward_direction=log_q_forward_direction,
        log_q_forward_selection=log_q_forward_selection,
        log_q_forward_auxiliary=log_q_forward_auxiliary,
        log_q_reverse_direction=log_q_reverse_direction,
        log_q_reverse_selection=log_q_reverse_selection,
        log_q_reverse_auxiliary=log_q_reverse_auxiliary,
        log_jacobian=0.0,
        move=move,
        node_id=node_id,
        coefficient_id=coefficient_id,
        secondary_node_id=secondary_node_id,
        block_leaf_count=block_leaf_count,
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
        fixed_coefficients=state.fixed_coefficients,
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
        fixed_coefficients=state.fixed_coefficients,
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


def propose_relocate(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    merge_parent_node_id: int,
    split_leaf_node_id: int,
    new_fraction: float,
) -> GammaBetaTransitionTerms:
    """Relocate one fixed-``K`` split by a sequential merge-then-split move.

    A cherry parent ``a`` is selected uniformly and merged. A different
    splittable leaf ``b`` is then selected uniformly from the intermediate
    frontier, excluding ``a``, and split using the explicit proposed
    fraction. The reverse move first merges ``b`` and then splits ``a``.
    Both sequential selection probabilities and both normalized Beta
    auxiliary densities are retained explicitly.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        merge_parent_node_id: Selected source cherry parent ``a``.
        split_leaf_node_id: Selected intermediate splittable leaf ``b``.
        new_fraction: Explicit allocation fraction introduced at ``b``.

    Returns:
        Immutable fixed-``K`` proposal accounting. Ineligible choices or an
        out-of-support fraction produce an invalid self-transition.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state pairing is invalid.
    """
    _validate_problem_state(problem, state)
    source_node_id = _node_id_or_none(merge_parent_node_id)
    destination_node_id = _node_id_or_none(split_leaf_node_id)
    mergeable = problem.tree.mergeable_parents(state.frontier)
    if not mergeable:
        return _invalid_transition(
            state,
            move="relocate",
            node_id=source_node_id,
            secondary_node_id=destination_node_id,
            reason="the source frontier has no mergeable cherry parents",
        )
    if source_node_id is None or source_node_id not in mergeable:
        return _invalid_transition(
            state,
            move="relocate",
            node_id=source_node_id,
            secondary_node_id=destination_node_id,
            reason="merge_parent_node_id is not a mergeable cherry parent",
        )

    intermediate = state.frontier.merge(problem.tree, source_node_id)
    destinations = tuple(
        node_id for node_id in problem.tree.splittable_nodes(intermediate) if node_id != source_node_id
    )
    if not destinations:
        return _invalid_transition(
            state,
            move="relocate",
            node_id=source_node_id,
            secondary_node_id=destination_node_id,
            reason="the intermediate frontier has no different splittable destination",
        )
    if destination_node_id is None or destination_node_id not in destinations:
        return _invalid_transition(
            state,
            move="relocate",
            node_id=source_node_id,
            secondary_node_id=destination_node_id,
            reason="split_leaf_node_id is not an eligible different destination",
        )
    fraction = _open_unit_value(new_fraction)
    if fraction is None:
        return _invalid_transition(
            state,
            move="relocate",
            node_id=source_node_id,
            secondary_node_id=destination_node_id,
            reason="new_fraction must lie strictly between zero and one",
        )

    fractions_by_node = _active_fraction_mapping(problem, state)
    removed_fraction = fractions_by_node.pop(source_node_id)
    fractions_by_node[destination_node_id] = fraction
    candidate_frontier = intermediate.split(problem.tree, destination_node_id)
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=candidate_frontier,
        root_total=state.root_total,
        active_fractions=_ordered_fractions(
            problem,
            candidate_frontier,
            fractions_by_node,
        ),
        fixed_coefficients=state.fixed_coefficients,
    )
    reverse_mergeable = problem.tree.mergeable_parents(candidate_frontier)
    reverse_intermediate = candidate_frontier.merge(problem.tree, destination_node_id)
    reverse_destinations = tuple(
        node_id
        for node_id in problem.tree.splittable_nodes(reverse_intermediate)
        if node_id != destination_node_id
    )
    if reverse_intermediate != intermediate or source_node_id not in reverse_destinations:
        raise RuntimeError("Internal relocation reverse support mismatch.")
    return _valid_transition(
        state,
        candidate,
        move="relocate",
        node_id=source_node_id,
        secondary_node_id=destination_node_id,
        log_q_forward_selection=-math.log(len(mergeable)) - math.log(len(destinations)),
        log_q_forward_auxiliary=problem.prior.log_fraction_density(
            destination_node_id,
            fraction,
        ),
        log_q_reverse_selection=-math.log(len(reverse_mergeable)) - math.log(len(reverse_destinations)),
        log_q_reverse_auxiliary=problem.prior.log_fraction_density(
            source_node_id,
            removed_fraction,
        ),
    )


def _subtree_frontier(
    problem: GammaBetaTreeProblem,
    frontier: DyadicFrontier,
    block_node_id: int,
) -> DyadicFrontier:
    """Return the active leaves contained inside one canonical subtree."""
    block = problem.tree.node(block_node_id)
    return DyadicFrontier(
        tuple(
            node_id
            for node_id in frontier.node_ids
            if (
                problem.tree.nodes[node_id].row_start >= block.row_start
                and problem.tree.nodes[node_id].row_stop <= block.row_stop
                and problem.tree.nodes[node_id].col_start >= block.col_start
                and problem.tree.nodes[node_id].col_stop <= block.col_stop
            )
        )
    )


def _eligible_retile_blocks_unchecked(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    index: SubtreePartitionIndex,
) -> tuple[tuple[int, int], ...]:
    """Return active split blocks with at least one same-size alternative."""
    eligible: list[tuple[int, int]] = []
    for node_id in state.frontier.active_split_nodes(problem.tree):
        subtree = _subtree_frontier(problem, state.frontier, node_id)
        block_k = len(subtree)
        if block_k <= index.max_k and index.count(node_id, block_k) > 1:
            eligible.append((node_id, block_k))
    return tuple(eligible)


def eligible_subtree_retile_blocks(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    index: SubtreePartitionIndex,
) -> tuple[tuple[int, int], ...]:
    """Return normalized subtree-retile block candidates as ``(node, m)``.

    A block is eligible when it is an active split node, its current subtree
    frontier has ``m <= index.max_k`` regions, and the index contains at least
    one different exact-``m`` frontier. The compound sampler should select
    uniformly from this tuple and persist ``index.max_k`` as part of its
    kernel configuration.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        index: Exact subtree index built for ``problem.tree``.

    Returns:
        Stable node-ID-order tuple of eligible block IDs and active region
        counts.

    Raises:
        TypeError: If a public argument has the wrong type.
        ValueError: If the problem/state or index/tree pairing is invalid.
    """
    _validate_problem_state(problem, state)
    if not isinstance(index, SubtreePartitionIndex):
        raise TypeError("index must be a SubtreePartitionIndex.")
    if index.tree is not problem.tree:
        raise ValueError("index must have been built for problem.tree.")
    return _eligible_retile_blocks_unchecked(problem, state, index)


def _explicit_fraction_mapping(
    values: Mapping[int, float],
    expected_node_ids: frozenset[int],
) -> dict[int, float] | None:
    """Normalize an exact node-keyed proposed-fraction mapping."""
    normalized: dict[int, float] = {}
    for node_id, value in values.items():
        selected_node_id = _node_id_or_none(node_id)
        fraction = _open_unit_value(value)
        if selected_node_id is None or fraction is None or selected_node_id in normalized:
            return None
        normalized[selected_node_id] = fraction
    if frozenset(normalized) != expected_node_ids:
        return None
    return normalized


def propose_subtree_retile(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    index: SubtreePartitionIndex,
    *,
    block_node_id: int,
    replacement_frontier: DyadicFrontier,
    new_fractions_by_node: Mapping[int, float],
) -> GammaBetaTransitionTerms:
    """Replace one active subtree partition by a different same-size tiling.

    Eligible active split blocks are selected uniformly. Conditional on a
    block containing ``m`` active regions, one of the other exact-``m``
    subtree frontiers is selected uniformly using ``index``. Fractions for
    active split nodes common to source and candidate are retained; the
    caller supplies explicit Beta-prior values for every newly active split
    node. The pointwise reverse density evaluates the removed source values.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        index: Exact subtree partition count/rank index for ``problem.tree``.
        block_node_id: Explicit eligible active split block.
        replacement_frontier: Explicit different exact-``m`` frontier that
            covers the selected subtree.
        new_fractions_by_node: Exact mapping from every newly active split
            node to its explicit proposed fraction.

    Returns:
        Immutable fixed-``K`` proposal accounting with the block's active
        region count recorded in ``block_leaf_count``.

    Raises:
        TypeError: If public object arguments have the wrong type.
        ValueError: If the problem/state or index/tree pairing is invalid.
    """
    _validate_problem_state(problem, state)
    if not isinstance(index, SubtreePartitionIndex):
        raise TypeError("index must be a SubtreePartitionIndex.")
    if index.tree is not problem.tree:
        raise ValueError("index must have been built for problem.tree.")
    if not isinstance(replacement_frontier, DyadicFrontier):
        raise TypeError("replacement_frontier must be a DyadicFrontier.")
    if not isinstance(new_fractions_by_node, Mapping):
        raise TypeError("new_fractions_by_node must be a mapping.")

    selected_node_id = _node_id_or_none(block_node_id)
    eligible = _eligible_retile_blocks_unchecked(problem, state, index)
    eligible_by_node = dict(eligible)
    if selected_node_id is None or selected_node_id not in eligible_by_node:
        return _invalid_transition(
            state,
            move="subtree_retile",
            node_id=selected_node_id,
            reason="block_node_id is not an eligible active split block",
        )
    block_k = eligible_by_node[selected_node_id]
    source_subtree = _subtree_frontier(
        problem,
        state.frontier,
        selected_node_id,
    )
    try:
        source_rank = index.rank(selected_node_id, block_k, source_subtree)
        replacement_rank = index.rank(
            selected_node_id,
            block_k,
            replacement_frontier,
        )
    except (KeyError, TypeError, ValueError):
        return _invalid_transition(
            state,
            move="subtree_retile",
            node_id=selected_node_id,
            block_leaf_count=block_k,
            reason="replacement_frontier must exactly cover the block with the same size",
        )
    if replacement_rank == source_rank:
        return _invalid_transition(
            state,
            move="subtree_retile",
            node_id=selected_node_id,
            block_leaf_count=block_k,
            reason="replacement_frontier must differ from the source subtree frontier",
        )

    source_subtree_ids = frozenset(source_subtree.node_ids)
    candidate_frontier = DyadicFrontier(
        tuple(node_id for node_id in state.frontier.node_ids if node_id not in source_subtree_ids)
        + replacement_frontier.node_ids
    )
    candidate_frontier.validate(problem.tree)
    source_fractions = _active_fraction_mapping(problem, state)
    candidate_split_nodes = frozenset(candidate_frontier.active_split_nodes(problem.tree))
    source_split_nodes = frozenset(source_fractions)
    added_nodes = candidate_split_nodes - source_split_nodes
    removed_nodes = source_split_nodes - candidate_split_nodes
    proposed_fractions = _explicit_fraction_mapping(
        new_fractions_by_node,
        added_nodes,
    )
    if proposed_fractions is None:
        return _invalid_transition(
            state,
            move="subtree_retile",
            node_id=selected_node_id,
            block_leaf_count=block_k,
            reason="new_fractions_by_node must exactly cover added nodes with open-unit values",
        )
    candidate_fractions = {
        node_id: value for node_id, value in source_fractions.items() if node_id in candidate_split_nodes
    }
    candidate_fractions.update(proposed_fractions)
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=candidate_frontier,
        root_total=state.root_total,
        active_fractions=_ordered_fractions(
            problem,
            candidate_frontier,
            candidate_fractions,
        ),
        fixed_coefficients=state.fixed_coefficients,
    )
    reverse_eligible = _eligible_retile_blocks_unchecked(problem, candidate, index)
    reverse_eligible_nodes = frozenset(node_id for node_id, _ in reverse_eligible)
    if selected_node_id not in reverse_eligible_nodes:
        raise RuntimeError("Internal subtree-retile reverse block support mismatch.")
    alternatives = index.count(selected_node_id, block_k) - 1
    return _valid_transition(
        state,
        candidate,
        move="subtree_retile",
        node_id=selected_node_id,
        block_leaf_count=block_k,
        log_q_forward_selection=-math.log(len(eligible)) - math.log(alternatives),
        log_q_forward_auxiliary=sum(
            problem.prior.log_fraction_density(node_id, proposed_fractions[node_id])
            for node_id in sorted(added_nodes)
        ),
        log_q_reverse_selection=-math.log(len(reverse_eligible)) - math.log(alternatives),
        log_q_reverse_auxiliary=sum(
            problem.prior.log_fraction_density(node_id, source_fractions[node_id])
            for node_id in sorted(removed_nodes)
        ),
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
        fixed_coefficients=state.fixed_coefficients,
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
        fixed_coefficients=state.fixed_coefficients,
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


def propose_fixed_coefficient(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    proposal_stdev: float,
) -> GammaBetaTransitionTerms:
    """Propose one deterministic-position fixed coefficient by Gaussian walk.

    The caller supplies the position because the compound schedule visits each
    configured fixed coefficient exactly once per cycle. The untruncated
    Gaussian proposal is symmetric, but both normalized directional densities
    are retained in the transition decomposition for auditability. A draw
    outside the positive coefficient support is an explicit invalid
    self-transition.

    Args:
        problem: Fixed-tree Gamma--Beta problem.
        state: Source active-frontier state.
        coefficient_position: Zero-based deterministic fixed-block position.
        proposed_coefficient: Explicit Gaussian random-walk proposal.
        proposal_stdev: Finite positive Gaussian standard deviation.

    Returns:
        Immutable fixed-dimensional proposal accounting.

    Raises:
        TypeError: If problem, state, position, or scale has the wrong type.
        ValueError: If the problem/state pairing or proposal scale is invalid.
    """
    _validate_problem_state(problem, state)
    scale = _positive_scale(proposal_stdev, name="proposal_stdev")
    if isinstance(coefficient_position, bool) or not isinstance(
        coefficient_position,
        Integral,
    ):
        raise TypeError("coefficient_position must be an integer.")
    position = int(coefficient_position)
    if position < 0 or position >= state.fixed_coefficients.size:
        return _invalid_transition(
            state,
            move="fixed_coefficient",
            node_id=None,
            coefficient_id=position,
            reason="coefficient_position must select a configured fixed coefficient",
        )
    value = _positive_value(proposed_coefficient)
    if value is None:
        return _invalid_transition(
            state,
            move="fixed_coefficient",
            node_id=None,
            coefficient_id=position,
            reason="proposed_coefficient must be finite and positive",
        )

    current = float(state.fixed_coefficients[position])
    fixed_coefficients = np.array(state.fixed_coefficients, copy=True)
    fixed_coefficients[position] = value
    candidate = build_gamma_beta_tree_state(
        problem,
        frontier=state.frontier,
        root_total=state.root_total,
        active_fractions=state.active_fractions,
        fixed_coefficients=fixed_coefficients,
    )
    return _valid_transition(
        state,
        candidate,
        move="fixed_coefficient",
        node_id=None,
        coefficient_id=position,
        log_q_forward_auxiliary=_normal_log_density(
            value,
            mean=current,
            stdev=scale,
        ),
        log_q_reverse_auxiliary=_normal_log_density(
            current,
            mean=value,
            stdev=scale,
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
    "eligible_subtree_retile_blocks",
    "propose_fraction_refresh",
    "propose_fixed_coefficient",
    "propose_merge",
    "propose_relocate",
    "propose_root_refresh",
    "propose_split",
    "propose_subtree_retile",
]
