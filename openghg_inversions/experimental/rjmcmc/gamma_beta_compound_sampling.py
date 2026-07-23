"""Full-posterior compound sampler for the fixed-tree Gamma--Beta model.

This module keeps the structural-only mobility oracle in
:mod:`openghg_inversions.experimental.rjmcmc.gamma_beta_sampling` unchanged
and provides the separate posterior sampler needed for synthetic and real-data
experiments. One versioned cycle contains two independently mixed split/merge
opportunities, configurable fixed-``K`` relocation and bounded-subtree retile
opportunities (one each by default), one independent-prior root-total refresh,
a configurable number of uniformly selected independent-prior active-fraction
refreshes (five by default), and one deterministic Gaussian random-walk slot
for each always-active fixed coefficient.

Every atomic slot consumes an acceptance uniform, including unavailable
structural directions and fraction refreshes at ``K=1``. Structural choices
respect the contiguous positive support of the declared marginal ``p(K)``;
selecting an unavailable direction at a support boundary is an explicit
self-transition. In-memory checkpoints preserve the exact PCG64 stream,
global retention coordinate, and schedule phase, including continuation from
the middle of a cycle.

The implementation is a correctness-first NumPy reference. It fully rebuilds
candidate states through the immutable target API and is not yet a Numba
performance kernel. Durable checkpoint and trace serialization are provided
separately by :mod:`openghg_inversions.experimental.rjmcmc.gamma_beta_io`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dyadic_tree import DyadicFrontier, SubtreePartitionIndex
from .gamma_beta_proposals import (
    GammaBetaTransitionTerms,
    accept_or_reject,
    eligible_subtree_retile_blocks,
    propose_fixed_coefficient,
    propose_fraction_refresh,
    propose_merge,
    propose_relocate,
    propose_root_refresh,
    propose_split,
    propose_subtree_retile,
)
from .gamma_beta_tree import GammaBetaTreeProblem, GammaBetaTreeState
from .retention import RetentionSettings
from .sampling import PCG64State


GAMMA_BETA_COMPOUND_SCHEDULE_ID = (
    "gamma_beta_2_mixed_structure_n_relocate_n_subtree_retile_1_root_n_fraction_fixed_sweep_v2"
)
"""Versioned identifier for the posterior compound schedule."""

CompoundSlot = Literal[
    "structural",
    "relocation",
    "subtree_retile",
    "root",
    "fraction",
    "fixed",
]
"""Atomic Gamma--Beta compound-schedule slot kind."""


def _readonly_vector(
    values: object,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    name: str,
) -> np.ndarray:
    """Return an owned read-only one-dimensional array.

    Args:
        values: Values convertible to a NumPy array.
        dtype: Required NumPy dtype.
        name: Public field name used in validation errors.

    Returns:
        Owned read-only one-dimensional array.

    Raises:
        ValueError: If the converted array is not one-dimensional.
    """
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    array.setflags(write=False)
    return array


def _readonly_matrix(
    values: object,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    name: str,
) -> np.ndarray:
    """Return an owned read-only two-dimensional array.

    Args:
        values: Values convertible to a NumPy array.
        dtype: Required NumPy dtype.
        name: Public field name used in validation errors.

    Returns:
        Owned read-only two-dimensional array.

    Raises:
        ValueError: If the converted array is not two-dimensional.
    """
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    array.setflags(write=False)
    return array


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite positive real sampler setting.

    Args:
        value: Candidate scalar.
        name: Public setting name used in validation errors.

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
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.ndim != 0:
            raise ValueError
        result = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive.") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _normalize_fixed_scale_input(
    value: float | ArrayLike,
) -> float | tuple[float, ...]:
    """Normalize a scalar or one-dimensional fixed proposal-scale setting.

    Args:
        value: Positive scalar or one-dimensional positive values.

    Returns:
        A positive scalar or immutable tuple of positive scales.

    Raises:
        TypeError: If a scalar or vector entry is Boolean.
        ValueError: If the input is not scalar or one-dimensional, or any
            scale is not finite and positive.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return _positive_float(array.item(), name="fixed_coefficient_proposal_sd")
    if array.ndim != 1:
        raise ValueError("fixed_coefficient_proposal_sd must be scalar or one-dimensional.")
    return tuple(_positive_float(item, name="fixed_coefficient_proposal_sd") for item in array.tolist())


def _resolve_fixed_scales(
    value: float | tuple[float, ...],
    *,
    n_fixed: int,
) -> tuple[float, ...]:
    """Resolve a scalar or vector scale against the fixed-block width.

    Args:
        value: Normalized shared scalar or vector of per-position scales.
        n_fixed: Number of fixed coefficients in the target problem.

    Returns:
        Immutable tuple with one scale per fixed coefficient.

    Raises:
        ValueError: If a supplied vector does not match ``n_fixed``.
    """
    if not isinstance(value, tuple):
        return (float(value),) * n_fixed
    if len(value) != n_fixed:
        raise ValueError("fixed_coefficient_proposal_sd vector must have one entry per fixed coefficient.")
    return value


def _positive_k_support(
    problem: GammaBetaTreeProblem,
    *,
    relocation_configured: bool,
    subtree_retile_max_leaves: int | None,
) -> tuple[int, int]:
    """Return and validate the contiguous positive support of marginal ``p(K)``.

    Args:
        problem: Gamma--Beta target containing the normalized partition prior.
        relocation_configured: Whether the schedule includes at least one
            relocation opportunity.
        subtree_retile_max_leaves: Retile cap when the schedule includes a
            subtree-retile opportunity, otherwise ``None``.

    Returns:
        Inclusive smallest and largest positive-mass region counts.

    Raises:
        ValueError: If no region count has positive mass or the positive
            support contains a gap.
    """
    probabilities = problem.partition_prior.marginal_probability_by_k
    positive = np.flatnonzero(probabilities > 0.0)
    if positive.size == 0:
        raise ValueError("partition prior must assign positive mass to at least one K.")
    lower = int(positive[0])
    upper = int(positive[-1])
    if not np.array_equal(positive, np.arange(lower, upper + 1)):
        raise ValueError("compound sampling requires contiguous positive p(K) support.")
    fixed_k_topology_is_effective = relocation_configured or (
        subtree_retile_max_leaves is not None and subtree_retile_max_leaves >= lower
    )
    if (
        lower == upper
        and problem.partition_prior.partition_counts[lower] > 1
        and not (fixed_k_topology_is_effective)
    ):
        raise ValueError(
            "singleton p(K) support has multiple frontiers but the compound "
            "schedule has no effective fixed-K topology move."
        )
    return lower, upper


@dataclass(frozen=True, slots=True)
class GammaBetaCompoundConfig:
    """Configuration for a fresh Gamma--Beta posterior segment.

    Args:
        iterations: Positive number of atomic transitions.
        seed: Non-negative PCG64 seed or ``None``.
        split_direction_probability: Probability of selecting split inside
            either mixed structural slot.
        fraction_refresh_slots: Non-negative number of independently selected
            active-fraction refreshes per cycle. Five matches the Lunt-style
            dynamic-coefficient opportunity count.
        relocation_slots: Non-negative number of fixed-``K`` cherry relocation
            opportunities per cycle.
        subtree_retile_slots: Non-negative number of fixed-``K`` bounded
            subtree-retile opportunities per cycle.
        max_subtree_leaves: Positive maximum active-leaf count in a subtree
            eligible for exact retile ranking and unranking.
        fixed_coefficient_proposal_sd: Positive scalar shared by all fixed
            coefficients or a positive one-dimensional vector resolved against
            the problem at sampling time.

    Raises:
        TypeError: If integer or probability settings have invalid types.
        ValueError: If a setting lies outside its supported range.
    """

    iterations: int
    seed: int | None = None
    split_direction_probability: float = 0.5
    fraction_refresh_slots: int = 5
    fixed_coefficient_proposal_sd: float | tuple[float, ...] = 0.4
    relocation_slots: int = 1
    subtree_retile_slots: int = 1
    max_subtree_leaves: int = 8

    def __post_init__(self) -> None:
        """Normalize and validate problem-independent sampler settings."""
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
        if isinstance(self.split_direction_probability, bool):
            raise TypeError("split_direction_probability must be a real number.")
        probability = float(self.split_direction_probability)
        if not isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("split_direction_probability must lie strictly between zero and one.")
        for name in (
            "fraction_refresh_slots",
            "relocation_slots",
            "subtree_retile_slots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        if isinstance(self.max_subtree_leaves, bool) or not isinstance(
            self.max_subtree_leaves,
            Integral,
        ):
            raise TypeError("max_subtree_leaves must be an integer.")
        if self.max_subtree_leaves < 1:
            raise ValueError("max_subtree_leaves must be positive.")
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "split_direction_probability", probability)
        object.__setattr__(self, "max_subtree_leaves", int(self.max_subtree_leaves))
        object.__setattr__(
            self,
            "fixed_coefficient_proposal_sd",
            _normalize_fixed_scale_input(self.fixed_coefficient_proposal_sd),
        )


@dataclass(frozen=True, slots=True)
class GammaBetaCompoundKernelSettings:
    """Immutable problem-resolved compound schedule settings.

    Attributes:
        split_direction_probability: Split probability in either structural
            slot.
        fraction_refresh_slots: Number of fraction slots per cycle.
        relocation_slots: Number of fixed-``K`` relocation slots per cycle.
        subtree_retile_slots: Number of bounded subtree-retile slots per cycle.
        max_subtree_leaves: Maximum active leaves in a selected retile block.
        fixed_coefficient_proposal_sd: Positive Gaussian scales in
            deterministic fixed-coefficient slot order.

    Raises:
        TypeError: If probability or slot-count settings have invalid types.
        ValueError: If probability, slot count, or any scale lies outside its
            supported range.
    """

    split_direction_probability: float
    fraction_refresh_slots: int
    fixed_coefficient_proposal_sd: tuple[float, ...]
    relocation_slots: int = 1
    subtree_retile_slots: int = 1
    max_subtree_leaves: int = 8

    def __post_init__(self) -> None:
        """Validate immutable resolved schedule settings."""
        if isinstance(self.split_direction_probability, bool):
            raise TypeError("split_direction_probability must be a real number.")
        probability = float(self.split_direction_probability)
        if not isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("split_direction_probability must lie strictly between zero and one.")
        for name in (
            "fraction_refresh_slots",
            "relocation_slots",
            "subtree_retile_slots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        if isinstance(self.max_subtree_leaves, bool) or not isinstance(
            self.max_subtree_leaves,
            Integral,
        ):
            raise TypeError("max_subtree_leaves must be an integer.")
        if self.max_subtree_leaves < 1:
            raise ValueError("max_subtree_leaves must be positive.")
        scales = tuple(
            _positive_float(value, name="fixed_coefficient_proposal_sd")
            for value in self.fixed_coefficient_proposal_sd
        )
        object.__setattr__(self, "split_direction_probability", probability)
        object.__setattr__(self, "max_subtree_leaves", int(self.max_subtree_leaves))
        object.__setattr__(self, "fixed_coefficient_proposal_sd", scales)

    @property
    def cycle_length(self) -> int:
        """Return the number of atomic transitions in one complete cycle.

        Returns:
            Two structural slots, configured relocation and retile slots, one
            root slot, configured fraction slots, and one slot per fixed
            coefficient.
        """
        return (
            3
            + self.relocation_slots
            + self.subtree_retile_slots
            + self.fraction_refresh_slots
            + len(self.fixed_coefficient_proposal_sd)
        )

    @property
    def has_fixed_k_topology_kernel(self) -> bool:
        """Return whether at least one fixed-``K`` topology slot is configured."""
        return self.relocation_slots > 0 or self.subtree_retile_slots > 0


@dataclass(frozen=True, slots=True)
class GammaBetaCompoundTrace:
    """Retained posterior states and every attempted atomic transition.

    Variable-dimensional frontiers and split fractions use immutable tuples.
    Fixed-width retained and every-attempt diagnostics are owned read-only
    arrays. ``global_transition`` is one-based, whereas ``state_transition``
    uses completed-transition coordinates and may include zero for the initial
    state.

    Attributes:
        frontiers: Retained canonical active frontiers.
        split_fractions: Retained read-only fraction vectors, each with shape
            ``(K-1,)`` in canonical active-split order.
        root_total: Retained positive root totals, shape ``(n_retained,)``.
        fixed_coefficients: Retained positive fixed coefficients, shape
            ``(n_retained, n_fixed)``.
        k: Retained active-region counts, shape ``(n_retained,)``.
        log_gaussian_likelihood: Retained raw normalized Gaussian log
            likelihoods.
        log_likelihood: Retained likelihood-power-scaled target components.
        log_root_prior: Retained normalized root Gamma log densities.
        log_fraction_prior: Retained normalized active Beta log densities.
        log_partition_prior: Retained normalized frontier log probabilities.
        log_fixed_coefficient_prior: Retained normalized fixed-block lognormal
            log densities.
        log_target: Retained complete log targets.
        state_transition: Global completed-transition coordinates of retained
            states.
        global_transition: One-based global coordinate for every attempted
            transition.
        slot: Compound slot kind for every attempt.
        move: Concrete proposal kernel name for every attempt.
        valid: Whether each attempted proposal was eligible for MH acceptance.
        accepted: Whether each attempt changed the visited state.
        node_id: Selected structural or fraction node, with ``-1`` when
            unavailable or not applicable.
        secondary_node_id: Selected relocation destination node, with ``-1``
            when unavailable or not applicable.
        block_leaf_count: Selected subtree-retile leaf count, with ``-1`` when
            unavailable or not applicable.
        coefficient_id: Selected fixed coefficient, with ``-1`` when not
            applicable.
        k_before: Source active-region count for every attempt.
        k_after: Visited active-region count after every attempt.
        log_acceptance_ratio: Raw untruncated MH log ratio for every attempt;
            invalid stays store negative infinity.

    Raises:
        TypeError: If a retained frontier has the wrong type.
        ValueError: If retained or attempted arrays violate shapes, ordering,
            finiteness, support, sentinel, or acceptance invariants.
    """

    frontiers: tuple[DyadicFrontier, ...]
    split_fractions: tuple[NDArray[np.float64], ...]
    root_total: NDArray[np.float64]
    fixed_coefficients: NDArray[np.float64]
    k: NDArray[np.int64]
    log_gaussian_likelihood: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    log_root_prior: NDArray[np.float64]
    log_fraction_prior: NDArray[np.float64]
    log_partition_prior: NDArray[np.float64]
    log_fixed_coefficient_prior: NDArray[np.float64]
    log_target: NDArray[np.float64]
    state_transition: NDArray[np.int64]
    global_transition: NDArray[np.int64]
    slot: NDArray[np.str_]
    move: NDArray[np.str_]
    valid: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    node_id: NDArray[np.int64]
    secondary_node_id: NDArray[np.int64]
    block_leaf_count: NDArray[np.int64]
    coefficient_id: NDArray[np.int64]
    k_before: NDArray[np.int64]
    k_after: NDArray[np.int64]
    log_acceptance_ratio: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own arrays and enforce retained and attempted axis contracts."""
        frontiers = tuple(self.frontiers)
        if any(not isinstance(frontier, DyadicFrontier) for frontier in frontiers):
            raise TypeError("frontiers must contain DyadicFrontier values.")
        object.__setattr__(self, "frontiers", frontiers)
        fractions: list[NDArray[np.float64]] = []
        for values in self.split_fractions:
            fraction = _readonly_vector(
                values,
                dtype=np.float64,
                name="split_fractions entry",
            )
            if np.any(~np.isfinite(fraction)) or np.any((fraction <= 0.0) | (fraction >= 1.0)):
                raise ValueError("retained split fractions must lie strictly between zero and one.")
            fractions.append(fraction)
        object.__setattr__(self, "split_fractions", tuple(fractions))

        retained_vectors = {
            "root_total": np.float64,
            "k": np.int64,
            "log_gaussian_likelihood": np.float64,
            "log_likelihood": np.float64,
            "log_root_prior": np.float64,
            "log_fraction_prior": np.float64,
            "log_partition_prior": np.float64,
            "log_fixed_coefficient_prior": np.float64,
            "log_target": np.float64,
            "state_transition": np.int64,
        }
        for name, dtype in retained_vectors.items():
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=dtype, name=name),
            )
        object.__setattr__(
            self,
            "fixed_coefficients",
            _readonly_matrix(
                self.fixed_coefficients,
                dtype=np.float64,
                name="fixed_coefficients",
            ),
        )
        attempted_vectors = {
            "global_transition": np.int64,
            "slot": np.dtype("U14"),
            "move": np.dtype("U17"),
            "valid": np.bool_,
            "accepted": np.bool_,
            "node_id": np.int64,
            "secondary_node_id": np.int64,
            "block_leaf_count": np.int64,
            "coefficient_id": np.int64,
            "k_before": np.int64,
            "k_after": np.int64,
            "log_acceptance_ratio": np.float64,
        }
        for name, dtype in attempted_vectors.items():
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=dtype, name=name),
            )

        retained = len(frontiers)
        if len(self.split_fractions) != retained:
            raise ValueError("split_fractions must contain one vector per retained frontier.")
        for name in retained_vectors:
            if getattr(self, name).shape != (retained,):
                raise ValueError(f"{name} must have one entry per retained frontier.")
        if self.fixed_coefficients.shape[0] != retained:
            raise ValueError("fixed_coefficients must have one row per retained frontier.")
        for position, (frontier, fraction) in enumerate(zip(frontiers, self.split_fractions, strict=True)):
            if fraction.shape != (len(frontier) - 1,):
                raise ValueError(f"split_fractions[{position}] must have K-1 entries.")
        if retained:
            finite_retained = (
                "root_total",
                "log_gaussian_likelihood",
                "log_likelihood",
                "log_root_prior",
                "log_fraction_prior",
                "log_partition_prior",
                "log_fixed_coefficient_prior",
                "log_target",
            )
            if (
                any(np.any(~np.isfinite(getattr(self, name))) for name in finite_retained)
                or np.any(~np.isfinite(self.fixed_coefficients))
                or np.any(self.root_total <= 0.0)
                or np.any(self.fixed_coefficients <= 0.0)
                or np.any(self.k < 1)
                or np.any(self.state_transition < 0)
                or np.any(np.diff(self.state_transition) <= 0)
            ):
                raise ValueError("retained state summaries contain invalid values.")
        if any(int(k) != len(frontier) for k, frontier in zip(self.k, frontiers, strict=True)):
            raise ValueError("each retained k must equal its frontier size.")

        attempted = self.global_transition.size
        for name in attempted_vectors:
            if getattr(self, name).shape != (attempted,):
                raise ValueError(f"{name} must have one entry per attempted transition.")
        if attempted and (
            np.any(self.global_transition < 1)
            or np.any(np.diff(self.global_transition) != 1)
            or np.any(self.k_before < 1)
            or np.any(self.k_after < 1)
        ):
            raise ValueError("attempt coordinates and K values are malformed.")
        if np.any(
            ~np.isin(
                self.slot,
                (
                    "structural",
                    "relocation",
                    "subtree_retile",
                    "root",
                    "fraction",
                    "fixed",
                ),
            )
        ):
            raise ValueError("slot contains an unsupported compound slot kind.")
        if np.any(
            ~np.isin(
                self.move,
                (
                    "split",
                    "merge",
                    "relocate",
                    "subtree_retile",
                    "root_refresh",
                    "fraction_refresh",
                    "fixed_coefficient",
                ),
            )
        ):
            raise ValueError("move contains an unsupported Gamma-Beta kernel.")
        if np.any(self.accepted & ~self.valid):
            raise ValueError("accepted transitions must be valid.")
        slot_moves = {
            "structural": ("split", "merge"),
            "relocation": ("relocate",),
            "subtree_retile": ("subtree_retile",),
            "root": ("root_refresh",),
            "fraction": ("fraction_refresh",),
            "fixed": ("fixed_coefficient",),
        }
        for slot_name, allowed_moves in slot_moves.items():
            if np.any((self.slot == slot_name) & ~np.isin(self.move, allowed_moves)):
                raise ValueError("slot and move diagnostics are inconsistent.")
        if np.any((self.move != "relocate") & (self.secondary_node_id != -1)):
            raise ValueError("secondary_node_id is only valid for relocation moves.")
        if np.any(
            (self.move == "relocate") & self.valid & ((self.node_id < 0) | (self.secondary_node_id < 0))
        ):
            raise ValueError("valid relocations require two non-negative node IDs.")
        if np.any((self.move != "subtree_retile") & (self.block_leaf_count != -1)):
            raise ValueError("block_leaf_count is only valid for subtree-retile moves.")
        if np.any((self.move == "subtree_retile") & self.valid & (self.block_leaf_count < 1)):
            raise ValueError("valid subtree retiles require a positive block leaf count.")
        if np.any((self.move == "subtree_retile") & self.valid & (self.node_id < 0)):
            raise ValueError("valid subtree retiles require a non-negative block node.")
        if np.any(np.isnan(self.log_acceptance_ratio)) or np.any(self.log_acceptance_ratio == np.inf):
            raise ValueError("log_acceptance_ratio cannot contain NaN or positive infinity.")

    @property
    def acceptance_rate(self) -> float:
        """Return the accepted fraction of all atomic opportunities.

        Returns:
            Mean every-attempt acceptance, including invalid stays.
        """
        return float(np.mean(self.accepted))


@dataclass(frozen=True, slots=True)
class GammaBetaCompoundCheckpoint:
    """Exact in-memory continuation boundary for the compound sampler.

    Attributes:
        problem: Exact immutable problem object required by identity.
        state: Final state at the segment boundary.
        rng_state: Exact NumPy PCG64 stream state.
        transitions_completed: Global number of attempted transitions.
        schedule_phase: Phase of the next transition in the compound cycle.
        kernel_settings: Complete immutable problem-resolved schedule settings.
        retention: Global warmup/thinning phase.
        schedule_id: Versioned schedule identifier.

    Raises:
        TypeError: If a field has the wrong public type.
        ValueError: If problem/state identity, target support, proposal-scale
            width, transition coordinate, schedule phase or version, or
            contiguous positive ``p(K)`` support is invalid.
    """

    problem: GammaBetaTreeProblem
    state: GammaBetaTreeState
    rng_state: PCG64State
    transitions_completed: int
    schedule_phase: int
    kernel_settings: GammaBetaCompoundKernelSettings
    retention: RetentionSettings
    schedule_id: str = GAMMA_BETA_COMPOUND_SCHEDULE_ID

    def __post_init__(self) -> None:
        """Validate the complete exact continuation contract."""
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
        if isinstance(self.schedule_phase, bool) or not isinstance(self.schedule_phase, Integral):
            raise TypeError("schedule_phase must be an integer.")
        if not isinstance(self.kernel_settings, GammaBetaCompoundKernelSettings):
            raise TypeError("kernel_settings must be GammaBetaCompoundKernelSettings.")
        if len(self.kernel_settings.fixed_coefficient_proposal_sd) != self.state.fixed_coefficients.size:
            raise ValueError("checkpoint fixed proposal scales do not match the problem.")
        expected_phase = int(self.transitions_completed) % self.kernel_settings.cycle_length
        if self.schedule_phase != expected_phase:
            raise ValueError("schedule_phase is inconsistent with transitions_completed.")
        if not isinstance(self.retention, RetentionSettings):
            raise TypeError("retention must be a RetentionSettings.")
        if self.schedule_id != GAMMA_BETA_COMPOUND_SCHEDULE_ID:
            raise ValueError("checkpoint schedule identifier is incompatible.")
        _positive_k_support(
            self.problem,
            relocation_configured=self.kernel_settings.relocation_slots > 0,
            subtree_retile_max_leaves=(
                self.kernel_settings.max_subtree_leaves
                if self.kernel_settings.subtree_retile_slots > 0
                else None
            ),
        )
        object.__setattr__(self, "transitions_completed", int(self.transitions_completed))
        object.__setattr__(self, "schedule_phase", int(self.schedule_phase))


@dataclass(frozen=True, slots=True)
class GammaBetaCompoundSamplingResult:
    """One compound sampling segment and its continuation boundary.

    Attributes:
        trace: Retained states and every-attempt diagnostics.
        final_state: State after the segment's final atomic transition.
        checkpoint: Exact in-memory continuation boundary after the segment.
    """

    trace: GammaBetaCompoundTrace
    final_state: GammaBetaTreeState
    checkpoint: GammaBetaCompoundCheckpoint


def _retained_transition_numbers(
    *,
    transitions_completed: int,
    iterations: int,
    retention: RetentionSettings,
    include_initial: bool,
) -> NDArray[np.int64]:
    """Return global transition coordinates retained in one segment.

    Args:
        transitions_completed: Global coordinate before the segment.
        iterations: Number of atomic transitions in the segment.
        retention: Global warmup and thinning rule.
        include_initial: Whether the state at ``transitions_completed`` is
            eligible for retention.

    Returns:
        Increasing global completed-transition coordinates to retain.
    """
    lower = transitions_completed if include_initial else transitions_completed + 1
    upper = transitions_completed + iterations
    first = max(lower, retention.warmup_transitions)
    remainder = (first - retention.warmup_transitions) % retention.thin
    if remainder:
        first += retention.thin - remainder
    if first > upper:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, upper + 1, retention.thin, dtype=np.int64)


def _slot_at_phase(
    phase: int,
    settings: GammaBetaCompoundKernelSettings,
) -> tuple[CompoundSlot, int | None]:
    """Resolve one zero-based schedule phase to its slot and fixed position.

    Args:
        phase: Zero-based phase within one compound cycle.
        settings: Problem-resolved schedule settings.

    Returns:
        Slot kind and fixed position, where the position is non-null only for
        a deterministic fixed-coefficient slot.
    """
    if phase < 2:
        return "structural", None
    cursor = 2
    relocation_end = cursor + settings.relocation_slots
    if phase < relocation_end:
        return "relocation", None
    cursor = relocation_end
    retile_end = cursor + settings.subtree_retile_slots
    if phase < retile_end:
        return "subtree_retile", None
    cursor = retile_end
    if phase == cursor:
        return "root", None
    cursor += 1
    fraction_end = cursor + settings.fraction_refresh_slots
    if phase < fraction_end:
        return "fraction", None
    return "fixed", phase - fraction_end


def _draw_structural_transition(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    split_direction_probability: float,
    minimum_k: int,
    maximum_k: int,
    rng: np.random.Generator,
) -> GammaBetaTransitionTerms:
    """Draw one support-aware mixed split/merge candidate.

    Args:
        problem: Fixed-tree posterior target.
        state: Source state.
        split_direction_probability: Probability of selecting split.
        minimum_k: Inclusive lower positive-prior support bound.
        maximum_k: Inclusive upper positive-prior support bound.
        rng: NumPy generator advanced by the direction and any eligible node
            and auxiliary-fraction draws.

    Returns:
        Complete split or merge proposal accounting. An unavailable selected
        direction is an explicit invalid self-transition.
    """
    select_split = float(rng.random()) < split_direction_probability
    if select_split:
        splittable = problem.tree.splittable_nodes(state.frontier)
        if state.k >= maximum_k or not splittable:
            return propose_split(
                problem,
                state,
                leaf_node_id=-1,
                new_fraction=0.5,
                split_direction_probability=split_direction_probability,
            )
        node_id = splittable[int(rng.integers(len(splittable)))]
        alpha, beta = problem.prior.beta_parameters(node_id)
        return propose_split(
            problem,
            state,
            leaf_node_id=node_id,
            new_fraction=float(rng.beta(alpha, beta)),
            split_direction_probability=split_direction_probability,
        )

    mergeable = problem.tree.mergeable_parents(state.frontier)
    if state.k <= minimum_k or not mergeable:
        return propose_merge(
            problem,
            state,
            parent_node_id=-1,
            split_direction_probability=split_direction_probability,
        )
    return propose_merge(
        problem,
        state,
        parent_node_id=mergeable[int(rng.integers(len(mergeable)))],
        split_direction_probability=split_direction_probability,
    )


def _draw_relocation_transition(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    rng: np.random.Generator,
) -> GammaBetaTransitionTerms:
    """Draw one sequential cherry-merge and intermediate-leaf split."""
    mergeable = problem.tree.mergeable_parents(state.frontier)
    if not mergeable:
        return propose_relocate(
            problem,
            state,
            merge_parent_node_id=-1,
            split_leaf_node_id=-1,
            new_fraction=0.5,
        )
    source_node_id = mergeable[int(rng.integers(len(mergeable)))]
    intermediate = state.frontier.merge(problem.tree, source_node_id)
    destinations = tuple(
        node_id for node_id in problem.tree.splittable_nodes(intermediate) if node_id != source_node_id
    )
    if not destinations:
        return propose_relocate(
            problem,
            state,
            merge_parent_node_id=source_node_id,
            split_leaf_node_id=-1,
            new_fraction=0.5,
        )
    destination_node_id = destinations[int(rng.integers(len(destinations)))]
    alpha, beta = problem.prior.beta_parameters(destination_node_id)
    return propose_relocate(
        problem,
        state,
        merge_parent_node_id=source_node_id,
        split_leaf_node_id=destination_node_id,
        new_fraction=float(rng.beta(alpha, beta)),
    )


def _subtree_frontier(
    problem: GammaBetaTreeProblem,
    frontier: DyadicFrontier,
    block_node_id: int,
) -> DyadicFrontier:
    """Return the active frontier contained in one canonical subtree."""
    block = problem.tree.node(block_node_id)
    return DyadicFrontier(
        tuple(
            node_id
            for node_id in frontier.node_ids
            if (
                problem.tree.node(node_id).row_start >= block.row_start
                and problem.tree.node(node_id).row_stop <= block.row_stop
                and problem.tree.node(node_id).col_start >= block.col_start
                and problem.tree.node(node_id).col_stop <= block.col_stop
            )
        )
    )


def _randbelow(rng: np.random.Generator, upper: int) -> int:
    """Draw uniformly below an arbitrary-precision positive integer bound."""
    if isinstance(upper, bool) or not isinstance(upper, Integral):
        raise TypeError("upper must be an integer.")
    bound = int(upper)
    if bound < 1:
        raise ValueError("upper must be positive.")
    if bound == 1:
        return 0
    n_bytes = max(1, ((bound - 1).bit_length() + 7) // 8)
    sample_space = 1 << (8 * n_bytes)
    acceptance_limit = sample_space - sample_space % bound
    while True:
        value = int.from_bytes(rng.bytes(n_bytes), byteorder="little")
        if value < acceptance_limit:
            return value % bound


def _draw_subtree_retile_transition(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    index: SubtreePartitionIndex,
    rng: np.random.Generator,
) -> GammaBetaTransitionTerms:
    """Draw one exact alternative bounded-subtree frontier and new fractions."""
    eligible = eligible_subtree_retile_blocks(problem, state, index)
    if not eligible:
        return propose_subtree_retile(
            problem,
            state,
            index,
            block_node_id=-1,
            replacement_frontier=DyadicFrontier.root(problem.tree),
            new_fractions_by_node={},
        )
    block_node_id, block_k = eligible[int(rng.integers(len(eligible)))]
    source_subtree = _subtree_frontier(
        problem,
        state.frontier,
        block_node_id,
    )
    source_rank = index.rank(block_node_id, block_k, source_subtree)
    alternative_count = index.count(block_node_id, block_k) - 1
    alternative_offset = _randbelow(rng, alternative_count)
    replacement_rank = alternative_offset if alternative_offset < source_rank else alternative_offset + 1
    replacement = index.unrank(block_node_id, block_k, replacement_rank)
    source_subtree_ids = frozenset(source_subtree.node_ids)
    candidate_frontier = DyadicFrontier(
        tuple(node_id for node_id in state.frontier.node_ids if node_id not in source_subtree_ids)
        + replacement.node_ids
    )
    source_split_nodes = frozenset(state.frontier.active_split_nodes(problem.tree))
    candidate_split_nodes = frozenset(candidate_frontier.active_split_nodes(problem.tree))
    new_fractions: dict[int, float] = {}
    for node_id in sorted(candidate_split_nodes - source_split_nodes):
        alpha, beta = problem.prior.beta_parameters(node_id)
        new_fractions[node_id] = float(rng.beta(alpha, beta))
    return propose_subtree_retile(
        problem,
        state,
        index,
        block_node_id=block_node_id,
        replacement_frontier=replacement,
        new_fractions_by_node=new_fractions,
    )


def _draw_transition(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
    *,
    phase: int,
    settings: GammaBetaCompoundKernelSettings,
    minimum_k: int,
    maximum_k: int,
    subtree_index: SubtreePartitionIndex | None,
    rng: np.random.Generator,
) -> tuple[CompoundSlot, GammaBetaTransitionTerms]:
    """Draw the candidate assigned to one global compound-schedule phase.

    Args:
        problem: Fixed-tree posterior target.
        state: Source state.
        phase: Zero-based phase within the compound cycle.
        settings: Problem-resolved schedule settings.
        minimum_k: Inclusive lower positive-prior support bound.
        maximum_k: Inclusive upper positive-prior support bound.
        subtree_index: Segment-level exact bounded subtree index, or ``None``
            when no retile slots are configured.
        rng: NumPy generator advanced by the selected kernel's proposal draws.

    Returns:
        Stable slot kind and complete proposal accounting. Fraction slots at
        ``K=1`` and unavailable structural directions return explicit invalid
        self-transitions.

    Raises:
        RuntimeError: If a fixed slot cannot be mapped to a coefficient.
    """
    slot, fixed_position = _slot_at_phase(phase, settings)
    if slot == "structural":
        return slot, _draw_structural_transition(
            problem,
            state,
            split_direction_probability=settings.split_direction_probability,
            minimum_k=minimum_k,
            maximum_k=maximum_k,
            rng=rng,
        )
    if slot == "relocation":
        return slot, _draw_relocation_transition(problem, state, rng=rng)
    if slot == "subtree_retile":
        if subtree_index is None:
            raise RuntimeError("subtree-retile slot requires a segment index.")
        return slot, _draw_subtree_retile_transition(
            problem,
            state,
            index=subtree_index,
            rng=rng,
        )
    if slot == "root":
        root_total = float(
            rng.gamma(
                shape=problem.prior.root_shape,
                scale=1.0 / problem.prior.root_rate,
            )
        )
        return slot, propose_root_refresh(
            problem,
            state,
            new_root_total=root_total,
        )
    if slot == "fraction":
        node_ids = state.frontier.active_split_nodes(problem.tree)
        if not node_ids:
            return slot, propose_fraction_refresh(
                problem,
                state,
                split_node_id=-1,
                new_fraction=0.5,
            )
        node_id = node_ids[int(rng.integers(len(node_ids)))]
        alpha, beta = problem.prior.beta_parameters(node_id)
        return slot, propose_fraction_refresh(
            problem,
            state,
            split_node_id=node_id,
            new_fraction=float(rng.beta(alpha, beta)),
        )
    if fixed_position is None:
        raise RuntimeError("fixed schedule slot did not resolve a coefficient position.")
    proposal_sd = settings.fixed_coefficient_proposal_sd[fixed_position]
    proposed = float(
        rng.normal(
            loc=float(state.fixed_coefficients[fixed_position]),
            scale=proposal_sd,
        )
    )
    return slot, propose_fixed_coefficient(
        problem,
        state,
        coefficient_position=fixed_position,
        proposed_coefficient=proposed,
        proposal_stdev=proposal_sd,
    )


def _run_segment(
    problem: GammaBetaTreeProblem,
    initial_state: GammaBetaTreeState,
    *,
    iterations: int,
    rng: np.random.Generator,
    transitions_completed: int,
    settings: GammaBetaCompoundKernelSettings,
    retention: RetentionSettings,
    include_initial: bool,
) -> GammaBetaCompoundSamplingResult:
    """Run one exact posterior segment using global schedule coordinates.

    Args:
        problem: Fixed-tree posterior target.
        initial_state: Source state at the segment boundary.
        iterations: Positive number of atomic transitions.
        rng: PCG64 generator at the exact segment-start state.
        transitions_completed: Global coordinate before the segment.
        settings: Complete problem-resolved kernel settings.
        retention: Global warmup and thinning rule.
        include_initial: Whether the boundary state is eligible for retention.

    Returns:
        Segment trace, final state, and exact next checkpoint.

    Raises:
        ValueError: If state/problem identity, finite target support, fixed
            scale width, contiguous ``p(K)`` support, initial ``K``, or
            planned retention count is inconsistent.

    Notes:
        Each loop iteration consumes one acceptance uniform after all
        proposal-specific draws, including invalid boundary opportunities.
    """
    if initial_state.problem is not problem:
        raise ValueError("initial_state must have been built for the supplied problem.")
    if not isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target support.")
    if len(settings.fixed_coefficient_proposal_sd) != initial_state.fixed_coefficients.size:
        raise ValueError("fixed proposal scales must match the problem fixed block.")
    minimum_k, maximum_k = _positive_k_support(
        problem,
        relocation_configured=settings.relocation_slots > 0,
        subtree_retile_max_leaves=(
            settings.max_subtree_leaves if settings.subtree_retile_slots > 0 else None
        ),
    )
    if settings.fraction_refresh_slots == 0 and maximum_k > 1:
        raise ValueError("fraction_refresh_slots must be positive when p(K) supports active fractions.")
    if not minimum_k <= initial_state.k <= maximum_k:
        raise ValueError("initial_state K lies outside positive partition-prior support.")
    subtree_index = (
        SubtreePartitionIndex(
            problem.tree,
            min(settings.max_subtree_leaves, len(problem.tree.leaf_ids)),
        )
        if settings.subtree_retile_slots
        else None
    )

    retained_transitions = _retained_transition_numbers(
        transitions_completed=transitions_completed,
        iterations=iterations,
        retention=retention,
        include_initial=include_initial,
    )
    retained_frontiers: list[DyadicFrontier] = []
    retained_fractions: list[NDArray[np.float64]] = []
    retained_root: list[float] = []
    retained_fixed: list[NDArray[np.float64]] = []
    retained_k: list[int] = []
    retained_log_gaussian: list[float] = []
    retained_log_likelihood: list[float] = []
    retained_log_root: list[float] = []
    retained_log_fraction: list[float] = []
    retained_log_partition: list[float] = []
    retained_log_fixed: list[float] = []
    retained_log_target: list[float] = []

    global_transition = np.arange(
        transitions_completed + 1,
        transitions_completed + iterations + 1,
        dtype=np.int64,
    )
    slots = np.empty(iterations, dtype="U14")
    moves = np.empty(iterations, dtype="U17")
    valid = np.empty(iterations, dtype=np.bool_)
    accepted = np.empty(iterations, dtype=np.bool_)
    node_id = np.full(iterations, -1, dtype=np.int64)
    secondary_node_id = np.full(iterations, -1, dtype=np.int64)
    block_leaf_count = np.full(iterations, -1, dtype=np.int64)
    coefficient_id = np.full(iterations, -1, dtype=np.int64)
    k_before = np.empty(iterations, dtype=np.int64)
    k_after = np.empty(iterations, dtype=np.int64)
    log_acceptance_ratio = np.empty(iterations, dtype=np.float64)
    state = initial_state
    retained_position = 0

    def retain(current: GammaBetaTreeState) -> None:
        """Copy one immutable state into variable-dimensional trace storage."""
        nonlocal retained_position
        retained_frontiers.append(current.frontier)
        fraction = np.array(current.split_fractions, copy=True)
        fraction.setflags(write=False)
        retained_fractions.append(fraction)
        fixed = np.array(current.fixed_coefficients, copy=True)
        fixed.setflags(write=False)
        retained_fixed.append(fixed)
        retained_root.append(current.root_total)
        retained_k.append(current.k)
        retained_log_gaussian.append(current.log_gaussian_likelihood)
        retained_log_likelihood.append(current.log_likelihood)
        retained_log_root.append(current.log_root_prior)
        retained_log_fraction.append(current.log_fraction_prior)
        retained_log_partition.append(current.log_partition_prior)
        retained_log_fixed.append(current.log_fixed_coefficient_prior)
        retained_log_target.append(current.log_target)
        retained_position += 1

    if retained_transitions.size and retained_transitions[0] == transitions_completed:
        retain(state)
    next_retained = (
        int(retained_transitions[retained_position])
        if retained_position < retained_transitions.size
        else None
    )

    for iteration in range(iterations):
        phase = (transitions_completed + iteration) % settings.cycle_length
        source = state
        slot, transition = _draw_transition(
            problem,
            source,
            phase=phase,
            settings=settings,
            minimum_k=minimum_k,
            maximum_k=maximum_k,
            subtree_index=subtree_index,
            rng=rng,
        )
        uniform = float(rng.random())
        log_uniform = -np.inf if uniform == 0.0 else log(uniform)
        state = accept_or_reject(source, transition, log_uniform=log_uniform)
        proposal_accepted = transition.valid and state is transition.candidate

        slots[iteration] = slot
        moves[iteration] = transition.move
        valid[iteration] = transition.valid
        accepted[iteration] = proposal_accepted
        if transition.node_id is not None:
            node_id[iteration] = transition.node_id
        if transition.secondary_node_id is not None:
            secondary_node_id[iteration] = transition.secondary_node_id
        if transition.block_leaf_count is not None:
            block_leaf_count[iteration] = transition.block_leaf_count
        if transition.coefficient_id is not None:
            coefficient_id[iteration] = transition.coefficient_id
        k_before[iteration] = source.k
        k_after[iteration] = state.k
        log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
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
    if retained_fixed:
        fixed_matrix = np.stack(retained_fixed, axis=0)
    else:
        fixed_matrix = np.empty(
            (0, len(settings.fixed_coefficient_proposal_sd)),
            dtype=np.float64,
        )
    total_transitions = transitions_completed + iterations
    checkpoint = GammaBetaCompoundCheckpoint(
        problem=problem,
        state=state,
        rng_state=PCG64State.from_generator(rng),
        transitions_completed=total_transitions,
        schedule_phase=total_transitions % settings.cycle_length,
        kernel_settings=settings,
        retention=retention,
    )
    return GammaBetaCompoundSamplingResult(
        trace=GammaBetaCompoundTrace(
            frontiers=tuple(retained_frontiers),
            split_fractions=tuple(retained_fractions),
            root_total=np.asarray(retained_root, dtype=np.float64),
            fixed_coefficients=fixed_matrix,
            k=np.asarray(retained_k, dtype=np.int64),
            log_gaussian_likelihood=np.asarray(retained_log_gaussian, dtype=np.float64),
            log_likelihood=np.asarray(retained_log_likelihood, dtype=np.float64),
            log_root_prior=np.asarray(retained_log_root, dtype=np.float64),
            log_fraction_prior=np.asarray(retained_log_fraction, dtype=np.float64),
            log_partition_prior=np.asarray(retained_log_partition, dtype=np.float64),
            log_fixed_coefficient_prior=np.asarray(retained_log_fixed, dtype=np.float64),
            log_target=np.asarray(retained_log_target, dtype=np.float64),
            state_transition=retained_transitions,
            global_transition=global_transition,
            slot=slots,
            move=moves,
            valid=valid,
            accepted=accepted,
            node_id=node_id,
            secondary_node_id=secondary_node_id,
            block_leaf_count=block_leaf_count,
            coefficient_id=coefficient_id,
            k_before=k_before,
            k_after=k_after,
            log_acceptance_ratio=log_acceptance_ratio,
        ),
        final_state=state,
        checkpoint=checkpoint,
    )


def sample_gamma_beta_compound(
    problem: GammaBetaTreeProblem,
    initial_state: GammaBetaTreeState,
    config: GammaBetaCompoundConfig,
    *,
    retention: RetentionSettings | None = None,
) -> GammaBetaCompoundSamplingResult:
    """Run a fresh seeded Gamma--Beta full-posterior compound segment.

    Args:
        problem: Fixed-tree observation model and normalized priors.
        initial_state: State built for the exact supplied problem object.
        config: Seed, transition count, schedule opportunities, and fixed
            Gaussian scales.
        retention: Optional global warmup/thinning coordinates.

    Returns:
        Segment trace, final state, and exact in-memory checkpoint.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If state/problem identity, partition support, or
            problem-resolved proposal scales are malformed.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(initial_state, GammaBetaTreeState):
        raise TypeError("initial_state must be a GammaBetaTreeState.")
    if not isinstance(config, GammaBetaCompoundConfig):
        raise TypeError("config must be a GammaBetaCompoundConfig.")
    retention_settings = RetentionSettings() if retention is None else retention
    if not isinstance(retention_settings, RetentionSettings):
        raise TypeError("retention must be a RetentionSettings or None.")
    scales = _resolve_fixed_scales(
        config.fixed_coefficient_proposal_sd,
        n_fixed=initial_state.fixed_coefficients.size,
    )
    settings = GammaBetaCompoundKernelSettings(
        split_direction_probability=config.split_direction_probability,
        fraction_refresh_slots=config.fraction_refresh_slots,
        relocation_slots=config.relocation_slots,
        subtree_retile_slots=config.subtree_retile_slots,
        max_subtree_leaves=config.max_subtree_leaves,
        fixed_coefficient_proposal_sd=scales,
    )
    return _run_segment(
        problem,
        initial_state,
        iterations=config.iterations,
        rng=np.random.Generator(np.random.PCG64(config.seed)),
        transitions_completed=0,
        settings=settings,
        retention=retention_settings,
        include_initial=True,
    )


def continue_gamma_beta_compound(
    problem: GammaBetaTreeProblem,
    checkpoint: GammaBetaCompoundCheckpoint,
    *,
    iterations: int,
) -> GammaBetaCompoundSamplingResult:
    """Continue a compound chain exactly from an in-memory checkpoint.

    Args:
        problem: Exact problem object retained by ``checkpoint``.
        checkpoint: Compound in-memory continuation boundary.
        iterations: Positive number of additional atomic transitions.

    Returns:
        Continued segment trace, final state, and next checkpoint.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If iterations, problem identity, schedule version, or
            checkpoint phase is incompatible.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(checkpoint, GammaBetaCompoundCheckpoint):
        raise TypeError("checkpoint must be a GammaBetaCompoundCheckpoint.")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral):
        raise TypeError("iterations must be an integer.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if checkpoint.problem is not problem:
        raise ValueError("continuation requires the exact in-memory problem object.")
    if checkpoint.schedule_id != GAMMA_BETA_COMPOUND_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible with this sampler.")
    expected_phase = checkpoint.transitions_completed % checkpoint.kernel_settings.cycle_length
    if checkpoint.schedule_phase != expected_phase:
        raise ValueError("checkpoint schedule phase is inconsistent.")
    return _run_segment(
        problem,
        checkpoint.state,
        iterations=int(iterations),
        rng=checkpoint.rng_state.generator(),
        transitions_completed=checkpoint.transitions_completed,
        settings=checkpoint.kernel_settings,
        retention=checkpoint.retention,
        include_initial=False,
    )


__all__ = [
    "GAMMA_BETA_COMPOUND_SCHEDULE_ID",
    "CompoundSlot",
    "GammaBetaCompoundCheckpoint",
    "GammaBetaCompoundConfig",
    "GammaBetaCompoundKernelSettings",
    "GammaBetaCompoundSamplingResult",
    "GammaBetaCompoundTrace",
    "continue_gamma_beta_compound",
    "sample_gamma_beta_compound",
]
