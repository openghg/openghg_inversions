"""Correctness-first fixed-``K`` full-tiling posterior sampler.

This module composes the deterministic full-tiling posterior proposals into a
small NumPy reference sampler suitable for real-data smoke tests.  One
versioned cycle contains two independently mixed structural opportunities,
each selecting an edge flip or resolution relocation with probability one
half, one independent-prior root-total refresh, a configurable number of
independent-prior pair-allocation refreshes, and one deterministic
Gaussian-random-walk opportunity for every fixed coefficient. This is not
convergence evidence, and irreducibility over the complete fixed-``K`` tiling
space is not claimed.

Structural selection is bounded to current-state geometry and avoids
exhaustive state-space enumeration. It enumerates only currently mergeable
midpoint-friend pairs. Edge flips choose the
perpendicular orientation.  Relocations choose from the fixed catalogue of
every intermediate leaf crossed with both axis labels, including invalid
choices.  Invalid attempts are explicit self-transitions and every atomic
slot consumes one acceptance uniform.  The sampler never calls the exhaustive
tiling or proposal-path enumeration oracles.

Retained states occur at the initial coordinate and complete global cycle
boundaries.  Every attempted transition is diagnosed independently of retained
states.  In-memory checkpoints preserve the exact PCG64 stream, global
transition coordinate, schedule phase, fixed region count, and immutable
kernel settings, so continuation from an awkward mid-cycle boundary exactly
reproduces an uninterrupted chain.  Durable serialization is intentionally
outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .full_tiling import Axis, MergeChoice, Rectangle, SplitChoice, merge_choices
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    PosteriorTransitionTerms,
    accept_or_reject,
    propose_fixed_coefficient,
    propose_pair_allocation_refresh,
    propose_posterior_edge_flip,
    propose_posterior_resolution_relocation,
    propose_root_total_refresh,
)
from .sampling import PCG64State


FULL_TILING_COMPOUND_SCHEDULE_ID = "full_tiling_2_mixed_structure_1_root_n_pair_allocation_fixed_sweep_v1"
"""Versioned identifier for the fixed-``K`` compound schedule."""

FullTilingCompoundSlot = Literal["structural", "root", "pair_allocation", "fixed"]
"""Atomic full-tiling compound-schedule slot kind."""


def _readonly_array(
    values: object,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    ndim: int,
    name: str,
) -> np.ndarray:
    """Return an owned read-only NumPy array of the requested rank.

    Args:
        values: Values convertible to a NumPy array.
        dtype: Required NumPy dtype.
        ndim: Required number of dimensions.
        name: Public field name used in validation errors.

    Returns:
        Owned read-only array.

    Raises:
        ValueError: If the converted array has the wrong rank.
    """
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    array.setflags(write=False)
    return array


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite positive real setting.

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


def _normalize_fixed_scale_input(value: float | ArrayLike) -> float | tuple[float, ...]:
    """Normalize a scalar or one-dimensional fixed proposal-scale setting.

    Args:
        value: Positive scalar or one-dimensional positive values.

    Returns:
        A positive scalar or immutable tuple of positive scales.

    Raises:
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


def _positive_integer(value: object, *, name: str, allow_zero: bool = False) -> int:
    """Validate and normalize one integer setting.

    Args:
        value: Candidate integer.
        name: Public setting name used in validation errors.
        allow_zero: Whether zero lies in the supported range.

    Returns:
        Normalized Python integer.

    Raises:
        TypeError: If the value is Boolean or non-integral.
        ValueError: If the value lies below the supported range.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        adjective = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {adjective}.")
    return result


@dataclass(frozen=True, slots=True)
class FullTilingCompoundConfig:
    """Configuration for a fresh fixed-``K`` full-tiling segment.

    Args:
        iterations: Positive number of atomic transitions.
        seed: Non-negative PCG64 seed or ``None``.
        pair_allocation_refresh_slots: Non-negative number of uniformly
            selected unordered-pair allocation refreshes per cycle.  The
            default five matches the dynamic-coefficient opportunity budget
            used by the real-data smoke schedule.
        fixed_coefficient_proposal_sd: Positive scalar shared by every fixed
            coefficient or a positive one-dimensional per-position vector.

    Raises:
        TypeError: If an integer setting has the wrong type.
        ValueError: If a setting lies outside its supported range.
    """

    iterations: int
    seed: int | None = None
    pair_allocation_refresh_slots: int = 5
    fixed_coefficient_proposal_sd: float | tuple[float, ...] = 0.4

    def __post_init__(self) -> None:
        """Normalize and validate problem-independent settings."""
        object.__setattr__(
            self,
            "iterations",
            _positive_integer(self.iterations, name="iterations"),
        )
        if self.seed is not None:
            seed = _positive_integer(self.seed, name="seed", allow_zero=True)
            object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "pair_allocation_refresh_slots",
            _positive_integer(
                self.pair_allocation_refresh_slots,
                name="pair_allocation_refresh_slots",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "fixed_coefficient_proposal_sd",
            _normalize_fixed_scale_input(self.fixed_coefficient_proposal_sd),
        )


@dataclass(frozen=True, slots=True)
class FullTilingCompoundKernelSettings:
    """Immutable problem-resolved fixed-``K`` kernel settings.

    Attributes:
        fixed_k: Region count preserved by every transition.
        pair_allocation_refresh_slots: Pair refresh opportunities per cycle.
        fixed_coefficient_proposal_sd: Gaussian scales in deterministic fixed
            coefficient order.

    Raises:
        TypeError: If an integer setting has the wrong type.
        ValueError: If a setting lies outside its supported range.
    """

    fixed_k: int
    pair_allocation_refresh_slots: int
    fixed_coefficient_proposal_sd: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and normalize resolved schedule settings."""
        object.__setattr__(
            self,
            "fixed_k",
            _positive_integer(self.fixed_k, name="fixed_k"),
        )
        object.__setattr__(
            self,
            "pair_allocation_refresh_slots",
            _positive_integer(
                self.pair_allocation_refresh_slots,
                name="pair_allocation_refresh_slots",
                allow_zero=True,
            ),
        )
        scales = tuple(
            _positive_float(value, name="fixed_coefficient_proposal_sd")
            for value in self.fixed_coefficient_proposal_sd
        )
        object.__setattr__(self, "fixed_coefficient_proposal_sd", scales)

    @property
    def cycle_length(self) -> int:
        """Return the atomic transition count in one complete cycle.

        Returns:
            Two structural slots, one root slot, configured pair-allocation
            slots, and one slot per fixed coefficient.
        """
        return 3 + self.pair_allocation_refresh_slots + len(self.fixed_coefficient_proposal_sd)


@dataclass(frozen=True, slots=True)
class FullTilingCompoundTrace:
    """Cycle-boundary posterior states and every attempted transition.

    All supplied arrays are copied and stored read-only. ``state_transition``
    uses completed-transition coordinates and includes
    zero for a fresh chain.  ``global_transition`` is one-based.  Retained
    geometry has exact shape ``(draw, K, 4)`` and retained masses have exact
    shape ``(draw, K)``; fixed ``K`` therefore requires no padding or mask.

    Attributes:
        rectangle_bounds: Canonical ``(row_start, row_stop, col_start,
            col_stop)`` bounds, shape ``(draw, K, 4)``.
        leaf_masses: Positive canonical leaf masses, shape ``(draw, K)``.
        root_total: Positive retained root totals.
        fixed_coefficients: Retained fixed coefficient matrix.
        log_gaussian_likelihood: Raw normalized Gaussian log likelihood.
        log_likelihood: Likelihood-power-scaled target component.
        log_root_prior: Normalized root Gamma log-density component.
        log_allocation_prior: Normalized conditional mass-allocation component.
        log_structural_prior: Zero structural log-ratio component; the unknown
            fixed-``K`` communication-component normalizer is omitted.
        log_fixed_coefficient_prior: Normalized fixed-block prior component.
        log_target: Retained log target up to the omitted fixed-``K``
            communication-component structural normalizer.
        state_transition: Global coordinates of retained states.
        global_transition: Global coordinates of every attempted transition.
        slot: Stable compound slot name per attempt.
        move: Concrete proposal-kernel name per attempt.
        valid: Whether each proposal reached an MH decision.
        accepted: Whether each proposal changed the visited state.
        log_acceptance_ratio: Raw untruncated MH log ratio, with negative
            infinity for invalid attempts.
        invalid_reason: Empty string for valid proposals and a compact
            diagnostic explanation for invalid proposals.

    Raises:
        ValueError: If arrays violate shape, support, ordering, finiteness, or
            transition-diagnostic invariants.
    """

    rectangle_bounds: NDArray[np.int64]
    leaf_masses: NDArray[np.float64]
    root_total: NDArray[np.float64]
    fixed_coefficients: NDArray[np.float64]
    log_gaussian_likelihood: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]
    log_root_prior: NDArray[np.float64]
    log_allocation_prior: NDArray[np.float64]
    log_structural_prior: NDArray[np.float64]
    log_fixed_coefficient_prior: NDArray[np.float64]
    log_target: NDArray[np.float64]
    state_transition: NDArray[np.int64]
    global_transition: NDArray[np.int64]
    slot: NDArray[np.str_]
    move: NDArray[np.str_]
    valid: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    log_acceptance_ratio: NDArray[np.float64]
    invalid_reason: NDArray[np.str_]

    def __post_init__(self) -> None:
        """Own arrays and enforce retained and attempted axis contracts."""
        retained_arrays = {
            "rectangle_bounds": (np.int64, 3),
            "leaf_masses": (np.float64, 2),
            "root_total": (np.float64, 1),
            "fixed_coefficients": (np.float64, 2),
            "log_gaussian_likelihood": (np.float64, 1),
            "log_likelihood": (np.float64, 1),
            "log_root_prior": (np.float64, 1),
            "log_allocation_prior": (np.float64, 1),
            "log_structural_prior": (np.float64, 1),
            "log_fixed_coefficient_prior": (np.float64, 1),
            "log_target": (np.float64, 1),
            "state_transition": (np.int64, 1),
        }
        for name, (dtype, ndim) in retained_arrays.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    dtype=dtype,
                    ndim=ndim,
                    name=name,
                ),
            )
        attempted_arrays = {
            "global_transition": (np.int64, 1),
            "slot": (np.dtype("U15"), 1),
            "move": (np.dtype("U24"), 1),
            "valid": (np.bool_, 1),
            "accepted": (np.bool_, 1),
            "log_acceptance_ratio": (np.float64, 1),
            "invalid_reason": (np.dtype("U96"), 1),
        }
        for name, (dtype, ndim) in attempted_arrays.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    dtype=dtype,
                    ndim=ndim,
                    name=name,
                ),
            )

        retained = self.state_transition.size
        if self.rectangle_bounds.shape[0] != retained:
            raise ValueError("rectangle_bounds must have one row per retained state.")
        if self.rectangle_bounds.shape[2:] != (4,):
            raise ValueError("rectangle_bounds must have shape (draw, K, 4).")
        fixed_k = self.rectangle_bounds.shape[1]
        if self.leaf_masses.shape != (retained, fixed_k):
            raise ValueError("leaf_masses must have shape (draw, K).")
        if self.fixed_coefficients.shape[0] != retained:
            raise ValueError("fixed_coefficients must have one row per retained state.")
        for name in retained_arrays:
            if name in ("rectangle_bounds", "leaf_masses", "fixed_coefficients"):
                continue
            if getattr(self, name).shape != (retained,):
                raise ValueError(f"{name} must have one entry per retained state.")
        if retained and (
            fixed_k < 1
            or np.any(~np.isfinite(self.leaf_masses))
            or np.any(self.leaf_masses <= 0.0)
            or np.any(~np.isfinite(self.root_total))
            or np.any(self.root_total <= 0.0)
            or np.any(~np.isfinite(self.fixed_coefficients))
            or np.any(self.fixed_coefficients <= 0.0)
            or np.any(np.diff(self.state_transition) <= 0)
            or np.any(self.state_transition < 0)
        ):
            raise ValueError("retained states contain invalid values.")
        finite_log_names = (
            "log_gaussian_likelihood",
            "log_likelihood",
            "log_root_prior",
            "log_allocation_prior",
            "log_structural_prior",
            "log_fixed_coefficient_prior",
            "log_target",
        )
        if any(np.any(~np.isfinite(getattr(self, name))) for name in finite_log_names):
            raise ValueError("retained log-target components must be finite.")
        if retained and not np.allclose(
            self.root_total,
            np.sum(self.leaf_masses, axis=1),
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("root_total must equal the retained leaf-mass sum.")
        if retained and not np.allclose(
            self.log_target,
            self.log_likelihood
            + self.log_root_prior
            + self.log_allocation_prior
            + self.log_structural_prior
            + self.log_fixed_coefficient_prior,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("log_target must equal the sum of retained target components.")

        attempted = self.global_transition.size
        for name in attempted_arrays:
            if getattr(self, name).shape != (attempted,):
                raise ValueError(f"{name} must have one entry per attempted transition.")
        if attempted and (np.any(self.global_transition < 1) or np.any(np.diff(self.global_transition) != 1)):
            raise ValueError("global_transition must contain consecutive positive coordinates.")
        if np.any(~np.isin(self.slot, ("structural", "root", "pair_allocation", "fixed"))):
            raise ValueError("slot contains an unsupported compound slot kind.")
        if np.any(
            ~np.isin(
                self.move,
                (
                    "edge_flip",
                    "resolution_relocation",
                    "root_total_refresh",
                    "pair_allocation_refresh",
                    "fixed_coefficient",
                ),
            )
        ):
            raise ValueError("move contains an unsupported full-tiling kernel.")
        if np.any(self.accepted & ~self.valid):
            raise ValueError("accepted transitions must be valid.")
        if np.any(np.isnan(self.log_acceptance_ratio)) or np.any(self.log_acceptance_ratio == np.inf):
            raise ValueError("log_acceptance_ratio cannot contain NaN or positive infinity.")
        if np.any(self.valid & (self.invalid_reason != "")):
            raise ValueError("valid proposals cannot have an invalid reason.")
        if np.any(~self.valid & (self.invalid_reason == "")):
            raise ValueError("invalid proposals must have a reason.")

    @property
    def k(self) -> int:
        """Return the fixed retained region count.

        Returns:
            Size of the exact region axis.
        """
        return int(self.rectangle_bounds.shape[1])

    @property
    def acceptance_rate(self) -> float:
        """Return the accepted fraction of all atomic opportunities.

        Returns:
            Mean every-attempt acceptance, including invalid stays, or NaN
            for a manually constructed trace with no attempted transitions.
        """
        return float(np.mean(self.accepted))


@dataclass(frozen=True, slots=True)
class FullTilingCompoundCheckpoint:
    """Exact in-memory continuation boundary for the full-tiling sampler.

    Attributes:
        problem: Exact immutable problem required by identity.
        state: Final posterior state at the segment boundary.
        rng_state: Exact NumPy PCG64 stream state.
        transitions_completed: Global number of attempted transitions.
        schedule_phase: Phase of the next transition in the compound cycle.
        kernel_settings: Complete immutable problem-resolved settings,
            including fixed ``K``.
        schedule_id: Versioned schedule identifier.

    Raises:
        TypeError: If a field has the wrong public type.
        ValueError: If problem identity, fixed ``K``, transition coordinate,
            schedule phase, scale width, or version is inconsistent.
    """

    problem: FullTilingProblem
    state: FullTilingPosteriorState
    rng_state: PCG64State
    transitions_completed: int
    schedule_phase: int
    kernel_settings: FullTilingCompoundKernelSettings
    schedule_id: str = FULL_TILING_COMPOUND_SCHEDULE_ID

    def __post_init__(self) -> None:
        """Validate the complete exact continuation contract."""
        if not isinstance(self.problem, FullTilingProblem):
            raise TypeError("problem must be a FullTilingProblem.")
        if not isinstance(self.state, FullTilingPosteriorState):
            raise TypeError("state must be a FullTilingPosteriorState.")
        if self.state.problem is not self.problem:
            raise ValueError("checkpoint state must belong to checkpoint problem.")
        if not isinstance(self.rng_state, PCG64State):
            raise TypeError("rng_state must be a PCG64State.")
        completed = _positive_integer(
            self.transitions_completed,
            name="transitions_completed",
            allow_zero=True,
        )
        phase = _positive_integer(
            self.schedule_phase,
            name="schedule_phase",
            allow_zero=True,
        )
        if not isinstance(
            self.kernel_settings,
            FullTilingCompoundKernelSettings,
        ):
            raise TypeError("kernel_settings must be a FullTilingCompoundKernelSettings.")
        if self.state.k != self.kernel_settings.fixed_k:
            raise ValueError("checkpoint state K does not match fixed kernel K.")
        if self.state.fixed_coefficients.size != len(self.kernel_settings.fixed_coefficient_proposal_sd):
            raise ValueError("checkpoint fixed proposal scales do not match the problem.")
        if phase != completed % self.kernel_settings.cycle_length:
            raise ValueError("schedule_phase is inconsistent with transitions_completed.")
        if self.schedule_id != FULL_TILING_COMPOUND_SCHEDULE_ID:
            raise ValueError("checkpoint schedule identifier is incompatible.")
        if not isfinite(self.state.log_target):
            raise ValueError("checkpoint state must have finite target support.")
        object.__setattr__(self, "transitions_completed", completed)
        object.__setattr__(self, "schedule_phase", phase)


@dataclass(frozen=True, slots=True)
class FullTilingCompoundSamplingResult:
    """One sampling segment and its exact continuation boundary.

    Attributes:
        trace: Cycle-boundary states and every-attempt diagnostics.
        final_state: State after the segment's final atomic transition.
        checkpoint: Exact in-memory continuation boundary after the segment.
    """

    trace: FullTilingCompoundTrace
    final_state: FullTilingPosteriorState
    checkpoint: FullTilingCompoundCheckpoint


def _retained_cycle_boundaries(
    *,
    transitions_completed: int,
    iterations: int,
    cycle_length: int,
    include_initial: bool,
) -> NDArray[np.int64]:
    """Return global complete-cycle coordinates retained by one segment.

    Args:
        transitions_completed: Global coordinate before the segment.
        iterations: Atomic transition count in this segment.
        cycle_length: Positive number of atomic slots per cycle.
        include_initial: Whether a fresh coordinate-zero state is retained.

    Returns:
        Increasing global completed-transition coordinates.  A fresh segment
        begins with zero; all other entries are positive cycle multiples.
    """
    upper = transitions_completed + iterations
    first = ((transitions_completed + 1 + cycle_length - 1) // cycle_length) * cycle_length
    if first <= upper:
        boundaries = np.arange(first, upper + 1, cycle_length, dtype=np.int64)
    else:
        boundaries = np.empty(0, dtype=np.int64)
    if include_initial:
        return np.concatenate((np.asarray([transitions_completed], dtype=np.int64), boundaries))
    return boundaries


def _slot_at_phase(
    phase: int,
    settings: FullTilingCompoundKernelSettings,
) -> tuple[FullTilingCompoundSlot, int | None]:
    """Resolve one zero-based cycle phase to a slot and fixed position.

    Args:
        phase: Zero-based phase within one compound cycle.
        settings: Immutable problem-resolved kernel settings.

    Returns:
        Slot kind and a fixed coefficient position only for fixed slots.
    """
    if phase < 2:
        return "structural", None
    if phase == 2:
        return "root", None
    pair_end = 3 + settings.pair_allocation_refresh_slots
    if phase < pair_end:
        return "pair_allocation", None
    return "fixed", phase - pair_end


def _invalid_merge_choice() -> MergeChoice:
    """Return an always out-of-domain merge choice for invalid self-attempts."""
    return MergeChoice(Rectangle(-2, 0, -2, 0), "horizontal")


def _invalid_split_choice() -> SplitChoice:
    """Return an always out-of-domain split choice for invalid self-attempts."""
    return SplitChoice(Rectangle(-4, -2, -4, -2), "horizontal")


def _draw_beta(
    problem: FullTilingProblem,
    first: Rectangle,
    second: Rectangle,
    *,
    rng: np.random.Generator,
) -> float:
    """Draw the additive-alpha Beta fraction for an ordered rectangle pair.

    Args:
        problem: Full-tiling problem owning the additive-alpha measure.
        first: Rectangle associated with the first Beta shape.
        second: Rectangle associated with the second Beta shape.
        rng: Generator advanced by one Beta draw.

    Returns:
        Open-unit Beta draw.
    """
    return float(
        rng.beta(
            problem.allocation_prior.alpha(first),
            problem.allocation_prior.alpha(second),
        )
    )


def _draw_structural_transition(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    rng: np.random.Generator,
) -> PosteriorTransitionTerms:
    """Draw one fixed half-and-half structural proposal.

    The component choice is never availability-renormalized.  Only source
    merge choices are enumerated.  Relocation uses the fixed intermediate
    ``leaf × axis`` catalogue, retaining invalid entries as explicit attempts.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        rng: Generator advanced by the component and proposal-specific draws.

    Returns:
        Complete deterministic posterior transition terms.
    """
    choose_edge_flip = float(rng.random()) < 0.5
    merges = merge_choices(state.tiling_state.tiling)
    if not merges:
        merge = _invalid_merge_choice()
        if choose_edge_flip:
            return propose_posterior_edge_flip(
                problem,
                state,
                merge_choice=merge,
                new_fraction=0.5,
            )
        return propose_posterior_resolution_relocation(
            problem,
            state,
            merge_choice=merge,
            split_choice=_invalid_split_choice(),
            new_fraction=0.5,
        )

    merge = merges[int(rng.integers(len(merges)))]
    if choose_edge_flip:
        target_axis: Axis = "vertical" if merge.axis == "horizontal" else "horizontal"
        if target_axis in merge.parent.admissible_axes:
            children = merge.parent.midpoint_children(target_axis)
            fraction = _draw_beta(problem, children[0], children[1], rng=rng)
        else:
            fraction = 0.5
        return propose_posterior_edge_flip(
            problem,
            state,
            merge_choice=merge,
            new_fraction=fraction,
        )

    intermediate = state.tiling_state.tiling.merge(merge)
    axes: tuple[Axis, Axis] = ("horizontal", "vertical")
    catalogue = tuple(SplitChoice(leaf, axis) for leaf in intermediate.leaves for axis in axes)
    if len(catalogue) != 2 * (state.k - 1):
        raise RuntimeError("relocation catalogue does not have fixed size 2 * (K - 1).")
    split = catalogue[int(rng.integers(len(catalogue)))]
    geometrically_admissible = split.leaf != merge.parent and split.axis in split.leaf.admissible_axes
    if geometrically_admissible:
        children = split.leaf.midpoint_children(split.axis)
        fraction = _draw_beta(problem, children[0], children[1], rng=rng)
    else:
        fraction = 0.5
    return propose_posterior_resolution_relocation(
        problem,
        state,
        merge_choice=merge,
        split_choice=split,
        new_fraction=fraction,
    )


def _draw_pair_refresh_transition(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    rng: np.random.Generator,
) -> PosteriorTransitionTerms:
    """Draw one uniform unordered-pair additive-alpha refresh.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        rng: Generator advanced by pair selection and an eligible Beta draw.

    Returns:
        Complete pair-refresh transition terms.  ``K=1`` is an explicit
        invalid self-attempt through the deterministic proposal API.
    """
    leaves = state.tiling_state.tiling.leaves
    pair_count = state.k * (state.k - 1) // 2
    if pair_count == 0:
        return propose_pair_allocation_refresh(
            problem,
            state,
            first_leaf=_invalid_split_choice().leaf,
            second_leaf=_invalid_merge_choice().parent,
            new_fraction=0.5,
        )
    selected = int(rng.integers(pair_count))
    offset = 0
    first_position = 0
    second_position = 1
    for first_position in range(state.k - 1):
        width = state.k - first_position - 1
        if selected < offset + width:
            second_position = first_position + 1 + selected - offset
            break
        offset += width
    first = leaves[first_position]
    second = leaves[second_position]
    fraction = _draw_beta(problem, first, second, rng=rng)
    return propose_pair_allocation_refresh(
        problem,
        state,
        first_leaf=first,
        second_leaf=second,
        new_fraction=fraction,
    )


def _draw_transition(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    phase: int,
    settings: FullTilingCompoundKernelSettings,
    rng: np.random.Generator,
) -> tuple[FullTilingCompoundSlot, PosteriorTransitionTerms]:
    """Draw the proposal assigned to one global schedule phase.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        phase: Zero-based phase within one cycle.
        settings: Immutable problem-resolved kernel settings.
        rng: Generator advanced by proposal-specific draws.

    Returns:
        Stable slot name and complete deterministic transition terms.

    Raises:
        RuntimeError: If a fixed phase does not map to a coefficient.
    """
    slot, fixed_position = _slot_at_phase(phase, settings)
    if slot == "structural":
        return slot, _draw_structural_transition(problem, state, rng=rng)
    if slot == "root":
        return slot, propose_root_total_refresh(
            problem,
            state,
            new_root_total=float(
                rng.gamma(
                    shape=problem.base.prior.root_shape,
                    scale=1.0 / problem.base.prior.root_rate,
                )
            ),
        )
    if slot == "pair_allocation":
        return slot, _draw_pair_refresh_transition(problem, state, rng=rng)
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


def _rectangle_bounds(state: FullTilingPosteriorState) -> NDArray[np.int64]:
    """Return canonical integer bounds for one posterior state's leaves."""
    return np.asarray(
        [
            (leaf.row_start, leaf.row_stop, leaf.col_start, leaf.col_stop)
            for leaf in state.tiling_state.tiling.leaves
        ],
        dtype=np.int64,
    )


def _run_segment(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    *,
    iterations: int,
    rng: np.random.Generator,
    transitions_completed: int,
    settings: FullTilingCompoundKernelSettings,
    include_initial: bool,
) -> FullTilingCompoundSamplingResult:
    """Run one exact segment using global schedule and retention coordinates.

    Args:
        problem: Fixed-``K`` full-tiling posterior target.
        initial_state: Source state at the segment boundary.
        iterations: Positive number of atomic transitions.
        rng: PCG64 generator at the exact segment-start state.
        transitions_completed: Global coordinate before the segment.
        settings: Complete immutable problem-resolved settings.
        include_initial: Whether the boundary is the fresh initial draw.

    Returns:
        Segment trace, final state, and exact next checkpoint.

    Raises:
        ValueError: If state identity, target support, fixed ``K``, proposal
            scale width, or pair-refresh availability is inconsistent.

    Notes:
        Each loop iteration consumes one acceptance uniform after all
        proposal-specific draws, including invalid structural or continuous
        attempts.
    """
    if initial_state.problem is not problem:
        raise ValueError("initial_state must have been built for the supplied problem.")
    if not isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target support.")
    if initial_state.k != settings.fixed_k:
        raise ValueError("initial_state K must match the immutable fixed K.")
    if initial_state.fixed_coefficients.size != len(settings.fixed_coefficient_proposal_sd):
        raise ValueError("fixed proposal scales must match the problem fixed block.")
    retained_transitions = _retained_cycle_boundaries(
        transitions_completed=transitions_completed,
        iterations=iterations,
        cycle_length=settings.cycle_length,
        include_initial=include_initial,
    )
    retained_bounds: list[NDArray[np.int64]] = []
    retained_masses: list[NDArray[np.float64]] = []
    retained_root: list[float] = []
    retained_fixed: list[NDArray[np.float64]] = []
    retained_log_gaussian: list[float] = []
    retained_log_likelihood: list[float] = []
    retained_log_root: list[float] = []
    retained_log_allocation: list[float] = []
    retained_log_fixed: list[float] = []
    retained_log_target: list[float] = []

    global_transition = np.arange(
        transitions_completed + 1,
        transitions_completed + iterations + 1,
        dtype=np.int64,
    )
    slots = np.empty(iterations, dtype="U15")
    moves = np.empty(iterations, dtype="U24")
    valid = np.empty(iterations, dtype=np.bool_)
    accepted = np.empty(iterations, dtype=np.bool_)
    log_acceptance_ratio = np.empty(iterations, dtype=np.float64)
    invalid_reason = np.empty(iterations, dtype="U96")
    state = initial_state
    retained_position = 0

    def retain(current: FullTilingPosteriorState) -> None:
        """Copy one immutable state into exact fixed-width trace storage."""
        nonlocal retained_position
        retained_bounds.append(_rectangle_bounds(current))
        retained_masses.append(np.array(current.leaf_masses, copy=True))
        retained_fixed.append(np.array(current.fixed_coefficients, copy=True))
        retained_root.append(current.root_total)
        retained_log_gaussian.append(current.log_gaussian_likelihood)
        retained_log_likelihood.append(current.log_likelihood)
        retained_log_root.append(current.log_root_prior)
        retained_log_allocation.append(current.log_allocation_prior)
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
        log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
        invalid_reason[iteration] = "" if transition.reason is None else transition.reason
        completed = transitions_completed + iteration + 1
        if next_retained == completed:
            retain(state)
            next_retained = (
                int(retained_transitions[retained_position])
                if retained_position < retained_transitions.size
                else None
            )

    if retained_position != retained_transitions.size:
        raise RuntimeError("retained full-tiling state count did not match planned coordinates.")
    if retained_bounds:
        bounds = np.stack(retained_bounds, axis=0)
        masses = np.stack(retained_masses, axis=0)
        fixed = np.stack(retained_fixed, axis=0)
    else:
        bounds = np.empty((0, settings.fixed_k, 4), dtype=np.int64)
        masses = np.empty((0, settings.fixed_k), dtype=np.float64)
        fixed = np.empty(
            (0, len(settings.fixed_coefficient_proposal_sd)),
            dtype=np.float64,
        )
    total_transitions = transitions_completed + iterations
    checkpoint = FullTilingCompoundCheckpoint(
        problem=problem,
        state=state,
        rng_state=PCG64State.from_generator(rng),
        transitions_completed=total_transitions,
        schedule_phase=total_transitions % settings.cycle_length,
        kernel_settings=settings,
    )
    return FullTilingCompoundSamplingResult(
        trace=FullTilingCompoundTrace(
            rectangle_bounds=bounds,
            leaf_masses=masses,
            root_total=np.asarray(retained_root, dtype=np.float64),
            fixed_coefficients=fixed,
            log_gaussian_likelihood=np.asarray(
                retained_log_gaussian,
                dtype=np.float64,
            ),
            log_likelihood=np.asarray(retained_log_likelihood, dtype=np.float64),
            log_root_prior=np.asarray(retained_log_root, dtype=np.float64),
            log_allocation_prior=np.asarray(
                retained_log_allocation,
                dtype=np.float64,
            ),
            log_structural_prior=np.zeros(len(retained_root), dtype=np.float64),
            log_fixed_coefficient_prior=np.asarray(
                retained_log_fixed,
                dtype=np.float64,
            ),
            log_target=np.asarray(retained_log_target, dtype=np.float64),
            state_transition=retained_transitions,
            global_transition=global_transition,
            slot=slots,
            move=moves,
            valid=valid,
            accepted=accepted,
            log_acceptance_ratio=log_acceptance_ratio,
            invalid_reason=invalid_reason,
        ),
        final_state=state,
        checkpoint=checkpoint,
    )


def sample_full_tiling_compound(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    config: FullTilingCompoundConfig,
) -> FullTilingCompoundSamplingResult:
    """Run a fresh seeded fixed-``K`` full-tiling posterior segment.

    Args:
        problem: Full-tiling observation model and normalized priors.
        initial_state: State built for the exact supplied problem object.
        config: Seed, atomic transition count, pair opportunities, and fixed
            Gaussian proposal scales.

    Returns:
        Segment trace, final state, and exact in-memory checkpoint.

    Raises:
        TypeError: If an argument has the wrong public type.
        ValueError: If state identity, fixed ``K``, or problem-resolved scales
            are malformed.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(initial_state, FullTilingPosteriorState):
        raise TypeError("initial_state must be a FullTilingPosteriorState.")
    if not isinstance(config, FullTilingCompoundConfig):
        raise TypeError("config must be a FullTilingCompoundConfig.")
    scales = _resolve_fixed_scales(
        config.fixed_coefficient_proposal_sd,
        n_fixed=initial_state.fixed_coefficients.size,
    )
    settings = FullTilingCompoundKernelSettings(
        fixed_k=initial_state.k,
        pair_allocation_refresh_slots=config.pair_allocation_refresh_slots,
        fixed_coefficient_proposal_sd=scales,
    )
    return _run_segment(
        problem,
        initial_state,
        iterations=config.iterations,
        rng=np.random.Generator(np.random.PCG64(config.seed)),
        transitions_completed=0,
        settings=settings,
        include_initial=True,
    )


def continue_full_tiling_compound(
    problem: FullTilingProblem,
    checkpoint: FullTilingCompoundCheckpoint,
    *,
    iterations: int,
) -> FullTilingCompoundSamplingResult:
    """Continue a full-tiling chain exactly from an in-memory checkpoint.

    Args:
        problem: Exact problem object retained by ``checkpoint``.
        checkpoint: In-memory continuation boundary.
        iterations: Positive number of additional atomic transitions.

    Returns:
        Continued segment trace, final state, and next checkpoint.

    Raises:
        TypeError: If arguments have the wrong public types.
        ValueError: If iterations, problem identity, schedule version, fixed
            ``K``, or checkpoint phase is incompatible.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(checkpoint, FullTilingCompoundCheckpoint):
        raise TypeError("checkpoint must be a FullTilingCompoundCheckpoint.")
    transition_count = _positive_integer(iterations, name="iterations")
    if checkpoint.problem is not problem:
        raise ValueError("continuation requires the exact in-memory problem object.")
    if checkpoint.schedule_id != FULL_TILING_COMPOUND_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible with this sampler.")
    expected_phase = checkpoint.transitions_completed % checkpoint.kernel_settings.cycle_length
    if checkpoint.schedule_phase != expected_phase:
        raise ValueError("checkpoint schedule phase is inconsistent.")
    if checkpoint.state.k != checkpoint.kernel_settings.fixed_k:
        raise ValueError("checkpoint state violates its fixed-K kernel setting.")
    return _run_segment(
        problem,
        checkpoint.state,
        iterations=transition_count,
        rng=checkpoint.rng_state.generator(),
        transitions_completed=checkpoint.transitions_completed,
        settings=checkpoint.kernel_settings,
        include_initial=False,
    )


__all__ = [
    "FULL_TILING_COMPOUND_SCHEDULE_ID",
    "FullTilingCompoundCheckpoint",
    "FullTilingCompoundConfig",
    "FullTilingCompoundKernelSettings",
    "FullTilingCompoundSamplingResult",
    "FullTilingCompoundSlot",
    "FullTilingCompoundTrace",
    "continue_full_tiling_compound",
    "sample_full_tiling_compound",
]
