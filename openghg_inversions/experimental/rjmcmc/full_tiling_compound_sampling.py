"""Correctness-first fixed-``K`` full-tiling posterior sampler.

This module composes the deterministic full-tiling posterior proposals into a
small NumPy reference sampler suitable for real-data smoke tests.  One
versioned cycle contains two independently mixed structural opportunities,
each selecting an edge flip or resolution relocation with probability one
half, one stepping-out/shrinkage slice update of the log root total, a
configurable number of
independent-prior pair-allocation refreshes, and one deterministic
Gaussian-random-walk opportunity for every fixed coefficient. This is not
convergence evidence, and irreducibility over the complete fixed-``K`` tiling
space is not claimed.

Structural selection is bounded to current-state geometry and avoids
exhaustive state-space enumeration. It enumerates only currently mergeable
midpoint-friend pairs. Edge flips choose the
perpendicular orientation.  Relocations choose from the fixed catalogue of
every intermediate leaf crossed with both axis labels, including invalid
choices.  Invalid attempts are explicit self-transitions. Every non-slice
atomic slot consumes one final acceptance uniform, while the slice slot uses
only its height, bracket, and shrinkage draws. The sampler never calls the
exhaustive tiling or proposal-path enumeration oracles.

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
from math import fsum, isfinite, log
from numbers import Integral
from time import perf_counter_ns
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .full_tiling import Axis, MergeChoice, Rectangle, SplitChoice, merge_choices
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    PosteriorTransitionTerms,
    accept_or_reject,
    log_root_total_slice_density,
    propose_fixed_coefficient,
    propose_pair_allocation_refresh,
    propose_posterior_edge_flip,
    propose_posterior_resolution_relocation,
    rescale_full_tiling_root_total,
)
from .sampling import PCG64State


FULL_TILING_COMPOUND_SCHEDULE_ID = (
    "full_tiling_2_mixed_structure_1_root_slice_n_pair_allocation_fixed_sweep_v2"
)
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
    if isinstance(value, (bool, np.bool_)):
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
        root_slice_width: Positive initial stepping-out bracket width in
            ``z = log(T)``.
        root_slice_max_steps: Positive finite stepping-out budget, including
            the initial bracket.
        root_slice_max_shrink_steps: Positive maximum number of shrinkage
            candidates before a guard error.

    Raises:
        TypeError: If an integer setting has the wrong type.
        ValueError: If a setting lies outside its supported range.
    """

    iterations: int
    seed: int | None = None
    pair_allocation_refresh_slots: int = 5
    fixed_coefficient_proposal_sd: float | tuple[float, ...] = 0.4
    root_slice_width: float = 1.0
    root_slice_max_steps: int = 100
    root_slice_max_shrink_steps: int = 1000

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
        root_slice_width = _positive_float(
            self.root_slice_width,
            name="root_slice_width",
        )
        root_slice_max_steps = _positive_integer(
            self.root_slice_max_steps,
            name="root_slice_max_steps",
        )
        if not isfinite(root_slice_width * root_slice_max_steps):
            raise ValueError("root_slice_width times root_slice_max_steps must be finite.")
        object.__setattr__(self, "root_slice_width", root_slice_width)
        object.__setattr__(self, "root_slice_max_steps", root_slice_max_steps)
        object.__setattr__(
            self,
            "root_slice_max_shrink_steps",
            _positive_integer(
                self.root_slice_max_shrink_steps,
                name="root_slice_max_shrink_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class FullTilingCompoundKernelSettings:
    """Immutable problem-resolved fixed-``K`` kernel settings.

    Attributes:
        fixed_k: Region count preserved by every transition.
        pair_allocation_refresh_slots: Pair refresh opportunities per cycle.
        fixed_coefficient_proposal_sd: Gaussian scales in deterministic fixed
            coefficient order.
        root_slice_width: Initial stepping-out bracket width in log-total
            coordinates.
        root_slice_max_steps: Finite stepping-out budget, including the
            initial bracket.
        root_slice_max_shrink_steps: Maximum shrinkage candidate draws.

    Raises:
        TypeError: If an integer setting has the wrong type.
        ValueError: If a setting lies outside its supported range.
    """

    fixed_k: int
    pair_allocation_refresh_slots: int
    fixed_coefficient_proposal_sd: tuple[float, ...]
    root_slice_width: float = 1.0
    root_slice_max_steps: int = 100
    root_slice_max_shrink_steps: int = 1000

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
        root_slice_width = _positive_float(
            self.root_slice_width,
            name="root_slice_width",
        )
        root_slice_max_steps = _positive_integer(
            self.root_slice_max_steps,
            name="root_slice_max_steps",
        )
        if not isfinite(root_slice_width * root_slice_max_steps):
            raise ValueError("root_slice_width times root_slice_max_steps must be finite.")
        object.__setattr__(self, "root_slice_width", root_slice_width)
        object.__setattr__(self, "root_slice_max_steps", root_slice_max_steps)
        object.__setattr__(
            self,
            "root_slice_max_shrink_steps",
            _positive_integer(
                self.root_slice_max_shrink_steps,
                name="root_slice_max_shrink_steps",
            ),
        )

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
    A root slice row is recorded as valid and accepted with
    ``log_acceptance_ratio == 0`` even though it is a direct conditional
    update rather than a Metropolis-Hastings decision. Historical v1
    ``root_total_refresh`` move labels remain readable, but cannot be resumed
    by the v2 checkpoint kernel.

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
                    "root_total_slice",
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
class FullTilingMovementDiagnostics:
    """Optional output-only movement measurements for every atomic transition.

    Candidate movement is recorded even when a valid Metropolis-Hastings
    proposal is rejected. Invalid proposals have zero movement and slice
    counters, and use ``-1`` for ``fixed_position``. Catalogue sizes describe
    the deterministic source-side choices already visited by proposal
    construction; diagnostics never enumerate merge choices themselves.
    Timing values are observational and are not part of scientific replay.
    Displacement norms may be positive infinity when the corresponding true
    finite-precision magnitude exceeds the largest representable float; this
    sentinel does not alter the chain.

    Args:
        global_transition: Consecutive one-based transition coordinates.
        move: Concrete proposal or slice-kernel name.
        valid: Whether an eligible candidate was constructed.
        accepted: Whether the candidate became the visited state.
        proposal_elapsed_ns: Proposal construction or slice-update elapsed
            nanoseconds.
        diagnostic_elapsed_ns: Movement-metric calculation elapsed
            nanoseconds.
        source_merge_count: Source merge-choice count for structural slots.
        destination_catalogue_size: Fixed relocation destination catalogue
            size, zero for other moves.
        pair_catalogue_size: Unordered allocation-pair catalogue size, zero
            for other moves.
        design_cache_misses: Rectangle design columns newly cached while
            constructing the proposal in the current process segment. This is
            a cost metric, not restart-stable Markov state.
        changed_native_cell_count: Native-cell area of source rectangles
            absent from the candidate tiling.
        changed_nominal_mass: Normalized nominal mass of those source
            rectangles.
        standardized_prediction_l2: Candidate-minus-source prediction norm
            after elementwise division by observation standard deviation.
        root_abs_displacement: Absolute candidate root-total displacement.
        root_abs_log_displacement: Absolute candidate log-root displacement.
        allocation_share_l1_displacement: L1 distance between source and
            candidate rectangle-keyed allocation-share vectors.
        fixed_position: Changed fixed coefficient position, or ``-1``.
        fixed_abs_displacement: Absolute fixed-coefficient displacement.
        fixed_abs_log_displacement: Absolute log fixed-coefficient
            displacement.
        slice_left_steps: Successful left stepping-out extensions.
        slice_right_steps: Successful right stepping-out extensions.
        slice_shrink_draws: Candidate draws made during slice shrinkage.
        slice_log_density_evaluations: Conditional log-density evaluations in
            the root slice update.

    Raises:
        ValueError: If arrays are not aligned or violate movement, catalogue,
            timing, sentinel, or slice-counter invariants.
    """

    global_transition: NDArray[np.int64]
    move: NDArray[np.str_]
    valid: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    proposal_elapsed_ns: NDArray[np.int64]
    diagnostic_elapsed_ns: NDArray[np.int64]
    source_merge_count: NDArray[np.int64]
    destination_catalogue_size: NDArray[np.int64]
    pair_catalogue_size: NDArray[np.int64]
    design_cache_misses: NDArray[np.int64]
    changed_native_cell_count: NDArray[np.int64]
    changed_nominal_mass: NDArray[np.float64]
    standardized_prediction_l2: NDArray[np.float64]
    root_abs_displacement: NDArray[np.float64]
    root_abs_log_displacement: NDArray[np.float64]
    allocation_share_l1_displacement: NDArray[np.float64]
    fixed_position: NDArray[np.int64]
    fixed_abs_displacement: NDArray[np.float64]
    fixed_abs_log_displacement: NDArray[np.float64]
    slice_left_steps: NDArray[np.int64]
    slice_right_steps: NDArray[np.int64]
    slice_shrink_draws: NDArray[np.int64]
    slice_log_density_evaluations: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Copy arrays read-only and enforce per-attempt invariants."""
        integer_fields = (
            "global_transition",
            "proposal_elapsed_ns",
            "diagnostic_elapsed_ns",
            "source_merge_count",
            "destination_catalogue_size",
            "pair_catalogue_size",
            "design_cache_misses",
            "changed_native_cell_count",
            "fixed_position",
            "slice_left_steps",
            "slice_right_steps",
            "slice_shrink_draws",
            "slice_log_density_evaluations",
        )
        float_fields = (
            "changed_nominal_mass",
            "standardized_prediction_l2",
            "root_abs_displacement",
            "root_abs_log_displacement",
            "allocation_share_l1_displacement",
            "fixed_abs_displacement",
            "fixed_abs_log_displacement",
        )
        for name in integer_fields:
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), dtype=np.int64, ndim=1, name=name),
            )
        for name in float_fields:
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), dtype=np.float64, ndim=1, name=name),
            )
        object.__setattr__(
            self,
            "move",
            _readonly_array(self.move, dtype=np.dtype("U24"), ndim=1, name="move"),
        )
        for name in ("valid", "accepted"):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), dtype=np.bool_, ndim=1, name=name),
            )

        size = self.global_transition.size
        for name in (*integer_fields, *float_fields, "move", "valid", "accepted"):
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have one entry per atomic transition.")
        if size and (np.any(self.global_transition < 1) or np.any(np.diff(self.global_transition) != 1)):
            raise ValueError("global_transition must contain consecutive positive coordinates.")
        supported_moves = (
            "edge_flip",
            "resolution_relocation",
            "root_total_slice",
            "pair_allocation_refresh",
            "fixed_coefficient",
        )
        if np.any(~np.isin(self.move, supported_moves)):
            raise ValueError("move contains an unsupported full-tiling kernel.")
        if np.any(self.accepted & ~self.valid):
            raise ValueError("accepted transitions must be valid.")
        nonnegative_integer_fields = tuple(
            name for name in integer_fields if name not in ("global_transition", "fixed_position")
        )
        if any(np.any(getattr(self, name) < 0) for name in nonnegative_integer_fields):
            raise ValueError("timings, catalogue sizes, counts, and slice counters must be non-negative.")
        if any(
            np.any(np.isnan(getattr(self, name))) or np.any(getattr(self, name) < 0.0)
            for name in float_fields
        ):
            raise ValueError("movement values must be non-negative and cannot contain NaN.")
        if np.any(~np.isfinite(self.changed_nominal_mass)) or np.any(self.changed_nominal_mass > 1.0):
            raise ValueError("changed_nominal_mass cannot exceed one.")
        invalid = ~self.valid
        movement_fields = (
            "changed_native_cell_count",
            *float_fields,
            "slice_left_steps",
            "slice_right_steps",
            "slice_shrink_draws",
            "slice_log_density_evaluations",
        )
        if any(np.any(getattr(self, name)[invalid] != 0) for name in movement_fields):
            raise ValueError("invalid proposals must have zero movement and slice counters.")
        if np.any(self.fixed_position[invalid] != -1):
            raise ValueError("invalid proposals must use the -1 fixed-position sentinel.")
        fixed = self.move == "fixed_coefficient"
        if np.any(self.fixed_position[~(fixed & self.valid)] != -1):
            raise ValueError("fixed_position is populated only for valid fixed proposals.")
        if np.any(self.fixed_position[fixed & self.valid] < 0):
            raise ValueError("valid fixed proposals must identify a fixed position.")
        root = self.move == "root_total_slice"
        for name in (
            "slice_left_steps",
            "slice_right_steps",
            "slice_shrink_draws",
            "slice_log_density_evaluations",
        ):
            if np.any(getattr(self, name)[~root] != 0):
                raise ValueError("root slice counters must be zero off root slots.")


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
        movement_diagnostics: Optional output-only per-transition metrics.
    """

    trace: FullTilingCompoundTrace
    final_state: FullTilingPosteriorState
    checkpoint: FullTilingCompoundCheckpoint
    movement_diagnostics: FullTilingMovementDiagnostics | None = None


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
) -> tuple[PosteriorTransitionTerms, "_ProposalCatalogueStats"]:
    """Draw one fixed half-and-half structural proposal.

    The component choice is never availability-renormalized.  Only source
    merge choices are enumerated.  Relocation uses the fixed intermediate
    ``leaf × axis`` catalogue, retaining invalid entries as explicit attempts.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        rng: Generator advanced by the component and proposal-specific draws.

    Returns:
        Complete deterministic posterior transition terms and source
        catalogue sizes already encountered while drawing them.
    """
    choose_edge_flip = float(rng.random()) < 0.5
    merges = merge_choices(state.tiling_state.tiling)
    merge_count = len(merges)
    if not merges:
        merge = _invalid_merge_choice()
        if choose_edge_flip:
            return (
                propose_posterior_edge_flip(
                    problem,
                    state,
                    merge_choice=merge,
                    new_fraction=0.5,
                ),
                _ProposalCatalogueStats(source_merge_count=merge_count),
            )
        return (
            propose_posterior_resolution_relocation(
                problem,
                state,
                merge_choice=merge,
                split_choice=_invalid_split_choice(),
                new_fraction=0.5,
            ),
            _ProposalCatalogueStats(
                source_merge_count=merge_count,
                destination_catalogue_size=2 * (state.k - 1),
            ),
        )

    merge = merges[int(rng.integers(len(merges)))]
    if choose_edge_flip:
        target_axis: Axis = "vertical" if merge.axis == "horizontal" else "horizontal"
        if target_axis in merge.parent.admissible_axes:
            children = merge.parent.midpoint_children(target_axis)
            fraction = _draw_beta(problem, children[0], children[1], rng=rng)
        else:
            fraction = 0.5
        return (
            propose_posterior_edge_flip(
                problem,
                state,
                merge_choice=merge,
                new_fraction=fraction,
            ),
            _ProposalCatalogueStats(source_merge_count=merge_count),
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
    return (
        propose_posterior_resolution_relocation(
            problem,
            state,
            merge_choice=merge,
            split_choice=split,
            new_fraction=fraction,
        ),
        _ProposalCatalogueStats(
            source_merge_count=merge_count,
            destination_catalogue_size=len(catalogue),
        ),
    )


def _draw_pair_refresh_transition(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    rng: np.random.Generator,
) -> tuple[PosteriorTransitionTerms, "_ProposalCatalogueStats"]:
    """Draw one uniform unordered-pair additive-alpha refresh.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        rng: Generator advanced by pair selection and an eligible Beta draw.

    Returns:
        Complete pair-refresh transition terms.  ``K=1`` is an explicit
        invalid self-attempt through the deterministic proposal API, paired
        with the already calculated pair-catalogue size.
    """
    leaves = state.tiling_state.tiling.leaves
    pair_count = state.k * (state.k - 1) // 2
    if pair_count == 0:
        return (
            propose_pair_allocation_refresh(
                problem,
                state,
                first_leaf=_invalid_split_choice().leaf,
                second_leaf=_invalid_merge_choice().parent,
                new_fraction=0.5,
            ),
            _ProposalCatalogueStats(pair_catalogue_size=pair_count),
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
    return (
        propose_pair_allocation_refresh(
            problem,
            state,
            first_leaf=first,
            second_leaf=second,
            new_fraction=fraction,
        ),
        _ProposalCatalogueStats(pair_catalogue_size=pair_count),
    )


@dataclass(frozen=True, slots=True)
class _ProposalCatalogueStats:
    """Catalogue sizes gathered without repeating proposal enumeration."""

    source_merge_count: int = 0
    destination_catalogue_size: int = 0
    pair_catalogue_size: int = 0


@dataclass(frozen=True, slots=True)
class _RootSliceCounters:
    """Variable-work counters from one log-root slice update."""

    left_steps: int
    right_steps: int
    shrink_draws: int
    log_density_evaluations: int


class _RandomSource(Protocol):
    """Minimal random interface required by the root slice update."""

    def random(self) -> float:
        """Return one uniform draw."""
        ...


def _draw_root_total_slice(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    width: float,
    max_steps: int,
    max_shrink_steps: int,
    rng: _RandomSource,
) -> tuple[FullTilingPosteriorState, _RootSliceCounters]:
    """Draw one exact finite stepping-out/shrinkage update in ``z=log(T)``.

    The conditional posterior function already includes the ``+z`` chart
    Jacobian. The finite stepping-out allocation is randomized as in the
    standard exact slice construction: at most ``max_steps - 1`` extensions
    are divided between the left and right endpoints. No Metropolis-Hastings
    acceptance uniform is drawn after a slice candidate is obtained.

    Args:
        problem: Full-tiling posterior target.
        source: State supplying geometry, allocation shares, and fixed block.
        width: Positive initial log-total bracket width.
        max_steps: Positive total finite stepping-out budget, including the
            initial bracket.
        max_shrink_steps: Positive candidate-draw guard.
        rng: Generator advanced by slice-specific variable draws.

    Returns:
        Accepted posterior state and exact variable-work counters.

    Raises:
        RuntimeError: If the current point lacks finite conditional density,
            an accepted log total cannot be represented as a positive
            scientific total, or the shrinkage guard is exhausted.
    """

    evaluations: int = 0

    def density(log_root_total: float) -> float:
        """Evaluate and count the exact conditional log density."""
        nonlocal evaluations
        evaluations += 1
        return log_root_total_slice_density(
            problem,
            source,
            log_root_total=log_root_total,
        )

    current = log(source.root_total)
    current_density = density(current)
    if not isfinite(current_density):
        raise RuntimeError("root slice current point must have finite log density.")

    height_uniform = float(rng.random())
    log_height = -np.inf if height_uniform == 0.0 else current_density + log(height_uniform)
    bracket_uniform = float(rng.random())
    left = current - width * bracket_uniform
    right = left + width

    allocation_uniform = float(rng.random())
    left_budget = min(int(max_steps * allocation_uniform), max_steps - 1)
    right_budget = max_steps - 1 - left_budget
    left_steps = 0
    while left_budget > 0 and density(left) > log_height:
        left -= width
        left_budget -= 1
        left_steps += 1
    right_steps = 0
    while right_budget > 0 and density(right) > log_height:
        right += width
        right_budget -= 1
        right_steps += 1

    for shrink_draws in range(1, max_shrink_steps + 1):
        candidate_uniform = float(rng.random())
        candidate_log_total = left + candidate_uniform * (right - left)
        candidate_density = density(candidate_log_total)
        if candidate_density > -np.inf and candidate_density >= log_height:
            candidate_total = float(np.exp(candidate_log_total))
            if not isfinite(candidate_total) or candidate_total <= 0.0:
                raise RuntimeError(
                    "root slice accepted a log total outside representable scientific support."
                )
            candidate = rescale_full_tiling_root_total(
                problem,
                source,
                new_root_total=candidate_total,
            )
            return candidate, _RootSliceCounters(
                left_steps=left_steps,
                right_steps=right_steps,
                shrink_draws=shrink_draws,
                log_density_evaluations=evaluations,
            )
        if candidate_log_total < current:
            left = candidate_log_total
        elif candidate_log_total > current:
            right = candidate_log_total
        else:
            raise RuntimeError("root slice shrinkage made no progress.")
    raise RuntimeError(f"root slice exceeded root_slice_max_shrink_steps={max_shrink_steps}.")


def _draw_transition(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    *,
    phase: int,
    settings: FullTilingCompoundKernelSettings,
    rng: np.random.Generator,
) -> tuple[
    FullTilingCompoundSlot,
    PosteriorTransitionTerms,
    _ProposalCatalogueStats,
    int,
]:
    """Draw the proposal assigned to one global schedule phase.

    Args:
        problem: Full-tiling posterior target.
        state: Source posterior state.
        phase: Zero-based phase within one cycle.
        settings: Immutable problem-resolved kernel settings.
        rng: Generator advanced by proposal-specific draws.

    Returns:
        Stable slot name, complete deterministic transition terms, catalogue
        sizes already calculated by proposal construction, and the attempted
        fixed position or ``-1``.

    Raises:
        RuntimeError: If called for the separately handled slice slot or if a
            fixed phase does not map to a coefficient.
    """
    slot, fixed_position = _slot_at_phase(phase, settings)
    if slot == "structural":
        transition, catalogue_stats = _draw_structural_transition(
            problem,
            state,
            rng=rng,
        )
        return slot, transition, catalogue_stats, -1
    if slot == "root":
        raise RuntimeError("root slice slots must be handled without a final acceptance uniform.")
    if slot == "pair_allocation":
        transition, catalogue_stats = _draw_pair_refresh_transition(
            problem,
            state,
            rng=rng,
        )
        return slot, transition, catalogue_stats, -1
    if fixed_position is None:
        raise RuntimeError("fixed schedule slot did not resolve a coefficient position.")
    proposal_sd = settings.fixed_coefficient_proposal_sd[fixed_position]
    proposed = float(
        rng.normal(
            loc=float(state.fixed_coefficients[fixed_position]),
            scale=proposal_sd,
        )
    )
    return (
        slot,
        propose_fixed_coefficient(
            problem,
            state,
            coefficient_position=fixed_position,
            proposed_coefficient=proposed,
            proposal_stdev=proposal_sd,
        ),
        _ProposalCatalogueStats(),
        fixed_position,
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


def _scaled_l2_norm(values: NDArray[np.float64]) -> float:
    """Return an overflow-resistant Euclidean norm."""
    absolute = np.abs(values)
    scale = float(np.max(absolute, initial=0.0))
    if scale == 0.0:
        return 0.0
    if not isfinite(scale):
        return np.inf
    return float(scale * np.sqrt(np.sum((absolute / scale) ** 2)))


@dataclass(frozen=True, slots=True)
class _MovementMetrics:
    """Candidate movement calculated only from existing state caches."""

    changed_native_cell_count: int = 0
    changed_nominal_mass: float = 0.0
    standardized_prediction_l2: float = 0.0
    root_abs_displacement: float = 0.0
    root_abs_log_displacement: float = 0.0
    allocation_share_l1_displacement: float = 0.0
    fixed_position: int = -1
    fixed_abs_displacement: float = 0.0
    fixed_abs_log_displacement: float = 0.0


def _movement_metrics(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    candidate: FullTilingPosteriorState,
    *,
    valid: bool,
    fixed_position: int,
) -> _MovementMetrics:
    """Calculate proposal movement without rebuilding geometry or predictions.

    Args:
        problem: Problem supplying nominal masses and observation scales.
        source: State before the attempted transition.
        candidate: Proposed state, which may equal ``source`` when invalid.
        valid: Whether candidate movement is meaningful.
        fixed_position: Attempted fixed position, or ``-1`` for other slots.

    Returns:
        Zero movement for invalid attempts or metrics from existing source and
        candidate caches for valid proposals.
    """
    if not valid:
        return _MovementMetrics()

    source_leaves = source.tiling_state.tiling.leaves
    candidate_leaf_set = set(candidate.tiling_state.tiling.leaves)
    changed_source_leaves = tuple(leaf for leaf in source_leaves if leaf not in candidate_leaf_set)
    changed_native_cell_count = sum(leaf.area for leaf in changed_source_leaves)
    changed_nominal_mass = min(
        1.0,
        max(
            0.0,
            fsum(problem.rectangle_nominal_mass(leaf) for leaf in changed_source_leaves),
        ),
    )

    source_shares = {
        leaf: float(mass / source.root_total)
        for leaf, mass in zip(source_leaves, source.leaf_masses, strict=True)
    }
    candidate_shares = {
        leaf: float(mass / candidate.root_total)
        for leaf, mass in zip(
            candidate.tiling_state.tiling.leaves,
            candidate.leaf_masses,
            strict=True,
        )
    }
    share_l1 = sum(
        abs(source_shares.get(leaf, 0.0) - candidate_shares.get(leaf, 0.0))
        for leaf in source_shares.keys() | candidate_shares.keys()
    )
    prediction_change = candidate.prediction - source.prediction
    root_abs_displacement = abs(candidate.root_total - source.root_total)
    root_abs_log_displacement = abs(log(candidate.root_total) - log(source.root_total))

    diagnosed_fixed_position = -1
    fixed_abs_displacement = 0.0
    fixed_abs_log_displacement = 0.0
    if fixed_position >= 0:
        diagnosed_fixed_position = fixed_position
        source_fixed = float(source.fixed_coefficients[fixed_position])
        candidate_fixed = float(candidate.fixed_coefficients[fixed_position])
        fixed_abs_displacement = abs(candidate_fixed - source_fixed)
        fixed_abs_log_displacement = abs(log(candidate_fixed) - log(source_fixed))

    return _MovementMetrics(
        changed_native_cell_count=changed_native_cell_count,
        changed_nominal_mass=changed_nominal_mass,
        standardized_prediction_l2=_scaled_l2_norm(prediction_change / problem.observation_sd),
        root_abs_displacement=root_abs_displacement,
        root_abs_log_displacement=root_abs_log_displacement,
        allocation_share_l1_displacement=share_l1,
        fixed_position=diagnosed_fixed_position,
        fixed_abs_displacement=fixed_abs_displacement,
        fixed_abs_log_displacement=fixed_abs_log_displacement,
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
    collect_movement_diagnostics: bool,
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
        collect_movement_diagnostics: Whether to collect output-only metrics.

    Returns:
        Segment trace, final state, and exact next checkpoint.

    Raises:
        ValueError: If state identity, target support, fixed ``K``, proposal
            scale width, or pair-refresh availability is inconsistent.

    Notes:
        Each non-slice loop iteration consumes one acceptance uniform after
        all proposal-specific draws, including invalid attempts. Root slice
        slots do not consume a final Metropolis-Hastings uniform.
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
    diagnostic_rows: dict[str, list[int | float | str | bool]] | None = None
    if collect_movement_diagnostics:
        diagnostic_rows = {
            name: []
            for name in (
                "global_transition",
                "move",
                "valid",
                "accepted",
                "proposal_elapsed_ns",
                "diagnostic_elapsed_ns",
                "source_merge_count",
                "destination_catalogue_size",
                "pair_catalogue_size",
                "design_cache_misses",
                "changed_native_cell_count",
                "changed_nominal_mass",
                "standardized_prediction_l2",
                "root_abs_displacement",
                "root_abs_log_displacement",
                "allocation_share_l1_displacement",
                "fixed_position",
                "fixed_abs_displacement",
                "fixed_abs_log_displacement",
                "slice_left_steps",
                "slice_right_steps",
                "slice_shrink_draws",
                "slice_log_density_evaluations",
            )
        }
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
        slot, _ = _slot_at_phase(phase, settings)
        cache_size_before = len(problem._design_cache) if diagnostic_rows is not None else 0
        proposal_started = perf_counter_ns() if diagnostic_rows is not None else 0
        if slot == "root":
            state, slice_counters = _draw_root_total_slice(
                problem,
                source,
                width=settings.root_slice_width,
                max_steps=settings.root_slice_max_steps,
                max_shrink_steps=settings.root_slice_max_shrink_steps,
                rng=rng,
            )
            candidate = state
            move = "root_total_slice"
            transition_valid = True
            proposal_accepted = True
            transition_log_acceptance_ratio = 0.0
            transition_reason = ""
            catalogue_stats = _ProposalCatalogueStats()
            fixed_position = -1
        else:
            (
                slot,
                transition,
                catalogue_stats,
                fixed_position,
            ) = _draw_transition(
                problem,
                source,
                phase=phase,
                settings=settings,
                rng=rng,
            )
            uniform = float(rng.random())
            log_uniform = -np.inf if uniform == 0.0 else log(uniform)
            state = accept_or_reject(source, transition, log_uniform=log_uniform)
            candidate = transition.candidate
            move = transition.move
            transition_valid = transition.valid
            proposal_accepted = transition.valid and state is transition.candidate
            transition_log_acceptance_ratio = transition.log_acceptance_ratio
            transition_reason = "" if transition.reason is None else transition.reason
            slice_counters = _RootSliceCounters(0, 0, 0, 0)
        proposal_elapsed = perf_counter_ns() - proposal_started if diagnostic_rows is not None else 0

        slots[iteration] = slot
        moves[iteration] = move
        valid[iteration] = transition_valid
        accepted[iteration] = proposal_accepted
        log_acceptance_ratio[iteration] = transition_log_acceptance_ratio
        invalid_reason[iteration] = transition_reason
        completed = transitions_completed + iteration + 1
        if diagnostic_rows is not None:
            diagnostic_started = perf_counter_ns()
            metrics = _movement_metrics(
                problem,
                source,
                candidate,
                valid=transition_valid,
                fixed_position=fixed_position,
            )
            diagnostic_elapsed = perf_counter_ns() - diagnostic_started
            row: dict[str, int | float | str | bool] = {
                "global_transition": completed,
                "move": move,
                "valid": transition_valid,
                "accepted": proposal_accepted,
                "proposal_elapsed_ns": proposal_elapsed,
                "diagnostic_elapsed_ns": diagnostic_elapsed,
                "source_merge_count": catalogue_stats.source_merge_count,
                "destination_catalogue_size": catalogue_stats.destination_catalogue_size,
                "pair_catalogue_size": catalogue_stats.pair_catalogue_size,
                "design_cache_misses": len(problem._design_cache) - cache_size_before,
                "changed_native_cell_count": metrics.changed_native_cell_count,
                "changed_nominal_mass": metrics.changed_nominal_mass,
                "standardized_prediction_l2": metrics.standardized_prediction_l2,
                "root_abs_displacement": metrics.root_abs_displacement,
                "root_abs_log_displacement": metrics.root_abs_log_displacement,
                "allocation_share_l1_displacement": (metrics.allocation_share_l1_displacement),
                "fixed_position": metrics.fixed_position,
                "fixed_abs_displacement": metrics.fixed_abs_displacement,
                "fixed_abs_log_displacement": metrics.fixed_abs_log_displacement,
                "slice_left_steps": slice_counters.left_steps,
                "slice_right_steps": slice_counters.right_steps,
                "slice_shrink_draws": slice_counters.shrink_draws,
                "slice_log_density_evaluations": (slice_counters.log_density_evaluations),
            }
            for name, value in row.items():
                diagnostic_rows[name].append(value)
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
    movement_diagnostics = None
    if diagnostic_rows is not None:
        movement_diagnostics = FullTilingMovementDiagnostics(
            global_transition=np.asarray(
                diagnostic_rows["global_transition"],
                dtype=np.int64,
            ),
            move=np.asarray(diagnostic_rows["move"], dtype="U24"),
            valid=np.asarray(diagnostic_rows["valid"], dtype=np.bool_),
            accepted=np.asarray(diagnostic_rows["accepted"], dtype=np.bool_),
            proposal_elapsed_ns=np.asarray(
                diagnostic_rows["proposal_elapsed_ns"],
                dtype=np.int64,
            ),
            diagnostic_elapsed_ns=np.asarray(
                diagnostic_rows["diagnostic_elapsed_ns"],
                dtype=np.int64,
            ),
            source_merge_count=np.asarray(
                diagnostic_rows["source_merge_count"],
                dtype=np.int64,
            ),
            destination_catalogue_size=np.asarray(
                diagnostic_rows["destination_catalogue_size"],
                dtype=np.int64,
            ),
            pair_catalogue_size=np.asarray(
                diagnostic_rows["pair_catalogue_size"],
                dtype=np.int64,
            ),
            design_cache_misses=np.asarray(
                diagnostic_rows["design_cache_misses"],
                dtype=np.int64,
            ),
            changed_native_cell_count=np.asarray(
                diagnostic_rows["changed_native_cell_count"],
                dtype=np.int64,
            ),
            changed_nominal_mass=np.asarray(
                diagnostic_rows["changed_nominal_mass"],
                dtype=np.float64,
            ),
            standardized_prediction_l2=np.asarray(
                diagnostic_rows["standardized_prediction_l2"],
                dtype=np.float64,
            ),
            root_abs_displacement=np.asarray(
                diagnostic_rows["root_abs_displacement"],
                dtype=np.float64,
            ),
            root_abs_log_displacement=np.asarray(
                diagnostic_rows["root_abs_log_displacement"],
                dtype=np.float64,
            ),
            allocation_share_l1_displacement=np.asarray(
                diagnostic_rows["allocation_share_l1_displacement"],
                dtype=np.float64,
            ),
            fixed_position=np.asarray(
                diagnostic_rows["fixed_position"],
                dtype=np.int64,
            ),
            fixed_abs_displacement=np.asarray(
                diagnostic_rows["fixed_abs_displacement"],
                dtype=np.float64,
            ),
            fixed_abs_log_displacement=np.asarray(
                diagnostic_rows["fixed_abs_log_displacement"],
                dtype=np.float64,
            ),
            slice_left_steps=np.asarray(
                diagnostic_rows["slice_left_steps"],
                dtype=np.int64,
            ),
            slice_right_steps=np.asarray(
                diagnostic_rows["slice_right_steps"],
                dtype=np.int64,
            ),
            slice_shrink_draws=np.asarray(
                diagnostic_rows["slice_shrink_draws"],
                dtype=np.int64,
            ),
            slice_log_density_evaluations=np.asarray(
                diagnostic_rows["slice_log_density_evaluations"],
                dtype=np.int64,
            ),
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
        movement_diagnostics=movement_diagnostics,
    )


def sample_full_tiling_compound(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    config: FullTilingCompoundConfig,
    *,
    collect_movement_diagnostics: bool = False,
) -> FullTilingCompoundSamplingResult:
    """Run a fresh seeded fixed-``K`` full-tiling posterior segment.

    Args:
        problem: Full-tiling observation model and normalized priors.
        initial_state: State built for the exact supplied problem object.
        config: Seed, atomic transition count, pair opportunities, and fixed
            Gaussian proposal scales.
        collect_movement_diagnostics: Whether to attach output-only
            per-transition movement metrics.

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
    if not isinstance(collect_movement_diagnostics, bool):
        raise TypeError("collect_movement_diagnostics must be a Boolean.")
    scales = _resolve_fixed_scales(
        config.fixed_coefficient_proposal_sd,
        n_fixed=initial_state.fixed_coefficients.size,
    )
    settings = FullTilingCompoundKernelSettings(
        fixed_k=initial_state.k,
        pair_allocation_refresh_slots=config.pair_allocation_refresh_slots,
        fixed_coefficient_proposal_sd=scales,
        root_slice_width=config.root_slice_width,
        root_slice_max_steps=config.root_slice_max_steps,
        root_slice_max_shrink_steps=config.root_slice_max_shrink_steps,
    )
    return _run_segment(
        problem,
        initial_state,
        iterations=config.iterations,
        rng=np.random.Generator(np.random.PCG64(config.seed)),
        transitions_completed=0,
        settings=settings,
        include_initial=True,
        collect_movement_diagnostics=collect_movement_diagnostics,
    )


def continue_full_tiling_compound(
    problem: FullTilingProblem,
    checkpoint: FullTilingCompoundCheckpoint,
    *,
    iterations: int,
    collect_movement_diagnostics: bool = False,
) -> FullTilingCompoundSamplingResult:
    """Continue a full-tiling chain exactly from an in-memory checkpoint.

    Args:
        problem: Exact problem object retained by ``checkpoint``.
        checkpoint: In-memory continuation boundary.
        iterations: Positive number of additional atomic transitions.
        collect_movement_diagnostics: Whether to attach output-only metrics
            for this segment. The choice is not persisted in the checkpoint.

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
    if not isinstance(collect_movement_diagnostics, bool):
        raise TypeError("collect_movement_diagnostics must be a Boolean.")
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
        collect_movement_diagnostics=collect_movement_diagnostics,
    )


__all__ = [
    "FULL_TILING_COMPOUND_SCHEDULE_ID",
    "FullTilingCompoundCheckpoint",
    "FullTilingCompoundConfig",
    "FullTilingCompoundKernelSettings",
    "FullTilingCompoundSamplingResult",
    "FullTilingCompoundSlot",
    "FullTilingCompoundTrace",
    "FullTilingMovementDiagnostics",
    "continue_full_tiling_compound",
    "sample_full_tiling_compound",
]
