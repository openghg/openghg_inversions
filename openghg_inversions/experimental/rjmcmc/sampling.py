"""Seeded reference sampler for spatial trans-dimensional MCMC.

Without an always-active fixed block, the single-chain driver repeats the
original four slots: a dynamic coefficient update, two identical mixed
birth/death slots, and a configurable nucleus-location slot. A problem with a
fixed block uses a versioned five-slot schedule that inserts one uniformly
selected fixed-coefficient update after the dynamic coefficient slot. Both use
a seeded NumPy PCG64 generator, and each dimension slot selects birth or death
independently with equal probability. By default, state traces include the
initial state and every subsequent state. Optional collection-time retention
saves a global warmup/thinning subsequence while preserving diagnostics for
every attempted transition. In-memory checkpoints preserve the random stream,
schedule phase, and retention phase for exact continuation. Durable
serialization, parallel chains, and parallel tempering are not yet provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.core import (
    Backend,
    TransDimensionalProblem,
    TransDimensionalState,
)
from openghg_inversions.experimental.rjmcmc.proposals import (
    TransitionTerms,
    accept_or_reject,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_fixed_coefficient,
    propose_global_move,
    propose_local_move,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings

SCHEDULE_ID = "four_slot_coefficient_dimension_dimension_nucleus_v1"
FIXED_BLOCK_SCHEDULE_ID = "five_slot_coefficient_fixed_coefficient_dimension_dimension_nucleus_v1"


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for the first single-chain reference sampler.

    Problems without a fixed block retain the original four-slot schedule.
    Problems with one insert a fixed-coefficient proposal as the second slot.

    Args:
        iterations: Number of attempted transitions.
        coefficient_proposal_sd: Gaussian random-walk scale for coefficients.
        birth_proposal_sd: Gaussian auxiliary-coefficient scale for birth/death.
        fixed_coefficient_proposal_sd: Gaussian random-walk scale for
            always-active fixed-block coefficients. ``None`` reuses
            ``coefficient_proposal_sd``. This setting does not alter the
            four-slot schedule when ``problem`` has no fixed block.
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
    fixed_coefficient_proposal_sd: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed sampler settings before allocating a trace."""
        if isinstance(self.iterations, bool) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer.")
        for name in ("coefficient_proposal_sd", "birth_proposal_sd"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.fixed_coefficient_proposal_sd is not None:
            fixed_coefficient_proposal_sd = float(self.fixed_coefficient_proposal_sd)
            if not isfinite(fixed_coefficient_proposal_sd) or fixed_coefficient_proposal_sd <= 0.0:
                raise ValueError("fixed_coefficient_proposal_sd must be finite and positive.")
            object.__setattr__(
                self,
                "fixed_coefficient_proposal_sd",
                fixed_coefficient_proposal_sd,
            )
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


@dataclass(frozen=True, slots=True)
class KernelSettings:
    """Immutable settings that define the transition kernel.

    Unlike :class:`SamplerConfig`, these settings exclude the segment length
    and initial seed. A checkpoint owns them so continuation cannot silently
    change the kernel midway through a chain.
    """

    coefficient_proposal_sd: float
    birth_proposal_sd: float
    backend: Backend
    nucleus_move: Literal["global", "local"]
    local_move_scale: float | None
    fixed_coefficient_proposal_sd: float | None = None

    @classmethod
    def from_config(cls, config: SamplerConfig) -> KernelSettings:
        """Extract immutable kernel settings from a sampler configuration."""
        return cls(
            coefficient_proposal_sd=config.coefficient_proposal_sd,
            birth_proposal_sd=config.birth_proposal_sd,
            backend=config.backend,
            nucleus_move=config.nucleus_move,
            local_move_scale=config.local_move_scale,
            fixed_coefficient_proposal_sd=config.fixed_coefficient_proposal_sd,
        )


@dataclass(frozen=True, slots=True)
class PCG64State:
    """Immutable exact state of NumPy's PCG64 bit generator."""

    state: int
    increment: int
    has_uint32: int
    uinteger: int
    algorithm: str = "PCG64"

    @classmethod
    def from_generator(cls, rng: np.random.Generator) -> PCG64State:
        """Capture an explicit PCG64 generator state without mutable mappings."""
        bit_generator = rng.bit_generator
        if not isinstance(bit_generator, np.random.PCG64):
            raise TypeError("RJMCMC continuation requires a NumPy PCG64 generator.")
        raw = bit_generator.state
        nested = raw["state"]
        if not isinstance(nested, dict):
            raise TypeError("Unexpected PCG64 state representation.")
        return cls(
            state=int(nested["state"]),
            increment=int(nested["inc"]),
            has_uint32=int(raw["has_uint32"]),
            uinteger=int(raw["uinteger"]),
        )

    def generator(self) -> np.random.Generator:
        """Restore a new generator at exactly this state."""
        if self.algorithm != "PCG64":
            raise ValueError(f"Unsupported checkpoint RNG algorithm {self.algorithm!r}.")
        bit_generator = np.random.PCG64()
        bit_generator.state = {
            "bit_generator": self.algorithm,
            "state": {"state": self.state, "inc": self.increment},
            "has_uint32": self.has_uint32,
            "uinteger": self.uinteger,
        }
        return np.random.Generator(bit_generator)


@dataclass(frozen=True, slots=True)
class SamplerCheckpoint:
    """Exact in-memory continuation state at a transition boundary.

    The checkpoint deliberately excludes a durable problem fingerprint and
    serialization schema. Those are required before persisted checkpoints can
    safely be loaded against scientific inputs, but are outside this in-memory
    continuation stage.

    Attributes:
        problem: Exact in-memory problem object used by this chain. Continuation
            rejects an equal-but-distinct object until durable fingerprints are
            implemented.
        state: Fully cached state after ``transitions_completed`` transitions.
        rng_state: Exact explicit PCG64 state for the next random draw.
        transitions_completed: Global number of attempted transitions.
        kernel_settings: Immutable transition-kernel settings.
        retention: Immutable collection-time retention phase.
        schedule_id: Identifier for the global transition schedule.
    """

    problem: TransDimensionalProblem
    state: TransDimensionalState
    rng_state: PCG64State
    transitions_completed: int
    kernel_settings: KernelSettings
    retention: RetentionSettings
    schedule_id: str = SCHEDULE_ID


@dataclass(frozen=True)
class SamplingTrace:
    """Fixed-capacity retained states and segment transition diagnostics.

    Attributes:
        k: Active-region count at every retained state. With default retention,
            shape is ``(iterations + 1,)`` and row zero is the initial state.
        nuclei: Padded nucleus indices with shape
            ``(n_retained, k_max)``.
        coefficients: Padded region coefficients with shape
            ``(n_retained, k_max)``.
        fixed_coefficients: Always-active fixed-block coefficients with shape
            ``(n_retained, n_fixed_coefficients)``. A manually constructed
            legacy trace that omits this field receives a zero-width second
            dimension.
        log_target: Normalized log target at every saved state, with shape
            ``(n_retained,)``.
        moves: Proposal name for each attempted transition, with shape
            ``(iterations,)``. Entry ``i`` describes segment transition ``i``;
            it need not align with retained-state rows. Mixed dimension slots
            record the selected proposal label, ``"birth"`` or ``"death"``,
            rather than a generic dimension-slot label. Diagnostics always
            describe the transitions attempted in this sampling segment,
            including transitions whose resulting states were not retained.
        accepted: Whether each attempted transition changed the chain state,
            with shape ``(iterations,)``.
        log_acceptance_ratio: Untruncated log Metropolis-Hastings ratio for
            each attempted transition, with shape ``(iterations,)``.
        state_transition: Global completed-transition number for each retained
            state. It can be empty when a segment contains no retained states.
            A missing value in a manually constructed legacy trace defaults to
            consecutive indices starting at zero.
    """

    k: NDArray[np.int64]
    nuclei: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    log_target: NDArray[np.float64]
    moves: NDArray[np.str_]
    accepted: NDArray[np.bool_]
    log_acceptance_ratio: NDArray[np.float64]
    state_transition: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fixed_coefficients: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        """Supply safe defaults for legacy transition and fixed-block fields."""
        transition = np.asarray(self.state_transition, dtype=np.int64)
        if transition.size == 0 and self.k.size:
            transition = np.arange(self.k.size, dtype=np.int64)
            object.__setattr__(self, "state_transition", transition)
        else:
            object.__setattr__(self, "state_transition", transition)

        fixed_coefficients = np.asarray(self.fixed_coefficients, dtype=np.float64)
        if fixed_coefficients.ndim == 1 and fixed_coefficients.size == 0:
            fixed_coefficients = np.empty((self.k.size, 0), dtype=np.float64)
        elif fixed_coefficients.ndim != 2 or fixed_coefficients.shape[0] != self.k.size:
            raise ValueError("fixed_coefficients must have shape (n_retained, n_fixed_coefficients).")
        object.__setattr__(self, "fixed_coefficients", fixed_coefficients)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of attempted transitions that changed the chain state."""
        return float(np.mean(self.accepted))


@dataclass(frozen=True)
class SamplingResult:
    """Reference sampler output, final cached state, and continuation state.

    Attributes:
        trace: Fixed-capacity states and transition diagnostics for the chain.
        final_state: Fully cached final state corresponding to the last trace
            row under default retention. It remains available even when a
            retained trace is empty.
        checkpoint: Exact in-memory continuation state after the segment.
    """

    trace: SamplingTrace
    final_state: TransDimensionalState
    checkpoint: SamplerCheckpoint


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


def _fixed_coefficient_proposal_sd(config: SamplerConfig | KernelSettings) -> float:
    """Return the explicit or dynamic-coefficient fallback proposal scale."""
    scale = config.fixed_coefficient_proposal_sd
    return config.coefficient_proposal_sd if scale is None else scale


def _schedule_id(problem: TransDimensionalProblem) -> str:
    """Return the schedule identity implied by a problem's fixed block."""
    return FIXED_BLOCK_SCHEDULE_ID if problem.n_fixed_coefficients else SCHEDULE_ID


def _draw_transition(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    config: SamplerConfig | KernelSettings,
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

    if move == "fixed_coefficient":
        if not problem.n_fixed_coefficients:
            raise ValueError("fixed_coefficient moves require a nonempty fixed block.")
        proposal_stdev = _fixed_coefficient_proposal_sd(config)
        position = int(rng.integers(problem.n_fixed_coefficients))
        value = float(state.fixed_coefficients[position] + rng.normal(scale=proposal_stdev))
        return propose_fixed_coefficient(
            problem,
            state,
            coefficient_position=position,
            proposed_coefficient=value,
            proposal_stdev=proposal_stdev,
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


def _validate_start_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """Validate the state invariants required before a sampling segment."""
    if state.capacity != problem.k_max:
        raise ValueError("initial_state capacity must equal problem.k_max.")
    if state.fixed_coefficients.shape != (problem.n_fixed_coefficients,):
        raise ValueError("initial_state fixed coefficients are incompatible with the problem.")
    if not np.isfinite(state.log_target):
        raise ValueError("initial_state must have finite target density.")


def _retained_transition_numbers(
    *,
    transitions_completed: int,
    iterations: int,
    retention: RetentionSettings,
    include_initial: bool,
) -> NDArray[np.int64]:
    """Return global transition numbers retained by one sampling segment."""
    lower = transitions_completed if include_initial else transitions_completed + 1
    upper = transitions_completed + iterations
    first = max(lower, retention.warmup_transitions)
    phase_adjustment = (-(first - retention.warmup_transitions)) % retention.thin
    first += phase_adjustment
    if first > upper:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, upper + 1, retention.thin, dtype=np.int64)


def _run_segment(
    problem: TransDimensionalProblem,
    initial_state: TransDimensionalState,
    *,
    kernel_settings: KernelSettings,
    retention: RetentionSettings,
    rng: np.random.Generator,
    transitions_completed: int,
    iterations: int,
    include_initial: bool,
) -> SamplingResult:
    """Run one segment using global schedule and retention phases."""
    _validate_start_state(problem, initial_state)
    state_transition = _retained_transition_numbers(
        transitions_completed=transitions_completed,
        iterations=iterations,
        retention=retention,
        include_initial=include_initial,
    )
    n_retained = int(state_transition.size)
    capacity = problem.k_max
    k_trace = np.empty(n_retained, dtype=np.int64)
    nuclei_trace = np.empty((n_retained, capacity), dtype=np.int64)
    coefficient_trace = np.empty((n_retained, capacity), dtype=np.float64)
    fixed_coefficient_trace = np.empty(
        (n_retained, problem.n_fixed_coefficients),
        dtype=np.float64,
    )
    log_target_trace = np.empty(n_retained, dtype=np.float64)
    moves = np.empty(
        iterations,
        dtype="U17" if problem.n_fixed_coefficients else "U16",
    )
    accepted = np.zeros(iterations, dtype=np.bool_)
    log_acceptance_ratio = np.empty(iterations, dtype=np.float64)

    state = initial_state
    retained_position = 0

    def retain(current_state: TransDimensionalState) -> None:
        """Copy one cached state into the next fixed-capacity retained row."""
        nonlocal retained_position
        k_trace[retained_position] = current_state.k
        nuclei_trace[retained_position] = current_state.nuclei
        coefficient_trace[retained_position] = current_state.coefficients
        fixed_coefficient_trace[retained_position] = current_state.fixed_coefficients
        log_target_trace[retained_position] = current_state.log_target
        retained_position += 1

    if include_initial and n_retained and state_transition[0] == transitions_completed:
        retain(state)

    nucleus_move = "global_move" if kernel_settings.nucleus_move == "global" else "local_move"
    if problem.n_fixed_coefficients:
        schedule = (
            "coefficient",
            "fixed_coefficient",
            "dimension",
            "dimension",
            nucleus_move,
        )
    else:
        schedule = ("coefficient", "dimension", "dimension", nucleus_move)

    for iteration in range(iterations):
        global_transition = transitions_completed + iteration
        move = schedule[global_transition % len(schedule)]
        transition = _draw_transition(problem, state, kernel_settings, rng, move)
        uniform = float(rng.random())
        log_uniform = log(uniform) if uniform > 0.0 else -np.inf
        next_state = accept_or_reject(state, transition, log_uniform=log_uniform)
        accepted[iteration] = transition.valid and next_state is transition.candidate
        moves[iteration] = transition.move
        log_acceptance_ratio[iteration] = transition.log_acceptance_ratio
        state = next_state

        completed = global_transition + 1
        if retained_position < n_retained and state_transition[retained_position] == completed:
            retain(state)

    total_transitions = transitions_completed + iterations
    checkpoint = SamplerCheckpoint(
        problem=problem,
        state=state,
        rng_state=PCG64State.from_generator(rng),
        transitions_completed=total_transitions,
        kernel_settings=kernel_settings,
        retention=retention,
        schedule_id=_schedule_id(problem),
    )
    return SamplingResult(
        trace=SamplingTrace(
            k=k_trace,
            nuclei=nuclei_trace,
            coefficients=coefficient_trace,
            fixed_coefficients=fixed_coefficient_trace,
            log_target=log_target_trace,
            moves=moves,
            accepted=accepted,
            log_acceptance_ratio=log_acceptance_ratio,
            state_transition=state_transition,
        ),
        final_state=state,
        checkpoint=checkpoint,
    )


def sample(
    problem: TransDimensionalProblem,
    initial_state: TransDimensionalState,
    config: SamplerConfig,
    retention: RetentionSettings | None = None,
) -> SamplingResult:
    """Run an auditable single-chain spatial RJMCMC segment from transition zero.

    Without a fixed block, the original four-slot schedule repeats a dynamic
    coefficient move, two identical dimension moves, and a nucleus move. A
    problem with fixed columns inserts one uniformly selected fixed-coefficient
    move after the dynamic coefficient slot, creating a separately identified
    five-slot schedule. Each dimension slot independently chooses birth or
    death with probability one half, making it an invariant paired RJ kernel.
    The equal move-type probabilities cancel from the reported
    Metropolis-Hastings ratio. Nucleus moves are globally uniform by default;
    setting ``nucleus_move="local"`` uses a normalized discrete-Gaussian
    destination kernel instead. Impossible boundary proposals remain explicit
    self-transitions rather than renormalizing the birth/death selection.

    Omitting ``retention`` preserves the original behavior exactly: the initial
    state and every subsequent state are saved. Collection-time retention does
    not alter the random stream or transition diagnostics.

    Args:
        problem: Immutable target and fine-grid numerical inputs.
        initial_state: Complete initial state compatible with ``problem``.
        config: Transition count, scales, seed, and numerical backend.
        retention: Optional global warmup and thinning policy.

    Returns:
        Retained fixed-capacity states, all segment diagnostics, the final
        cached state, and an exact in-memory continuation checkpoint.

    Raises:
        ValueError: If the initial state capacity is incompatible with the
            problem or its target density is not finite.
    """
    retention_settings = RetentionSettings() if retention is None else retention
    if not isinstance(retention_settings, RetentionSettings):
        raise TypeError("retention must be a RetentionSettings instance or None.")
    rng = np.random.Generator(np.random.PCG64(config.seed))
    return _run_segment(
        problem,
        initial_state,
        kernel_settings=KernelSettings.from_config(config),
        retention=retention_settings,
        rng=rng,
        transitions_completed=0,
        iterations=config.iterations,
        include_initial=True,
    )


def continue_sample(
    problem: TransDimensionalProblem,
    checkpoint: SamplerCheckpoint,
    *,
    iterations: int,
) -> SamplingResult:
    """Continue a chain exactly from an in-memory transition boundary.

    Kernel settings, schedule identity, random state, and retention policy all
    come from ``checkpoint``. The incoming boundary state is never retained a
    second time, even when it lies on the retention phase.

    Args:
        problem: The same immutable numerical target used by the original run.
            Durable problem identity checks are deferred until checkpoint
            serialization is implemented.
        checkpoint: Exact result checkpoint from an earlier segment.
        iterations: Positive number of additional attempted transitions.

    Returns:
        Results for only the newly attempted segment and a new checkpoint.

    Raises:
        TypeError: If ``checkpoint`` has the wrong type.
        ValueError: If the segment length or schedule identity is invalid.
    """
    if not isinstance(checkpoint, SamplerCheckpoint):
        raise TypeError("checkpoint must be a SamplerCheckpoint instance.")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    supported_schedules = {SCHEDULE_ID, FIXED_BLOCK_SCHEDULE_ID}
    if checkpoint.schedule_id not in supported_schedules:
        raise ValueError(f"Unsupported sampler schedule {checkpoint.schedule_id!r}.")
    if checkpoint.transitions_completed < 0:
        raise ValueError("checkpoint transitions_completed must be non-negative.")
    if problem is not checkpoint.problem:
        raise ValueError("continuation requires the exact in-memory problem object.")
    expected_schedule_id = _schedule_id(problem)
    if checkpoint.schedule_id != expected_schedule_id:
        raise ValueError(
            f"Checkpoint schedule {checkpoint.schedule_id!r} is incompatible with "
            f"this problem; expected {expected_schedule_id!r}."
        )
    return _run_segment(
        problem,
        checkpoint.state,
        kernel_settings=checkpoint.kernel_settings,
        retention=checkpoint.retention,
        rng=checkpoint.rng_state.generator(),
        transitions_completed=checkpoint.transitions_completed,
        iterations=int(iterations),
        include_initial=False,
    )


__all__ = [
    "FIXED_BLOCK_SCHEDULE_ID",
    "KernelSettings",
    "PCG64State",
    "SCHEDULE_ID",
    "SamplerCheckpoint",
    "SamplerConfig",
    "SamplingResult",
    "SamplingTrace",
    "continue_sample",
    "sample",
]
