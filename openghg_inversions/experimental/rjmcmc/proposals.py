"""Deterministic proposal accounting for the spatial TDMCMC sampler.

Random choices are supplied explicitly to every function in this module. This
keeps proposal construction reproducible and allows NumPy and Numba state
builders to be checked with identical draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .core import (
    Backend,
    TransDimensionalProblem,
    TransDimensionalState,
    update_dynamic_coefficient_state,
    update_error_model_state,
    update_fixed_coefficient_state,
    update_shared_hierarchy_state,
    update_structural_state,
)


_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class TransitionTerms:
    """A proposed state and the terms in its Metropolis-Hastings ratio.

    Attributes:
        candidate: Candidate state. Invalid proposals retain the source state.
        log_target_delta: Candidate log target minus source log target.
        log_q_forward: Conditional forward proposal log density or probability.
        log_q_reverse: Conditional reverse proposal log density or probability.
        log_jacobian: Log absolute Jacobian determinant for the dimension match.
        move: Stable proposal name used by sampler diagnostics.
        valid: Whether the proposal produced a candidate eligible for acceptance.
        reason: Explanation for an invalid self-transition, otherwise ``None``.
        log_acceptance_ratio: Complete untruncated log Metropolis-Hastings ratio.
            Invalid self-transitions use negative infinity so they cannot be
            accepted by :func:`accept_or_reject`.

    Raises:
        TypeError: If candidate or validity metadata have the wrong type.
        ValueError: If move metadata or log-ratio terms are malformed.
    """

    candidate: TransDimensionalState
    log_target_delta: float
    log_q_forward: float
    log_q_reverse: float
    log_jacobian: float
    move: str
    valid: bool = True
    reason: str | None = None
    log_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate transition metadata and calculate the acceptance ratio."""
        if not isinstance(self.candidate, TransDimensionalState):
            raise TypeError("candidate must be a TransDimensionalState.")
        if not isinstance(self.move, str) or not self.move:
            raise ValueError("move must be a non-empty string.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a Boolean.")
        if self.valid and self.reason is not None:
            raise ValueError("a valid transition cannot have an invalidity reason.")
        if not self.valid and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("an invalid transition must provide a non-empty reason.")

        terms = (
            float(self.log_target_delta),
            float(self.log_q_forward),
            float(self.log_q_reverse),
            float(self.log_jacobian),
        )
        if any(math.isnan(value) for value in terms):
            raise ValueError("transition log terms cannot be NaN.")
        object.__setattr__(self, "log_target_delta", terms[0])
        object.__setattr__(self, "log_q_forward", terms[1])
        object.__setattr__(self, "log_q_reverse", terms[2])
        object.__setattr__(self, "log_jacobian", terms[3])

        log_acceptance_ratio = terms[0] + terms[2] - terms[1] + terms[3] if self.valid else -math.inf
        if math.isnan(log_acceptance_ratio):
            raise ValueError("the calculated log acceptance ratio cannot be NaN.")
        object.__setattr__(self, "log_acceptance_ratio", log_acceptance_ratio)


def _normal_log_density(value: float, *, mean: float, stdev: float) -> float:
    """Return a normalized univariate Gaussian log density."""
    standardized = (value - mean) / stdev
    return -0.5 * standardized * standardized - math.log(stdev) - 0.5 * _LOG_TWO_PI


def _validate_proposal_scale(value: float, *, name: str) -> float:
    """Return a finite positive proposal scale or raise ``ValueError``."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _validate_backend(backend: Backend) -> None:
    """Raise if a state-builder backend name is invalid."""
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be either 'numpy' or 'numba'.")


def _validate_problem_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """Check the shape-level contract shared by a problem and source state.

    A state is required to have been built or durably loaded for ``problem``.
    The state representation does not retain a problem object reference, so a
    same-shaped state from a different problem cannot be identified here.
    """
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem.")
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if state.capacity != problem.k_max:
        raise ValueError("state capacity must equal problem.k_max.")
    if state.labels.size != problem.ncell or state.design.shape != (problem.nobs, problem.k_max):
        raise ValueError("state dimensions are incompatible with the problem.")
    if state.fixed_coefficients.shape != (problem.n_fixed_coefficients,):
        raise ValueError("state fixed coefficients are incompatible with the problem.")
    if not problem.k_min <= state.k <= problem.k_max:
        raise ValueError("state.k must lie within the problem's permitted range.")


def _invalid_transition(
    state: TransDimensionalState,
    *,
    move: str,
    reason: str,
) -> TransitionTerms:
    """Return a rejected-by-construction self-transition."""
    return TransitionTerms(
        candidate=state,
        log_target_delta=0.0,
        log_q_forward=0.0,
        log_q_reverse=0.0,
        log_jacobian=0.0,
        move=move,
        valid=False,
        reason=reason,
    )


def _active_position(value: object, *, k: int) -> int | None:
    """Return an active zero-based position, or ``None`` for an invalid choice."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    position = int(value)
    return position if 0 <= position < k else None


def _grid_cell(value: object, *, ncell: int) -> int | None:
    """Return a zero-based grid-cell index, or ``None`` for an invalid choice."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    cell = int(value)
    return cell if 0 <= cell < ncell else None


def _positive_coefficient(value: float) -> float | None:
    """Return a finite positive coefficient, or ``None`` outside prior support."""
    if isinstance(value, bool):
        return None
    try:
        coefficient = float(value)
    except (TypeError, ValueError):
        return None
    return coefficient if math.isfinite(coefficient) and coefficient > 0.0 else None


def _nearest_coefficient(
    problem: TransDimensionalProblem,
    nuclei: np.ndarray,
    coefficients: np.ndarray,
    cell: int,
) -> float:
    """Return the coefficient belonging to the nearest supplied nucleus."""
    differences = problem.grid_coordinates[nuclei] - problem.grid_coordinates[cell]
    squared_distances = np.sum(np.square(differences), axis=1)
    return float(coefficients[int(np.argmin(squared_distances))])


def _valid_transition(
    source: TransDimensionalState,
    candidate: TransDimensionalState,
    *,
    log_q_forward: float,
    log_q_reverse: float,
    move: str,
) -> TransitionTerms:
    """Build transition terms for a valid identity-Jacobian proposal."""
    return TransitionTerms(
        candidate=candidate,
        log_target_delta=candidate.log_target - source.log_target,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        log_jacobian=0.0,
        move=move,
    )


def propose_coefficient(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a random-walk update for one active coefficient.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state built or durably loaded for ``problem``.
            A same-shaped state associated with a different problem is outside
            the proposal contract.
        coefficient_position: Zero-based active coefficient position selected
            uniformly by the eventual sampler.
        proposed_coefficient: Explicit proposed value.
        proposal_stdev: Standard deviation of the symmetric Gaussian random walk.
        backend: State-building implementation for candidate caches.

    Returns:
        Complete proposal accounting. Invalid positions or coefficients outside
        positive lognormal support produce invalid self-transitions.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_stdev = _validate_proposal_scale(proposal_stdev, name="proposal_stdev")
    position = _active_position(coefficient_position, k=state.k)
    if position is None:
        return _invalid_transition(
            state,
            move="coefficient",
            reason="coefficient_position must select an active coefficient.",
        )
    value = _positive_coefficient(proposed_coefficient)
    if value is None:
        return _invalid_transition(
            state,
            move="coefficient",
            reason="proposed_coefficient must be finite and positive.",
        )

    current = float(state.active_coefficients[position])
    candidate = update_dynamic_coefficient_state(
        problem,
        state,
        coefficient_position=position,
        proposed_coefficient=value,
        backend=backend,
    )
    log_position_probability = -math.log(state.k)
    log_q_forward = log_position_probability + _normal_log_density(
        value,
        mean=current,
        stdev=proposal_stdev,
    )
    log_q_reverse = log_position_probability + _normal_log_density(
        current,
        mean=value,
        stdev=proposal_stdev,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="coefficient",
    )


def propose_fixed_coefficient(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    proposal_stdev: float,
    backend: Backend = "numpy",
    position_selected_deterministically: bool = False,
) -> TransitionTerms:
    """Construct a random-walk update for one always-active coefficient.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state built or durably loaded for ``problem``.
            A same-shaped state associated with a different problem is outside
            the proposal contract.
        coefficient_position: Zero-based fixed-block coefficient position,
            selected uniformly by the eventual sampler.
        proposed_coefficient: Explicit proposed value.
        proposal_stdev: Standard deviation of the symmetric Gaussian random walk.
        backend: State-building implementation for candidate caches.
        position_selected_deterministically: Whether the schedule selected the
            supplied position with probability one. The default accounts for a
            uniform random choice among fixed positions; deterministic slots
            omit that common selection factor from both proposal densities.

    Returns:
        Complete proposal accounting. A missing fixed block, invalid position,
        or coefficient outside positive lognormal support produces an invalid
        self-transition.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_stdev = _validate_proposal_scale(proposal_stdev, name="proposal_stdev")
    if not isinstance(position_selected_deterministically, bool):
        raise TypeError("position_selected_deterministically must be a boolean.")
    position = _active_position(coefficient_position, k=problem.n_fixed_coefficients)
    if position is None:
        return _invalid_transition(
            state,
            move="fixed_coefficient",
            reason="coefficient_position must select an always-active fixed coefficient.",
        )
    value = _positive_coefficient(proposed_coefficient)
    if value is None:
        return _invalid_transition(
            state,
            move="fixed_coefficient",
            reason="proposed_coefficient must be finite and positive.",
        )

    current = float(state.fixed_coefficients[position])
    candidate = update_fixed_coefficient_state(
        problem,
        state,
        coefficient_position=position,
        proposed_coefficient=value,
        backend=backend,
    )
    log_position_probability = (
        0.0 if position_selected_deterministically else -math.log(problem.n_fixed_coefficients)
    )
    log_q_forward = log_position_probability + _normal_log_density(
        value,
        mean=current,
        stdev=proposal_stdev,
    )
    log_q_reverse = log_position_probability + _normal_log_density(
        current,
        mean=value,
        stdev=proposal_stdev,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="fixed_coefficient",
    )


def _propose_error_parameter(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    parameter: str,
    parameter_position: int,
    proposed_value: float,
    proposal_stdev: float,
    backend: Backend,
) -> TransitionTerms:
    """Construct one bounded OU-parameter random-walk update.

    The Gaussian walk is defined on the whole real line. Draws outside the
    bounded-uniform target support are therefore explicit invalid
    self-transitions rather than draws from a silently truncated proposal.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_stdev = _validate_proposal_scale(proposal_stdev, name="proposal_stdev")
    error_model = problem.error_model
    move = "mismatch_sd" if parameter == "mismatch_sd" else "correlation_timescale"
    if error_model is None:
        return _invalid_transition(
            state,
            move=move,
            reason=f"{move} proposals require an inferred OU error model.",
        )

    if parameter == "mismatch_sd":
        current_values = state.mismatch_sd
        lower = error_model.mismatch_sd_prior_lower
        upper = error_model.mismatch_sd_prior_upper
    else:
        current_values = state.correlation_timescale
        lower = error_model.correlation_timescale_prior_lower
        upper = error_model.correlation_timescale_prior_upper
    position = _active_position(parameter_position, k=current_values.size)
    if position is None:
        return _invalid_transition(
            state,
            move=move,
            reason=f"{move}_position must select a configured parameter.",
        )
    if isinstance(proposed_value, bool):
        value = None
    else:
        try:
            numeric_value = float(proposed_value)
        except (TypeError, ValueError):
            value = None
        else:
            value = numeric_value if math.isfinite(numeric_value) else None
    if value is None or value < lower[position] or value > upper[position]:
        return _invalid_transition(
            state,
            move=move,
            reason=(
                f"proposed_{move} must be finite and within its inclusive bounded-uniform prior support."
            ),
        )

    current = float(current_values[position])
    if parameter == "mismatch_sd":
        candidate = update_error_model_state(
            problem,
            state,
            mismatch_sd_position=position,
            proposed_mismatch_sd=value,
            backend=backend,
        )
    else:
        candidate = update_error_model_state(
            problem,
            state,
            correlation_timescale_position=position,
            proposed_correlation_timescale=value,
            backend=backend,
        )
    log_position_probability = -math.log(current_values.size)
    log_q_forward = log_position_probability + _normal_log_density(
        value,
        mean=current,
        stdev=proposal_stdev,
    )
    log_q_reverse = log_position_probability + _normal_log_density(
        current,
        mean=value,
        stdev=proposal_stdev,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move=move,
    )


def propose_mismatch_sd(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    mismatch_sd_position: int,
    proposed_mismatch_sd: float,
    proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a random-walk update for one OU mismatch amplitude.

    Args:
        problem: Target containing an inferred OU error model.
        state: Immutable source state built or durably loaded for ``problem``.
        mismatch_sd_position: Zero-based mismatch group chosen uniformly by
            the eventual sampler.
        proposed_mismatch_sd: Explicit additive-Gaussian proposal draw.
        proposal_stdev: Positive standard deviation of the Gaussian walk.
        backend: OU likelihood implementation for the candidate target.

    Returns:
        Complete proposal accounting. Missing configuration, invalid positions,
        and draws outside the inclusive bounded-uniform prior support produce
        invalid self-transitions. The proposal is not truncated at the bounds.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    return _propose_error_parameter(
        problem,
        state,
        parameter="mismatch_sd",
        parameter_position=mismatch_sd_position,
        proposed_value=proposed_mismatch_sd,
        proposal_stdev=proposal_stdev,
        backend=backend,
    )


def propose_correlation_timescale(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    correlation_timescale_position: int,
    proposed_correlation_timescale: float,
    proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a random-walk update for one OU correlation timescale.

    Args:
        problem: Target containing an inferred OU error model.
        state: Immutable source state built or durably loaded for ``problem``.
        correlation_timescale_position: Zero-based timescale parameter chosen
            uniformly by the eventual sampler.
        proposed_correlation_timescale: Explicit additive-Gaussian proposal.
        proposal_stdev: Positive standard deviation of the Gaussian walk.
        backend: OU likelihood implementation for the candidate target.

    Returns:
        Complete proposal accounting. Missing configuration, invalid positions,
        and draws outside the inclusive bounded-uniform prior support produce
        invalid self-transitions. The proposal is not truncated at the bounds.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    return _propose_error_parameter(
        problem,
        state,
        parameter="correlation_timescale",
        parameter_position=correlation_timescale_position,
        proposed_value=proposed_correlation_timescale,
        proposal_stdev=proposal_stdev,
        backend=backend,
    )


def propose_shared_hierarchy(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    proposed_eta: float,
    proposed_zeta: float,
    eta_proposal_stdev: float,
    zeta_proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a joint Gaussian update of the shared hierarchy state.

    Args:
        problem: Target containing a shared coefficient hierarchy.
        state: Immutable source state built or durably loaded for ``problem``.
        proposed_eta: Explicit proposal for log arithmetic prior mean.
        proposed_zeta: Explicit proposal for log arithmetic prior SD.
        eta_proposal_stdev: Positive Gaussian walk scale for ``eta``.
        zeta_proposal_stdev: Positive Gaussian walk scale for ``zeta``.
        backend: Hierarchy-kernel implementation for the candidate target.

    Returns:
        Complete proposal accounting for the product of two independent,
        symmetric Gaussian random walks. Missing hierarchy configuration or a
        nonfinite coordinate produces an invalid self-transition.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, a proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    eta_scale = _validate_proposal_scale(
        eta_proposal_stdev,
        name="eta_proposal_stdev",
    )
    zeta_scale = _validate_proposal_scale(
        zeta_proposal_stdev,
        name="zeta_proposal_stdev",
    )
    if problem.coefficient_hierarchy is None:
        return _invalid_transition(
            state,
            move="shared_hierarchy",
            reason="shared hierarchy proposals require coefficient_hierarchy.",
        )
    eta = math.nan
    zeta = math.nan
    if isinstance(proposed_eta, bool) or isinstance(proposed_zeta, bool):
        valid_coordinates = False
    else:
        try:
            eta = float(proposed_eta)
            zeta = float(proposed_zeta)
        except (TypeError, ValueError):
            valid_coordinates = False
        else:
            valid_coordinates = math.isfinite(eta) and math.isfinite(zeta)
    if not valid_coordinates:
        return _invalid_transition(
            state,
            move="shared_hierarchy",
            reason="proposed_eta and proposed_zeta must be finite.",
        )

    candidate = update_shared_hierarchy_state(
        problem,
        state,
        proposed_eta=eta,
        proposed_zeta=zeta,
        backend=backend,
    )
    log_q_forward = _normal_log_density(
        eta,
        mean=state.eta,
        stdev=eta_scale,
    ) + _normal_log_density(
        zeta,
        mean=state.zeta,
        stdev=zeta_scale,
    )
    log_q_reverse = _normal_log_density(
        state.eta,
        mean=eta,
        stdev=eta_scale,
    ) + _normal_log_density(
        state.zeta,
        mean=zeta,
        stdev=zeta_scale,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="shared_hierarchy",
    )


def propose_birth(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    new_nucleus: int,
    proposed_coefficient: float,
    proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a birth at an explicitly supplied unused grid cell.

    The new coefficient is proposed from a Gaussian centred on the coefficient
    of the closest existing nucleus. The cell is conditionally uniform among
    the ``ncell - k`` unoccupied cells, and the reverse death deletes one of the
    ``k + 1`` candidate nuclei uniformly.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state.
        new_nucleus: Explicit fine-grid cell selected for the new nucleus.
        proposed_coefficient: Explicit auxiliary coefficient draw.
        proposal_stdev: Standard deviation of the birth coefficient proposal.
        backend: State-building implementation for candidate caches.

    Returns:
        Complete birth proposal accounting. A full-capacity state, invalid grid
        cell, occupied cell, or coefficient outside positive prior support yields
        an invalid self-transition.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_stdev = _validate_proposal_scale(proposal_stdev, name="proposal_stdev")
    if state.k >= problem.k_max:
        return _invalid_transition(
            state,
            move="birth",
            reason="birth is unavailable when state.k equals problem.k_max.",
        )
    cell = _grid_cell(new_nucleus, ncell=problem.ncell)
    if cell is None:
        return _invalid_transition(
            state,
            move="birth",
            reason="new_nucleus must identify a fine-grid cell.",
        )
    if np.any(state.active_nuclei == cell):
        return _invalid_transition(
            state,
            move="birth",
            reason="new_nucleus must be unoccupied.",
        )
    value = _positive_coefficient(proposed_coefficient)
    if value is None:
        return _invalid_transition(
            state,
            move="birth",
            reason="proposed_coefficient must be finite and positive.",
        )

    parent_coefficient = _nearest_coefficient(
        problem,
        state.active_nuclei,
        state.active_coefficients,
        cell,
    )
    candidate = update_structural_state(
        problem,
        state,
        np.append(state.active_nuclei, cell),
        np.append(state.active_coefficients, value),
        backend=backend,
    )
    log_q_forward = -math.log(problem.ncell - state.k) + _normal_log_density(
        value,
        mean=parent_coefficient,
        stdev=proposal_stdev,
    )
    log_q_reverse = -math.log(state.k + 1)
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="birth",
    )


def propose_death(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    remove_position: int,
    proposal_stdev: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Construct a death of an explicitly selected active nucleus.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state.
        remove_position: Zero-based active nucleus position selected uniformly
            by the eventual sampler.
        proposal_stdev: Standard deviation used by the reverse birth coefficient
            proposal.
        backend: State-building implementation for candidate caches.

    Returns:
        Complete death proposal accounting. A minimum-size state or invalid
        active position produces an invalid self-transition.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_stdev = _validate_proposal_scale(proposal_stdev, name="proposal_stdev")
    if state.k <= problem.k_min:
        return _invalid_transition(
            state,
            move="death",
            reason="death is unavailable when state.k equals problem.k_min.",
        )
    position = _active_position(remove_position, k=state.k)
    if position is None:
        return _invalid_transition(
            state,
            move="death",
            reason="remove_position must select an active nucleus.",
        )

    removed_nucleus = int(state.active_nuclei[position])
    removed_coefficient = float(state.active_coefficients[position])
    surviving_nuclei = np.delete(state.active_nuclei, position)
    surviving_coefficients = np.delete(state.active_coefficients, position)
    reverse_parent_coefficient = _nearest_coefficient(
        problem,
        surviving_nuclei,
        surviving_coefficients,
        removed_nucleus,
    )
    candidate = update_structural_state(
        problem,
        state,
        surviving_nuclei,
        surviving_coefficients,
        backend=backend,
    )
    log_q_forward = -math.log(state.k)
    log_q_reverse = -math.log(problem.ncell - (state.k - 1)) + _normal_log_density(
        removed_coefficient,
        mean=reverse_parent_coefficient,
        stdev=proposal_stdev,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="death",
    )


def propose_global_move(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    move_position: int,
    new_nucleus: int,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Move one selected nucleus uniformly to an unoccupied grid cell.

    The coefficient moves with its nucleus. Both directions choose one of ``k``
    active nuclei and one of ``ncell - k`` unoccupied cells, so the normalized
    forward and reverse probabilities are equal but are still reported.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state.
        move_position: Zero-based active nucleus position selected uniformly.
        new_nucleus: Explicit unoccupied destination grid cell.
        backend: State-building implementation for candidate caches.

    Returns:
        Complete global-move accounting. Invalid positions, grid cells, or
        occupied destinations produce invalid self-transitions.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract or backend is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    position = _active_position(move_position, k=state.k)
    if position is None:
        return _invalid_transition(
            state,
            move="global_move",
            reason="move_position must select an active nucleus.",
        )
    cell = _grid_cell(new_nucleus, ncell=problem.ncell)
    if cell is None:
        return _invalid_transition(
            state,
            move="global_move",
            reason="new_nucleus must identify a fine-grid cell.",
        )
    if np.any(state.active_nuclei == cell):
        return _invalid_transition(
            state,
            move="global_move",
            reason="new_nucleus must be unoccupied.",
        )

    nuclei = np.array(state.active_nuclei, copy=True)
    coefficients = np.array(state.active_coefficients, copy=True)
    nuclei[position] = cell
    candidate = update_structural_state(
        problem,
        state,
        nuclei,
        coefficients,
        backend=backend,
    )
    log_q = -math.log(state.k) - math.log(problem.ncell - state.k)
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q,
        log_q_reverse=log_q,
        move="global_move",
    )


def _local_move_log_probability(
    problem: TransDimensionalProblem,
    active_nuclei: np.ndarray,
    *,
    origin: int,
    destination: int,
    proposal_scale: float,
) -> float:
    """Return the normalized log probability of one local nucleus move.

    Args:
        problem: Target distribution and fine-grid coordinates.
        active_nuclei: Nuclei occupied before the proposed move.
        origin: Grid cell containing the selected nucleus.
        destination: Currently unoccupied destination grid cell.
        proposal_scale: Positive Gaussian distance scale in coordinate units.

    Returns:
        Joint log probability of selecting the nucleus uniformly and selecting
        ``destination`` from the normalized discrete Gaussian kernel.
    """
    available = np.ones(problem.ncell, dtype=bool)
    available[active_nuclei] = False
    destinations = np.flatnonzero(available)
    differences = (problem.grid_coordinates[destinations] - problem.grid_coordinates[origin]) / proposal_scale
    squared_distances = np.einsum("ij,ij->i", differences, differences)
    log_weights = -0.5 * squared_distances
    maximum = float(np.max(log_weights))
    log_normalizer = maximum + math.log(float(np.exp(log_weights - maximum).sum()))
    destination_offset = int(np.flatnonzero(destinations == destination)[0])
    return -math.log(active_nuclei.size) + float(log_weights[destination_offset]) - log_normalizer


def propose_local_move(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    move_position: int,
    new_nucleus: int,
    proposal_scale: float,
    backend: Backend = "numpy",
) -> TransitionTerms:
    """Move one selected nucleus using a discrete Gaussian location kernel.

    Conditional on choosing one of the ``k`` active nuclei uniformly, every
    currently unoccupied fine-grid cell is a possible destination. Its weight
    is proportional to ``exp(-distance_squared / (2 * proposal_scale**2))``.
    The coefficient moves with its nucleus. Because the available destination
    set changes after the move, the reverse normalization is recomputed from
    the candidate state rather than assumed to equal the forward normalization.

    Args:
        problem: Target distribution and fine-grid numerical inputs.
        state: Immutable source state.
        move_position: Zero-based active nucleus position selected uniformly.
        new_nucleus: Explicit currently unoccupied destination grid cell.
        proposal_scale: Positive Gaussian distance scale in coordinate units.
        backend: State-building implementation for candidate caches.

    Returns:
        Complete local-move accounting with exact normalized forward and reverse
        log probabilities. Invalid positions, grid cells, or occupied
        destinations produce invalid self-transitions.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If the problem/state contract, proposal scale, or backend
            is malformed.
    """
    _validate_backend(backend)
    _validate_problem_state(problem, state)
    proposal_scale = _validate_proposal_scale(proposal_scale, name="proposal_scale")
    position = _active_position(move_position, k=state.k)
    if position is None:
        return _invalid_transition(
            state,
            move="local_move",
            reason="move_position must select an active nucleus.",
        )
    cell = _grid_cell(new_nucleus, ncell=problem.ncell)
    if cell is None:
        return _invalid_transition(
            state,
            move="local_move",
            reason="new_nucleus must identify a fine-grid cell.",
        )
    if np.any(state.active_nuclei == cell):
        return _invalid_transition(
            state,
            move="local_move",
            reason="new_nucleus must be unoccupied.",
        )

    old_nucleus = int(state.active_nuclei[position])
    nuclei = np.array(state.active_nuclei, copy=True)
    coefficients = np.array(state.active_coefficients, copy=True)
    nuclei[position] = cell
    candidate = update_structural_state(
        problem,
        state,
        nuclei,
        coefficients,
        backend=backend,
    )
    log_q_forward = _local_move_log_probability(
        problem,
        state.active_nuclei,
        origin=old_nucleus,
        destination=cell,
        proposal_scale=proposal_scale,
    )
    log_q_reverse = _local_move_log_probability(
        problem,
        candidate.active_nuclei,
        origin=cell,
        destination=old_nucleus,
        proposal_scale=proposal_scale,
    )
    return _valid_transition(
        state,
        candidate,
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        move="local_move",
    )


def accept_or_reject(
    state: TransDimensionalState,
    transition: TransitionTerms,
    *,
    log_uniform: float,
) -> TransDimensionalState:
    """Select a candidate using an explicitly supplied log-uniform draw.

    Args:
        state: Source state to retain on rejection.
        transition: Proposal accounting and candidate state.
        log_uniform: Natural logarithm of a draw in ``(0, 1]``. Values must be
            non-positive; negative infinity is accepted as the limiting value.

    Returns:
        ``transition.candidate`` when ``log_uniform`` is below the truncated log
        acceptance ratio, otherwise the unchanged source ``state``.

    Raises:
        TypeError: If state or transition has the wrong type.
        ValueError: If ``log_uniform`` is NaN or positive.
    """
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if not isinstance(transition, TransitionTerms):
        raise TypeError("transition must be a TransitionTerms instance.")
    log_uniform = float(log_uniform)
    if math.isnan(log_uniform) or log_uniform > 0.0:
        raise ValueError("log_uniform must be non-positive and cannot be NaN.")
    threshold = min(0.0, transition.log_acceptance_ratio)
    if transition.valid and log_uniform < threshold:
        return transition.candidate
    return state


__all__ = [
    "TransitionTerms",
    "accept_or_reject",
    "propose_birth",
    "propose_coefficient",
    "propose_correlation_timescale",
    "propose_death",
    "propose_fixed_coefficient",
    "propose_global_move",
    "propose_local_move",
    "propose_mismatch_sd",
    "propose_shared_hierarchy",
]
