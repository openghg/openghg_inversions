"""Structural-proposal diagnostics for the experimental Voronoi RJMCMC.

The sampler can optionally record one row for every dimension or
nucleus-location proposal.  Rows describe the cached source, proposed
candidate, and resulting chain state without rebuilding a state, drawing
additional random numbers, or multiplying the fine-grid sensitivity transpose
by a residual.  Consequently, enabling diagnostics does not alter the
transition kernel, random stream, checkpoint schema, or retained-state schema.

Cell ownership is represented by stable nucleus cell identifiers,
``state.active_nuclei[state.labels]``.  This is essential because canonical
region positions change after structural edits.  Prediction standardization
uses the fixed ``problem.observation_sd`` elementwise.  It is exact diagonal
Gaussian whitening only when the independent fixed-error likelihood is active;
for the inferred OU likelihood it is a useful observation-error scale, not
whitening by the complete model-data mismatch covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
)
from openghg_inversions.experimental.rjmcmc.proposals import TransitionTerms

if TYPE_CHECKING:
    from collections.abc import Sequence


IntArray = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]

STRUCTURAL_MOVES: Final[tuple[str, ...]] = (
    "birth",
    "death",
    "global_move",
    "local_move",
)

STRUCTURAL_INVALID_REASON_CODES: Final[dict[str, int]] = {
    "": 0,
    "birth is unavailable when state.k equals problem.k_max.": 1,
    "death is unavailable when state.k equals problem.k_min.": 2,
    "new_nucleus must identify a fine-grid cell.": 3,
    "new_nucleus must be unoccupied.": 4,
    "proposed_coefficient must be finite and positive.": 5,
    "remove_position must select an active nucleus.": 6,
    "move_position must select an active nucleus.": 7,
}
STRUCTURAL_INVALID_REASON_LABELS: Final[dict[int, str]] = {
    code: reason for reason, code in STRUCTURAL_INVALID_REASON_CODES.items()
}
_ARRAY_FIELDS: Final[tuple[str, ...]] = (
    "transition",
    "move",
    "invalid_reason_code",
    "valid",
    "accepted",
    "source_k",
    "candidate_k",
    "result_k",
    "delta_log_likelihood",
    "delta_log_coefficient_prior",
    "delta_log_fixed_coefficient_prior",
    "delta_log_k_prior",
    "delta_log_nucleus_prior",
    "delta_log_error_model_prior",
    "delta_log_coefficient_hyperprior",
    "delta_log_target",
    "log_q_forward",
    "log_q_reverse",
    "log_jacobian",
    "log_acceptance_ratio",
    "source_nucleus",
    "candidate_nucleus",
    "owner_changed_cell_count",
    "owner_changed_cell_fraction",
    "affected_candidate_design_column_count",
    "prediction_change_l2",
    "observation_error_standardized_prediction_change_l2",
    "event_region_observation_error_standardized_design_l2",
    "coefficient_contrast",
)

_UINT8_FIELDS: Final[tuple[str, ...]] = ("invalid_reason_code",)

_FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "delta_log_likelihood",
    "delta_log_coefficient_prior",
    "delta_log_fixed_coefficient_prior",
    "delta_log_k_prior",
    "delta_log_nucleus_prior",
    "delta_log_error_model_prior",
    "delta_log_coefficient_hyperprior",
    "delta_log_target",
    "log_q_forward",
    "log_q_reverse",
    "log_jacobian",
    "log_acceptance_ratio",
    "owner_changed_cell_fraction",
    "prediction_change_l2",
    "observation_error_standardized_prediction_change_l2",
    "event_region_observation_error_standardized_design_l2",
    "coefficient_contrast",
)

_INTEGER_FIELDS: Final[tuple[str, ...]] = (
    "transition",
    "source_k",
    "candidate_k",
    "result_k",
    "source_nucleus",
    "candidate_nucleus",
    "owner_changed_cell_count",
    "affected_candidate_design_column_count",
)

_DELTA_FIELDS: Final[tuple[str, ...]] = (
    "delta_log_likelihood",
    "delta_log_coefficient_prior",
    "delta_log_fixed_coefficient_prior",
    "delta_log_k_prior",
    "delta_log_nucleus_prior",
    "delta_log_error_model_prior",
    "delta_log_coefficient_hyperprior",
    "delta_log_target",
)


def _readonly_vector(values: object, *, dtype: DTypeLike, name: str) -> np.ndarray:
    """Return an owned, immutable one-dimensional NumPy array."""
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    array.setflags(write=False)
    return array


def _invalid_reason_code(reason: str | None) -> int:
    """Return the compact code for a structural proposal invalidity reason."""
    if reason is None:
        return 0
    try:
        return STRUCTURAL_INVALID_REASON_CODES[reason]
    except KeyError as error:
        raise ValueError(f"unregistered structural invalidity reason: {reason!r}.") from error


def _scaled_l2_norm(values: FloatArray) -> float:
    """Return an overflow-resistant Euclidean norm.

    A positive infinity is returned only when the mathematical norm exceeds
    the representable float64 range. Finite inputs around ``1e200`` therefore
    remain diagnostic rather than overflowing during squaring.
    """
    absolute = np.abs(values)
    scale = float(np.max(absolute, initial=0.0))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.sqrt(np.sum((absolute / scale) ** 2)))


@dataclass(frozen=True, slots=True)
class StructuralDiagnosticsProvenance:
    """Identity required before structural diagnostic segments can be joined.

    Args:
        chain_id: Stable identifier shared by every segment of one chain.
        problem_fingerprint: Lowercase hexadecimal SHA-256 of the numerical
            target and frozen inputs. The checkpoint I/O problem fingerprint
            is suitable.

    Raises:
        TypeError: If either identifier is not a string.
        ValueError: If either identifier is empty or contains only whitespace.
    """

    chain_id: str
    problem_fingerprint: str

    def __post_init__(self) -> None:
        """Reject provenance that cannot distinguish a chain and problem."""
        for name in ("chain_id", "problem_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
            value = value.strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        fingerprint = self.problem_fingerprint.lower()
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("problem_fingerprint must be a 64-character hexadecimal SHA-256.")
        object.__setattr__(self, "problem_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class StructuralDiagnostics:
    """Immutable per-proposal structural diagnostics.

    All arrays have one row per attempted ``birth``, ``death``,
    ``global_move``, or ``local_move``.  ``transition`` is the global number
    of completed atomic transitions, so the first proposal in a fresh chain
    has coordinate one.  Candidate deltas and norms describe the proposal even
    when it was rejected; ``result_k`` describes the visited chain state.

    ``source_nucleus`` and ``candidate_nucleus`` are event endpoint cell IDs.
    An insertion is ``(-1, new)``, a deletion is ``(removed, -1)``, and a
    nucleus move is ``(old, new)``.  Invalid proposals use ``(-1, -1)``.
    ``coefficient_contrast`` is a signed log coefficient ratio: insertion
    coefficient over the source coefficient owning the inserted cell; removed
    coefficient over the candidate coefficient owning the removed cell; or
    moved coefficient over the source coefficient owning the destination cell.

    ``affected_candidate_design_column_count`` counts candidate regions whose
    design columns need rebuilding after ownership identities change.  It
    includes surviving donors and recipients, while excluding a removed
    nucleus.  An active empty region is still counted when its ownership
    identity participates in the edit.

    Args:
        transition: Global completed-transition coordinates.
        move: Stable structural proposal names.
        invalid_reason_code: Zero for valid proposals and a compact
            deterministic invalidity code otherwise. Labels are exposed by
            :attr:`invalid_reason`.
        valid: Whether each proposal produced an eligible candidate.
        accepted: Whether each candidate became the next chain state.
        source_k: Active-region count before the proposal.
        candidate_k: Active-region count in the proposed candidate.
        result_k: Active-region count after acceptance or rejection.
        delta_log_likelihood: Candidate-minus-source cached likelihood.
        delta_log_coefficient_prior: Candidate-minus-source dynamic prior.
        delta_log_fixed_coefficient_prior: Candidate-minus-source fixed prior.
        delta_log_k_prior: Candidate-minus-source region-count prior.
        delta_log_nucleus_prior: Candidate-minus-source nucleus-set prior.
        delta_log_error_model_prior: Candidate-minus-source error-model prior.
        delta_log_coefficient_hyperprior: Candidate-minus-source hyperprior.
        delta_log_target: Candidate-minus-source complete target.
        log_q_forward: Forward proposal log density or probability conditional
            on the selected structural direction. The sampler's equal
            up/down selection probabilities cancel from the MH ratio.
        log_q_reverse: Corresponding conditional reverse proposal term.
        log_jacobian: Dimension-matching log absolute Jacobian.
        log_acceptance_ratio: Untruncated log Metropolis-Hastings ratio.
        source_nucleus: Removed or moved-from nucleus cell ID.
        candidate_nucleus: Added or moved-to nucleus cell ID.
        owner_changed_cell_count: Fine-grid cells whose owner identity changes.
        owner_changed_cell_fraction: Owner-change count divided by the fixed
            fine-grid cell count.
        affected_candidate_design_column_count: Candidate design columns
            affected by those owner changes.
        prediction_change_l2: Candidate-minus-source prediction Euclidean
            norm. Positive infinity denotes a norm beyond float64 range.
        observation_error_standardized_prediction_change_l2: Prediction
            difference divided elementwise by ``observation_sd`` before taking
            its Euclidean norm.
        event_region_observation_error_standardized_design_l2: Event-region
            design column divided elementwise by ``observation_sd`` before
            taking its Euclidean norm.  The candidate column is used for an
            insertion or move and the source column for a deletion.
        coefficient_contrast: Signed event coefficient contrast described
            above.
        initial_nuclei: Canonical active nucleus IDs at the segment boundary
            before its first transition.
        final_nuclei: Canonical active nucleus IDs after its last transition.
        segment_transition_start: Completed atomic-transition count at the
            incoming segment boundary.
        segment_transition_end: Completed atomic-transition count at the
            outgoing segment boundary.
        n_grid_cells: Fine-grid cell count of the sampled problem.
        n_observations: Observation count of the sampled problem.
        provenance: Stable chain and problem identity. Concatenation rejects
            segments whose provenance differs even when their transition and
            nucleus endpoints happen to agree.

    Raises:
        TypeError: If ``provenance`` is not a
            :class:`StructuralDiagnosticsProvenance`.
        ValueError: If shapes, coordinates, move names, sentinels, or numerical
            invariants are inconsistent.
    """

    transition: IntArray
    move: StringArray
    invalid_reason_code: UInt8Array
    valid: BoolArray
    accepted: BoolArray
    source_k: IntArray
    candidate_k: IntArray
    result_k: IntArray
    delta_log_likelihood: FloatArray
    delta_log_coefficient_prior: FloatArray
    delta_log_fixed_coefficient_prior: FloatArray
    delta_log_k_prior: FloatArray
    delta_log_nucleus_prior: FloatArray
    delta_log_error_model_prior: FloatArray
    delta_log_coefficient_hyperprior: FloatArray
    delta_log_target: FloatArray
    log_q_forward: FloatArray
    log_q_reverse: FloatArray
    log_jacobian: FloatArray
    log_acceptance_ratio: FloatArray
    source_nucleus: IntArray
    candidate_nucleus: IntArray
    owner_changed_cell_count: IntArray
    owner_changed_cell_fraction: FloatArray
    affected_candidate_design_column_count: IntArray
    prediction_change_l2: FloatArray
    observation_error_standardized_prediction_change_l2: FloatArray
    event_region_observation_error_standardized_design_l2: FloatArray
    coefficient_contrast: FloatArray
    initial_nuclei: IntArray
    final_nuclei: IntArray
    segment_transition_start: int
    segment_transition_end: int
    n_grid_cells: int
    n_observations: int
    provenance: StructuralDiagnosticsProvenance

    def __post_init__(self) -> None:
        """Own arrays and validate cross-field structural invariants."""
        if not isinstance(self.provenance, StructuralDiagnosticsProvenance):
            raise TypeError("provenance must be a StructuralDiagnosticsProvenance instance.")
        if (
            isinstance(self.n_grid_cells, bool)
            or not isinstance(self.n_grid_cells, (int, np.integer))
            or self.n_grid_cells < 1
        ):
            raise ValueError("n_grid_cells must be a positive integer.")
        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, (int, np.integer))
            or self.n_observations < 1
        ):
            raise ValueError("n_observations must be a positive integer.")
        object.__setattr__(self, "n_grid_cells", int(self.n_grid_cells))
        object.__setattr__(self, "n_observations", int(self.n_observations))
        for name in ("segment_transition_start", "segment_transition_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))
        if self.segment_transition_start < 0 or self.segment_transition_end <= self.segment_transition_start:
            raise ValueError("segment transition bounds must define a positive interval.")
        for name in ("initial_nuclei", "final_nuclei"):
            nuclei = _readonly_vector(getattr(self, name), dtype=np.int64, name=name)
            if (
                nuclei.size < 1
                or np.any(nuclei < 0)
                or np.any(nuclei >= self.n_grid_cells)
                or np.any(np.diff(nuclei) <= 0)
            ):
                raise ValueError(f"{name} must contain sorted unique fine-grid cell identifiers.")
            object.__setattr__(self, name, nuclei)

        for name in _INTEGER_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.int64, name=name),
            )
        for name in _UINT8_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.uint8, name=name),
            )
        for name in _FLOAT_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.float64, name=name),
            )
        object.__setattr__(
            self,
            "move",
            _readonly_vector(self.move, dtype="U11", name="move"),
        )
        for name in ("valid", "accepted"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.bool_, name=name),
            )

        size = int(self.transition.size)
        for name in _ARRAY_FIELDS:
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},).")
        if size and (self.transition[0] < 1 or np.any(np.diff(self.transition) <= 0)):
            raise ValueError("transition must contain strictly increasing positive coordinates.")
        if size and (
            self.transition[0] <= self.segment_transition_start
            or self.transition[-1] > self.segment_transition_end
        ):
            raise ValueError("structural transitions must lie inside the segment bounds.")
        if np.any(~np.isin(self.move, STRUCTURAL_MOVES)):
            raise ValueError(f"move entries must be one of {STRUCTURAL_MOVES!r}.")
        known_reason_codes = np.asarray([*STRUCTURAL_INVALID_REASON_LABELS], dtype=np.uint8)
        if np.any(~np.isin(self.invalid_reason_code, known_reason_codes)):
            raise ValueError("invalid_reason_code contains an unknown code.")
        if np.any(self.valid & (self.invalid_reason_code != 0)) or np.any(
            ~self.valid & (self.invalid_reason_code == 0)
        ):
            raise ValueError("invalid_reason_code must be zero exactly for valid proposals.")
        if np.any(self.accepted & ~self.valid):
            raise ValueError("accepted structural proposals must be valid.")
        for name in _FLOAT_FIELDS:
            if np.any(np.isnan(getattr(self, name))):
                raise ValueError(f"{name} cannot contain NaN.")
        for name in (
            "prediction_change_l2",
            "observation_error_standardized_prediction_change_l2",
            "event_region_observation_error_standardized_design_l2",
        ):
            values = getattr(self, name)
            if np.any(np.isnan(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain non-negative values without NaN.")

        for name in ("source_k", "candidate_k", "result_k"):
            values = getattr(self, name)
            if np.any(values < 1) or np.any(values > self.n_grid_cells):
                raise ValueError(f"{name} must lie within the fine-grid cell count.")
        expected_result = np.where(self.accepted, self.candidate_k, self.source_k)
        if not np.array_equal(self.result_k, expected_result):
            raise ValueError("result_k must select candidate_k on acceptance and source_k otherwise.")
        if size and (
            self.initial_nuclei.size != self.source_k[0] or self.final_nuclei.size != self.result_k[-1]
        ):
            raise ValueError("segment endpoint nucleus counts must match structural rows.")
        for name in ("source_nucleus", "candidate_nucleus"):
            values = getattr(self, name)
            if np.any(values < -1) or np.any(values >= self.n_grid_cells):
                raise ValueError(f"{name} must contain valid cell IDs or the -1 sentinel.")
        if np.any(self.owner_changed_cell_count < 0) or np.any(
            self.owner_changed_cell_count > self.n_grid_cells
        ):
            raise ValueError("owner_changed_cell_count lies outside valid bounds.")
        if np.any(~np.isfinite(self.owner_changed_cell_fraction)) or np.any(
            (self.owner_changed_cell_fraction < 0.0) | (self.owner_changed_cell_fraction > 1.0)
        ):
            raise ValueError("owner_changed_cell_fraction must lie within [0, 1].")
        if not np.allclose(
            self.owner_changed_cell_fraction,
            self.owner_changed_cell_count / self.n_grid_cells,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("owner_changed_cell_fraction is inconsistent with its count.")
        if np.any(self.affected_candidate_design_column_count < 0) or np.any(
            self.affected_candidate_design_column_count > self.candidate_k
        ):
            raise ValueError("affected candidate design-column counts lie outside valid bounds.")

        invalid = ~self.valid
        for name in _DELTA_FIELDS:
            if np.any(getattr(self, name)[invalid] != 0.0):
                raise ValueError("invalid proposals must have zero cached-target deltas.")
        for name in (
            "log_q_forward",
            "log_q_reverse",
            "log_jacobian",
            "prediction_change_l2",
            "owner_changed_cell_fraction",
            "observation_error_standardized_prediction_change_l2",
            "event_region_observation_error_standardized_design_l2",
            "coefficient_contrast",
        ):
            if np.any(getattr(self, name)[invalid] != 0.0):
                raise ValueError(f"invalid proposals must have zero {name}.")
        if np.any(self.log_acceptance_ratio[invalid] != -np.inf):
            raise ValueError("invalid proposals must have log_acceptance_ratio equal to -inf.")
        if np.any(self.source_nucleus[invalid] != -1) or np.any(self.candidate_nucleus[invalid] != -1):
            raise ValueError("invalid proposals must use -1 nucleus sentinels.")
        if np.any(self.owner_changed_cell_count[invalid] != 0) or np.any(
            self.affected_candidate_design_column_count[invalid] != 0
        ):
            raise ValueError("invalid proposals must have zero structural change counts.")

    @property
    def size(self) -> int:
        """Number of structural proposal rows."""
        return int(self.transition.size)

    @property
    def invalid_reason(self) -> StringArray:
        """Human-readable invalidity labels decoded on demand."""
        labels = np.asarray(
            [
                STRUCTURAL_INVALID_REASON_LABELS.get(
                    int(code),
                    "other structural invalid proposal",
                )
                for code in self.invalid_reason_code
            ],
            dtype="U64",
        )
        labels.setflags(write=False)
        return labels


@dataclass(frozen=True, slots=True)
class DerivedStructuralDiagnostics:
    """Quantities derived from concatenated structural proposal rows.

    Arrays align with the input :class:`StructuralDiagnostics`.  Reversal
    flags require consecutive global atomic transition coordinates.
    ``exact_endpoint_reversal`` means the nucleus endpoint IDs are exactly
    swapped; it does not claim equality of every coefficient or cached state
    component.

    ``removed_region_lineage_age`` is populated only for accepted deletions.
    It is the difference in global transition coordinates from the accepted
    insertion that created the coefficient-bearing region lineage. An accepted
    location move transfers that creation time to the destination nucleus
    rather than resetting it. A deletion of a lineage already present at the
    beginning of the concatenated diagnostics uses ``-1`` and sets the
    corresponding left-censor flag.

    ``exact_endpoint_reversal`` includes either an immediate up/down undo or an
    immediate inverse nucleus-location move. It requires consecutive global
    atomic transitions; consecutive structural rows separated by another move
    are not immediate reversals.

    Args:
        k_step: Accepted change in active-region count, or zero.
        adjacent_accepted_opposite_k_reversal: Whether the current and previous
            global atomic transitions were accepted opposite nonzero ``k``
            steps.
        exact_endpoint_reversal: Whether adjacent accepted structural
            transitions exchange the same nucleus endpoint cell IDs.
        removed_region_lineage_age: Accepted deletion age measured from the
            lineage's insertion, or ``-1`` when not applicable or
            left-censored.
        removed_region_lineage_age_left_censored: Whether an accepted deletion
            removes a region lineage present at the diagnostic start.

    Raises:
        ValueError: If arrays do not share a one-dimensional shape.
    """

    k_step: IntArray
    adjacent_accepted_opposite_k_reversal: BoolArray
    exact_endpoint_reversal: BoolArray
    removed_region_lineage_age: IntArray
    removed_region_lineage_age_left_censored: BoolArray

    def __post_init__(self) -> None:
        """Own arrays and require a common one-dimensional shape."""
        object.__setattr__(
            self,
            "k_step",
            _readonly_vector(self.k_step, dtype=np.int64, name="k_step"),
        )
        for name in (
            "adjacent_accepted_opposite_k_reversal",
            "exact_endpoint_reversal",
            "removed_region_lineage_age_left_censored",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.bool_, name=name),
            )
        object.__setattr__(
            self,
            "removed_region_lineage_age",
            _readonly_vector(
                self.removed_region_lineage_age,
                dtype=np.int64,
                name="removed_region_lineage_age",
            ),
        )
        size = self.k_step.size
        for name in (
            "adjacent_accepted_opposite_k_reversal",
            "exact_endpoint_reversal",
            "removed_region_lineage_age",
            "removed_region_lineage_age_left_censored",
        ):
            if getattr(self, name).shape != (size,):
                raise ValueError("derived structural diagnostic arrays must share one shape.")


@dataclass(frozen=True, slots=True)
class NucleusResidenceIntervals:
    """Observable accepted-nucleus residence intervals.

    A row closes whenever an accepted deletion or nucleus move removes a cell
    ID. Nuclei already active before the concatenated diagnostics have unknown
    starts represented by ``start_transition == -1`` and ``left_censored``.
    Accepted insertions and move destinations that remain active at the final
    recorded transition have ``end_transition == -1`` and ``right_censored``.
    Initial nuclei that are never edited remain observable because the
    diagnostic table stores the segment's complete initial and final nucleus
    sets.

    Args:
        nucleus: Fine-grid nucleus cell identifier.
        start_transition: Accepted creation coordinate or ``-1``.
        end_transition: Accepted removal coordinate or ``-1``.
        left_censored: Whether creation predates the diagnostic table.
        right_censored: Whether removal follows the diagnostic table.

    Raises:
        ValueError: If shapes, nucleus identifiers, transition sentinels, or
            censor flags are inconsistent.
    """

    nucleus: IntArray
    start_transition: IntArray
    end_transition: IntArray
    left_censored: BoolArray
    right_censored: BoolArray

    def __post_init__(self) -> None:
        """Own interval arrays and validate censor sentinels."""
        for name in ("nucleus", "start_transition", "end_transition"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.int64, name=name),
            )
        for name in ("left_censored", "right_censored"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.bool_, name=name),
            )
        size = self.nucleus.size
        for name in (
            "start_transition",
            "end_transition",
            "left_censored",
            "right_censored",
        ):
            if getattr(self, name).shape != (size,):
                raise ValueError("nucleus residence interval arrays must share one shape.")
        if np.any(self.nucleus < 0):
            raise ValueError("nucleus residence intervals require non-negative cell IDs.")
        if np.any(self.left_censored != (self.start_transition == -1)):
            raise ValueError("left-censored intervals must use start_transition == -1.")
        if np.any(self.right_censored != (self.end_transition == -1)):
            raise ValueError("right-censored intervals must use end_transition == -1.")
        if np.any((self.start_transition != -1) & (self.start_transition < 1)):
            raise ValueError("observed interval starts must be positive transitions.")
        if np.any((self.end_transition != -1) & (self.end_transition < 1)):
            raise ValueError("observed interval ends must be positive transitions.")
        complete = ~self.left_censored & ~self.right_censored
        if np.any(self.end_transition[complete] <= self.start_transition[complete]):
            raise ValueError("complete residence intervals must have positive duration.")


@dataclass(frozen=True, slots=True)
class RegionLineageIntervals:
    """Accepted region-lineage intervals across nucleus relocations.

    ``lineage_id`` is allocated monotonically within the diagnostic stream and
    therefore remains unique even when a deleted nucleus cell is later reused.
    ``origin_nucleus`` records the initial or insertion nucleus. A location
    move changes the current nucleus endpoint without closing the lineage
    interval.

    Args:
        lineage_id: Stable diagnostic lineage identifier.
        origin_nucleus: Nucleus cell at the left boundary or insertion.
        start_transition: Accepted insertion coordinate or ``-1``.
        end_transition: Accepted deletion coordinate or ``-1``.
        left_censored: Whether insertion predates the diagnostic table.
        right_censored: Whether deletion follows the diagnostic table.

    Raises:
        ValueError: If arrays, transition sentinels, or censor flags are
            inconsistent.
    """

    lineage_id: IntArray
    origin_nucleus: IntArray
    start_transition: IntArray
    end_transition: IntArray
    left_censored: BoolArray
    right_censored: BoolArray

    def __post_init__(self) -> None:
        """Own interval arrays and validate censor sentinels."""
        for name in ("lineage_id", "origin_nucleus", "start_transition", "end_transition"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.int64, name=name),
            )
        for name in ("left_censored", "right_censored"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), dtype=np.bool_, name=name),
            )
        size = self.lineage_id.size
        for name in (
            "origin_nucleus",
            "start_transition",
            "end_transition",
            "left_censored",
            "right_censored",
        ):
            if getattr(self, name).shape != (size,):
                raise ValueError("region lineage interval arrays must share one shape.")
        if np.any(self.lineage_id < 0) or np.unique(self.lineage_id).size != size:
            raise ValueError("region lineage identifiers must be non-negative and unique.")
        if np.any(self.origin_nucleus < 0):
            raise ValueError("region lineage origins must be non-negative nucleus IDs.")
        if np.any(self.left_censored != (self.start_transition == -1)):
            raise ValueError("left-censored lineages must use start_transition == -1.")
        if np.any(self.right_censored != (self.end_transition == -1)):
            raise ValueError("right-censored lineages must use end_transition == -1.")
        if np.any((self.start_transition != -1) & (self.start_transition < 1)):
            raise ValueError("observed lineage starts must be positive transitions.")
        if np.any((self.end_transition != -1) & (self.end_transition < 1)):
            raise ValueError("observed lineage ends must be positive transitions.")
        complete = ~self.left_censored & ~self.right_censored
        if np.any(self.end_transition[complete] <= self.start_transition[complete]):
            raise ValueError("complete region lineage intervals must have positive duration.")


def concatenate_structural_diagnostics(
    diagnostics: Sequence[StructuralDiagnostics],
) -> StructuralDiagnostics:
    """Concatenate globally ordered structural diagnostic segments.

    Args:
        diagnostics: One or more segment diagnostics in chronological order.
            Metadata must match and global transition coordinates must remain
            strictly increasing across segment boundaries.

    Returns:
        One immutable diagnostic table suitable for chain-wide derivations.

    Raises:
        ValueError: If no segments are supplied or metadata/ordering differs.
        TypeError: If any item is not :class:`StructuralDiagnostics`.
    """
    if not diagnostics:
        raise ValueError("diagnostics must contain at least one segment.")
    if any(not isinstance(item, StructuralDiagnostics) for item in diagnostics):
        raise TypeError("all diagnostics must be StructuralDiagnostics instances.")
    first = diagnostics[0]
    for item in diagnostics[1:]:
        if (
            item.n_grid_cells != first.n_grid_cells
            or item.n_observations != first.n_observations
            or item.provenance != first.provenance
        ):
            raise ValueError("structural diagnostic metadata must match across segments.")
    for previous, current in zip(diagnostics, diagnostics[1:]):
        if previous.segment_transition_end != current.segment_transition_start:
            raise ValueError("structural diagnostic segment transition bounds are discontinuous.")
        if not np.array_equal(previous.final_nuclei, current.initial_nuclei):
            raise ValueError("structural diagnostic segment nucleus endpoints are discontinuous.")
    arrays = {name: np.concatenate([getattr(item, name) for item in diagnostics]) for name in _ARRAY_FIELDS}
    return StructuralDiagnostics(
        **arrays,
        initial_nuclei=first.initial_nuclei,
        final_nuclei=diagnostics[-1].final_nuclei,
        segment_transition_start=first.segment_transition_start,
        segment_transition_end=diagnostics[-1].segment_transition_end,
        n_grid_cells=first.n_grid_cells,
        n_observations=first.n_observations,
        provenance=first.provenance,
    )


def derive_structural_diagnostics(
    diagnostics: StructuralDiagnostics,
) -> DerivedStructuralDiagnostics:
    """Derive chain-mixing events from globally concatenated diagnostics.

    Args:
        diagnostics: Chronologically ordered structural proposal diagnostics.

    Returns:
        Accepted ``k`` steps, adjacent reversal flags, and accepted-deletion
        region-lineage ages. Global transition coordinates make age valid
        across sampling segment boundaries after concatenation.

    Raises:
        TypeError: If ``diagnostics`` has the wrong type.
        ValueError: If accepted event endpoints are inconsistent with the
            segment endpoint nucleus sets.
    """
    if not isinstance(diagnostics, StructuralDiagnostics):
        raise TypeError("diagnostics must be a StructuralDiagnostics instance.")
    size = diagnostics.size
    k_step = diagnostics.result_k - diagnostics.source_k
    opposite = np.zeros(size, dtype=np.bool_)
    endpoints = np.zeros(size, dtype=np.bool_)
    if size > 1:
        adjacent_accepted = (
            diagnostics.accepted[1:] & diagnostics.accepted[:-1] & (np.diff(diagnostics.transition) == 1)
        )
        opposite[1:] = adjacent_accepted & (k_step[1:] != 0) & (k_step[1:] == -k_step[:-1])
        endpoints[1:] = (
            adjacent_accepted
            & (diagnostics.source_k[1:] == diagnostics.result_k[:-1])
            & (diagnostics.result_k[1:] == diagnostics.source_k[:-1])
            & (diagnostics.source_nucleus[1:] == diagnostics.candidate_nucleus[:-1])
            & (diagnostics.candidate_nucleus[1:] == diagnostics.source_nucleus[:-1])
        )

    lineage_age = np.full(size, -1, dtype=np.int64)
    left_censored = np.zeros(size, dtype=np.bool_)
    created_at: dict[int, int | None] = {int(nucleus): None for nucleus in diagnostics.initial_nuclei}
    for position in range(size):
        if not diagnostics.accepted[position]:
            continue
        move = str(diagnostics.move[position])
        source_nucleus = int(diagnostics.source_nucleus[position])
        candidate_nucleus = int(diagnostics.candidate_nucleus[position])
        transition = int(diagnostics.transition[position])
        if move == "death":
            if source_nucleus not in created_at:
                raise ValueError("deletion removes a lineage absent from the diagnostic state.")
            created_transition = created_at.pop(source_nucleus)
            if created_transition is None:
                left_censored[position] = True
            else:
                lineage_age[position] = transition - created_transition
        elif move in ("global_move", "local_move"):
            if source_nucleus not in created_at or candidate_nucleus in created_at:
                raise ValueError("location move endpoints are inconsistent with lineage state.")
            created_at[candidate_nucleus] = created_at.pop(source_nucleus)
        elif move == "birth":
            if candidate_nucleus in created_at:
                raise ValueError("insertion creates an already-active lineage endpoint.")
            created_at[candidate_nucleus] = transition

    if set(created_at) != set(map(int, diagnostics.final_nuclei)):
        raise ValueError("accepted structural events do not reproduce final_nuclei.")
    return DerivedStructuralDiagnostics(
        k_step=k_step,
        adjacent_accepted_opposite_k_reversal=opposite,
        exact_endpoint_reversal=endpoints,
        removed_region_lineage_age=lineage_age,
        removed_region_lineage_age_left_censored=left_censored,
    )


def derive_nucleus_residence_intervals(
    diagnostics: StructuralDiagnostics,
) -> NucleusResidenceIntervals:
    """Derive observable nucleus residence intervals across concatenated segments.

    Args:
        diagnostics: Globally ordered structural diagnostics, normally produced
            by :func:`concatenate_structural_diagnostics`.

    Returns:
        Closed removal intervals plus right-censored intervals for every
        accepted insertion or move destination still active at the end.

    Raises:
        TypeError: If ``diagnostics`` has the wrong type.
        ValueError: If accepted event endpoints are inconsistent with the
            segment endpoint nucleus sets.
    """
    if not isinstance(diagnostics, StructuralDiagnostics):
        raise TypeError("diagnostics must be a StructuralDiagnostics instance.")
    active_since: dict[int, int | None] = {int(nucleus): None for nucleus in diagnostics.initial_nuclei}
    intervals: list[tuple[int, int, int, bool, bool]] = []
    for position in range(diagnostics.size):
        if not diagnostics.accepted[position]:
            continue
        move = str(diagnostics.move[position])
        source_nucleus = int(diagnostics.source_nucleus[position])
        candidate_nucleus = int(diagnostics.candidate_nucleus[position])
        transition = int(diagnostics.transition[position])
        if move in ("death", "global_move", "local_move"):
            if source_nucleus not in active_since:
                raise ValueError("removal endpoint is absent from the nucleus residence state.")
            start = active_since.pop(source_nucleus)
            intervals.append(
                (
                    source_nucleus,
                    -1 if start is None else start,
                    transition,
                    start is None,
                    False,
                )
            )
        if move in ("birth", "global_move", "local_move"):
            if candidate_nucleus in active_since:
                raise ValueError("creation endpoint is already present in nucleus residence state.")
            active_since[candidate_nucleus] = transition
    for nucleus, start in sorted(active_since.items()):
        intervals.append((nucleus, -1 if start is None else start, -1, start is None, True))
    if set(active_since) != set(map(int, diagnostics.final_nuclei)):
        raise ValueError("accepted structural events do not reproduce final_nuclei.")
    return NucleusResidenceIntervals(
        nucleus=np.asarray([row[0] for row in intervals], dtype=np.int64),
        start_transition=np.asarray([row[1] for row in intervals], dtype=np.int64),
        end_transition=np.asarray([row[2] for row in intervals], dtype=np.int64),
        left_censored=np.asarray([row[3] for row in intervals], dtype=np.bool_),
        right_censored=np.asarray([row[4] for row in intervals], dtype=np.bool_),
    )


def derive_region_lineage_intervals(
    diagnostics: StructuralDiagnostics,
) -> RegionLineageIntervals:
    """Derive region-lineage intervals while transferring through moves.

    Args:
        diagnostics: Globally ordered structural diagnostics, normally produced
            by :func:`concatenate_structural_diagnostics`.

    Returns:
        Closed deletion intervals and right-censored surviving lineages.

    Raises:
        TypeError: If ``diagnostics`` has the wrong type.
        ValueError: If accepted event endpoints are inconsistent with the
            segment endpoint nucleus sets.
    """
    if not isinstance(diagnostics, StructuralDiagnostics):
        raise TypeError("diagnostics must be a StructuralDiagnostics instance.")
    active: dict[int, tuple[int, int, int | None]] = {
        int(nucleus): (lineage_id, int(nucleus), None)
        for lineage_id, nucleus in enumerate(diagnostics.initial_nuclei)
    }
    next_lineage_id = len(active)
    intervals: list[tuple[int, int, int, int, bool, bool]] = []
    for position in range(diagnostics.size):
        if not diagnostics.accepted[position]:
            continue
        move = str(diagnostics.move[position])
        source_nucleus = int(diagnostics.source_nucleus[position])
        candidate_nucleus = int(diagnostics.candidate_nucleus[position])
        transition = int(diagnostics.transition[position])
        if move == "death":
            if source_nucleus not in active:
                raise ValueError("deletion removes a lineage absent from the diagnostic state.")
            lineage_id, origin_nucleus, start = active.pop(source_nucleus)
            intervals.append(
                (
                    lineage_id,
                    origin_nucleus,
                    -1 if start is None else start,
                    transition,
                    start is None,
                    False,
                )
            )
        elif move in ("global_move", "local_move"):
            if source_nucleus not in active or candidate_nucleus in active:
                raise ValueError("location move endpoints are inconsistent with lineage state.")
            active[candidate_nucleus] = active.pop(source_nucleus)
        elif move == "birth":
            if candidate_nucleus in active:
                raise ValueError("insertion creates an already-active lineage endpoint.")
            active[candidate_nucleus] = (next_lineage_id, candidate_nucleus, transition)
            next_lineage_id += 1
    if set(active) != set(map(int, diagnostics.final_nuclei)):
        raise ValueError("accepted structural events do not reproduce final_nuclei.")
    for _, (lineage_id, origin_nucleus, start) in sorted(active.items()):
        intervals.append(
            (
                lineage_id,
                origin_nucleus,
                -1 if start is None else start,
                -1,
                start is None,
                True,
            )
        )
    return RegionLineageIntervals(
        lineage_id=np.asarray([row[0] for row in intervals], dtype=np.int64),
        origin_nucleus=np.asarray([row[1] for row in intervals], dtype=np.int64),
        start_transition=np.asarray([row[2] for row in intervals], dtype=np.int64),
        end_transition=np.asarray([row[3] for row in intervals], dtype=np.int64),
        left_censored=np.asarray([row[4] for row in intervals], dtype=np.bool_),
        right_censored=np.asarray([row[5] for row in intervals], dtype=np.bool_),
    )


def _coefficient_for_nucleus(state: TransDimensionalState, nucleus: int) -> float:
    """Return the coefficient aligned with one active nucleus identity."""
    position = int(np.searchsorted(state.active_nuclei, nucleus))
    if position >= state.k or int(state.active_nuclei[position]) != nucleus:
        raise ValueError("event nucleus is not active in the supplied state.")
    return float(state.active_coefficients[position])


def _event_nuclei(
    source: TransDimensionalState,
    candidate: TransDimensionalState,
) -> tuple[int, int]:
    """Return removed and added nucleus identities with ``-1`` sentinels."""
    removed = np.setdiff1d(
        source.active_nuclei,
        candidate.active_nuclei,
        assume_unique=True,
    )
    added = np.setdiff1d(
        candidate.active_nuclei,
        source.active_nuclei,
        assume_unique=True,
    )
    if removed.size > 1 or added.size > 1:
        raise ValueError("structural diagnostics support only one-nucleus edits.")
    source_nucleus = -1 if removed.size == 0 else int(removed[0])
    candidate_nucleus = -1 if added.size == 0 else int(added[0])
    return source_nucleus, candidate_nucleus


def _event_design_column(
    source: TransDimensionalState,
    candidate: TransDimensionalState,
    *,
    move: str,
    source_nucleus: int,
    candidate_nucleus: int,
) -> FloatArray:
    """Return the source or candidate event-region design column."""
    if move == "death":
        state = source
        nucleus = source_nucleus
    else:
        state = candidate
        nucleus = candidate_nucleus
    position = int(np.searchsorted(state.active_nuclei, nucleus))
    if position >= state.k or int(state.active_nuclei[position]) != nucleus:
        raise ValueError("event region is not active in the selected state.")
    return state.design[:, position]


def _coefficient_contrast(
    source: TransDimensionalState,
    candidate: TransDimensionalState,
    *,
    move: str,
    source_nucleus: int,
    candidate_nucleus: int,
) -> float:
    """Return the documented signed coefficient contrast for one edit."""
    if move == "birth":
        event_coefficient = _coefficient_for_nucleus(candidate, candidate_nucleus)
        other_owner = int(source.active_nuclei[source.labels[candidate_nucleus]])
        return float(np.log(event_coefficient) - np.log(_coefficient_for_nucleus(source, other_owner)))
    if move == "death":
        event_coefficient = _coefficient_for_nucleus(source, source_nucleus)
        other_owner = int(candidate.active_nuclei[candidate.labels[source_nucleus]])
        return float(np.log(event_coefficient) - np.log(_coefficient_for_nucleus(candidate, other_owner)))
    event_coefficient = _coefficient_for_nucleus(candidate, candidate_nucleus)
    other_owner = int(source.active_nuclei[source.labels[candidate_nucleus]])
    return float(np.log(event_coefficient) - np.log(_coefficient_for_nucleus(source, other_owner)))


def _structural_metrics(
    problem: TransDimensionalProblem,
    source: TransDimensionalState,
    transition: TransitionTerms,
) -> dict[str, float | int]:
    """Calculate structural metrics solely from source and candidate caches."""
    if not transition.valid:
        return {
            "source_nucleus": -1,
            "candidate_nucleus": -1,
            "owner_changed_cell_count": 0,
            "owner_changed_cell_fraction": 0.0,
            "affected_candidate_design_column_count": 0,
            "prediction_change_l2": 0.0,
            "observation_error_standardized_prediction_change_l2": 0.0,
            "event_region_observation_error_standardized_design_l2": 0.0,
            "coefficient_contrast": 0.0,
        }
    candidate = transition.candidate
    source_nucleus, candidate_nucleus = _event_nuclei(source, candidate)
    source_owner = source.active_nuclei[source.labels]
    candidate_owner = candidate.active_nuclei[candidate.labels]
    changed = source_owner != candidate_owner
    changed_count = int(np.count_nonzero(changed))
    if changed_count:
        affected_owner_ids = np.union1d(
            source_owner[changed],
            candidate_owner[changed],
        )
        affected_candidate_count = int(np.count_nonzero(np.isin(candidate.active_nuclei, affected_owner_ids)))
    else:
        affected_candidate_count = 0
    prediction_change = candidate.prediction - source.prediction
    event_design = _event_design_column(
        source,
        candidate,
        move=transition.move,
        source_nucleus=source_nucleus,
        candidate_nucleus=candidate_nucleus,
    )
    return {
        "source_nucleus": source_nucleus,
        "candidate_nucleus": candidate_nucleus,
        "owner_changed_cell_count": changed_count,
        "owner_changed_cell_fraction": changed_count / problem.n_grid_cells,
        "affected_candidate_design_column_count": affected_candidate_count,
        "prediction_change_l2": _scaled_l2_norm(prediction_change),
        "observation_error_standardized_prediction_change_l2": _scaled_l2_norm(
            prediction_change / problem.observation_sd
        ),
        "event_region_observation_error_standardized_design_l2": _scaled_l2_norm(
            event_design / problem.observation_sd
        ),
        "coefficient_contrast": _coefficient_contrast(
            source,
            candidate,
            move=transition.move,
            source_nucleus=source_nucleus,
            candidate_nucleus=candidate_nucleus,
        ),
    }


class _StructuralDiagnosticsBuffer:
    """Exact-capacity mutable collector finalized as immutable diagnostics."""

    __slots__ = (
        "_arrays",
        "_initial_nuclei",
        "_position",
        "_problem",
        "_provenance",
        "_segment_transition_start",
    )

    def __init__(
        self,
        problem: TransDimensionalProblem,
        initial_state: TransDimensionalState,
        capacity: int,
        *,
        provenance: StructuralDiagnosticsProvenance,
        segment_transition_start: int = 0,
    ) -> None:
        """Allocate scalar arrays for a known number of structural events."""
        if capacity < 0:
            raise ValueError("capacity must be non-negative.")
        if not isinstance(provenance, StructuralDiagnosticsProvenance):
            raise TypeError("provenance must be a StructuralDiagnosticsProvenance instance.")
        self._problem = problem
        self._provenance = provenance
        self._initial_nuclei = np.array(initial_state.active_nuclei, copy=True)
        self._segment_transition_start = int(segment_transition_start)
        self._position = 0
        arrays: dict[str, np.ndarray] = {}
        for name in _INTEGER_FIELDS:
            arrays[name] = np.empty(capacity, dtype=np.int64)
        for name in _FLOAT_FIELDS:
            arrays[name] = np.empty(capacity, dtype=np.float64)
        for name in _UINT8_FIELDS:
            arrays[name] = np.empty(capacity, dtype=np.uint8)
        arrays["move"] = np.empty(capacity, dtype="U11")
        arrays["valid"] = np.empty(capacity, dtype=np.bool_)
        arrays["accepted"] = np.empty(capacity, dtype=np.bool_)
        self._arrays = arrays

    def append(
        self,
        *,
        transition_number: int,
        source: TransDimensionalState,
        transition: TransitionTerms,
        result: TransDimensionalState,
        accepted: bool,
    ) -> None:
        """Append one structural proposal using already-cached state values."""
        if transition.move not in STRUCTURAL_MOVES:
            raise ValueError("only structural proposals can be appended.")
        position = self._position
        if position >= self._arrays["transition"].size:
            raise RuntimeError("structural diagnostic capacity was underestimated.")
        candidate = transition.candidate
        metrics = _structural_metrics(self._problem, source, transition)
        values: dict[str, object] = {
            "transition": transition_number,
            "move": transition.move,
            "invalid_reason_code": _invalid_reason_code(transition.reason),
            "valid": transition.valid,
            "accepted": accepted,
            "source_k": source.k,
            "candidate_k": candidate.k,
            "result_k": result.k,
            "delta_log_likelihood": candidate.log_likelihood - source.log_likelihood,
            "delta_log_coefficient_prior": (candidate.log_coefficient_prior - source.log_coefficient_prior),
            "delta_log_fixed_coefficient_prior": (
                candidate.log_fixed_coefficient_prior - source.log_fixed_coefficient_prior
            ),
            "delta_log_k_prior": candidate.log_k_prior - source.log_k_prior,
            "delta_log_nucleus_prior": (candidate.log_nucleus_prior - source.log_nucleus_prior),
            "delta_log_error_model_prior": (candidate.log_error_model_prior - source.log_error_model_prior),
            "delta_log_coefficient_hyperprior": (
                candidate.log_coefficient_hyperprior - source.log_coefficient_hyperprior
            ),
            "delta_log_target": candidate.log_target - source.log_target,
            "log_q_forward": transition.log_q_forward,
            "log_q_reverse": transition.log_q_reverse,
            "log_jacobian": transition.log_jacobian,
            "log_acceptance_ratio": transition.log_acceptance_ratio,
            **metrics,
        }
        for name, value in values.items():
            self._arrays[name][position] = value
        self._position += 1

    def finalize(
        self,
        final_state: TransDimensionalState,
        *,
        segment_transition_end: int | None = None,
    ) -> StructuralDiagnostics:
        """Return an immutable copy and verify the exact capacity count."""
        if self._position != self._arrays["transition"].size:
            raise RuntimeError("structural diagnostic capacity did not match recorded events.")
        if segment_transition_end is None:
            if self._arrays["transition"].size == 0:
                raise ValueError("segment_transition_end is required when no structural rows exist.")
            segment_transition_end = int(self._arrays["transition"][-1])
        return StructuralDiagnostics(
            **self._arrays,
            initial_nuclei=self._initial_nuclei,
            final_nuclei=final_state.active_nuclei,
            segment_transition_start=self._segment_transition_start,
            segment_transition_end=segment_transition_end,
            n_grid_cells=self._problem.n_grid_cells,
            n_observations=self._problem.n_observations,
            provenance=self._provenance,
        )


__all__ = [
    "DerivedStructuralDiagnostics",
    "NucleusResidenceIntervals",
    "RegionLineageIntervals",
    "STRUCTURAL_INVALID_REASON_LABELS",
    "STRUCTURAL_MOVES",
    "StructuralDiagnostics",
    "StructuralDiagnosticsProvenance",
    "concatenate_structural_diagnostics",
    "derive_nucleus_residence_intervals",
    "derive_region_lineage_intervals",
    "derive_structural_diagnostics",
]
