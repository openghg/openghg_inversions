"""Exact tiny-state MH-flow audit for likelihood-aware full-tiling moves.

The audit enumerates the actual structural-attempt catalogue used by the
compound sampler.  In particular, relocation attempts retain every
``intermediate leaf x axis`` slot, including geometrically invalid slots that
become explicit self-transitions.  This differs from
:func:`~openghg_inversions.experimental.rjmcmc.full_tiling.relocation_paths`,
which lists only valid geometry paths.

The oracle is deliberately independent of the stochastic sampler loop.  It
uses the public deterministic posterior proposal builders, an exactly
reversible binary64 mass construction, and a nonconstant direct-observation
likelihood.  Its result is a machine-readable R0 correctness certificate, not
a scientific sampling result.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .full_tiling import (
    Axis,
    LeafTiling,
    MergeChoice,
    SplitChoice,
    TilingState,
    edge_flip_paths,
    enumerate_tilings,
    merge_choices,
    relocation_paths,
)
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    PosteriorTransitionTerms,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    propose_posterior_edge_flip,
    propose_posterior_resolution_relocation,
)
from .full_tiling_compound_sampling import FULL_TILING_COMPOUND_SCHEDULE_ID
from .gamma_beta_adapter import gamma_beta_problem_from_rhime_inputs

FloatArray: TypeAlias = NDArray[np.float64]

AUDIT_SCHEMA = "openghg_inversions.mh_local_search_flow_oracle.v1"
COMPLETION_SCHEMA = "openghg_inversions.mh_local_search_flow_oracle_completion.v1"
AUDIT_FILENAME = "audit.json"
COMPLETION_FILENAME = "complete.json"

_SHAPE = (2, 3)
_K = 4
_FRACTIONS = (0.25, 0.5, 0.75)
_OBSERVATIONS = (0.2, 1.3, 2.7, 0.6, 1.9, 3.4)
_OBSERVATION_SD = (0.35, 0.45, 0.55, 0.40, 0.50, 0.60)
_NOMINAL_WEIGHT = ((1.0, 2.0, 3.0), (5.0, 7.0, 11.0))
_MOVE_LOG_WEIGHT = math.log(0.5)
_ABSOLUTE_TOLERANCE_FLOOR = 5.0e-13
_ULP_TOLERANCE_FACTOR = 512
_EXPECTED_COMPOUND_SCHEDULE_ID = "full_tiling_2_mixed_structure_1_root_slice_n_pair_allocation_fixed_sweep_v2"
_EXPECTED_TOPOLOGY_SHA256 = (
    "e55a203b2811f7ae209e531a4b84ca29783688f6ca1655422acd873febb52385",
    "5dc6c9b9b8069221f9c5b801243488017666863bb0ded065c92a05f18f2c16dc",
    "e5a2668a59f951649ac1e8bd324027807acc15c8662b26d9449b9c6e577c51d6",
    "5df9ab4e1f8c756315f4a5cf722bf03cf7fb2195446e5fa21211a4dabe95a609",
    "a2f66d645b28f6ba26474136516a6f858685052f1f56058f1fddf7a3e8a42b11",
    "ca309de90e5bbb3e1438afe593e87289f2d8b1a4a0c75f7c6c112f5dd5c22e6d",
    "e1cb7dd854063d1c2ea001fea80e4837e0c6c7ee9cebdfac7ce04cdcf982ffe2",
    "961f6fbd882c3803a608c37f652d7fef30412f7d521586c4330c249483bc2fa0",
)
_EXPECTED_COUNTS = {
    "edge_attempts": 45,
    "edge_valid": 6,
    "edge_invalid": 39,
    "edge_unique_valid_geometry_paths": 2,
    "relocation_attempts": 270,
    "relocation_valid": 48,
    "relocation_invalid": 222,
    "relocation_unique_valid_geometry_paths": 16,
    "valid_reverses": 54,
    "exact_invalid_self_transitions": 261,
}
_CHECK_KEYS = frozenset(
    (
        "actual_compound_attempt_catalogue",
        "geometry_oracle_destination_sets_match",
        "invalid_entries_are_explicit_self_transitions",
        "authoritative_reverse_topology_and_masses_bit_exact",
        "discrete_selection_terms_swap_bit_exact",
        "continuous_accounting_swaps_within_tolerance",
        "log_acceptance_ratio_is_antisymmetric",
        "accepted_pointwise_mh_flow_is_equal",
        "likelihood_is_nonconstant",
    )
)
_MAXIMUM_KEYS = frozenset(
    (
        "auxiliary_swap",
        "component_antisymmetry",
        "target_recovery",
        "jacobian_antisymmetry",
        "ratio_antisymmetry",
        "accepted_flow_equality",
    )
)
_PAYLOAD_KEYS = frozenset(
    (
        "status",
        "purpose",
        "source_revision",
        "runtime",
        "model",
        "model_sha256",
        "sampler_law",
        "catalogue",
        "checks",
        "likelihood",
        "tolerance_policy",
        "maximum_discrepancies",
    )
)
_CATALOGUE_KEYS = frozenset(
    (
        "topologies",
        "topology_sha256",
        "fractions",
        "relocation_destination_slots_per_merge",
        "move_mixture_weights",
        "counts",
    )
)
_LIKELIHOOD_KEYS = frozenset(
    (
        "minimum_log_likelihood",
        "maximum_log_likelihood",
        "range",
        "nonzero_transition_deltas",
    )
)
_MAXIMUM_VALUE_KEYS = frozenset(
    (
        "absolute_difference",
        "tolerance",
        "tolerance_fraction",
        "ulps",
        "context",
    )
)
_AUDIT_ENVELOPE_KEYS = frozenset(("schema", "payload", "payload_sha256"))
_COMPLETION_KEYS = frozenset(("schema", "files"))
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> str:
    """Return deterministic strict JSON without a trailing newline."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON value."""
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the streaming SHA-256 of one file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> object:
    """Load strict JSON while rejecting non-standard floating constants."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path.name} is not strict JSON") from error


def _write_create_only(path: Path, value: Mapping[str, object]) -> None:
    """Durably create one strict JSON file without replacing existing data."""
    text = _canonical_json(dict(value)) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _bounds(tiling: LeafTiling) -> tuple[tuple[int, int, int, int], ...]:
    """Return canonical half-open rectangle bounds."""
    return tuple((leaf.row_start, leaf.row_stop, leaf.col_start, leaf.col_stop) for leaf in tiling.leaves)


def _topology_sha256(tiling: LeafTiling) -> str:
    """Return a stable digest of canonical rectangle bounds."""
    return _json_sha256(_bounds(tiling))


def _expected_model_identity() -> dict[str, object]:
    """Return the exact immutable scientific-model identity."""
    return {
        "shape": list(_SHAPE),
        "k": _K,
        "operator": "six_direct_native_cell_rows",
        "observations": list(_OBSERVATIONS),
        "observation_sd": list(_OBSERVATION_SD),
        "nominal_weight": [list(row) for row in _NOMINAL_WEIGHT],
        "root_prior": {"shape": 4.0, "rate": 4.0},
        "allocation_concentration": 8.0,
        "likelihood_power": 1.0,
        "source_leaf_masses": [1.0, 1.0, 1.0, 1.0],
    }


def _expected_sampler_law() -> dict[str, object]:
    """Return the exact compound structural-attempt law audited here."""
    return {
        "schedule_id": _EXPECTED_COMPOUND_SCHEDULE_ID,
        "structural_slots_per_cycle": 2,
        "slot_component_draws_are_independent": True,
        "move_mixture_weights": {
            "edge_flip": 0.5,
            "resolution_relocation": 0.5,
        },
        "merge_selection": "uniform_over_current_midpoint_friend_merges",
        "edge_destination": "deterministic_perpendicular_axis",
        "relocation_destination": (
            "uniform_over_intermediate_canonical_leaf_cross_"
            "ordered_horizontal_vertical_axes_including_invalid"
        ),
        "relocation_destination_slots_formula": "2 * (K - 1)",
        "invalid_attempt_policy": "explicit_self_transition",
        "availability_renormalization": False,
    }


def _expected_tolerance_policy() -> dict[str, object]:
    """Return the exact immutable discrepancy-tolerance policy."""
    return {
        "formula": "max(5e-13, 512 * ulp(max(1, abs(a), abs(b))))",
        "absolute_floor": _ABSOLUTE_TOLERANCE_FLOOR,
        "ulp_factor": _ULP_TOLERANCE_FACTOR,
    }


def _is_lower_hex(value: object, pattern: re.Pattern[str]) -> bool:
    """Return whether a value is a complete lowercase hexadecimal digest."""
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> dict[str, object]:
    """Return one mapping after enforcing its exact schema keys."""
    if not isinstance(value, dict) or frozenset(value) != expected:
        raise ValueError(f"{name} keys are incompatible")
    return cast(dict[str, object], value)


def _require_finite_float(value: object, *, name: str) -> float:
    """Return one exact JSON float after rejecting integers and non-finite data."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _clean_git_head(repository: Path = _REPOSITORY_ROOT) -> str:
    """Return the clean worktree's lowercase full Git HEAD or fail closed."""
    try:
        top_level = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "--show-toplevel"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            (
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("git is required to certify flow-oracle provenance") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("could not inspect flow-oracle Git provenance") from error
    if Path(top_level).resolve() != repository.resolve():
        raise RuntimeError("flow-oracle source is not the expected Git worktree root")
    if not _is_lower_hex(head, _LOWER_HEX_40):
        raise RuntimeError("Git HEAD is not a lowercase full 40-hex revision")
    if status:
        raise RuntimeError("flow-oracle publication requires a clean Git worktree")
    return head


def _problem() -> FullTilingProblem:
    """Build the frozen nonconstant direct-observation posterior problem."""
    operator = np.eye(6, dtype=np.float64).reshape(6, *_SHAPE)
    observations = np.asarray(_OBSERVATIONS, dtype=np.float64)
    observation_sd = np.asarray(_OBSERVATION_SD, dtype=np.float64)
    nominal_weight = np.asarray(_NOMINAL_WEIGHT, dtype=np.float64)
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), operator),
            "mf": ("nmeasure", observations),
            "mf_error": ("nmeasure", observation_sd),
        },
        coords={
            "nmeasure": np.arange(6, dtype=np.int64),
            "lat": np.arange(_SHAPE[0], dtype=np.float64),
            "lon": np.arange(_SHAPE[1], dtype=np.float64),
        },
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=nominal_weight,
        k_min=_K,
        k_max=_K,
        concentration=float(2 * _K),
        root_variance=0.25,
        likelihood_power=1.0,
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=float(2 * _K),
    )
    prior = problem.base.prior
    if (
        prior.root_shape != 4.0
        or prior.root_rate != 4.0
        or problem.allocation_prior.concentration != 8.0
        or problem.base.likelihood_power != 1.0
    ):
        raise RuntimeError("the frozen scientific model identity drifted")
    return problem


def _source_state(
    problem: FullTilingProblem,
    tiling: LeafTiling,
) -> FullTilingPosteriorState:
    """Build a source whose structural involutions are exact in binary64."""
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            tiling,
            np.ones(_K, dtype=np.float64),
        ),
    )


def _scaled_tolerance(first: float, second: float) -> tuple[float, float]:
    """Return the frozen absolute/ULP tolerance and scale ULP."""
    scale = max(1.0, abs(first), abs(second))
    ulp = math.ulp(scale)
    return max(_ABSOLUTE_TOLERANCE_FLOOR, _ULP_TOLERANCE_FACTOR * ulp), ulp


@dataclass(slots=True)
class _Maximum:
    """Largest discrepancy observed for one audited numerical identity."""

    absolute_difference: float = -1.0
    tolerance: float = 0.0
    tolerance_fraction: float = 0.0
    ulps: float = 0.0
    context: str = ""

    def check(
        self,
        first: float,
        second: float,
        *,
        context: str,
    ) -> None:
        """Record and enforce one equality under the frozen tolerance."""
        if not math.isfinite(first) or not math.isfinite(second):
            raise RuntimeError(f"{context}: compared values must be finite")
        difference = abs(first - second)
        tolerance, ulp = _scaled_tolerance(first, second)
        if difference > tolerance:
            raise RuntimeError(
                f"{context}: discrepancy {difference:.17g} exceeds frozen tolerance {tolerance:.17g}"
            )
        if difference > self.absolute_difference:
            self.absolute_difference = difference
            self.tolerance = tolerance
            self.tolerance_fraction = difference / tolerance
            self.ulps = difference / ulp
            self.context = context

    def payload(self) -> dict[str, object]:
        """Return strict-JSON-compatible maximum-discrepancy metadata."""
        if self.absolute_difference < 0.0:
            raise RuntimeError("a discrepancy accumulator was never exercised")
        return {
            "absolute_difference": self.absolute_difference,
            "tolerance": self.tolerance,
            "tolerance_fraction": self.tolerance_fraction,
            "ulps": self.ulps,
            "context": self.context,
        }


def _reverse_fraction(
    source: FullTilingPosteriorState,
    transition: PosteriorTransitionTerms,
) -> float:
    """Return the exact source-child fraction for the unique reverse split."""
    split = transition.reverse_split_choice
    if split is None:
        raise RuntimeError("valid structural transition has no reverse split")
    first, second = split.leaf.midpoint_children(split.axis)
    first_mass = source.allocation.mass(first)
    second_mass = source.allocation.mass(second)
    fraction = first_mass / (first_mass + second_mass)
    if fraction != 0.5:
        raise RuntimeError("frozen equal source masses did not produce an exact half fraction")
    return fraction


def _reverse_transition(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    forward: PosteriorTransitionTerms,
) -> PosteriorTransitionTerms:
    """Construct the unique exact reverse of one valid forward transition."""
    reverse_merge = forward.reverse_merge_choice
    reverse_split = forward.reverse_split_choice
    if reverse_merge is None or reverse_split is None:
        raise RuntimeError("valid structural transition lacks reverse metadata")
    fraction = _reverse_fraction(source, forward)
    if forward.move == "edge_flip":
        return propose_posterior_edge_flip(
            problem,
            forward.candidate,
            merge_choice=reverse_merge,
            new_fraction=fraction,
        )
    return propose_posterior_resolution_relocation(
        problem,
        forward.candidate,
        merge_choice=reverse_merge,
        split_choice=reverse_split,
        new_fraction=fraction,
    )


def _audit_valid_transition(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    forward: PosteriorTransitionTerms,
    *,
    context: str,
    maxima: Mapping[str, _Maximum],
) -> None:
    """Audit exact recovery, accounting swaps, antisymmetry, and MH flow."""
    if not forward.valid:
        raise RuntimeError(f"{context}: expected a valid transition")
    reverse = _reverse_transition(problem, source, forward)
    if not reverse.valid:
        raise RuntimeError(f"{context}: unique reverse is invalid: {reverse.reason}")
    if reverse.candidate.allocation.tiling != source.allocation.tiling:
        raise RuntimeError(f"{context}: reverse topology recovery is not exact")
    if not np.array_equal(reverse.candidate.leaf_masses, source.leaf_masses):
        raise RuntimeError(f"{context}: reverse authoritative mass recovery is not bit-exact")
    if not np.array_equal(
        reverse.candidate.fixed_coefficients,
        source.fixed_coefficients,
    ):
        raise RuntimeError(f"{context}: reverse fixed-coordinate recovery is not bit-exact")

    if (
        forward.log_q_forward_selection != reverse.log_q_reverse_selection
        or forward.log_q_reverse_selection != reverse.log_q_forward_selection
    ):
        raise RuntimeError(f"{context}: discrete selection terms do not swap bit-exactly")

    maxima["auxiliary_swap"].check(
        forward.log_q_forward_auxiliary,
        reverse.log_q_reverse_auxiliary,
        context=f"{context}:forward-aux",
    )
    maxima["auxiliary_swap"].check(
        forward.log_q_reverse_auxiliary,
        reverse.log_q_forward_auxiliary,
        context=f"{context}:reverse-aux",
    )
    maxima["component_antisymmetry"].check(
        forward.delta_log_likelihood,
        -reverse.delta_log_likelihood,
        context=f"{context}:likelihood",
    )
    maxima["component_antisymmetry"].check(
        forward.delta_log_allocation_prior,
        -reverse.delta_log_allocation_prior,
        context=f"{context}:allocation-prior",
    )
    maxima["target_recovery"].check(
        source.log_target,
        reverse.candidate.log_target,
        context=f"{context}:target-recovery",
    )
    maxima["jacobian_antisymmetry"].check(
        forward.log_jacobian,
        -reverse.log_jacobian,
        context=context,
    )
    maxima["ratio_antisymmetry"].check(
        forward.log_acceptance_ratio,
        -reverse.log_acceptance_ratio,
        context=context,
    )

    forward_flow = (
        source.log_target + _MOVE_LOG_WEIGHT + forward.log_q_forward + min(0.0, forward.log_acceptance_ratio)
    )
    reverse_flow_on_forward_chart = (
        forward.candidate.log_target
        + _MOVE_LOG_WEIGHT
        + reverse.log_q_forward
        + min(0.0, reverse.log_acceptance_ratio)
        + forward.log_jacobian
    )
    maxima["accepted_flow_equality"].check(
        forward_flow,
        reverse_flow_on_forward_chart,
        context=context,
    )


def _invalid_transition_is_explicit_self(
    transition: PosteriorTransitionTerms,
    source: FullTilingPosteriorState,
) -> bool:
    """Return whether one invalid attempt has the required self-transition form."""
    return bool(
        not transition.valid
        and transition.candidate is source
        and transition.reason
        and transition.reverse_merge_choice is None
        and transition.reverse_split_choice is None
        and transition.log_acceptance_ratio == -math.inf
    )


def _relocation_catalogue(
    tiling: LeafTiling,
    merge: MergeChoice,
) -> tuple[SplitChoice, ...]:
    """Return the compound sampler's complete fixed-size destination catalogue."""
    intermediate = tiling.merge(merge)
    axes: tuple[Axis, Axis] = ("horizontal", "vertical")
    catalogue = tuple(SplitChoice(leaf, axis) for leaf in intermediate.leaves for axis in axes)
    expected = 2 * (_K - 1)
    if len(catalogue) != expected:
        raise RuntimeError("relocation destination catalogue has an unexpected size")
    return catalogue


def run_flow_oracle(*, source_revision: str) -> dict[str, object]:
    """Run the complete deterministic R0 flow audit and return its payload."""
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("source_revision must be a non-empty string")
    if FULL_TILING_COMPOUND_SCHEDULE_ID != _EXPECTED_COMPOUND_SCHEDULE_ID:
        raise RuntimeError(
            "the mobile compound schedule identity drifted; review and version "
            "the flow oracle before rerunning it"
        )
    problem = _problem()
    tilings = enumerate_tilings(_SHAPE, _K)
    if len(tilings) != 8:
        raise RuntimeError("the frozen 2x3 K=4 catalogue must contain eight tilings")
    tiling_set = frozenset(tilings)
    topology_hashes = tuple(_topology_sha256(tiling) for tiling in tilings)
    if topology_hashes != _EXPECTED_TOPOLOGY_SHA256:
        raise RuntimeError("the frozen tiny-catalogue topology identity or order drifted")

    maxima = {
        name: _Maximum()
        for name in (
            "auxiliary_swap",
            "component_antisymmetry",
            "target_recovery",
            "jacobian_antisymmetry",
            "ratio_antisymmetry",
            "accepted_flow_equality",
        )
    }
    counts = {
        "edge_attempts": 0,
        "edge_valid": 0,
        "edge_invalid": 0,
        "edge_unique_valid_geometry_paths": 0,
        "relocation_attempts": 0,
        "relocation_valid": 0,
        "relocation_invalid": 0,
        "relocation_unique_valid_geometry_paths": 0,
        "valid_reverses": 0,
        "exact_invalid_self_transitions": 0,
    }
    likelihood_values: list[float] = []
    nonzero_likelihood_deltas = 0

    for tiling_index, tiling in enumerate(tilings):
        source = _source_state(problem, tiling)
        likelihood_values.append(source.log_likelihood)
        merges = merge_choices(tiling)
        expected_edge = frozenset(path.merge for path in edge_flip_paths(tiling))
        expected_relocations = frozenset((path.merge, path.split) for path in relocation_paths(tiling))
        counts["edge_unique_valid_geometry_paths"] += len(expected_edge)
        counts["relocation_unique_valid_geometry_paths"] += len(expected_relocations)

        for fraction in _FRACTIONS:
            actual_edge: set[MergeChoice] = set()
            for merge_index, merge in enumerate(merges):
                context = f"tiling={tiling_index}:edge:merge={merge_index}:fraction={fraction:.2f}"
                transition = propose_posterior_edge_flip(
                    problem,
                    source,
                    merge_choice=merge,
                    new_fraction=fraction,
                )
                counts["edge_attempts"] += 1
                if transition.valid:
                    counts["edge_valid"] += 1
                    counts["valid_reverses"] += 1
                    actual_edge.add(merge)
                    likelihood_values.append(transition.candidate.log_likelihood)
                    if transition.delta_log_likelihood != 0.0:
                        nonzero_likelihood_deltas += 1
                    if transition.candidate.allocation.tiling not in tiling_set:
                        raise RuntimeError(f"{context}: candidate is outside tiny catalogue")
                    expected_selection = -math.log(len(merges))
                    if transition.log_q_forward_selection != expected_selection:
                        raise RuntimeError(f"{context}: edge selection law drifted")
                    _audit_valid_transition(
                        problem,
                        source,
                        transition,
                        context=context,
                        maxima=maxima,
                    )
                else:
                    counts["edge_invalid"] += 1
                    if not _invalid_transition_is_explicit_self(transition, source):
                        raise RuntimeError(
                            f"{context}: invalid edge attempt is not an explicit self-transition"
                        )
                    counts["exact_invalid_self_transitions"] += 1
            if frozenset(actual_edge) != expected_edge:
                raise RuntimeError(
                    f"tiling={tiling_index}: posterior edge validity differs from geometry oracle"
                )

            actual_relocations: set[tuple[MergeChoice, SplitChoice]] = set()
            for merge_index, merge in enumerate(merges):
                catalogue = _relocation_catalogue(tiling, merge)
                for split_index, split in enumerate(catalogue):
                    context = (
                        f"tiling={tiling_index}:relocation:merge={merge_index}:"
                        f"split={split_index}:fraction={fraction:.2f}"
                    )
                    transition = propose_posterior_resolution_relocation(
                        problem,
                        source,
                        merge_choice=merge,
                        split_choice=split,
                        new_fraction=fraction,
                    )
                    counts["relocation_attempts"] += 1
                    if transition.valid:
                        counts["relocation_valid"] += 1
                        counts["valid_reverses"] += 1
                        actual_relocations.add((merge, split))
                        likelihood_values.append(transition.candidate.log_likelihood)
                        if transition.delta_log_likelihood != 0.0:
                            nonzero_likelihood_deltas += 1
                        if transition.candidate.allocation.tiling not in tiling_set:
                            raise RuntimeError(f"{context}: candidate is outside tiny catalogue")
                        expected_selection = -math.log(len(merges)) - math.log(2 * (_K - 1))
                        if transition.log_q_forward_selection != expected_selection:
                            raise RuntimeError(f"{context}: relocation selection law drifted")
                        _audit_valid_transition(
                            problem,
                            source,
                            transition,
                            context=context,
                            maxima=maxima,
                        )
                    else:
                        counts["relocation_invalid"] += 1
                        if not _invalid_transition_is_explicit_self(transition, source):
                            raise RuntimeError(
                                f"{context}: invalid relocation attempt is not an explicit self-transition"
                            )
                        counts["exact_invalid_self_transitions"] += 1
            if frozenset(actual_relocations) != expected_relocations:
                missing = expected_relocations - frozenset(actual_relocations)
                extra = frozenset(actual_relocations) - expected_relocations
                raise RuntimeError(
                    f"tiling={tiling_index}: posterior relocation validity differs "
                    f"from geometry oracle; missing={len(missing)}, extra={len(extra)}"
                )

    likelihood_min = min(likelihood_values)
    likelihood_max = max(likelihood_values)
    if likelihood_min == likelihood_max or nonzero_likelihood_deltas == 0:
        raise RuntimeError("the frozen likelihood did not distinguish structural states")
    if counts["relocation_invalid"] == 0:
        raise RuntimeError("the fixed relocation catalogue did not exercise invalid entries")
    if counts != _EXPECTED_COUNTS:
        raise RuntimeError("the frozen tiny-catalogue attempt counts drifted")

    model_identity = _expected_model_identity()
    payload: dict[str, object] = {
        "status": "pass",
        "purpose": "exact_tiny_likelihood_aware_structural_mh_flow_audit",
        "source_revision": source_revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "model": model_identity,
        "model_sha256": _json_sha256(model_identity),
        "sampler_law": _expected_sampler_law(),
        "catalogue": {
            "topologies": len(tilings),
            "topology_sha256": list(topology_hashes),
            "fractions": list(_FRACTIONS),
            "relocation_destination_slots_per_merge": 2 * (_K - 1),
            "move_mixture_weights": {
                "edge_flip": 0.5,
                "resolution_relocation": 0.5,
            },
            "counts": counts,
        },
        "checks": {
            "actual_compound_attempt_catalogue": True,
            "geometry_oracle_destination_sets_match": True,
            "invalid_entries_are_explicit_self_transitions": True,
            "authoritative_reverse_topology_and_masses_bit_exact": True,
            "discrete_selection_terms_swap_bit_exact": True,
            "continuous_accounting_swaps_within_tolerance": True,
            "log_acceptance_ratio_is_antisymmetric": True,
            "accepted_pointwise_mh_flow_is_equal": True,
            "likelihood_is_nonconstant": True,
        },
        "likelihood": {
            "minimum_log_likelihood": likelihood_min,
            "maximum_log_likelihood": likelihood_max,
            "range": likelihood_max - likelihood_min,
            "nonzero_transition_deltas": nonzero_likelihood_deltas,
        },
        "tolerance_policy": _expected_tolerance_policy(),
        "maximum_discrepancies": {name: maximum.payload() for name, maximum in maxima.items()},
    }
    return payload


def _audit_envelope(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the checksummed audit envelope."""
    payload_copy = dict(payload)
    return {
        "schema": AUDIT_SCHEMA,
        "payload": payload_copy,
        "payload_sha256": _json_sha256(payload_copy),
    }


def _validate_audit_payload(payload: object) -> dict[str, object]:
    """Validate the complete semantic content of one published audit payload."""
    result = _require_exact_keys(payload, _PAYLOAD_KEYS, name="audit payload")
    if result["status"] != "pass":
        raise ValueError("audit payload is not a passing certificate")
    if result["purpose"] != "exact_tiny_likelihood_aware_structural_mh_flow_audit":
        raise ValueError("audit purpose is incompatible")
    if not _is_lower_hex(result["source_revision"], _LOWER_HEX_40):
        raise ValueError("audit source revision must be lowercase full 40-hex")

    runtime = _require_exact_keys(
        result["runtime"],
        frozenset(("python", "numpy")),
        name="runtime",
    )
    if any(not isinstance(runtime[name], str) or not runtime[name] for name in runtime):
        raise ValueError("runtime versions must be non-empty strings")

    model = result["model"]
    if _canonical_json(model) != _canonical_json(_expected_model_identity()):
        raise ValueError("audit scientific model identity is incompatible")
    if not _is_lower_hex(result["model_sha256"], _LOWER_HEX_64):
        raise ValueError("audit model digest must be lowercase 64-hex")
    if result["model_sha256"] != _json_sha256(model):
        raise ValueError("audit model digest mismatch")
    if _canonical_json(result["sampler_law"]) != _canonical_json(_expected_sampler_law()):
        raise ValueError("audit sampler-law identity is incompatible")
    if FULL_TILING_COMPOUND_SCHEDULE_ID != _EXPECTED_COMPOUND_SCHEDULE_ID:
        raise ValueError("current compound schedule identity differs from the audited law")

    catalogue = _require_exact_keys(
        result["catalogue"],
        _CATALOGUE_KEYS,
        name="catalogue",
    )
    if type(catalogue["topologies"]) is not int or catalogue["topologies"] != 8:
        raise ValueError("catalogue topology count is incompatible")
    if catalogue["topology_sha256"] != list(_EXPECTED_TOPOLOGY_SHA256):
        raise ValueError("catalogue topology hashes or order are incompatible")
    if catalogue["fractions"] != list(_FRACTIONS) or any(
        type(value) is not float for value in cast(list[object], catalogue["fractions"])
    ):
        raise ValueError("catalogue allocation fractions are incompatible")
    if type(catalogue["relocation_destination_slots_per_merge"]) is not int or catalogue[
        "relocation_destination_slots_per_merge"
    ] != 2 * (_K - 1):
        raise ValueError("catalogue relocation slot count is incompatible")
    expected_mixture = {
        "edge_flip": 0.5,
        "resolution_relocation": 0.5,
    }
    if _canonical_json(catalogue["move_mixture_weights"]) != _canonical_json(expected_mixture):
        raise ValueError("catalogue move-mixture weights are incompatible")
    counts = _require_exact_keys(
        catalogue["counts"],
        frozenset(_EXPECTED_COUNTS),
        name="catalogue counts",
    )
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("catalogue counts must be non-negative JSON integers")
    if counts != _EXPECTED_COUNTS:
        raise ValueError("catalogue counts differ from the frozen tiny oracle")
    integer_counts = cast(dict[str, int], counts)
    if integer_counts["edge_attempts"] != (integer_counts["edge_valid"] + integer_counts["edge_invalid"]):
        raise ValueError("edge attempt count relation is inconsistent")
    if integer_counts["relocation_attempts"] != (
        integer_counts["relocation_valid"] + integer_counts["relocation_invalid"]
    ):
        raise ValueError("relocation attempt count relation is inconsistent")
    if integer_counts["valid_reverses"] != (
        integer_counts["edge_valid"] + integer_counts["relocation_valid"]
    ):
        raise ValueError("valid reverse count relation is inconsistent")
    if integer_counts["exact_invalid_self_transitions"] != (
        integer_counts["edge_invalid"] + integer_counts["relocation_invalid"]
    ):
        raise ValueError("invalid self-transition count relation is inconsistent")
    if integer_counts["edge_valid"] != (
        integer_counts["edge_unique_valid_geometry_paths"] * len(_FRACTIONS)
    ) or integer_counts["relocation_valid"] != (
        integer_counts["relocation_unique_valid_geometry_paths"] * len(_FRACTIONS)
    ):
        raise ValueError("valid path/fraction count relation is inconsistent")

    checks = _require_exact_keys(result["checks"], _CHECK_KEYS, name="checks")
    if any(type(value) is not bool or value is not True for value in checks.values()):
        raise ValueError("every audit check must be literal true")

    likelihood = _require_exact_keys(
        result["likelihood"],
        _LIKELIHOOD_KEYS,
        name="likelihood",
    )
    minimum = _require_finite_float(
        likelihood["minimum_log_likelihood"],
        name="minimum_log_likelihood",
    )
    maximum = _require_finite_float(
        likelihood["maximum_log_likelihood"],
        name="maximum_log_likelihood",
    )
    likelihood_range = _require_finite_float(likelihood["range"], name="likelihood range")
    nonzero = likelihood["nonzero_transition_deltas"]
    if type(nonzero) is not int or nonzero < 1:
        raise ValueError("nonzero likelihood-delta count must be a positive integer")
    if maximum <= minimum or likelihood_range != maximum - minimum:
        raise ValueError("likelihood range is inconsistent or non-positive")

    if _canonical_json(result["tolerance_policy"]) != _canonical_json(_expected_tolerance_policy()):
        raise ValueError("audit tolerance policy is incompatible")
    maxima = _require_exact_keys(
        result["maximum_discrepancies"],
        _MAXIMUM_KEYS,
        name="maximum discrepancies",
    )
    for name, raw_maximum in maxima.items():
        maximum_value = _require_exact_keys(
            raw_maximum,
            _MAXIMUM_VALUE_KEYS,
            name=f"maximum discrepancy {name}",
        )
        difference = _require_finite_float(
            maximum_value["absolute_difference"],
            name=f"{name} absolute_difference",
        )
        tolerance = _require_finite_float(
            maximum_value["tolerance"],
            name=f"{name} tolerance",
        )
        fraction = _require_finite_float(
            maximum_value["tolerance_fraction"],
            name=f"{name} tolerance_fraction",
        )
        ulps = _require_finite_float(maximum_value["ulps"], name=f"{name} ulps")
        context = maximum_value["context"]
        if difference < 0.0 or tolerance <= 0.0 or difference > tolerance:
            raise ValueError(f"{name} discrepancy exceeds its positive tolerance")
        if not 0.0 <= fraction <= 1.0 or fraction != difference / tolerance:
            raise ValueError(f"{name} tolerance fraction is inconsistent")
        if ulps < 0.0:
            raise ValueError(f"{name} ULP discrepancy must be non-negative")
        if difference == 0.0:
            if ulps != 0.0:
                raise ValueError(f"{name} zero discrepancy must have zero ULPs")
        else:
            if ulps <= 0.0:
                raise ValueError(f"{name} nonzero discrepancy must have positive ULPs")
            reconstructed_ulp = difference / ulps
            expected_tolerance = max(
                _ABSOLUTE_TOLERANCE_FLOOR,
                _ULP_TOLERANCE_FACTOR * reconstructed_ulp,
            )
            if tolerance != expected_tolerance:
                raise ValueError(f"{name} tolerance is inconsistent with its ULP scale")
        if not isinstance(context, str) or not context:
            raise ValueError(f"{name} discrepancy context must be non-empty")
    return result


def _read_audit(path: Path) -> dict[str, object]:
    """Read and validate one audit envelope."""
    value = _strict_json(path)
    if not isinstance(value, dict) or frozenset(value) != _AUDIT_ENVELOPE_KEYS:
        raise ValueError("audit envelope keys are incompatible")
    if value["schema"] != AUDIT_SCHEMA or not isinstance(value["payload"], dict):
        raise ValueError("audit schema or payload is incompatible")
    if not _is_lower_hex(value["payload_sha256"], _LOWER_HEX_64):
        raise ValueError("audit payload checksum must be lowercase 64-hex")
    if value["payload_sha256"] != _json_sha256(value["payload"]):
        raise ValueError("audit payload checksum mismatch")
    return _validate_audit_payload(value["payload"])


def validate_flow_oracle_bundle(directory: Path) -> dict[str, object]:
    """Validate a completed published flow-oracle bundle."""
    completion_path = directory / COMPLETION_FILENAME
    completion = _strict_json(completion_path)
    if not isinstance(completion, dict) or frozenset(completion) != _COMPLETION_KEYS:
        raise ValueError("completion certificate keys are incompatible")
    if completion["schema"] != COMPLETION_SCHEMA:
        raise ValueError("completion certificate schema is incompatible")
    files = completion["files"]
    if not isinstance(files, dict) or frozenset(files) != frozenset((AUDIT_FILENAME,)):
        raise ValueError("completion certificate file catalogue is incompatible")
    audit_path = directory / AUDIT_FILENAME
    if not _is_lower_hex(files[AUDIT_FILENAME], _LOWER_HEX_64):
        raise ValueError("completion audit checksum must be lowercase 64-hex")
    if files[AUDIT_FILENAME] != _file_sha256(audit_path):
        raise ValueError("completion certificate audit checksum mismatch")
    return _read_audit(audit_path)


def publish_flow_oracle(
    output_directory: Path,
    *,
    source_revision: str,
) -> dict[str, object]:
    """Run, publish, independently validate, and certify the deterministic audit."""
    if not _is_lower_hex(source_revision, _LOWER_HEX_40):
        raise ValueError("source_revision must be lowercase full 40-hex")
    current_head = _clean_git_head()
    if source_revision != current_head:
        raise ValueError("source_revision does not equal the current clean Git HEAD")
    output_directory.mkdir(parents=False, exist_ok=False)
    payload = run_flow_oracle(source_revision=source_revision)
    audit_path = output_directory / AUDIT_FILENAME
    _write_create_only(audit_path, _audit_envelope(payload))
    reopened = _read_audit(audit_path)
    if reopened != payload:
        raise RuntimeError("reopened audit payload differs from the in-memory result")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "files": {AUDIT_FILENAME: _file_sha256(audit_path)},
    }
    _write_create_only(output_directory / COMPLETION_FILENAME, completion)
    validated = validate_flow_oracle_bundle(output_directory)
    if validated != payload:
        raise RuntimeError("completed audit bundle differs from the in-memory result")
    return payload


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "publish_flow_oracle",
    "run_flow_oracle",
    "validate_flow_oracle_bundle",
]
