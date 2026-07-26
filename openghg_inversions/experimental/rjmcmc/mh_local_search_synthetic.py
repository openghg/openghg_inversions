"""Deterministic synthetic experiments for MH-guided local partition search.

This module deliberately separates the practical sampler input from the
sealed evaluation payload.  A :class:`SyntheticTrainingArtifact` contains
only the operator and observations available to the fixed/mobile sampler.
Truth, held-out data, the planted topology, and its construction witness live
only in :class:`SyntheticEvaluationArtifact`.

The experiment is a finite-budget algorithm comparison.  It is not a
partition-posterior convergence workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .fixed_basis_nuts import FixedBasisNUTSData, prepare_fixed_basis_nuts
from .full_tiling import (
    LeafTiling,
    Rectangle,
    SplitChoice,
    TilingState,
    edge_flip_paths,
    relocation_paths,
)
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from .gamma_beta_adapter import gamma_beta_problem_from_rhime_inputs
from .full_tiling_compound_sampling import FullTilingMovementDiagnostics

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Stage = Literal["s0", "s1"]
Scenario = Literal["aligned", "edge-one", "relocation-one"]

DEFINITION_SCHEMA = "openghg_inversions.mh_local_search_synthetic_definition.v1"
TRAINING_SCHEMA = "openghg_inversions.mh_local_search_synthetic_training.v1"
EVALUATION_SCHEMA = "openghg_inversions.mh_local_search_synthetic_evaluation.v1"

_TRACE_RETAINED_FIELDS = (
    "rectangle_bounds",
    "leaf_masses",
    "root_total",
    "fixed_coefficients",
    "log_gaussian_likelihood",
    "log_likelihood",
    "log_root_prior",
    "log_allocation_prior",
    "log_structural_prior",
    "log_fixed_coefficient_prior",
    "log_target",
    "state_transition",
)
_TRACE_ATTEMPT_FIELDS = (
    "global_transition",
    "slot",
    "move",
    "valid",
    "accepted",
    "log_acceptance_ratio",
    "invalid_reason",
)
_MOVEMENT_FIELDS = tuple(item.name for item in fields(FullTilingMovementDiagnostics))

_ENVELOPE_KEYS = frozenset(("schema", "payload", "payload_sha256"))
_TRAINING_KEYS = frozenset(
    (
        "stage",
        "replicate",
        "definition_sha256",
        "cell_id",
        "generation_commitment",
        "shape",
        "k",
        "operator",
        "observations",
        "observation_sd",
        "nominal_weight",
        "row_ids",
        "p0_bounds",
        "conditioning_seed",
        "fixed_seed",
        "mobile_seed",
    )
)
_EVALUATION_KEYS = frozenset(
    (
        "stage",
        "scenario",
        "replicate",
        "definition_sha256",
        "cell_id",
        "generation_commitment",
        "training_observations_sha256",
        "shape",
        "k",
        "truth",
        "heldout_operator",
        "heldout_noiseless",
        "heldout_observations",
        "heldout_sd",
        "heldout_row_ids",
        "pstar_bounds",
        "witness",
    )
)
_DEFINITION_KEYS = frozenset(
    (
        "stage",
        "shape",
        "k",
        "settings",
        "nominal_weight",
        "p0_bounds",
        "p0_sha256",
        "training_operator",
        "training_operator_sha256",
        "training_row_ids",
        "heldout_operator",
        "heldout_operator_sha256",
        "heldout_row_ids",
        "scenarios",
    )
)
_STAGE_SETTINGS: dict[Stage, dict[str, object]] = {
    "s0": {
        "shape": (2, 4),
        "k": 4,
        "observation_sd": 0.05,
        "training_operator_seed": 60000,
        "heldout_operator_seed": 60001,
        "noise_seeds": (61001, 61002, 61003, 61004),
        "conditioning_seeds": (61501, 61502, 61503, 61504),
        "oracle_conditioning_seeds": (61601, 61602, 61603, 61604),
        "fixed_seeds": (62001, 62002, 62003, 62004),
        "oracle_seeds": (62501, 62502, 62503, 62504),
        "mobile_seeds": (63001, 63002, 63003, 63004),
        "conditioning_cycles": 2_000,
        "production_cycles": 5_000,
    },
    "s1": {
        "shape": (8, 8),
        "k": 8,
        "observation_sd": 0.08,
        "training_operator_seed": 71000,
        "heldout_operator_seed": 71001,
        "noise_seeds": (71101, 71102, 71103, 71104),
        "conditioning_seeds": (71501, 71502, 71503, 71504),
        "oracle_conditioning_seeds": (71601, 71602, 71603, 71604),
        "fixed_seeds": (72001, 72002, 72003, 72004),
        "oracle_seeds": (72501, 72502, 72503, 72504),
        "mobile_seeds": (73001, 73002, 73003, 73004),
        "conditioning_cycles": 10_000,
        "production_cycles": 50_000,
    },
}


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _sha256_text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return value


def canonical_json(value: object) -> str:
    """Return strict, deterministic JSON without a trailing newline."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON value."""
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def cell_commitment(
    definition_sha256: str,
    scenario: Scenario,
    replicate: int,
) -> str:
    """Return an opaque commitment to one sealed stage/scenario/replicate cell."""
    definition_digest = _sha256_text(
        definition_sha256,
        name="definition_sha256",
    )
    if not isinstance(scenario, str) or scenario not in ("aligned", "edge-one", "relocation-one"):
        raise ValueError("unsupported scenario")
    replicate_index = _exact_int(replicate, name="replicate")
    return json_sha256(
        {
            "definition_sha256": definition_digest,
            "scenario": scenario,
            "replicate": replicate_index,
        }
    )


def observation_generation_commitment(
    definition_sha256: str,
    cell_id: str,
    training_observations: FloatArray,
    heldout_observations: FloatArray,
) -> str:
    """Commit opaquely to the exact paired deterministic noise realization."""
    return json_sha256(
        {
            "schema": "mh_local_search_observation_generation_v1",
            "definition_sha256": _sha256_text(
                definition_sha256,
                name="definition_sha256",
            ),
            "cell_id": _sha256_text(cell_id, name="cell_id"),
            "training_observations_sha256": json_sha256(
                np.asarray(training_observations, dtype=np.float64).tolist()
            ),
            "heldout_observations_sha256": json_sha256(
                np.asarray(heldout_observations, dtype=np.float64).tolist()
            ),
        }
    )


def file_sha256(path: Path) -> str:
    """Return the streaming SHA-256 of one file."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def topology_bounds(tiling: LeafTiling) -> tuple[tuple[int, int, int, int], ...]:
    """Return canonical rectangle bounds for ``tiling``."""
    return tuple((leaf.row_start, leaf.row_stop, leaf.col_start, leaf.col_stop) for leaf in tiling.leaves)


def topology_sha256(tiling: LeafTiling) -> str:
    """Return a canonical topology digest."""
    return json_sha256(topology_bounds(tiling))


def tiling_from_bounds(
    shape: tuple[int, int],
    bounds: object,
) -> LeafTiling:
    """Build a validated canonical tiling from integer rectangle bounds."""
    try:
        rows = tuple(
            tuple(_exact_int(value, name="rectangle bound") for value in cast(Any, row))
            for row in cast(Any, bounds)
        )
    except TypeError as error:
        raise ValueError("rectangle bounds must be an iterable of integer quadruples") from error
    if any(len(row) != 4 for row in rows):
        raise ValueError("every rectangle bound must contain four integers")
    return LeafTiling(shape, tuple(Rectangle(*row) for row in rows))


def validate_local_reference_trace(
    path: Path,
    *,
    manifest: Mapping[str, object],
    shape: tuple[int, int],
    k: int,
    expected_bounds: IntArray | None = None,
    problem: FullTilingProblem | None = None,
) -> dict[str, NDArray[Any]]:
    """Load and exactly validate one fixed-basis local-reference trace.

    This is the single raw-trace validator used both when the producer writes
    a local-reference chain and when the conditional-reference certifier
    reopens it.  Besides the exact archive schema, it checks all transition
    coordinates and diagnostic arrays.  Supplying ``problem`` additionally
    rebuilds every retained scientific state and requires exact target
    components, so a self-consistently rehashed trace cannot substitute
    altered scientific coordinates or cached targets.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected_keys = {
        *_TRACE_RETAINED_FIELDS,
        *(f"attempt_{name}" for name in _TRACE_ATTEMPT_FIELDS),
        *(f"movement_{name}" for name in _MOVEMENT_FIELDS),
        "cycle",
        "chunk_end_cycle",
        "chunk_sampler_seconds",
    }
    if set(trace) != expected_keys:
        raise ValueError("fixed local-reference trace fields are incompatible")

    expected_dtypes: dict[str, np.dtype[Any]] = {
        "rectangle_bounds": np.dtype(np.int64),
        "leaf_masses": np.dtype(np.float64),
        "root_total": np.dtype(np.float64),
        "fixed_coefficients": np.dtype(np.float64),
        "log_gaussian_likelihood": np.dtype(np.float64),
        "log_likelihood": np.dtype(np.float64),
        "log_root_prior": np.dtype(np.float64),
        "log_allocation_prior": np.dtype(np.float64),
        "log_structural_prior": np.dtype(np.float64),
        "log_fixed_coefficient_prior": np.dtype(np.float64),
        "log_target": np.dtype(np.float64),
        "state_transition": np.dtype(np.int64),
        "attempt_global_transition": np.dtype(np.int64),
        "attempt_slot": np.dtype("U15"),
        "attempt_move": np.dtype("U24"),
        "attempt_valid": np.dtype(np.bool_),
        "attempt_accepted": np.dtype(np.bool_),
        "attempt_log_acceptance_ratio": np.dtype(np.float64),
        "attempt_invalid_reason": np.dtype("U96"),
        "cycle": np.dtype(np.int64),
        "chunk_end_cycle": np.dtype(np.int64),
        "chunk_sampler_seconds": np.dtype(np.float64),
    }
    movement_integer = (
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
    movement_float = (
        "changed_nominal_mass",
        "standardized_prediction_l2",
        "root_abs_displacement",
        "root_abs_log_displacement",
        "allocation_share_l1_displacement",
        "fixed_abs_displacement",
        "fixed_abs_log_displacement",
    )
    for name in movement_integer:
        expected_dtypes[f"movement_{name}"] = np.dtype(np.int64)
    for name in movement_float:
        expected_dtypes[f"movement_{name}"] = np.dtype(np.float64)
    expected_dtypes["movement_move"] = np.dtype("U24")
    expected_dtypes["movement_valid"] = np.dtype(np.bool_)
    expected_dtypes["movement_accepted"] = np.dtype(np.bool_)
    if any(trace[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("fixed local-reference trace has an incompatible exact dtype")

    cycles = _exact_int(
        manifest.get("production_cycles"),
        name="manifest production_cycles",
        minimum=1,
    )
    pair_slots = _exact_int(
        manifest.get("pair_slots"),
        name="manifest pair_slots",
        minimum=1,
    )
    chunk_cycles = _exact_int(
        manifest.get("chunk_cycles"),
        name="manifest chunk_cycles",
        minimum=1,
    )
    cycle_length = 1 + pair_slots
    attempts = cycles * cycle_length
    if not np.array_equal(trace["cycle"], np.arange(1, cycles + 1, dtype=np.int64)):
        raise ValueError("fixed local-reference retained cycle coordinates are inconsistent")
    if not np.array_equal(
        trace["state_transition"],
        cycle_length * np.arange(1, cycles + 1, dtype=np.int64),
    ):
        raise ValueError("fixed local-reference retained transition coordinates are inconsistent")
    if not np.array_equal(
        trace["attempt_global_transition"],
        np.arange(1, attempts + 1, dtype=np.int64),
    ):
        raise ValueError("fixed local-reference attempt coordinates are inconsistent")
    expected_slots = np.tile(
        np.asarray(("root", *("pair_allocation" for _ in range(pair_slots))), dtype="U15"),
        cycles,
    )
    expected_moves = np.tile(
        np.asarray(
            ("root_total_slice", *("pair_allocation_refresh" for _ in range(pair_slots))), dtype="U24"
        ),
        cycles,
    )
    if not np.array_equal(trace["attempt_slot"], expected_slots) or not np.array_equal(
        trace["attempt_move"],
        expected_moves,
    ):
        raise ValueError("fixed local-reference attempt schedule is inconsistent")
    expected_chunks = np.arange(chunk_cycles, cycles + 1, chunk_cycles, dtype=np.int64)
    if (
        cycles % chunk_cycles
        or not np.array_equal(trace["chunk_end_cycle"], expected_chunks)
        or trace["chunk_sampler_seconds"].shape != expected_chunks.shape
        or np.any(~np.isfinite(trace["chunk_sampler_seconds"]))
        or np.any(trace["chunk_sampler_seconds"] < 0.0)
    ):
        raise ValueError("fixed local-reference chunk timing coordinates are inconsistent")

    bounds = trace["rectangle_bounds"]
    masses = trace["leaf_masses"]
    roots = trace["root_total"]
    fixed = trace["fixed_coefficients"]
    if (
        bounds.shape != (cycles, k, 4)
        or masses.shape != (cycles, k)
        or roots.shape != (cycles,)
        or fixed.ndim != 2
        or fixed.shape[0] != cycles
        or np.any(~np.isfinite(masses))
        or np.any(masses <= 0.0)
        or np.any(~np.isfinite(roots))
        or np.any(roots <= 0.0)
        or not np.array_equal(roots, np.sum(masses, axis=1))
    ):
        raise ValueError("fixed local-reference scientific coordinates are inconsistent")
    expected_bounds_array: IntArray | None = None
    if expected_bounds is not None:
        expected_bounds_array = np.asarray(expected_bounds)
        if (
            expected_bounds_array.dtype != np.dtype(np.int64)
            or expected_bounds_array.shape != (k, 4)
            or np.any(bounds != expected_bounds_array[None, :, :])
        ):
            raise ValueError("fixed local-reference topology differs from the certified topology")
    elif np.any(bounds != bounds[0]):
        raise ValueError("fixed local-reference trace changed topology")

    target_fields = (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_structural_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    )
    if any(trace[name].shape != (cycles,) or np.any(~np.isfinite(trace[name])) for name in target_fields):
        raise ValueError("fixed local-reference retained target arrays are inconsistent")
    if np.any(trace["log_structural_prior"] != 0.0):
        raise ValueError("fixed local-reference structural target must be identically zero")

    for name in _TRACE_ATTEMPT_FIELDS:
        if trace[f"attempt_{name}"].shape != (attempts,):
            raise ValueError(f"fixed local-reference attempt {name} shape is inconsistent")
    if (
        np.any(trace["attempt_accepted"] & ~trace["attempt_valid"])
        or np.any(np.isnan(trace["attempt_log_acceptance_ratio"]))
        or np.any(trace["attempt_log_acceptance_ratio"] == math.inf)
    ):
        raise ValueError("fixed local-reference attempt diagnostics are inconsistent")
    movement = FullTilingMovementDiagnostics(**{name: trace[f"movement_{name}"] for name in _MOVEMENT_FIELDS})
    for name in ("global_transition", "move", "valid", "accepted"):
        if not np.array_equal(
            getattr(movement, name),
            trace[f"attempt_{name}"],
        ):
            raise ValueError("fixed local-reference movement diagnostics disagree with attempts")
    if np.any(np.isin(trace["attempt_move"], ("edge_flip", "resolution_relocation"))):
        raise ValueError("fixed local-reference trace entered structural proposal code")

    if problem is not None:
        if problem.shape != shape:
            raise ValueError("fixed local-reference problem shape is incompatible")
        for draw in range(cycles):
            tiling = tiling_from_bounds(shape, bounds[draw])
            if tiling.k != k:
                raise ValueError("fixed local-reference retained topology has inconsistent K")
            state = build_full_tiling_posterior_state(
                problem,
                allocation=TilingState(tiling, masses[draw]),
                fixed_coefficients=fixed[draw],
            )
            exact = {
                "root_total": state.root_total,
                "log_gaussian_likelihood": state.log_gaussian_likelihood,
                "log_likelihood": state.log_likelihood,
                "log_root_prior": state.log_root_prior,
                "log_allocation_prior": state.log_allocation_prior,
                "log_structural_prior": 0.0,
                "log_fixed_coefficient_prior": state.log_fixed_coefficient_prior,
                "log_target": state.log_target,
            }
            if any(trace[name][draw].item() != expected for name, expected in exact.items()):
                raise ValueError(
                    f"fixed local-reference scientific target at cycle {draw + 1} does not rebuild exactly"
                )
    else:
        for draw, draw_bounds in enumerate(bounds):
            if tiling_from_bounds(shape, draw_bounds).k != k:
                raise ValueError(
                    f"fixed local-reference retained topology at cycle {draw + 1} has inconsistent K"
                )
    return trace


def _readonly_float(values: object, *, name: str, ndim: int) -> FloatArray:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if source.dtype.kind not in "fiu" or source.dtype.kind == "b":
        raise ValueError(f"{name} must contain only JSON numeric values")
    array = np.asarray(source, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _shape(value: object) -> tuple[int, int]:
    try:
        values = tuple(_exact_int(item, name="shape entry", minimum=1) for item in cast(Any, value))
    except TypeError as error:
        raise ValueError("shape must contain two integers") from error
    if len(values) != 2 or any(item < 1 for item in values):
        raise ValueError("shape must contain two positive integers")
    return values


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    try:
        result = tuple(cast(Any, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of strings") from error
    if any(not isinstance(item, str) for item in result):
        raise ValueError(f"{name} must contain only strings")
    return cast(tuple[str, ...], result)


@dataclass(frozen=True, slots=True)
class SyntheticTrainingArtifact:
    """Practical-sampler input with no evaluation payload."""

    stage: Stage
    replicate: int
    definition_sha256: str
    cell_id: str
    generation_commitment: str
    shape: tuple[int, int]
    k: int
    operator: FloatArray
    observations: FloatArray
    observation_sd: FloatArray
    nominal_weight: FloatArray
    row_ids: tuple[str, ...]
    p0_bounds: tuple[tuple[int, int, int, int], ...]
    conditioning_seed: int
    fixed_seed: int
    mobile_seed: int

    def __post_init__(self) -> None:
        """Validate and freeze the training-only payload."""
        if not isinstance(self.stage, str) or self.stage not in _STAGE_SETTINGS:
            raise ValueError("unsupported stage")
        definition = build_stage_definition(self.stage)
        definition_digest = json_sha256(definition)
        if _sha256_text(self.definition_sha256, name="definition_sha256") != definition_digest:
            raise ValueError("training definition_sha256 does not identify the frozen definition")
        _sha256_text(self.cell_id, name="cell_id")
        _sha256_text(
            self.generation_commitment,
            name="generation_commitment",
        )
        shape = _shape(self.shape)
        replicate = _exact_int(self.replicate, name="replicate")
        k = _exact_int(self.k, name="k", minimum=1)
        settings = cast(Mapping[str, object], definition["settings"])
        if replicate >= len(cast(Any, settings["noise_seeds"])):
            raise ValueError("replicate lies outside the frozen stage")
        if shape != _shape(definition["shape"]) or k != _exact_int(
            definition["k"],
            name="definition k",
            minimum=1,
        ):
            raise ValueError("training shape or k differs from the frozen definition")
        operator = _readonly_float(self.operator, name="operator", ndim=2)
        observations = _readonly_float(self.observations, name="observations", ndim=1)
        observation_sd = _readonly_float(
            self.observation_sd,
            name="observation_sd",
            ndim=1,
        )
        nominal = _readonly_float(self.nominal_weight, name="nominal_weight", ndim=2)
        if operator.shape != (observations.size, shape[0] * shape[1]):
            raise ValueError("operator shape is inconsistent with observations and grid")
        if observation_sd.shape != observations.shape or np.any(observation_sd <= 0.0):
            raise ValueError("observation_sd must be positive and aligned")
        if nominal.shape != shape or np.any(nominal <= 0.0):
            raise ValueError("nominal_weight must be positive and match shape")
        if (
            len(self.row_ids) != observations.size
            or any(not isinstance(item, str) for item in self.row_ids)
            or len(set(self.row_ids)) != len(self.row_ids)
        ):
            raise ValueError("training row IDs must be unique and aligned")
        expected_row_ids = tuple(str(item) for item in cast(Any, definition["training_row_ids"]))
        if (
            self.row_ids != expected_row_ids
            or not np.array_equal(
                operator,
                np.asarray(definition["training_operator"], dtype=np.float64),
            )
            or not np.array_equal(
                nominal,
                np.asarray(definition["nominal_weight"], dtype=np.float64),
            )
            or not np.array_equal(
                observation_sd,
                np.full(
                    observations.size,
                    float(cast(Any, settings["observation_sd"])),
                ),
            )
        ):
            raise ValueError("training design differs from the frozen definition")
        p0 = tiling_from_bounds(shape, self.p0_bounds)
        if p0.k != k or topology_bounds(p0) != tuple(
            tuple(row) for row in cast(Any, definition["p0_bounds"])
        ):
            raise ValueError("P0 must contain exactly k leaves")
        seeds = {
            "conditioning_seed": settings["conditioning_seeds"],
            "fixed_seed": settings["fixed_seeds"],
            "mobile_seed": settings["mobile_seeds"],
        }
        for name, catalogue in seeds.items():
            supplied = _exact_int(getattr(self, name), name=name)
            expected = _exact_int(cast(Any, catalogue)[replicate], name=f"frozen {name}")
            if supplied != expected:
                raise ValueError(f"{name} differs from the frozen definition")
            object.__setattr__(self, name, supplied)
        object.__setattr__(self, "replicate", replicate)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_sd", observation_sd)
        object.__setattr__(self, "nominal_weight", nominal)
        object.__setattr__(self, "p0_bounds", topology_bounds(p0))

    def payload(self) -> dict[str, object]:
        """Return a strict JSON training-only payload."""
        return {
            "stage": self.stage,
            "replicate": self.replicate,
            "definition_sha256": self.definition_sha256,
            "cell_id": self.cell_id,
            "generation_commitment": self.generation_commitment,
            "shape": list(self.shape),
            "k": self.k,
            "operator": self.operator.tolist(),
            "observations": self.observations.tolist(),
            "observation_sd": self.observation_sd.tolist(),
            "nominal_weight": self.nominal_weight.tolist(),
            "row_ids": list(self.row_ids),
            "p0_bounds": [list(row) for row in self.p0_bounds],
            "conditioning_seed": self.conditioning_seed,
            "fixed_seed": self.fixed_seed,
            "mobile_seed": self.mobile_seed,
        }


@dataclass(frozen=True, slots=True)
class SyntheticEvaluationArtifact:
    """Sealed truth and held-out payload unavailable to practical sampling."""

    stage: Stage
    scenario: Scenario
    replicate: int
    definition_sha256: str
    cell_id: str
    generation_commitment: str
    training_observations_sha256: str
    shape: tuple[int, int]
    k: int
    truth: FloatArray
    heldout_operator: FloatArray
    heldout_noiseless: FloatArray
    heldout_observations: FloatArray
    heldout_sd: FloatArray
    heldout_row_ids: tuple[str, ...]
    pstar_bounds: tuple[tuple[int, int, int, int], ...]
    witness: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate and freeze the evaluation-only payload."""
        if not isinstance(self.stage, str) or self.stage not in _STAGE_SETTINGS:
            raise ValueError("unsupported stage")
        if not isinstance(self.scenario, str) or self.scenario not in (
            "aligned",
            "edge-one",
            "relocation-one",
        ):
            raise ValueError("unsupported scenario")
        replicate = _exact_int(self.replicate, name="replicate")
        definition = build_stage_definition(self.stage)
        definition_digest = json_sha256(definition)
        if _sha256_text(self.definition_sha256, name="definition_sha256") != definition_digest:
            raise ValueError("evaluation definition_sha256 does not identify the frozen definition")
        if _sha256_text(self.cell_id, name="cell_id") != cell_commitment(
            definition_digest,
            self.scenario,
            replicate,
        ):
            raise ValueError("evaluation cell_id does not match its sealed cell")
        _sha256_text(
            self.generation_commitment,
            name="generation_commitment",
        )
        _sha256_text(
            self.training_observations_sha256,
            name="training_observations_sha256",
        )
        settings = cast(Mapping[str, object], definition["settings"])
        if replicate >= len(cast(Any, settings["noise_seeds"])):
            raise ValueError("replicate lies outside the frozen stage")
        shape = _shape(self.shape)
        k = _exact_int(self.k, name="k", minimum=1)
        if shape != _shape(definition["shape"]) or k != _exact_int(
            definition["k"],
            name="definition k",
            minimum=1,
        ):
            raise ValueError("evaluation shape or k differs from the frozen definition")
        truth = _readonly_float(self.truth, name="truth", ndim=2)
        heldout = _readonly_float(
            self.heldout_operator,
            name="heldout_operator",
            ndim=2,
        )
        noiseless = _readonly_float(
            self.heldout_noiseless,
            name="heldout_noiseless",
            ndim=1,
        )
        observations = _readonly_float(
            self.heldout_observations,
            name="heldout_observations",
            ndim=1,
        )
        sd = _readonly_float(self.heldout_sd, name="heldout_sd", ndim=1)
        if truth.shape != shape:
            raise ValueError("truth must match shape")
        if heldout.shape != (noiseless.size, truth.size):
            raise ValueError("heldout operator has incompatible shape")
        if observations.shape != noiseless.shape or sd.shape != noiseless.shape:
            raise ValueError("heldout vectors must align")
        if np.any(sd <= 0.0):
            raise ValueError("heldout_sd must be positive")
        if not isinstance(self.witness, Mapping):
            raise ValueError("witness must be a mapping")
        if (
            len(self.heldout_row_ids) != noiseless.size
            or any(not isinstance(item, str) for item in self.heldout_row_ids)
            or len(set(self.heldout_row_ids)) != len(self.heldout_row_ids)
        ):
            raise ValueError("heldout row IDs must align")
        pstar = tiling_from_bounds(shape, self.pstar_bounds)
        scenario_map = cast(
            Mapping[str, object],
            cast(Mapping[str, object], definition["scenarios"])[self.scenario],
        )
        expected_ids = tuple(str(item) for item in cast(Any, definition["heldout_row_ids"]))
        if (
            pstar.k != k
            or topology_bounds(pstar) != tuple(tuple(row) for row in cast(Any, scenario_map["pstar_bounds"]))
            or self.heldout_row_ids != expected_ids
            or not np.array_equal(
                heldout,
                np.asarray(definition["heldout_operator"], dtype=np.float64),
            )
            or not np.array_equal(
                truth,
                np.asarray(scenario_map["truth"], dtype=np.float64),
            )
            or canonical_json(dict(self.witness))
            != canonical_json(dict(cast(Mapping[str, object], scenario_map["witness"])))
            or not np.array_equal(
                sd,
                np.full(
                    observations.size,
                    float(cast(Any, settings["observation_sd"])),
                ),
            )
        ):
            raise ValueError("evaluation payload differs from the frozen definition")
        if not np.array_equal(noiseless, heldout @ truth.ravel(order="C")):
            raise ValueError("heldout_noiseless must equal heldout_operator @ truth")
        (
            _,
            expected_training,
            expected_heldout_noiseless,
            expected_heldout,
        ) = _replay_observations(
            definition,
            scenario=self.scenario,
            replicate=replicate,
        )
        expected_training_digest = json_sha256(expected_training.tolist())
        expected_generation = observation_generation_commitment(
            definition_digest,
            self.cell_id,
            expected_training,
            expected_heldout,
        )
        if (
            self.training_observations_sha256 != expected_training_digest
            or self.generation_commitment != expected_generation
            or not np.array_equal(noiseless, expected_heldout_noiseless)
            or not np.array_equal(observations, expected_heldout)
        ):
            raise ValueError("evaluation observation realization does not replay")
        if pstar.k != k:
            raise ValueError("Pstar must contain exactly k leaves")
        object.__setattr__(self, "replicate", replicate)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "truth", truth)
        object.__setattr__(self, "heldout_operator", heldout)
        object.__setattr__(self, "heldout_noiseless", noiseless)
        object.__setattr__(self, "heldout_observations", observations)
        object.__setattr__(self, "heldout_sd", sd)
        object.__setattr__(self, "pstar_bounds", topology_bounds(pstar))
        object.__setattr__(self, "witness", dict(self.witness))

    def payload(self) -> dict[str, object]:
        """Return a strict JSON evaluation payload."""
        return {
            "stage": self.stage,
            "scenario": self.scenario,
            "replicate": self.replicate,
            "definition_sha256": self.definition_sha256,
            "cell_id": self.cell_id,
            "generation_commitment": self.generation_commitment,
            "training_observations_sha256": self.training_observations_sha256,
            "shape": list(self.shape),
            "k": self.k,
            "truth": self.truth.tolist(),
            "heldout_operator": self.heldout_operator.tolist(),
            "heldout_noiseless": self.heldout_noiseless.tolist(),
            "heldout_observations": self.heldout_observations.tolist(),
            "heldout_sd": self.heldout_sd.tolist(),
            "heldout_row_ids": list(self.heldout_row_ids),
            "pstar_bounds": [list(row) for row in self.pstar_bounds],
            "witness": dict(self.witness),
        }


def _envelope(schema: str, payload: Mapping[str, object]) -> dict[str, object]:
    payload_copy = dict(payload)
    return {
        "schema": schema,
        "payload": payload_copy,
        "payload_sha256": json_sha256(payload_copy),
    }


def write_envelope(path: Path, schema: str, payload: Mapping[str, object]) -> str:
    """Create one strict checksum envelope and return its file digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(_envelope(schema, payload)) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
    return file_sha256(path)


def read_envelope(path: Path, *, schema: str) -> dict[str, object]:
    """Read one strict checksum envelope, rejecting drift and corruption."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("artifact is not strict JSON") from error
    if not isinstance(value, dict) or frozenset(value) != _ENVELOPE_KEYS:
        raise ValueError("artifact envelope keys are incompatible")
    if value["schema"] != schema or not isinstance(value["payload"], dict):
        raise ValueError("artifact schema or payload is incompatible")
    if _sha256_text(value["payload_sha256"], name="payload_sha256") != json_sha256(value["payload"]):
        raise ValueError("artifact payload checksum mismatch")
    return cast(dict[str, object], value["payload"])


def _training_from_payload(payload: Mapping[str, object]) -> SyntheticTrainingArtifact:
    if frozenset(payload) != _TRAINING_KEYS:
        raise ValueError("training payload keys are incompatible")
    return SyntheticTrainingArtifact(
        stage=cast(Stage, payload["stage"]),
        replicate=_exact_int(payload["replicate"], name="replicate"),
        definition_sha256=_sha256_text(
            payload["definition_sha256"],
            name="definition_sha256",
        ),
        cell_id=_sha256_text(payload["cell_id"], name="cell_id"),
        generation_commitment=_sha256_text(
            payload["generation_commitment"],
            name="generation_commitment",
        ),
        shape=_shape(payload["shape"]),
        k=_exact_int(payload["k"], name="k", minimum=1),
        operator=cast(Any, payload["operator"]),
        observations=cast(Any, payload["observations"]),
        observation_sd=cast(Any, payload["observation_sd"]),
        nominal_weight=cast(Any, payload["nominal_weight"]),
        row_ids=_string_tuple(payload["row_ids"], name="row_ids"),
        p0_bounds=cast(Any, payload["p0_bounds"]),
        conditioning_seed=_exact_int(
            payload["conditioning_seed"],
            name="conditioning_seed",
        ),
        fixed_seed=_exact_int(payload["fixed_seed"], name="fixed_seed"),
        mobile_seed=_exact_int(payload["mobile_seed"], name="mobile_seed"),
    )


def _evaluation_from_payload(
    payload: Mapping[str, object],
) -> SyntheticEvaluationArtifact:
    if frozenset(payload) != _EVALUATION_KEYS:
        raise ValueError("evaluation payload keys are incompatible")
    return SyntheticEvaluationArtifact(
        stage=cast(Stage, payload["stage"]),
        scenario=cast(Scenario, payload["scenario"]),
        replicate=_exact_int(payload["replicate"], name="replicate"),
        definition_sha256=_sha256_text(
            payload["definition_sha256"],
            name="definition_sha256",
        ),
        cell_id=_sha256_text(payload["cell_id"], name="cell_id"),
        generation_commitment=_sha256_text(
            payload["generation_commitment"],
            name="generation_commitment",
        ),
        training_observations_sha256=_sha256_text(
            payload["training_observations_sha256"],
            name="training_observations_sha256",
        ),
        shape=_shape(payload["shape"]),
        k=_exact_int(payload["k"], name="k", minimum=1),
        truth=cast(Any, payload["truth"]),
        heldout_operator=cast(Any, payload["heldout_operator"]),
        heldout_noiseless=cast(Any, payload["heldout_noiseless"]),
        heldout_observations=cast(Any, payload["heldout_observations"]),
        heldout_sd=cast(Any, payload["heldout_sd"]),
        heldout_row_ids=_string_tuple(
            payload["heldout_row_ids"],
            name="heldout_row_ids",
        ),
        pstar_bounds=cast(Any, payload["pstar_bounds"]),
        witness=cast(Mapping[str, object], payload["witness"]),
    )


def load_training_artifact(path: Path) -> SyntheticTrainingArtifact:
    """Load and validate one training-only artifact."""
    return _training_from_payload(read_envelope(path, schema=TRAINING_SCHEMA))


def load_evaluation_artifact(path: Path) -> SyntheticEvaluationArtifact:
    """Load and validate one sealed evaluation artifact."""
    return _evaluation_from_payload(read_envelope(path, schema=EVALUATION_SCHEMA))


def validate_artifact_pair(
    training: SyntheticTrainingArtifact,
    evaluation: SyntheticEvaluationArtifact,
) -> None:
    """Require one practical/sealed pair and its exact replayed noise realization."""
    if (
        training.stage,
        training.replicate,
        training.definition_sha256,
        training.cell_id,
        training.shape,
        training.k,
        training.generation_commitment,
    ) != (
        evaluation.stage,
        evaluation.replicate,
        evaluation.definition_sha256,
        evaluation.cell_id,
        evaluation.shape,
        evaluation.k,
        evaluation.generation_commitment,
    ):
        raise ValueError("training and evaluation artifacts do not identify the same cell")
    if json_sha256(training.observations.tolist()) != evaluation.training_observations_sha256:
        raise ValueError("training observations do not match the sealed noise replay")


def _build_problem(
    *,
    shape: tuple[int, int],
    k: int,
    operator: FloatArray,
    observations: FloatArray,
    observation_sd: FloatArray,
    nominal_weight: FloatArray,
) -> FullTilingProblem:
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("nmeasure", "lat", "lon"),
                operator.reshape(operator.shape[0], *shape),
            ),
            "mf": ("nmeasure", observations),
            "mf_error": ("nmeasure", observation_sd),
        },
        coords={
            "nmeasure": np.arange(operator.shape[0], dtype=np.int64),
            "lat": np.arange(shape[0], dtype=np.float64),
            "lon": np.arange(shape[1], dtype=np.float64),
        },
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=nominal_weight,
        k_min=k,
        k_max=k,
        concentration=float(2 * k),
        root_variance=0.25,
        likelihood_power=1.0,
    )
    return full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=float(2 * k),
    )


def problem_from_training(
    training: SyntheticTrainingArtifact,
) -> FullTilingProblem:
    """Construct the exact full-tiling problem from training-only data."""
    return _build_problem(
        shape=training.shape,
        k=training.k,
        operator=training.operator,
        observations=training.observations,
        observation_sd=training.observation_sd,
        nominal_weight=training.nominal_weight,
    )


def state_on_tiling(
    problem: FullTilingProblem,
    tiling: LeafTiling,
) -> FullTilingPosteriorState:
    """Build the prior-mean continuous state on an arbitrary valid tiling."""
    if tiling.shape != problem.shape:
        raise ValueError("tiling and problem shapes differ")
    root_mean = problem.base.prior.root_shape / problem.base.prior.root_rate
    nominal = np.asarray(
        [problem.rectangle_nominal_mass(leaf) for leaf in tiling.leaves],
        dtype=np.float64,
    )
    masses = root_mean * nominal / float(nominal.sum())
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(tiling, masses),
    )


def prepare_fixed_basis_reference(
    training: SyntheticTrainingArtifact,
    bounds: object,
) -> tuple[FullTilingProblem, FullTilingPosteriorState, FixedBasisNUTSData]:
    """Prepare the existing NUTS bridge on a caller-supplied topology."""
    problem = problem_from_training(training)
    state = state_on_tiling(problem, tiling_from_bounds(training.shape, bounds))
    return problem, state, prepare_fixed_basis_nuts(problem, state)


def _operator_s0() -> tuple[FloatArray, tuple[str, ...], FloatArray, tuple[str, ...]]:
    shape = (2, 4)
    direct = np.eye(8, dtype=np.float64)
    training = np.repeat(direct, 2, axis=0)
    train_ids = [f"direct-r{cell // 4}-c{cell % 4}-rep{rep}" for cell in range(8) for rep in range(2)]
    permutation = np.random.Generator(np.random.PCG64(60000)).permutation(training.shape[0])
    training = training[permutation]
    train_ids = [train_ids[int(index)] for index in permutation]

    rows: list[FloatArray] = []
    held_ids: list[str] = []
    for row in range(shape[0]):
        for column in range(shape[1] - 1):
            values = np.zeros(8, dtype=np.float64)
            values[row * 4 + column : row * 4 + column + 2] = 0.5
            rows.append(values)
            held_ids.append(f"horizontal-r{row}-c{column}:{column + 2}")
    for row in range(shape[0] - 1):
        for column in range(shape[1]):
            values = np.zeros(8, dtype=np.float64)
            values[row * 4 + column] = 0.5
            values[(row + 1) * 4 + column] = 0.5
            rows.append(values)
            held_ids.append(f"vertical-r{row}:{row + 2}-c{column}")
    heldout = np.stack(rows)
    held_permutation = np.random.Generator(np.random.PCG64(60001)).permutation(len(rows))
    return (
        training,
        tuple(train_ids),
        heldout[held_permutation],
        tuple(held_ids[int(index)] for index in held_permutation),
    )


def _gaussian_operator(
    *,
    rows: int,
    seed: int,
    shape: tuple[int, int] = (8, 8),
) -> tuple[FloatArray, tuple[str, ...]]:
    generator = np.random.Generator(np.random.PCG64(seed))
    row_coordinate = np.arange(shape[0], dtype=np.float64) + 0.5
    column_coordinate = np.arange(shape[1], dtype=np.float64) + 0.5
    output = np.empty((rows, shape[0] * shape[1]), dtype=np.float64)
    identifiers: list[str] = []
    for index in range(rows):
        center_row = float(generator.uniform(0.0, float(shape[0])))
        center_column = float(generator.uniform(0.0, float(shape[1])))
        width = float(generator.uniform(0.55, 1.35))
        squared = (row_coordinate[:, None] - center_row) ** 2 + (
            column_coordinate[None, :] - center_column
        ) ** 2
        raw = np.exp(-squared / (2.0 * width * width)).ravel(order="C")
        total = math.fsum(float(item) for item in raw)
        normalized = raw / total
        if abs(float(normalized.sum()) - 1.0) > 8.0 * np.spacing(1.0):
            raise RuntimeError("Gaussian footprint normalization exceeded eight ULP")
        output[index] = normalized
        identifiers.append(
            f"gaussian-{seed}-{index:03d}-r{center_row.hex()}-c{center_column.hex()}-w{width.hex()}"
        )
    return output, tuple(identifiers)


def stage_operators(
    stage: Stage,
) -> tuple[FloatArray, tuple[str, ...], FloatArray, tuple[str, ...]]:
    """Return deterministic training and held-out operators with row IDs."""
    if stage == "s0":
        return _operator_s0()
    if stage == "s1":
        training, train_ids = _gaussian_operator(rows=96, seed=71000)
        heldout, held_ids = _gaussian_operator(rows=48, seed=71001)
        return training, train_ids, heldout, held_ids
    raise ValueError("unsupported stage")


def _apply_edge(tiling: LeafTiling, index: int) -> LeafTiling:
    path = edge_flip_paths(tiling)[index]
    intermediate = tiling.merge(path.merge)
    return intermediate.split(SplitChoice(path.merge.parent, path.target_axis))


def _apply_relocation(tiling: LeafTiling, index: int) -> LeafTiling:
    path = relocation_paths(tiling)[index]
    return tiling.merge(path.merge).split(path.split)


def scenario_topology(
    p0: LeafTiling,
    scenario: Scenario,
) -> tuple[LeafTiling, dict[str, object]]:
    """Resolve a planted topology and exact deterministic one-move witness."""
    source_bounds = topology_bounds(p0)
    if scenario == "aligned":
        return p0, {
            "move": "identity",
            "path_index": None,
            "source_bounds": source_bounds,
            "destination_bounds": source_bounds,
            "source_sha256": topology_sha256(p0),
            "destination_sha256": topology_sha256(p0),
            "certified_distance": 0,
        }
    edge_destination: LeafTiling | None = None
    edge_index: int | None = None
    for index, _ in enumerate(edge_flip_paths(p0)):
        candidate = _apply_edge(p0, index)
        if candidate != p0:
            edge_destination = candidate
            edge_index = index
            break
    if edge_destination is None or edge_index is None:
        raise RuntimeError("P0 has no nontrivial edge-flip witness")
    if scenario == "edge-one":
        destination = edge_destination
        move = "edge_flip"
        path_index = edge_index
    elif scenario == "relocation-one":
        destination = None
        path_index = -1
        for index, _ in enumerate(relocation_paths(p0)):
            candidate = _apply_relocation(p0, index)
            if candidate != p0 and candidate != edge_destination:
                destination = candidate
                path_index = index
                break
        if destination is None:
            raise RuntimeError("P0 has no distinct nontrivial relocation witness")
        move = "resolution_relocation"
    else:
        raise ValueError("unsupported scenario")
    return destination, {
        "move": move,
        "path_index": path_index,
        "source_bounds": source_bounds,
        "destination_bounds": topology_bounds(destination),
        "source_sha256": topology_sha256(p0),
        "destination_sha256": topology_sha256(destination),
        "certified_distance": 1,
    }


def truth_on_tiling(
    tiling: LeafTiling,
    nominal_weight: FloatArray,
) -> FloatArray:
    """Return the frozen distinct positive planted scaling field."""
    if tiling.k < 2:
        raise ValueError("the frozen distinct-scaling formula requires K at least two")
    raw = np.asarray(
        [math.exp(-0.7 + 1.4 * leaf_index / (tiling.k - 1)) for leaf_index in range(tiling.k)],
        dtype=np.float64,
    )
    leaf_weight = np.asarray(
        [
            math.fsum(
                float(item)
                for item in nominal_weight[
                    leaf.row_start : leaf.row_stop,
                    leaf.col_start : leaf.col_stop,
                ].ravel(order="C")
            )
            for leaf in tiling.leaves
        ],
        dtype=np.float64,
    )
    total_weight = math.fsum(float(item) for item in leaf_weight)
    raw_mean = (
        math.fsum(float(value) * float(weight) for value, weight in zip(raw, leaf_weight, strict=True))
        / total_weight
    )
    scaling = raw / raw_mean
    partial = math.fsum(
        float(value) * float(weight) for value, weight in zip(scaling[:-1], leaf_weight[:-1], strict=True)
    )
    final_value = (total_weight - partial) / float(leaf_weight[-1])
    lower = final_value
    upper = final_value
    found = False
    for _ in range(256):
        for candidate in (final_value, lower, upper):
            scaling[-1] = candidate
            if (
                math.fsum(
                    float(value) * float(weight) for value, weight in zip(scaling, leaf_weight, strict=True)
                )
                == total_weight
            ):
                found = True
                break
        if found:
            break
        lower = float(np.nextafter(lower, 0.0))
        upper = float(np.nextafter(upper, math.inf))
    if not found:
        raise RuntimeError("could not audit an exact weighted-mean-one truth field")
    truth = np.empty(tiling.shape, dtype=np.float64)
    for leaf, value in zip(tiling.leaves, scaling, strict=True):
        truth[leaf.row_start : leaf.row_stop, leaf.col_start : leaf.col_stop] = value
    native_mean = math.fsum(
        float(value) * float(weight)
        for value, weight in zip(
            truth.ravel(order="C"),
            nominal_weight.ravel(order="C"),
            strict=True,
        )
    ) / math.fsum(float(item) for item in nominal_weight.ravel(order="C"))
    if native_mean != 1.0:
        raise RuntimeError("truth field does not have nominal-weighted mean one")
    truth.setflags(write=False)
    return truth


def build_stage_definition(stage: Stage) -> dict[str, object]:
    """Resolve all noiseless stage inputs before any sampling."""
    if not isinstance(stage, str) or stage not in _STAGE_SETTINGS:
        raise ValueError("unsupported stage")
    settings = _STAGE_SETTINGS[stage]
    shape = cast(tuple[int, int], settings["shape"])
    k = int(cast(Any, settings["k"]))
    train, train_ids, heldout, heldout_ids = stage_operators(stage)
    nominal = np.full(shape, 1.0 / (shape[0] * shape[1]), dtype=np.float64)
    dummy = _build_problem(
        shape=shape,
        k=k,
        operator=train,
        observations=np.zeros(train.shape[0], dtype=np.float64),
        observation_sd=np.full(
            train.shape[0],
            float(cast(Any, settings["observation_sd"])),
        ),
        nominal_weight=nominal,
    )
    p0 = initialize_full_tiling_posterior_state(dummy, k=k).allocation.tiling
    scenarios: dict[str, object] = {}
    for scenario in ("aligned", "edge-one", "relocation-one"):
        pstar, witness = scenario_topology(p0, cast(Scenario, scenario))
        truth = truth_on_tiling(pstar, nominal)
        scenarios[scenario] = {
            "pstar_bounds": topology_bounds(pstar),
            "pstar_sha256": topology_sha256(pstar),
            "truth": truth.tolist(),
            "truth_sha256": json_sha256(truth.tolist()),
            "witness": witness,
        }
    return {
        "stage": stage,
        "shape": shape,
        "k": k,
        "settings": settings,
        "nominal_weight": nominal.tolist(),
        "p0_bounds": topology_bounds(p0),
        "p0_sha256": topology_sha256(p0),
        "training_operator": train.tolist(),
        "training_operator_sha256": json_sha256(train.tolist()),
        "training_row_ids": train_ids,
        "heldout_operator": heldout.tolist(),
        "heldout_operator_sha256": json_sha256(heldout.tolist()),
        "heldout_row_ids": heldout_ids,
        "scenarios": scenarios,
    }


def validate_stage_definition(
    definition: Mapping[str, object],
) -> dict[str, object]:
    """Replay and require exact equality with the code-frozen stage definition."""
    if frozenset(definition) != _DEFINITION_KEYS:
        raise ValueError("definition payload keys are incompatible")
    stage = definition.get("stage")
    if not isinstance(stage, str) or stage not in _STAGE_SETTINGS:
        raise ValueError("definition stage is unsupported")
    expected = build_stage_definition(cast(Stage, stage))
    if canonical_json(definition) != canonical_json(expected):
        raise ValueError("definition does not exactly replay the code-frozen stage")
    return expected


def frozen_stage_budgets(stage: Stage) -> tuple[int, int, int]:
    """Return conditioning cycles, production cycles, and pair slots."""
    if not isinstance(stage, str) or stage not in _STAGE_SETTINGS:
        raise ValueError("unsupported stage")
    settings = _STAGE_SETTINGS[stage]
    return (
        _exact_int(
            settings["conditioning_cycles"],
            name="conditioning_cycles",
            minimum=1,
        ),
        _exact_int(
            settings["production_cycles"],
            name="production_cycles",
            minimum=1,
        ),
        5,
    )


def frozen_oracle_settings(
    stage: Stage,
    replicate: int,
) -> tuple[int, int, int, int, int]:
    """Return oracle conditioning/production budgets, seeds, and pair slots."""
    conditioning_cycles, production_cycles, pair_slots = frozen_stage_budgets(stage)
    replicate_index = _exact_int(replicate, name="replicate")
    settings = _STAGE_SETTINGS[stage]
    conditioning_catalogue = cast(Any, settings["oracle_conditioning_seeds"])
    sampler_catalogue = cast(Any, settings["oracle_seeds"])
    if replicate_index >= len(conditioning_catalogue):
        raise ValueError("replicate lies outside the frozen oracle seed catalogue")
    return (
        conditioning_cycles,
        production_cycles,
        _exact_int(
            conditioning_catalogue[replicate_index],
            name="oracle_conditioning_seed",
        ),
        _exact_int(sampler_catalogue[replicate_index], name="oracle_seed"),
        pair_slots,
    )


def frozen_local_reference_seeds(stage: Stage) -> tuple[int, int, int, int]:
    """Return the four predeclared fixed-basis local-reference seeds."""
    if not isinstance(stage, str) or stage not in _STAGE_SETTINGS:
        raise ValueError("unsupported stage")
    first = 64_201 if stage == "s0" else 74_201
    return cast(tuple[int, int, int, int], tuple(range(first, first + 4)))


def _replay_observations(
    definition: Mapping[str, object],
    *,
    scenario: Scenario,
    replicate: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    settings = cast(Mapping[str, object], definition["settings"])
    scenario_map = cast(
        Mapping[str, object],
        cast(Mapping[str, object], definition["scenarios"])[scenario],
    )
    operator = np.asarray(definition["training_operator"], dtype=np.float64)
    heldout = np.asarray(definition["heldout_operator"], dtype=np.float64)
    truth_vector = np.asarray(scenario_map["truth"], dtype=np.float64).ravel(order="C")
    noiseless_training = operator @ truth_vector
    noiseless_heldout = heldout @ truth_vector
    observation_sd = float(cast(Any, settings["observation_sd"]))
    noise_seed = _exact_int(
        cast(Any, settings["noise_seeds"])[replicate],
        name="noise_seed",
    )
    generator = np.random.Generator(np.random.PCG64(noise_seed))
    training = noiseless_training + generator.normal(
        0.0,
        observation_sd,
        size=noiseless_training.size,
    )
    heldout_observations = noiseless_heldout + generator.normal(
        0.0,
        observation_sd,
        size=noiseless_heldout.size,
    )
    return noiseless_training, training, noiseless_heldout, heldout_observations


def materialize_replicate(
    definition: Mapping[str, object],
    *,
    scenario: Scenario,
    replicate: int,
) -> tuple[SyntheticTrainingArtifact, SyntheticEvaluationArtifact]:
    """Create one paired training/evaluation payload from a frozen definition."""
    definition = validate_stage_definition(definition)
    stage = cast(Stage, definition["stage"])
    settings = cast(Mapping[str, object], definition["settings"])
    replicate = _exact_int(replicate, name="replicate")
    if replicate >= len(cast(Any, settings["noise_seeds"])):
        raise ValueError("replicate index lies outside the frozen seed catalogue")
    if not isinstance(scenario, str) or scenario not in ("aligned", "edge-one", "relocation-one"):
        raise ValueError("unsupported scenario")
    scenario_map = cast(
        Mapping[str, object],
        cast(Mapping[str, object], definition["scenarios"])[scenario],
    )
    definition_digest = json_sha256(definition)
    cell_id = cell_commitment(definition_digest, scenario, replicate)
    shape = _shape(definition["shape"])
    k = _exact_int(definition["k"], name="k", minimum=1)
    operator = np.asarray(definition["training_operator"], dtype=np.float64)
    heldout = np.asarray(definition["heldout_operator"], dtype=np.float64)
    truth = np.asarray(scenario_map["truth"], dtype=np.float64)
    _, noisy_train, noiseless_heldout, noisy_heldout = _replay_observations(
        definition,
        scenario=scenario,
        replicate=replicate,
    )
    sd_value = float(cast(Any, settings["observation_sd"]))
    generation_commitment = observation_generation_commitment(
        definition_digest,
        cell_id,
        noisy_train,
        noisy_heldout,
    )
    nominal = np.asarray(definition["nominal_weight"], dtype=np.float64)
    training = SyntheticTrainingArtifact(
        stage=stage,
        replicate=replicate,
        definition_sha256=definition_digest,
        cell_id=cell_id,
        generation_commitment=generation_commitment,
        shape=shape,
        k=k,
        operator=operator,
        observations=noisy_train,
        observation_sd=np.full(noisy_train.size, sd_value),
        nominal_weight=nominal,
        row_ids=tuple(str(item) for item in cast(Any, definition["training_row_ids"])),
        p0_bounds=cast(Any, definition["p0_bounds"]),
        conditioning_seed=_exact_int(
            cast(Any, settings["conditioning_seeds"])[replicate],
            name="conditioning_seed",
        ),
        fixed_seed=_exact_int(
            cast(Any, settings["fixed_seeds"])[replicate],
            name="fixed_seed",
        ),
        mobile_seed=_exact_int(
            cast(Any, settings["mobile_seeds"])[replicate],
            name="mobile_seed",
        ),
    )
    evaluation = SyntheticEvaluationArtifact(
        stage=stage,
        scenario=scenario,
        replicate=replicate,
        definition_sha256=definition_digest,
        cell_id=cell_id,
        generation_commitment=generation_commitment,
        training_observations_sha256=json_sha256(noisy_train.tolist()),
        shape=shape,
        k=k,
        truth=truth,
        heldout_operator=heldout,
        heldout_noiseless=noiseless_heldout,
        heldout_observations=noisy_heldout,
        heldout_sd=np.full(noisy_heldout.size, sd_value),
        heldout_row_ids=tuple(str(item) for item in cast(Any, definition["heldout_row_ids"])),
        pstar_bounds=cast(Any, scenario_map["pstar_bounds"]),
        witness=cast(Mapping[str, object], scenario_map["witness"]),
    )
    if set(training.row_ids) & set(evaluation.heldout_row_ids):
        raise RuntimeError("training and held-out row identities overlap")
    return training, evaluation


def reconstruct_native_fields(
    rectangle_bounds: IntArray,
    leaf_masses: FloatArray,
    nominal_weight: FloatArray,
) -> FloatArray:
    """Reconstruct common-grid scaling fields for every retained state."""
    bounds = np.asarray(rectangle_bounds, dtype=np.int64)
    masses = np.asarray(leaf_masses, dtype=np.float64)
    weight = np.asarray(nominal_weight, dtype=np.float64)
    if bounds.ndim != 3 or bounds.shape[2] != 4:
        raise ValueError("rectangle_bounds must have shape (draw, K, 4)")
    if masses.shape != bounds.shape[:2]:
        raise ValueError("leaf_masses must align with rectangle bounds")
    output = np.zeros((bounds.shape[0], *weight.shape), dtype=np.float64)
    normalized_weight = weight / float(weight.sum())
    for draw in range(bounds.shape[0]):
        for region in range(bounds.shape[1]):
            row_start, row_stop, column_start, column_stop = bounds[draw, region]
            leaf_weight = float(normalized_weight[row_start:row_stop, column_start:column_stop].sum())
            output[draw, row_start:row_stop, column_start:column_stop] = masses[draw, region] / leaf_weight
    if not np.all(np.isfinite(output)):
        raise ValueError("reconstructed fields are non-finite")
    return output


def common_native_totals(field: FloatArray, nominal_weight: FloatArray) -> FloatArray:
    """Return nine domain-normalized nominal-weighted regional contributions."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3:
        raise ValueError("field must have shape (draw, row, column) or (row, column)")
    rows, columns = nominal_weight.shape
    row_midpoint = rows // 2
    column_midpoint = columns // 2
    masks = (
        (0, rows, 0, columns),
        (0, row_midpoint, 0, columns),
        (row_midpoint, rows, 0, columns),
        (0, rows, 0, column_midpoint),
        (0, rows, column_midpoint, columns),
        (0, row_midpoint, 0, column_midpoint),
        (0, row_midpoint, column_midpoint, columns),
        (row_midpoint, rows, 0, column_midpoint),
        (row_midpoint, rows, column_midpoint, columns),
    )
    result = np.empty((values.shape[0], len(masks)), dtype=np.float64)
    domain_weight = math.fsum(float(item) for item in nominal_weight.ravel(order="C"))
    for index, (r0, r1, c0, c1) in enumerate(masks):
        weights = nominal_weight[r0:r1, c0:c1]
        result[:, index] = (
            np.sum(
                values[:, r0:r1, c0:c1] * weights[np.newaxis, :, :],
                axis=(1, 2),
            )
            / domain_weight
        )
    return result


__all__ = [
    "DEFINITION_SCHEMA",
    "EVALUATION_SCHEMA",
    "TRAINING_SCHEMA",
    "SyntheticEvaluationArtifact",
    "SyntheticTrainingArtifact",
    "build_stage_definition",
    "canonical_json",
    "cell_commitment",
    "common_native_totals",
    "file_sha256",
    "frozen_stage_budgets",
    "frozen_local_reference_seeds",
    "frozen_oracle_settings",
    "json_sha256",
    "load_evaluation_artifact",
    "load_training_artifact",
    "materialize_replicate",
    "observation_generation_commitment",
    "prepare_fixed_basis_reference",
    "problem_from_training",
    "read_envelope",
    "reconstruct_native_fields",
    "scenario_topology",
    "stage_operators",
    "state_on_tiling",
    "tiling_from_bounds",
    "topology_bounds",
    "topology_sha256",
    "truth_on_tiling",
    "validate_artifact_pair",
    "validate_local_reference_trace",
    "validate_stage_definition",
    "write_envelope",
]
