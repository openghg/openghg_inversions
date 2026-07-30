"""Coarse-to-fine SMC for one fixed Gamma--Dirichlet root allocation.

This opt-in experimental module progressively reveals the independent Beta
coordinates of a binary factorization of one native Dirichlet allocation.  A
Gaussian moment closure supplies intermediate SMC potentials, but the
terminal potential is always the ordinary normalized native Gaussian
likelihood.  Compatible trees are therefore computational charts for the
same continuous target rather than different scientific priors.

The implementation is deliberately small and auditable:

* tree leaves carry stable scientific cell identities;
* schedules are deterministic, parent first, and independently hashed;
* conditional observation means and unresolved covariances update locally;
* prior-proposal SMC supports no resampling, ESS-triggered multinomial
  resampling, and resampling at every nonterminal refinement;
* direct IID allocation averaging can consume exactly the same split paths;
* checkpoints retain the complete scientific state and both PCG64 streams;
  and
* every checkpoint is content hashed and fails closed against tree, schedule,
  input, configuration, path, or source-provenance mismatches.

The Gaussian closure is not exported as an exact marginal likelihood.  It is
used only inside :func:`run_resolution_smc` to guide finite particles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import resource
import time
from typing import Literal, Mapping, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
NestedCellChart: TypeAlias = int | tuple["NestedCellChart", "NestedCellChart"]
ResamplingPolicy: TypeAlias = Literal["never", "ess", "always"]

__all__ = [
    "DirectIIDResult",
    "GaussianGuideSpecification",
    "ParticleFrontier",
    "ResolutionBatch",
    "ResolutionSchedule",
    "ResolutionSMCCheckpoint",
    "ResolutionSMCConfig",
    "ResolutionSMCResult",
    "ResolutionTree",
    "ResolutionTreeNode",
    "SMCLevelDiagnostic",
    "breadth_first_schedule",
    "direct_iid_likelihood_average",
    "draw_prior_allocation_paths",
    "parent_first_priority_schedule",
    "run_resolution_smc",
]

_SCHEMA = "aggregation-error-resolution-smc-v1"
_CHECKPOINT_SCHEMA = "aggregation-error-resolution-smc-checkpoint-v1"
_LOG_TWO_PI = math.log(2.0 * math.pi)


def _canonical_json(value: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: object) -> str:
    """Hash one strict canonical JSON value."""
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _array_sha256(values: NDArray[np.generic]) -> str:
    """Return a dtype-, shape-, and value-sensitive array digest."""
    array = np.ascontiguousarray(values)
    header = _canonical_json(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    )
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_float(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite owned read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _readonly_int(values: ArrayLike, *, name: str) -> IntArray:
    """Return an owned read-only int64 array without silent float coercion."""
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be an integer array.")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: int, *, name: str) -> int:
    """Return a strictly positive built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    """Return a non-negative built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive_float(value: float, *, name: str) -> float:
    """Return a finite strictly positive float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on Linux and macOS."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if value > 10**10 else value * 1024


@dataclass(frozen=True, slots=True)
class ResolutionTreeNode:
    """One immutable node in a compatible Gamma--Beta chart."""

    node_id: int
    parent_id: int | None
    left_id: int | None
    right_id: int | None
    depth: int
    cell_ids: tuple[int, ...]
    cell_indices: tuple[int, ...]
    alpha_total: float

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a native cell."""
        return self.left_id is None


@dataclass(frozen=True, slots=True, eq=False)
class ResolutionTree:
    """A fixed binary chart of one native Dirichlet allocation."""

    cell_ids: IntArray
    cell_alphas: FloatArray
    nodes: tuple[ResolutionTreeNode, ...]
    root_id: int
    identity: str
    _cell_index_by_id: Mapping[int, int] = field(repr=False)

    @classmethod
    def from_nested_chart(
        cls,
        cell_ids: ArrayLike,
        cell_alphas: ArrayLike,
        chart: NestedCellChart,
    ) -> ResolutionTree:
        """Build a deterministic tree from a nested stable-cell-ID chart."""
        ids = _readonly_int(cell_ids, name="cell_ids")
        alphas = _readonly_float(cell_alphas, name="cell_alphas")
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("cell_ids must be a non-empty one-dimensional array.")
        if alphas.shape != ids.shape or np.any(alphas <= 0.0):
            raise ValueError("cell_alphas must be positive with one entry per cell ID.")
        if np.unique(ids).size != ids.size:
            raise ValueError("cell_ids must be unique.")
        index_by_id = {int(cell_id): index for index, cell_id in enumerate(ids)}
        records: list[ResolutionTreeNode | None] = []
        chart_leaves: list[int] = []

        def build(value: NestedCellChart, parent_id: int | None, depth: int) -> int:
            node_id = len(records)
            records.append(None)
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError("each internal chart entry must have exactly two children.")
                left_id = build(value[0], node_id, depth + 1)
                right_id = build(value[1], node_id, depth + 1)
                left = records[left_id]
                right = records[right_id]
                assert left is not None and right is not None
                descendant_ids = left.cell_ids + right.cell_ids
                descendant_indices = left.cell_indices + right.cell_indices
            else:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError("chart leaves must be integer scientific cell IDs.")
                cell_id = int(value)
                if cell_id not in index_by_id:
                    raise ValueError(f"chart contains unknown scientific cell ID {cell_id}.")
                chart_leaves.append(cell_id)
                left_id = None
                right_id = None
                descendant_ids = (cell_id,)
                descendant_indices = (index_by_id[cell_id],)
            alpha_total = math.fsum(float(alphas[index]) for index in descendant_indices)
            records[node_id] = ResolutionTreeNode(
                node_id=node_id,
                parent_id=parent_id,
                left_id=left_id,
                right_id=right_id,
                depth=depth,
                cell_ids=descendant_ids,
                cell_indices=descendant_indices,
                alpha_total=alpha_total,
            )
            return node_id

        root_id = build(chart, None, 0)
        if len(chart_leaves) != ids.size or set(chart_leaves) != set(int(value) for value in ids):
            raise ValueError("chart must contain every scientific cell ID exactly once.")
        if len(set(chart_leaves)) != len(chart_leaves):
            raise ValueError("chart must not repeat scientific cell IDs.")
        nodes = tuple(cast(ResolutionTreeNode, record) for record in records)
        payload = {
            "schema": _SCHEMA,
            "cell_ids": ids.tolist(),
            "cell_alphas": alphas.tolist(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "left_id": node.left_id,
                    "right_id": node.right_id,
                    "depth": node.depth,
                    "cell_ids": list(node.cell_ids),
                    "alpha_total": node.alpha_total,
                }
                for node in nodes
            ],
        }
        return cls(
            cell_ids=ids,
            cell_alphas=alphas,
            nodes=nodes,
            root_id=root_id,
            identity=_sha256_json(payload),
            _cell_index_by_id=index_by_id,
        )

    @property
    def internal_node_ids(self) -> tuple[int, ...]:
        """Return internal node IDs in deterministic construction order."""
        return tuple(node.node_id for node in self.nodes if not node.is_leaf)

    @property
    def leaf_node_ids(self) -> tuple[int, ...]:
        """Return leaf node IDs in scientific cell order."""
        by_cell_id = {node.cell_ids[0]: node.node_id for node in self.nodes if node.is_leaf}
        return tuple(by_cell_id[int(cell_id)] for cell_id in self.cell_ids)

    def node(self, node_id: int) -> ResolutionTreeNode:
        """Return one node after validating its ID."""
        normalized = _nonnegative_integer(node_id, name="node_id")
        if normalized >= len(self.nodes):
            raise ValueError("node_id lies outside the tree.")
        return self.nodes[normalized]

    def is_compatible(self, other: ResolutionTree) -> bool:
        """Whether two trees parameterize the same scientific Dirichlet law."""
        if not isinstance(other, ResolutionTree):
            return False
        self_map = {
            int(cell_id): float(alpha) for cell_id, alpha in zip(self.cell_ids, self.cell_alphas, strict=True)
        }
        other_map = {
            int(cell_id): float(alpha)
            for cell_id, alpha in zip(other.cell_ids, other.cell_alphas, strict=True)
        }
        return self_map == other_map


@dataclass(frozen=True, slots=True)
class ResolutionBatch:
    """One deterministic batch of simultaneously eligible internal nodes."""

    node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ValueError("a resolution batch cannot be empty.")
        if any(isinstance(node_id, bool) or not isinstance(node_id, Integral) for node_id in self.node_ids):
            raise TypeError("resolution-batch node IDs must be integers.")
        normalized = tuple(int(node_id) for node_id in self.node_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("a resolution batch cannot repeat a node.")
        object.__setattr__(self, "node_ids", normalized)


@dataclass(frozen=True, slots=True, eq=False)
class ResolutionSchedule:
    """A complete deterministic parent-first reveal schedule."""

    batches: tuple[ResolutionBatch, ...]
    identity: str
    name: str

    @classmethod
    def build(
        cls,
        tree: ResolutionTree,
        batches: Sequence[Sequence[int]],
        *,
        name: str,
    ) -> ResolutionSchedule:
        """Validate and freeze a complete parent-first schedule."""
        if not isinstance(tree, ResolutionTree):
            raise TypeError("tree must be a ResolutionTree.")
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("schedule name cannot be empty.")
        built = tuple(ResolutionBatch(tuple(batch)) for batch in batches)
        if not built:
            raise ValueError("a resolution schedule cannot be empty.")
        frontier = {tree.root_id}
        revealed: set[int] = set()
        for batch in built:
            batch_set = set(batch.node_ids)
            if not batch_set.issubset(frontier):
                raise ValueError("every scheduled node must be on the current frontier.")
            if any(tree.node(node_id).is_leaf for node_id in batch.node_ids):
                raise ValueError("native leaves cannot be scheduled for refinement.")
            if batch_set & revealed:
                raise ValueError("an internal node cannot be revealed twice.")
            for node_id in batch.node_ids:
                node = tree.node(node_id)
                assert node.left_id is not None and node.right_id is not None
                frontier.remove(node_id)
                frontier.add(node.left_id)
                frontier.add(node.right_id)
                revealed.add(node_id)
        if revealed != set(tree.internal_node_ids):
            raise ValueError("the schedule must reveal every internal node exactly once.")
        if frontier != set(tree.leaf_node_ids):
            raise ValueError("the terminal schedule frontier must contain every native leaf.")
        payload = {
            "schema": _SCHEMA,
            "tree_identity": tree.identity,
            "name": normalized_name,
            "batches": [list(batch.node_ids) for batch in built],
        }
        return cls(
            batches=built,
            identity=_sha256_json(payload),
            name=normalized_name,
        )

    @property
    def coordinate_node_ids(self) -> tuple[int, ...]:
        """Return the complete split-coordinate order."""
        return tuple(node_id for batch in self.batches for node_id in batch.node_ids)

    def frontier_after(self, tree: ResolutionTree, completed_levels: int) -> tuple[int, ...]:
        """Return the deterministic frontier after a number of batches."""
        levels = _nonnegative_integer(completed_levels, name="completed_levels")
        if levels > len(self.batches):
            raise ValueError("completed_levels exceeds the schedule length.")
        frontier = {tree.root_id}
        for batch in self.batches[:levels]:
            for node_id in batch.node_ids:
                node = tree.node(node_id)
                assert node.left_id is not None and node.right_id is not None
                frontier.remove(node_id)
                frontier.add(node.left_id)
                frontier.add(node.right_id)
        return tuple(sorted(frontier))


def breadth_first_schedule(tree: ResolutionTree, *, name: str = "breadth-first") -> ResolutionSchedule:
    """Return deterministic complete-depth batches."""
    by_depth: dict[int, list[int]] = {}
    for node in tree.nodes:
        if not node.is_leaf:
            by_depth.setdefault(node.depth, []).append(node.node_id)
    batches = [tuple(sorted(by_depth[depth])) for depth in sorted(by_depth)]
    return ResolutionSchedule.build(tree, batches, name=name)


@dataclass(frozen=True, slots=True, eq=False)
class GaussianGuideSpecification:
    """Observation model used by the intermediate closure and exact terminal."""

    observation: FloatArray
    mean_offset: FloatArray
    design: FloatArray
    noise_sd: FloatArray
    identity: str

    @classmethod
    def build(
        cls,
        observation: ArrayLike,
        design: ArrayLike,
        noise_sd: ArrayLike,
        *,
        mean_offset: ArrayLike | None = None,
    ) -> GaussianGuideSpecification:
        """Validate and freeze one diagonal-noise observation model."""
        observed = _readonly_float(observation, name="observation")
        matrix = _readonly_float(design, name="design")
        if observed.ndim != 1 or observed.size == 0:
            raise ValueError("observation must be a non-empty vector.")
        if matrix.ndim != 2 or matrix.shape[0] != observed.size or matrix.shape[1] == 0:
            raise ValueError("design must have one row per observation and at least one cell column.")
        raw_noise = np.asarray(noise_sd, dtype=np.float64)
        if raw_noise.ndim == 0:
            raw_noise = np.full(observed.size, float(raw_noise), dtype=np.float64)
        scale = _readonly_float(raw_noise, name="noise_sd")
        if scale.shape != observed.shape or np.any(scale <= 0.0):
            raise ValueError("noise_sd must be positive with one entry per observation.")
        if mean_offset is None:
            offset = _readonly_float(np.zeros(observed.size), name="mean_offset")
        else:
            offset = _readonly_float(mean_offset, name="mean_offset")
            if offset.shape != observed.shape:
                raise ValueError("mean_offset must have one entry per observation.")
        payload = {
            "schema": _SCHEMA,
            "observation_sha256": _array_sha256(observed),
            "mean_offset_sha256": _array_sha256(offset),
            "design_sha256": _array_sha256(matrix),
            "noise_sd_sha256": _array_sha256(scale),
        }
        return cls(
            observation=observed,
            mean_offset=offset,
            design=matrix,
            noise_sd=scale,
            identity=_sha256_json(payload),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _TreeGuideMoments:
    """Per-node exact conditional first and second observation moments."""

    means: FloatArray
    covariances: FloatArray


def _tree_guide_moments(
    tree: ResolutionTree,
    guide: GaussianGuideSpecification,
) -> _TreeGuideMoments:
    if guide.design.shape[1] != tree.cell_ids.size:
        raise ValueError("guide design must have one column per tree cell.")
    observation_count = guide.observation.size
    means = np.empty((len(tree.nodes), observation_count), dtype=np.float64)
    covariances = np.empty(
        (len(tree.nodes), observation_count, observation_count),
        dtype=np.float64,
    )
    for node in reversed(tree.nodes):
        indices = np.asarray(node.cell_indices, dtype=np.int64)
        alphas = tree.cell_alphas[indices]
        proportions = alphas / node.alpha_total
        columns = guide.design[:, indices]
        mean = columns @ proportions
        means[node.node_id] = mean
        if node.is_leaf:
            covariances[node.node_id] = 0.0
        else:
            centered = columns - mean[:, np.newaxis]
            covariance = ((centered * proportions[np.newaxis, :]) @ centered.T) / (node.alpha_total + 1.0)
            covariances[node.node_id] = 0.5 * (covariance + covariance.T)
    means.setflags(write=False)
    covariances.setflags(write=False)
    return _TreeGuideMoments(means=means, covariances=covariances)


def parent_first_priority_schedule(
    tree: ResolutionTree,
    guide: GaussianGuideSpecification,
    *,
    root_total: float,
    favorable: bool = True,
    batch_size: int = 1,
    name: str | None = None,
) -> ResolutionSchedule:
    """Order eligible nodes by frozen prior/operator observation energy."""
    total = _positive_float(root_total, name="root_total")
    width = _positive_integer(batch_size, name="batch_size")
    moments = _tree_guide_moments(tree, guide)
    root_alpha = tree.node(tree.root_id).alpha_total
    inverse_noise = 1.0 / guide.noise_sd
    scores: dict[int, float] = {}
    for node in tree.nodes:
        if node.is_leaf:
            continue
        assert node.left_id is not None and node.right_id is not None
        left = tree.node(node.left_id)
        right = tree.node(node.right_id)
        expected_squared_mass = (
            total**2 * node.alpha_total * (node.alpha_total + 1.0) / (root_alpha * (root_alpha + 1.0))
        )
        beta_variance = (
            left.alpha_total * right.alpha_total / (node.alpha_total**2 * (node.alpha_total + 1.0))
        )
        contrast = (moments.means[node.left_id] - moments.means[node.right_id]) * inverse_noise
        scores[node.node_id] = expected_squared_mass * beta_variance * float(contrast @ contrast)

    frontier = {tree.root_id}
    remaining = set(tree.internal_node_ids)
    batches: list[tuple[int, ...]] = []
    while remaining:
        eligible = sorted(frontier & remaining)
        if not eligible:
            raise RuntimeError("priority schedule has no eligible internal node.")
        ordered = sorted(
            eligible,
            key=lambda node_id: (
                -scores[node_id] if favorable else scores[node_id],
                node_id,
            ),
        )
        selected = tuple(ordered[:width])
        batches.append(selected)
        for node_id in selected:
            node = tree.node(node_id)
            assert node.left_id is not None and node.right_id is not None
            frontier.remove(node_id)
            frontier.add(node.left_id)
            frontier.add(node.right_id)
            remaining.remove(node_id)
    default_name = "observation-energy" if favorable else "unfavorable-observation-energy"
    return ResolutionSchedule.build(tree, batches, name=default_name if name is None else name)


@dataclass(frozen=True, slots=True)
class ResolutionSMCConfig:
    """Finite-particle and resampling configuration."""

    particle_count: int
    seed: int
    resampling_policy: ResamplingPolicy = "ess"
    ess_fraction: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "particle_count",
            _positive_integer(self.particle_count, name="particle_count"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer.")
        normalized_seed = int(self.seed)
        if normalized_seed < 0 or normalized_seed >= 2**128:
            raise ValueError("seed must lie in [0, 2**128).")
        object.__setattr__(self, "seed", normalized_seed)
        if self.resampling_policy not in ("never", "ess", "always"):
            raise ValueError("resampling_policy must be 'never', 'ess', or 'always'.")
        threshold = float(self.ess_fraction)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("ess_fraction must lie in [0, 1].")
        object.__setattr__(self, "ess_fraction", threshold)
        if self.resampling_policy == "ess" and threshold == 0.0:
            raise ValueError("ESS-triggered resampling requires a positive ess_fraction.")

    @property
    def identity(self) -> str:
        """Return a canonical identity including both RNG and particle policy."""
        return _sha256_json(
            {
                "schema": _SCHEMA,
                "particle_count": self.particle_count,
                "seed": self.seed,
                "resampling_policy": self.resampling_policy,
                "ess_fraction": self.ess_fraction,
                "bit_generator": "PCG64",
                "resampling_method": "multinomial",
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class ParticleFrontier:
    """Complete particle state at one deterministic resolution boundary."""

    node_masses: FloatArray
    observation_mean: FloatArray
    unresolved_covariance: FloatArray

    @property
    def storage_nbytes(self) -> int:
        """Bytes owned by the three scientific state arrays."""
        return int(self.node_masses.nbytes + self.observation_mean.nbytes + self.unresolved_covariance.nbytes)


@dataclass(frozen=True, slots=True)
class SMCLevelDiagnostic:
    """Per-level estimator, ancestry, cost, timing, and memory diagnostics."""

    level: int
    node_ids: tuple[int, ...]
    terminal: bool
    resampled: bool
    ess: float
    ess_fraction: float
    incremental_weight_cv: float
    max_normalized_weight: float
    shannon_perplexity: float
    unique_ancestor_count: int
    incremental_log_weight_min: float
    incremental_log_weight_mean: float
    incremental_log_weight_max: float
    incremental_log_weight_sd: float
    linear_likelihood_correction_mean: float
    linear_likelihood_correction_variance: float
    log_normalizer_increment: float
    beta_draw_count: int
    beta_endpoint_repair_count: int
    forward_update_count: int
    likelihood_evaluation_count: int
    elapsed_seconds: float
    state_bytes: int
    peak_rss_bytes: int
    max_mass_conservation_error: float
    max_mean_update_error: float
    max_covariance_update_error: float
    max_terminal_prediction_error: float
    max_terminal_unresolved_covariance: float


@dataclass(frozen=True, slots=True, eq=False)
class DirectIIDResult:
    """Direct complete-path IID likelihood average."""

    likelihood: float
    log_likelihood: float
    terminal_log_likelihoods: FloatArray
    leaf_masses: FloatArray
    allocation_paths_sha256: str
    elapsed_seconds: float
    peak_rss_bytes: int
    beta_draw_count: int
    beta_endpoint_repair_count: int
    likelihood_evaluation_count: int


@dataclass(frozen=True, slots=True, eq=False)
class ResolutionSMCResult:
    """Completed non-negative normalizing-constant estimate."""

    likelihood: float
    log_likelihood: float
    accumulator_log_likelihood: float
    no_resampling_accumulator_error: float | None
    terminal_log_likelihoods: FloatArray
    normalized_log_weights: FloatArray
    terminal_leaf_masses: FloatArray
    lineage: IntArray
    ancestry: IntArray
    diagnostics: tuple[SMCLevelDiagnostic, ...]
    tree_identity: str
    schedule_identity: str
    guide_identity: str
    config_identity: str
    allocation_paths_sha256: str | None
    source_identity: str
    scientific_fingerprint: str
    elapsed_seconds: float
    peak_rss_bytes: int


def _batched_gaussian_log_likelihood(
    guide: GaussianGuideSpecification,
    means: FloatArray,
    unresolved_covariances: FloatArray,
) -> FloatArray:
    """Evaluate normalized Gaussian guides for a particle batch."""
    if means.ndim != 2 or means.shape[1] != guide.observation.size:
        raise ValueError("means must have one row per particle and one column per observation.")
    expected_covariance_shape = (
        means.shape[0],
        guide.observation.size,
        guide.observation.size,
    )
    if unresolved_covariances.shape != expected_covariance_shape:
        raise ValueError("unresolved_covariances has an incompatible shape.")
    covariance = np.array(unresolved_covariances, copy=True)
    diagonal = np.arange(guide.observation.size)
    covariance[:, diagonal, diagonal] += np.square(guide.noise_sd)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    sign, log_determinant = np.linalg.slogdet(covariance)
    if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
        raise ValueError("Gaussian guide covariance is not positive definite.")
    residual = guide.observation[np.newaxis, :] - (guide.mean_offset[np.newaxis, :] + means)
    try:
        solved = np.linalg.solve(covariance, residual[..., np.newaxis])[..., 0]
    except np.linalg.LinAlgError as error:
        raise ValueError("Gaussian guide covariance solve failed.") from error
    quadratic = np.einsum("ni,ni->n", residual, solved)
    result = -0.5 * (guide.observation.size * _LOG_TWO_PI + log_determinant + quadratic)
    if not np.all(np.isfinite(result)):
        raise ValueError("Gaussian guide produced non-finite log likelihoods.")
    return np.asarray(result, dtype=np.float64)


def _exact_terminal_log_likelihood(
    guide: GaussianGuideSpecification,
    leaf_masses: FloatArray,
) -> FloatArray:
    """Evaluate the ordinary normalized native Gaussian likelihood."""
    if leaf_masses.ndim != 2 or leaf_masses.shape[1] != guide.design.shape[1]:
        raise ValueError("leaf_masses must have one column per native design cell.")
    mean = leaf_masses @ guide.design.T + guide.mean_offset[np.newaxis, :]
    residual = (guide.observation[np.newaxis, :] - mean) / guide.noise_sd[np.newaxis, :]
    result = -0.5 * (
        guide.observation.size * _LOG_TWO_PI
        + 2.0 * float(np.log(guide.noise_sd).sum())
        + np.einsum("ni,ni->n", residual, residual)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("terminal native Gaussian likelihood is non-finite.")
    return np.asarray(result, dtype=np.float64)


def _logmeanexp(values: FloatArray) -> float:
    """Return a stable log arithmetic mean."""
    reduced = cast(float, logsumexp(values))
    return float(reduced - math.log(values.size))


def _linear_from_log(log_value: float) -> float:
    """Return a non-negative linear representation, allowing underflow."""
    if not math.isfinite(log_value):
        raise ValueError("normalizing-constant log value must be finite.")
    value = float(math.exp(log_value))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("normalizing constant must be finite and non-negative.")
    return value


def _leaf_masses(tree: ResolutionTree, node_masses: FloatArray) -> FloatArray:
    """Return terminal masses in the tree's scientific cell order."""
    result = np.asarray(node_masses[:, tree.leaf_node_ids], dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("terminal native masses must be finite and strictly positive.")
    return result


def _recompute_frontier(
    active_node_ids: Sequence[int],
    node_masses: FloatArray,
    moments: _TreeGuideMoments,
) -> tuple[FloatArray, FloatArray]:
    """Recompute exact frontier moments independently of local updates."""
    particle_count = node_masses.shape[0]
    observation_count = moments.means.shape[1]
    mean = np.zeros((particle_count, observation_count), dtype=np.float64)
    covariance = np.zeros(
        (particle_count, observation_count, observation_count),
        dtype=np.float64,
    )
    for node_id in active_node_ids:
        mass = node_masses[:, node_id]
        mean += mass[:, np.newaxis] * moments.means[node_id][np.newaxis, :]
        covariance += (
            np.square(mass)[:, np.newaxis, np.newaxis] * moments.covariances[node_id][np.newaxis, :, :]
        )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    return mean, covariance


def _initial_frontier(
    tree: ResolutionTree,
    moments: _TreeGuideMoments,
    *,
    root_total: float,
    particle_count: int,
) -> ParticleFrontier:
    masses = np.zeros((particle_count, len(tree.nodes)), dtype=np.float64)
    masses[:, tree.root_id] = root_total
    mean = np.repeat(
        (root_total * moments.means[tree.root_id])[np.newaxis, :],
        particle_count,
        axis=0,
    )
    covariance = np.repeat(
        (root_total**2 * moments.covariances[tree.root_id])[np.newaxis, :, :],
        particle_count,
        axis=0,
    )
    return ParticleFrontier(
        node_masses=masses,
        observation_mean=mean,
        unresolved_covariance=covariance,
    )


def _path_identity(paths: FloatArray | None) -> str | None:
    return None if paths is None else _array_sha256(paths)


def _open_beta_draw(
    generator: np.random.Generator,
    first_shape: float,
    second_shape: float,
    *,
    size: int,
) -> tuple[FloatArray, int]:
    """Draw Beta values and repair finite-representation endpoints.

    A continuous Beta variate is almost surely in the open unit interval, but
    for shapes below one NumPy can round a legitimate tail draw to exactly
    zero or one.  The nearest interior float64 value is the least invasive
    represented repair and prevents a zero child mass.  The caller records
    every repair as a numerical diagnostic.
    """
    values = np.asarray(
        generator.beta(first_shape, second_shape, size=size),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("PCG64 produced a non-finite Beta variate.")
    endpoint = (values <= 0.0) | (values >= 1.0)
    repair_count = int(np.count_nonzero(endpoint))
    if repair_count:
        values = np.array(values, copy=True)
        values[values <= 0.0] = np.nextafter(0.0, 1.0)
        values[values >= 1.0] = np.nextafter(1.0, 0.0)
    return values, repair_count


def _split_parent_mass(
    parent_mass: FloatArray,
    fraction: FloatArray,
) -> tuple[FloatArray, FloatArray, int]:
    """Split represented parent masses into positive exact complements."""
    left_mass = parent_mass * fraction
    endpoint = (left_mass <= 0.0) | (left_mass >= parent_mass)
    repair_count = int(np.count_nonzero(endpoint))
    if repair_count:
        left_mass = np.array(left_mass, copy=True)
        lower = np.nextafter(np.zeros_like(parent_mass), parent_mass)
        upper = np.nextafter(parent_mass, np.zeros_like(parent_mass))
        left_mass = np.where(left_mass <= 0.0, lower, left_mass)
        left_mass = np.where(left_mass >= parent_mass, upper, left_mass)
    right_mass = parent_mass - left_mass
    return left_mass, right_mass, repair_count


def draw_prior_allocation_paths(
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    *,
    particle_count: int,
    seed: int,
) -> FloatArray:
    """Draw complete independent Beta paths in one schedule coordinate order."""
    count = _positive_integer(particle_count, name="particle_count")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**128:
        raise ValueError("seed must be an integer in [0, 2**128).")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    result = np.empty((count, len(schedule.coordinate_node_ids)), dtype=np.float64)
    for coordinate, node_id in enumerate(schedule.coordinate_node_ids):
        node = tree.node(node_id)
        assert node.left_id is not None and node.right_id is not None
        left = tree.node(node.left_id)
        right = tree.node(node.right_id)
        values, _ = _open_beta_draw(
            generator,
            left.alpha_total,
            right.alpha_total,
            size=count,
        )
        result[:, coordinate] = values
    result.setflags(write=False)
    return result


def _validate_paths(
    paths: ArrayLike | None,
    *,
    config: ResolutionSMCConfig,
    schedule: ResolutionSchedule,
) -> FloatArray | None:
    if paths is None:
        return None
    result = _readonly_float(paths, name="allocation_paths")
    expected = (config.particle_count, len(schedule.coordinate_node_ids))
    if result.shape != expected:
        raise ValueError("allocation_paths must have shape (particle_count, split_coordinate_count).")
    if np.any((result <= 0.0) | (result >= 1.0)):
        raise ValueError("allocation_paths must lie strictly inside the Beta support.")
    if config.resampling_policy != "never":
        raise ValueError("preassigned complete paths are supported only when resampling is disabled.")
    return result


def _terminal_masses_from_paths(
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    paths: FloatArray,
    *,
    root_total: float,
) -> FloatArray:
    masses = np.zeros((paths.shape[0], len(tree.nodes)), dtype=np.float64)
    masses[:, tree.root_id] = root_total
    coordinate = 0
    for batch in schedule.batches:
        for node_id in batch.node_ids:
            node = tree.node(node_id)
            assert node.left_id is not None and node.right_id is not None
            parent = masses[:, node_id]
            fraction = paths[:, coordinate]
            left, right, _ = _split_parent_mass(parent, fraction)
            masses[:, node.left_id] = left
            masses[:, node.right_id] = right
            coordinate += 1
    leaf = _leaf_masses(tree, masses)
    total_error = np.max(np.abs(np.sum(leaf, axis=1) - root_total))
    tolerance = 8.0 * np.spacing(root_total) * max(1, tree.cell_ids.size)
    if total_error > tolerance:
        raise ValueError("complete allocation paths do not conserve the root mass.")
    return leaf


def direct_iid_likelihood_average(
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    guide: GaussianGuideSpecification,
    *,
    root_total: float,
    config: ResolutionSMCConfig,
    allocation_paths: ArrayLike | None = None,
) -> DirectIIDResult:
    """Average exact terminal native likelihoods over complete prior paths."""
    if config.resampling_policy != "never":
        raise ValueError("direct IID evaluation requires resampling_policy='never'.")
    total = _positive_float(root_total, name="root_total")
    start = time.perf_counter()
    paths = _validate_paths(allocation_paths, config=config, schedule=schedule)
    if paths is None:
        generator = np.random.Generator(np.random.PCG64(config.seed))
        generated = np.empty(
            (config.particle_count, len(schedule.coordinate_node_ids)),
            dtype=np.float64,
        )
        beta_endpoint_repair_count = 0
        for coordinate, node_id in enumerate(schedule.coordinate_node_ids):
            node = tree.node(node_id)
            assert node.left_id is not None and node.right_id is not None
            values, repairs = _open_beta_draw(
                generator,
                tree.node(node.left_id).alpha_total,
                tree.node(node.right_id).alpha_total,
                size=config.particle_count,
            )
            generated[:, coordinate] = values
            beta_endpoint_repair_count += repairs
        generated.setflags(write=False)
        paths = generated
        beta_draw_count = config.particle_count * len(schedule.coordinate_node_ids)
    else:
        beta_draw_count = 0
        beta_endpoint_repair_count = 0
    leaf = _terminal_masses_from_paths(tree, schedule, paths, root_total=total)
    terminal = _exact_terminal_log_likelihood(guide, leaf)
    log_estimate = _logmeanexp(terminal)
    elapsed = time.perf_counter() - start
    terminal.setflags(write=False)
    leaf.setflags(write=False)
    return DirectIIDResult(
        likelihood=_linear_from_log(log_estimate),
        log_likelihood=log_estimate,
        terminal_log_likelihoods=terminal,
        leaf_masses=leaf,
        allocation_paths_sha256=_array_sha256(paths),
        elapsed_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        beta_draw_count=beta_draw_count,
        beta_endpoint_repair_count=beta_endpoint_repair_count,
        likelihood_evaluation_count=config.particle_count,
    )


def _rng_states(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    sequence = np.random.SeedSequence(seed)
    propagation_seed, resampling_seed = sequence.spawn(2)
    return (
        np.random.Generator(np.random.PCG64(propagation_seed)),
        np.random.Generator(np.random.PCG64(resampling_seed)),
    )


def _json_safe_rng_state(generator: np.random.Generator) -> dict[str, object]:
    state = generator.bit_generator.state
    # PCG64's state consists only of strings and Python-sized integers.
    return json.loads(_canonical_json(state))


def _generator_from_state(state: Mapping[str, object]) -> np.random.Generator:
    if state.get("bit_generator") != "PCG64":
        raise ValueError("checkpoint RNG state must use PCG64.")
    generator = np.random.Generator(np.random.PCG64())
    try:
        generator.bit_generator.state = dict(state)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint contains an invalid PCG64 state.") from error
    return generator


def _scientific_fingerprint(
    *,
    log_likelihood: float,
    terminal_log_likelihoods: FloatArray,
    normalized_log_weights: FloatArray,
    terminal_leaf_masses: FloatArray,
    lineage: IntArray,
    ancestry: IntArray,
    diagnostics: Sequence[SMCLevelDiagnostic],
) -> str:
    diagnostic_payload = []
    for diagnostic in diagnostics:
        payload = asdict(diagnostic)
        for operational in ("elapsed_seconds", "peak_rss_bytes"):
            payload.pop(operational)
        diagnostic_payload.append(payload)
    return _sha256_json(
        {
            "schema": _SCHEMA,
            "log_likelihood": log_likelihood,
            "terminal_log_likelihoods_sha256": _array_sha256(terminal_log_likelihoods),
            "normalized_log_weights_sha256": _array_sha256(normalized_log_weights),
            "terminal_leaf_masses_sha256": _array_sha256(terminal_leaf_masses),
            "lineage_sha256": _array_sha256(lineage),
            "ancestry_sha256": _array_sha256(ancestry),
            "diagnostics": diagnostic_payload,
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class ResolutionSMCCheckpoint:
    """Exact restart boundary with content and provenance authentication."""

    completed_levels: int
    coordinate_cursor: int
    frontier_node_ids: tuple[int, ...]
    frontier: ParticleFrontier
    normalized_log_weights: FloatArray
    current_guide_log_likelihoods: FloatArray
    log_normalizer_accumulator: float
    lineage: IntArray
    ancestry: IntArray
    diagnostics: tuple[SMCLevelDiagnostic, ...]
    propagation_rng_state: Mapping[str, object]
    resampling_rng_state: Mapping[str, object]
    tree_identity: str
    schedule_identity: str
    guide_identity: str
    config_identity: str
    allocation_paths_sha256: str | None
    source_identity: str

    def _metadata(self) -> dict[str, object]:
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "completed_levels": self.completed_levels,
            "coordinate_cursor": self.coordinate_cursor,
            "frontier_node_ids": list(self.frontier_node_ids),
            "log_normalizer_accumulator": self.log_normalizer_accumulator,
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "propagation_rng_state": self.propagation_rng_state,
            "resampling_rng_state": self.resampling_rng_state,
            "tree_identity": self.tree_identity,
            "schedule_identity": self.schedule_identity,
            "guide_identity": self.guide_identity,
            "config_identity": self.config_identity,
            "allocation_paths_sha256": self.allocation_paths_sha256,
            "source_identity": self.source_identity,
        }

    def _arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "node_masses": self.frontier.node_masses,
            "observation_mean": self.frontier.observation_mean,
            "unresolved_covariance": self.frontier.unresolved_covariance,
            "normalized_log_weights": self.normalized_log_weights,
            "current_guide_log_likelihoods": self.current_guide_log_likelihoods,
            "lineage": self.lineage,
            "ancestry": self.ancestry,
        }

    @property
    def content_sha256(self) -> str:
        """Return the authenticated content identity."""
        payload = {
            "metadata": self._metadata(),
            "arrays": {name: _array_sha256(values) for name, values in sorted(self._arrays().items())},
        }
        return _sha256_json(payload)

    def save(self, path: str | Path) -> None:
        """Write one compressed NPZ checkpoint atomically."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        metadata["content_sha256"] = self.content_sha256
        temporary = destination.with_name(f".{destination.name}.tmp.npz")
        arrays = self._arrays()
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(_canonical_json(metadata)),
            node_masses=arrays["node_masses"],
            observation_mean=arrays["observation_mean"],
            unresolved_covariance=arrays["unresolved_covariance"],
            normalized_log_weights=arrays["normalized_log_weights"],
            current_guide_log_likelihoods=arrays["current_guide_log_likelihoods"],
            lineage=arrays["lineage"],
            ancestry=arrays["ancestry"],
        )
        temporary.replace(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        tree: ResolutionTree,
        schedule: ResolutionSchedule,
        guide: GaussianGuideSpecification,
        config: ResolutionSMCConfig,
        allocation_paths_sha256: str | None,
        source_identity: str,
    ) -> ResolutionSMCCheckpoint:
        """Load and authenticate a checkpoint against the complete run."""
        expected_arrays = {
            "node_masses",
            "observation_mean",
            "unresolved_covariance",
            "normalized_log_weights",
            "current_guide_log_likelihoods",
            "lineage",
            "ancestry",
            "metadata_json",
        }
        with np.load(Path(path), allow_pickle=False) as stored:
            if set(stored.files) != expected_arrays:
                raise ValueError("checkpoint contains missing or unexpected arrays.")
            try:
                metadata = json.loads(str(stored["metadata_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("checkpoint metadata is not valid JSON.") from error
            arrays = {
                name: np.array(stored[name], copy=True) for name in expected_arrays if name != "metadata_json"
            }
        if metadata.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema is not supported.")
        supplied_digest = metadata.pop("content_sha256", None)
        try:
            diagnostics = tuple(
                SMCLevelDiagnostic(
                    **{
                        **payload,
                        "node_ids": tuple(payload["node_ids"]),
                    }
                )
                for payload in metadata["diagnostics"]
            )
            checkpoint = cls(
                completed_levels=int(metadata["completed_levels"]),
                coordinate_cursor=int(metadata["coordinate_cursor"]),
                frontier_node_ids=tuple(int(value) for value in metadata["frontier_node_ids"]),
                frontier=ParticleFrontier(
                    node_masses=_readonly_float(arrays["node_masses"], name="node_masses"),
                    observation_mean=_readonly_float(arrays["observation_mean"], name="observation_mean"),
                    unresolved_covariance=_readonly_float(
                        arrays["unresolved_covariance"],
                        name="unresolved_covariance",
                    ),
                ),
                normalized_log_weights=_readonly_float(
                    arrays["normalized_log_weights"],
                    name="normalized_log_weights",
                ),
                current_guide_log_likelihoods=_readonly_float(
                    arrays["current_guide_log_likelihoods"],
                    name="current_guide_log_likelihoods",
                ),
                log_normalizer_accumulator=float(metadata["log_normalizer_accumulator"]),
                lineage=_readonly_int(arrays["lineage"], name="lineage"),
                ancestry=_readonly_int(arrays["ancestry"], name="ancestry"),
                diagnostics=diagnostics,
                propagation_rng_state=cast(Mapping[str, object], metadata["propagation_rng_state"]),
                resampling_rng_state=cast(Mapping[str, object], metadata["resampling_rng_state"]),
                tree_identity=str(metadata["tree_identity"]),
                schedule_identity=str(metadata["schedule_identity"]),
                guide_identity=str(metadata["guide_identity"]),
                config_identity=str(metadata["config_identity"]),
                allocation_paths_sha256=cast(str | None, metadata["allocation_paths_sha256"]),
                source_identity=str(metadata["source_identity"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint metadata or arrays are malformed.") from error
        if supplied_digest != checkpoint.content_sha256:
            raise ValueError("checkpoint content digest does not match its payload.")
        expected_identities = {
            "tree identity": (checkpoint.tree_identity, tree.identity),
            "schedule identity": (checkpoint.schedule_identity, schedule.identity),
            "guide/input identity": (checkpoint.guide_identity, guide.identity),
            "configuration identity": (checkpoint.config_identity, config.identity),
            "allocation-path identity": (
                checkpoint.allocation_paths_sha256,
                allocation_paths_sha256,
            ),
            "source identity": (checkpoint.source_identity, source_identity),
        }
        for name, (actual, expected) in expected_identities.items():
            if actual != expected:
                raise ValueError(f"checkpoint {name} does not match the requested run.")
        expected_frontier = schedule.frontier_after(tree, checkpoint.completed_levels)
        if checkpoint.frontier_node_ids != expected_frontier:
            raise ValueError("checkpoint frontier does not match its completed level.")
        expected_cursor = sum(
            len(batch.node_ids) for batch in schedule.batches[: checkpoint.completed_levels]
        )
        if checkpoint.coordinate_cursor != expected_cursor:
            raise ValueError("checkpoint coordinate cursor does not match its completed level.")
        particle_count = config.particle_count
        observation_count = guide.observation.size
        expected_shapes = {
            "node_masses": (particle_count, len(tree.nodes)),
            "observation_mean": (particle_count, observation_count),
            "unresolved_covariance": (
                particle_count,
                observation_count,
                observation_count,
            ),
            "normalized_log_weights": (particle_count,),
            "current_guide_log_likelihoods": (particle_count,),
            "lineage": (particle_count,),
            "ancestry": (checkpoint.completed_levels, particle_count),
        }
        for name, expected_shape in expected_shapes.items():
            actual = (
                getattr(checkpoint.frontier, name)
                if name in {"node_masses", "observation_mean", "unresolved_covariance"}
                else getattr(checkpoint, name)
            )
            if actual.shape != expected_shape:
                raise ValueError(f"checkpoint {name} has an invalid shape.")
        if len(checkpoint.diagnostics) != checkpoint.completed_levels:
            raise ValueError("checkpoint diagnostics do not match completed levels.")
        if not math.isfinite(checkpoint.log_normalizer_accumulator):
            raise ValueError("checkpoint log normalizer is non-finite.")
        if not np.isclose(
            float(cast(float, logsumexp(checkpoint.normalized_log_weights))),
            0.0,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise ValueError("checkpoint normalized log weights do not sum to one.")
        _generator_from_state(checkpoint.propagation_rng_state)
        _generator_from_state(checkpoint.resampling_rng_state)
        return checkpoint


def _make_checkpoint(
    *,
    completed_levels: int,
    coordinate_cursor: int,
    frontier_node_ids: tuple[int, ...],
    node_masses: FloatArray,
    observation_mean: FloatArray,
    unresolved_covariance: FloatArray,
    normalized_log_weights: FloatArray,
    current_guide_log_likelihoods: FloatArray,
    log_normalizer_accumulator: float,
    lineage: IntArray,
    ancestry: Sequence[IntArray],
    diagnostics: Sequence[SMCLevelDiagnostic],
    propagation_rng: np.random.Generator,
    resampling_rng: np.random.Generator,
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    guide: GaussianGuideSpecification,
    config: ResolutionSMCConfig,
    allocation_paths_sha256: str | None,
    source_identity: str,
) -> ResolutionSMCCheckpoint:
    ancestry_array = (
        np.stack(ancestry, axis=0) if ancestry else np.empty((0, config.particle_count), dtype=np.int64)
    )
    return ResolutionSMCCheckpoint(
        completed_levels=completed_levels,
        coordinate_cursor=coordinate_cursor,
        frontier_node_ids=frontier_node_ids,
        frontier=ParticleFrontier(
            node_masses=_readonly_float(node_masses, name="node_masses"),
            observation_mean=_readonly_float(observation_mean, name="observation_mean"),
            unresolved_covariance=_readonly_float(
                unresolved_covariance,
                name="unresolved_covariance",
            ),
        ),
        normalized_log_weights=_readonly_float(
            normalized_log_weights,
            name="normalized_log_weights",
        ),
        current_guide_log_likelihoods=_readonly_float(
            current_guide_log_likelihoods,
            name="current_guide_log_likelihoods",
        ),
        log_normalizer_accumulator=float(log_normalizer_accumulator),
        lineage=_readonly_int(lineage, name="lineage"),
        ancestry=_readonly_int(ancestry_array, name="ancestry"),
        diagnostics=tuple(diagnostics),
        propagation_rng_state=_json_safe_rng_state(propagation_rng),
        resampling_rng_state=_json_safe_rng_state(resampling_rng),
        tree_identity=tree.identity,
        schedule_identity=schedule.identity,
        guide_identity=guide.identity,
        config_identity=config.identity,
        allocation_paths_sha256=allocation_paths_sha256,
        source_identity=source_identity,
    )


def run_resolution_smc(
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    guide: GaussianGuideSpecification,
    *,
    root_total: float,
    config: ResolutionSMCConfig,
    source_identity: str,
    allocation_paths: ArrayLike | None = None,
    checkpoint: ResolutionSMCCheckpoint | None = None,
    stop_after_level: int | None = None,
) -> tuple[ResolutionSMCResult | None, ResolutionSMCCheckpoint]:
    """Run or resume prior-proposal resolution SMC.

    ``stop_after_level`` counts completed schedule batches and may be zero.
    The returned checkpoint is always a complete exact restart boundary.
    """
    if not isinstance(tree, ResolutionTree):
        raise TypeError("tree must be a ResolutionTree.")
    if not isinstance(schedule, ResolutionSchedule):
        raise TypeError("schedule must be a ResolutionSchedule.")
    if not isinstance(guide, GaussianGuideSpecification):
        raise TypeError("guide must be a GaussianGuideSpecification.")
    if not isinstance(config, ResolutionSMCConfig):
        raise TypeError("config must be a ResolutionSMCConfig.")
    total = _positive_float(root_total, name="root_total")
    normalized_source_identity = str(source_identity).strip()
    if not normalized_source_identity:
        raise ValueError("source_identity cannot be empty.")
    paths = _validate_paths(allocation_paths, config=config, schedule=schedule)
    paths_sha256 = _path_identity(paths)
    target_level = (
        len(schedule.batches)
        if stop_after_level is None
        else _nonnegative_integer(
            stop_after_level,
            name="stop_after_level",
        )
    )
    if target_level > len(schedule.batches):
        raise ValueError("stop_after_level exceeds the schedule length.")
    moments = _tree_guide_moments(tree, guide)
    run_start = time.perf_counter()

    if checkpoint is None:
        propagation_rng, resampling_rng = _rng_states(config.seed)
        frontier = _initial_frontier(
            tree,
            moments,
            root_total=total,
            particle_count=config.particle_count,
        )
        node_masses = np.array(frontier.node_masses, copy=True)
        observation_mean = np.array(frontier.observation_mean, copy=True)
        unresolved_covariance = np.array(frontier.unresolved_covariance, copy=True)
        normalized_log_weights = np.full(
            config.particle_count,
            -math.log(config.particle_count),
            dtype=np.float64,
        )
        current_guide = _batched_gaussian_log_likelihood(
            guide,
            observation_mean,
            unresolved_covariance,
        )
        if not np.all(current_guide == current_guide[0]):
            raise RuntimeError("initial particles must have one common Gaussian guide value.")
        log_normalizer = float(current_guide[0])
        lineage = np.arange(config.particle_count, dtype=np.int64)
        ancestry: list[IntArray] = []
        diagnostics: list[SMCLevelDiagnostic] = []
        completed_levels = 0
        coordinate_cursor = 0
    else:
        if checkpoint.tree_identity != tree.identity:
            raise ValueError("checkpoint tree identity does not match.")
        if checkpoint.schedule_identity != schedule.identity:
            raise ValueError("checkpoint schedule identity does not match.")
        if checkpoint.guide_identity != guide.identity:
            raise ValueError("checkpoint guide/input identity does not match.")
        if checkpoint.config_identity != config.identity:
            raise ValueError("checkpoint configuration identity does not match.")
        if checkpoint.allocation_paths_sha256 != paths_sha256:
            raise ValueError("checkpoint allocation-path identity does not match.")
        if checkpoint.source_identity != normalized_source_identity:
            raise ValueError("checkpoint source identity does not match.")
        if checkpoint.completed_levels > target_level:
            raise ValueError("checkpoint is beyond the requested stopping level.")
        expected_frontier = schedule.frontier_after(tree, checkpoint.completed_levels)
        if checkpoint.frontier_node_ids != expected_frontier:
            raise ValueError("checkpoint frontier does not match the schedule.")
        propagation_rng = _generator_from_state(checkpoint.propagation_rng_state)
        resampling_rng = _generator_from_state(checkpoint.resampling_rng_state)
        node_masses = np.array(checkpoint.frontier.node_masses, copy=True)
        observation_mean = np.array(checkpoint.frontier.observation_mean, copy=True)
        unresolved_covariance = np.array(
            checkpoint.frontier.unresolved_covariance,
            copy=True,
        )
        normalized_log_weights = np.array(
            checkpoint.normalized_log_weights,
            copy=True,
        )
        current_guide = np.array(
            checkpoint.current_guide_log_likelihoods,
            copy=True,
        )
        log_normalizer = float(checkpoint.log_normalizer_accumulator)
        lineage = np.array(checkpoint.lineage, copy=True)
        ancestry = [
            np.array(checkpoint.ancestry[level], dtype=np.int64, copy=True)
            for level in range(checkpoint.ancestry.shape[0])
        ]
        diagnostics = list(checkpoint.diagnostics)
        completed_levels = checkpoint.completed_levels
        coordinate_cursor = checkpoint.coordinate_cursor

    active_frontier = set(schedule.frontier_after(tree, completed_levels))
    for level_index in range(completed_levels, target_level):
        level_start = time.perf_counter()
        batch = schedule.batches[level_index]
        previous_guide = np.array(current_guide, copy=True)
        max_conservation_error = 0.0
        beta_endpoint_repair_count = 0
        for node_id in batch.node_ids:
            if node_id not in active_frontier:
                raise RuntimeError("scheduled node is not active at this resolution.")
            node = tree.node(node_id)
            assert node.left_id is not None and node.right_id is not None
            left_node = tree.node(node.left_id)
            right_node = tree.node(node.right_id)
            parent_mass = node_masses[:, node_id]
            if paths is None:
                fraction, fraction_repairs = _open_beta_draw(
                    propagation_rng,
                    left_node.alpha_total,
                    right_node.alpha_total,
                    size=config.particle_count,
                )
                beta_endpoint_repair_count += fraction_repairs
            else:
                fraction = paths[:, coordinate_cursor]
            coordinate_cursor += 1
            if not np.all(np.isfinite(fraction)) or np.any((fraction <= 0.0) | (fraction >= 1.0)):
                raise ValueError("represented Beta fractions must lie strictly inside (0, 1).")
            left_mass, right_mass, mass_repairs = _split_parent_mass(
                parent_mass,
                fraction,
            )
            beta_endpoint_repair_count += mass_repairs
            if (
                not np.all(np.isfinite(left_mass))
                or not np.all(np.isfinite(right_mass))
                or np.any(left_mass <= 0.0)
                or np.any(right_mass <= 0.0)
            ):
                raise ValueError("child masses must remain finite and strictly positive.")
            conservation_error = float(np.max(np.abs((left_mass + right_mass) - parent_mass)))
            max_conservation_error = max(max_conservation_error, conservation_error)
            tolerance = float(np.max(np.spacing(parent_mass)))
            if conservation_error > tolerance:
                raise ValueError("a Gamma--Beta refinement did not conserve parent mass.")
            node_masses[:, node.left_id] = left_mass
            node_masses[:, node.right_id] = right_mass
            prior_left = left_node.alpha_total / node.alpha_total
            contrast = moments.means[node.left_id] - moments.means[node.right_id]
            observation_mean += (parent_mass * (fraction - prior_left))[:, np.newaxis] * contrast[
                np.newaxis, :
            ]
            unresolved_covariance -= (
                np.square(parent_mass)[:, np.newaxis, np.newaxis]
                * moments.covariances[node_id][np.newaxis, :, :]
            )
            unresolved_covariance += (
                np.square(left_mass)[:, np.newaxis, np.newaxis]
                * moments.covariances[node.left_id][np.newaxis, :, :]
            )
            unresolved_covariance += (
                np.square(right_mass)[:, np.newaxis, np.newaxis]
                * moments.covariances[node.right_id][np.newaxis, :, :]
            )
            unresolved_covariance = 0.5 * (unresolved_covariance + np.swapaxes(unresolved_covariance, 1, 2))
            active_frontier.remove(node_id)
            active_frontier.add(node.left_id)
            active_frontier.add(node.right_id)

        recomputed_mean, recomputed_covariance = _recompute_frontier(
            sorted(active_frontier),
            node_masses,
            moments,
        )
        max_mean_update_error = float(np.max(np.abs(observation_mean - recomputed_mean), initial=0.0))
        max_covariance_update_error = float(
            np.max(
                np.abs(unresolved_covariance - recomputed_covariance),
                initial=0.0,
            )
        )
        scale = max(
            1.0,
            float(np.max(np.abs(recomputed_mean), initial=0.0)),
            float(np.max(np.abs(recomputed_covariance), initial=0.0)),
        )
        update_tolerance = (
            4096.0 * np.finfo(np.float64).eps * max(1, len(tree.nodes), guide.observation.size) * scale
        )
        if max_mean_update_error > update_tolerance:
            raise ValueError("local conditional-mean update disagrees with exact frontier recomputation.")
        if max_covariance_update_error > update_tolerance:
            raise ValueError("local covariance update disagrees with exact frontier recomputation.")
        # Re-anchor at the independently recomputed exact frontier.  This
        # makes the terminal all-leaf unresolved covariance literally zero.
        observation_mean = recomputed_mean
        unresolved_covariance = recomputed_covariance

        terminal = level_index == len(schedule.batches) - 1
        terminal_prediction_error = 0.0
        terminal_covariance_error = 0.0
        if terminal:
            leaf_masses = _leaf_masses(tree, node_masses)
            mass_error = float(np.max(np.abs(np.sum(leaf_masses, axis=1) - total)))
            mass_tolerance = 8.0 * np.spacing(total) * max(1, tree.cell_ids.size)
            if mass_error > mass_tolerance:
                raise ValueError("terminal native allocation does not conserve the retained root mass.")
            terminal_prediction = leaf_masses @ guide.design.T
            terminal_prediction_error = float(
                np.max(np.abs(observation_mean - terminal_prediction), initial=0.0)
            )
            prediction_scale = max(
                1.0,
                float(np.max(np.abs(terminal_prediction), initial=0.0)),
            )
            if terminal_prediction_error > (
                4096.0 * np.finfo(np.float64).eps * max(1, tree.cell_ids.size) * prediction_scale
            ):
                raise ValueError("terminal frontier prediction disagrees with the native design.")
            terminal_covariance_error = float(np.max(np.abs(unresolved_covariance), initial=0.0))
            if terminal_covariance_error != 0.0:
                raise ValueError("terminal unresolved covariance must be exactly zero.")
            current_guide = _exact_terminal_log_likelihood(guide, leaf_masses)
        else:
            current_guide = _batched_gaussian_log_likelihood(
                guide,
                observation_mean,
                unresolved_covariance,
            )

        incremental = current_guide - previous_guide
        if not np.all(np.isfinite(incremental)):
            raise ValueError("SMC incremental log weights are non-finite.")
        log_increment = float(cast(float, logsumexp(normalized_log_weights + incremental)))
        log_second = float(cast(float, logsumexp(normalized_log_weights + 2.0 * incremental)))
        squared_cv = max(0.0, math.exp(log_second - 2.0 * log_increment) - 1.0)
        incremental_cv = math.sqrt(squared_cv)
        updated_log_weights = normalized_log_weights + incremental - log_increment
        weights = np.exp(updated_log_weights)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=2.0e-13)
        ):
            raise ValueError("SMC normalized weights are invalid.")
        ess = float(1.0 / np.sum(np.square(weights)))
        positive = weights > 0.0
        entropy = -float(np.sum(weights[positive] * updated_log_weights[positive]))
        perplexity = float(math.exp(entropy))
        should_resample = False
        if not terminal:
            if config.resampling_policy == "always":
                should_resample = True
            elif config.resampling_policy == "ess":
                should_resample = ess < config.ess_fraction * config.particle_count
        with np.errstate(under="ignore"):
            linear_correction = np.exp(current_guide) - np.exp(previous_guide)
        if not np.all(np.isfinite(linear_correction)):
            raise ValueError("linear likelihood correction is non-finite.")
        if should_resample:
            selected = np.asarray(
                resampling_rng.choice(
                    config.particle_count,
                    size=config.particle_count,
                    replace=True,
                    p=weights,
                ),
                dtype=np.int64,
            )
            node_masses = node_masses[selected]
            observation_mean = observation_mean[selected]
            unresolved_covariance = unresolved_covariance[selected]
            current_guide = current_guide[selected]
            lineage = lineage[selected]
            normalized_log_weights = np.full(
                config.particle_count,
                -math.log(config.particle_count),
                dtype=np.float64,
            )
        else:
            selected = np.arange(config.particle_count, dtype=np.int64)
            normalized_log_weights = updated_log_weights
        ancestry.append(selected)
        log_normalizer += log_increment
        correction_variance = float(np.var(linear_correction, ddof=1)) if config.particle_count > 1 else 0.0
        level_elapsed = time.perf_counter() - level_start
        state_bytes = int(
            node_masses.nbytes
            + observation_mean.nbytes
            + unresolved_covariance.nbytes
            + normalized_log_weights.nbytes
            + current_guide.nbytes
            + lineage.nbytes
            + sum(item.nbytes for item in ancestry)
        )
        diagnostics.append(
            SMCLevelDiagnostic(
                level=level_index + 1,
                node_ids=batch.node_ids,
                terminal=terminal,
                resampled=should_resample,
                ess=ess,
                ess_fraction=ess / config.particle_count,
                incremental_weight_cv=incremental_cv,
                max_normalized_weight=float(np.max(weights)),
                shannon_perplexity=perplexity,
                unique_ancestor_count=int(np.unique(lineage).size),
                incremental_log_weight_min=float(np.min(incremental)),
                incremental_log_weight_mean=float(np.mean(incremental)),
                incremental_log_weight_max=float(np.max(incremental)),
                incremental_log_weight_sd=float(np.std(incremental)),
                linear_likelihood_correction_mean=float(np.mean(linear_correction)),
                linear_likelihood_correction_variance=correction_variance,
                log_normalizer_increment=log_increment,
                beta_draw_count=0 if paths is not None else config.particle_count * len(batch.node_ids),
                beta_endpoint_repair_count=beta_endpoint_repair_count,
                forward_update_count=config.particle_count * len(batch.node_ids),
                likelihood_evaluation_count=config.particle_count,
                elapsed_seconds=level_elapsed,
                state_bytes=state_bytes,
                peak_rss_bytes=_peak_rss_bytes(),
                max_mass_conservation_error=max_conservation_error,
                max_mean_update_error=max_mean_update_error,
                max_covariance_update_error=max_covariance_update_error,
                max_terminal_prediction_error=terminal_prediction_error,
                max_terminal_unresolved_covariance=terminal_covariance_error,
            )
        )
        completed_levels = level_index + 1

    final_checkpoint = _make_checkpoint(
        completed_levels=completed_levels,
        coordinate_cursor=coordinate_cursor,
        frontier_node_ids=tuple(sorted(active_frontier)),
        node_masses=node_masses,
        observation_mean=observation_mean,
        unresolved_covariance=unresolved_covariance,
        normalized_log_weights=normalized_log_weights,
        current_guide_log_likelihoods=current_guide,
        log_normalizer_accumulator=log_normalizer,
        lineage=lineage,
        ancestry=ancestry,
        diagnostics=diagnostics,
        propagation_rng=propagation_rng,
        resampling_rng=resampling_rng,
        tree=tree,
        schedule=schedule,
        guide=guide,
        config=config,
        allocation_paths_sha256=paths_sha256,
        source_identity=normalized_source_identity,
    )
    if completed_levels != len(schedule.batches):
        return None, final_checkpoint

    terminal_leaf_masses = _leaf_masses(tree, node_masses)
    terminal_log_likelihoods = _exact_terminal_log_likelihood(
        guide,
        terminal_leaf_masses,
    )
    accumulator_log_likelihood = log_normalizer
    if config.resampling_policy == "never":
        # Algebraically this is the same normalizer as the telescoping
        # accumulator.  The terminal expression is authoritative so a
        # path-matched direct IID calculation is bitwise identical.
        log_likelihood = _logmeanexp(terminal_log_likelihoods)
        accumulator_error: float | None = accumulator_log_likelihood - log_likelihood
    else:
        log_likelihood = accumulator_log_likelihood
        accumulator_error = None
    ancestry_array = np.stack(ancestry, axis=0)
    fingerprint = _scientific_fingerprint(
        log_likelihood=log_likelihood,
        terminal_log_likelihoods=terminal_log_likelihoods,
        normalized_log_weights=normalized_log_weights,
        terminal_leaf_masses=terminal_leaf_masses,
        lineage=lineage,
        ancestry=ancestry_array,
        diagnostics=diagnostics,
    )
    terminal_log_likelihoods.setflags(write=False)
    normalized_log_weights.setflags(write=False)
    terminal_leaf_masses.setflags(write=False)
    lineage.setflags(write=False)
    ancestry_array.setflags(write=False)
    result = ResolutionSMCResult(
        likelihood=_linear_from_log(log_likelihood),
        log_likelihood=log_likelihood,
        accumulator_log_likelihood=accumulator_log_likelihood,
        no_resampling_accumulator_error=accumulator_error,
        terminal_log_likelihoods=terminal_log_likelihoods,
        normalized_log_weights=normalized_log_weights,
        terminal_leaf_masses=terminal_leaf_masses,
        lineage=lineage,
        ancestry=ancestry_array,
        diagnostics=tuple(diagnostics),
        tree_identity=tree.identity,
        schedule_identity=schedule.identity,
        guide_identity=guide.identity,
        config_identity=config.identity,
        allocation_paths_sha256=paths_sha256,
        source_identity=normalized_source_identity,
        scientific_fingerprint=fingerprint,
        elapsed_seconds=time.perf_counter() - run_start,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    return result, final_checkpoint
