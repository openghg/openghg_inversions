"""Non-enumerating PyMC adapter for Gaussian dyadic product spaces.

This experimental adapter represents a partition by one binary value for each
possible dyadic split. Valid vectors are ancestry closed: a descendant can be
split only when every ancestor is split. A custom Metropolis-Hastings step
proposes one legal split or merge at a time, while native PyMC NUTS updates a
fixed root-and-contrast vector and any always-active outer coefficients.

Unlike :mod:`.pymc_product_space`, model construction does not enumerate valid
partitions or allocate a design tensor with a partition axis. The likelihood
uses one static finest-grid contrast design whose columns are enabled by the
split mask. The implementation remains a Gaussian proof of concept and does
not integrate with production RHIME sampling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

import pymc as pm  # type: ignore[import-untyped]  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
from pymc.blocking import RaveledVars, StatsType  # type: ignore[import-untyped]  # noqa: E402
from pymc.step_methods.arraystep import ArrayStep  # type: ignore[import-untyped]  # noqa: E402
from pymc.step_methods.compound import (  # type: ignore[import-untyped]  # noqa: E402
    StatDtype,
    StatShape,
    StepMethodState,
)
from pymc.step_methods.state import dataclass_state  # type: ignore[import-untyped]  # noqa: E402
from pymc.util import (  # type: ignore[import-untyped]  # noqa: E402
    RandomGenerator,
    get_value_vars_from_user_vars,
)
from pytensor.configdefaults import config  # noqa: E402
from pytensor.graph.basic import Variable  # noqa: E402

from .contrast import TreeContrastLayout  # noqa: E402
from .gaussian_product_space import GaussianProductSpaceTarget  # noqa: E402
from .partition_prior import RegionCountPartitionPrior  # noqa: E402
from .product_space import enumerate_partition_neighbors  # noqa: E402
from .state import PartitionState  # noqa: E402
from .tree import DyadicTree  # noqa: E402


@dataclass_state
class DyadicSplitMaskStepState(StepMethodState):
    """Serializable PyMC state for the parameter-free split-mask step."""

    tune: bool


class DyadicSplitMaskStep(ArrayStep):
    """Local split/merge MH update for one PyMC split-mask variable.

    The step decodes the current canonical mask, chooses uniformly from all
    unique one-split or one-merge neighbors, and evaluates PyMC's compiled joint
    log density with all continuous values held fixed. The reverse-neighbor
    count supplies the Hastings correction.

    Args:
        vars: One vector PyMC Bernoulli variable or its value variable.
        layout: Fixed contrast layout defining split-mask order and decoding.
        model: PyMC model containing ``vars``.
        rng: Seed or random generator accepted by PyMC.
        compile_kwargs: Optional arguments for :meth:`pymc.Model.compile_logp`.
        blocked: PyMC blocked-step flag. This step always owns one variable.
    """

    name = "dyadic_split_mask"
    default_blocked = True
    stats_dtypes_shapes: dict[str, tuple[StatDtype, StatShape]] = {
        "accepted": (bool, []),
        "log_acceptance_ratio": (np.float64, []),
        "partition_regions": (np.int64, []),
        "proposal_degree": (np.int64, []),
        "proposed_split": (bool, []),
        "reverse_degree": (np.int64, []),
        "tune": (bool, []),
    }
    _state_class = DyadicSplitMaskStepState

    def __init__(
        self,
        vars: Variable | Sequence[Variable],
        *,
        layout: TreeContrastLayout,
        model: pm.Model | None = None,
        rng: RandomGenerator = None,
        compile_kwargs: Mapping[str, Any] | None = None,
        blocked: bool = True,
    ) -> None:
        """Validate the split-mask variable and compile the joint log density."""
        if not blocked:
            raise ValueError("DyadicSplitMaskStep must be blocked.")
        if not isinstance(layout, TreeContrastLayout):
            raise TypeError("layout must be a TreeContrastLayout.")
        if not layout.split_node_ids:
            raise ValueError("DyadicSplitMaskStep requires at least one possible split.")

        model = pm.modelcontext(model)
        value_vars = get_value_vars_from_user_vars(vars, model)
        if len(value_vars) != 1:
            raise ValueError("DyadicSplitMaskStep requires exactly one split-mask variable.")
        initial_point = model.initial_point()
        initial_value = np.asarray(initial_point[cast(str, value_vars[0].name)])
        expected_shape = (len(layout.split_node_ids),)
        if initial_value.shape != expected_shape or initial_value.dtype.kind not in "iu":
            raise ValueError(f"The split-mask value variable must be an integer vector with shape {expected_shape}.")
        layout.partition_from_split_mask(_binary_mask(initial_value, expected_shape=expected_shape))

        self.layout = layout
        self.tune = True
        kwargs = dict(compile_kwargs or {})
        super().__init__(
            value_vars,
            [model.compile_logp(**kwargs)],
            blocked=blocked,
            rng=rng,
        )

    def astep(self, apoint: RaveledVars, *args: Any) -> tuple[RaveledVars, StatsType]:
        """Propose one legal neighboring mask and apply the Hastings correction."""
        expected_shape = (len(self.layout.split_node_ids),)
        current_mask = _binary_mask(apoint.data, expected_shape=expected_shape)
        current_partition = self.layout.partition_from_split_mask(current_mask)
        neighbors = enumerate_partition_neighbors(self.layout.tree, current_partition)
        if not neighbors:  # pragma: no cover - constructor rejects a one-cell tree.
            return apoint, [
                self._stats(
                    len(current_partition.active),
                    accepted=False,
                    log_acceptance_ratio=-math.inf,
                    proposal_degree=0,
                    proposed_split=False,
                    reverse_degree=0,
                )
            ]

        neighbor = neighbors[int(self.rng.integers(len(neighbors)))]
        candidate_mask = self.layout.split_mask(neighbor.partition)
        candidate_data = candidate_mask.astype(apoint.data.dtype, copy=False)
        candidate = RaveledVars(candidate_data, apoint.point_map_info)

        reverse_neighbors = enumerate_partition_neighbors(self.layout.tree, neighbor.partition)
        reverse = next(
            (item for item in reverse_neighbors if item.partition == current_partition),
            None,
        )
        if reverse is None:  # pragma: no cover - local move kernel invariant.
            raise ValueError("Every local split-mask proposal must have a reverse edge.")

        logp = args[0]
        current_logp = float(logp(apoint))
        candidate_logp = float(logp(candidate))
        if not math.isfinite(current_logp):
            raise ValueError("Current PyMC joint log density must be finite.")
        if math.isnan(candidate_logp) or candidate_logp == math.inf:
            raise ValueError("Candidate PyMC joint log density must be finite or negative infinity.")

        log_acceptance_ratio = candidate_logp - current_logp + reverse.log_q - neighbor.log_q
        uniform = float(self.rng.random())
        log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
        accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
        selected = candidate if accepted else apoint
        selected_regions = len(neighbor.partition.active) if accepted else len(current_partition.active)
        return selected, [
            self._stats(
                selected_regions,
                accepted=accepted,
                log_acceptance_ratio=log_acceptance_ratio,
                proposal_degree=len(neighbors),
                proposed_split=len(neighbor.partition.active) > len(current_partition.active),
                reverse_degree=len(reverse_neighbors),
            )
        ]

    def reset_tuning(self) -> None:
        """Retain PyMC's no-op tuning contract for this parameter-free step."""
        self.tune = True

    def _stats(
        self,
        partition_regions: int,
        *,
        accepted: bool,
        log_acceptance_ratio: float,
        proposal_degree: int,
        proposed_split: bool,
        reverse_degree: int,
    ) -> dict[str, bool | float | int]:
        """Build one sample-stat record for the selected partition.

        Args:
            partition_regions: Number of regions after accept/reject selection.
            accepted: Whether the candidate mask was accepted.
            log_acceptance_ratio: Unclipped Metropolis-Hastings log ratio.
            proposal_degree: Number of legal neighbors at the source.
            proposed_split: Whether the candidate had one more region.
            reverse_degree: Number of legal neighbors at the candidate.

        Returns:
            Scalar statistics matching :attr:`stats_dtypes_shapes`.
        """
        return {
            "accepted": accepted,
            "log_acceptance_ratio": float(log_acceptance_ratio),
            "partition_regions": partition_regions,
            "proposal_degree": proposal_degree,
            "proposed_split": proposed_split,
            "reverse_degree": reverse_degree,
            "tune": self.tune,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PyMCSplitMaskProductSpaceModel:
    """Built non-enumerating PyMC model for a Gaussian product-space target.

    Attributes:
        model: PyMC model with a static contrast design.
        target: Framework-independent Gaussian reference target.
        partition_prior: Symbolic and framework-independent partition prior.
        split_mask: Canonical binary split-mask variable.
        inner_whitened: Permanent standard-normal inner coordinates.
        outer_whitened: Standard-normal outer coordinates, or ``None`` when the
            target has no outer regions.
    """

    model: pm.Model
    target: GaussianProductSpaceTarget
    partition_prior: RegionCountPartitionPrior
    split_mask: Variable
    inner_whitened: Variable
    outer_whitened: Variable | None

    def step_methods(
        self,
        *,
        partition_rng: RandomGenerator = None,
        nuts_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[DyadicSplitMaskStep, pm.NUTS]:
        """Create partition-first compound steps with native PyMC NUTS.

        Args:
            partition_rng: Initial seed or generator for the local partition
                step. Pass ``random_seed`` to :func:`pymc.sample` for
                reproducible compound chains.
            nuts_kwargs: Optional arguments forwarded to :class:`pymc.NUTS`.

        Returns:
            ``(split_mask_step, nuts_step)`` in execution order. Only NUTS has
            adaptive tuning parameters.
        """
        continuous = [self.inner_whitened]
        if self.outer_whitened is not None:
            continuous.append(self.outer_whitened)
        with self.model:
            partition_step = cast(
                DyadicSplitMaskStep,
                DyadicSplitMaskStep(
                    self.split_mask,
                    layout=self.target.contrast_layout,
                    model=self.model,
                    rng=partition_rng,
                ),
            )
            nuts_step = cast(pm.NUTS, pm.NUTS(continuous, **dict(nuts_kwargs or {})))
        return partition_step, nuts_step

    def physical_state(
        self,
        split_mask: npt.ArrayLike,
        inner_whitened: npt.ArrayLike,
        outer_whitened: npt.ArrayLike | None = None,
    ) -> tuple[PartitionState, np.ndarray, np.ndarray]:
        """Map sampled mask and whitened values to physical coordinates.

        Args:
            split_mask: Binary mask in stable layout order.
            inner_whitened: Permanent standard-normal inner vector.
            outer_whitened: Standard-normal outer vector, omitted only for a
                target without outer coefficients.

        Returns:
            Partition, physical inner contrasts, and physical outer
            coefficients.
        """
        mask = _binary_mask(
            split_mask,
            expected_shape=(len(self.target.contrast_layout.split_node_ids),),
        )
        partition = self.target.contrast_layout.partition_from_split_mask(mask)
        inner = _finite_vector(inner_whitened, name="inner_whitened")
        if inner.shape != self.target.inner_prior_variances.shape:
            raise ValueError("inner_whitened has the wrong permanent dimension.")
        inner_coordinates = np.sqrt(self.target.inner_prior_variances) * inner

        outer_count = self.target.outer_design.shape[1]
        if outer_whitened is None:
            if outer_count:
                raise ValueError("outer_whitened is required for this target.")
            outer: npt.NDArray[np.float64] = np.empty(0, dtype=float)
        else:
            outer_raw = _finite_vector(outer_whitened, name="outer_whitened")
            if outer_raw.shape != (outer_count,):
                raise ValueError("outer_whitened has the wrong outer-region dimension.")
            outer = np.linalg.cholesky(self.target.outer_prior_covariance) @ outer_raw
        return partition, inner_coordinates, outer


def build_pymc_split_mask_product_space_model(
    target: GaussianProductSpaceTarget,
    *,
    partition_prior: RegionCountPartitionPrior | None = None,
    initial_partition: PartitionState | None = None,
) -> PyMCSplitMaskProductSpaceModel:
    """Build a non-enumerating fixed-shape PyMC product-space model.

    The target must use equal active and inactive Gaussian variances so that a
    permanent whitened coordinate has the same prior meaning under every
    partition. The partition prior must be a :class:`RegionCountPartitionPrior`
    because an arbitrary Python callback cannot be represented in a symbolic
    PyTensor graph.

    Args:
        target: Validated Gaussian reference target.
        partition_prior: Symbolic region-count prior. When omitted,
            ``target.partition_log_prior`` must already be such an object.
        initial_partition: Optional valid positive-mass initial frontier. The
            default is a deterministic frontier at the smallest supported K.

    Returns:
        Built model and variables ready for
        :meth:`PyMCSplitMaskProductSpaceModel.step_methods`.

    Raises:
        TypeError: If the target or partition prior has the wrong type.
        ValueError: If whitening is partition dependent, the tree has only one
            grid cell, prior trees disagree, or the initial state is invalid.
    """
    if not isinstance(target, GaussianProductSpaceTarget):
        raise TypeError("target must be a GaussianProductSpaceTarget.")
    if not np.array_equal(
        target.inactive_pseudo_prior_variances,
        target.inner_prior_variances,
    ):
        raise ValueError(
            "The split-mask PyMC adapter requires inactive pseudo-prior "
            "variances to equal active prior variances."
        )
    if not target.contrast_layout.split_node_ids:
        raise ValueError("The split-mask PyMC adapter requires at least two finest grid cells.")

    prior = target.partition_log_prior if partition_prior is None else partition_prior
    if not isinstance(prior, RegionCountPartitionPrior):
        raise TypeError(
            "partition_prior must be a RegionCountPartitionPrior; arbitrary "
            "Python callbacks cannot be represented symbolically."
        )
    if prior.tree != target.tree:
        raise ValueError("partition_prior and target must use the same dyadic tree.")

    if initial_partition is None:
        supported_k = np.flatnonzero(np.isfinite(prior.log_probability_by_k))
        initial = _partition_with_region_count(target.tree, int(supported_k[0]))
    else:
        if not isinstance(initial_partition, PartitionState):
            raise TypeError("initial_partition must be a PartitionState.")
        initial_partition.validate(target.tree)
        initial = initial_partition
    if not math.isfinite(prior(initial)):
        raise ValueError("initial_partition must have positive prior mass.")

    layout = target.contrast_layout
    initial_mask = layout.split_mask(initial).astype(np.int8)
    finest_grid_design = target.inner_design.values[:, target.tree.leaf_ids]
    full_inner_design = layout.full_contrast_design(finest_grid_design)
    whitened_inner_design = _model_float(
        full_inner_design * np.sqrt(target.inner_prior_variances)
    )

    outer_count = target.outer_design.shape[1]
    if outer_count:
        whitened_outer_design = _model_float(
            target.outer_design @ np.linalg.cholesky(target.outer_prior_covariance)
        )
    else:
        whitened_outer_design = _model_float(
            np.empty((target.observations.size, 0), dtype=float)
        )

    child_indices, parent_indices = _split_ancestry_indices(layout)
    prior_table = _model_float(prior.log_probability_by_k)
    with pm.Model() as model:
        split_mask = pm.Bernoulli(
            "split_mask",
            p=0.5,
            shape=len(layout.split_node_ids),
            initval=initial_mask,
        )
        split_mask_tensor: Any = split_mask
        if child_indices.size:
            canonical = pt.all(
                split_mask_tensor[child_indices] <= split_mask_tensor[parent_indices]
            )
            pm.Potential(
                "canonical_partition",
                cast(
                    Any,
                    pt.switch(canonical, _model_float(0.0), _model_float(-math.inf)),
                ),
            )
        split_count: Any = pt.sum(split_mask_tensor)
        basis_region_count = pm.Deterministic(
            "basis_region_count",
            1 + split_count,
        )
        pm.Potential(
            "partition_prior",
            pt.as_tensor_variable(prior_table)[basis_region_count],
        )

        inner_whitened = pm.Normal(
            "inner_whitened",
            mu=0.0,
            sigma=1.0,
            shape=layout.coordinate_count,
        )
        coordinate_mask = pt.concatenate(
            (
                pt.ones((1,), dtype=config.floatX),
                pt.cast(split_mask_tensor, config.floatX),
            )
        )
        prediction = pt.as_tensor_variable(_model_float(target.observation_mean)) + (
            pt.as_tensor_variable(whitened_inner_design)
            @ (inner_whitened * coordinate_mask)
        )

        outer_whitened: Variable | None
        if outer_count:
            outer_whitened = pm.Normal(
                "outer_whitened",
                mu=0.0,
                sigma=1.0,
                shape=outer_count,
            )
            prediction = prediction + pt.as_tensor_variable(whitened_outer_design) @ outer_whitened
        else:
            outer_whitened = None

        pm.Deterministic(
            "inner_coordinates",
            pt.as_tensor_variable(_model_float(np.sqrt(target.inner_prior_variances)))
            * inner_whitened,
        )
        if outer_whitened is not None:
            pm.Deterministic(
                "outer_coefficients",
                pt.as_tensor_variable(
                    _model_float(np.linalg.cholesky(target.outer_prior_covariance))
                )
                @ outer_whitened,
            )
        pm.MvNormal(
            "observations",
            mu=prediction,
            cov=_model_float(target.observation_covariance),
            observed=_model_float(target.observations),
        )

    return PyMCSplitMaskProductSpaceModel(
        model=model,
        target=target,
        partition_prior=prior,
        split_mask=split_mask,
        inner_whitened=inner_whitened,
        outer_whitened=outer_whitened,
    )


def _split_ancestry_indices(
    layout: TreeContrastLayout,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return child and parent split-mask indices for canonical constraints.

    Args:
        layout: Complete tree contrast layout.

    Returns:
        Parallel integer arrays for every non-root splittable node.
    """
    index_by_node = {node_id: index for index, node_id in enumerate(layout.split_node_ids)}
    child_indices: list[int] = []
    parent_indices: list[int] = []
    for child_id in layout.split_node_ids:
        parent_id = layout.tree.parent(child_id)
        if parent_id is None:
            continue
        child_indices.append(index_by_node[child_id])
        parent_indices.append(index_by_node[parent_id])
    return np.asarray(child_indices, dtype=np.int64), np.asarray(parent_indices, dtype=np.int64)


def _partition_with_region_count(tree: DyadicTree, region_count: int) -> PartitionState:
    """Construct a deterministic valid frontier with exactly ``region_count`` regions.

    Args:
        tree: Complete canonical dyadic tree.
        region_count: Target count between one and the number of leaves.

    Returns:
        Valid frontier obtained by repeatedly splitting the first active
        splittable node in stable order.
    """
    partition = PartitionState.root(tree)
    while len(partition.active) < region_count:
        node_id = next(
            node
            for node in partition.ordered_active()
            if tree.children(node)
        )
        partition = partition.split(tree, node_id)
    return partition


def _binary_mask(
    values: npt.ArrayLike,
    *,
    expected_shape: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return a Boolean mask after exact shape and binary-value checks.

    Args:
        values: Candidate Boolean or integer mask.
        expected_shape: Required one-dimensional shape.

    Returns:
        Copied Boolean mask.

    Raises:
        ValueError: If shape, dtype, or values do not define a binary mask.
    """
    source = np.asarray(values)
    if source.shape != expected_shape:
        raise ValueError(f"split_mask must have shape {expected_shape}.")
    if source.dtype.kind not in "biu" or np.any((source != 0) & (source != 1)):
        raise ValueError("split_mask must contain only binary values.")
    return np.asarray(source, dtype=np.bool_).copy()


def _finite_vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return a finite real floating-point vector."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(source, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _model_float(values: npt.ArrayLike) -> np.ndarray:
    """Cast numeric constants to the configured PyTensor floating dtype."""
    return np.asarray(values, dtype=config.floatX)


__all__ = [
    "DyadicSplitMaskStep",
    "DyadicSplitMaskStepState",
    "PyMCSplitMaskProductSpaceModel",
    "build_pymc_split_mask_product_space_model",
]
