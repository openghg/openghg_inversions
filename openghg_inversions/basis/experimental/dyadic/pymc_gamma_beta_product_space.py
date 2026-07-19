"""Non-enumerating PyMC product space for positive Gamma--Beta forests.

The model keeps a permanent root Gamma coordinate for every stochastic semantic
group and a permanent Beta allocation coordinate for every possible forest
split.  A canonical Bernoulli split mask selects one active frontier without
changing the continuous dimension or rebuilding the observation design.

Inactive Beta coordinates retain their normalized Gamma--Beta prior.  Because
the prior is projectively consistent, these are natural product-space
pseudo-priors: integrating coordinates below an inactive node leaves the parent
model unchanged.  A custom local Metropolis--Hastings step updates the partition
mask, and native PyMC NUTS updates all permanent positive coordinates.

This remains an experimental Gaussian-likelihood prototype and is not wired to
production RHIME entry points.
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

from .gamma_beta_partition import (  # noqa: E402
    GammaBetaPartitionLayout,
    GammaBetaRegionCountPrior,
)
from .gamma_beta_product_space import GammaBetaProductSpaceTarget  # noqa: E402


@dataclass_state
class GammaBetaSplitMaskStepState(StepMethodState):
    """Serializable PyMC state for the parameter-free partition step."""

    tune: bool


class GammaBetaSplitMaskStep(ArrayStep):
    """Local split/merge/swap Metropolis--Hastings update for a forest mask.

    The proposal selects uniformly among valid one-split, one-merge, and
    optionally fixed-K merge-then-split neighbors. Continuous Gamma and Beta
    coordinates remain fixed during this block. Source and reverse neighbor
    counts provide the Hastings correction.

    Args:
        vars: One vector PyMC Bernoulli variable or its value variable.
        layout: Fixed Gamma-Beta forest partition layout.
        model: PyMC model containing ``vars``.
        rng: Seed or generator accepted by PyMC.
        compile_kwargs: Optional arguments for :meth:`pymc.Model.compile_logp`.
        include_swap_moves: Include fixed-K merge/split neighbors.
        blocked: PyMC blocked-step flag.  This step always owns one variable.
    """

    name = "gamma_beta_split_mask"
    default_blocked = True
    stats_dtypes_shapes: dict[str, tuple[StatDtype, StatShape]] = {
        "accepted": (bool, []),
        "log_acceptance_ratio": (np.float64, []),
        "partition_regions": (np.int64, []),
        "proposal_degree": (np.int64, []),
        "proposed_split": (bool, []),
        "proposed_swap": (bool, []),
        "reverse_degree": (np.int64, []),
        "tune": (bool, []),
    }
    _state_class = GammaBetaSplitMaskStepState

    def __init__(
        self,
        vars: Variable | Sequence[Variable],
        *,
        layout: GammaBetaPartitionLayout,
        model: pm.Model | None = None,
        rng: RandomGenerator = None,
        compile_kwargs: Mapping[str, Any] | None = None,
        include_swap_moves: bool = True,
        blocked: bool = True,
    ) -> None:
        """Validate the mask variable and compile the full joint density."""
        if not blocked:
            raise ValueError("GammaBetaSplitMaskStep must be blocked.")
        if not isinstance(include_swap_moves, bool):
            raise TypeError("include_swap_moves must be Boolean.")
        if not isinstance(layout, GammaBetaPartitionLayout):
            raise TypeError("layout must be a GammaBetaPartitionLayout.")
        if not layout.split_node_ids:
            raise ValueError("GammaBetaSplitMaskStep requires at least one possible split.")

        model = pm.modelcontext(model)
        value_vars = get_value_vars_from_user_vars(vars, model)
        if len(value_vars) != 1:
            raise ValueError("GammaBetaSplitMaskStep requires exactly one mask variable.")
        initial_point = model.initial_point()
        initial_value = np.asarray(initial_point[cast(str, value_vars[0].name)])
        expected_shape = (layout.split_count,)
        if initial_value.shape != expected_shape or initial_value.dtype.kind not in "iu":
            raise ValueError(
                "The split-mask value variable must be an integer vector with "
                f"shape {expected_shape}."
            )
        layout.canonical_split_mask(initial_value)

        self.layout = layout
        self.include_swap_moves = include_swap_moves
        self.tune = True
        super().__init__(
            value_vars,
            [model.compile_logp(**dict(compile_kwargs or {}))],
            blocked=blocked,
            rng=rng,
        )

    def astep(self, apoint: RaveledVars, *args: Any) -> tuple[RaveledVars, StatsType]:
        """Propose one legal neighboring mask and accept or reject it."""
        current_mask = self.layout.canonical_split_mask(apoint.data)
        neighbors = self.layout.neighbors(
            current_mask,
            include_swaps=self.include_swap_moves,
        )
        if not neighbors:  # pragma: no cover - constructor rejects no-split forests.
            return apoint, [
                self._stats(
                    self.layout.region_count(current_mask),
                    accepted=False,
                    log_acceptance_ratio=-math.inf,
                    proposal_degree=0,
                    proposed_split=False,
                    proposed_swap=False,
                    reverse_degree=0,
                )
            ]

        move = neighbors[int(self.rng.integers(len(neighbors)))]
        candidate_data = move.split_mask.astype(apoint.data.dtype, copy=False)
        candidate = RaveledVars(candidate_data, apoint.point_map_info)
        reverse_neighbors = self.layout.neighbors(
            move.split_mask,
            include_swaps=self.include_swap_moves,
        )
        reverse = next(
            (
                candidate_move
                for candidate_move in reverse_neighbors
                if np.array_equal(candidate_move.split_mask, current_mask)
            ),
            None,
        )
        if reverse is None:  # pragma: no cover - local kernel invariant.
            raise ValueError("Every local Gamma-Beta partition move must be reversible.")

        logp = args[0]
        current_logp = float(logp(apoint))
        candidate_logp = float(logp(candidate))
        if not math.isfinite(current_logp):
            raise ValueError("Current PyMC joint log density must be finite.")
        if math.isnan(candidate_logp) or candidate_logp == math.inf:
            raise ValueError(
                "Candidate PyMC joint log density must be finite or negative infinity."
            )

        log_acceptance_ratio = (
            candidate_logp - current_logp + reverse.log_q - move.log_q
        )
        uniform = float(self.rng.random())
        log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
        accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
        selected = candidate if accepted else apoint
        selected_mask = move.split_mask if accepted else current_mask
        return selected, [
            self._stats(
                self.layout.region_count(selected_mask),
                accepted=accepted,
                log_acceptance_ratio=log_acceptance_ratio,
                proposal_degree=len(neighbors),
                proposed_split=move.kind == "split",
                proposed_swap=move.kind == "swap",
                reverse_degree=len(reverse_neighbors),
            )
        ]

    def reset_tuning(self) -> None:
        """Retain PyMC's no-op tuning contract for this discrete step."""
        self.tune = True

    def _stats(
        self,
        partition_regions: int,
        *,
        accepted: bool,
        log_acceptance_ratio: float,
        proposal_degree: int,
        proposed_split: bool,
        proposed_swap: bool,
        reverse_degree: int,
    ) -> dict[str, bool | float | int]:
        """Build one sample-stat record for the selected partition.

        Args:
            partition_regions: Region count after accept/reject selection.
            accepted: Whether the candidate was accepted.
            log_acceptance_ratio: Unclipped Metropolis--Hastings log ratio.
            proposal_degree: Number of source neighbors.
            proposed_split: Whether the proposed move was a split.
            proposed_swap: Whether the proposed move was a fixed-K swap.
            reverse_degree: Number of candidate-state neighbors.

        Returns:
            Scalar statistics matching :attr:`stats_dtypes_shapes`.
        """
        return {
            "accepted": accepted,
            "log_acceptance_ratio": float(log_acceptance_ratio),
            "partition_regions": partition_regions,
            "proposal_degree": proposal_degree,
            "proposed_split": proposed_split,
            "proposed_swap": proposed_swap,
            "reverse_degree": reverse_degree,
            "tune": self.tune,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PyMCGammaBetaProductSpaceModel:
    """Built non-enumerating PyMC positive product-space model.

    Attributes:
        model: Static-shape PyMC model.
        target: Framework-independent numerical target.
        partition_prior: Exact symbolic forest partition prior.
        split_mask: Canonical binary partition variable.
        stochastic_group_root_scalings: Positive group-root variable, or
            ``None`` when every group root is fixed at one.
        split_fractions: Permanent Beta allocation vector.
        fixed_split_mask: Exact point-mass partition, or ``None`` for latent P.
    """

    model: pm.Model
    target: GammaBetaProductSpaceTarget
    partition_prior: GammaBetaRegionCountPrior
    split_mask: Variable
    stochastic_group_root_scalings: Variable | None
    split_fractions: Variable
    fixed_split_mask: npt.NDArray[np.bool_] | None

    def step_methods(
        self,
        *,
        partition_rng: RandomGenerator = None,
        include_swap_moves: bool = True,
        nuts_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[GammaBetaSplitMaskStep, pm.NUTS]:
        """Create partition-first compound steps with native PyMC NUTS.

        Args:
            partition_rng: Seed or generator for structural proposals.
            include_swap_moves: Include fixed-K merge/split proposals.
            nuts_kwargs: Optional arguments forwarded to :class:`pymc.NUTS`.

        Returns:
            ``(split_mask_step, nuts_step)`` in execution order.
        """
        continuous: list[Variable] = []
        if self.stochastic_group_root_scalings is not None:
            continuous.append(self.stochastic_group_root_scalings)
        continuous.append(self.split_fractions)
        with self.model:
            partition_step = cast(
                GammaBetaSplitMaskStep,
                GammaBetaSplitMaskStep(
                    self.split_mask,
                    layout=self.partition_prior.layout,
                    model=self.model,
                    rng=partition_rng,
                    include_swap_moves=include_swap_moves,
                ),
            )
            nuts_step = cast(pm.NUTS, pm.NUTS(continuous, **dict(nuts_kwargs or {})))
        return partition_step, nuts_step

    def physical_state(
        self,
        split_mask: npt.ArrayLike,
        stochastic_group_root_scalings: npt.ArrayLike,
        split_fractions: npt.ArrayLike,
    ) -> tuple[
        tuple[int, ...],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Map sampled free variables to one active positive forest state.

        Args:
            split_mask: Canonical partition mask.
            stochastic_group_root_scalings: Positive values in the target
                layout's stochastic-group order.  Supply an empty vector when
                all roots are fixed.
            split_fractions: Permanent Beta fractions in split-node order.

        Returns:
            Active node IDs, full group-root vector, and all node scalings.

        Raises:
            ValueError: If free root or split vectors are invalid.
        """
        partition_layout = self.partition_prior.layout
        active = partition_layout.active_node_ids(split_mask)
        coordinate_layout = self.target.coordinate_layout
        free_roots = _finite_vector(
            stochastic_group_root_scalings,
            name="stochastic_group_root_scalings",
        )
        expected_shape = (len(coordinate_layout.stochastic_group_indices),)
        if free_roots.shape != expected_shape or np.any(free_roots <= 0.0):
            raise ValueError(
                "stochastic_group_root_scalings must be positive with shape "
                f"{expected_shape}."
            )
        group_roots = np.ones(len(coordinate_layout.forest.groups), dtype=np.float64)
        group_roots[list(coordinate_layout.stochastic_group_indices)] = free_roots
        node_scalings = coordinate_layout.node_scalings(group_roots, split_fractions)
        return active, group_roots, node_scalings


def build_pymc_gamma_beta_product_space_model(
    target: GammaBetaProductSpaceTarget,
    partition_prior: GammaBetaRegionCountPrior,
    *,
    initial_split_mask: npt.ArrayLike | None = None,
    fixed_split_mask: npt.ArrayLike | None = None,
) -> PyMCGammaBetaProductSpaceModel:
    """Build a fixed-shape positive Gamma--Beta product-space model.

    Args:
        target: Validated framework-independent observation target.
        partition_prior: Exact normalized forest partition prior.
        initial_split_mask: Optional positive-prior canonical mask.  The
            default is a deterministic mask at the smallest supported K.
        fixed_split_mask: Optional exact point-mass partition. When supplied,
            every other mask has zero probability in the built model.

    Returns:
        Built PyMC model and variables ready for compound sampling.

    Raises:
        TypeError: If target or prior has the wrong type.
        ValueError: If forests differ, no split is available, or the initial
            partition has zero prior mass.
    """
    if not isinstance(target, GammaBetaProductSpaceTarget):
        raise TypeError("target must be a GammaBetaProductSpaceTarget.")
    if not isinstance(partition_prior, GammaBetaRegionCountPrior):
        raise TypeError("partition_prior must be a GammaBetaRegionCountPrior.")
    layout = partition_prior.layout
    coordinate_layout = target.coordinate_layout
    if layout.forest is not coordinate_layout.forest:
        raise ValueError("partition_prior and target must use the same forest instance.")
    if not layout.split_count:
        raise ValueError("The Gamma-Beta product space requires at least one possible split.")

    fixed_mask = (
        None
        if fixed_split_mask is None
        else layout.canonical_split_mask(fixed_split_mask)
    )
    if fixed_mask is not None and not math.isfinite(partition_prior(fixed_mask)):
        raise ValueError("fixed_split_mask must have positive prior mass.")

    if initial_split_mask is None and fixed_mask is not None:
        initial_mask = fixed_mask
    elif initial_split_mask is None:
        supported_k = np.flatnonzero(np.isfinite(partition_prior.log_probability_by_k))
        initial_mask = layout.initial_split_mask(int(supported_k[0]))
    else:
        initial_mask = layout.canonical_split_mask(initial_split_mask)
    if fixed_mask is not None and not np.array_equal(initial_mask, fixed_mask):
        raise ValueError("initial_split_mask must equal fixed_split_mask when both are supplied.")
    if not math.isfinite(partition_prior(initial_mask)):
        raise ValueError("initial_split_mask must have positive prior mass.")

    child_indices, parent_indices = _split_ancestry_indices(layout)
    ancestor_matrix = _model_float(
        coordinate_layout.left_path + coordinate_layout.right_path
    )
    split_index_by_node = np.maximum(layout.split_index_by_node, 0)
    splittable_node = _model_float(layout.split_index_by_node >= 0)

    stochastic_indices = np.asarray(
        coordinate_layout.stochastic_group_indices,
        dtype=np.int64,
    )
    stochastic_variances = np.asarray(
        [
            layout.forest.groups[group_index].root_variance
            for group_index in stochastic_indices
        ],
        dtype=np.float64,
    )
    root_shapes = 1.0 / stochastic_variances
    expected_fractions = coordinate_layout.expected_fraction_by_split
    kappas = coordinate_layout.kappa_by_split
    first_shapes = kappas * expected_fractions
    second_shapes = kappas * (1.0 - expected_fractions)

    with pm.Model() as model:
        split_mask = pm.Bernoulli(
            "split_mask",
            p=0.5,
            shape=layout.split_count,
            initval=initial_mask.astype(np.int8),
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
        pm.Potential(
            "split_mask_base_measure",
            cast(
                Any,
                pt.as_tensor_variable(
                    _model_float(layout.split_count * math.log(2.0))
                ),
            ),
        )
        basis_region_count = pm.Deterministic(
            "basis_region_count",
            layout.minimum_regions + split_count,
        )
        pm.Potential(
            "partition_prior",
            pt.as_tensor_variable(_model_float(partition_prior.log_probability_by_k))[
                basis_region_count
            ],
        )
        if fixed_mask is not None:
            fixed_partition = pt.all(
                pt.eq(
                    split_mask_tensor,
                    pt.as_tensor_variable(fixed_mask.astype(np.int8)),
                )
            )
            pm.Potential(
                "fixed_partition",
                cast(
                    Any,
                    pt.switch(
                        fixed_partition,
                        _model_float(-partition_prior(fixed_mask)),
                        _model_float(-math.inf),
                    ),
                ),
            )

        stochastic_group_root_scalings: Variable | None
        if stochastic_indices.size:
            stochastic_group_root_scalings = pm.Gamma(
                "stochastic_group_root_scalings",
                alpha=_model_float(root_shapes),
                beta=_model_float(root_shapes),
                shape=stochastic_indices.size,
            )
            group_roots: Any = pt.ones(
                (len(layout.forest.groups),),
                dtype=config.floatX,
            )
            group_roots = pt.set_subtensor(
                group_roots[stochastic_indices],
                stochastic_group_root_scalings,
            )
        else:
            stochastic_group_root_scalings = None
            group_roots = pt.ones(
                (len(layout.forest.groups),),
                dtype=config.floatX,
            )

        split_fractions = pm.Beta(
            "split_fractions",
            alpha=_model_float(first_shapes),
            beta=_model_float(second_shapes),
            shape=layout.split_count,
        )
        split_fractions_tensor: Any = split_fractions
        first_log_ratio = cast(Any, pt.log(split_fractions_tensor)) - cast(
            Any,
            pt.log(pt.as_tensor_variable(_model_float(expected_fractions))),
        )
        second_log_ratio = cast(Any, pt.log1p(-split_fractions_tensor)) - cast(
            Any,
            pt.log1p(-pt.as_tensor_variable(_model_float(expected_fractions))),
        )
        node_scalings = pm.Deterministic(
            "node_scalings",
            pt.exp(
                pt.log(group_roots[coordinate_layout.group_index_by_node])
                + pt.as_tensor_variable(_model_float(coordinate_layout.left_path))
                @ first_log_ratio
                + pt.as_tensor_variable(_model_float(coordinate_layout.right_path))
                @ second_log_ratio
            ),
        )

        mask_float = pt.cast(split_mask_tensor, config.floatX)
        reached = pt.prod(
            1.0
            - pt.as_tensor_variable(ancestor_matrix)
            + pt.as_tensor_variable(ancestor_matrix) * mask_float,
            axis=1,
        )
        unsplit = 1.0 - pt.as_tensor_variable(splittable_node) * mask_float[
            split_index_by_node
        ]
        active_node_mask = pm.Deterministic(
            "active_node_mask",
            reached * unsplit,
        )
        prediction = pm.Deterministic(
            "observation_prediction",
            pt.as_tensor_variable(_model_float(target.observation_mean))
            + pt.as_tensor_variable(_model_float(target.node_design))
            @ (node_scalings * active_node_mask),
        )
        observation_sd = _diagonal_standard_deviations(
            target.observation_covariance
        )
        if observation_sd is None:
            pm.MvNormal(
                "observations",
                mu=prediction,
                cov=_model_float(target.observation_covariance),
                observed=_model_float(target.observations),
            )
        else:
            pm.Normal(
                "observations",
                mu=prediction,
                sigma=_model_float(observation_sd),
                observed=_model_float(target.observations),
            )

    return PyMCGammaBetaProductSpaceModel(
        model=model,
        target=target,
        partition_prior=partition_prior,
        split_mask=split_mask,
        stochastic_group_root_scalings=stochastic_group_root_scalings,
        split_fractions=split_fractions,
        fixed_split_mask=fixed_mask,
    )


def _split_ancestry_indices(
    layout: GammaBetaPartitionLayout,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return child and parent mask indices for canonical constraints.

    Args:
        layout: Compiled Gamma-Beta partition layout.

    Returns:
        Parallel integer arrays for every non-root internal node.
    """
    child_indices: list[int] = []
    parent_indices: list[int] = []
    for node_id in layout.split_node_ids:
        parent_id = layout.forest.nodes[node_id].parent_id
        if parent_id is None:
            continue
        parent_index = layout.split_index_by_node[parent_id]
        if parent_index >= 0:
            child_indices.append(int(layout.split_index_by_node[node_id]))
            parent_indices.append(int(parent_index))
    return np.asarray(child_indices, dtype=np.int64), np.asarray(parent_indices, dtype=np.int64)


def _finite_vector(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Return one finite real floating-point vector."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(source, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _model_float(values: npt.ArrayLike | float) -> npt.NDArray[np.floating[Any]]:
    """Cast numeric constants to the configured PyTensor floating dtype."""
    return np.asarray(values, dtype=config.floatX)


def _diagonal_standard_deviations(
    covariance: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """Return diagonal standard deviations, or ``None`` for full covariance."""
    diagonal = np.diag(covariance)
    if not np.array_equal(covariance, np.diag(diagonal)):
        return None
    return np.sqrt(diagonal)


__all__ = [
    "GammaBetaSplitMaskStep",
    "GammaBetaSplitMaskStepState",
    "PyMCGammaBetaProductSpaceModel",
    "build_pymc_gamma_beta_product_space_model",
]
