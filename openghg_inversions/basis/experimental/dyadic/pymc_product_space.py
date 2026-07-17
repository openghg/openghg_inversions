"""Native PyMC adapter for the fixed-contrast Gaussian product space.

This experimental adapter keeps a scalar partition index beside fixed-shape
whitened inner and outer coordinates. A custom local split/merge step updates
only the partition index, then native PyMC NUTS updates every continuous
coordinate. The inactive inner coordinates therefore remain part of NUTS; this
is a small exact reference construction, not yet a performance-oriented active-
coordinate sampler.

The implementation is pinned to the custom-step contracts in PyMC 5.25 and
5.26. It deliberately does not integrate with production RHIME sampling.
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

from .gaussian_product_space import GaussianProductSpaceTarget  # noqa: E402
from .product_space import enumerate_partition_neighbors  # noqa: E402
from .state import PartitionState  # noqa: E402
from .tree import DyadicTree  # noqa: E402


@dataclass_state
class DyadicPartitionStepState(StepMethodState):
    """Serializable PyMC sampling state for the parameter-free partition step."""

    tune: bool


@dataclass(frozen=True, slots=True)
class _PartitionEdge:
    """Precomputed local proposal edge between enumerated partitions."""

    candidate_index: int
    log_q_forward: float
    log_q_reverse: float


class DyadicPartitionStep(ArrayStep):
    """Local split/merge MH update for one PyMC partition-index variable.

    The step evaluates PyMC's compiled joint log density at the source and
    proposed indices while every continuous value is held fixed. It has no
    tuning parameters. When composed in the returned order from
    :meth:`PyMCProductSpaceModel.step_methods`, this update runs before NUTS.

    Args:
        vars: One scalar PyMC categorical variable or its value variable.
        tree: Dyadic tree shared by every partition.
        partitions: Distinct valid partitions closed under one-step split and
            merge moves.
        model: PyMC model containing ``vars``.
        rng: Seed or random generator accepted by PyMC.
        compile_kwargs: Optional arguments for :meth:`pymc.Model.compile_logp`.
        blocked: PyMC blocked-step flag. This step always owns one variable.
    """

    name = "dyadic_partition"
    default_blocked = True
    stats_dtypes_shapes: dict[str, tuple[StatDtype, StatShape]] = {
        "accepted": (bool, []),
        "log_acceptance_ratio": (np.float64, []),
        "partition_regions": (np.int64, []),
        "tune": (bool, []),
    }
    _state_class = DyadicPartitionStepState

    def __init__(
        self,
        vars: Variable | Sequence[Variable],
        *,
        tree: DyadicTree,
        partitions: Sequence[PartitionState],
        model: pm.Model | None = None,
        rng: RandomGenerator = None,
        compile_kwargs: Mapping[str, Any] | None = None,
        blocked: bool = True,
    ) -> None:
        """Compile the joint log density and precompute all local proposal edges."""
        if not blocked:
            raise ValueError("DyadicPartitionStep must be blocked.")
        model = pm.modelcontext(model)
        value_vars = get_value_vars_from_user_vars(vars, model)
        if len(value_vars) != 1:
            raise ValueError("DyadicPartitionStep requires exactly one partition-index variable.")
        initial_point = model.initial_point()
        initial_value = np.asarray(initial_point[cast(str, value_vars[0].name)])
        if initial_value.shape != () or initial_value.dtype.kind not in "iu":
            raise ValueError("The partition-index value variable must be a scalar integer.")

        self.tree = tree
        self.partitions = _validated_partitions(tree, partitions)
        self._partition_indices = {partition: index for index, partition in enumerate(self.partitions)}
        self._edges = _partition_edges(tree, self.partitions, self._partition_indices)
        self.tune = True
        kwargs = dict(compile_kwargs or {})
        super().__init__(
            value_vars,
            [model.compile_logp(**kwargs)],
            blocked=blocked,
            rng=rng,
        )

    def astep(self, apoint: RaveledVars, *args: Any) -> tuple[RaveledVars, StatsType]:
        """Propose one neighboring partition and apply the Hastings correction."""
        if apoint.data.size != 1:
            raise ValueError("The raveled partition index must contain exactly one value.")
        current_index = int(apoint.data[0])
        if current_index < 0 or current_index >= len(self.partitions):
            raise ValueError("Current partition index lies outside the configured partitions.")

        edges = self._edges[current_index]
        if not edges:
            return apoint, [self._stats(current_index, accepted=False, log_acceptance_ratio=-math.inf)]

        edge = edges[int(self.rng.integers(len(edges)))]
        candidate_data = apoint.data.copy()
        candidate_data[0] = edge.candidate_index
        candidate = RaveledVars(candidate_data, apoint.point_map_info)
        logp = args[0]
        current_logp = float(logp(apoint))
        candidate_logp = float(logp(candidate))
        if not math.isfinite(current_logp):
            raise ValueError("Current PyMC joint log density must be finite.")
        if math.isnan(candidate_logp) or candidate_logp == math.inf:
            raise ValueError("Candidate PyMC joint log density must be finite or negative infinity.")

        log_acceptance_ratio = candidate_logp - current_logp + edge.log_q_reverse - edge.log_q_forward
        uniform = float(self.rng.random())
        log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
        accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
        selected_index = edge.candidate_index if accepted else current_index
        selected = candidate if accepted else apoint
        return selected, [
            self._stats(
                selected_index,
                accepted=accepted,
                log_acceptance_ratio=log_acceptance_ratio,
            )
        ]

    def reset_tuning(self) -> None:
        """Retain PyMC's no-op tuning contract for this parameter-free step."""
        self.tune = True

    def _stats(
        self,
        partition_index: int,
        *,
        accepted: bool,
        log_acceptance_ratio: float,
    ) -> dict[str, bool | float | int]:
        """Build one PyMC sample-stat record for the selected partition."""
        return {
            "accepted": accepted,
            "log_acceptance_ratio": float(log_acceptance_ratio),
            "partition_regions": len(self.partitions[partition_index].active),
            "tune": self.tune,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PyMCProductSpaceModel:
    """Built PyMC model and variables for one Gaussian product-space target.

    Attributes:
        model: PyMC model with the fixed-shape augmented target.
        target: Framework-independent Gaussian reference target.
        partitions: Stable mapping from sampled integer index to partition.
        partition_index: Sampled categorical partition variable.
        inner_whitened: Permanent standard-normal inner coordinates.
        outer_whitened: Standard-normal outer coordinates, or ``None`` when the
            target has no outer regions.
    """

    model: pm.Model
    target: GaussianProductSpaceTarget
    partitions: tuple[PartitionState, ...]
    partition_index: Variable
    inner_whitened: Variable
    outer_whitened: Variable | None

    def step_methods(
        self,
        *,
        partition_rng: RandomGenerator = None,
        nuts_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[DyadicPartitionStep, pm.NUTS]:
        """Create partition-first compound steps with native PyMC NUTS.

        Args:
            partition_rng: Initial seed or generator for the local partition
                step. PyMC reseeds compound methods when ``pm.sample`` starts;
                pass ``random_seed`` to ``pm.sample`` for reproducible chains.
            nuts_kwargs: Optional arguments forwarded to :class:`pymc.NUTS`.

        Returns:
            ``(partition_step, nuts_step)``. PyMC's ``CompoundStep`` executes
            these in this exact order. Only NUTS adapts during tuning.
        """
        continuous = [self.inner_whitened]
        if self.outer_whitened is not None:
            continuous.append(self.outer_whitened)
        with self.model:
            partition_step = cast(
                DyadicPartitionStep,
                DyadicPartitionStep(
                    self.partition_index,
                    tree=self.target.tree,
                    partitions=self.partitions,
                    model=self.model,
                    rng=partition_rng,
                ),
            )
            nuts_step = cast(pm.NUTS, pm.NUTS(continuous, **dict(nuts_kwargs or {})))
        return partition_step, nuts_step

    def physical_state(
        self,
        partition_index: int,
        inner_whitened: npt.ArrayLike,
        outer_whitened: npt.ArrayLike | None = None,
    ) -> tuple[PartitionState, np.ndarray, np.ndarray]:
        """Map sampled whitened values to framework-independent coordinates.

        Args:
            partition_index: Integer index into :attr:`partitions`.
            inner_whitened: Permanent standard-normal inner vector.
            outer_whitened: Standard-normal outer vector, omitted only for a
                target without outer coefficients.

        Returns:
            Partition, physical inner contrasts, and physical outer
            coefficients.
        """
        if isinstance(partition_index, bool) or not isinstance(partition_index, (int, np.integer)):
            raise TypeError("partition_index must be an integer.")
        index = int(partition_index)
        if index < 0 or index >= len(self.partitions):
            raise ValueError("partition_index lies outside the configured partitions.")
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
        return self.partitions[index], inner_coordinates, outer


def build_pymc_product_space_model(
    target: GaussianProductSpaceTarget,
    partitions: Sequence[PartitionState],
    *,
    initial_partition: PartitionState | None = None,
) -> PyMCProductSpaceModel:
    """Build the fixed-shape PyMC form of a Gaussian product-space target.

    The current reference requires inactive pseudo-prior variances to equal the
    corresponding active prior variances. This makes whitening independent of
    the partition: a partition update changes only the active likelihood
    design, not the meaning or order of any continuous coordinate.

    Args:
        target: Validated Gaussian reference target.
        partitions: Distinct valid partitions closed under local split/merge
            moves.
        initial_partition: Optional initial partition. Defaults to the first
            supplied state.

    Returns:
        Built model and variables ready for :meth:`PyMCProductSpaceModel.step_methods`.

    Raises:
        TypeError: If ``target`` is not a Gaussian product-space target.
        ValueError: If pseudo-prior whitening is partition dependent, the
            partition list is invalid, or every partition has zero prior mass.
    """
    if not isinstance(target, GaussianProductSpaceTarget):
        raise TypeError("target must be a GaussianProductSpaceTarget.")
    if not np.array_equal(
        target.inactive_pseudo_prior_variances,
        target.inner_prior_variances,
    ):
        raise ValueError(
            "The initial PyMC adapter requires inactive pseudo-prior variances "
            "to equal active prior variances."
        )
    states = _validated_partitions(target.tree, partitions)
    partition_indices = {partition: index for index, partition in enumerate(states)}
    _partition_edges(target.tree, states, partition_indices)
    partition_log_weights = _partition_log_weights(target, states)

    if initial_partition is None:
        initial_index = int(np.flatnonzero(np.isfinite(partition_log_weights))[0])
    else:
        try:
            initial_index = partition_indices[initial_partition]
        except KeyError as error:
            raise ValueError("initial_partition must be one of the supplied partitions.") from error
        if not np.isfinite(partition_log_weights[initial_index]):
            raise ValueError("initial_partition must have positive prior mass.")

    whitened_inner_design = _model_float(_whitened_inner_design(target, states))
    outer_count = target.outer_design.shape[1]
    if outer_count:
        whitened_outer_design = _model_float(
            target.outer_design @ np.linalg.cholesky(target.outer_prior_covariance)
        )
    else:
        whitened_outer_design = _model_float(np.empty((target.observations.size, 0), dtype=float))

    with pm.Model() as model:
        partition_index = pm.Categorical(
            "partition_index",
            p=_model_float(np.full(len(states), 1.0 / len(states))),
            initval=initial_index,
        )
        pm.Potential(
            "partition_prior",
            pt.as_tensor_variable(partition_log_weights)[partition_index],
        )
        inner_whitened = pm.Normal(
            "inner_whitened",
            mu=0.0,
            sigma=1.0,
            shape=target.contrast_layout.coordinate_count,
        )
        selected_inner_design = pt.take(
            pt.as_tensor_variable(whitened_inner_design),
            partition_index,
            axis=0,
        )
        prediction = pt.as_tensor_variable(_model_float(target.observation_mean)) + (
            selected_inner_design @ inner_whitened
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
            "basis_region_count",
            pt.as_tensor_variable(np.array([len(state.active) for state in states], dtype=np.int64))[
                partition_index
            ],
        )
        pm.Deterministic(
            "inner_coordinates",
            pt.as_tensor_variable(_model_float(np.sqrt(target.inner_prior_variances))) * inner_whitened,
        )
        if outer_whitened is not None:
            pm.Deterministic(
                "outer_coefficients",
                pt.as_tensor_variable(_model_float(np.linalg.cholesky(target.outer_prior_covariance)))
                @ outer_whitened,
            )
        pm.MvNormal(
            "observations",
            mu=prediction,
            cov=_model_float(target.observation_covariance),
            observed=_model_float(target.observations),
        )

    return PyMCProductSpaceModel(
        model=model,
        target=target,
        partitions=states,
        partition_index=partition_index,
        inner_whitened=inner_whitened,
        outer_whitened=outer_whitened,
    )


def _validated_partitions(
    tree: DyadicTree,
    partitions: Sequence[PartitionState],
) -> tuple[PartitionState, ...]:
    """Return distinct valid partitions without changing caller order."""
    if not isinstance(tree, DyadicTree):
        raise TypeError("tree must be a DyadicTree.")
    states = tuple(partitions)
    if not states:
        raise ValueError("partitions must not be empty.")
    if len(set(states)) != len(states):
        raise ValueError("partitions must be distinct.")
    for state in states:
        if not isinstance(state, PartitionState):
            raise TypeError("partitions must contain only PartitionState values.")
        state.validate(tree)
    return states


def _partition_edges(
    tree: DyadicTree,
    partitions: tuple[PartitionState, ...],
    indices: Mapping[PartitionState, int],
) -> tuple[tuple[_PartitionEdge, ...], ...]:
    """Precompute local proposal indices and both Hastings probabilities."""
    neighbor_rows = tuple(enumerate_partition_neighbors(tree, partition) for partition in partitions)
    rows: list[tuple[_PartitionEdge, ...]] = []
    for source_index, neighbors in enumerate(neighbor_rows):
        edges: list[_PartitionEdge] = []
        for neighbor in neighbors:
            try:
                candidate_index = indices[neighbor.partition]
            except KeyError as error:
                raise ValueError("partitions must be closed under local split and merge moves.") from error
            reverse = next(
                (
                    item
                    for item in neighbor_rows[candidate_index]
                    if item.partition == partitions[source_index]
                ),
                None,
            )
            if reverse is None:  # pragma: no cover - framework kernel invariant.
                raise ValueError("Every local partition proposal must have a reverse edge.")
            edges.append(
                _PartitionEdge(
                    candidate_index=candidate_index,
                    log_q_forward=neighbor.log_q,
                    log_q_reverse=reverse.log_q,
                )
            )
        rows.append(tuple(edges))
    return tuple(rows)


def _partition_log_weights(
    target: GaussianProductSpaceTarget,
    partitions: tuple[PartitionState, ...],
) -> np.ndarray:
    """Return shifted partition log weights in the configured model dtype."""
    log_prior = np.array([float(target.partition_log_prior(state)) for state in partitions])
    if np.any(np.isnan(log_prior)) or np.any(np.isposinf(log_prior)):
        raise ValueError("partition_log_prior must return finite values or negative infinity.")
    if np.all(np.isneginf(log_prior)):
        raise ValueError("At least one partition must have positive prior mass.")
    finite_prior = np.isfinite(log_prior)
    finite_maximum = float(log_prior[np.isfinite(log_prior)].max())
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = log_prior - finite_maximum
    if np.any(finite_prior & ~np.isfinite(shifted)):
        raise ValueError("partition_log_prior exceeds the configured model dtype range.")
    model_weights = _model_float(shifted)
    if np.any(finite_prior & ~np.isfinite(model_weights)):
        raise ValueError("partition_log_prior exceeds the configured model dtype range.")
    return model_weights


def _whitened_inner_design(
    target: GaussianProductSpaceTarget,
    partitions: tuple[PartitionState, ...],
) -> np.ndarray:
    """Build one full fixed-coordinate likelihood design per partition."""
    result = np.zeros(
        (
            len(partitions),
            target.observations.size,
            target.contrast_layout.coordinate_count,
        ),
        dtype=float,
    )
    standard_deviations = np.sqrt(target.inner_prior_variances)
    for partition_index, partition in enumerate(partitions):
        active_design, active_indices = target.active_design(partition)
        inner_active = active_design[:, : len(active_indices)]
        result[partition_index][:, list(active_indices)] = (
            inner_active * standard_deviations[list(active_indices)]
        )
    return result


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
    "DyadicPartitionStep",
    "DyadicPartitionStepState",
    "PyMCProductSpaceModel",
    "build_pymc_product_space_model",
]
