"""Vectorized permanent coordinates for grouped Gamma--Beta forests.

The existing :class:`~.gamma_beta.GammaBetaForest` samples node scalings in a
clear top-down loop. Product-space inference needs the same transformation in
a fixed symbolic graph with hundreds of possible splits. This module compiles
the recursion into static left/right path matrices so every node scaling can be
evaluated from one group-root vector and one permanent Beta-fraction vector.

It also aggregates a finest-grid observation design into one static column per
forest node. Partition masks can then activate a frontier without rebuilding
the design or changing the continuous coordinate dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from .gamma_beta import GammaBetaForest, KappaStrategy


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaCoordinateLayout:
    """Static vectorization metadata for one grouped Gamma--Beta forest.

    Construct layouts with :meth:`from_forest`. Internal split coordinates are
    ordered by node ID. Rows of the path matrices are forest nodes and columns
    are split coordinates.

    Attributes:
        forest: Immutable grouped forest described by the layout.
        internal_node_ids: Internal forest nodes in stable coordinate order.
        stochastic_group_indices: Groups whose root variance is positive.
        group_index_by_node: Semantic group index for every forest node.
        expected_fraction_by_split: Expected first-child mass fraction per
            internal node.
        kappa_by_split: Positive Beta concentration per internal node.
        left_path: Indicator that a node descends through the first child of a
            split ancestor.
        right_path: Indicator that a node descends through the second child of
            a split ancestor.
    """

    forest: GammaBetaForest
    internal_node_ids: tuple[int, ...]
    stochastic_group_indices: tuple[int, ...]
    group_index_by_node: npt.NDArray[np.int64]
    expected_fraction_by_split: npt.NDArray[np.float64]
    kappa_by_split: npt.NDArray[np.float64]
    left_path: npt.NDArray[np.int8]
    right_path: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        """Validate dimensions and freeze every numeric array."""
        if not isinstance(self.forest, GammaBetaForest):
            raise TypeError("forest must be a GammaBetaForest.")
        expected_internal = tuple(
            node.node_id for node in self.forest.nodes if node.child_ids
        )
        if self.internal_node_ids != expected_internal:
            raise ValueError("internal_node_ids must contain all internal nodes in node order.")
        expected_stochastic = tuple(
            index
            for index, group in enumerate(self.forest.groups)
            if group.root_variance > 0.0
        )
        if self.stochastic_group_indices != expected_stochastic:
            raise ValueError(
                "stochastic_group_indices must contain every positive-variance group."
            )

        node_count = len(self.forest.nodes)
        split_count = len(self.internal_node_ids)
        arrays = {
            "group_index_by_node": (
                self.group_index_by_node,
                (node_count,),
                np.int64,
            ),
            "expected_fraction_by_split": (
                self.expected_fraction_by_split,
                (split_count,),
                np.float64,
            ),
            "kappa_by_split": (
                self.kappa_by_split,
                (split_count,),
                np.float64,
            ),
            "left_path": (
                self.left_path,
                (node_count, split_count),
                np.int8,
            ),
            "right_path": (
                self.right_path,
                (node_count, split_count),
                np.int8,
            ),
        }
        for name, (source, shape, dtype) in arrays.items():
            values = np.asarray(source, dtype=dtype)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if name in {"left_path", "right_path"} and np.any(
                (values != 0) & (values != 1)
            ):
                raise ValueError(f"{name} must contain only zero and one.")
            if name not in {"left_path", "right_path", "group_index_by_node"}:
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{name} must contain only finite values.")
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

        if np.any(self.group_index_by_node < 0) or np.any(
            self.group_index_by_node >= len(self.forest.groups)
        ):
            raise ValueError("group_index_by_node contains an unknown group.")
        if np.any(self.expected_fraction_by_split <= 0.0) or np.any(
            self.expected_fraction_by_split >= 1.0
        ):
            raise ValueError("expected_fraction_by_split must lie strictly between zero and one.")
        if np.any(self.kappa_by_split <= 0.0):
            raise ValueError("kappa_by_split must be positive.")
        if np.any(self.left_path & self.right_path):
            raise ValueError("A node cannot take both branches of one split ancestor.")

    @classmethod
    def from_forest(
        cls,
        forest: GammaBetaForest,
        *,
        kappa_strategy: KappaStrategy,
    ) -> GammaBetaCoordinateLayout:
        """Compile vectorized path and prior metadata from a forest.

        Args:
            forest: Fixed maximum Gamma-Beta forest.
            kappa_strategy: Concentration policy evaluated once per possible
                split.

        Returns:
            Immutable coordinate layout for NumPy or symbolic evaluation.

        Raises:
            TypeError: If ``forest`` has the wrong type.
            ValueError: If a concentration is not positive and finite.
        """
        if not isinstance(forest, GammaBetaForest):
            raise TypeError("forest must be a GammaBetaForest.")
        internal_node_ids = tuple(node.node_id for node in forest.nodes if node.child_ids)
        split_index_by_node = {
            node_id: index for index, node_id in enumerate(internal_node_ids)
        }
        left_path = np.zeros(
            (len(forest.nodes), len(internal_node_ids)),
            dtype=np.int8,
        )
        right_path = np.zeros_like(left_path)

        for node in forest.nodes:
            descendant_id = node.node_id
            parent_id = node.parent_id
            while parent_id is not None:
                parent = forest.nodes[parent_id]
                split_index = split_index_by_node[parent_id]
                first_id, second_id = parent.child_ids
                if descendant_id == first_id:
                    left_path[node.node_id, split_index] = 1
                elif descendant_id == second_id:
                    right_path[node.node_id, split_index] = 1
                else:  # pragma: no cover - immutable forest topology invariant.
                    raise ValueError("Forest parent/child topology is inconsistent.")
                descendant_id = parent_id
                parent_id = parent.parent_id

        expected_fractions = np.empty(len(internal_node_ids), dtype=np.float64)
        kappas = np.empty(len(internal_node_ids), dtype=np.float64)
        for split_index, node_id in enumerate(internal_node_ids):
            context = forest.split_context(node_id)
            first_mass, second_mass = context.child_expected_masses
            expected_fractions[split_index] = first_mass / (first_mass + second_mass)
            kappa = float(kappa_strategy(context))
            if not math.isfinite(kappa) or kappa <= 0.0:
                raise ValueError(
                    f"Kappa strategy returned invalid value {kappa!r} for node {node_id}."
                )
            kappas[split_index] = kappa

        return cls(
            forest=forest,
            internal_node_ids=internal_node_ids,
            stochastic_group_indices=tuple(
                index
                for index, group in enumerate(forest.groups)
                if group.root_variance > 0.0
            ),
            group_index_by_node=np.fromiter(
                (node.group_index for node in forest.nodes),
                dtype=np.int64,
                count=len(forest.nodes),
            ),
            expected_fraction_by_split=expected_fractions,
            kappa_by_split=kappas,
            left_path=left_path,
            right_path=right_path,
        )

    @property
    def split_count(self) -> int:
        """Return the permanent number of possible Beta splits."""
        return len(self.internal_node_ids)

    def node_scalings(
        self,
        group_root_scalings: npt.ArrayLike,
        split_fractions: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Evaluate every node scaling through vectorized path products.

        Args:
            group_root_scalings: Positive mean-one root scalings with shape
                ``(group,)``. Fixed zero-variance groups must equal one.
            split_fractions: Fractions in ``(0, 1)`` with shape
                ``(possible_split,)``.

        Returns:
            Positive scaling for every forest node in node order.

        Raises:
            ValueError: If a vector has the wrong shape or invalid values.
        """
        roots = _finite_vector(
            group_root_scalings,
            expected_shape=(len(self.forest.groups),),
            name="group_root_scalings",
        )
        if np.any(roots <= 0.0):
            raise ValueError("group_root_scalings must be positive.")
        fixed_groups = tuple(
            index
            for index, group in enumerate(self.forest.groups)
            if group.root_variance == 0.0
        )
        if fixed_groups and not np.array_equal(
            roots[list(fixed_groups)],
            np.ones(len(fixed_groups)),
        ):
            raise ValueError("Zero-variance group root scalings must equal one.")

        fractions = _finite_vector(
            split_fractions,
            expected_shape=(self.split_count,),
            name="split_fractions",
        )
        if np.any(fractions <= 0.0) or np.any(fractions >= 1.0):
            raise ValueError("split_fractions must lie strictly between zero and one.")

        first_log_ratio = np.log(fractions) - np.log(
            self.expected_fraction_by_split
        )
        second_log_ratio = np.log1p(-fractions) - np.log1p(
            -self.expected_fraction_by_split
        )
        log_scalings = (
            np.log(roots[self.group_index_by_node])
            + self.left_path @ first_log_ratio
            + self.right_path @ second_log_ratio
        )
        scalings = np.exp(log_scalings)
        if not np.all(np.isfinite(scalings)) or np.any(scalings <= 0.0):
            raise ValueError("Gamma-Beta coordinates produced invalid node scalings.")
        return scalings

    def node_design(self, finest_grid_design: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Aggregate a finest-grid design into one column per forest node.

        Args:
            finest_grid_design: Finite values with shape
                ``(observation, row, column)`` matching :attr:`forest`.

        Returns:
            Static matrix with shape ``(observation, forest_node)``. A column
            is the sum over the hard support of that node.

        Raises:
            ValueError: If shape, dtype, or values are invalid.
        """
        source = np.asarray(finest_grid_design)
        if np.iscomplexobj(source):
            raise ValueError("finest_grid_design must be real-valued.")
        values = np.asarray(source, dtype=np.float64)
        expected_shape = (
            (values.shape[0], *self.forest.shape) if values.ndim == 3 else None
        )
        if values.ndim != 3 or values.shape != expected_shape:
            raise ValueError(
                "finest_grid_design must have shape "
                f"(observation, {self.forest.shape[0]}, {self.forest.shape[1]})."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("finest_grid_design must contain only finite values.")

        flattened = values.reshape(values.shape[0], -1)
        result = np.empty(
            (values.shape[0], len(self.forest.nodes)),
            dtype=np.float64,
        )
        for node in self.forest.nodes:
            result[:, node.node_id] = flattened[:, node.flat_indices].sum(axis=1)
        return result

    def render_frontier_scalings(
        self,
        active_node_ids: tuple[int, ...],
        node_scalings: npt.ArrayLike,
        *,
        fill_value: float = np.nan,
    ) -> npt.NDArray[np.float64]:
        """Render active node scalings onto the full spatial grid.

        Args:
            active_node_ids: Forest frontier nodes whose supports must not
                overlap.
            node_scalings: Positive scaling vector for every forest node.
            fill_value: Value retained outside declared active supports.

        Returns:
            Grid with each active node scaling broadcast over its support.

        Raises:
            ValueError: If IDs, scalings, overlap, or coverage are invalid.
        """
        scalings = _finite_vector(
            node_scalings,
            expected_shape=(len(self.forest.nodes),),
            name="node_scalings",
        )
        if np.any(scalings <= 0.0):
            raise ValueError("node_scalings must be positive.")
        grid = np.full(self.forest.shape, fill_value, dtype=np.float64)
        flattened = grid.reshape(-1)
        covered = np.zeros(flattened.size, dtype=np.bool_)
        for node_id in active_node_ids:
            if isinstance(node_id, bool) or not isinstance(node_id, (int, np.integer)):
                raise ValueError("active_node_ids must contain valid integer node IDs.")
            try:
                node = self.forest.nodes[int(node_id)]
            except IndexError as error:
                raise ValueError(f"Unknown active forest node {node_id!r}.") from error
            if np.any(covered[node.flat_indices]):
                raise ValueError("Active forest node supports must not overlap.")
            flattened[node.flat_indices] = scalings[node.node_id]
            covered[node.flat_indices] = True

        declared = np.zeros(flattened.size, dtype=np.bool_)
        for group in self.forest.groups:
            declared |= group.mask.reshape(-1)
        if not np.array_equal(covered, declared):
            raise ValueError("Active forest nodes must cover every declared group exactly once.")
        return grid


def _finite_vector(
    values: npt.ArrayLike,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.float64]:
    """Return a finite real vector with one exact shape."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(source, dtype=np.float64)
    if result.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


__all__ = ["GammaBetaCoordinateLayout"]
