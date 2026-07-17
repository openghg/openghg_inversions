"""Fixed root-and-contrast coordinates for dyadic partition frontiers.

The coordinate vector has a permanent layout for one complete dyadic tree. Its
first value is the mean anomaly over the root tile and every non-terminal tree
node contributes one mass-preserving child contrast. A partition activates the
root coordinate and only the contrasts above its active frontier. Partition
updates can therefore leave the coordinate vector fixed while changing which
entries enter the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .state import PartitionState
from .tree import DyadicTree, NodeId


@dataclass(frozen=True, slots=True)
class TreeContrastLayout:
    """Stable root-and-split coordinate layout for one dyadic tree.

    Construct layouts with :meth:`from_tree`. Coordinate zero is the root mean
    anomaly. Subsequent coordinates correspond to splittable nodes in stable
    node-ID order.

    Attributes:
        tree: Complete canonical dyadic tree described by the layout.
        split_node_ids: Splittable node IDs in coordinate order after the root
            mean coordinate.
    """

    tree: DyadicTree
    split_node_ids: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        """Reject layouts that do not exactly describe their complete tree."""
        if not isinstance(self.tree, DyadicTree):
            raise TypeError("tree must be a DyadicTree.")
        expected = tuple(tile.node_id for tile in self.tree.nodes if self.tree.children(tile.node_id))
        if self.split_node_ids != expected:
            raise ValueError("split_node_ids must contain every splittable node in stable tree order.")
        if self.coordinate_count != len(self.tree.leaf_ids):
            raise ValueError("A binary contrast layout must have one coordinate per finest grid cell.")

    @classmethod
    def from_tree(cls, tree: DyadicTree) -> TreeContrastLayout:
        """Construct the fixed contrast layout for a complete dyadic tree.

        Args:
            tree: Tree whose non-terminal nodes define contrast coordinates.

        Returns:
            Stable layout with one coordinate per finest-grid degree of
            freedom.

        Raises:
            TypeError: If ``tree`` is not a :class:`DyadicTree`.
            ValueError: If a non-terminal node does not have exactly two
                children or the resulting layout is not square at the finest
                grid scale.
        """
        if not isinstance(tree, DyadicTree):
            raise TypeError("tree must be a DyadicTree.")

        split_node_ids: list[NodeId] = []
        for tile in tree.nodes:
            children = tree.children(tile.node_id)
            if children and len(children) != 2:
                raise ValueError("Every dyadic split must have exactly two children.")
            if children:
                split_node_ids.append(tile.node_id)

        layout = cls(tree=tree, split_node_ids=tuple(split_node_ids))
        if layout.coordinate_count != len(tree.leaf_ids):
            raise ValueError("A binary contrast layout must have one coordinate per finest grid cell.")
        return layout

    @property
    def coordinate_count(self) -> int:
        """Return the permanent number of inner product-space coordinates."""
        return 1 + len(self.split_node_ids)

    def contrast_index(self, node_id: NodeId) -> int:
        """Return the permanent coordinate index for one possible split.

        Args:
            node_id: Non-terminal node whose child contrast is requested.

        Returns:
            Coordinate index after the root mean coordinate.

        Raises:
            KeyError: If ``node_id`` is not a splittable tree node.
        """
        try:
            return self.split_node_ids.index(node_id) + 1
        except ValueError as error:
            raise KeyError(f"Node {node_id!r} does not define a split contrast.") from error

    def active_split_ids(self, partition: PartitionState) -> tuple[NodeId, ...]:
        """Return split nodes above a partition's active frontier.

        Args:
            partition: Valid active frontier on :attr:`tree`.

        Returns:
            Split-node IDs in stable tree order. A split is active exactly when
            the frontier descends through that node.
        """
        partition.validate(self.tree)
        active_splits: set[NodeId] = set()
        pending = [self.tree.root_id]
        while pending:
            node_id = pending.pop()
            if node_id in partition.active:
                continue
            children = self.tree.children(node_id)
            if not children:  # pragma: no cover - guarded by partition validation.
                raise ValueError("Partition frontier does not cover the tree root.")
            active_splits.add(node_id)
            pending.extend(children)
        return tuple(node_id for node_id in self.split_node_ids if node_id in active_splits)

    def active_coordinate_indices(self, partition: PartitionState) -> tuple[int, ...]:
        """Return the root and active-contrast coordinate indices."""
        return (0, *(self.contrast_index(node_id) for node_id in self.active_split_ids(partition)))

    def inactive_coordinate_indices(self, partition: PartitionState) -> tuple[int, ...]:
        """Return contrast indices below the partition frontier."""
        active = set(self.active_coordinate_indices(partition))
        return tuple(index for index in range(1, self.coordinate_count) if index not in active)

    def decoder(self, partition: PartitionState) -> np.ndarray:
        """Build the linear map from fixed coordinates to active-region means.

        For a split ``G = L union R`` with grid-cell counts ``n_L`` and
        ``n_R``, the child means are

        ``a_L = a_G + n_R / n_G * delta_G`` and
        ``a_R = a_G - n_L / n_G * delta_G``.

        Args:
            partition: Valid active frontier on :attr:`tree`.

        Returns:
            Matrix with shape ``(active_region, coordinate)``. Rows follow
            :meth:`PartitionState.ordered_active`; inactive columns are zero.
        """
        partition.validate(self.tree)
        root_row: npt.NDArray[np.float64] = np.zeros(self.coordinate_count, dtype=float)
        root_row[0] = 1.0
        rows_by_node: dict[NodeId, np.ndarray] = {}
        pending: list[tuple[NodeId, np.ndarray]] = [(self.tree.root_id, root_row)]

        while pending:
            node_id, parent_row = pending.pop()
            if node_id in partition.active:
                rows_by_node[node_id] = parent_row
                continue

            left_id, right_id = self.tree.children(node_id)
            left_area = self.tree.tile(left_id).area
            right_area = self.tree.tile(right_id).area
            total_area = left_area + right_area
            contrast_index = self.contrast_index(node_id)

            left_row = parent_row.copy()
            left_row[contrast_index] += right_area / total_area
            right_row = parent_row.copy()
            right_row[contrast_index] -= left_area / total_area
            pending.append((right_id, right_row))
            pending.append((left_id, left_row))

        return np.vstack([rows_by_node[node_id] for node_id in partition.ordered_active()])

    def decode(self, partition: PartitionState, coordinates: npt.ArrayLike) -> np.ndarray:
        """Decode fixed coordinates into active-region mean anomalies.

        Args:
            partition: Valid active frontier on :attr:`tree`.
            coordinates: Finite one-dimensional vector with
                :attr:`coordinate_count` values.

        Returns:
            Active-region means in stable partition order.

        Raises:
            ValueError: If ``coordinates`` has the wrong shape or contains a
                non-finite value.
        """
        source = np.asarray(coordinates)
        if np.iscomplexobj(source):
            raise ValueError("coordinates must be real-valued.")
        values = np.asarray(source, dtype=float)
        if values.shape != (self.coordinate_count,):
            raise ValueError(f"coordinates must have shape ({self.coordinate_count},).")
        if not np.all(np.isfinite(values)):
            raise ValueError("coordinates must contain only finite values.")
        return self.decoder(partition) @ values

    def prior_variances(self, scale: float) -> np.ndarray:
        """Return primitive Gaussian variances for all fixed coordinates.

        These variances are induced by independent finest-grid anomalies with
        standard deviation ``scale``. The root mean has variance
        ``scale**2 / n_root`` and a split contrast has variance
        ``scale**2 * (1 / n_left + 1 / n_right)``.

        Args:
            scale: Positive finite finest-grid anomaly standard deviation.

        Returns:
            Positive variances in permanent coordinate order.

        Raises:
            ValueError: If ``scale`` is not positive and finite.
        """
        if isinstance(scale, bool):
            raise ValueError("scale must be positive and finite.")
        try:
            scale_value = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("scale must be positive and finite.") from error
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("scale must be positive and finite.")

        with np.errstate(over="ignore", invalid="ignore"):
            variance = float(np.multiply(scale_value, scale_value))
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("scale produces non-finite primitive variances.")

        result: npt.NDArray[np.float64] = np.empty(self.coordinate_count, dtype=float)
        result[0] = variance / self.tree.tile(self.tree.root_id).area
        for node_id in self.split_node_ids:
            left_id, right_id = self.tree.children(node_id)
            left_area = self.tree.tile(left_id).area
            right_area = self.tree.tile(right_id).area
            result[self.contrast_index(node_id)] = variance * (1.0 / left_area + 1.0 / right_area)
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("scale produces non-finite primitive variances.")
        return result


__all__ = ["TreeContrastLayout"]
