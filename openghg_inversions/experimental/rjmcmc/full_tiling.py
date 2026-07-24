"""Exact tiny-state oracle for canonical iterated-bisection tilings.

The scientific geometry in this module is the final canonical set of leaf
rectangles, not a decomposition tree or chronological split history.  The
module deliberately uses exhaustive enumeration and raster cover validation:
it is a correctness oracle for small fixed-``K`` experiments, not a production
tiling implementation.

Two fixed-dimensional structural proposals are implemented.  An edge flip
merges a midpoint-friend pair and splits the same parent in the perpendicular
orientation.  A resolution relocation merges one friend pair and splits a
different leaf.  Both retain the complete merge/split choice as a temporary
auxiliary path, so their pointwise reverse probabilities are explicit.

Leaf masses use a Dirichlet allocation whose shapes are supplied by one
globally additive cell measure.  This makes the allocation independent of the
binary construction used to describe a tiling.  Proposal constructors are
deterministic and consume no random numbers.

The main entry points are :func:`enumerate_tilings`,
:func:`edge_flip_paths`, :func:`relocation_paths`,
:func:`propose_edge_flip`, and :func:`propose_resolution_relocation`.  This
experimental research API is a tiny-state correctness oracle, not a stable
production sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np


Axis: TypeAlias = Literal["horizontal", "vertical"]
"""Orientation of a midpoint bisection."""

Move: TypeAlias = Literal["edge_flip", "resolution_relocation"]
"""Name of a fixed-``K`` full-tiling proposal."""


def _normalise_axis(axis: Axis) -> Axis:
    """Validate and return one bisection orientation."""
    if axis not in ("horizontal", "vertical"):
        raise ValueError("axis must be 'horizontal' or 'vertical'.")
    return axis


@dataclass(frozen=True, slots=True, order=True)
class Rectangle:
    """One half-open rectangular block of native-grid cells.

    Args:
        row_start: Inclusive first row.
        row_stop: Exclusive final row.
        col_start: Inclusive first column.
        col_stop: Exclusive final column.

    Raises:
        TypeError: If a bound is not an integer.
        ValueError: If either half-open interval is empty.
    """

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def __post_init__(self) -> None:
        """Validate and normalize the rectangle bounds."""
        values = (self.row_start, self.row_stop, self.col_start, self.col_stop)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("rectangle bounds must be integers.")
        for name, value in zip(
            ("row_start", "row_stop", "col_start", "col_stop"),
            values,
        ):
            object.__setattr__(self, name, int(value))
        if self.row_start >= self.row_stop or self.col_start >= self.col_stop:
            raise ValueError("rectangle intervals must be non-empty.")

    @property
    def height(self) -> int:
        """Return the number of covered rows."""
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        """Return the number of covered columns."""
        return self.col_stop - self.col_start

    @property
    def area(self) -> int:
        """Return the number of covered native-grid cells."""
        return self.height * self.width

    @property
    def admissible_axes(self) -> tuple[Axis, ...]:
        """Return all orientations along which this block can be bisected."""
        axes: list[Axis] = []
        if self.height >= 2:
            axes.append("horizontal")
        if self.width >= 2:
            axes.append("vertical")
        return tuple(axes)

    def midpoint_children(self, axis: Axis) -> tuple[Rectangle, Rectangle]:
        """Bisect the rectangle at the integer midpoint of one axis.

        Odd lengths assign the smaller part to the first child.  This
        convention is deterministic but does not require dyadic interval
        endpoints.

        Args:
            axis: Row-wise (horizontal) or column-wise (vertical) split.

        Returns:
            The ordered first and second child rectangles.

        Raises:
            ValueError: If the axis is invalid or has length one.
        """
        axis = _normalise_axis(axis)
        if axis == "horizontal":
            if self.height < 2:
                raise ValueError("a one-row rectangle cannot be split horizontally.")
            midpoint = self.row_start + self.height // 2
            return (
                Rectangle(self.row_start, midpoint, self.col_start, self.col_stop),
                Rectangle(midpoint, self.row_stop, self.col_start, self.col_stop),
            )
        if self.width < 2:
            raise ValueError("a one-column rectangle cannot be split vertically.")
        midpoint = self.col_start + self.width // 2
        return (
            Rectangle(self.row_start, self.row_stop, self.col_start, midpoint),
            Rectangle(self.row_start, self.row_stop, midpoint, self.col_stop),
        )


@dataclass(frozen=True, slots=True, order=True)
class SplitChoice:
    """A labelled leaf and orientation selected for bisection.

    Args:
        leaf: Existing rectangular leaf.
        axis: Requested midpoint-bisection orientation.

    Raises:
        TypeError: If ``leaf`` is not a rectangle.
        ValueError: If ``axis`` is not a supported orientation.
    """

    leaf: Rectangle
    axis: Axis

    def __post_init__(self) -> None:
        """Validate the labelled split."""
        if not isinstance(self.leaf, Rectangle):
            raise TypeError("leaf must be a Rectangle.")
        object.__setattr__(self, "axis", _normalise_axis(self.axis))


@dataclass(frozen=True, slots=True, order=True)
class MergeChoice:
    """A labelled parent and orientation whose midpoint children are leaves.

    Args:
        parent: Rectangle recovered by merging two midpoint friends.
        axis: Orientation that produced the ordered friend pair.

    Raises:
        TypeError: If ``parent`` is not a rectangle.
        ValueError: If the axis is invalid or cannot bisect the parent.
    """

    parent: Rectangle
    axis: Axis

    def __post_init__(self) -> None:
        """Validate the labelled merge."""
        if not isinstance(self.parent, Rectangle):
            raise TypeError("parent must be a Rectangle.")
        object.__setattr__(self, "axis", _normalise_axis(self.axis))
        self.parent.midpoint_children(self.axis)

    @property
    def children(self) -> tuple[Rectangle, Rectangle]:
        """Return the ordered leaf pair removed by the merge."""
        return self.parent.midpoint_children(self.axis)


def _normalise_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize a two-dimensional native-grid shape."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a two-integer tuple.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in shape):
        raise TypeError("shape entries must be integers.")
    normalized = (int(shape[0]), int(shape[1]))
    if normalized[0] < 1 or normalized[1] < 1:
        raise ValueError("shape entries must be positive.")
    return normalized


@dataclass(frozen=True, slots=True)
class LeafTiling:
    """A construction-history-free rectangular tiling of one grid.

    Args:
        shape: Native-grid row and column counts.
        leaves: Rectangles that cover every cell exactly once.  Input order is
            ignored and stored canonically.

    Raises:
        TypeError: If the shape or leaves have the wrong type.
        ValueError: If leaves are duplicated, out of bounds, overlapping, or
            do not cover the complete grid.
    """

    shape: tuple[int, int]
    leaves: tuple[Rectangle, ...]

    def __post_init__(self) -> None:
        """Canonicalize leaves and validate exact cover."""
        shape = _normalise_shape(self.shape)
        if not isinstance(self.leaves, tuple) or not self.leaves:
            raise TypeError("leaves must be a non-empty tuple of Rectangle objects.")
        if any(not isinstance(leaf, Rectangle) for leaf in self.leaves):
            raise TypeError("leaves must contain only Rectangle objects.")
        leaves = tuple(sorted(self.leaves))
        if len(set(leaves)) != len(leaves):
            raise ValueError("leaves cannot contain duplicate rectangles.")
        cover = np.zeros(shape, dtype=np.uint8)
        for leaf in leaves:
            if (
                leaf.row_start < 0
                or leaf.col_start < 0
                or leaf.row_stop > shape[0]
                or leaf.col_stop > shape[1]
            ):
                raise ValueError("every leaf must lie within the grid.")
            block = cover[leaf.row_start : leaf.row_stop, leaf.col_start : leaf.col_stop]
            if np.any(block):
                raise ValueError("leaves cannot overlap.")
            block[...] = 1
        if not np.all(cover):
            raise ValueError("leaves must cover the complete grid.")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "leaves", leaves)

    @classmethod
    def root(cls, shape: tuple[int, int]) -> LeafTiling:
        """Return the one-leaf tiling of a grid.

        Args:
            shape: Native-grid row and column counts.

        Returns:
            The root tiling.
        """
        shape = _normalise_shape(shape)
        return cls(shape, (Rectangle(0, shape[0], 0, shape[1]),))

    @property
    def k(self) -> int:
        """Return the number of leaves."""
        return len(self.leaves)

    def split(self, choice: SplitChoice) -> LeafTiling:
        """Apply one admissible midpoint split.

        Args:
            choice: Existing leaf and admissible orientation.

        Returns:
            A new canonical tiling with one additional leaf.

        Raises:
            TypeError: If ``choice`` is not a :class:`SplitChoice`.
            ValueError: If the labelled split is unavailable.
        """
        if not isinstance(choice, SplitChoice):
            raise TypeError("choice must be a SplitChoice.")
        if choice not in split_choices(self):
            raise ValueError("the split choice is not available in this tiling.")
        children = choice.leaf.midpoint_children(choice.axis)
        leaves = tuple(leaf for leaf in self.leaves if leaf != choice.leaf) + children
        return LeafTiling(self.shape, leaves)

    def merge(self, choice: MergeChoice) -> LeafTiling:
        """Apply one available midpoint-friend merge.

        Args:
            choice: Parent and source split orientation.

        Returns:
            A new canonical tiling with one fewer leaf.

        Raises:
            TypeError: If ``choice`` is not a :class:`MergeChoice`.
            ValueError: If the labelled merge is unavailable.
        """
        if not isinstance(choice, MergeChoice):
            raise TypeError("choice must be a MergeChoice.")
        if choice not in merge_choices(self):
            raise ValueError("the merge choice is not available in this tiling.")
        children = frozenset(choice.children)
        leaves = tuple(leaf for leaf in self.leaves if leaf not in children) + (choice.parent,)
        return LeafTiling(self.shape, leaves)


def split_choices(tiling: LeafTiling) -> tuple[SplitChoice, ...]:
    """Return every labelled midpoint split available in a tiling.

    Args:
        tiling: Source canonical leaf tiling.

    Returns:
        Canonically ordered leaf-orientation choices.

    Raises:
        TypeError: If ``tiling`` is not a :class:`LeafTiling`.
    """
    if not isinstance(tiling, LeafTiling):
        raise TypeError("tiling must be a LeafTiling.")
    return tuple(SplitChoice(leaf, axis) for leaf in tiling.leaves for axis in leaf.admissible_axes)


@lru_cache(maxsize=256)
def _cached_merge_choices(tiling: LeafTiling) -> tuple[MergeChoice, ...]:
    """Return the cached midpoint-friend catalogue for one immutable tiling."""
    choices: set[MergeChoice] = set()
    right_neighbors = {(leaf.row_start, leaf.row_stop, leaf.col_start): leaf for leaf in tiling.leaves}
    bottom_neighbors = {(leaf.col_start, leaf.col_stop, leaf.row_start): leaf for leaf in tiling.leaves}
    for first in tiling.leaves:
        second = right_neighbors.get((first.row_start, first.row_stop, first.col_stop))
        if second is not None:
            parent = Rectangle(
                first.row_start,
                first.row_stop,
                first.col_start,
                second.col_stop,
            )
            if parent.midpoint_children("vertical") == (first, second):
                choices.add(MergeChoice(parent, "vertical"))

        second = bottom_neighbors.get((first.col_start, first.col_stop, first.row_stop))
        if second is not None:
            parent = Rectangle(
                first.row_start,
                second.row_stop,
                first.col_start,
                first.col_stop,
            )
            if parent.midpoint_children("horizontal") == (first, second):
                choices.add(MergeChoice(parent, "horizontal"))
    return tuple(sorted(choices))


def merge_choices(tiling: LeafTiling) -> tuple[MergeChoice, ...]:
    """Infer every midpoint-friend pair directly from leaf geometry.

    Args:
        tiling: Source canonical leaf tiling.

    Returns:
        Canonically ordered parent-orientation choices.

    Raises:
        TypeError: If ``tiling`` is not a :class:`LeafTiling`.
    """
    if not isinstance(tiling, LeafTiling):
        raise TypeError("tiling must be a LeafTiling.")
    return _cached_merge_choices(tiling)


def enumerate_tilings(shape: tuple[int, int], k: int) -> tuple[LeafTiling, ...]:
    """Exhaustively enumerate unique midpoint-bisection tilings.

    This routine expands all split orders and orientations and deduplicates
    their final canonical leaf sets.  Its cost is intentionally exponential.

    Args:
        shape: Native-grid row and column counts.
        k: Required positive leaf count.

    Returns:
        Canonically ordered unique tilings with exactly ``k`` leaves.

    Raises:
        TypeError: If ``k`` is not an integer.
        ValueError: If ``k`` is outside the grid-cell range.
    """
    shape = _normalise_shape(shape)
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer.")
    k = int(k)
    if k < 1 or k > shape[0] * shape[1]:
        raise ValueError("k must lie between one and the native-grid cell count.")
    current = {LeafTiling.root(shape)}
    while next(iter(current)).k < k:
        expanded: set[LeafTiling] = set()
        for tiling in current:
            expanded.update(tiling.split(choice) for choice in split_choices(tiling))
        if not expanded:
            return ()
        current = expanded
    return tuple(sorted(current, key=lambda tiling: tiling.leaves))


@lru_cache(maxsize=256)
def _cached_is_recursive_bisection_tiling(tiling: LeafTiling) -> bool:
    """Return cached recursive-bisection membership for one immutable tiling."""
    leaf_set = frozenset(tiling.leaves)

    @lru_cache(maxsize=None)
    def admissible(block: Rectangle) -> bool:
        """Return whether one block decomposes exactly into active leaves."""
        if block in leaf_set:
            return True
        contained = tuple(
            leaf
            for leaf in tiling.leaves
            if (
                leaf.row_start >= block.row_start
                and leaf.row_stop <= block.row_stop
                and leaf.col_start >= block.col_start
                and leaf.col_stop <= block.col_stop
            )
        )
        if sum(leaf.area for leaf in contained) != block.area:
            return False
        return any(
            all(admissible(child) for child in block.midpoint_children(axis))
            for axis in block.admissible_axes
        )

    root = Rectangle(0, tiling.shape[0], 0, tiling.shape[1])
    return admissible(root)


def is_recursive_bisection_tiling(tiling: LeafTiling) -> bool:
    """Test whether a leaf set has at least one midpoint-bisection decomposition.

    A canonical leaf set can have several valid decomposition trees.  This
    dynamic programme asks only whether at least one exists and therefore does
    not attach construction multiplicity to the state.

    Args:
        tiling: Exact rectangular cover to test.

    Returns:
        Whether repeated midpoint bisection of the root can produce the leaves.

    Raises:
        TypeError: If ``tiling`` is not a :class:`LeafTiling`.
    """
    if not isinstance(tiling, LeafTiling):
        raise TypeError("tiling must be a LeafTiling.")
    return _cached_is_recursive_bisection_tiling(tiling)


@dataclass(frozen=True, slots=True, eq=False)
class AdditiveAlphaPrior:
    """Globally additive Dirichlet base measure on native-grid cells.

    Args:
        cell_weights: Strictly positive finite two-dimensional relative
            weights.
        concentration: Positive total Dirichlet concentration.

    Raises:
        TypeError: If the weights are not two-dimensional or concentration is
            not scalar.
        ValueError: If parameters are non-finite/non-positive or their relative
            normalization is not representable in ``float64``.
    """

    cell_weights: np.ndarray
    concentration: float
    _scaled_weights: np.ndarray = field(init=False, repr=False)
    _scaled_weight_total: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Own and validate the additive cell measure."""
        weights = np.asarray(self.cell_weights, dtype=np.float64)
        if weights.ndim != 2:
            raise TypeError("cell_weights must be a two-dimensional array.")
        if weights.size == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("cell_weights must be finite and strictly positive.")
        if isinstance(self.concentration, bool):
            raise TypeError("concentration must be a scalar.")
        concentration = float(self.concentration)
        if not math.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("concentration must be finite and positive.")
        scaled_weights = weights / float(np.max(weights))
        scaled_weight_total = float(np.sum(scaled_weights))
        if (
            np.any(scaled_weights <= 0.0)
            or not math.isfinite(scaled_weight_total)
            or scaled_weight_total <= 0.0
        ):
            raise ValueError("relative cell weights must have a representable finite normalization.")
        cell_alphas = concentration * scaled_weights / scaled_weight_total
        if np.any(cell_alphas <= 0.0) or not np.all(np.isfinite(cell_alphas)):
            raise ValueError("native-cell Dirichlet concentrations must remain representably positive.")
        try:
            log_normalizers_finite = math.isfinite(math.lgamma(concentration)) and all(
                math.isfinite(math.lgamma(float(alpha))) for alpha in cell_alphas.flat
            )
        except (OverflowError, ValueError):
            log_normalizers_finite = False
        if not log_normalizers_finite:
            raise ValueError("Dirichlet log normalizers must be representable in float64.")
        weights = weights.copy()
        weights.setflags(write=False)
        scaled_weights = scaled_weights.copy()
        scaled_weights.setflags(write=False)
        object.__setattr__(self, "cell_weights", weights)
        object.__setattr__(self, "concentration", concentration)
        object.__setattr__(self, "_scaled_weights", scaled_weights)
        object.__setattr__(self, "_scaled_weight_total", scaled_weight_total)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native-grid shape."""
        return (int(self.cell_weights.shape[0]), int(self.cell_weights.shape[1]))

    def alpha(self, rectangle: Rectangle) -> float:
        """Return the additive Dirichlet shape of one rectangle.

        Args:
            rectangle: In-domain rectangular leaf or block.

        Returns:
            Positive dimensionless Dirichlet concentration proportional to
            contained cell weight.

        Raises:
            TypeError: If ``rectangle`` is not a :class:`Rectangle`.
            ValueError: If the rectangle lies outside the prior grid.
        """
        if not isinstance(rectangle, Rectangle):
            raise TypeError("rectangle must be a Rectangle.")
        if (
            rectangle.row_start < 0
            or rectangle.col_start < 0
            or rectangle.row_stop > self.shape[0]
            or rectangle.col_stop > self.shape[1]
        ):
            raise ValueError("rectangle must lie within the prior grid.")
        weight = np.sum(
            self._scaled_weights[
                rectangle.row_start : rectangle.row_stop,
                rectangle.col_start : rectangle.col_stop,
            ]
        )
        return self.concentration * float(weight) / self._scaled_weight_total

    def leaf_alphas(self, tiling: LeafTiling) -> np.ndarray:
        """Return owned read-only Dirichlet shapes in canonical leaf order.

        Args:
            tiling: Tiling defined on the same native grid.

        Returns:
            Read-only ``float64`` array of shape ``(tiling.k,)`` in canonical
            leaf order.

        Raises:
            ValueError: If the tiling and prior grids differ.
        """
        if tiling.shape != self.shape:
            raise ValueError("tiling and additive-alpha prior shapes must match.")
        values = np.asarray([self.alpha(leaf) for leaf in tiling.leaves], dtype=np.float64)
        values.setflags(write=False)
        return values

    def log_beta_density(
        self,
        children: tuple[Rectangle, Rectangle],
        fraction: float,
    ) -> float:
        """Evaluate a normalized split-fraction Beta log density.

        Args:
            children: Ordered disjoint child rectangles.
            fraction: ``children[0]`` mass fraction in the open unit interval.

        Returns:
            Dimensionless normalized Beta log density, or negative infinity
            outside support.

        Raises:
            TypeError: If ``children`` is not a two-rectangle tuple.
            ValueError: If children are not an ordered midpoint-friend pair or
                either lies outside the prior grid.
        """
        if (
            not isinstance(children, tuple)
            or len(children) != 2
            or any(not isinstance(child, Rectangle) for child in children)
        ):
            raise TypeError("children must be a two-Rectangle tuple.")
        first_child, second_child = children
        valid_midpoint_pair = False
        if (
            first_child.row_start == second_child.row_start
            and first_child.row_stop == second_child.row_stop
            and first_child.col_stop == second_child.col_start
        ):
            parent = Rectangle(
                first_child.row_start,
                first_child.row_stop,
                first_child.col_start,
                second_child.col_stop,
            )
            valid_midpoint_pair = parent.midpoint_children("vertical") == children
        if (
            first_child.col_start == second_child.col_start
            and first_child.col_stop == second_child.col_stop
            and first_child.row_stop == second_child.row_start
        ):
            parent = Rectangle(
                first_child.row_start,
                second_child.row_stop,
                first_child.col_start,
                first_child.col_stop,
            )
            valid_midpoint_pair = parent.midpoint_children("horizontal") == children
        if not valid_midpoint_pair:
            raise ValueError("children must be an ordered midpoint-friend pair.")
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            return -math.inf
        first = self.alpha(children[0])
        second = self.alpha(children[1])
        return (
            math.lgamma(first + second)
            - math.lgamma(first)
            - math.lgamma(second)
            + (first - 1.0) * math.log(fraction)
            + (second - 1.0) * math.log1p(-fraction)
        )

    def log_mass_allocation_density(self, state: TilingState) -> float:
        """Evaluate the normalized conditional density of physical leaf masses.

        The density is with respect to ``K - 1`` independent masses at the
        state's fixed positive total.  It equals the Dirichlet density of the
        mass fractions times ``total**(-(K - 1))``.

        Args:
            state: Canonical tiling and aligned positive leaf masses.

        Returns:
            Normalized conditional log density.

        Raises:
            TypeError: If ``state`` is not a :class:`TilingState`.
            ValueError: If the state and prior grids differ.

        Note:
            Like every continuous density, its numeric value depends on the
            common physical mass unit used by ``state.leaf_masses``.
        """
        if not isinstance(state, TilingState):
            raise TypeError("state must be a TilingState.")
        alphas = self.leaf_alphas(state.tiling)
        total = state.total_mass
        return (
            math.lgamma(self.concentration)
            - sum(math.lgamma(float(alpha)) for alpha in alphas)
            + float(np.dot(alphas - 1.0, np.log(state.leaf_masses / total)))
            - (state.tiling.k - 1) * math.log(total)
        )


@dataclass(frozen=True, slots=True, eq=False)
class TilingState:
    """A canonical leaf tiling with aligned positive physical masses.

    Args:
        tiling: Canonical geometry state.
        leaf_masses: One-dimensional ``(tiling.k,)`` vector in canonical leaf
            order. Values share one physical unit, are copied, and are stored
            read-only.

    Raises:
        TypeError: If ``tiling`` is not a :class:`LeafTiling`.
        ValueError: If the tiling is outside the recursive family, masses are
            mis-shaped/non-finite/non-positive, or their total overflows.
    """

    tiling: LeafTiling
    leaf_masses: np.ndarray

    def __post_init__(self) -> None:
        """Own and validate the leaf-mass coordinate vector."""
        if not isinstance(self.tiling, LeafTiling):
            raise TypeError("tiling must be a LeafTiling.")
        if not is_recursive_bisection_tiling(self.tiling):
            raise ValueError("tiling must belong to the recursive midpoint-bisection family.")
        masses = np.asarray(self.leaf_masses, dtype=np.float64)
        if masses.ndim != 1 or masses.shape != (self.tiling.k,):
            raise ValueError("leaf_masses must have one entry per canonical leaf.")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("leaf_masses must be finite and strictly positive.")
        largest_mass = float(np.max(masses))
        scaled_total = float(np.sum(masses / largest_mass))
        if largest_mass > np.finfo(np.float64).max / scaled_total:
            raise ValueError("the total leaf mass must be finite.")
        masses = masses.copy()
        masses.setflags(write=False)
        object.__setattr__(self, "leaf_masses", masses)

    @property
    def total_mass(self) -> float:
        """Return the conserved total leaf mass."""
        return float(np.sum(self.leaf_masses))

    def mass(self, leaf: Rectangle) -> float:
        """Return the mass aligned with one active leaf.

        Args:
            leaf: Active canonical leaf.

        Returns:
            Its positive physical mass.

        Raises:
            KeyError: If the rectangle is not active.
        """
        try:
            position = self.tiling.leaves.index(leaf)
        except ValueError as exc:
            raise KeyError("rectangle is not an active leaf.") from exc
        return float(self.leaf_masses[position])


@dataclass(frozen=True, slots=True, order=True)
class EdgeFlipPath:
    """Auxiliary path for one merge followed by a perpendicular split.

    Args:
        merge: Source midpoint-friend merge.
        target_axis: Perpendicular destination split orientation.

    Raises:
        TypeError: If ``merge`` has the wrong type.
        ValueError: If the target axis is invalid, parallel, or unavailable.
    """

    merge: MergeChoice
    target_axis: Axis

    def __post_init__(self) -> None:
        """Validate the labelled edge-flip path."""
        if not isinstance(self.merge, MergeChoice):
            raise TypeError("merge must be a MergeChoice.")
        target_axis = _normalise_axis(self.target_axis)
        if target_axis == self.merge.axis:
            raise ValueError("an edge flip must use the perpendicular orientation.")
        self.merge.parent.midpoint_children(target_axis)
        object.__setattr__(self, "target_axis", target_axis)


@dataclass(frozen=True, slots=True, order=True)
class RelocationPath:
    """Auxiliary path for merging at one location and splitting another.

    Args:
        merge: Source midpoint-friend merge.
        split: Labelled split in the merged intermediate tiling.

    Raises:
        TypeError: If either choice has the wrong type.
        ValueError: If the split simply reverses the selected merge.
    """

    merge: MergeChoice
    split: SplitChoice

    def __post_init__(self) -> None:
        """Validate path metadata independent of a source tiling."""
        if not isinstance(self.merge, MergeChoice):
            raise TypeError("merge must be a MergeChoice.")
        if not isinstance(self.split, SplitChoice):
            raise TypeError("split must be a SplitChoice.")
        if self.split.leaf == self.merge.parent:
            raise ValueError("resolution relocation must split a different leaf.")


AuxiliaryPath: TypeAlias = EdgeFlipPath | RelocationPath
"""Complete labelled discrete path of a fixed-``K`` proposal."""


def edge_flip_paths(tiling: LeafTiling) -> tuple[EdgeFlipPath, ...]:
    """Return all available perpendicular edge-flip paths.

    Args:
        tiling: Source canonical tiling.

    Returns:
        Canonically ordered eligible paths.

    Raises:
        ValueError: If the source tiling is outside the recursive-bisection
            family.
    """
    if not is_recursive_bisection_tiling(tiling):
        raise ValueError("source tiling must belong to the recursive midpoint-bisection family.")
    paths: list[EdgeFlipPath] = []
    for merge in merge_choices(tiling):
        target: Axis = "vertical" if merge.axis == "horizontal" else "horizontal"
        if target in merge.parent.admissible_axes:
            path = EdgeFlipPath(merge, target)
            intermediate = tiling.merge(merge)
            candidate = intermediate.split(SplitChoice(merge.parent, target))
            if is_recursive_bisection_tiling(candidate):
                paths.append(path)
    return tuple(sorted(paths))


def relocation_paths(tiling: LeafTiling) -> tuple[RelocationPath, ...]:
    """Return all labelled sequential merge/remote-split paths.

    The selection law is uniform over source merge choices and then uniform
    over destination split operations conditional on the selected merge.  The
    returned tuple is a catalogue, not an assertion that all paths have equal
    probability.

    Args:
        tiling: Source canonical tiling.

    Returns:
        Canonically ordered eligible paths.

    Raises:
        ValueError: If the source tiling is outside the recursive-bisection
            family.
    """
    if not is_recursive_bisection_tiling(tiling):
        raise ValueError("source tiling must belong to the recursive midpoint-bisection family.")
    paths: list[RelocationPath] = []
    for merge in merge_choices(tiling):
        paths.extend(RelocationPath(merge, split) for split in _relocation_destinations(tiling, merge))
    return tuple(sorted(paths))


def _relocation_destinations(tiling: LeafTiling, merge: MergeChoice) -> tuple[SplitChoice, ...]:
    """Return recursively admissible remote splits after one labelled merge.

    Args:
        tiling: Source canonical tiling.
        merge: Available source merge.

    Returns:
        Canonically ordered split choices in the intermediate tiling, excluding
        the trivial reverse split and candidates outside the recursive family.
    """
    intermediate = tiling.merge(merge)
    destinations: list[SplitChoice] = []
    for choice in split_choices(intermediate):
        if choice.leaf != merge.parent:
            candidate = intermediate.split(choice)
            if is_recursive_bisection_tiling(candidate):
                destinations.append(choice)
    return tuple(destinations)


def relocation_path_log_probability(tiling: LeafTiling, path: RelocationPath) -> float:
    """Return the exact sequential discrete log probability of one path.

    Args:
        tiling: Source canonical tiling.
        path: Available merge then remote-split path.

    Returns:
        ``-log(number of merges) - log(number of conditional splits)``.

    Raises:
        ValueError: If the source is outside the recursive family or the path
            is unavailable.
    """
    if not is_recursive_bisection_tiling(tiling):
        raise ValueError("source tiling must belong to the recursive midpoint-bisection family.")
    merges = merge_choices(tiling)
    if path.merge not in merges:
        raise ValueError("the relocation merge is unavailable.")
    destinations = _relocation_destinations(tiling, path.merge)
    if path.split not in destinations:
        raise ValueError("the relocation destination is unavailable.")
    return -math.log(len(merges)) - math.log(len(destinations))


@dataclass(frozen=True, slots=True, eq=False)
class TilingTransitionTerms:
    """Candidate state and exact decomposed fixed-``K`` MH terms.

    Attributes:
        candidate: Proposed state, or the source object when invalid.
        reverse_path: Unique labelled pointwise reverse for a valid proposal.
        delta_log_allocation_prior: Candidate-minus-source normalized
            conditional Dirichlet mass density.
        delta_log_structural_prior: Candidate-minus-source geometry log prior;
            zero for the declared uniform fixed-``K`` target.
        log_q_forward_selection: Forward discrete path-selection probability.
        log_q_forward_auxiliary: Forward normalized Beta log density.
        log_q_reverse_selection: Reverse discrete path-selection probability.
        log_q_reverse_auxiliary: Reverse normalized Beta log density.
        log_jacobian: Log absolute augmented-coordinate Jacobian.
        move: Stable proposal name.
        valid: Whether the candidate can enter an MH decision.
        reason: Invalid self-transition explanation.
        log_acceptance_ratio: Complete untruncated within-kernel log MH ratio.

    All delta-log and ``log_*`` values use natural logarithms.
    State-independent symmetric move-mixture weights cancel outside these
    terms. A sampler with state-dependent or availability-renormalized move
    weights must add their forward/reverse log probabilities.

    Raises:
        TypeError: If state or validity metadata have the wrong type.
        ValueError: If move metadata, validity metadata, or log terms are
            inconsistent.
    """

    candidate: TilingState
    reverse_path: AuxiliaryPath | None
    delta_log_allocation_prior: float
    delta_log_structural_prior: float
    log_q_forward_selection: float
    log_q_forward_auxiliary: float
    log_q_reverse_selection: float
    log_q_reverse_auxiliary: float
    log_jacobian: float
    move: Move
    valid: bool = True
    reason: str | None = None
    log_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate terms and calculate the aggregate log acceptance ratio."""
        if not isinstance(self.candidate, TilingState):
            raise TypeError("candidate must be a TilingState.")
        if self.move not in ("edge_flip", "resolution_relocation"):
            raise ValueError("move must name a full-tiling proposal.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be Boolean.")
        if self.valid and self.reverse_path is None:
            raise ValueError("a valid transition must retain its reverse path.")
        if self.valid and self.reason is not None:
            raise ValueError("a valid transition cannot have an invalidity reason.")
        if not self.valid and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("an invalid transition must provide a reason.")
        names = (
            "delta_log_allocation_prior",
            "delta_log_structural_prior",
            "log_q_forward_selection",
            "log_q_forward_auxiliary",
            "log_q_reverse_selection",
            "log_q_reverse_auxiliary",
            "log_jacobian",
        )
        values = tuple(float(getattr(self, name)) for name in names)
        if any(math.isnan(value) for value in values):
            raise ValueError("transition log terms cannot be NaN.")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)
        if self.valid:
            log_ratio = (
                self.delta_log_allocation_prior
                + self.delta_log_structural_prior
                + self.log_q_reverse_selection
                + self.log_q_reverse_auxiliary
                - self.log_q_forward_selection
                - self.log_q_forward_auxiliary
                + self.log_jacobian
            )
        else:
            log_ratio = -math.inf
        object.__setattr__(self, "log_acceptance_ratio", float(log_ratio))


def _invalid_transition(source: TilingState, move: Move, reason: str) -> TilingTransitionTerms:
    """Return an explicit invalid self-transition.

    The exact ``source`` object is retained, the reverse path is absent,
    decomposed terms are zero, and the aggregate log acceptance ratio is
    negative infinity.

    Args:
        source: State retained by identity.
        move: Attempted kernel name.
        reason: Non-empty invalidity explanation.

    Returns:
        Invalid transition terms.
    """
    return TilingTransitionTerms(
        candidate=source,
        reverse_path=None,
        delta_log_allocation_prior=0.0,
        delta_log_structural_prior=0.0,
        log_q_forward_selection=0.0,
        log_q_forward_auxiliary=0.0,
        log_q_reverse_selection=0.0,
        log_q_reverse_auxiliary=0.0,
        log_jacobian=0.0,
        move=move,
        valid=False,
        reason=reason,
    )


def _state_from_mass_map(tiling: LeafTiling, masses: dict[Rectangle, float]) -> TilingState:
    """Build a state by aligning a rectangle-to-mass map canonically.

    Args:
        tiling: Candidate canonical tiling.
        masses: Physical mass keyed by every candidate leaf.

    Returns:
        State with masses in canonical leaf order.

    Raises:
        KeyError: If any canonical leaf is absent from ``masses``.
    """
    return TilingState(tiling, np.asarray([masses[leaf] for leaf in tiling.leaves]))


def propose_edge_flip(
    prior: AdditiveAlphaPrior,
    source: TilingState,
    *,
    path: EdgeFlipPath,
    new_fraction: float,
) -> TilingTransitionTerms:
    """Construct one deterministic edge-flip proposal.

    This function neither samples the auxiliary fraction nor performs the MH
    accept/reject decision. The prior and source must use the same native-grid
    shape.

    Args:
        prior: Globally additive allocation prior.
        source: Source tiling and physical masses.
        path: Available merge and perpendicular-resplit path.
        new_fraction: Proposed first-child fraction in the open unit interval.

    Returns:
        Exact decomposed proposal terms.  Invalid inputs within the proposal
        support become explicit self-transitions.

    Raises:
        TypeError: If the prior, source, or path has the wrong type.
        ValueError: If the prior and source grids differ.
        RuntimeError: If a constructed candidate lacks its required reverse.
    """
    if not isinstance(prior, AdditiveAlphaPrior):
        raise TypeError("prior must be an AdditiveAlphaPrior.")
    if not isinstance(source, TilingState):
        raise TypeError("source must be a TilingState.")
    if not isinstance(path, EdgeFlipPath):
        raise TypeError("path must be an EdgeFlipPath.")
    paths = edge_flip_paths(source.tiling)
    if path not in paths:
        return _invalid_transition(source, "edge_flip", "edge-flip path is unavailable")
    new_fraction = float(new_fraction)
    if not math.isfinite(new_fraction) or not 0.0 < new_fraction < 1.0:
        return _invalid_transition(source, "edge_flip", "new fraction lies outside support")

    source_children = path.merge.children
    old_total = source.mass(source_children[0]) + source.mass(source_children[1])
    old_fraction = source.mass(source_children[0]) / old_total
    intermediate = source.tiling.merge(path.merge)
    target = SplitChoice(path.merge.parent, path.target_axis)
    candidate_tiling = intermediate.split(target)
    target_children = target.leaf.midpoint_children(target.axis)
    masses = {leaf: source.mass(leaf) for leaf in source.tiling.leaves if leaf not in source_children}
    masses[target_children[0]] = old_total * new_fraction
    masses[target_children[1]] = old_total * (1.0 - new_fraction)
    candidate = _state_from_mass_map(candidate_tiling, masses)
    reverse = EdgeFlipPath(
        MergeChoice(path.merge.parent, path.target_axis),
        path.merge.axis,
    )
    reverse_paths = edge_flip_paths(candidate_tiling)
    if reverse not in reverse_paths:
        raise RuntimeError("constructed edge flip has no pointwise reverse.")
    return TilingTransitionTerms(
        candidate=candidate,
        reverse_path=reverse,
        delta_log_allocation_prior=(
            prior.log_mass_allocation_density(candidate) - prior.log_mass_allocation_density(source)
        ),
        delta_log_structural_prior=0.0,
        log_q_forward_selection=-math.log(len(paths)),
        log_q_forward_auxiliary=prior.log_beta_density(target_children, new_fraction),
        log_q_reverse_selection=-math.log(len(reverse_paths)),
        log_q_reverse_auxiliary=prior.log_beta_density(source_children, old_fraction),
        log_jacobian=0.0,
        move="edge_flip",
    )


def propose_resolution_relocation(
    prior: AdditiveAlphaPrior,
    source: TilingState,
    *,
    path: RelocationPath,
    new_fraction: float,
) -> TilingTransitionTerms:
    """Construct one deterministic resolution-relocation proposal.

    The physical-mass augmented map has absolute Jacobian equal to the
    destination leaf mass divided by the merged source-pair mass. This
    function neither samples the auxiliary fraction nor performs the MH
    accept/reject decision. The prior and source must use the same native-grid
    shape.

    Args:
        prior: Globally additive allocation prior.
        source: Source tiling and physical masses.
        path: Available merge and remote-split path.
        new_fraction: Proposed first-child fraction in the open unit interval.

    Returns:
        Exact decomposed proposal terms.  Invalid inputs within the proposal
        support become explicit self-transitions.

    Raises:
        TypeError: If the prior, source, or path has the wrong type.
        ValueError: If the prior and source grids differ.
        RuntimeError: If a constructed candidate lacks its required reverse.
    """
    if not isinstance(prior, AdditiveAlphaPrior):
        raise TypeError("prior must be an AdditiveAlphaPrior.")
    if not isinstance(source, TilingState):
        raise TypeError("source must be a TilingState.")
    if not isinstance(path, RelocationPath):
        raise TypeError("path must be a RelocationPath.")
    if path not in relocation_paths(source.tiling):
        return _invalid_transition(
            source,
            "resolution_relocation",
            "resolution-relocation path is unavailable",
        )
    new_fraction = float(new_fraction)
    if not math.isfinite(new_fraction) or not 0.0 < new_fraction < 1.0:
        return _invalid_transition(
            source,
            "resolution_relocation",
            "new fraction lies outside support",
        )

    source_children = path.merge.children
    source_total = source.mass(source_children[0]) + source.mass(source_children[1])
    old_fraction = source.mass(source_children[0]) / source_total
    destination_total = source.mass(path.split.leaf)
    intermediate = source.tiling.merge(path.merge)
    candidate_tiling = intermediate.split(path.split)
    destination_children = path.split.leaf.midpoint_children(path.split.axis)
    removed = frozenset((*source_children, path.split.leaf))
    masses = {leaf: source.mass(leaf) for leaf in source.tiling.leaves if leaf not in removed}
    masses[path.merge.parent] = source_total
    masses[destination_children[0]] = destination_total * new_fraction
    masses[destination_children[1]] = destination_total * (1.0 - new_fraction)
    candidate = _state_from_mass_map(candidate_tiling, masses)
    reverse = RelocationPath(
        MergeChoice(path.split.leaf, path.split.axis),
        SplitChoice(path.merge.parent, path.merge.axis),
    )
    if reverse not in relocation_paths(candidate_tiling):
        raise RuntimeError("constructed resolution relocation has no pointwise reverse.")
    return TilingTransitionTerms(
        candidate=candidate,
        reverse_path=reverse,
        delta_log_allocation_prior=(
            prior.log_mass_allocation_density(candidate) - prior.log_mass_allocation_density(source)
        ),
        delta_log_structural_prior=0.0,
        log_q_forward_selection=relocation_path_log_probability(source.tiling, path),
        log_q_forward_auxiliary=prior.log_beta_density(
            destination_children,
            new_fraction,
        ),
        log_q_reverse_selection=relocation_path_log_probability(
            candidate_tiling,
            reverse,
        ),
        log_q_reverse_auxiliary=prior.log_beta_density(source_children, old_fraction),
        log_jacobian=math.log(destination_total / source_total),
        move="resolution_relocation",
    )
