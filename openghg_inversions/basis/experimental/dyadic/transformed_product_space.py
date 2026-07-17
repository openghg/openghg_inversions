"""Auxiliary-variable split/merge proposals for fixed dyadic product spaces.

This module adds coefficient transport to the fixed-dimensional product-space
kernel.  A local split or merge draws a replacement value for the affected
permanent contrast coordinate and swaps the old value into the reverse
auxiliary variable.  The Metropolis ratio includes both auxiliary densities,
both structure-proposal probabilities, and the swap Jacobian.

The accounting resembles a reversible-jump move, but the sampler is not
trans-dimensional: every root/contrast coordinate remains in the augmented
state under every partition.  The coordinate swap and the additive
parent/children transform both have unit absolute Jacobian.  A packed active-
only implementation would instead be genuine RJMCMC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .contrast import TreeContrastLayout
from .product_space import (
    LogAugmentedDensity,
    PartitionMove,
    ProductSpaceState,
    _checked_log_density,
    _frozen_vector,
    enumerate_partition_neighbors,
)
from .proposals import MergeMove, SplitMove


@dataclass(frozen=True, slots=True)
class AdditiveCoefficientTransform:
    """Mass-preserving map between one parent and two child coefficients.

    The canonical contrast is ``left - right``.  ``left_mass`` and
    ``right_mass`` may represent area, prior flux, or another declared positive
    measure, but callers must use the same measure in both directions.

    Attributes:
        left_mass: Positive finite mass of the canonical left child.
        right_mass: Positive finite mass of the canonical right child.
    """

    left_mass: float
    right_mass: float

    def __post_init__(self) -> None:
        """Validate the two child masses."""
        left_mass = _positive_scalar(self.left_mass, name="left_mass")
        right_mass = _positive_scalar(self.right_mass, name="right_mass")
        with np.errstate(over="ignore", invalid="ignore"):
            total_mass = left_mass + right_mass
        if not math.isfinite(total_mass):
            raise ValueError("Combined child mass must be finite.")
        object.__setattr__(self, "left_mass", left_mass)
        object.__setattr__(
            self,
            "right_mass",
            right_mass,
        )

    @property
    def total_mass(self) -> float:
        """Return the parent mass."""
        return self.left_mass + self.right_mass

    @property
    def log_abs_jacobian(self) -> float:
        """Return log absolute Jacobian for ``(parent, contrast) -> children``."""
        return 0.0

    def split(self, parent: float, contrast: float) -> tuple[float, float]:
        """Map a parent common mode and contrast to ordered child values.

        Args:
            parent: Finite parent coefficient.
            contrast: Finite canonical ``left - right`` contrast.

        Returns:
            Ordered ``(left, right)`` coefficients preserving weighted mass.
        """
        parent_value = _finite_scalar(parent, name="parent")
        contrast_value = _finite_scalar(contrast, name="contrast")
        left = parent_value + self.right_mass / self.total_mass * contrast_value
        right = parent_value - self.left_mass / self.total_mass * contrast_value
        return left, right

    def merge(self, left: float, right: float) -> tuple[float, float]:
        """Invert :meth:`split` and recover the parent and contrast.

        Args:
            left: Finite canonical left-child coefficient.
            right: Finite canonical right-child coefficient.

        Returns:
            ``(parent, contrast)`` using the configured masses.
        """
        left_value = _finite_scalar(left, name="left")
        right_value = _finite_scalar(right, name="right")
        parent = (self.left_mass * left_value + self.right_mass * right_value) / self.total_mass
        contrast = left_value - right_value
        return parent, contrast


class ContrastAuxiliaryProposal(Protocol):
    """Proposal density for the contrast affected by a split or merge."""

    def draw(
        self,
        state: ProductSpaceState,
        move: PartitionMove,
        coordinate_index: int,
        rng: np.random.Generator,
    ) -> float:
        """Draw the contrast value placed in the proposed state."""
        ...

    def log_density(
        self,
        value: float,
        state: ProductSpaceState,
        move: PartitionMove,
        coordinate_index: int,
    ) -> float:
        """Evaluate the normalized density used by :meth:`draw`."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class GaussianContrastProposal:
    """Independent Gaussian proposals for active and inactive contrasts.

    A split draws from the active arrays because its destination activates the
    affected contrast.  A merge draws from the inactive arrays because its
    destination deactivates that contrast.  Means and variances use the
    permanent contrast-coordinate order.

    Attributes:
        active_means: Mean used when a split activates each coordinate.
        active_variances: Positive variance used for split proposals.
        inactive_means: Mean used when a merge deactivates each coordinate.
        inactive_variances: Positive variance used for merge proposals.
    """

    active_means: np.ndarray
    active_variances: np.ndarray
    inactive_means: np.ndarray
    inactive_variances: np.ndarray

    def __post_init__(self) -> None:
        """Validate and freeze all Gaussian parameters."""
        active_means = _frozen_vector(self.active_means, name="active_means")
        inactive_means = _frozen_vector(self.inactive_means, name="inactive_means")
        active_variances = _positive_vector(
            self.active_variances,
            name="active_variances",
        )
        inactive_variances = _positive_vector(
            self.inactive_variances,
            name="inactive_variances",
        )
        shapes = {
            active_means.shape,
            active_variances.shape,
            inactive_means.shape,
            inactive_variances.shape,
        }
        if len(shapes) != 1:
            raise ValueError("Gaussian proposal arrays must have the same shape.")
        object.__setattr__(self, "active_means", active_means)
        object.__setattr__(self, "active_variances", active_variances)
        object.__setattr__(self, "inactive_means", inactive_means)
        object.__setattr__(self, "inactive_variances", inactive_variances)

    @classmethod
    def centered(
        cls,
        active_variances: npt.ArrayLike,
        inactive_variances: npt.ArrayLike,
    ) -> GaussianContrastProposal:
        """Construct zero-mean active and inactive Gaussian proposals.

        Args:
            active_variances: Positive permanent-coordinate variances for splits.
            inactive_variances: Positive permanent-coordinate variances for merges.

        Returns:
            Proposal with zero means and copied read-only parameter arrays.
        """
        active = _positive_vector(active_variances, name="active_variances")
        inactive = _positive_vector(inactive_variances, name="inactive_variances")
        if active.shape != inactive.shape:
            raise ValueError("active and inactive variances must have the same shape.")
        zeros = np.zeros(active.shape, dtype=float)
        return cls(zeros, active, zeros, inactive)

    def draw(
        self,
        state: ProductSpaceState,
        move: PartitionMove,
        coordinate_index: int,
        rng: np.random.Generator,
    ) -> float:
        """Draw one destination-aware Gaussian contrast value."""
        del state
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        mean, variance = self._parameters(move, coordinate_index)
        return float(rng.normal(mean, math.sqrt(variance)))

    def log_density(
        self,
        value: float,
        state: ProductSpaceState,
        move: PartitionMove,
        coordinate_index: int,
    ) -> float:
        """Evaluate one normalized destination-aware Gaussian density."""
        del state
        sample = _finite_scalar(value, name="value")
        mean, variance = self._parameters(move, coordinate_index)
        return -0.5 * (math.log(2.0 * math.pi * variance) + (sample - mean) ** 2 / variance)

    def _parameters(self, move: PartitionMove, coordinate_index: int) -> tuple[float, float]:
        """Return mean and variance for the destination status of a move."""
        if isinstance(coordinate_index, bool) or not isinstance(coordinate_index, int):
            raise TypeError("coordinate_index must be an integer.")
        if coordinate_index < 0 or coordinate_index >= self.active_means.size:
            raise IndexError("coordinate_index lies outside the proposal arrays.")
        if isinstance(move, SplitMove):
            return (
                float(self.active_means[coordinate_index]),
                float(self.active_variances[coordinate_index]),
            )
        if isinstance(move, MergeMove):
            return (
                float(self.inactive_means[coordinate_index]),
                float(self.inactive_variances[coordinate_index]),
            )
        raise TypeError("move must be a SplitMove or MergeMove.")


@dataclass(frozen=True, slots=True, eq=False)
class ContrastSwap:
    """Result of swapping one permanent contrast with an auxiliary value."""

    coordinates: np.ndarray
    reverse_auxiliary: float
    log_abs_jacobian: float = 0.0

    def __post_init__(self) -> None:
        """Freeze coordinates and validate scalar diagnostics."""
        object.__setattr__(
            self,
            "coordinates",
            _frozen_vector(self.coordinates, name="coordinates"),
        )
        object.__setattr__(
            self,
            "reverse_auxiliary",
            _finite_scalar(self.reverse_auxiliary, name="reverse_auxiliary"),
        )
        object.__setattr__(
            self,
            "log_abs_jacobian",
            _finite_scalar(self.log_abs_jacobian, name="log_abs_jacobian"),
        )


@dataclass(frozen=True, slots=True)
class TransportedProductSpaceTransition:
    """Diagnostic record for one auxiliary-variable partition transition."""

    state: ProductSpaceState
    candidate: ProductSpaceState
    move: PartitionMove | None
    accepted: bool
    coordinate_index: int | None
    forward_auxiliary: float | None
    reverse_auxiliary: float | None
    current_log_density: float
    candidate_log_density: float
    log_partition_q_forward: float
    log_partition_q_reverse: float
    log_auxiliary_q_forward: float
    log_auxiliary_q_reverse: float
    log_abs_jacobian: float
    log_acceptance_ratio: float


def swap_contrast_coordinate(
    coordinates: npt.ArrayLike,
    coordinate_index: int,
    forward_auxiliary: float,
) -> ContrastSwap:
    """Swap a permanent coordinate with a forward auxiliary value.

    Applying this function again to the returned coordinates with
    ``reverse_auxiliary`` exactly restores the source.  The swap matrix has
    determinant ``-1`` and therefore unit absolute Jacobian.

    Args:
        coordinates: Finite permanent root/contrast vector.
        coordinate_index: Coordinate replaced by the auxiliary value.
        forward_auxiliary: Finite value drawn for the proposed state.

    Returns:
        Proposed coordinates, reverse auxiliary, and zero log-Jacobian.
    """
    values = _frozen_vector(coordinates, name="coordinates")
    if isinstance(coordinate_index, bool) or not isinstance(coordinate_index, int):
        raise TypeError("coordinate_index must be an integer.")
    if coordinate_index < 0 or coordinate_index >= values.size:
        raise IndexError("coordinate_index lies outside coordinates.")
    auxiliary = _finite_scalar(forward_auxiliary, name="forward_auxiliary")
    proposed = values.copy()
    reverse_auxiliary = float(proposed[coordinate_index])
    proposed[coordinate_index] = auxiliary
    return ContrastSwap(proposed, reverse_auxiliary, 0.0)


def transported_partition_metropolis_step(
    layout: TreeContrastLayout,
    current: ProductSpaceState,
    *,
    log_density: LogAugmentedDensity,
    auxiliary_proposal: ContrastAuxiliaryProposal,
    rng: np.random.Generator,
) -> TransportedProductSpaceTransition:
    """Apply an exact split/merge move with an affected-contrast proposal.

    Args:
        layout: Permanent contrast layout and dyadic tree.
        current: Current fixed-dimensional augmented state.
        log_density: Complete normalized augmented target density.
        auxiliary_proposal: Normalized proposal used for the destination
            status of the affected contrast.
        rng: Caller-owned NumPy random generator.

    Returns:
        Accepted state and complete target, proposal, auxiliary, and Jacobian
        diagnostics.

    Raises:
        TypeError: If an interface or random generator is invalid.
        ValueError: If dimensions or density values violate kernel invariants.
    """
    if not isinstance(layout, TreeContrastLayout):
        raise TypeError("layout must be a TreeContrastLayout.")
    if not isinstance(current, ProductSpaceState):
        raise TypeError("current must be a ProductSpaceState.")
    if current.inner_coordinates.shape != (layout.coordinate_count,):
        raise ValueError("current inner coordinates do not match the contrast layout.")
    if not callable(log_density):
        raise TypeError("log_density must be callable.")
    if not callable(getattr(auxiliary_proposal, "draw", None)) or not callable(
        getattr(auxiliary_proposal, "log_density", None)
    ):
        raise TypeError("auxiliary_proposal must define draw and log_density methods.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    current.partition.validate(layout.tree)
    current_log_density = _checked_log_density(log_density(current), current=True)
    neighbors = enumerate_partition_neighbors(layout.tree, current.partition)
    if not neighbors:
        return TransportedProductSpaceTransition(
            state=current,
            candidate=current,
            move=None,
            accepted=False,
            coordinate_index=None,
            forward_auxiliary=None,
            reverse_auxiliary=None,
            current_log_density=current_log_density,
            candidate_log_density=current_log_density,
            log_partition_q_forward=0.0,
            log_partition_q_reverse=0.0,
            log_auxiliary_q_forward=0.0,
            log_auxiliary_q_reverse=0.0,
            log_abs_jacobian=0.0,
            log_acceptance_ratio=-math.inf,
        )

    neighbor = neighbors[int(rng.integers(len(neighbors)))]
    coordinate_index = _coordinate_index(layout, neighbor.move)
    forward_auxiliary = _finite_scalar(
        auxiliary_proposal.draw(current, neighbor.move, coordinate_index, rng),
        name="forward auxiliary draw",
    )
    swap = swap_contrast_coordinate(
        current.inner_coordinates,
        coordinate_index,
        forward_auxiliary,
    )
    candidate = ProductSpaceState(
        partition=neighbor.partition,
        inner_coordinates=swap.coordinates,
        outer_coefficients=current.outer_coefficients,
    )
    candidate_log_density = _checked_log_density(log_density(candidate), current=False)
    reverse_neighbors = enumerate_partition_neighbors(layout.tree, candidate.partition)
    reverse = next(
        (item for item in reverse_neighbors if item.partition == current.partition),
        None,
    )
    if reverse is None:  # pragma: no cover - protects future proposal extensions.
        raise ValueError("Proposed partition has no reverse split-or-merge move.")
    reverse_index = _coordinate_index(layout, reverse.move)
    if reverse_index != coordinate_index:  # pragma: no cover - tree invariant.
        raise ValueError("Forward and reverse moves must affect the same contrast coordinate.")

    log_auxiliary_q_forward = _checked_auxiliary_log_density(
        auxiliary_proposal.log_density(
            forward_auxiliary,
            current,
            neighbor.move,
            coordinate_index,
        ),
        forward=True,
    )
    log_auxiliary_q_reverse = _checked_auxiliary_log_density(
        auxiliary_proposal.log_density(
            swap.reverse_auxiliary,
            candidate,
            reverse.move,
            coordinate_index,
        ),
        forward=False,
    )
    log_acceptance_ratio = (
        candidate_log_density
        - current_log_density
        + reverse.log_q
        - neighbor.log_q
        + log_auxiliary_q_reverse
        - log_auxiliary_q_forward
        + swap.log_abs_jacobian
    )
    uniform = float(rng.random())
    log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
    accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
    return TransportedProductSpaceTransition(
        state=candidate if accepted else current,
        candidate=candidate,
        move=neighbor.move,
        accepted=accepted,
        coordinate_index=coordinate_index,
        forward_auxiliary=forward_auxiliary,
        reverse_auxiliary=swap.reverse_auxiliary,
        current_log_density=current_log_density,
        candidate_log_density=candidate_log_density,
        log_partition_q_forward=neighbor.log_q,
        log_partition_q_reverse=reverse.log_q,
        log_auxiliary_q_forward=log_auxiliary_q_forward,
        log_auxiliary_q_reverse=log_auxiliary_q_reverse,
        log_abs_jacobian=swap.log_abs_jacobian,
        log_acceptance_ratio=log_acceptance_ratio,
    )


def _coordinate_index(layout: TreeContrastLayout, move: PartitionMove) -> int:
    """Return the permanent contrast index activated or deactivated by a move."""
    if isinstance(move, SplitMove):
        return layout.contrast_index(move.node_id)
    if isinstance(move, MergeMove):
        return layout.contrast_index(move.parent_id)
    raise TypeError("move must be a SplitMove or MergeMove.")


def _positive_vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return a copied, read-only vector of positive finite values."""
    result = _frozen_vector(values, name=name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return result


def _finite_scalar(value: float, *, name: str) -> float:
    """Return one finite real scalar."""
    source = np.asarray(value)
    if source.shape != () or np.iscomplexobj(source):
        raise ValueError(f"{name} must be a real scalar.")
    result = float(source)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_scalar(value: float, *, name: str) -> float:
    """Return one positive finite real scalar."""
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _checked_auxiliary_log_density(value: float, *, forward: bool) -> float:
    """Validate a proposal log density, allowing zero reverse density."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("auxiliary log density must be a real scalar.") from error
    if math.isnan(result) or result == math.inf or (forward and result == -math.inf):
        if forward:
            raise ValueError("forward auxiliary log density must be finite.")
        raise ValueError("reverse auxiliary log density must be finite or negative infinity.")
    return result


__all__ = [
    "AdditiveCoefficientTransform",
    "ContrastAuxiliaryProposal",
    "ContrastSwap",
    "GaussianContrastProposal",
    "TransportedProductSpaceTransition",
    "swap_contrast_coordinate",
    "transported_partition_metropolis_step",
]
