"""Framework-independent target for Gamma--Beta partition inference.

The grouped :mod:`.gamma_beta` prior supplies a fixed maximum forest and
projectively consistent positive coordinates.  A partition chooses one active
frontier of that forest.  This module joins those pieces to an observation
model without depending on PyMC: it validates a finest-grid sensitivity design,
precomputes one design column per possible forest node, and evaluates predictions
and Gaussian log likelihoods for arbitrary valid frontiers.

Keeping this target independent of the sampler provides a small numerical oracle
for the experimental product-space implementation.  It does not prescribe a
partition prior or a transition kernel, and it is not integrated with production
RHIME inversion entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from .gamma_beta import GammaBetaForest, KappaStrategy
from .gamma_beta_coordinates import GammaBetaCoordinateLayout


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaProductSpaceTarget:
    """Validated positive partition-inference observation target.

    Construct targets with :meth:`from_grid`.  All arrays are copied and made
    read-only so sampler code cannot mutate the reference calculation.

    Attributes:
        observations: Finite observation vector.
        observation_mean: Fixed baseline or offset in observation space.
        observation_covariance: Symmetric positive-definite Gaussian residual
            covariance.
        coordinate_layout: Permanent grouped Gamma--Beta coordinate layout.
        node_design: Static observation-by-forest-node sensitivity matrix.
    """

    observations: npt.NDArray[np.float64]
    observation_mean: npt.NDArray[np.float64]
    observation_covariance: npt.NDArray[np.float64]
    coordinate_layout: GammaBetaCoordinateLayout
    node_design: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and freeze the complete numerical target."""
        if not isinstance(self.coordinate_layout, GammaBetaCoordinateLayout):
            raise TypeError("coordinate_layout must be a GammaBetaCoordinateLayout.")

        observations = _finite_array(
            self.observations,
            ndim=1,
            name="observations",
        )
        observation_mean = _finite_array(
            self.observation_mean,
            ndim=1,
            name="observation_mean",
        )
        if observation_mean.shape != observations.shape:
            raise ValueError("observation_mean must match observations.")

        covariance = _finite_array(
            self.observation_covariance,
            ndim=2,
            name="observation_covariance",
        )
        expected_covariance_shape = (observations.size, observations.size)
        if covariance.shape != expected_covariance_shape:
            raise ValueError(
                "observation_covariance must have shape "
                f"{expected_covariance_shape}."
            )
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("observation_covariance must be symmetric.")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "observation_covariance must be positive definite."
            ) from error

        node_design = _finite_array(self.node_design, ndim=2, name="node_design")
        expected_design_shape = (
            observations.size,
            len(self.coordinate_layout.forest.nodes),
        )
        if node_design.shape != expected_design_shape:
            raise ValueError(f"node_design must have shape {expected_design_shape}.")

        for name, values in (
            ("observations", observations),
            ("observation_mean", observation_mean),
            ("observation_covariance", covariance),
            ("node_design", node_design),
        ):
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @classmethod
    def from_grid(
        cls,
        *,
        observations: npt.ArrayLike,
        finest_grid_design: npt.ArrayLike,
        forest: GammaBetaForest,
        kappa_strategy: KappaStrategy,
        observation_covariance: npt.ArrayLike | None = None,
        observation_sd: npt.ArrayLike | float = 1.0,
        observation_mean: npt.ArrayLike | float = 0.0,
    ) -> GammaBetaProductSpaceTarget:
        """Build a target from a finest-grid sensitivity design.

        Args:
            observations: Finite one-dimensional observation vector.
            finest_grid_design: Finite array with shape
                ``(observation, row, column)``.  Columns already include prior
                flux when the inferred positive variables are scaling factors.
            forest: Fixed maximum grouped Gamma--Beta forest on the design grid.
            kappa_strategy: Positive concentration policy for all permanent
                split coordinates.
            observation_covariance: Optional full residual covariance.  When
                omitted, a diagonal covariance is built from ``observation_sd``.
            observation_sd: Positive scalar or observation-length vector used
                only when ``observation_covariance`` is omitted.
            observation_mean: Scalar or observation-length fixed baseline.

        Returns:
            Immutable framework-independent target.

        Raises:
            TypeError: If ``forest`` has the wrong type.
            ValueError: If observation, design, or covariance inputs are
                incompatible or invalid.
        """
        if not isinstance(forest, GammaBetaForest):
            raise TypeError("forest must be a GammaBetaForest.")
        observation_values = _finite_array(observations, ndim=1, name="observations")
        mean_values = _broadcast_vector(
            observation_mean,
            length=observation_values.size,
            name="observation_mean",
        )
        if observation_covariance is None:
            standard_deviations = _broadcast_vector(
                observation_sd,
                length=observation_values.size,
                name="observation_sd",
            )
            if np.any(standard_deviations <= 0.0):
                raise ValueError("observation_sd must be positive.")
            covariance = np.diag(np.square(standard_deviations))
        else:
            covariance = np.asarray(observation_covariance)

        layout = GammaBetaCoordinateLayout.from_forest(
            forest,
            kappa_strategy=kappa_strategy,
        )
        node_design = layout.node_design(finest_grid_design)
        if node_design.shape[0] != observation_values.size:
            raise ValueError(
                "finest_grid_design must have the same observation count as observations."
            )
        return cls(
            observations=observation_values,
            observation_mean=mean_values,
            observation_covariance=covariance,
            coordinate_layout=layout,
            node_design=node_design,
        )

    def prediction(
        self,
        active_node_ids: tuple[int, ...],
        group_root_scalings: npt.ArrayLike,
        split_fractions: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Return the observation prediction for one active frontier.

        Args:
            active_node_ids: Non-overlapping forest frontier covering every
                declared group.
            group_root_scalings: Positive scaling for every semantic group.
            split_fractions: Permanent Beta fraction for every possible split.

        Returns:
            Fixed observation mean plus active regional contributions.

        Raises:
            ValueError: If coordinates or the active frontier are invalid.
        """
        node_scalings = self.coordinate_layout.node_scalings(
            group_root_scalings,
            split_fractions,
        )
        # Rendering validates exact, non-overlapping group coverage.  The grid
        # itself is intentionally discarded because prediction uses node sums.
        self.coordinate_layout.render_frontier_scalings(
            active_node_ids,
            node_scalings,
        )
        active = np.asarray(active_node_ids, dtype=np.int64)
        return self.observation_mean + self.node_design[:, active] @ node_scalings[active]

    def log_likelihood(
        self,
        active_node_ids: tuple[int, ...],
        group_root_scalings: npt.ArrayLike,
        split_fractions: npt.ArrayLike,
    ) -> float:
        """Return the normalized Gaussian log likelihood for one state.

        Args:
            active_node_ids: Non-overlapping forest frontier covering every
                declared group.
            group_root_scalings: Positive scaling for every semantic group.
            split_fractions: Permanent Beta fraction for every possible split.

        Returns:
            Scalar Gaussian log likelihood including its normalization.
        """
        prediction = self.prediction(
            active_node_ids,
            group_root_scalings,
            split_fractions,
        )
        residual = self.observations - prediction
        cholesky = np.linalg.cholesky(self.observation_covariance)
        whitened = np.linalg.solve(cholesky, residual)
        log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
        return float(
            -0.5
            * (
                observations_size_log_two_pi(self.observations.size)
                + log_determinant
                + whitened @ whitened
            )
        )


def observations_size_log_two_pi(observation_count: int) -> float:
    """Return ``observation_count * log(2 pi)`` after validation."""
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise TypeError("observation_count must be an integer.")
    if observation_count < 0:
        raise ValueError("observation_count must be non-negative.")
    return observation_count * math.log(2.0 * math.pi)


def _broadcast_vector(
    values: npt.ArrayLike | float,
    *,
    length: int,
    name: str,
) -> npt.NDArray[np.float64]:
    """Broadcast a scalar or validate one finite vector."""
    source = np.asarray(values)
    if source.ndim == 0:
        source = np.full(length, source.item())
    result = _finite_array(source, ndim=1, name=name)
    if result.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape ({length},).")
    return result


def _finite_array(
    values: npt.ArrayLike,
    *,
    ndim: int,
    name: str,
) -> npt.NDArray[np.float64]:
    """Return one finite real floating-point array of known rank."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(source, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


__all__ = ["GammaBetaProductSpaceTarget"]
