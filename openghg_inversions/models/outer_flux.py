r"""Select outer-region flux states and hold or marginalise them.

Outer-region flux states are the retained flux states whose ``basis_group``
label is in a chosen set. By default the CO2 recipes sample them with the
inner states, uncentred, beside any boundary correction. This option selects
them once on the flux prior's state axis and holds the selected states ``O``
at their prior mean ``m_O`` through state activity, so ``H_O m_O`` joins the
fixed part of the forward model. The two modes differ in what happens to the
prior uncertainty ``Sigma_OO`` of the held states:

- ``"fixed"`` drops it. The held states have zero uncertainty and nothing is
  added to the observation error. It is a what-if for a known outer prior,
  not a stand-in for leaving the outer flux unsampled.
- ``"marginalise"`` keeps it as an error term. The low-rank covariance
  :math:`H_O \Sigma_{OO} H_O^T` (rank at most the number of held states) is
  added to the fixed aggregation-error covariance, beside its own low-rank
  factor, by :meth:`OuterStates.add_marginal_error`. For a linear Gaussian
  model with ``O`` uncorrelated with the other states, the posterior of the
  other states is then exactly their marginal posterior when ``O`` is
  sampled uncentred. Under the default LogNormal-copula prior it is an
  approximation: ``Sigma_OO`` is the prior's arithmetic covariance, so the
  outer contribution is replaced by a Gaussian with the same first two
  moments. Correlated outer states are conditioned on the others, below.

``"marginalise"`` also accepts outer states correlated with the other states
``I`` (cross-covariance ``Sigma_OI``). The outer states are then replaced by
their best linear predictor from ``I`` under the prior's arithmetic moments,
``x_O = m_O + B (x_I - m_I) + r`` with ``B = Sigma_OI Sigma_II^+``, so

- the inner design becomes ``H_I + H_O B`` (:meth:`OuterStates.condition_sensitivity`);
- the fixed term gains ``-H_O B m_I``;
- the error covariance gains ``H_O Sigma_O|I H_O^T`` with
  ``Sigma_O|I = Sigma_OO - B Sigma_IO``.

The prior of ``I`` is unchanged. For a Gaussian prior this is exact (the
residual ``r`` is independent of ``x_I``). Under the LogNormal copula ``r`` is
uncorrelated with ``x_I`` but not independent of it, a moment-level
approximation of the same kind as the coherent reduction. States held by state
activity or pruning are part of ``I`` at their fixed values. ``"fixed"`` still
requires zero cross-covariance, because holding ``O`` at ``m_O`` exactly
would need the inner prior conditioned on it.

Recipe builders take the :class:`OuterFluxChoice` and resolve it with
:meth:`OuterFluxChoice.resolve_states` on the prior they sample (once for the
shared linked CO2/O2 state). They apply the result with
:meth:`OuterStates.condition_sensitivity` before exact-zero pruning,
:meth:`OuterStates.fix_states` after it, and
:meth:`OuterStates.add_marginal_error` before the likelihood is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, get_args

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.boundary import Outputs
from openghg_inversions.models.state_activity import ResolvedStateActivity
from openghg_inversions.observation_error import (
    DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR,
    AggregationError,
)


OuterMode = Literal["fixed", "marginalise"]

_RANK_DIM = "agg_rank"


@dataclass(frozen=True)
class OuterFluxChoice:
    """Outer-region flux option, chosen before any arrays are attached.

    Attributes:
        mode: ``"fixed"`` holds the selected outer states at their prior mean
            with zero uncertainty. ``"marginalise"`` also holds them at their
            prior mean and adds :math:`H_O \\Sigma_{OO} H_O^T` to the error
            covariance, keeping their prior uncertainty without sampling them.
        outer_groups: ``basis_group`` labels of the outer states.
        fixed_sources: ``source`` labels of the outer states to hold (or
            marginalise), for example ``("ocean",)``. Empty selects every
            outer state; other outer states stay sampled as usual.
        group_coord: State coordinate holding the basis-group labels.
        source_coord: State coordinate holding the source labels.
    """

    mode: OuterMode
    outer_groups: tuple[str, ...]
    fixed_sources: tuple[str, ...] = ()
    group_coord: str = "basis_group"
    source_coord: str = "source"

    def __post_init__(self) -> None:
        if self.mode not in get_args(OuterMode):
            raise ValueError(f"Unknown outer-flux mode {self.mode!r}; expected one of {get_args(OuterMode)}.")
        object.__setattr__(self, "outer_groups", tuple(self.outer_groups))
        object.__setattr__(self, "fixed_sources", tuple(self.fixed_sources))
        if not self.outer_groups:
            raise ValueError("The outer-flux option needs at least one outer basis group.")

    def outputs(self, prefix: str = "") -> Outputs:
        """Record this option's mode and selectors as run metadata.

        Args:
            prefix: Graph-name prefix of the recipe's variables.

        Returns:
            No roles or units, and metadata recording the mode and selectors.
        """
        metadata = {
            "outer_flux_mode": self.mode,
            "outer_groups": list(self.outer_groups),
            "outer_fixed_sources": list(self.fixed_sources),
        }
        return Outputs({}, {}, metadata).prefixed(prefix)

    def resolve_states(self, flux_prior: CorrelatedPrior) -> OuterStates:
        """Select the held outer states on the flux prior's state axis.

        This is the state-axis work, done once per prior even for linked
        channels.

        Args:
            flux_prior: The flux prior the model samples. Its mean carries
                the state labels and selector coordinates.

        Returns:
            The held outer selector and the prior mean.

        Raises:
            ValueError: If a selector coordinate is missing or not on the state
                axis, any label in ``outer_groups`` matches no state, any label
                in ``fixed_sources`` matches no outer state, or, in ``"fixed"``
                mode, a held state has nonzero prior covariance with any other
                state.
        """
        mean = flux_prior.mean

        # Selector labels are small; materialise them once so lazy coordinates
        # can be matched and masked.
        groups = _state_labels(mean, self.group_coord).compute()
        missing_groups = _missing(self.outer_groups, groups)
        if missing_groups:
            raise ValueError(f"No flux state has {self.group_coord} {missing_groups!r}.")
        outer = groups.isin(self.outer_groups)
        fixed = outer
        if self.fixed_sources:
            sources = _state_labels(mean, self.source_coord).compute()
            missing_sources = _missing(self.fixed_sources, sources.where(outer, drop=True))
            if missing_sources:
                raise ValueError(
                    f"No outer flux state (with {self.group_coord} in {self.outer_groups!r}) has "
                    f"{self.source_coord} {missing_sources!r}."
                )
            fixed = outer & sources.isin(self.fixed_sources)

        covariance = flux_prior.arithmetic_covariance.to_numpy()
        fixed_positions = np.flatnonzero(fixed.to_numpy())
        other_positions = np.flatnonzero(~fixed.to_numpy())
        cross = covariance[np.ix_(fixed_positions, other_positions)]
        correlated = bool(np.any(cross != 0))
        # Holding F at its prior mean while sampling the marginal prior of the
        # other states is exact only when F is uncorrelated with them.
        if correlated and self.mode == "fixed":
            raise ValueError("Fixed outer flux states must have zero prior covariance with every other state.")

        prior_factor = regression = None
        if self.mode == "marginalise":
            conditional = covariance[np.ix_(fixed_positions, fixed_positions)]
            if correlated:
                regression = _regression(covariance[np.ix_(other_positions, other_positions)], cross)
                conditional = conditional - regression @ cross.T
            prior_factor = _psd_factor(conditional)

        return OuterStates(
            choice=self,
            state_dim=flux_prior.state_dim,
            fixed=fixed.rename("fixed_outer"),
            prior_mean=mean,
            prior_factor=prior_factor,
            regression=regression,
        )


def _missing(requested: tuple[str, ...], labels: xr.DataArray) -> list[str]:
    """Return the requested labels, in request order, that ``labels`` lacks."""
    present = set(labels.to_numpy().tolist())
    return [label for label in requested if label not in present]


def _state_labels(mean: xr.DataArray, name: str) -> xr.DataArray:
    """Return a selector coordinate that lies on the state axis."""
    labels = mean.coords.get(name)
    if labels is None or labels.dims != mean.dims:
        raise ValueError(f"Outer-flux selection needs a {name!r} coordinate on the flux state axis.")
    return labels.reset_coords(drop=True)


def _psd_factor(covariance: np.ndarray) -> np.ndarray:
    """Return ``L`` with ``L @ L.T == covariance``, dropping only numerically zero modes.

    An eigendecomposition is used rather than a Cholesky factor, so held states
    with zero or exactly collinear prior variance are accepted. Eigenvalues
    below a scale-relative roundoff tolerance are treated as zero; a clearly
    negative eigenvalue means the covariance is not PSD and is rejected.
    """
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.size == 0:
        return np.zeros((0, 0))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = covariance.shape[0] * np.finfo(float).eps * scale
    if eigenvalues[0] < -1e-8 * scale:
        raise ValueError("The prior covariance of the marginalised outer states is not positive semidefinite.")
    keep = eigenvalues > tolerance
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])


def _regression(inner_covariance: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """Return ``B = Sigma_OI Sigma_II^+``, the best linear predictor of ``O`` from ``I``.

    A Hermitian pseudo-inverse with a scale-relative cutoff keeps
    zero-variance or collinear other states from dominating.
    """
    inner_covariance = 0.5 * (inner_covariance + inner_covariance.T)
    cutoff = inner_covariance.shape[0] * np.finfo(float).eps
    return cross @ np.linalg.pinv(inner_covariance, rcond=cutoff, hermitian=True)


@dataclass(frozen=True)
class OuterStates:
    """Held outer states on the flux prior's state axis, shared by every channel.

    Attributes:
        choice: The choice these selectors resolve.
        state_dim: Flux state dimension.
        fixed: Boolean mask of outer states ``F`` held at their prior mean.
        prior_mean: Labelled baseline arithmetic prior mean ``m``.
        prior_factor: In ``"marginalise"`` mode, a factor ``L_F`` of the held
            states' arithmetic prior covariance given the other states,
            ``L_F L_F^T = Sigma_F|I`` (``Sigma_FF`` when uncorrelated), with
            rows in state order of ``F``; ``None`` in ``"fixed"`` mode.
        regression: In ``"marginalise"`` mode with correlated held states,
            ``B = Sigma_FI Sigma_II^+`` with rows in state order of ``F`` and
            columns in state order of the other states; otherwise ``None``.
    """

    choice: OuterFluxChoice
    state_dim: str
    fixed: xr.DataArray
    prior_mean: xr.DataArray
    prior_factor: np.ndarray | None = field(default=None, compare=False)
    regression: np.ndarray | None = field(default=None, compare=False)

    def condition_sensitivity(
        self, flux_sensitivity: xr.DataArray, fixed_prior_contribution: xr.DataArray, *, output_dim: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Move correlated outer response onto the other states' design.

        For held states correlated with the other states, the outer
        contribution is ``H_F (m_F + B (x_I - m_I))``. Holding ``F`` at ``m_F``
        then needs ``H_F B`` added to the other states' columns and
        ``-H_F B m_I`` added to the fixed contribution. Without correlation
        (or in ``"fixed"`` mode) the inputs are returned unchanged.

        Args:
            flux_sensitivity: Flux sensitivity on ``(output_dim, state_dim)``,
                possibly sparse, before exact-zero pruning.
            fixed_prior_contribution: Fixed contribution on ``output_dim``.
            output_dim: Observation dimension.

        Returns:
            The conditioned sensitivity (dense when changed) and fixed
            contribution.

        Raises:
            ValueError: If the sensitivity's state labels differ from the prior's.
        """
        if self.regression is None:
            return flux_sensitivity, fixed_prior_contribution
        sensitivity, fixed, mean = xr.align(
            flux_sensitivity, self.fixed.reset_coords(drop=True), self.prior_mean.reset_coords(drop=True),
            join="exact", copy=False,
        )
        sensitivity = to_dense(sensitivity).transpose(output_dim, self.state_dim)
        held = fixed.to_numpy()
        values = np.array(sensitivity.values, dtype=np.float64)
        moved = values[:, held] @ self.regression
        values[:, ~held] += moved
        shift = -(moved @ mean.to_numpy()[~held])
        conditioned = sensitivity.copy(data=values)
        offset = sensitivity.isel({self.state_dim: 0}, drop=True).reset_coords(drop=True).copy(data=shift)
        contribution = (fixed_prior_contribution + offset).transpose(*fixed_prior_contribution.dims)
        return conditioned, contribution.assign_attrs(fixed_prior_contribution.attrs).rename(
            fixed_prior_contribution.name,
        )

    def fix_states(self, activity: ResolvedStateActivity) -> ResolvedStateActivity:
        """Hold fixed outer states at their prior mean in a resolved activity.

        Args:
            activity: Activity resolved against the flux sensitivity, possibly
                with the caller's own policy.

        Returns:
            The activity with ``F`` inactive at ``m_F``; other states keep
            their activity and fixed values.

        Raises:
            ValueError: If the activity's state labels differ from the prior's,
                or the caller's policy explicitly fixes a held state at a value
                other than its prior mean. A state the policy made inactive
                counts as explicitly fixed even if its sensitivity column is
                also exactly zero; one inactive only through exact-zero
                pruning does not.
        """
        # Without a recorded policy, any inactive state with a response was
        # fixed explicitly; states removed only by exact-zero pruning were not.
        explicit = (
            activity.explicitly_inactive
            if activity.explicitly_inactive is not None
            else ~activity.active & ~activity.zero_sensitivity
        )
        active, fixed, prior_mean, caller_value, explicit = xr.align(
            activity.active,
            self.fixed,
            self.prior_mean.reset_coords(drop=True),
            activity.fixed_value.reset_coords(drop=True),
            explicit.reset_coords(drop=True),
            join="exact",
            copy=False,
        )
        conflict = fixed & explicit & (caller_value != prior_mean)
        if bool(conflict.any()):
            labels = conflict[conflict.dims[0]].where(conflict, drop=True).to_numpy().tolist()
            raise ValueError(
                f"State activity fixes outer flux state(s) {labels!r} at values other than their prior "
                "mean, which the fixed outer-flux option holds them at."
            )
        return replace(
            activity,
            active=(active & ~fixed).rename(activity.active.name),
            fixed_value=xr.where(fixed, prior_mean, activity.fixed_value).rename(activity.fixed_value.name),
        )

    def marginal_error_factor(self, flux_sensitivity: xr.DataArray, *, output_dim: str) -> xr.DataArray:
        """Return the observation-space factor ``H_F L_F`` of the marginalised outer flux.

        Its outer product ``H_F Sigma_F|I H_F^T`` is the covariance the held
        states would add to the observations if they were sampled, beyond
        what the other states predict (``Sigma_FF`` when uncorrelated).

        Args:
            flux_sensitivity: Eager or lazy flux sensitivity on
                ``(output_dim, state_dim)``, aligned with the observations;
                possibly sparse.
            output_dim: Observation dimension.

        Returns:
            Dense factor on ``(output_dim, "agg_rank")``; its column count is
            the numerical rank of ``Sigma_F|I``.

        Raises:
            ValueError: If the mode is not ``"marginalise"``, or the
                sensitivity's state labels differ from the prior's.
        """
        if self.prior_factor is None:
            raise ValueError('Only "marginalise" outer flux states add an observation-error term.')
        sensitivity, fixed = xr.align(
            flux_sensitivity, self.fixed.reset_coords(drop=True), join="exact", copy=False,
        )
        held = to_dense(sensitivity.isel({self.state_dim: np.flatnonzero(fixed.to_numpy())}))
        values = np.asarray(held.transpose(output_dim, self.state_dim).values, dtype=np.float64)
        factor = values @ self.prior_factor
        return xr.DataArray(
            factor,
            dims=(output_dim, _RANK_DIM),
            coords={output_dim: held.coords[output_dim], _RANK_DIM: np.arange(factor.shape[1])},
            name=LOW_RANK_FACTOR,
        )

    def add_marginal_error(
        self, flux_sensitivity: xr.DataArray, aggregation_error: AggregationError, *, output_dim: str,
    ) -> AggregationError:
        """Add the marginalised outer covariance to a fixed aggregation error.

        In ``"fixed"`` mode the aggregation error is returned unchanged. In
        ``"marginalise"`` mode ``H_F L_F`` is appended to the low-rank factor
        (a ``"none"`` or ``"diagonal"`` error becomes low-rank with that
        factor) or, for a dense error, ``H_F Sigma_F|I H_F^T`` is added to the
        dense covariance. The marginal variance grows by the row sums of
        squares of ``H_F L_F``.

        Args:
            flux_sensitivity: Flux sensitivity on ``(output_dim, state_dim)``,
                in the row order of ``aggregation_error``.
            aggregation_error: Fixed aggregation error of the likelihood.
            output_dim: Observation dimension.

        Returns:
            The aggregation error the likelihood should use.
        """
        if self.choice.mode == "fixed":
            return aggregation_error
        factor = self.marginal_error_factor(flux_sensitivity, output_dim=output_dim)
        values = factor.to_numpy()
        marginal_variance = np.asarray(aggregation_error.marginal_variance, dtype=np.float64) + np.einsum(
            "ij,ij->i", values, values,
        )
        if aggregation_error.mode == "dense":
            assert aggregation_error.covariance is not None
            covariance = aggregation_error.covariance + values @ values.T
            return replace(aggregation_error, covariance=covariance, marginal_variance=marginal_variance)
        if aggregation_error.mode == "low_rank":
            assert aggregation_error.factor is not None
            existing = aggregation_error.factor.transpose(output_dim, ...)
            rank_dim = existing.dims[1]
            combined = np.hstack([np.asarray(existing.values, dtype=np.float64), values])
            factor = xr.DataArray(
                combined,
                dims=(output_dim, rank_dim),
                coords={output_dim: existing.coords[output_dim], rank_dim: np.arange(combined.shape[1])},
                name=existing.name,
            )
            return replace(aggregation_error, factor=factor, marginal_variance=marginal_variance)
        if aggregation_error.mode == "diagonal":
            assert aggregation_error.diagonal_variance is not None
            diagonal = aggregation_error.diagonal_variance
        else:
            diagonal = xr.DataArray(
                np.zeros(values.shape[0]), dims=output_dim, coords={output_dim: factor.coords[output_dim]},
            )
        return AggregationError(
            mode="low_rank",
            marginal_variance=marginal_variance,
            factor=factor,
            diagonal_variance=diagonal.rename(DIAGONAL_RESIDUAL_VARIANCE),
        )


__all__ = ["OuterFluxChoice", "OuterMode", "OuterStates"]
