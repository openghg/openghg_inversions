"""Select outer-region flux states and hold them at their prior mean.

Outer-region flux states are the retained flux states whose ``basis_group``
label is in a chosen set. By default the CO2 recipes sample them with the
inner states, uncentred, beside any boundary correction. This option selects
them once on the flux prior's state axis and, in ``"fixed"`` mode, holds the
selected states at their prior mean through state activity.

``"fixed"`` is a what-if for a known outer prior: the held states have zero
uncertainty, and nothing is added to the observation error for them. It is
not a stand-in for leaving the outer flux unsampled. That would marginalise
the outer states, adding ``H_O Sigma_OO H_O^T`` to the error covariance, which
this module does not do.

Recipe builders take the :class:`OuterFluxChoice` and resolve it with
:meth:`OuterFluxChoice.resolve_states` on the prior they sample (once for the
shared linked CO2/O2 state), then apply the result with
:meth:`OuterStates.fix_states` after exact-zero pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args

import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.boundary import Outputs
from openghg_inversions.models.state_activity import ResolvedStateActivity


OuterMode = Literal["fixed"]


@dataclass(frozen=True)
class OuterFluxChoice:
    """Outer-region flux option, chosen before any arrays are attached.

    Attributes:
        mode: ``"fixed"`` holds the selected outer states at their prior mean
            with zero uncertainty.
        outer_groups: ``basis_group`` labels of the outer states.
        fixed_sources: ``source`` labels of the outer states to hold, for
            example ``("ocean",)``. Empty holds every outer state; other outer
            states stay sampled as usual.
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
                in ``fixed_sources`` matches no outer state, or a held state has
                nonzero prior covariance with any other state.
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

        # Holding F at its prior mean while sampling the marginal prior of the
        # other states is exact only when F is uncorrelated with them.
        covariance = flux_prior.arithmetic_covariance.to_numpy()
        fixed_positions = np.flatnonzero(fixed.to_numpy())
        other_positions = np.flatnonzero(~fixed.to_numpy())
        if np.any(covariance[np.ix_(fixed_positions, other_positions)] != 0):
            raise ValueError("Fixed outer flux states must have zero prior covariance with every other state.")

        return OuterStates(
            choice=self,
            state_dim=flux_prior.state_dim,
            fixed=fixed.rename("fixed_outer"),
            prior_mean=mean,
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


@dataclass(frozen=True)
class OuterStates:
    """Held outer states on the flux prior's state axis, shared by every channel.

    Attributes:
        choice: The choice these selectors resolve.
        state_dim: Flux state dimension.
        fixed: Boolean mask of outer states ``F`` held at their prior mean.
        prior_mean: Labelled baseline arithmetic prior mean ``m``.
    """

    choice: OuterFluxChoice
    state_dim: str
    fixed: xr.DataArray
    prior_mean: xr.DataArray

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

    def outputs(self, prefix: str = "") -> Outputs:
        """Record the resolved choice's mode and selectors as run metadata.

        Args:
            prefix: Graph-name prefix of the recipe's variables.

        Returns:
            See :meth:`OuterFluxChoice.outputs`.
        """
        return self.choice.outputs(prefix)


__all__ = ["OuterFluxChoice", "OuterMode", "OuterStates"]
