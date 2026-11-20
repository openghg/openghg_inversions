"""Outer-region flux corrections that leave the atmospheric level to the boundary.

Outer-region fluxes and the boundary both raise every observation by a similar
amount, so a model that samples both cannot separate their common level. This
option keeps the outer flux states in the ordinary retained flux state and
removes their level instead: the sampled outer columns of ``H`` are centred on
the fit's rows, and the level they lose is restored at the prior mean as one
fixed scalar. The boundary's mean shift is then the only sampled level, and
recipes reject this option without one.

With equal-site row weights ``w`` (``w.sum() == 1``), prior mean ``m`` and
``mu_S = w @ H_S`` for the sampled outer columns ``S``::

    H'_S = H_S - 1 mu_S                (centred sampled outer columns)
    f'   = f + (mu_S @ m_S) 1          (level restored at the prior mean)
    H x + f = H_I x_I + H_F m_F + H'_S x_S + f' + [mu_S @ (x_S - m_S)] 1

Fixed outer states ``F`` are held at their prior mean through state activity,
so their raw columns stay. The bracketed level is what the boundary mean shift
absorbs. In additive mode every outer state is fixed and centred standard
normal outer patterns are added as their own term.

Work is split by what it depends on. :meth:`OuterFluxChoice.resolve_states`
reads the flux prior once and selects states by ``basis_group`` and
``source`` labels; :meth:`OuterStates.bind_rows` computes the row weights and
column means for one channel's rows. Recipes prune exact-zero columns of the
raw ``H`` first, then call :meth:`OuterFlux.centre` on the retained columns
and :meth:`OuterFlux.restore_level` on the fixed term, so pruning is
unchanged by this option.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args

import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.boundary import AdditiveBoundary, AnomalyScalingBoundary, Bound, Outputs
from openghg_inversions.models.components import (
    AffineBlock,
    AffineConcentrationTerm,
    add_linear_component,
)
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
)


OuterMode = Literal["fixed", "multiplicative", "additive"]
ADDITIVE_SOURCES = {"FF", "net_bio"}
DENSITY_UNITS = "mol m-2 s-1"


@dataclass(frozen=True)
class OuterFluxChoice:
    """Outer-region flux option, chosen before any arrays are attached.

    Attributes:
        mode: ``"fixed"`` holds every outer state at its prior mean;
            ``"multiplicative"`` samples outer states not in
            ``fixed_sources`` and centres their columns; ``"additive"`` holds
            every outer state at its prior mean and adds centred outer
            patterns.
        outer_groups: ``basis_group`` labels of the outer states.
        fixed_sources: ``source`` labels of outer states held at their prior
            mean in multiplicative mode, for example ``("ocean",)``.
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
        if self.fixed_sources and self.mode != "multiplicative":
            raise ValueError("fixed_sources applies only to multiplicative outer flux; other modes fix every outer state.")

    def resolve_states(self, flux_prior: CorrelatedPrior) -> OuterStates:
        """Select the sampled and fixed outer states on the flux prior's state axis.

        This is the state-axis work, done once per prior even for linked
        channels.

        Args:
            flux_prior: The flux prior the model samples. Its mean carries
                the state labels and selector coordinates.

        Returns:
            Sampled and fixed outer selectors and the prior mean.

        Raises:
            ValueError: If a selector coordinate is missing or not on the state
                axis, no state is in ``outer_groups``, or a fixed outer state
                has nonzero prior covariance with any other state.
        """
        mean = flux_prior.mean
        state_dim = flux_prior.state_dim

        outer = _state_labels(mean, self.group_coord).isin(self.outer_groups)
        if not bool(outer.any()):
            raise ValueError(f"No flux state has {self.group_coord} in {self.outer_groups!r}.")
        if self.mode == "multiplicative":
            held = (
                _state_labels(mean, self.source_coord).isin(self.fixed_sources)
                if self.fixed_sources else xr.zeros_like(outer)
            )
            fixed = outer & held
        else:
            fixed = outer
        sampled = outer & ~fixed

        # Holding F at its prior mean while sampling the marginal prior of the
        # other states is exact only when F is uncorrelated with them.
        covariance = flux_prior.arithmetic_covariance.to_numpy()
        fixed_positions = np.flatnonzero(fixed.to_numpy())
        other_positions = np.flatnonzero(~fixed.to_numpy())
        if np.any(covariance[np.ix_(fixed_positions, other_positions)] != 0):
            raise ValueError("Fixed outer flux states must have zero prior covariance with every other state.")

        return OuterStates(
            choice=self,
            state_dim=state_dim,
            sampled=sampled.rename("sampled_outer"),
            fixed=fixed.rename("fixed_outer"),
            prior_mean=mean,
        )


def _state_labels(mean: xr.DataArray, name: str) -> xr.DataArray:
    """Return a selector coordinate that lies on the state axis."""
    labels = mean.coords.get(name)
    if labels is None or labels.dims != mean.dims:
        raise ValueError(f"Outer-flux selection needs a {name!r} coordinate on the flux state axis.")
    return labels.reset_coords(drop=True)


@dataclass(frozen=True)
class OuterStates:
    """Outer selectors on the flux prior's state axis, shared by every channel.

    Attributes:
        choice: The choice these selectors resolve.
        state_dim: Flux state dimension.
        sampled: Boolean mask of sampled outer states ``S``, whose columns are
            centred.
        fixed: Boolean mask of outer states ``F`` held at their prior mean.
        prior_mean: Labelled baseline arithmetic prior mean ``m``.
    """

    choice: OuterFluxChoice
    state_dim: str
    sampled: xr.DataArray
    fixed: xr.DataArray
    prior_mean: xr.DataArray

    def bind_rows(
        self,
        observations: xr.DataArray,
        flux_sensitivity: xr.DataArray,
        *,
        additive_design: xr.DataArray | None = None,
        additive_density_sd: xr.DataArray | None = None,
    ) -> OuterFlux:
        """Compute one channel's row weights, column means and level.

        Preparation guarantees that ``flux_sensitivity`` is in the
        observations' units and on the prior's state labels, so neither is
        rechecked here. The column means stay lazy.

        Args:
            observations: One channel's observation rows, on their row
                dimension (``observations.dims[0]``), with a ``site``
                coordinate.
            flux_sensitivity: That channel's raw ``H`` on the same rows and
                the full flux state axis.
            additive_design: Additive mode only: concentration responses of
                the outer patterns on ``(row, "outer_pattern")``, with an
                ``outer_source`` label of ``"FF"`` or ``"net_bio"`` per
                pattern and the observations' units.
            additive_density_sd: Additive mode only: positive density SD per
                ``outer_pattern`` with units convertible to ``mol m-2 s-1``.
                The design columns are responses to one SD.

        Returns:
            The bound option for this channel.

        Raises:
            ValueError: If observations lack row-aligned site labels, the
                sensitivity or additive design rows differ from the
                observation rows, the density SDs and design patterns differ,
                or the additive inputs are missing, supplied outside additive
                mode, or invalid.
        """
        row = str(observations.dims[0])
        site = observations.coords.get("site")
        if site is None or site.dims != (row,):
            raise ValueError("Equal-site centring needs a site coordinate on the observation rows.")

        # Each site has equal total weight, spread evenly over its rows.
        _, site_index, site_counts = np.unique(site.to_numpy(), return_inverse=True, return_counts=True)
        weights = xr.DataArray(
            1 / (site_counts.size * site_counts[site_index]), dims=row, coords={row: observations[row]},
        )

        # xr.dot inner-joins rows; a row mismatch must fail rather than shrink the means.
        weights, flux_sensitivity = xr.align(weights, flux_sensitivity, join="exact", copy=False)
        column_mean = xr.dot(weights, flux_sensitivity, dim=row).where(self.sampled, 0)
        level_compensation = xr.dot(column_mean, self.prior_mean, dim=self.state_dim)

        data = xr.Dataset({
            "centering_weights": weights,
            "outer_column_mean": column_mean.reset_coords(drop=True),
            "sampled_outer": self.sampled.reset_coords(drop=True),
            "fixed_outer": self.fixed.reset_coords(drop=True),
            "level_compensation": level_compensation.reset_coords(drop=True),
        })

        additive = self.choice.mode == "additive"
        if additive != (additive_design is not None) or additive != (additive_density_sd is not None):
            raise ValueError("Additive outer flux needs additive_design and additive_density_sd; other modes take neither.")
        if additive_design is not None and additive_density_sd is not None:
            data = data.merge(
                _additive_patterns(weights, additive_design, additive_density_sd, units=observations.attrs.get("units")),
            )
        return OuterFlux(self, data.assign_attrs(units=observations.attrs.get("units")))

    def fix_states(self, activity: ResolvedStateActivity) -> ResolvedStateActivity:
        """Hold fixed outer states at their prior mean in a resolved activity.

        Args:
            activity: Activity resolved against the flux sensitivity, possibly
                with the caller's own policy.

        Returns:
            The activity with ``F`` inactive at ``m_F``; other states keep
            their activity and fixed values.

        Raises:
            ValueError: If the activity's state labels differ from the prior's.
        """
        active, fixed, prior_mean = xr.align(
            activity.active, self.fixed, self.prior_mean.reset_coords(drop=True), join="exact", copy=False,
        )
        return replace(
            activity,
            active=(active & ~fixed).rename(activity.active.name),
            fixed_value=xr.where(fixed, prior_mean, activity.fixed_value).rename(activity.fixed_value.name),
        )


def _additive_patterns(
    weights: xr.DataArray,
    design: xr.DataArray,
    density_sd: xr.DataArray,
    *,
    units: str | None,
) -> xr.Dataset:
    """Check one channel's additive outer patterns and centre them on its rows."""
    row = str(weights.dims[0])
    if design.dims != (row, "outer_pattern") or not design.sizes["outer_pattern"]:
        raise ValueError(f"Additive outer patterns need a nonempty ({row!r}, 'outer_pattern') design.")
    if design.attrs.get("units") != units:
        raise ValueError("Additive outer patterns must use the observations' concentration units.")
    if not design.get_index("outer_pattern").is_unique:
        raise ValueError("Additive outer pattern labels must be unique.")
    source = design.coords.get("outer_source")
    if source is None or source.dims != ("outer_pattern",) or not set(source.to_numpy().tolist()) <= ADDITIVE_SOURCES:
        raise ValueError("Additive outer patterns need outer_source labels of 'FF' or 'net_bio'.")
    if density_sd.dims != ("outer_pattern",) or not density_sd.attrs.get("units"):
        raise ValueError("Additive density SDs need outer_pattern labels and declared units.")
    # Both joins below would otherwise fill missing rows or widths with NaN.
    weights, design = xr.align(weights, design, join="exact", copy=False)
    design, density_sd = xr.align(design, density_sd, join="exact", copy=False)
    density_sd = density_sd.pint.quantify().pint.to(DENSITY_UNITS).pint.dequantify().assign_attrs(units=DENSITY_UNITS)
    if not np.isfinite(density_sd.to_numpy()).all() or bool((density_sd <= 0).any()):
        raise ValueError("Additive density SDs must be finite and positive.")

    pattern_mean = xr.dot(weights, design, dim=row)
    return xr.Dataset({
        "additive_centered": (design - pattern_mean).assign_attrs(units=units),
        "additive_mean": pattern_mean.assign_attrs(units=units),
        "additive_density_sd": density_sd,
    })


@dataclass(frozen=True)
class OuterFlux:
    """The outer-flux option bound to one channel's rows.

    ``data`` is an ``xr.Dataset`` with:

    * ``centering_weights[row]``: equal-site row weights summing to one;
    * ``outer_column_mean[state]``: ``mu``, zero off the sampled outer states;
    * ``sampled_outer[state]`` and ``fixed_outer[state]``: the selectors;
    * ``level_compensation[]``: ``mu_S @ m_S`` in concentration units;
    * additive mode only: ``additive_centered[row, outer_pattern]``,
      ``additive_mean[outer_pattern]`` and ``additive_density_sd[outer_pattern]``.

    Its ``units`` attribute is the observations' concentration unit. Values
    may be lazy until the recipe materializes them with its other inputs.

    Attributes:
        states: State selectors, shared by reference between linked channels.
        data: The bound record described above.
    """

    states: OuterStates
    data: xr.Dataset

    @property
    def choice(self) -> OuterFluxChoice:
        """Return the choice this option was bound from."""
        return self.states.choice

    def centre(self, prepared: PreparedLinearSensitivity, *, rows: xr.DataArray | None = None) -> PreparedLinearSensitivity:
        """Subtract the outer column means from a pruned flux sensitivity.

        Pruning runs first on the raw ``H``. A column that is zero there has a
        zero mean, so it would stay zero after centring; a sampled outer column
        that only becomes zero after centring stays in the retained design.
        Means are taken by retained position, so this works whether or not
        pruning renamed the state dimension.

        Args:
            prepared: The raw flux sensitivity after exact-zero pruning, on the
                full state axis this option was resolved on.
            rows: Optional boolean mask on the output dimension; only these
                rows are centred. Linked recipes pass one channel's rows.

        Returns:
            The same preparation with centred retained columns and an
            unchanged ``removed`` mask.

        Raises:
            ValueError: If the preparation's state labels differ from the prior's.
        """
        column_mean, _ = xr.align(
            self.data.outer_column_mean, prepared.removed.reset_coords(drop=True), join="exact", copy=False,
        )
        sensitivity = prepared.sensitivity
        retained_dim = next(str(dim) for dim in sensitivity.dims if dim != prepared.output_dim)
        offset = xr.DataArray(column_mean.to_numpy()[prepared.retained_indices], dims=retained_dim)
        if rows is not None:
            offset = offset * rows.astype(offset.dtype)
        centred = (sensitivity - offset).transpose(*sensitivity.dims).assign_attrs(sensitivity.attrs)
        return replace(prepared, sensitivity=centred)

    def restore_level(self, fixed_prior_contribution: xr.DataArray, *, rows: xr.DataArray | None = None) -> xr.DataArray:
        """Add the level removed by :meth:`centre` back at the prior mean.

        Args:
            fixed_prior_contribution: The recipe's fixed concentration term.
            rows: Optional boolean mask on the output dimension, as for
                :meth:`centre`.

        Returns:
            The fixed term plus ``mu_S @ m_S`` on the selected rows, with its
            name and attributes kept.
        """
        level = self.data.level_compensation
        if rows is not None:
            level = level * rows.astype(level.dtype)
        restored = fixed_prior_contribution + level
        return restored.rename(fixed_prior_contribution.name).assign_attrs(fixed_prior_contribution.attrs)

    def add(self, *, output_dim: str, prefix: str = "") -> AffineConcentrationTerm:
        """Add the centred additive outer patterns to the current model.

        Additive mode only. Every pattern is kept, including one that is
        constant over the rows, so its coefficient keeps its standard normal
        prior.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.

        Returns:
            The pattern term, with standard normal coefficients.
        """
        design = self.data.additive_centered.copy(deep=False)
        design["outer_pattern"].attrs = {"units": "1", "prior": "independent standard normal"}
        every_pattern = PreparedLinearSensitivity(
            sensitivity=design,
            removed=xr.zeros_like(design.outer_pattern, dtype=bool),
            output_dim=output_dim,
        )
        pattern = add_linear_component(
            every_pattern,
            data_name=f"{prefix}outer_pattern_operator",
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            var_name=f"{prefix}outer_pattern_coefficient",
            output_name=f"{prefix}outer_pattern_contribution",
            output_dim=output_dim,
        )
        return AffineConcentrationTerm.absolute(
            pattern.output,
            fixed_contribution=xr.zeros_like(design.isel(outer_pattern=0, drop=True)),
            blocks=(AffineBlock(design, pattern.state),),
            sampled_rvs=(pattern.latent,) if pattern.latent is not None else (),
        )

    def outputs(self, prefix: str = "") -> Outputs:
        """Declare the outer pattern roles and units and this option's metadata.

        Args:
            prefix: Graph-name prefix used for :meth:`add`.

        Returns:
            Roles and units for additive mode's pattern outputs, and metadata
            recording the mode, selectors and the boundary mean shift's
            interpretation as the unattributed atmospheric level.
        """
        roles: dict[str, str] = {}
        units: dict[str, str] = {}
        if self.choice.mode == "additive":
            roles = {
                "outer_flux_pattern_coefficient": "outer_pattern_coefficient",
                "outer_flux_pattern_concentration": "outer_pattern_contribution",
            }
            units = {"outer_pattern_coefficient": "1", "outer_pattern_contribution": self.data.attrs["units"]}
        metadata = {
            "outer_flux_mode": self.choice.mode,
            "outer_groups": list(self.choice.outer_groups),
            "outer_fixed_sources": list(self.choice.fixed_sources),
            "outer_flux_centering": "equal_site_rows",
            "boundary_mean_shift_interpretation": "unattributed atmospheric level",
        }
        return Outputs(roles, units, metadata).prefixed(prefix)


def check_outer_flux_boundary(outer_flux: OuterFlux | None, boundary: Bound | None) -> None:
    """Require a centred boundary mean shift to own the level outer centring removes.

    Args:
        outer_flux: The bound outer-flux option, if any.
        boundary: The recipe's bound boundary, if any.

    Raises:
        ValueError: If the outer-flux option is used without a centred boundary
            that has a mean-shift prior.
    """
    if outer_flux is None:
        return
    if not (isinstance(boundary, AdditiveBoundary | AnomalyScalingBoundary) and boundary.choice.mean_prior is not None):
        raise ValueError(
            "The outer-flux option needs an additive or anomaly-scaling boundary with a mean shift prior; "
            "that mean shift is the model's only atmospheric level."
        )


__all__ = ["OuterFlux", "OuterFluxChoice", "OuterMode", "OuterStates", "check_outer_flux_boundary"]
