"""Explicit outer-region state treatment for the CO2 RHIME recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import (
    add_correlated_lognormal_state_with_activity,
    add_state_vector,
    apply_linear_sensitivity,
)
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError


OuterRegionMode: TypeAlias = Literal["fixed", "marginalized", "inferred"]


@dataclass(frozen=True)
class CollapsedOuterStates:
    """Outer-sector sensitivities collapsed onto explicitly labelled states.

    Attributes:
        sensitivity: Observation-by-collapsed-state outer sensitivity.
        members: Metadata mapping original outer states to collapsed states.
    """

    sensitivity: xr.DataArray
    members: xr.Dataset


@dataclass(frozen=True)
class OuterRegionTreatment:
    """Mutually exclusive outer-region mean, covariance, or sampled state.

    ``prepared_sensitivity`` is the sole authority for the observation and
    state dimensions, structurally retained columns, and full-state mapping.
    Fixed and inferred treatments use that contract to build a full public
    state and apply only the retained operator. A marginalized treatment uses
    the same prepared roles while exposing its Gaussian mean contribution and
    observation-space factor without sampling outer states.

    Marginalization is outer-scoped here because these states describe flux
    outside the scored inversion domain: callers may propagate their prior
    uncertainty while keeping them out of the sampled retained state. Upstream
    preparation selects the mode explicitly. General marginalization of weak
    states inside any basis group belongs in a shared state-disposition API,
    rather than being inferred implicitly by this outer-region recipe.

    Atmospheric boundary conditions and outer flux remain separate model
    components. Reporting code may group them when presenting a baseline.

    This API is currently motivated by the PARIS Verification Games, where
    synthetic observations have known flux and transport truth. Current
    evidence uses inferred outer states; wider use is not yet established.

    Attributes:
        mode: Selected fixed, marginalized, or inferred treatment.
        prepared_sensitivity: Prepared outer sensitivity and full-state
            mapping.
        mean_contribution: Fixed observation-space mean for marginalized
            treatment, otherwise ``None``.
        observation_factor: Observation-by-rank covariance factor for
            marginalized treatment, otherwise ``None``.
        prior_mean: Arithmetic prior mean retained by inferred treatment.
        prior_covariance: Arithmetic prior covariance retained by inferred
            treatment.
        state_metadata: Scientific metadata and resolved treatment for each
            original outer state.
        resolved_activity: Fixed or inferred state-activity contract, or
            ``None`` for marginalized treatment.
    """

    mode: OuterRegionMode
    prepared_sensitivity: PreparedLinearSensitivity
    mean_contribution: xr.DataArray | None
    observation_factor: xr.DataArray | None
    prior_mean: xr.DataArray | None
    prior_covariance: xr.DataArray | None
    state_metadata: xr.Dataset
    resolved_activity: ResolvedStateActivity | None


def _matrix_state_dim(sensitivity: xr.DataArray, observation_dim: str) -> str:
    matrix = sensitivity.transpose(observation_dim, ...)
    state_dims = [str(dim) for dim in matrix.dims if dim != observation_dim]
    if len(state_dims) != 1:
        raise ValueError(
            f"Outer sensitivity must have one observation and one state dimension; got {matrix.dims!r}."
        )
    state_dim = state_dims[0]
    if state_dim not in matrix.coords or not matrix.get_index(state_dim).is_unique:
        raise ValueError(f"Outer sensitivity requires unique {state_dim!r} state labels.")
    return state_dim


def _aligned_state_vector(
    value: float | xr.DataArray,
    sensitivity: xr.DataArray,
    *,
    state_dim: str,
    name: str,
) -> xr.DataArray:
    if not isinstance(value, xr.DataArray):
        value = xr.full_like(
            sensitivity.coords[state_dim],
            float(value),
            dtype=np.float64,
        )
    if value.dims != (state_dim,):
        raise ValueError(f"{name} must have dims ({state_dim!r},); got {value.dims!r}.")
    try:
        value, _ = xr.align(value, sensitivity.coords[state_dim], join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"{name} labels must exactly match outer sensitivity labels.") from exc
    values = np.asarray(value.compute().values)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return value.transpose(state_dim)


def _aligned_state_covariance(
    covariance: xr.DataArray,
    sensitivity: xr.DataArray,
    *,
    state_dim: str,
) -> tuple[xr.DataArray, str]:
    covariance_dims = [str(dim) for dim in covariance.dims if dim != state_dim]
    if len(covariance_dims) != 1:
        raise ValueError(
            "Outer prior covariance must use the state dimension and one covariance-column "
            f"dimension; got {covariance.dims!r}."
        )
    covariance_dim = covariance_dims[0]
    covariance = covariance.transpose(state_dim, covariance_dim)
    state_index = sensitivity.get_index(state_dim)
    if (
        covariance.sizes[state_dim] != len(state_index)
        or covariance.sizes[covariance_dim] != len(state_index)
        or not covariance.get_index(state_dim).equals(state_index)
    ):
        raise ValueError("Outer prior covariance labels must exactly match outer state labels.")
    if covariance_dim in covariance.coords and list(covariance[covariance_dim].values) != list(
        state_index.values
    ):
        raise ValueError("Outer prior covariance labels must exactly match outer state labels.")
    values = np.asarray(covariance.compute().values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Outer prior covariance must contain finite values.")
    tolerance = 1e-10 * max(float(np.abs(values).max()), 1.0)
    if not np.allclose(values, values.T, rtol=1e-10, atol=tolerance):
        raise ValueError("Outer prior covariance must be symmetric.")
    values = 0.5 * (values + values.T)
    if float(np.linalg.eigvalsh(values).min()) < -tolerance:
        raise ValueError("Outer prior covariance must be positive semidefinite.")
    return covariance.copy(data=values), covariance_dim


def _outer_forward(
    sensitivity: xr.DataArray,
    state: xr.DataArray,
    *,
    observation_dim: str,
    state_dim: str,
) -> xr.DataArray:
    return xr.dot(sensitivity, state, dim=state_dim).transpose(observation_dim)


def _outer_observation_factor(
    sensitivity: xr.DataArray,
    covariance: xr.DataArray,
    *,
    observation_dim: str,
    state_dim: str,
) -> xr.DataArray:
    values = np.asarray(covariance.values, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    rank_dim = "outer_covariance_rank"
    state_factor = xr.DataArray(
        (eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]) @ eigenvectors.T,
        dims=(state_dim, rank_dim),
        coords={state_dim: sensitivity[state_dim], rank_dim: np.arange(eigenvalues.size)},
    )
    return xr.dot(sensitivity, state_factor, dim=state_dim).transpose(
        observation_dim,
        rank_dim,
    )


def _state_metadata(sensitivity: xr.DataArray, state_dim: str) -> xr.Dataset:
    variables = {str(name): coord for name, coord in sensitivity.coords.items() if coord.dims == (state_dim,)}
    return xr.Dataset(coords=variables)


def _label_outer_basis_group(sensitivity: xr.DataArray, state_dim: str) -> xr.DataArray:
    """Declare that every state passed through the outer-treatment API is outer."""
    if "basis_group" in sensitivity.coords:
        basis_group = sensitivity["basis_group"]
        if basis_group.dims != (state_dim,) or not bool((basis_group == "outer").all().compute().item()):
            raise ValueError("Outer sensitivity basis_group metadata must label every state 'outer'.")
        return sensitivity
    return sensitivity.assign_coords(
        basis_group=(state_dim, np.full(sensitivity.sizes[state_dim], "outer", dtype=object))
    )


def _set_metadata_activity(
    metadata: xr.Dataset,
    activity: ResolvedStateActivity | None,
) -> xr.Dataset:
    """Record the activity of each state or collapsed-state member."""
    metadata = metadata.copy()
    metadata_dim = next(iter(metadata.dims))
    if activity is None:
        values = np.zeros(metadata.sizes[metadata_dim], dtype=bool)
    elif metadata_dim == activity.state_dim:
        values = np.asarray(activity.active.values, dtype=bool)
    elif "collapsed_state" in metadata:
        active_by_state = dict(
            zip(
                activity.active[activity.state_dim].values,
                np.asarray(activity.active.values, dtype=bool),
                strict=True,
            )
        )
        values = np.asarray(
            [active_by_state[label] for label in metadata["collapsed_state"].values],
            dtype=bool,
        )
    else:
        raise ValueError("Outer state metadata cannot be aligned to resolved activity.")
    metadata["activity"] = (metadata_dim, values)
    return metadata


def collapse_outer_sectors(
    outer_sensitivity: xr.DataArray,
    *,
    group_labels: xr.DataArray,
    observation_dim: str = "nmeasure",
    collapsed_dim: str = "outer_state",
    source_label: str = "outer_total",
    sector_label: str = "outer_total",
) -> CollapsedOuterStates:
    """Sum sector columns into explicitly grouped shared outer scaling states.

    ``group_labels`` is required and state-aligned; no region-kind strings or
    positional conventions are parsed.  The returned member table retains all
    original state-aligned source, sector, domain, and region coordinates.

    Args:
        outer_sensitivity: Observation-by-state outer sensitivity.
        group_labels: State-aligned label selecting the collapsed state for
            each original column.
        observation_dim: Observation dimension in ``outer_sensitivity``.
        collapsed_dim: Dimension name for the collapsed states.
        source_label: Source metadata assigned to every collapsed state.
        sector_label: Sector metadata assigned to every collapsed state.

    Returns:
        Collapsed sensitivity and original-member metadata.

    Raises:
        ValueError: If dimensions or labels are incompatible, group labels are
            missing, or one collapsed group spans multiple domains.
    """
    state_dim = _matrix_state_dim(outer_sensitivity, observation_dim)
    outer_sensitivity = _label_outer_basis_group(outer_sensitivity, state_dim)
    if group_labels.dims != (state_dim,):
        raise ValueError(f"group_labels must have dims ({state_dim!r},); got {group_labels.dims!r}.")
    try:
        groups, _ = xr.align(
            group_labels,
            outer_sensitivity.coords[state_dim],
            join="exact",
            copy=False,
        )
    except ValueError as exc:
        raise ValueError("group_labels must exactly match outer sensitivity labels.") from exc
    group_values = np.asarray(groups.values)
    if pd.isna(group_values).any():
        raise ValueError("group_labels must not contain missing values.")
    unique_groups = pd.Index(group_values).drop_duplicates()
    indicator = xr.DataArray(
        group_values[:, np.newaxis] == unique_groups.to_numpy()[np.newaxis, :],
        dims=(state_dim, collapsed_dim),
        coords={
            state_dim: outer_sensitivity.coords[state_dim],
            collapsed_dim: unique_groups,
        },
    )
    collapsed = xr.dot(
        outer_sensitivity.transpose(observation_dim, state_dim),
        indicator,
        dim=state_dim,
    ).transpose(observation_dim, collapsed_dim)
    collapsed = collapsed.assign_coords(
        source=(collapsed_dim, np.full(len(unique_groups), source_label, dtype=object)),
        sector=(collapsed_dim, np.full(len(unique_groups), sector_label, dtype=object)),
        basis_group=(collapsed_dim, np.full(len(unique_groups), "outer", dtype=object)),
        activity=(collapsed_dim, np.ones(len(unique_groups), dtype=bool)),
        treatment=(collapsed_dim, np.full(len(unique_groups), "collapsed", dtype=object)),
    ).rename("outer_sensitivity")

    # A domain remains a valid collapsed-state label only when every member of
    # that group has the same domain. Original member domains are always kept.
    if "domain" in outer_sensitivity.coords and outer_sensitivity["domain"].dims == (state_dim,):
        domains = np.asarray(outer_sensitivity["domain"].values)
        collapsed_domains = []
        for group in unique_groups:
            values = pd.Index(domains[group_values == group]).drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"Collapsed outer group {group!r} spans multiple domains.")
            collapsed_domains.append(values[0])
        collapsed = collapsed.assign_coords(domain=(collapsed_dim, collapsed_domains))

    members = _state_metadata(outer_sensitivity, state_dim)
    members["collapsed_state"] = (state_dim, group_values)
    return CollapsedOuterStates(sensitivity=collapsed, members=members)


def prepare_outer_region_treatment(
    outer_sensitivity: xr.DataArray | CollapsedOuterStates,
    *,
    mode: OuterRegionMode,
    prior_mean: xr.DataArray | None = None,
    prior_covariance: xr.DataArray | None = None,
    fixed_scale: float | xr.DataArray = 1.0,
    observation_dim: str = "nmeasure",
) -> OuterRegionTreatment:
    """Prepare one exclusive fixed, Gaussian-marginalized, or inferred mode.

    The two-dimensional preparation boundary resolves the observation and
    state roles once and stores the resulting
    :class:`PreparedLinearSensitivity`. Graph construction consumes that
    contract without accepting another dimension authority or rediscovering
    the state axis.

    Marginalized outer flux is currently motivated by the PARIS Verification
    Games, where context outside the scored domain can propagate Gaussian
    prior uncertainty without joining the sampled state. Its usefulness as a
    general outer-state policy has not yet been established; marginalizing
    weak states in any basis group belongs in a shared state-disposition API.

    Args:
        outer_sensitivity: Observation-by-state outer sensitivity, optionally
            with a prior sector-collapse mapping.
        mode: Fixed, Gaussian-marginalized, or inferred outer-state treatment.
        prior_mean: Arithmetic prior mean required by marginalized and inferred
            modes.
        prior_covariance: Arithmetic prior covariance required by marginalized
            and inferred modes.
        fixed_scale: Exact outer scaling used by fixed mode.
        observation_dim: Observation dimension in the outer sensitivity.

    Returns:
        Prepared treatment consumed by CO2 graph construction.

    Raises:
        ValueError: If the mode is unsupported or required dimensions, labels,
            prior moments, or fixed values are inconsistent.
    """
    if mode not in ("fixed", "marginalized", "inferred"):
        raise ValueError(f"Outer-region mode must be 'fixed', 'marginalized', or 'inferred'; got {mode!r}.")
    if isinstance(outer_sensitivity, CollapsedOuterStates):
        sensitivity = outer_sensitivity.sensitivity
        metadata = outer_sensitivity.members.copy()
    else:
        sensitivity = outer_sensitivity
        state_dim = _matrix_state_dim(sensitivity, observation_dim)
        metadata = _state_metadata(sensitivity, state_dim)

    state_dim = _matrix_state_dim(sensitivity, observation_dim)
    sensitivity = _label_outer_basis_group(
        sensitivity.transpose(observation_dim, state_dim),
        state_dim,
    )
    prepared = prepare_linear_sensitivity(sensitivity, output_dim=observation_dim)
    metadata_dim = next(iter(metadata.dims))
    if "basis_group" not in metadata:
        metadata = metadata.assign_coords(
            basis_group=(
                metadata_dim,
                np.full(metadata.sizes[metadata_dim], "outer", dtype=object),
            )
        )
    metadata["treatment"] = (
        metadata_dim,
        np.full(metadata.sizes[metadata_dim], mode, dtype=object),
    )

    if mode == "fixed":
        scale = _aligned_state_vector(
            fixed_scale,
            sensitivity,
            state_dim=state_dim,
            name="fixed_scale",
        )
        activity = resolve_state_activity(
            prepared.removed,
            StateActivity(active=False, fixed_value=scale),
        )
        return OuterRegionTreatment(
            mode=mode,
            prepared_sensitivity=prepared,
            mean_contribution=None,
            observation_factor=None,
            prior_mean=None,
            prior_covariance=None,
            state_metadata=_set_metadata_activity(metadata, activity),
            resolved_activity=activity,
        )

    if prior_mean is None or prior_covariance is None:
        raise ValueError(f"Outer-region mode {mode!r} requires prior_mean and prior_covariance.")
    mean = _aligned_state_vector(
        prior_mean,
        sensitivity,
        state_dim=state_dim,
        name="prior_mean",
    )
    covariance, _ = _aligned_state_covariance(
        prior_covariance,
        sensitivity,
        state_dim=state_dim,
    )

    if mode == "marginalized":
        return OuterRegionTreatment(
            mode=mode,
            prepared_sensitivity=prepared,
            mean_contribution=_outer_forward(
                sensitivity,
                mean,
                observation_dim=observation_dim,
                state_dim=state_dim,
            ).rename("outer_flux_contribution"),
            observation_factor=_outer_observation_factor(
                sensitivity,
                covariance,
                observation_dim=observation_dim,
                state_dim=state_dim,
            ).rename("outer_observation_factor"),
            prior_mean=None,
            prior_covariance=None,
            state_metadata=_set_metadata_activity(metadata, None),
            resolved_activity=None,
        )

    activity = resolve_state_activity(prepared.removed)
    return OuterRegionTreatment(
        mode=mode,
        prepared_sensitivity=prepared,
        mean_contribution=None,
        observation_factor=None,
        prior_mean=mean,
        prior_covariance=covariance,
        state_metadata=_set_metadata_activity(metadata, activity),
        resolved_activity=activity,
    )


def _namespace_outer_prepared(
    prepared: PreparedLinearSensitivity,
) -> tuple[PreparedLinearSensitivity, dict[str, str]]:
    """Namespace one prepared outer contract at the PyMC coordinate boundary."""
    state_dim = prepared.state_dim
    state_rename = {state_dim: f"{state_dim}_outer"}
    state_rename.update(
        {
            str(name): f"{name}_outer"
            for name, coord in prepared.removed.coords.items()
            if name not in prepared.removed.dims and state_dim in coord.dims
        }
    )
    removed = prepared.removed.rename(state_rename)

    retained_dim = str(prepared.sensitivity.dims[1])
    retained_rename = {
        retained_dim: (
            state_rename[state_dim] if retained_dim == state_dim else f"{state_rename[state_dim]}_retained"
        )
    }
    retained_rename.update(
        {
            str(name): f"{name}_outer"
            for name, coord in prepared.sensitivity.coords.items()
            if name not in prepared.sensitivity.dims and retained_dim in coord.dims
        }
    )
    return (
        PreparedLinearSensitivity(
            sensitivity=prepared.sensitivity.rename(retained_rename),
            removed=removed,
            output_dim=prepared.output_dim,
        ),
        state_rename,
    )


def add_outer_state_component(
    treatment: OuterRegionTreatment,
    *,
    var_name: str = "outer_flux_scaling",
    sensitivity_name: str = "outer_sensitivity",
    output_name: str = "outer_flux_contribution",
) -> Any:
    """Build and apply a fixed or inferred outer state from its prepared contract.

    Args:
        treatment: Prepared fixed or inferred outer-region treatment.
        var_name: Name for the complete outer state vector.
        sensitivity_name: Name for retained outer-sensitivity model data.
        output_name: Name for the observation-space outer contribution.

    Returns:
        Symbolic observation-space outer flux contribution.

    Raises:
        ValueError: If the treatment does not retain a state or lacks its
            resolved activity or inferred prior moments.
    """
    if treatment.mode not in ("fixed", "inferred"):
        raise ValueError("Only fixed and inferred outer treatments retain a state vector.")
    if treatment.resolved_activity is None:
        raise ValueError(f"Outer-region treatment {treatment.mode!r} is missing its state contract.")
    prepared, rename = _namespace_outer_prepared(treatment.prepared_sensitivity)
    state_dim = prepared.state_dim
    resolved = treatment.resolved_activity

    def rename_activity_array(array: xr.DataArray) -> xr.DataArray:
        return array.rename(
            {
                name: namespaced
                for name, namespaced in rename.items()
                if name in array.dims or name in array.coords
            }
        )

    activity = ResolvedStateActivity(
        state_dim=state_dim,
        active=rename_activity_array(resolved.active),
        fixed_value=rename_activity_array(resolved.fixed_value),
        zero_sensitivity=rename_activity_array(resolved.zero_sensitivity),
    )
    if treatment.mode == "fixed":
        state = add_state_vector(activity, {}, var_name=var_name).state
    else:
        if treatment.prior_mean is None or treatment.prior_covariance is None:
            raise ValueError("Inferred outer treatment is missing its prior moments.")
        original_state_dim = treatment.prepared_sensitivity.state_dim
        covariance_dims = [str(dim) for dim in treatment.prior_covariance.dims if dim != original_state_dim]
        if len(covariance_dims) != 1:
            raise ValueError("Inferred outer treatment prior covariance must have one column dimension.")
        covariance_dim = covariance_dims[0]
        mean = treatment.prior_mean.rename(
            {
                name: namespaced
                for name, namespaced in rename.items()
                if name in treatment.prior_mean.dims or name in treatment.prior_mean.coords
            }
        )
        covariance_rename = {
            name: namespaced
            for name, namespaced in rename.items()
            if name in treatment.prior_covariance.dims or name in treatment.prior_covariance.coords
        }
        covariance_rename[covariance_dim] = f"{covariance_dim}_outer"
        covariance = treatment.prior_covariance.rename(covariance_rename)
        prior = CorrelatedLognormalPrior(
            mean,
            covariance,
            covariance_dim=covariance_rename[covariance_dim],
        )
        state = add_correlated_lognormal_state_with_activity(
            activity,
            prior,
            var_name=var_name,
        ).state
    return apply_linear_sensitivity(
        prepared,
        state,
        data_name=sensitivity_name,
        output_name=output_name,
    )


def add_outer_observation_covariance(
    aggregation_error: AggregationError,
    treatment: OuterRegionTreatment,
) -> AggregationError:
    """Add exact Gaussian outer-state covariance without densifying structured error.

    Args:
        aggregation_error: Existing fixed aggregation-error representation.
        treatment: Prepared outer-region treatment.

    Returns:
        The unchanged error for fixed or inferred treatment, or an error with
        the marginalized outer covariance added in its structured form.

    Raises:
        ValueError: If a marginalized treatment lacks its observation factor
            or labelled error components cannot align.
    """
    if treatment.mode != "marginalized":
        return aggregation_error
    if treatment.observation_factor is None:
        raise ValueError("Marginalized outer treatment requires an observation factor.")
    observation_dim = treatment.prepared_sensitivity.output_dim
    outer = treatment.observation_factor.transpose(observation_dim, ...)
    outer_rank_dim = str(outer.dims[1])
    outer_marginal = np.asarray((outer**2).sum(outer_rank_dim).compute().values)
    if not outer_marginal.any():
        return aggregation_error

    if aggregation_error.mode == "none":
        diagonal = xr.zeros_like(outer.isel({outer_rank_dim: 0}, drop=True)).rename(
            "diagonal_residual_variance"
        )
        return AggregationError(
            mode="low_rank",
            marginal_variance=outer_marginal,
            factor=outer.rename("low_rank_factor"),
            diagonal_variance=diagonal,
        )

    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        outer, covariance = xr.align(
            outer,
            aggregation_error.covariance,
            join="exact",
            copy=False,
        )
        outer_values = np.asarray(outer.compute().values, dtype=np.float64)
        values = np.asarray(covariance.values, dtype=np.float64) + outer_values @ outer_values.T
        return AggregationError(
            mode="dense",
            marginal_variance=aggregation_error.marginal_variance + outer_marginal,
            covariance=covariance.copy(data=values),
        )

    if aggregation_error.mode == "diagonal":
        assert aggregation_error.diagonal_variance is not None
        outer, diagonal = xr.align(
            outer,
            aggregation_error.diagonal_variance,
            join="exact",
            copy=False,
        )
        return AggregationError(
            mode="low_rank",
            marginal_variance=aggregation_error.marginal_variance + outer_marginal,
            factor=outer.rename("low_rank_factor"),
            diagonal_variance=diagonal,
        )

    assert aggregation_error.factor is not None
    assert aggregation_error.diagonal_variance is not None
    factor, outer, diagonal = xr.align(
        aggregation_error.factor,
        outer,
        aggregation_error.diagonal_variance,
        join="exact",
        copy=False,
    )
    factor_rank_dim = str(factor.dims[1])
    factor_rank = factor.sizes[factor_rank_dim]
    combined_rank_dim = "aggregation_error_rank"
    if factor_rank_dim != combined_rank_dim:
        factor = factor.rename({factor_rank_dim: combined_rank_dim})
    factor = factor.assign_coords({combined_rank_dim: np.arange(factor_rank)})
    outer_rank = outer.sizes[outer_rank_dim]
    outer = outer.rename({outer_rank_dim: combined_rank_dim}).assign_coords(
        {
            combined_rank_dim: np.arange(
                factor_rank,
                factor_rank + outer_rank,
            )
        }
    )
    combined = xr.concat((factor, outer), dim=combined_rank_dim).rename("low_rank_factor")
    return AggregationError(
        mode="low_rank",
        marginal_variance=aggregation_error.marginal_variance + outer_marginal,
        factor=combined.transpose(observation_dim, combined_rank_dim),
        diagonal_variance=diagonal,
    )


__all__ = [
    "CollapsedOuterStates",
    "OuterRegionMode",
    "OuterRegionTreatment",
    "add_outer_state_component",
    "add_outer_observation_covariance",
    "collapse_outer_sectors",
    "prepare_outer_region_treatment",
]
