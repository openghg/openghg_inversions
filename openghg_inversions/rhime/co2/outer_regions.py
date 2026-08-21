"""Explicit outer-region state treatment for the CO2 RHIME recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import add_correlated_lognormal_state_with_activity, add_model_data
from openghg_inversions.models.state_activity import detect_zero_sensitivity, resolve_state_activity
from openghg_inversions.observation_error import AggregationError


OuterRegionMode: TypeAlias = Literal["fixed", "marginalized", "inferred"]


@dataclass(frozen=True)
class CollapsedOuterStates:
    """Outer-sector sensitivities collapsed onto explicitly labelled states."""

    sensitivity: xr.DataArray
    members: xr.Dataset


@dataclass(frozen=True)
class OuterRegionTreatment:
    """Mutually exclusive outer-region mean, covariance, or sampled state.

    ``fixed_contribution`` is added to the atmospheric baseline.  Only a
    marginalized treatment has non-zero ``observation_covariance``; only an
    inferred treatment retains ``sensitivity`` and state-prior moments.  This
    partition prevents one outer contribution being counted in both the
    baseline and the sampled state.
    """

    mode: OuterRegionMode
    fixed_contribution: xr.DataArray
    observation_covariance: xr.DataArray
    sensitivity: xr.DataArray | None
    prior_mean: xr.DataArray | None
    prior_covariance: xr.DataArray | None
    state_metadata: xr.Dataset


def composite_baseline(
    atmospheric_baseline: Any | None,
    fixed_outer_contribution: Any | None,
) -> Any | None:
    """Return atmospheric boundary conditions plus fixed outer flux.

    Inputs may be labelled arrays or backend tensors.  The function deliberately
    does only the scientifically named addition; state treatment is resolved
    before model construction.
    """
    if atmospheric_baseline is None:
        return fixed_outer_contribution
    if fixed_outer_contribution is None:
        return atmospheric_baseline
    return atmospheric_baseline + fixed_outer_contribution


def _matrix_state_dim(sensitivity: xr.DataArray, observation_dim: str) -> str:
    matrix = sensitivity.transpose(observation_dim, ...)
    state_dims = [str(dim) for dim in matrix.dims if dim != observation_dim]
    if len(state_dims) != 1:
        raise ValueError(
            "Outer sensitivity must have one observation and one state dimension; "
            f"got {matrix.dims!r}."
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
    if float(np.linalg.eigvalsh(values).min()) < -tolerance:
        raise ValueError("Outer prior covariance must be positive semidefinite.")
    return covariance, covariance_dim


def _outer_forward(
    sensitivity: xr.DataArray,
    state: xr.DataArray,
    *,
    observation_dim: str,
    state_dim: str,
) -> xr.DataArray:
    return xr.dot(sensitivity, state, dim=state_dim).transpose(observation_dim)


def _outer_observation_covariance(
    sensitivity: xr.DataArray,
    covariance: xr.DataArray,
    *,
    observation_dim: str,
    state_dim: str,
    covariance_dim: str,
) -> xr.DataArray:
    observation_covariance_dim = f"{observation_dim}_cov"
    right = xr.DataArray(
        sensitivity.data,
        dims=(observation_covariance_dim, covariance_dim),
        coords={
            observation_covariance_dim: np.asarray(
                sensitivity.coords.get(
                    observation_dim,
                    xr.DataArray(np.arange(sensitivity.sizes[observation_dim])),
                ).values
            ),
        },
    )
    return xr.dot(
        xr.dot(sensitivity, covariance, dim=state_dim),
        right,
        dim=covariance_dim,
    ).transpose(observation_dim, observation_covariance_dim)


def _zero_observation_covariance(
    sensitivity: xr.DataArray,
    *,
    observation_dim: str,
) -> xr.DataArray:
    observation_covariance_dim = f"{observation_dim}_cov"
    rows = xr.zeros_like(sensitivity.isel({sensitivity.dims[1]: 0}, drop=True))
    columns = rows.rename({observation_dim: observation_covariance_dim})
    return (rows * columns).transpose(observation_dim, observation_covariance_dim)


def _state_metadata(sensitivity: xr.DataArray, state_dim: str) -> xr.Dataset:
    variables = {
        str(name): coord
        for name, coord in sensitivity.coords.items()
        if coord.dims == (state_dim,)
    }
    return xr.Dataset(coords=variables)


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
    """
    state_dim = _matrix_state_dim(outer_sensitivity, observation_dim)
    if group_labels.dims != (state_dim,):
        raise ValueError(
            f"group_labels must have dims ({state_dim!r},); got {group_labels.dims!r}."
        )
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
    """Prepare one exclusive fixed, Gaussian-marginalized, or inferred mode."""
    if mode not in ("fixed", "marginalized", "inferred"):
        raise ValueError(
            "Outer-region mode must be 'fixed', 'marginalized', or 'inferred'; "
            f"got {mode!r}."
        )
    if isinstance(outer_sensitivity, CollapsedOuterStates):
        sensitivity = outer_sensitivity.sensitivity
        metadata = outer_sensitivity.members.copy()
    else:
        sensitivity = outer_sensitivity
        state_dim = _matrix_state_dim(sensitivity, observation_dim)
        metadata = _state_metadata(sensitivity, state_dim)

    state_dim = _matrix_state_dim(sensitivity, observation_dim)
    sensitivity = sensitivity.transpose(observation_dim, state_dim)
    zero_covariance = _zero_observation_covariance(
        sensitivity,
        observation_dim=observation_dim,
    ).rename("outer_observation_covariance")
    zero_contribution = xr.zeros_like(
        sensitivity.isel({state_dim: 0}, drop=True)
    ).rename("fixed_outer_contribution")

    metadata_dim = next(iter(metadata.dims))
    metadata["activity"] = (
        metadata_dim,
        np.full(metadata.sizes[metadata_dim], mode == "inferred", dtype=bool),
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
        return OuterRegionTreatment(
            mode=mode,
            fixed_contribution=_outer_forward(
                sensitivity,
                scale,
                observation_dim=observation_dim,
                state_dim=state_dim,
            ).rename("fixed_outer_contribution"),
            observation_covariance=zero_covariance,
            sensitivity=None,
            prior_mean=None,
            prior_covariance=None,
            state_metadata=metadata,
        )

    if prior_mean is None or prior_covariance is None:
        raise ValueError(f"Outer-region mode {mode!r} requires prior_mean and prior_covariance.")
    mean = _aligned_state_vector(
        prior_mean,
        sensitivity,
        state_dim=state_dim,
        name="prior_mean",
    )
    covariance, covariance_dim = _aligned_state_covariance(
        prior_covariance,
        sensitivity,
        state_dim=state_dim,
    )

    if mode == "marginalized":
        return OuterRegionTreatment(
            mode=mode,
            fixed_contribution=_outer_forward(
                sensitivity,
                mean,
                observation_dim=observation_dim,
                state_dim=state_dim,
            ).rename("fixed_outer_contribution"),
            observation_covariance=_outer_observation_covariance(
                sensitivity,
                covariance,
                observation_dim=observation_dim,
                state_dim=state_dim,
                covariance_dim=covariance_dim,
            ).rename("outer_observation_covariance"),
            sensitivity=None,
            prior_mean=None,
            prior_covariance=None,
            state_metadata=metadata,
        )

    return OuterRegionTreatment(
        mode=mode,
        fixed_contribution=zero_contribution,
        observation_covariance=zero_covariance,
        sensitivity=sensitivity,
        prior_mean=mean,
        prior_covariance=covariance,
        state_metadata=metadata,
    )


def add_inferred_outer_component(
    treatment: OuterRegionTreatment,
    *,
    var_name: str = "x_outer",
    sensitivity_name: str = "h_outer",
    output_name: str = "mu_outer",
    observation_dim: str = "nmeasure",
) -> Any:
    """Add the sampled collapsed-or-sector-resolved outer contribution."""
    if (
        treatment.mode != "inferred"
        or treatment.sensitivity is None
        or treatment.prior_mean is None
        or treatment.prior_covariance is None
    ):
        raise ValueError("Only an inferred outer-region treatment has a sampled state.")
    sensitivity = treatment.sensitivity.transpose(observation_dim, ...)
    state_dim = _matrix_state_dim(sensitivity, observation_dim)
    covariance_dim = next(
        str(dim) for dim in treatment.prior_covariance.dims if dim != state_dim
    )
    rename = {state_dim: f"{state_dim}_outer"}
    rename.update(
        {
            str(name): f"{name}_outer"
            for name, coord in sensitivity.coords.items()
            if name not in sensitivity.dims and state_dim in coord.dims
        }
    )
    sensitivity = sensitivity.rename(rename)
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
        if name in treatment.prior_covariance.dims
        or name in treatment.prior_covariance.coords
    }
    covariance_rename[covariance_dim] = f"{covariance_dim}_outer"
    covariance = treatment.prior_covariance.rename(covariance_rename)
    state_dim = rename[state_dim]
    covariance_dim = covariance_rename[covariance_dim]
    prior = CorrelatedLognormalPrior(
        mean,
        covariance,
        covariance_dim=covariance_dim,
    )
    activity = resolve_state_activity(detect_zero_sensitivity(sensitivity))
    state = add_correlated_lognormal_state_with_activity(
        activity,
        prior,
        var_name=var_name,
    ).state
    design = add_model_data(sensitivity, sensitivity_name)
    return pm.Deterministic(
        output_name,
        pt.dot(design, state),
        dims=observation_dim,
    )


def add_outer_observation_covariance(
    aggregation_error: AggregationError,
    treatment: OuterRegionTreatment,
) -> AggregationError:
    """Add exact Gaussian outer-state covariance to fixed observation error."""
    if treatment.mode != "marginalized":
        return aggregation_error
    outer = treatment.observation_covariance
    outer_values = np.asarray(outer.values, dtype=np.float64)
    nmeasure = outer.shape[0]
    if aggregation_error.mode == "none":
        base = np.zeros((nmeasure, nmeasure), dtype=np.float64)
    elif aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        base = np.asarray(aggregation_error.covariance.values, dtype=np.float64)
    elif aggregation_error.mode == "diagonal":
        assert aggregation_error.diagonal_variance is not None
        base = np.diag(np.asarray(aggregation_error.diagonal_variance.values, dtype=np.float64))
    else:
        assert aggregation_error.factor is not None
        assert aggregation_error.diagonal_variance is not None
        factor = np.asarray(aggregation_error.factor.values, dtype=np.float64)
        base = factor @ factor.T + np.diag(
            np.asarray(aggregation_error.diagonal_variance.values, dtype=np.float64)
        )
    covariance = outer.copy(data=base + outer_values).rename("aggregation_error_covariance")
    return AggregationError(
        mode="dense",
        marginal_variance=np.diag(covariance.values).copy(),
        covariance=covariance,
    )


__all__ = [
    "CollapsedOuterStates",
    "OuterRegionMode",
    "OuterRegionTreatment",
    "add_inferred_outer_component",
    "add_outer_observation_covariance",
    "collapse_outer_sectors",
    "composite_baseline",
    "prepare_outer_region_treatment",
]
