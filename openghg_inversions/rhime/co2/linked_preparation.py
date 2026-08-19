"""Labelled scientific inputs for the linked CO2/O2 recipe.

CO2 and O2 keep distinct, potentially unequal observation axes at this public
boundary. They are stacked only after their labels, state meanings, covariance
blocks, and units have been checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AggregationError,
    resolve_aggregation_error,
)


@dataclass(frozen=True, slots=True, eq=False)
class Co2O2PreparedInputs:
    """Backend-neutral joint inputs with shared land and split-ocean states."""

    observations: xr.DataArray
    prior_forward_mean: xr.DataArray
    effective_observation_operator: xr.DataArray
    aggregation_error: AggregationError
    retained_prior: CorrelatedLognormalPrior
    co2_observation_dim: str
    o2_observation_dim: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _axis(array: xr.DataArray, name: str) -> str:
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.dims!r}.")
    dim = str(array.dims[0])
    if dim not in array.indexes or not array.indexes[dim].is_unique:
        raise ValueError(f"{name} requires unique labels on {dim!r}.")
    return dim


def _same_axis(reference: xr.DataArray, candidate: xr.DataArray, name: str) -> None:
    dim = str(reference.dims[0])
    if candidate.dims != (dim,) or not candidate.indexes[dim].equals(reference.indexes[dim]):
        raise ValueError(f"{name} labels must exactly match its observations.")


def _state(prior: CorrelatedLognormalPrior) -> tuple[xr.DataArray, str]:
    mean = prior.mean
    state_dim = prior.state_dim
    if any(name not in mean.coords for name in ("source", "tracer_scope")):
        raise ValueError("Retained states require source and tracer_scope coordinates.")
    pairs = {
        (str(source).lower(), str(scope).lower())
        for source, scope in zip(mean["source"].values, mean["tracer_scope"].values, strict=True)
    }
    required = {
        ("gpp", "shared"),
        ("ter", "shared"),
        ("ff", "shared"),
        ("ocean", "co2"),
        ("ocean", "o2"),
    }
    if pairs != required:
        raise ValueError(
            "Retained states must contain only shared GPP/TER/FF and tracer-specific CO2/O2 ocean states."
        )
    return mean, state_dim


def _operator(
    value: xr.DataArray,
    observation: xr.DataArray,
    state_mean: xr.DataArray,
    name: str,
) -> None:
    observation_dim = str(observation.dims[0])
    state_dim = str(state_mean.dims[0])
    if value.dims != (observation_dim, state_dim):
        raise ValueError(f"{name} operator must have dimensions {(observation_dim, state_dim)!r}.")
    if not value.indexes[observation_dim].equals(observation.indexes[observation_dim]):
        raise ValueError(f"{name} operator rows do not match its observations.")
    if not value.indexes[state_dim].equals(state_mean.indexes[state_dim]):
        raise ValueError(f"{name} operator state labels do not match the retained prior.")


def _covariance(
    value: xr.DataArray,
    row: xr.DataArray,
    column: xr.DataArray,
    name: str,
) -> np.ndarray:
    row_dim = str(row.dims[0])
    column_dim = str(column.dims[0])
    if value.ndim != 2 or value.dims[0] != row_dim or value.shape != (row.size, column.size):
        raise ValueError(f"{name} shape or row dimension does not match its observation axes.")
    value_column_dim = str(value.dims[1])
    if not value.indexes[row_dim].equals(row.indexes[row_dim]):
        raise ValueError(f"{name} row labels do not match its observations.")
    if not value.indexes[value_column_dim].equals(column.indexes[column_dim]):
        raise ValueError(f"{name} column labels do not match its observations.")
    result = np.asarray(value.values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _stack(
    co2: xr.DataArray,
    o2: xr.DataArray,
    *,
    co2_units: str,
    o2_units: str,
    name: str,
) -> xr.DataArray:
    nco2, no2 = co2.size, o2.size
    coords: dict[str, object] = {
        "observation": np.arange(nco2 + no2),
        "tracer": ("observation", np.repeat(("co2", "o2"), (nco2, no2))),
        "within_tracer_observation": (
            "observation",
            [str(label) for label in (*co2.indexes[co2.dims[0]], *o2.indexes[o2.dims[0]])],
        ),
        "observation_units": (
            "observation",
            np.repeat((co2_units, o2_units), (nco2, no2)),
        ),
    }
    for coordinate_name in ("site", "time"):
        if coordinate_name not in co2.coords or coordinate_name not in o2.coords:
            continue
        co2_coordinate = co2.coords[coordinate_name]
        o2_coordinate = o2.coords[coordinate_name]
        if co2_coordinate.dims == co2.dims and o2_coordinate.dims == o2.dims:
            coords[coordinate_name] = (
                "observation",
                np.concatenate((co2_coordinate.values, o2_coordinate.values)),
            )
    return xr.DataArray(
        np.concatenate((co2.values, o2.values)),
        dims=("observation",),
        coords=coords,
        name=name,
        attrs={"units": "mixed; see observation_units coordinate"},
    )


def prepare_linked_co2_o2_inputs(
    *,
    co2_observations: xr.DataArray,
    o2_observations: xr.DataArray,
    co2_prior_forward_mean: xr.DataArray,
    o2_prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    co2_aggregation_covariance: xr.DataArray,
    co2_o2_aggregation_covariance: xr.DataArray,
    o2_aggregation_covariance: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    co2_units: str,
    o2_units: str,
    provenance: Mapping[str, Any] | None = None,
) -> Co2O2PreparedInputs:
    """Validate separate tracer inputs and form one labelled joint likelihood."""
    if not co2_units.strip() or not o2_units.strip():
        raise ValueError("CO2 and O2 channel units must be non-empty.")
    try:
        prepared_provenance = json.loads(json.dumps(dict(provenance or {})))
    except (TypeError, ValueError) as exc:
        raise ValueError("Linked CO2/O2 provenance must be JSON serializable.") from exc

    co2_dim = _axis(co2_observations, "CO2 observations")
    o2_dim = _axis(o2_observations, "O2 observations")
    if co2_dim == o2_dim:
        raise ValueError("CO2 and O2 require distinct pre-stacking dimension names.")
    _same_axis(co2_observations, co2_prior_forward_mean, "CO2 prior forward mean")
    _same_axis(o2_observations, o2_prior_forward_mean, "O2 prior forward mean")
    state_mean, state_dim = _state(retained_prior)
    _operator(co2_operator, co2_observations, state_mean, "CO2")
    _operator(o2_operator, o2_observations, state_mean, "O2")

    co2_covariance = _covariance(
        co2_aggregation_covariance,
        co2_observations,
        co2_observations,
        "CO2 covariance",
    )
    cross_covariance = _covariance(
        co2_o2_aggregation_covariance,
        co2_observations,
        o2_observations,
        "CO2/O2 cross-covariance",
    )
    o2_covariance = _covariance(
        o2_aggregation_covariance,
        o2_observations,
        o2_observations,
        "O2 covariance",
    )
    joint_covariance = np.block([[co2_covariance, cross_covariance], [cross_covariance.T, o2_covariance]])

    observations = _stack(
        co2_observations,
        o2_observations,
        co2_units=co2_units,
        o2_units=o2_units,
        name="observed_concentration",
    )
    prior_forward = _stack(
        co2_prior_forward_mean,
        o2_prior_forward_mean,
        co2_units=co2_units,
        o2_units=o2_units,
        name="prior_forward_concentration",
    )
    operator = xr.DataArray(
        np.concatenate((co2_operator.values, o2_operator.values), axis=0),
        dims=("observation", state_dim),
        coords={"observation": observations["observation"], state_dim: state_mean[state_dim]},
        name="effective_observation_operator",
        attrs={"units": "observation_units per dimensionless flux scale"},
    )
    operator_coords: dict[str, xr.DataArray] = {
        "tracer": observations["tracer"],
        "observation_units": observations["observation_units"],
    }
    for coordinate_name in ("source", "tracer_scope"):
        # MultiIndex state levels are already part of the index. Reassigning
        # only some levels would corrupt that index in modern xarray.
        if coordinate_name not in operator.coords:
            operator_coords[coordinate_name] = state_mean[coordinate_name]
    operator = operator.assign_coords(operator_coords)
    covariance = xr.DataArray(
        joint_covariance,
        dims=("observation", "observation_cov"),
        coords={
            "observation": observations["observation"],
            "observation_cov": observations["observation"].values,
            "tracer": observations["tracer"],
            "tracer_cov": ("observation_cov", observations["tracer"].values),
            "observation_units": observations["observation_units"],
            "observation_units_cov": (
                "observation_cov",
                observations["observation_units"].values,
            ),
        },
        name=AGGREGATION_ERROR_COVARIANCE,
        attrs={"units": "observation_units * observation_units_cov"},
    )
    aggregation_error = resolve_aggregation_error(
        xr.Dataset(
            {
                observations.name: observations,
                AGGREGATION_ERROR_COVARIANCE: covariance,
            }
        ),
        "dense",
        output_dim="observation",
        covariance_dim="observation_cov",
    )
    return Co2O2PreparedInputs(
        observations=observations,
        prior_forward_mean=prior_forward,
        effective_observation_operator=operator,
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        co2_observation_dim=co2_dim,
        o2_observation_dim=o2_dim,
        provenance=prepared_provenance,
    )
