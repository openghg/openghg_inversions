"""Labelled scientific inputs for the CO2/O2 recipe.

CO2 and O2 keep distinct, potentially unequal observation axes at this public
boundary. They are stacked only after their labels, state meanings, covariance
blocks, and units have been checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from dask import compute as dask_compute
import numpy as np
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AggregationError,
    resolve_aggregation_error,
)

O2OperatorRatioConvention = Literal["embedded_signed_o2_per_co2"]
O2_OPERATOR_RATIO_CONVENTION: O2OperatorRatioConvention = "embedded_signed_o2_per_co2"


@dataclass(frozen=True, slots=True, eq=False)
class Co2O2PreparedInputs:
    """Backend-neutral joint inputs with shared land and split-ocean states."""

    observations: xr.DataArray
    prior_forward_mean: xr.DataArray
    co2_operator: xr.DataArray
    o2_operator: xr.DataArray
    o2_operator_ratio_convention: O2OperatorRatioConvention
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


def _validate_ocean_loadings(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    state_mean: xr.DataArray,
) -> None:
    """Reject loadings from one channel onto the other channel's ocean state."""
    state_dim = str(state_mean.dims[0])
    roles = [
        (str(source).lower(), str(scope).lower())
        for source, scope in zip(
            state_mean["source"].values,
            state_mean["tracer_scope"].values,
            strict=True,
        )
    ]
    co2_ocean = [index for index, role in enumerate(roles) if role == ("ocean", "co2")]
    o2_ocean = [index for index, role in enumerate(roles) if role == ("ocean", "o2")]
    co2_cross, o2_cross = dask_compute(
        co2_operator.isel({state_dim: o2_ocean}).data,
        o2_operator.isel({state_dim: co2_ocean}).data,
    )
    if np.any(co2_cross != 0):
        raise ValueError("CO2 operator must have zero loadings for O2-specific ocean states.")
    if np.any(o2_cross != 0):
        raise ValueError("O2 operator must have zero loadings for CO2-specific ocean states.")


def _covariance_block(
    value: xr.DataArray,
    row: xr.DataArray,
    column: xr.DataArray,
    name: str,
) -> xr.DataArray:
    row_dim = str(row.dims[0])
    column_dim = str(column.dims[0])
    if value.ndim != 2 or value.dims[0] != row_dim or value.shape != (row.size, column.size):
        raise ValueError(f"{name} shape or row dimension does not match its observation axes.")
    value_column_dim = str(value.dims[1])
    if not value.indexes[row_dim].equals(row.indexes[row_dim]):
        raise ValueError(f"{name} row labels do not match its observations.")
    if not value.indexes[value_column_dim].equals(column.indexes[column_dim]):
        raise ValueError(f"{name} column labels do not match its observations.")
    return value


def _channel_vector(
    value: xr.DataArray,
    *,
    units: str,
    tracer: str,
    name: str,
) -> xr.DataArray:
    dim = str(value.dims[0])
    return value.rename(name).assign_coords(
        tracer=(dim, np.repeat(tracer, value.size)),
        observation_units=(dim, np.repeat(units, value.size)),
    )


def _stack(
    co2: xr.DataArray,
    o2: xr.DataArray,
    *,
    co2_units: str,
    o2_units: str,
    name: str,
) -> xr.DataArray:
    """Stack labelled channel vectors while preserving their lazy payloads."""

    def channel(
        value: xr.DataArray,
        *,
        start: int,
        tracer: str,
        units: str,
    ) -> xr.DataArray:
        original_dim = str(value.dims[0])
        labels = [str(label) for label in value.indexes[original_dim]]
        result = _channel_vector(value, units=units, tracer=tracer, name=name).rename(
            {original_dim: "observation"}
        )
        return result.assign_coords(
            observation=np.arange(start, start + value.size),
            within_tracer_observation=("observation", labels),
        )

    stacked = xr.concat(
        (
            channel(co2, start=0, tracer="co2", units=co2_units),
            channel(o2, start=co2.size, tracer="o2", units=o2_units),
        ),
        dim="observation",
        join="exact",
    )
    stacked.attrs["units"] = "mixed; see observation_units coordinate"
    return stacked


def _joint_covariance(
    co2_covariance: xr.DataArray,
    cross_covariance: xr.DataArray,
    o2_covariance: xr.DataArray,
    *,
    nco2: int,
    no2: int,
) -> xr.DataArray:
    """Combine validated labelled channel blocks without materializing them."""
    co2_labels = np.arange(nco2)
    o2_labels = np.arange(nco2, nco2 + no2)

    def labelled(
        block: xr.DataArray,
        row_labels: np.ndarray,
        column_labels: np.ndarray,
    ) -> xr.DataArray:
        return xr.DataArray(
            block.data,
            dims=("observation", "observation_cov"),
            coords={"observation": row_labels, "observation_cov": column_labels},
        )

    co2 = labelled(co2_covariance, co2_labels, co2_labels)
    cross = labelled(cross_covariance, co2_labels, o2_labels)
    cross_transpose = labelled(cross_covariance.transpose(), o2_labels, co2_labels)
    o2 = labelled(o2_covariance, o2_labels, o2_labels)
    top = xr.concat((co2, cross), dim="observation_cov", join="exact")
    bottom = xr.concat((cross_transpose, o2), dim="observation_cov", join="exact")
    covariance = xr.concat((top, bottom), dim="observation", join="exact")
    tracer = np.repeat(("co2", "o2"), (nco2, no2))
    return covariance.assign_coords(
        tracer=("observation", tracer),
        tracer_cov=("observation_cov", tracer),
    ).rename(AGGREGATION_ERROR_COVARIANCE)


def prepare_co2_o2_inputs(
    *,
    co2_observations: xr.DataArray,
    o2_observations: xr.DataArray,
    co2_prior_forward_mean: xr.DataArray,
    o2_prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    o2_operator_ratio_convention: O2OperatorRatioConvention,
    co2_aggregation_covariance: xr.DataArray,
    co2_o2_aggregation_covariance: xr.DataArray,
    o2_aggregation_covariance: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    co2_units: str,
    o2_units: str,
    provenance: Mapping[str, Any] | None = None,
) -> Co2O2PreparedInputs:
    """Validate separate tracer inputs and form one labelled joint likelihood."""
    if o2_operator_ratio_convention != O2_OPERATOR_RATIO_CONVENTION:
        raise ValueError(
            "The O2 operator must declare signed O2-per-CO2 oxidation ratios embedded "
            "for shared GPP/TER/FF states; the O2 ocean state is applied directly."
        )
    if not co2_units.strip() or not o2_units.strip():
        raise ValueError("CO2 and O2 channel units must be non-empty.")
    try:
        prepared_provenance = json.loads(json.dumps(dict(provenance or {})))
    except (TypeError, ValueError) as exc:
        raise ValueError("CO2/O2 provenance must be JSON serializable.") from exc

    co2_dim = _axis(co2_observations, "CO2 observations")
    o2_dim = _axis(o2_observations, "O2 observations")
    if co2_dim == o2_dim:
        raise ValueError("CO2 and O2 require distinct pre-stacking dimension names.")
    _same_axis(co2_observations, co2_prior_forward_mean, "CO2 prior forward mean")
    _same_axis(o2_observations, o2_prior_forward_mean, "O2 prior forward mean")
    state_mean, _ = _state(retained_prior)
    _operator(co2_operator, co2_observations, state_mean, "CO2")
    _operator(o2_operator, o2_observations, state_mean, "O2")
    _validate_ocean_loadings(co2_operator, o2_operator, state_mean)

    co2_covariance = _covariance_block(
        co2_aggregation_covariance,
        co2_observations,
        co2_observations,
        "CO2 covariance",
    )
    cross_covariance = _covariance_block(
        co2_o2_aggregation_covariance,
        co2_observations,
        o2_observations,
        "CO2/O2 cross-covariance",
    )
    o2_covariance = _covariance_block(
        o2_aggregation_covariance,
        o2_observations,
        o2_observations,
        "O2 covariance",
    )
    covariance = _joint_covariance(
        co2_covariance,
        cross_covariance,
        o2_covariance,
        nco2=co2_observations.size,
        no2=o2_observations.size,
    ).assign_coords(
        observation_units=(
            "observation",
            np.repeat((co2_units, o2_units), (co2_observations.size, o2_observations.size)),
        ),
        observation_units_cov=(
            "observation_cov",
            np.repeat((co2_units, o2_units), (co2_observations.size, o2_observations.size)),
        ),
    )
    observations = _stack(
        co2_observations,
        o2_observations,
        co2_units=co2_units,
        o2_units=o2_units,
        name="observed_concentration",
    )
    prior_forward_mean = _stack(
        co2_prior_forward_mean,
        o2_prior_forward_mean,
        co2_units=co2_units,
        o2_units=o2_units,
        name="prior_forward_concentration",
    )
    operator_coords = {name: state_mean[name] for name in ("source", "tracer_scope")}
    co2_operator = co2_operator.rename("co2_effective_observation_operator").assign_coords(
        **{
            name: coordinate
            for name, coordinate in operator_coords.items()
            if name not in co2_operator.coords
        },
    ).assign_attrs(units=f"{co2_units} per dimensionless flux scale")
    o2_operator = o2_operator.rename("o2_effective_observation_operator").assign_coords(
        **{
            name: coordinate
            for name, coordinate in operator_coords.items()
            if name not in o2_operator.coords
        },
    ).assign_attrs(
        units=f"{o2_units} per dimensionless flux scale",
        oxidation_ratio_convention=o2_operator_ratio_convention,
        oxidation_ratio_direction="O2 flux per CO2 flux",
        oxidation_ratio_sign="signed; positive CO2 flux has negative O2 loading",
        oxidation_ratio_scope="shared GPP/TER/FF states; O2 ocean applied directly",
    )
    covariance.attrs["units"] = "observation_units * observation_units_cov"
    aggregation_error = resolve_aggregation_error(
        xr.Dataset(
            {
                AGGREGATION_ERROR_COVARIANCE: covariance,
            }
        ),
        "dense",
        output_dim="observation",
        covariance_dim="observation_cov",
    )
    return Co2O2PreparedInputs(
        observations=observations,
        prior_forward_mean=prior_forward_mean,
        co2_operator=co2_operator,
        o2_operator=o2_operator,
        o2_operator_ratio_convention=o2_operator_ratio_convention,
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        co2_observation_dim=co2_dim,
        o2_observation_dim=o2_dim,
        provenance=prepared_provenance,
    )
