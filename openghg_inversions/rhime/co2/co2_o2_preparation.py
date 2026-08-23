"""Labelled scientific inputs for the CO2/O2 recipe.

CO2 and O2 keep distinct, potentially unequal observation axes at this public
boundary. They are stacked only after their labels, state meanings, covariance
blocks, and units have been checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any

from dask import compute as dask_compute
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import concat_gather_data_arrays
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AggregationError,
    resolve_aggregation_error,
)


@dataclass(frozen=True, slots=True, eq=False)
class Co2O2PreparedInputs:
    """Backend-neutral joint inputs with shared land and split-ocean states.

    Callers should normally obtain this handoff from
    :func:`prepare_co2_o2_inputs` rather than constructing it directly.

    Attributes:
        observations: CO2 followed by O2 observations on ``("observation",)``.
            Its gathered MultiIndex records ``species`` and the native channel
            labels, while ``observation_units`` records channel units. Aligned
            ``site`` and ``time`` coordinates are retained when supplied.
        fixed_prior_contribution: Joint affine intercept
            ``H m - H_alpha Pi m`` on the same labelled ``observation`` axis
            and in the channel-specific observation units.
        co2_operator: Effective CO2 operator on
            ``(co2_observation_dim, retained_prior.state_dim)`` with row labels
            matching the native CO2 observations and state labels matching the
            retained prior. Units are CO2 observation units per dimensionless
            flux scale.
        o2_operator: Effective O2 operator on
            ``(o2_observation_dim, retained_prior.state_dim)`` with the
            corresponding O2 row labels and retained-state labels. Shared-state
            columns already contain signed O2-per-CO2 ratios; the O2-ocean
            column is applied directly.
        o2_co2_flux_ratio: Optional signed, finite, negative O2-per-CO2 ratios
            on ``(retained_prior.state_dim,)`` for exactly the shared GPP, TER,
            and FF states. The indexed state labels and ``source`` coordinate
            match the retained prior, while attrs record direction, sign
            convention, and provenance. The borrowed payload may remain lazy.
        o2_co2_flux_ratio_unavailable_reason: Non-empty explanation when
            scalar state-resolved ratios cannot be exposed because the paired
            native O2 flux embeds spatial ratios before convolution. Exactly
            one of this value and ``o2_co2_flux_ratio`` is present.
        aggregation_error: Validated dense joint aggregation error. Covariance
            rows use ``observation`` and columns use ``observation_cov`` in
            block order ``[[CO2, CO2/O2], [CO2/O2.T, O2]]``; per-axis unit
            coordinates describe mixed-unit entries.
        retained_prior: Correlated prior over shared GPP/TER/FF and separate
            CO2- and O2-ocean retained states.
        co2_observation_dim: Native indexed CO2 observation dimension name.
        o2_observation_dim: Native indexed O2 observation dimension name.
        provenance: JSON-serializable preparation and data provenance.
    """

    observations: xr.DataArray
    fixed_prior_contribution: xr.DataArray
    co2_operator: xr.DataArray
    o2_operator: xr.DataArray
    o2_co2_flux_ratio: xr.DataArray | None
    o2_co2_flux_ratio_unavailable_reason: str | None
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
    if candidate.dims != (dim,) or not _same_index(
        candidate.indexes[dim], reference.indexes[dim]
    ):
        raise ValueError(f"{name} labels must exactly match its observations.")


def _same_index(left: pd.Index, right: pd.Index) -> bool:
    """Compare index values and metadata used by xarray alignment."""
    return left.equals(right) and left.names == right.names


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
    if not _same_index(value.indexes[observation_dim], observation.indexes[observation_dim]):
        raise ValueError(f"{name} operator rows do not match its observations.")
    if not _same_index(value.indexes[state_dim], state_mean.indexes[state_dim]):
        raise ValueError(
            f"{name} operator state labels and index level names must match the retained prior."
        )


def _materialize_and_validate_ocean_loadings_and_ratio(
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    state_mean: xr.DataArray,
    o2_co2_flux_ratio: xr.DataArray | None,
) -> np.ndarray:
    """Jointly materialize and validate cross-ocean loadings and ratio values."""
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
    collections = [
        co2_operator.isel({state_dim: o2_ocean}).data,
        o2_operator.isel({state_dim: co2_ocean}).data,
    ]
    if o2_co2_flux_ratio is not None:
        collections.append(o2_co2_flux_ratio.data)
    computed = dask_compute(*collections)
    co2_cross, o2_cross = computed[:2]
    if np.any(co2_cross != 0):
        raise ValueError("CO2 operator must have zero loadings for O2-specific ocean states.")
    if np.any(o2_cross != 0):
        raise ValueError("O2 operator must have zero loadings for CO2-specific ocean states.")
    if o2_co2_flux_ratio is None:
        return np.empty(0)
    ratio_values = np.asarray(computed[2])
    if not np.isfinite(ratio_values).all() or np.any(ratio_values >= 0):
        raise ValueError("Available O2/CO2 flux ratios must contain only finite negative values.")
    return ratio_values


def _ratio_provenance(
    value: xr.DataArray | None,
    unavailable_reason: str | None,
    state_mean: xr.DataArray,
) -> xr.DataArray | None:
    """Validate signed O2-per-CO2 ratios against the retained shared states."""
    reason = str(unavailable_reason or "").strip()
    if (value is None) == (not reason):
        raise ValueError(
            "Supply exactly one of labelled O2/CO2 flux ratios or a non-empty unavailable reason."
        )
    if value is None:
        return None
    state_dim = str(state_mean.dims[0])
    shared = [
        index
        for index, scope in enumerate(state_mean["tracer_scope"].values)
        if str(scope).lower() == "shared"
    ]
    shared_mean = state_mean.isel({state_dim: shared})
    if value.dims != (state_dim,) or state_dim not in value.indexes:
        raise ValueError(f"O2/CO2 flux ratios must have one indexed {state_dim!r} dimension.")
    if not _same_index(value.indexes[state_dim], shared_mean.indexes[state_dim]):
        raise ValueError("O2/CO2 flux ratio state labels must match the retained shared states.")
    if "source" not in value.coords or value["source"].dims != (state_dim,):
        raise ValueError("O2/CO2 flux ratios require a source coordinate on the shared states.")
    if not np.array_equal(value["source"].values, shared_mean["source"].values):
        raise ValueError("O2/CO2 flux ratio sources must match the retained shared states.")
    if value.attrs.get("direction") != "O2 flux per CO2 flux":
        raise ValueError("O2/CO2 flux ratio direction must be 'O2 flux per CO2 flux'.")
    expected_sign = "signed; positive CO2 flux has negative O2 loading"
    if value.attrs.get("sign_convention") != expected_sign:
        raise ValueError(f"O2/CO2 flux ratio sign_convention must be {expected_sign!r}.")
    if not str(value.attrs.get("provenance", "")).strip():
        raise ValueError("Available O2/CO2 flux ratios require non-empty provenance metadata.")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("O2/CO2 flux ratios must be numeric.")
    return value.rename("o2_co2_flux_ratio")


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
    if not _same_index(value.indexes[row_dim], row.indexes[row_dim]):
        raise ValueError(f"{name} row labels do not match its observations.")
    if not value.indexes[value_column_dim].equals(column.indexes[column_dim]):
        raise ValueError(f"{name} column labels do not match its observations.")
    return value


def _channel_vector(
    value: xr.DataArray,
    *,
    units: str,
    name: str,
) -> xr.DataArray:
    dim = str(value.dims[0])
    return value.rename(name).assign_coords(
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
    channels = {
        species: _channel_vector(value, units=units, name=name).rename(
            {str(value.dims[0]): "channel_observation"}
        )
        for species, value, units in (
            ("co2", co2, co2_units),
            ("o2", o2, o2_units),
        )
    }
    stacked = concat_gather_data_arrays(
        channels,
        key_dim="species",
        ragged_dim="channel_observation",
        stack_dim="observation",
        join="exact",
    )
    stacked.attrs["units"] = "mixed; see observation_units coordinate"
    return stacked


def _joint_covariance(
    co2_covariance: xr.DataArray,
    cross_covariance: xr.DataArray,
    o2_covariance: xr.DataArray,
    *,
    observation_index: pd.MultiIndex,
) -> xr.DataArray:
    """Combine validated labelled channel blocks without materializing them."""
    nco2 = co2_covariance.shape[0]
    no2 = o2_covariance.shape[0]
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
    covariance = covariance.drop_indexes(("observation", "observation_cov")).drop_vars(
        ("observation", "observation_cov")
    )
    column_index = observation_index.set_names(
        [f"{name}_cov" for name in observation_index.names]
    )
    return covariance.assign_coords(
        xr.Coordinates.from_pandas_multiindex(observation_index, "observation")
    ).assign_coords(
        xr.Coordinates.from_pandas_multiindex(column_index, "observation_cov")
    ).rename(AGGREGATION_ERROR_COVARIANCE)


def prepare_co2_o2_inputs(
    *,
    co2_observations: xr.DataArray,
    o2_observations: xr.DataArray,
    co2_prior_forward_mean: xr.DataArray,
    o2_prior_forward_mean: xr.DataArray,
    co2_operator: xr.DataArray,
    o2_operator: xr.DataArray,
    o2_co2_flux_ratio: xr.DataArray | None,
    o2_co2_flux_ratio_unavailable_reason: str | None,
    co2_aggregation_covariance: xr.DataArray,
    co2_o2_aggregation_covariance: xr.DataArray,
    o2_aggregation_covariance: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    co2_units: str,
    o2_units: str,
    provenance: Mapping[str, Any] | None = None,
) -> Co2O2PreparedInputs:
    """Validate coherent-reduction channel products and form one joint likelihood.

    The concrete recipe treats the O2 operator as already containing fixed,
    signed O2-per-CO2 ratios for shared states. Supply their labelled values
    when they remain available, or an explicit reason why a native paired-flux
    construction cannot expose scalar ratios at this boundary.

    Args:
        co2_observations: One-dimensional CO2 observations with a unique
            indexed native observation coordinate.
        o2_observations: One-dimensional O2 observations with a unique indexed
            coordinate whose dimension name differs from the CO2 dimension.
            Times and lengths may differ between channels.
        co2_prior_forward_mean: Native CO2 prior mean ``H m`` on exactly the
            CO2 observation dimension and labels.
        o2_prior_forward_mean: Native O2 prior mean ``H m`` on exactly the O2
            observation dimension and labels.
        co2_operator: CO2 effective operator with dimensions
            ``(CO2 observation, retained state)`` and exact observation and
            retained-prior indexes. Its O2-ocean column must be zero.
        o2_operator: O2 effective operator with dimensions
            ``(O2 observation, retained state)`` and exact observation and
            retained-prior indexes. Its CO2-ocean column must be zero; signed
            O2-per-CO2 ratios are already embedded in shared-state columns.
        o2_co2_flux_ratio: Optional labelled ratios for exactly the shared
            retained states. Values must be finite and negative, the ``source``
            coordinate must match the prior, and attrs must declare direction
            ``"O2 flux per CO2 flux"``, the signed convention, and provenance.
        o2_co2_flux_ratio_unavailable_reason: Explanation used only when
            labelled scalar ratios are unavailable. Exactly one of this
            argument and ``o2_co2_flux_ratio`` must be supplied.
        co2_aggregation_covariance: CO2-by-CO2 dense covariance. Rows use the
            CO2 observation dimension; its distinct column dimension carries
            the same CO2 labels in the same order. Entries have squared CO2
            observation units.
        co2_o2_aggregation_covariance: CO2-row by O2-column cross-covariance,
            labelled by the native CO2 and O2 observation indexes. Entries
            have CO2 observation units times O2 observation units.
        o2_aggregation_covariance: O2-by-O2 dense covariance. Rows use the O2
            observation dimension; its distinct column dimension carries the
            same O2 labels in the same order. Entries have squared O2
            observation units.
        retained_prior: Retained correlated prior whose indexed state axis has
            ``source`` and ``tracer_scope`` coordinates for shared GPP/TER/FF,
            CO2 ocean, and O2 ocean states.
        co2_units: Non-empty units label for CO2 observations and operator rows.
        o2_units: Non-empty units label for O2 observations and operator rows.
        provenance: Optional JSON-serializable preparation provenance.

    Returns:
        Labelled, backend-neutral joint inputs. Observation vectors, affine
        intercept, operators, and available ratio provenance retain borrowed
        lazy payloads; dense covariance validation is the explicit eager
        aggregation-error boundary.

    Raises:
        ValueError: If units or provenance are invalid; observation, operator,
            state, ratio, or covariance dimensions/indexes disagree; the
            ratio exactly-one, direction, sign, provenance, or numerical-value
            contract fails; cross-tracer ocean loadings are nonzero; or the
            assembled dense covariance is non-finite, asymmetric, or not
            positive semidefinite.
    """
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
    o2_co2_flux_ratio = _ratio_provenance(
        o2_co2_flux_ratio,
        o2_co2_flux_ratio_unavailable_reason,
        state_mean,
    )
    ratio_values = _materialize_and_validate_ocean_loadings_and_ratio(
        co2_operator,
        o2_operator,
        state_mean,
        o2_co2_flux_ratio,
    )

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
    observations = _stack(
        co2_observations,
        o2_observations,
        co2_units=co2_units,
        o2_units=o2_units,
        name="observed_concentration",
    )
    observation_index = observations.indexes["observation"]
    if not isinstance(observation_index, pd.MultiIndex):  # pragma: no cover - helper invariant
        raise TypeError("Joint observations require a gathered MultiIndex.")
    covariance = _joint_covariance(
        co2_covariance,
        cross_covariance,
        o2_covariance,
        observation_index=observation_index,
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
    state_dim = retained_prior.state_dim
    co2_intercept = co2_prior_forward_mean - xr.dot(
        co2_operator,
        retained_prior.mean,
        dim=state_dim,
    )
    o2_intercept = o2_prior_forward_mean - xr.dot(
        o2_operator,
        retained_prior.mean,
        dim=state_dim,
    )
    fixed_prior_contribution = _stack(
        co2_intercept,
        o2_intercept,
        co2_units=co2_units,
        o2_units=o2_units,
        name="fixed_prior_contribution",
    )
    fixed_prior_contribution.attrs["mathematical_name"] = "H m - H_alpha Pi m"
    operator_coords = {name: state_mean[name] for name in ("source", "tracer_scope")}
    state_index = state_mean.indexes[state_dim]
    if isinstance(state_index, pd.MultiIndex):
        # The validated state index already owns these level coordinates;
        # replacing individual levels would corrupt the MultiIndex.
        operator_coords = {
            name: coordinate
            for name, coordinate in operator_coords.items()
            if name not in state_index.names
        }
    co2_operator = co2_operator.rename("co2_effective_observation_operator").assign_coords(
        **operator_coords,
    ).assign_attrs(units=f"{co2_units} per dimensionless flux scale")
    ratio_direction = "O2 flux per CO2 flux"
    ratio_sign = "signed; positive CO2 flux has negative O2 loading"
    ratio_status = "available" if o2_co2_flux_ratio is not None else "unavailable"
    ratio_record: dict[str, object] = {
        "status": ratio_status,
        "direction": ratio_direction,
        "sign_convention": ratio_sign,
    }
    if o2_co2_flux_ratio is not None:
        ratio_record.update(
            state=[str(label) for label in o2_co2_flux_ratio.indexes[state_dim]],
            source=[str(source) for source in o2_co2_flux_ratio["source"].values],
            value=ratio_values.tolist(),
            provenance=o2_co2_flux_ratio.attrs["provenance"],
        )
    else:
        ratio_record["unavailable_reason"] = o2_co2_flux_ratio_unavailable_reason
    o2_operator = o2_operator.rename("o2_effective_observation_operator").assign_coords(
        **operator_coords,
    ).assign_attrs(
        units=f"{o2_units} per dimensionless flux scale",
        oxidation_ratio_convention="embedded_signed_o2_per_co2",
        oxidation_ratio_direction=ratio_direction,
        oxidation_ratio_sign=ratio_sign,
        oxidation_ratio_scope="shared GPP/TER/FF states; O2 ocean applied directly",
        oxidation_ratio_provenance=json.dumps(ratio_record, sort_keys=True),
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
        fixed_prior_contribution=fixed_prior_contribution,
        co2_operator=co2_operator,
        o2_operator=o2_operator,
        o2_co2_flux_ratio=o2_co2_flux_ratio,
        o2_co2_flux_ratio_unavailable_reason=(
            None
            if o2_co2_flux_ratio is not None
            else str(o2_co2_flux_ratio_unavailable_reason).strip()
        ),
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        co2_observation_dim=co2_dim,
        o2_observation_dim=o2_dim,
        provenance=prepared_provenance,
    )
