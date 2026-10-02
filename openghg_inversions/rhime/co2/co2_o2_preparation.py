"""Labelled scientific inputs for the CO2/O2 recipe.

CO2 and O2 keep distinct, potentially unequal observation axes at this public
boundary. They are stacked only after their labels, state meanings, covariance
blocks, and units have been checked.

Prepared inputs support a versioned DataTree handoff and NetCDF or Zarr
persistence. In-memory conversion preserves borrowed lazy payloads. Saving
passes lazily densified arrays to xarray's writer, which computes their chunks
while writing; loading eagerly restores and validates the saved artifact.
Preparation checks independent-error labels and eager values while leaving lazy
error payloads and unit coordinates for the loading or model boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Self, cast

from dask import compute as dask_compute
from dask.array import Array as DaskArray
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import concat_gather_data_arrays, select_gathered_data_array, to_dense
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AggregationError,
    resolve_aggregation_error,
)
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    open_datatree_loaded,
    save_datatree,
)


CO2_O2_PREPARED_INPUTS_SCHEMA = "openghg_inversions.co2_o2_prepared_inputs"
CO2_O2_PREPARED_INPUTS_SCHEMA_VERSION = 1


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
        co2_sensitivity: Effective CO2 sensitivity on the native CO2 observation and
            retained-state dimensions, with row labels
            matching the native CO2 observations and state labels matching the
            retained prior. Units are CO2 observation units per dimensionless
            flux scale.
        o2_sensitivity: Effective O2 sensitivity on the native O2 observation and
            retained-state dimensions, with corresponding row and state
            labels. Shared-state columns already contain signed O2-per-CO2
            ratios; the O2-ocean column is applied directly.
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
        provenance: JSON-serializable preparation and data provenance.
        boundary_sensitivity: Optional mapping keyed by co2 and o2, containing
            native-channel H_bc arrays (observation, boundary state).
        independent_error_sd: Optional positive independent-error standard
            deviations on the joint observation axis, with matching row units.
            The borrowed payload may remain lazy.
    """

    observations: xr.DataArray
    fixed_prior_contribution: xr.DataArray
    co2_sensitivity: xr.DataArray
    o2_sensitivity: xr.DataArray
    o2_co2_flux_ratio: xr.DataArray | None
    o2_co2_flux_ratio_unavailable_reason: str | None
    aggregation_error: AggregationError
    retained_prior: CorrelatedLognormalPrior
    provenance: Mapping[str, Any] = field(default_factory=dict)
    boundary_sensitivity: Mapping[str, xr.DataArray] = field(default_factory=dict)
    independent_error_sd: xr.DataArray | None = None

    def to_datatree(self) -> xr.DataTree:
        """Build a versioned tree without writing or materializing payloads.

        Native channels and the shared-state ratio occupy separate nodes so Dataset
        alignment cannot pad ragged observations or ratio states.

        Returns:
            A versioned DataTree retaining borrowed lazy array payloads.

        Raises:
            ValueError: If dense aggregation covariance is absent, boundary
                channels are unsupported, or metadata cannot be serialized.
        """
        covariance = self.aggregation_error.covariance
        if self.aggregation_error.mode != "dense" or covariance is None:
            raise ValueError("CO2/O2 prepared inputs require dense aggregation error.")
        if set(self.boundary_sensitivity) - {"co2", "o2"}:
            raise ValueError("boundary_sensitivity must be keyed only by co2 and o2.")
        arrays = {
            "joint": {
                "observed_concentration": self.observations,
                "fixed_prior_contribution": self.fixed_prior_contribution,
                AGGREGATION_ERROR_COVARIANCE: covariance,
            },
            "co2_sensitivity": {"co2_effective_sensitivity": self.co2_sensitivity},
            "o2_sensitivity": {"o2_effective_sensitivity": self.o2_sensitivity},
            "retained_prior": {
                "arithmetic_mean": self.retained_prior.mean,
                "arithmetic_covariance": self.retained_prior.arithmetic_covariance,
            },
        }
        if self.o2_co2_flux_ratio is not None:
            arrays["flux_ratio"] = {"o2_co2_flux_ratio": self.o2_co2_flux_ratio}
        if self.independent_error_sd is not None:
            arrays["independent_error"] = {"independent_error_sd": self.independent_error_sd}
        for channel, boundary in self.boundary_sensitivity.items():
            arrays[f"{channel}_boundary"] = {"boundary_sensitivity": boundary}
        datasets: dict[str, xr.Dataset] = {}
        for name, variables in arrays.items():
            dataset = xr.Dataset(variables)
            indexes = [
                cast(str, dim) for dim in dataset.dims if isinstance(dataset.indexes.get(dim), pd.MultiIndex)
            ]
            if indexes:
                dataset = encode_cf_multiindexes(dataset, indexes)
            datasets[name] = dataset.assign_attrs(
                multiindex_dims_json=json.dumps(indexes),
                array_names_json=json.dumps({variable: array.name for variable, array in variables.items()}),
            )
        datasets["retained_prior"].attrs.update(
            state_dim=self.retained_prior.state_dim, covariance_dim=self.retained_prior.covariance_dim
        )
        tree = xr.DataTree.from_dict(datasets)
        tree.attrs = {
            "schema": CO2_O2_PREPARED_INPUTS_SCHEMA,
            "schema_version": CO2_O2_PREPARED_INPUTS_SCHEMA_VERSION,
            "provenance_json": json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False),
            "o2_co2_flux_ratio_unavailable_reason": self.o2_co2_flux_ratio_unavailable_reason or "",
        }
        return tree

    @classmethod
    def from_datatree(cls, tree: xr.DataTree) -> Self:
        """Restore and validate a version-1 artifact, preserving its saved intercept.

        Args:
            tree: Tree using the dedicated CO2/O2 prepared-input schema.

        Returns:
            Validated scientific inputs. Observations, fixed intercept, dense
            covariance, prior moments, ocean-loading slices, available ratio
            values, and independent error are materialized for validation;
            sensitivity payloads may otherwise remain lazy.

        Raises:
            KeyError: If a required node or scientific variable is absent.
            ValueError: If schema metadata or scientific contents are invalid.
        """
        if tree.attrs.get("schema") != CO2_O2_PREPARED_INPUTS_SCHEMA:
            raise ValueError(f"Expected Co2O2PreparedInputs schema {CO2_O2_PREPARED_INPUTS_SCHEMA!r}.")
        version = tree.attrs.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, Integral) or version != 1:
            raise ValueError(f"Expected Co2O2PreparedInputs schema_version 1; got {version!r}.")
        unexpected_boundaries = {
            name
            for name in tree.children
            if name.endswith("_boundary") and name not in {"co2_boundary", "o2_boundary"}
        }
        if unexpected_boundaries:
            raise ValueError(f"Unsupported serialized boundary nodes: {sorted(unexpected_boundaries)!r}.")
        datasets: dict[str, xr.Dataset] = {}
        for name, node in tree.children.items():
            dataset = node.to_dataset()
            try:
                indexes = json.loads(dataset.attrs["multiindex_dims_json"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid MultiIndex metadata in node {name!r}.") from exc
            if not isinstance(indexes, list) or any(not isinstance(dim, str) for dim in indexes):
                raise ValueError(f"Invalid MultiIndex metadata in node {name!r}.")
            compressed = {
                str(dim) for dim in dataset.dims if dim in dataset.coords and "compress" in dataset[dim].attrs
            }
            if set(indexes) != compressed or len(indexes) != len(compressed):
                raise ValueError(f"Declared MultiIndexes do not match CF coordinates in node {name!r}.")
            datasets[name] = decode_cf_multiindexes(dataset, indexes) if indexes else dataset
        try:
            provenance = json.loads(tree.attrs["provenance_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Serialized CO2/O2 provenance must be a JSON object.") from exc
        if not isinstance(provenance, dict):
            raise ValueError("Serialized CO2/O2 provenance must be a JSON object.")
        prior_data = datasets["retained_prior"]
        prior = CorrelatedLognormalPrior(
            _stored_array(prior_data, "arithmetic_mean"),
            _stored_array(prior_data, "arithmetic_covariance"),
            covariance_dim=prior_data.attrs["covariance_dim"],
        )
        if prior.state_dim != prior_data.attrs["state_dim"]:
            raise ValueError("Serialized retained-prior state_dim does not match its arithmetic mean.")
        joint = datasets["joint"]
        aggregation_error = resolve_aggregation_error(
            joint, "dense", output_dim="observation", covariance_dim="observation_cov"
        )
        covariance = aggregation_error.covariance
        if covariance is None:
            raise ValueError("CO2/O2 prepared inputs require dense aggregation error.")
        aggregation_error = replace(
            aggregation_error,
            covariance=covariance.rename(_stored_array(joint, AGGREGATION_ERROR_COVARIANCE).name),
        )
        reason = tree.attrs.get("o2_co2_flux_ratio_unavailable_reason")
        if not isinstance(reason, str):
            raise ValueError("Serialized ratio unavailable reason must be a string.")
        prepared = cls(
            observations=_stored_array(joint, "observed_concentration"),
            fixed_prior_contribution=_stored_array(joint, "fixed_prior_contribution"),
            co2_sensitivity=_stored_array(datasets["co2_sensitivity"], "co2_effective_sensitivity"),
            o2_sensitivity=_stored_array(datasets["o2_sensitivity"], "o2_effective_sensitivity"),
            o2_co2_flux_ratio=(
                _stored_array(datasets["flux_ratio"], "o2_co2_flux_ratio")
                if "flux_ratio" in datasets
                else None
            ),
            o2_co2_flux_ratio_unavailable_reason=reason.strip() or None,
            aggregation_error=aggregation_error,
            retained_prior=prior,
            provenance=provenance,
            boundary_sensitivity={
                channel: _stored_array(datasets[f"{channel}_boundary"], "boundary_sensitivity")
                for channel in ("co2", "o2")
                if f"{channel}_boundary" in datasets
            },
            independent_error_sd=(
                _stored_array(datasets["independent_error"], "independent_error_sd")
                if "independent_error" in datasets
                else None
            ),
        )
        _validate_prepared_inputs(prepared)
        return prepared

    def save(
        self,
        output_file: str | Path,
        output_format: Literal["netcdf", "zarr"] | None = None,
    ) -> None:
        """Save prepared inputs through xarray's chunked writer.

        Inputs are borrowed and are not mutated. Sparse Dask chunks are lazily
        densified; the Zarr writer lazily regularizes chunks for storage before
        executing the related array graphs. Prepared scientific values are trusted here;
        external artifacts are validated when loaded. An existing destination
        artifact is replaced.

        Args:
            output_file: Destination NetCDF file or Zarr store.
            output_format: Explicit "netcdf" or "zarr" format. When omitted,
                infer the format from the destination's .nc or .zarr suffix.

        Raises:
            ValueError: If schema metadata or the output format is invalid.
        """
        tree = self.to_datatree()
        for name in list(tree.children):
            dataset = tree[name].to_dataset()
            tree[name] = dataset.assign(
                {
                    variable: array.copy(deep=False, data=to_dense(array).data)
                    for variable, array in dataset.data_vars.items()
                }
            )
        save_datatree(tree, output_file, output_format)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """Eagerly load and validate a NetCDF file or Zarr store.

        Args:
            file_path: Prepared-input artifact to read.

        Returns:
            Fully materialized inputs with no references to closed file handles.

        Raises:
            KeyError: If a required node or scientific variable is absent.
            ValueError: If the schema or scientific contents are invalid.
            OSError: If the artifact cannot be opened.
        """
        return cls.from_datatree(open_datatree_loaded(file_path))


def _axis(array: xr.DataArray, name: str) -> str:
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.dims!r}.")
    dim = str(array.dims[0])
    if dim not in array.indexes or not array.indexes[dim].is_unique:
        raise ValueError(f"{name} requires unique labels on {dim!r}.")
    return dim


def _stored_array(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    """Restore a scientific array's name without altering its borrowed payload.

    Args:
        dataset: Decoded node containing array_names_json metadata.
        variable: Schema field whose original name should be restored.

    Returns:
        The selected array with its recorded original name and existing data.

    Raises:
        KeyError: If the required scientific field is absent.
        ValueError: If its name metadata is missing, malformed, or unsupported.
    """
    try:
        names = json.loads(dataset.attrs["array_names_json"])
        name = names[variable]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid array-name metadata for {variable!r}.") from exc
    if name is not None and not isinstance(name, (str, int, float)):
        raise ValueError(f"Invalid saved array name for {variable!r}.")
    return dataset[variable].rename(name)


def _same_axis(reference: xr.DataArray, candidate: xr.DataArray, name: str) -> None:
    dim = str(reference.dims[0])
    if candidate.dims != (dim,) or not _same_index(candidate.indexes[dim], reference.indexes[dim]):
        raise ValueError(f"{name} labels must exactly match its observations.")


def _same_index(left: pd.Index, right: pd.Index) -> bool:
    """Compare index values and metadata used by xarray alignment."""
    return left.equals(right) and left.names == right.names


def _validate_independent_error(
    observations: xr.DataArray,
    independent_error_sd: xr.DataArray | None,
    *,
    materialize: bool = False,
) -> None:
    """Check error structure, deferring borrowed lazy values until requested.

    Args:
        observations: Canonical gathered observation labels and row units.
        independent_error_sd: Optional labelled error standard deviations.
        materialize: Compute payload and unit coordinates together to validate
            their values when loading an external artifact. Otherwise,
            inspect only already-eager values and preserve lazy execution.

    Raises:
        ValueError: If labels, row-unit structure, or available values disagree,
            or error values are not finite positive real numbers.
    """
    if independent_error_sd is None:
        return
    _same_axis(observations, independent_error_sd, "independent_error_sd")
    coordinate = independent_error_sd.coords.get("observation_units")
    if coordinate is None or coordinate.dims != observations.dims:
        raise ValueError("independent_error_sd requires observation-aligned observation_units.")
    values = to_dense(independent_error_sd).data
    units = coordinate.data
    expected_units = observations["observation_units"].data
    if materialize:
        values, units, expected_units = dask_compute(values, units, expected_units)
    if (
        not isinstance(units, DaskArray)
        and not isinstance(expected_units, DaskArray)
        and not np.array_equal(units, expected_units)
    ):
        raise ValueError("independent_error_sd observation_units must match prepared observations.")
    if isinstance(values, DaskArray):
        return
    values = np.asarray(values)
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("independent_error_sd must contain only finite positive real numeric values.")


def _validate_prepared_inputs(prepared: Co2O2PreparedInputs) -> None:
    """Validate the external scientific handoff without rebuilding its intercept.

    Args:
        prepared: Restored inputs whose dense covariance and correlated prior
            have already been validated by their owning constructors.

    Raises:
        ValueError: If numeric payloads, labels, row units, state meanings,
            signed-ratio provenance, or boundary channels violate the schema.

    Notes:
        Called at the loading boundary. Observation and intercept
        payloads are computed together for finite real-number checks. Ocean
        slices, available ratios, independent error, and auxiliary units are
        materialized explicitly for their remaining scientific checks.
    """
    observations = prepared.observations
    _axis(observations, "Joint observations")
    index = observations.indexes.get("observation")
    if observations.dims != ("observation",) or not isinstance(index, pd.MultiIndex):
        raise ValueError("Joint observations require the observation MultiIndex.")
    observation_values, intercept_values = dask_compute(
        to_dense(observations).data, to_dense(prepared.fixed_prior_contribution).data
    )
    for name, values in (
        ("observations", observation_values),
        ("fixed_prior_contribution", intercept_values),
    ):
        values = np.asarray(values)
        if (
            not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{name} must contain only finite real numeric values.")
    if index.names[0] != "species" or set(index.get_level_values("species")) != {"co2", "o2"}:
        raise ValueError("Joint observation labels must gather exactly CO2 and O2 by species.")
    species = index.get_level_values("species").to_numpy()
    if not np.array_equal(species, np.concatenate((species[species == "co2"], species[species == "o2"]))):
        raise ValueError("Joint observations must be ordered CO2 followed by O2.")
    units = observations.coords.get("observation_units")
    if units is None or units.dims != ("observation",):
        raise ValueError("Joint observations require observation-aligned observation_units.")
    unit_values = np.asarray(dask_compute(units.data)[0]).astype(str)
    channel_units: dict[str, str] = {}
    native_observations: dict[str, xr.DataArray] = {}
    state_mean = _state(prepared.retained_prior)
    for channel, sensitivity in (("co2", prepared.co2_sensitivity), ("o2", prepared.o2_sensitivity)):
        if sensitivity.ndim != 2:
            raise ValueError(f"{channel} sensitivity must have native observation and state dimensions.")
        channel_values = np.unique(unit_values[species == channel])
        if channel_values.size != 1 or not channel_values[0].strip():
            raise ValueError(f"{channel} observations must have one non-empty unit label.")
        channel_units[channel] = str(channel_values[0])
        native = select_gathered_data_array(
            observations,
            key=channel,
            key_dim="species",
            ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename({"observation": sensitivity.dims[0]})
        native_observations[channel] = native
        _sensitivity(sensitivity, native, state_mean, channel.upper())
        for coordinate in ("source", "tracer_scope"):
            if coordinate not in sensitivity.coords or not sensitivity[coordinate].variable.equals(
                state_mean[coordinate].variable
            ):
                raise ValueError(f"{channel} sensitivity {coordinate} must match the retained prior.")
    if prepared.co2_sensitivity.dims[0] == prepared.o2_sensitivity.dims[0]:
        raise ValueError("CO2 and O2 require distinct native observation dimension names.")
    _same_axis(observations, prepared.fixed_prior_contribution, "fixed_prior_contribution")
    ratio = _ratio_provenance(
        prepared.o2_co2_flux_ratio, prepared.o2_co2_flux_ratio_unavailable_reason, state_mean
    )
    if ratio is None and not str(prepared.o2_co2_flux_ratio_unavailable_reason or "").strip():
        raise ValueError("Unavailable O2/CO2 flux ratios require a non-empty reason.")
    _materialize_and_validate_ocean_loadings_and_ratio(
        prepared.co2_sensitivity, prepared.o2_sensitivity, state_mean, ratio
    )
    try:
        actual_record = json.loads(prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("O2 sensitivity requires valid signed-ratio provenance.") from exc
    if not isinstance(actual_record, dict):
        raise ValueError("O2 sensitivity signed-ratio provenance must be a JSON object.")
    covariance = prepared.aggregation_error.covariance
    if prepared.aggregation_error.mode != "dense" or covariance is None:
        raise ValueError("CO2/O2 prepared inputs require dense aggregation error.")
    column_units = covariance.coords.get("observation_units_cov")
    if (
        column_units is None
        or column_units.dims != ("observation_cov",)
        or not np.array_equal(dask_compute(column_units.data)[0], unit_values)
    ):
        raise ValueError(
            "Aggregation covariance observation_units_cov must match prepared observation units."
        )
    if set(prepared.boundary_sensitivity) - {"co2", "o2"}:
        raise ValueError("boundary_sensitivity must be keyed only by co2 and o2.")
    if prepared.boundary_sensitivity and channel_units["co2"] != channel_units["o2"]:
        raise ValueError("Linked boundary sensitivity currently requires identical channel units.")
    for channel, boundary in prepared.boundary_sensitivity.items():
        native = native_observations[channel]
        if boundary.ndim != 2 or boundary.dims[0] != native.dims[0]:
            raise ValueError(
                f"{channel} boundary sensitivity requires native observation rows and one state axis."
            )
        xr.align(boundary, native, join="exact", copy=False)
        if boundary.dims[1] not in boundary.indexes or not boundary.indexes[boundary.dims[1]].is_unique:
            raise ValueError(f"{channel} boundary states require unique labels.")
    _validate_independent_error(observations, prepared.independent_error_sd, materialize=True)


def _ratio_record(
    ratio: xr.DataArray | None,
    unavailable_reason: str | None,
    ratio_values: np.ndarray,
) -> dict[str, object]:
    """Describe signed ratios already embedded in shared-state sensitivities."""
    record: dict[str, object] = {
        "status": "available" if ratio is not None else "unavailable",
        "direction": "O2 flux per CO2 flux",
        "sign_convention": "signed; positive CO2 flux has negative O2 loading",
    }
    if ratio is None:
        record["unavailable_reason"] = unavailable_reason
    else:
        record.update(
            state=[str(label) for label in ratio.indexes[str(ratio.dims[0])]],
            source=[str(source) for source in ratio["source"].values],
            value=ratio_values.tolist(),
            provenance=ratio.attrs["provenance"],
        )
    return record


def _state(prior: CorrelatedLognormalPrior) -> xr.DataArray:
    mean = prior.mean
    if any(name not in mean.coords for name in ("source", "tracer_scope")):
        raise ValueError("Retained states require source and tracer_scope coordinates.")
    sources = {str(source) for source in mean["source"].values}
    if len(sources) != len({source.lower() for source in sources}):
        raise ValueError(
            f"Retained source labels must use consistent spelling; found case variants in {sorted(sources)}."
        )
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
    return mean


def _sensitivity(
    value: xr.DataArray,
    observation: xr.DataArray,
    state_mean: xr.DataArray,
    name: str,
) -> None:
    observation_dim = str(observation.dims[0])
    state_dim = str(state_mean.dims[0])
    if value.dims != (observation_dim, state_dim):
        raise ValueError(f"{name} sensitivity must have dimensions {(observation_dim, state_dim)!r}.")
    if not _same_index(value.indexes[observation_dim], observation.indexes[observation_dim]):
        raise ValueError(f"{name} sensitivity rows do not match its observations.")
    if not _same_index(value.indexes[state_dim], state_mean.indexes[state_dim]):
        raise ValueError(
            f"{name} sensitivity state labels and index level names must match the retained prior."
        )


def _materialize_and_validate_ocean_loadings_and_ratio(
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
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
        co2_sensitivity.isel({state_dim: o2_ocean}).data,
        o2_sensitivity.isel({state_dim: co2_ocean}).data,
    ]
    if o2_co2_flux_ratio is not None:
        collections.append(o2_co2_flux_ratio.data)
    computed = dask_compute(*collections)
    co2_cross, o2_cross = computed[:2]
    if np.any(co2_cross != 0):
        raise ValueError("CO2 sensitivity must have zero loadings for O2-specific ocean states.")
    if np.any(o2_cross != 0):
        raise ValueError("O2 sensitivity must have zero loadings for CO2-specific ocean states.")
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
    if (value is None) == (unavailable_reason is None):
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
        species: value.rename(name)
        .assign_coords(
            observation_units=(value.dims[0], np.full(value.size, units)),
        )
        .rename({value.dims[0]: "channel_observation"})
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
    column_index = observation_index.set_names([f"{name}_cov" for name in observation_index.names])
    return (
        covariance.assign_coords(xr.Coordinates.from_pandas_multiindex(observation_index, "observation"))
        .assign_coords(xr.Coordinates.from_pandas_multiindex(column_index, "observation_cov"))
        .rename(AGGREGATION_ERROR_COVARIANCE)
    )


def prepare_co2_o2_inputs(
    *,
    co2_observations: xr.DataArray,
    o2_observations: xr.DataArray,
    co2_prior_forward_mean: xr.DataArray,
    o2_prior_forward_mean: xr.DataArray,
    co2_sensitivity: xr.DataArray,
    o2_sensitivity: xr.DataArray,
    o2_co2_flux_ratio: xr.DataArray | None,
    o2_co2_flux_ratio_unavailable_reason: str | None,
    co2_aggregation_covariance: xr.DataArray,
    co2_o2_aggregation_covariance: xr.DataArray,
    o2_aggregation_covariance: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    co2_units: str,
    o2_units: str,
    provenance: Mapping[str, Any] | None = None,
    boundary_sensitivity: Mapping[str, xr.DataArray] | None = None,
    independent_error_sd: xr.DataArray | None = None,
) -> Co2O2PreparedInputs:
    """Validate coherent-reduction channel products and form one joint likelihood.

    The concrete recipe treats the O2 sensitivity as already containing fixed,
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
        co2_sensitivity: CO2 effective sensitivity with dimensions
            ``(CO2 observation, retained state)`` and exact observation and
            retained-prior indexes. Its O2-ocean column must be zero.
        o2_sensitivity: O2 effective sensitivity with dimensions
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
            CO2 ocean, and O2 ocean states. Repeated source labels must use
            one consistent spelling, including case.
        co2_units: Non-empty units label for CO2 observations and sensitivity rows.
        o2_units: Non-empty units label for O2 observations and sensitivity rows.
        provenance: Optional JSON-serializable preparation provenance.
        boundary_sensitivity: Optional co2/o2 mapping of labelled H_bc arrays
            on native observation rows and one unique boundary-state axis.
            Channels must currently have identical units. Payloads remain borrowed.
        independent_error_sd: Optional finite positive independent-error standard
            deviations on the gathered observation axis, with matching labels
            and observation_units. Eager values are checked here; lazy payloads
            and unit coordinates are checked at the loading or model
            boundary. This borrowed array is retained for replay.

    Returns:
        Labelled, backend-neutral joint inputs. Observation vectors, affine
        intercept, sensitivities, independent error, and available ratio
        provenance retain borrowed lazy payloads; dense covariance validation is the explicit eager
        aggregation-error boundary.

    Raises:
        ValueError: If units or provenance are invalid; observation, sensitivity,
            state, ratio, or covariance dimensions/indexes disagree; source
            labels contain inconsistent case variants; the
            ratio exactly-one, direction, sign, provenance, or numerical-value
            contract fails; cross-tracer ocean loadings are nonzero; or the
            assembled dense covariance is non-finite, asymmetric, or not
            positive semidefinite.
    """
    if not co2_units.strip() or not o2_units.strip():
        raise ValueError("CO2 and O2 channel units must be non-empty.")
    try:
        prepared_provenance = json.loads(json.dumps(dict(provenance or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("CO2/O2 provenance must be JSON serializable.") from exc

    boundary_sensitivity = dict(boundary_sensitivity or {})
    if set(boundary_sensitivity) - {"co2", "o2"}:
        raise ValueError("boundary_sensitivity must be keyed only by co2 and o2.")
    if boundary_sensitivity and co2_units != o2_units:
        raise ValueError("Linked boundary sensitivity currently requires identical channel units.")
    for channel, observations in (("co2", co2_observations), ("o2", o2_observations)):
        if channel in boundary_sensitivity:
            boundary = boundary_sensitivity[channel]
            if boundary.ndim != 2 or boundary.dims[0] != observations.dims[0]:
                raise ValueError(
                    f"{channel} boundary sensitivity requires native observation rows and one state axis."
                )
            boundary, _ = xr.align(boundary, observations, join="exact", copy=False)
            declared_units = boundary.attrs.get("units")
            if declared_units is not None and declared_units not in (
                co2_units,
                f"{co2_units} per dimensionless boundary scale",
            ):
                raise ValueError(f"{channel} boundary sensitivity units must match its channel units.")
            state_dim = boundary.dims[1]
            if state_dim not in boundary.indexes or not boundary.indexes[state_dim].is_unique:
                raise ValueError(f"{channel} boundary states require unique labels.")
            boundary_sensitivity[channel] = boundary.assign_attrs(
                units=f"{co2_units} per dimensionless boundary scale"
            )

    co2_dim = _axis(co2_observations, "CO2 observations")
    o2_dim = _axis(o2_observations, "O2 observations")
    if co2_dim == o2_dim:
        raise ValueError("CO2 and O2 require distinct pre-stacking dimension names.")
    _same_axis(co2_observations, co2_prior_forward_mean, "CO2 prior forward mean")
    _same_axis(o2_observations, o2_prior_forward_mean, "O2 prior forward mean")
    state_mean = _state(retained_prior)
    _sensitivity(co2_sensitivity, co2_observations, state_mean, "CO2")
    _sensitivity(o2_sensitivity, o2_observations, state_mean, "O2")
    o2_co2_flux_ratio_unavailable_reason = str(o2_co2_flux_ratio_unavailable_reason or "").strip() or None
    o2_co2_flux_ratio = _ratio_provenance(
        o2_co2_flux_ratio,
        o2_co2_flux_ratio_unavailable_reason,
        state_mean,
    )
    ratio_values = _materialize_and_validate_ocean_loadings_and_ratio(
        co2_sensitivity,
        o2_sensitivity,
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
    _validate_independent_error(observations, independent_error_sd)
    observation_index = observations.indexes["observation"]
    if not isinstance(observation_index, pd.MultiIndex):  # pragma: no cover - helper invariant
        raise TypeError("Joint observations require a gathered MultiIndex.")
    covariance = _joint_covariance(
        co2_covariance,
        cross_covariance,
        o2_covariance,
        observation_index=observation_index,
    ).assign_coords(
        observation_units=observations["observation_units"],
        observation_units_cov=(
            "observation_cov",
            observations["observation_units"].values,
        ),
    )
    state_dim = retained_prior.state_dim
    co2_intercept = co2_prior_forward_mean - xr.dot(
        co2_sensitivity,
        retained_prior.mean,
        dim=state_dim,
    )
    o2_intercept = o2_prior_forward_mean - xr.dot(
        o2_sensitivity,
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
    sensitivity_coords = {name: state_mean[name] for name in ("source", "tracer_scope")}
    state_index = state_mean.indexes[state_dim]
    if isinstance(state_index, pd.MultiIndex):
        # The validated state index already owns these level coordinates;
        # replacing individual levels would corrupt the MultiIndex.
        sensitivity_coords = {
            name: coordinate
            for name, coordinate in sensitivity_coords.items()
            if name not in state_index.names
        }
    co2_sensitivity = (
        co2_sensitivity.rename("co2_effective_sensitivity")
        .assign_coords(sensitivity_coords)
        .assign_attrs(units=f"{co2_units} per dimensionless flux scale")
    )
    ratio_direction = "O2 flux per CO2 flux"
    ratio_sign = "signed; positive CO2 flux has negative O2 loading"
    ratio_record = _ratio_record(o2_co2_flux_ratio, o2_co2_flux_ratio_unavailable_reason, ratio_values)
    o2_sensitivity = (
        o2_sensitivity.rename("o2_effective_sensitivity")
        .assign_coords(sensitivity_coords)
        .assign_attrs(
            units=f"{o2_units} per dimensionless flux scale",
            oxidation_ratio_convention="embedded_signed_o2_per_co2",
            oxidation_ratio_direction=ratio_direction,
            oxidation_ratio_sign=ratio_sign,
            oxidation_ratio_scope="shared GPP/TER/FF states; O2 ocean applied directly",
            oxidation_ratio_provenance=json.dumps(ratio_record, sort_keys=True),
        )
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
        co2_sensitivity=co2_sensitivity,
        o2_sensitivity=o2_sensitivity,
        o2_co2_flux_ratio=o2_co2_flux_ratio,
        o2_co2_flux_ratio_unavailable_reason=o2_co2_flux_ratio_unavailable_reason,
        aggregation_error=aggregation_error,
        retained_prior=retained_prior,
        provenance=prepared_provenance,
        boundary_sensitivity=boundary_sensitivity,
        independent_error_sd=independent_error_sd,
    )
