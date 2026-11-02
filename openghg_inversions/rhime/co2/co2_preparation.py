"""Prepare durable inputs for the coherent CO2 model recipes.

This module composes canonical RHIME inputs with one coherent reduction and
owns the selected dense or low-rank aggregation-error representation. Inputs
are borrowed; serialization and low-rank factorization are explicit eager
boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast

import pandas as pd
import numpy as np
import xarray as xr
from openghg.util import (  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
    cf_ureg,
)

from openghg_inversions._serialization_codecs import _copy_json_mapping, _decode_json_mapping
from openghg_inversions.array_ops import require_unique_index, same_index, validate_covariance_coordinates
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AGGREGATION_ERROR_SD,
    DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR,
    prepare_low_rank_aggregation_error,
    resolve_aggregation_error,
)
from openghg_inversions.serialization import open_datatree_loaded, save_datatree

from .co2_o2_preparation import Co2O2PreparedInputs


CO2_PREPARED_INPUTS_SCHEMA = "openghg_inversions.co2_prepared_inputs"
CO2_PREPARED_INPUTS_SCHEMA_VERSION = 2
Co2AggregationErrorMode = Literal["dense", "low_rank"]
_AGGREGATION_PAYLOAD_NAMES = (
    AGGREGATION_ERROR_COVARIANCE,
    AGGREGATION_ERROR_SD,
    LOW_RANK_FACTOR,
    DIAGONAL_RESIDUAL_VARIANCE,
)
_AGGREGATION_REPRESENTATION_DIMS = {"agg_rank", "nmeasure_cov"}


def _json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable JSON-safe copy."""
    return MappingProxyType(_copy_json_mapping(value, "CO2 preparation provenance", sort_keys=True))


_require_axis = require_unique_index


def _require_same_axis(
    array: xr.DataArray,
    dim: str,
    expected: pd.Index,
    *,
    name: str,
) -> None:
    """Require exact axis labels, including MultiIndex level names."""
    if not same_index(_require_axis(array, dim, name=name), expected):
        raise ValueError(f"{name} labels must exactly match the canonical inputs.")


def _renamed(array: xr.DataArray, names: Mapping[str, str], *, name: str) -> xr.DataArray:
    """Rename scientific axes and the concrete variable without copying data."""
    return array.rename({old: new for old, new in names.items() if old != new}).rename(name)


def _borrow_without_axis_coordinates(array: xr.DataArray, dim: str) -> xr.DataArray:
    """Let the destination dataset provide coordinates for an aligned axis."""
    return xr.DataArray(
        array.data,
        dims=array.dims,
        coords={name: coord for name, coord in array.coords.items() if dim not in coord.dims},
        attrs=array.attrs,
        name=array.name,
    )


def _without_aggregation_payload(inputs: xr.Dataset, *, target_dim: str) -> xr.Dataset:
    """Remove the old aggregation payload without discarding shared consumers.

    One-dimensional coordinates on an aggregation representation dimension
    are owned by that dimension and are removed with it. Data variables and
    multidimensional coordinates using that dimension are rejected.

    Args:
        inputs: Dataset that may contain an existing aggregation payload.
        target_dim: Representation dimension required by the replacement
            payload.

    Returns:
        A dataset without the old aggregation payload or its owned dimensions.

    Raises:
        ValueError: If an owned or target representation dimension has a
            non-aggregation consumer.
    """
    owned_dims = {
        dim
        for name in _AGGREGATION_PAYLOAD_NAMES
        if name in inputs
        for dim in inputs[name].dims
        if dim in _AGGREGATION_REPRESENTATION_DIMS
    }
    for dim in owned_dims:
        consumers = [
            name
            for name, value in inputs.data_vars.items()
            if name not in _AGGREGATION_PAYLOAD_NAMES and dim in value.dims
        ]
        coordinate_consumers = [
            name
            for name, value in inputs.coords.items()
            if dim in value.dims and value.dims != (dim,)
        ]
        if consumers or coordinate_consumers:
            raise ValueError(
                f"Aggregation representation dimension {dim!r} is also used by "
                f"non-aggregation variable(s): {consumers + coordinate_consumers!r}."
            )

    cleaned = inputs
    for dim in owned_dims:
        cleaned = cleaned.drop_dims(dim)
    cleaned = cleaned.drop_vars(_AGGREGATION_PAYLOAD_NAMES, errors="ignore")
    if target_dim in cleaned.dims:
        consumers = [
            name
            for name, value in cleaned.variables.items()
            if target_dim in value.dims
        ]
        raise ValueError(
            f"Aggregation representation dimension {target_dim!r} is already used by "
            f"non-aggregation variable(s): {consumers!r}."
        )
    return cleaned


def _require_equivalent_units(actual: Any, expected: str, *, name: str) -> None:
    """Require units to have the same dimension and numeric scale."""
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError(f"{name} requires non-empty units.")
    try:
        actual_quantity = cf_ureg.parse_expression(actual)
        expected_quantity = cf_ureg.parse_expression(expected)
        if not hasattr(actual_quantity, "to"):
            actual_quantity = cf_ureg.Quantity(actual_quantity)
        if not hasattr(expected_quantity, "to"):
            expected_quantity = cf_ureg.Quantity(expected_quantity)
        scale = float(actual_quantity.to(expected_quantity.units).magnitude / expected_quantity.magnitude)
    except Exception as exc:
        raise ValueError(f"{name} units {actual!r} are incompatible with {expected!r}.") from exc
    if not np.isclose(scale, 1.0, rtol=1e-12, atol=0.0):
        raise ValueError(f"{name} units {actual!r} do not have the same numeric scale as {expected!r}.")


def _validate_co2_dataset(inputs: xr.Dataset) -> None:
    """Validate the scientific layout owned by the CO2 artifact."""
    from openghg_inversions.boundary_sensitivity import validate_boundary_operands

    validate_boundary_operands(inputs)
    required = (
        "H",
        "alpha_prior_mean",
        "alpha_prior_covariance",
        "fixed_prior_contribution",
        "mf",
        "mf_error",
    )
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"CO2 prepared inputs are missing required variable(s): {missing!r}.")

    observations = inputs["mf"]
    if observations.dims != ("nmeasure",):
        raise ValueError("CO2 observations must have dimensions ('nmeasure',).")
    observation_index = _require_axis(observations, "nmeasure", name="CO2 observations")
    sensitivity = inputs["H"]
    if sensitivity.ndim != 2 or sensitivity.dims[0] != "nmeasure":
        raise ValueError("CO2 H must be observation-by-state with nmeasure first.")
    state_dim = str(sensitivity.dims[1])
    state_index = _require_axis(sensitivity, state_dim, name="CO2 H")
    _require_same_axis(sensitivity, "nmeasure", observation_index, name="CO2 H")

    mean = inputs["alpha_prior_mean"]
    if mean.dims != (state_dim,):
        raise ValueError("CO2 alpha_prior_mean must use the H retained-state dimension.")
    _require_same_axis(mean, state_dim, state_index, name="CO2 alpha_prior_mean")
    covariance_dim = f"{state_dim}_cov"
    covariance = inputs["alpha_prior_covariance"]
    validate_covariance_coordinates(
        covariance,
        dim=state_dim,
        covariance_dim=covariance_dim,
    )
    for name in ("fixed_prior_contribution", "mf_error"):
        value = inputs[name]
        if value.dims != ("nmeasure",):
            raise ValueError(f"CO2 {name} must have dimensions ('nmeasure',).")
        _require_same_axis(value, "nmeasure", observation_index, name=f"CO2 {name}")

    concentration_units = observations.attrs.get("units")
    if not concentration_units:
        raise ValueError("CO2 observations require units.")
    mole_fraction_unit_scale(str(concentration_units), context="CO2 prepared observations")
    for name in ("H", "fixed_prior_contribution", "mf_error"):
        _require_equivalent_units(
            inputs[name].attrs.get("units"),
            str(concentration_units),
            name=f"CO2 {name}",
        )
    if mean.attrs.get("units") != "1" or covariance.attrs.get("units") != "1":
        raise ValueError("CO2 retained prior mean and covariance must be dimensionless.")

    squared_units = f"({concentration_units})**2"
    if AGGREGATION_ERROR_COVARIANCE in inputs:
        _require_equivalent_units(
            inputs[AGGREGATION_ERROR_COVARIANCE].attrs.get("units"),
            squared_units,
            name="CO2 aggregation_error_covariance",
        )
    if LOW_RANK_FACTOR in inputs:
        _require_equivalent_units(
            inputs[LOW_RANK_FACTOR].attrs.get("units"),
            str(concentration_units),
            name="CO2 low_rank_factor",
        )
    if DIAGONAL_RESIDUAL_VARIANCE in inputs:
        _require_equivalent_units(
            inputs[DIAGONAL_RESIDUAL_VARIANCE].attrs.get("units"),
            squared_units,
            name="CO2 diagonal_residual_variance",
        )

    for name in ("state_is_active", "state_fixed_value"):
        if name in inputs and inputs[name].dims != (state_dim,):
            raise ValueError(f"CO2 {name} must use the retained-state dimension.")


@dataclass(frozen=True, slots=True, eq=False)
class Co2PreparedInputs:
    """Durable coherent-reduction inputs for the CO2 model recipes.

    The public :attr:`rhime_inputs` member makes composition explicit for
    shared RHIME consumers. This CO2 boundary additionally owns one
    unambiguous aggregation-error representation and preparation provenance.

    Callers should normally construct this value with
    :func:`prepare_co2_inputs` rather than calling the dataclass constructor.
    The frozen dataclass prevents attribute rebinding, but its composed xarray
    objects are borrowed and may remain mutable. Construction raises
    ``ValueError`` when the selected representation, labels, units, or
    provenance violate this boundary's contract.

    Attributes:
        rhime_inputs: Canonical RHIME inputs composed into this recipe-specific
            boundary and passed directly to shared RHIME consumers.
        aggregation_error_mode: Concrete dense or low-rank representation
            stored in :attr:`inv_inputs` and consumed by both CO2 runners.
        provenance: Top-level read-only JSON-safe preparation record. Low-rank
            artifacts include approximation diagnostics.
    """

    rhime_inputs: PreparedInputs
    aggregation_error_mode: Co2AggregationErrorMode
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aggregation_error_mode not in ("dense", "low_rank"):
            raise ValueError("CO2 aggregation_error_mode must be 'dense' or 'low_rank'.")
        inputs = self.rhime_inputs.inv_inputs
        _validate_co2_dataset(inputs)
        co2_supplied_baseline_input_names(inputs)
        dense = AGGREGATION_ERROR_COVARIANCE in inputs
        factor_present = LOW_RANK_FACTOR in inputs
        diagonal_present = DIAGONAL_RESIDUAL_VARIANCE in inputs
        if factor_present != diagonal_present:
            missing = DIAGONAL_RESIDUAL_VARIANCE if factor_present else LOW_RANK_FACTOR
            raise ValueError(
                "Low-rank CO2 inputs require low_rank_factor and "
                f"diagonal_residual_variance together; missing {missing!r}."
            )
        low_rank = factor_present and diagonal_present
        if self.aggregation_error_mode == "dense" and (not dense or low_rank):
            raise ValueError("Dense CO2 inputs must contain only the dense aggregation covariance.")
        if self.aggregation_error_mode == "low_rank" and (dense or not low_rank):
            raise ValueError("Low-rank CO2 inputs must contain only the factor and diagonal residual.")
        if dense:
            validate_covariance_coordinates(
                inputs[AGGREGATION_ERROR_COVARIANCE],
                dim="nmeasure",
                covariance_dim="nmeasure_cov",
            )
        if low_rank:
            factor = inputs[LOW_RANK_FACTOR]
            diagonal = inputs[DIAGONAL_RESIDUAL_VARIANCE]
            if factor.dims != ("nmeasure", "agg_rank"):
                raise ValueError("CO2 low_rank_factor must have dimensions ('nmeasure', 'agg_rank').")
            if diagonal.dims != ("nmeasure",):
                raise ValueError("CO2 diagonal_residual_variance must have dimensions ('nmeasure',).")
        object.__setattr__(self, "provenance", _json_mapping(self.provenance))

    @property
    def inv_inputs(self) -> xr.Dataset:
        """Return the labelled dataset consumed by CO2 runners."""
        return self.rhime_inputs.inv_inputs

    @property
    def basis_functions(self) -> Any:
        """Return the retained basis object."""
        return self.rhime_inputs.basis_functions

    @property
    def site_metadata(self) -> xr.Dataset:
        """Return authoritative site metadata."""
        return self.rhime_inputs.site_metadata

    @property
    def sites(self) -> tuple[str, ...]:
        """Return site labels in indicator-decoding order."""
        return self.rhime_inputs.sites

    @property
    def averaging_period(self) -> tuple[str | None, ...]:
        """Return averaging periods aligned to :attr:`sites`."""
        return self.rhime_inputs.averaging_period

    @property
    def basis_artifact_source(self) -> str:
        """Return retained basis provenance."""
        return self.rhime_inputs.basis_artifact_source

    @property
    def basis_artifact_path(self) -> str | None:
        """Return the provenance-only basis artifact path."""
        return self.rhime_inputs.basis_artifact_path

    def to_datatree(self) -> xr.DataTree:
        """Build the versioned in-memory serialization tree.

        Aggregation-error arrays are validated and materialized together at
        this serialization boundary. This method performs no file I/O.

        Returns:
            A versioned DataTree containing the CO2 artifact.

        Raises:
            ValueError: If the aggregation-error representation is invalid or
                provenance cannot be serialized.
        """
        aggregation_error = resolve_aggregation_error(
            self.inv_inputs,
            self.aggregation_error_mode,
        )
        variables = dict(self.inv_inputs.variables)
        if aggregation_error.covariance is not None:
            variables[AGGREGATION_ERROR_COVARIANCE] = variables[
                AGGREGATION_ERROR_COVARIANCE
            ].copy(deep=False, data=aggregation_error.covariance.data)
        if aggregation_error.factor is not None:
            variables[LOW_RANK_FACTOR] = variables[LOW_RANK_FACTOR].copy(
                deep=False,
                data=aggregation_error.factor.data,
            )
        if aggregation_error.diagonal_variance is not None:
            variables[DIAGONAL_RESIDUAL_VARIANCE] = variables[
                DIAGONAL_RESIDUAL_VARIANCE
            ].copy(deep=False, data=aggregation_error.diagonal_variance.data)
        serializable_inputs = self.inv_inputs._replace(variables=variables)
        serializable = PreparedInputs(
            inv_inputs=serializable_inputs,
            basis_functions=self.basis_functions,
            site_metadata=self.site_metadata,
        )
        tree = xr.DataTree.from_dict({"rhime_inputs": serializable.to_datatree()})
        tree.attrs = {
            "schema": CO2_PREPARED_INPUTS_SCHEMA,
            "schema_version": CO2_PREPARED_INPUTS_SCHEMA_VERSION if "H_bc_anomaly" in self.inv_inputs else 1,
            "aggregation_error_mode": self.aggregation_error_mode,
            "provenance_json": json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False),
        }
        return tree

    @classmethod
    def from_datatree(cls, tree: xr.DataTree) -> Self:
        """Restore absolute or centered CO2 inputs from the dedicated schema.

        Args:
            tree: In-memory tree using the CO2 prepared-input schema.

        Returns:
            The validated prepared CO2 artifact.

        Raises:
            KeyError: If the required ``rhime_inputs`` node is absent.
            ValueError: If schema metadata, provenance, or the aggregation
                representation is invalid.
        """
        if tree.attrs.get("schema") != CO2_PREPARED_INPUTS_SCHEMA:
            raise ValueError(
                f"Expected Co2PreparedInputs schema {CO2_PREPARED_INPUTS_SCHEMA!r}, "
                f"got {tree.attrs.get('schema')!r}."
            )
        version = tree.attrs.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, Integral) or version not in (1, 2):
            raise ValueError(
                f"Expected Co2PreparedInputs schema_version 1 or 2, got {version!r}."
            )
        if "rhime_inputs" not in tree.children:
            raise KeyError("Missing required Co2PreparedInputs node 'rhime_inputs'.")
        mode = tree.attrs.get("aggregation_error_mode")
        if mode not in ("dense", "low_rank"):
            raise ValueError(f"Invalid serialized CO2 aggregation_error_mode {mode!r}.")
        provenance = _decode_json_mapping(
            tree.attrs.get("provenance_json", "{}"), "Serialized CO2 provenance"
        )
        prepared = cls(
            PreparedInputs.from_datatree(cast(xr.DataTree, tree["rhime_inputs"])),
            aggregation_error_mode=cast(Co2AggregationErrorMode, mode),
            provenance=provenance,
        )
        if version == 1 and "H_bc_anomaly" in prepared.inv_inputs:
            raise ValueError("Centered boundary operands require CO2 prepared schema version 2.")
        resolve_aggregation_error(prepared.inv_inputs, prepared.aggregation_error_mode)
        return prepared

    def save(
        self,
        output_file: str | Path,
        output_format: Literal["netcdf", "zarr"] | None = None,
    ) -> None:
        """Materialize and save the prepared CO2 artifact.

        Args:
            output_file: Destination NetCDF file or Zarr store.
            output_format: Explicit ``"netcdf"`` or ``"zarr"`` format. When
                omitted, infer the format from ``output_file``.

        Raises:
            ValueError: If the artifact or requested output format is invalid.
        """
        save_datatree(self.to_datatree(), output_file, output_format)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """Load and validate a prepared CO2 artifact from disk.

        Args:
            file_path: NetCDF file or Zarr store to load eagerly.

        Returns:
            The fully materialized prepared CO2 artifact.

        Raises:
            KeyError: If a required schema node is absent.
            ValueError: If schema metadata or scientific contents are invalid.
        """
        return cls.from_datatree(open_datatree_loaded(file_path))


def requested_co2_facts(
    prepared: Co2PreparedInputs | Co2O2PreparedInputs, *, start_date: str | None = None,
    end_date: str | None = None, domain: str | None = None,
    recorded_facts: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Return authoritative requested facts without inferring observation extrema.

    Explicit values fill missing provenance facts. Repeated facts must agree;
    dates compare by timestamp meaning. ``recorded_facts`` supplies declared
    configuration or handoff facts alongside numerical preparation provenance;
    all claims must agree. Only start/end dates and domain are consumed.
    Imported preparations missing facts require explicit caller values;
    observation extrema cannot recover the original requested window.
    """
    supplied = {"start_date": start_date, "end_date": end_date, "domain": domain}
    facts: dict[str, str] = {}
    for name, value in supplied.items():
        for claim in (prepared.provenance.get(name), (recorded_facts or {}).get(name), value):
            if claim is None:
                continue
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(f"CO2 preparation requires explicit requested facts: {[name]!r}.")
            if name in facts:
                recorded = facts[name]
                equivalent = (
                    pd.Timestamp(claim) == pd.Timestamp(recorded)
                    if name in {"start_date", "end_date"} else claim == recorded
                )
                if not equivalent:
                    raise ValueError(
                        f"CO2 requested {name} conflicts with preparation or declared request: {claim!r} != {recorded!r}."
                    )
            else:
                facts[name] = claim
    missing = [name for name in supplied if name not in facts]
    if missing:
        raise ValueError(f"CO2 preparation requires explicit requested facts: {missing!r}.")
    start, end = pd.Timestamp(facts["start_date"]), pd.Timestamp(facts["end_date"])
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("CO2 requested start_date must precede a valid end_date.")
    return facts


def prepare_co2_inputs(
    canonical_inputs: PreparedInputs,
    reduction: CoherentGaussianReduction,
    *,
    aggregation_error_rank: int | None = 512,
    provenance: Mapping[str, Any] | None = None,
) -> Co2PreparedInputs:
    """Map one coherent Gaussian reduction into the CO2 replay contract.

    The reduction replaces the canonical sensitivity with its effective
    operator and supplies the linked retained prior, affine intercept, and
    unresolved covariance. By default, the handoff retains at most 512 LRPD
    modes. ``None`` stores the unresolved covariance exactly; requested ranks
    larger than the observation count are capped at that count, and the actual
    factor width may be smaller when the covariance has lower numerical rank.

    Inputs are borrowed and are not mutated. The returned artifact retains the
    canonical observations, observation error, optional boundary data, state
    activity, basis functions, and site metadata. Observation and retained-state
    dimensions must be labelled and aligned between the canonical inputs and
    reduction. Concentration units must be compatible at the same numeric scale;
    values are not converted.

    Selecting LRPD storage eagerly materializes the complete unresolved
    covariance and computes its full dense eigendecomposition. The retained
    rank reduces the saved payload and downstream structured-likelihood cost,
    not this preparation cost.

    Args:
        canonical_inputs: Canonical RHIME observations and metadata whose
            observation and retained-state labels match ``reduction``.
        reduction: One coherent Gaussian reduction containing all linked
            retained-prior and observation products.
        aggregation_error_rank: Positive LRPD rank, defaulting to 512. Values
            larger than the observation count are capped at that count; the
            actual factor width is further limited by numerical positive rank.
            Pass ``None`` to store the exact dense covariance.
        provenance: Optional JSON-serializable project or preparation
            provenance. The reduction strategy and LRPD diagnostics are added
            by this boundary.

    Returns:
        A structurally validated, serializable artifact accepted by both CO2
        runners.

    Raises:
        ValueError: If labels, dimensions, units, rank, covariance
            representation, or provenance are invalid.
    """
    canonical = canonical_inputs.validated()
    inputs = canonical.inv_inputs
    if "H" not in inputs or "mf" not in inputs:
        raise ValueError("Canonical CO2 inputs require 'H' and 'mf'.")
    observations = inputs["mf"]
    if observations.dims != ("nmeasure",):
        raise ValueError("Canonical CO2 observations must have dimensions ('nmeasure',).")
    observation_index = _require_axis(observations, "nmeasure", name="Canonical observations")

    canonical_sensitivity = inputs["H"]
    state_dims = [str(dim) for dim in canonical_sensitivity.dims if dim != "nmeasure"]
    if canonical_sensitivity.ndim != 2 or len(state_dims) != 1:
        raise ValueError("Canonical CO2 H must have one observation and one retained-state dimension.")
    state_dim = state_dims[0]
    state_index = _require_axis(canonical_sensitivity, state_dim, name="Canonical sensitivity")

    reduced_mean = reduction.retained_mean
    if reduced_mean.ndim != 1:
        raise ValueError("Coherent retained_mean must be one-dimensional.")
    reduced_state_dim = str(reduced_mean.dims[0])
    _require_same_axis(reduced_mean, reduced_state_dim, state_index, name="Coherent retained_mean")

    intercept = reduction.observation_intercept
    if intercept.ndim != 1:
        raise ValueError("Coherent observation_intercept must be one-dimensional.")
    observation_dim = str(intercept.dims[0])
    for name, array in (
        ("native_observation_mean", reduction.native_observation_mean),
        ("observation_intercept", intercept),
    ):
        if array.dims != (observation_dim,):
            raise ValueError(f"Coherent {name} must have dimensions ({observation_dim!r},).")
        _require_same_axis(array, observation_dim, observation_index, name=f"Coherent {name}")

    effective = reduction.effective_observation_operator
    if effective.dims != (observation_dim, reduced_state_dim):
        raise ValueError("Coherent effective_observation_operator must be observation-by-state.")
    _require_same_axis(effective, observation_dim, observation_index, name="Coherent operator")
    _require_same_axis(effective, reduced_state_dim, state_index, name="Coherent operator")

    retained_covariance = reduction.retained_covariance
    if retained_covariance.ndim != 2 or retained_covariance.dims[0] != reduced_state_dim:
        raise ValueError("Coherent retained_covariance must be state-by-state.")
    reduced_covariance_dim = str(retained_covariance.dims[1])
    _require_same_axis(
        retained_covariance,
        reduced_state_dim,
        state_index,
        name="Coherent retained covariance rows",
    )
    if not retained_covariance.indexes[reduced_covariance_dim].equals(state_index):
        raise ValueError("Coherent retained covariance columns must match the retained states.")

    unresolved = reduction.unresolved_observation_covariance
    if unresolved.ndim != 2 or unresolved.dims[0] != observation_dim:
        raise ValueError("Coherent unresolved_observation_covariance must be observation-by-observation.")
    observation_covariance_dim = str(unresolved.dims[1])
    _require_same_axis(unresolved, observation_dim, observation_index, name="Unresolved covariance rows")
    if not unresolved.indexes[observation_covariance_dim].equals(observation_index):
        raise ValueError("Unresolved covariance columns must match the observations.")

    observation_units = observations.attrs.get("units")
    if not observation_units:
        raise ValueError("Canonical CO2 observations require units.")
    for name, array in (
        ("effective_observation_operator", effective),
        ("native_observation_mean", reduction.native_observation_mean),
        ("observation_intercept", intercept),
    ):
        _require_equivalent_units(
            array.attrs.get("units"),
            str(observation_units),
            name=f"Coherent {name}",
        )
    if reduced_mean.attrs.get("units") != "1" or retained_covariance.attrs.get("units") != "1":
        raise ValueError("Coherent retained prior mean and covariance must be dimensionless.")
    _require_equivalent_units(
        unresolved.attrs.get("units"),
        f"({observation_units})**2",
        name="Coherent unresolved_observation_covariance",
    )
    if (
        aggregation_error_rank is not None
        and (
            isinstance(aggregation_error_rank, bool)
            or not isinstance(aggregation_error_rank, int)
            or aggregation_error_rank < 1
        )
    ):
        raise ValueError("aggregation_error_rank must be a positive integer or None.")

    target_representation_dim = "nmeasure_cov" if aggregation_error_rank is None else "agg_rank"
    mapped = _without_aggregation_payload(
        inputs,
        target_dim=target_representation_dim,
    ).drop_vars(
        (
            "H",
            "alpha_prior_mean",
            "alpha_prior_covariance",
            "fixed_prior_contribution",
        ),
        errors="ignore",
    )
    mapped["H"] = _borrow_without_axis_coordinates(
        _renamed(
            effective,
            {observation_dim: "nmeasure", reduced_state_dim: state_dim},
            name="H",
        ),
        "nmeasure",
    )
    mapped["alpha_prior_mean"] = _renamed(
        reduced_mean,
        {reduced_state_dim: state_dim},
        name="alpha_prior_mean",
    )
    state_covariance_dim = f"{state_dim}_cov"
    mapped["alpha_prior_covariance"] = _renamed(
        retained_covariance,
        {reduced_state_dim: state_dim, reduced_covariance_dim: state_covariance_dim},
        name="alpha_prior_covariance",
    )
    mapped["fixed_prior_contribution"] = _borrow_without_axis_coordinates(
        _renamed(
            intercept,
            {observation_dim: "nmeasure"},
            name="fixed_prior_contribution",
        ),
        "nmeasure",
    )

    approximation = None
    if aggregation_error_rank is None:
        mode: Co2AggregationErrorMode = "dense"
        mapped[AGGREGATION_ERROR_COVARIANCE] = _borrow_without_axis_coordinates(
            _renamed(
                unresolved,
                {observation_dim: "nmeasure", observation_covariance_dim: "nmeasure_cov"},
                name=AGGREGATION_ERROR_COVARIANCE,
            ),
            "nmeasure",
        )
    else:
        approximation = prepare_low_rank_aggregation_error(
            unresolved,
            rank=min(aggregation_error_rank, unresolved.sizes[observation_dim]),
            output_dim=observation_dim,
            covariance_dim=observation_covariance_dim,
        )
        selected = approximation.aggregation_error
        assert selected.factor is not None and selected.diagonal_variance is not None
        factor_dim = str(selected.factor.dims[0])
        diagonal_dim = str(selected.diagonal_variance.dims[0])
        mapped[LOW_RANK_FACTOR] = _borrow_without_axis_coordinates(
            _renamed(
                selected.factor,
                {factor_dim: "nmeasure"},
                name=LOW_RANK_FACTOR,
            ).assign_attrs({**selected.factor.attrs, "units": observation_units}),
            "nmeasure",
        )
        mapped[DIAGONAL_RESIDUAL_VARIANCE] = _borrow_without_axis_coordinates(
            _renamed(
                selected.diagonal_variance,
                {diagonal_dim: "nmeasure"},
                name=DIAGONAL_RESIDUAL_VARIANCE,
            ).assign_attrs(
                {
                    **selected.diagonal_variance.attrs,
                    "units": unresolved.attrs.get("units", f"({observation_units})^2"),
                }
            ),
            "nmeasure",
        )
        mode = "low_rank"

    metadata = dict(provenance or {})
    declared_strategy = metadata.get("projection_strategy")
    if declared_strategy is not None and declared_strategy != reduction.projection_strategy:
        raise ValueError("CO2 provenance projection_strategy conflicts with the coherent reduction.")
    metadata["projection_strategy"] = reduction.projection_strategy
    metadata["aggregation_error_rank"] = aggregation_error_rank
    if approximation is not None:
        metadata["aggregation_error"] = dict(approximation.diagnostics)
    return Co2PreparedInputs(
        PreparedInputs(
            inv_inputs=mapped,
            basis_functions=canonical.basis_functions,
            site_metadata=canonical.site_metadata,
        ),
        aggregation_error_mode=mode,
        provenance=metadata,
    )


def co2_supplied_baseline_input_names(inputs: xr.Dataset, *, use_bc: bool = False) -> tuple[str, ...]:
    """Validate the absolute-background CO2 contract without computing arrays."""
    from ..baseline import supplied_baseline_input_names

    if "supplied_baseline" not in inputs:
        return ()
    if use_bc or any(name in inputs for name in ("H_bc", "H_bc_anomaly", "H_bc_mass", "bc_reference")):
        raise ValueError("A supplied absolute CO2 baseline replaces all sampled boundary inputs.")
    if "inner_fixed_contribution" in inputs:
        raise ValueError("CO2 inner affine contributions belong only in fixed_prior_contribution.")
    if inputs.supplied_baseline.attrs.get("aggregation_error_policy") != "inherited":
        raise ValueError("Supplied CO2 baselines require explicit inherited aggregation-error accounting.")
    supplied_baseline_input_names(
        inputs.assign(inner_fixed_contribution=inputs.fixed_prior_contribution), species="co2", use_bc=False,
    )
    return ("supplied_baseline",)


def _inner_basis(prepared: Co2PreparedInputs, positions: np.ndarray, support: xr.DataArray) -> Any:
    """Retain selected bucket geometry for native outputs, never for coherent H."""
    from dataclasses import replace
    from dask import is_dask_collection
    from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator
    from .._domain_support import remove_domain_overlap

    basis = prepared.basis_functions
    operator = basis.operator
    state_dim = operator.meta.state_dim
    original = operator.basis_matrix.get_index(state_dim)
    retained = prepared.inv_inputs.H.get_index(state_dim)
    if original.names != retained.names or not original.is_unique:
        raise ValueError("Coherent H and basis states must have matching unique labelled axes.")
    positions = original.get_indexer(retained[positions])
    if np.any(positions < 0):
        raise ValueError("Selected coherent H states are missing from the retained basis geometry.")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("Coherent H state order must follow the retained basis geometry.")
    grid_dims = operator.meta.grid_dims
    allowed = set(grid_dims) | ({"source"} if basis.source_labels is not None else set())
    if set(support.dims) not in (set(grid_dims), allowed) or support.dtype.kind != "b":
        raise ValueError("Inner support must be a Boolean native-grid mask, optionally indexed by source.")
    if is_dask_collection(support):
        raise ValueError("Materialize the inner-domain support mask explicitly before preparation.")
    for dim in support.dims:
        expected = (
            pd.Index(operator.source_labels, name="source") if dim == "source"
            else operator.basis_matrix.get_index(dim)
        )
        _require_same_axis(support, str(dim), expected, name="Inner support")
    if not bool(support.any()):
        raise ValueError("Inner support must contain at least one cell.")
    if isinstance(operator, BucketBasisOperator):
        if basis.source_labels is not None:
            raise ValueError("Coherent source states require a MultiSourceBucketBasisOperator.")
        flat = operator.basis_flat
        if is_dask_collection(flat):
            raise ValueError("Materialize retained bucket geometry explicitly before baseline preparation.")
        raw_labels = np.unique(flat.to_numpy()[np.isfinite(flat.to_numpy())])
        flat = flat.where(flat.isin(raw_labels[positions]) & support)
        metadata = operator.state_metadata
        selected_metadata = None if metadata is None else metadata.isel({state_dim: positions})
        selected = BucketBasisOperator(flat, meta=operator.meta, region_labels="basis_values")
        if selected_metadata is not None:
            selected_metadata = selected_metadata.assign_coords({state_dim: selected.basis_matrix[state_dim]})
            selected = BucketBasisOperator(
                flat, meta=operator.meta, region_labels="basis_values", state_metadata=selected_metadata,
            )
    elif isinstance(operator, MultiSourceBucketBasisOperator):
        chosen = original[positions]
        flats = {}
        for source in operator.source_labels:
            source_positions = np.flatnonzero(original.get_level_values(operator.source_dim) == source)
            chosen_positions = np.flatnonzero(np.isin(source_positions, positions))
            if not len(chosen_positions):
                continue
            flat = operator.basis_flat[source]
            if is_dask_collection(flat):
                raise ValueError("Materialize retained bucket geometry explicitly before baseline preparation.")
            raw_labels = np.unique(flat.to_numpy()[np.isfinite(flat.to_numpy())])
            mask = support.sel(source=source, drop=True) if "source" in support.dims else support
            flats[source] = flat.where(flat.isin(raw_labels[chosen_positions]) & mask)
        selected = MultiSourceBucketBasisOperator(
            flats, meta=operator.meta, source_dim=operator.source_dim,
            region_in_source_dim=operator.region_in_source_dim,
        )
        if selected.basis_matrix.sizes[state_dim] != len(chosen):
            raise ValueError("Inner support must retain native cells for every selected state.")
    else:
        raise ValueError("Supplied coherent backgrounds currently require retained bucket geometry.")
    if selected.basis_matrix.sizes[state_dim] != len(positions):
        raise ValueError("Inner support must retain native cells for every selected state.")
    flux = basis.flux
    if "source" in support.dims and "source" not in flux.dims:
        flux = flux.expand_dims(source=list(selected.source_labels))
    if isinstance(selected, MultiSourceBucketBasisOperator) and "source" in flux.dims:
        flux = flux.sel(source=list(selected.source_labels))
        if "source" in support.dims:
            support = support.sel(source=list(selected.source_labels))
    flux = remove_domain_overlap(flux, ~support).assign_coords(flux_support=support)
    return replace(basis, operator=selected, flux=flux)


def prepare_co2_supplied_baseline(
    prepared: Co2PreparedInputs,
    baseline: xr.DataArray | xr.Dataset,
    *,
    inner_states: xr.DataArray,
    inner_fixed_contribution: xr.DataArray,
    inner_support: xr.DataArray,
    baseline_site_metadata: xr.Dataset,
    species: str,
    provenance: str,
    constituents: tuple[str, ...],
    aggregation_error_policy: Literal["inherited"] = "inherited",
) -> Co2PreparedInputs:
    """Replace the exterior mean and boundaries of an existing coherent CO2 preparation.

    ``inner_states`` is an eager, labelled Boolean mask on the complete retained
    state axis. H columns are selected verbatim, and the retained arithmetic prior
    uses its marginal mean and covariance principal block. No conditioning at
    outer scaling one, or recomputation of coherent H from bucket geometry, occurs.
    ``inner_fixed_contribution`` supplies the complete inner-only affine term;
    a labelled zero series is required when it vanishes. Native reconstruction
    uses the explicit whole-cell ``inner_support``; it must preserve every selected
    state. State labels follow the retained operator after subset reconstruction;
    original selected labels are recorded in provenance.

    Only ``aggregation_error_policy='inherited'`` is supported: the original dense
    or low-rank payload is borrowed unchanged. Its unresolved exterior uncertainty
    may remain. This is not an inner-only recomputation of coherent residual error.
    Baseline uncertainty itself is omitted. All sampled boundary operands are
    removed. Baseline series and native flux stay lazy until the ordinary model or
    serialization boundary; concentration finiteness is checked by the model.
    """
    from dask import is_dask_collection
    from ..baseline import _prepare_baseline_series

    if not isinstance(prepared, Co2PreparedInputs):
        raise TypeError("prepare_co2_supplied_baseline requires Co2PreparedInputs.")
    if species.lower() != "co2":
        raise ValueError("The coherent supplied-baseline recipe requires species='co2'.")
    if aggregation_error_policy != "inherited":
        raise ValueError("Only aggregation_error_policy='inherited' is supported; no covariance is recomputed.")
    inputs = prepared.inv_inputs
    state_dim = str(inputs.H.dims[1])
    if inner_states.dims != (state_dim,) or inner_states.dtype.kind != "b":
        raise ValueError("inner_states must be a labelled Boolean mask on the complete retained-state axis.")
    _require_same_axis(inner_states, state_dim, inputs.H.get_index(state_dim), name="inner_states")
    if is_dask_collection(inner_states):
        raise ValueError("Materialize the inner_states selection mask explicitly before preparation.")
    positions = np.flatnonzero(inner_states.to_numpy())
    if not positions.size:
        raise ValueError("inner_states must select at least one retained state.")
    supplied, fixed, components = _prepare_baseline_series(
        prepared.rhime_inputs, baseline, inner_fixed_contribution=inner_fixed_contribution,
        baseline_site_metadata=baseline_site_metadata, species=species, provenance=provenance,
        constituents=constituents,
    )
    basis = _inner_basis(prepared, positions, inner_support)
    covariance_dim = f"{state_dim}_cov"
    selected = inputs.isel({state_dim: positions, covariance_dim: positions})
    # Boundary states and their coordinates no longer belong to the fitted model.
    selected = selected.drop_vars(("H_bc", "H_bc_anomaly", "H_bc_mass", "bc_reference"), errors="ignore")
    if "bc_region" in selected.dims:
        selected = selected.drop_dims("bc_region")
    selected = selected.drop_vars(("fixed_prior_contribution", "inner_fixed_contribution"), errors="ignore")
    new_index = basis.operator.basis_matrix.get_index(state_dim)
    # Regenerated source bucket labels are compact; both covariance axes follow H.
    for dim in (state_dim, covariance_dim):
        index = selected.indexes[dim]
        selected = selected.drop_vars(list(index.names) if isinstance(index, pd.MultiIndex) else dim)
    selected = selected.assign_coords(xr.Coordinates.from_pandas_multiindex(new_index, state_dim)
        if isinstance(new_index, pd.MultiIndex) else {state_dim: new_index})
    from openghg_inversions._labelled_matrices import renamed_column_coordinates
    selected = selected.assign_coords(renamed_column_coordinates(
        selected.H, row_dim=state_dim, column_dim=covariance_dim,
    ))
    original_index = inputs.H.get_index(state_dim)[positions]
    original_name = "original_region_in_source" if isinstance(original_index, pd.MultiIndex) else "original_state_label"
    original_values = (
        original_index.get_level_values(basis.operator.region_in_source_dim).to_numpy()
        if isinstance(original_index, pd.MultiIndex) else original_index.to_numpy()
    )
    selected = selected.assign_coords({original_name: (state_dim, original_values)})
    supplied = supplied.assign_attrs(
        aggregation_error_policy="inherited",
        aggregation_error_warning="Inherited aggregation error may retain unresolved exterior uncertainty.",
    )
    selected = selected.assign(
        supplied_baseline=supplied, fixed_prior_contribution=fixed.rename("fixed_prior_contribution"),
        inner_domain_support=(inner_support.rename(source="native_source")
            if "source" in inner_support.dims else inner_support), **components,
    )
    metadata = {
        **prepared.provenance,
        "supplied_baseline": {
            "provenance": provenance, "aggregation_error_policy": "inherited",
            "aggregation_error_warning": supplied.attrs["aggregation_error_warning"],
            "retained_prior_policy": "marginal",
            "original_inner_states": [str(value) for value in inputs.H.get_index(state_dim)[positions]],
        },
    }
    result = Co2PreparedInputs(
        PreparedInputs(selected, basis, prepared.site_metadata), prepared.aggregation_error_mode, metadata,
    )
    co2_supplied_baseline_input_names(result.inv_inputs)
    return result


__all__ = [
    "CO2_PREPARED_INPUTS_SCHEMA",
    "CO2_PREPARED_INPUTS_SCHEMA_VERSION",
    "Co2PreparedInputs",
    "prepare_co2_inputs",
    "prepare_co2_supplied_baseline",
    "co2_supplied_baseline_input_names",
    "requested_co2_facts",
]
