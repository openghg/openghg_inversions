"""Recipe-specific prepared inputs for coherent CO2 reductions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import pandas as pd
import numpy as np
import xarray as xr
from openghg.util import (  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
    cf_ureg,
)
from typing_extensions import Self

from openghg_inversions.array_ops import validate_covariance_coordinates
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import RhimePreparedInputs
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


CO2_PREPARED_INPUTS_SCHEMA = "openghg_inversions.co2_prepared_inputs"
CO2_PREPARED_INPUTS_SCHEMA_VERSION = 1
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
    try:
        encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("CO2 preparation provenance must be JSON-serializable.") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # Mapping always encodes as an object; keep the boundary explicit.
        raise ValueError("CO2 preparation provenance must encode as a JSON object.")
    return MappingProxyType(decoded)


def _same_index(left: pd.Index, right: pd.Index) -> bool:
    """Compare labels and MultiIndex level meaning, independent of dimension name."""
    if not left.equals(right):
        return False
    if isinstance(left, pd.MultiIndex) or isinstance(right, pd.MultiIndex):
        return (
            isinstance(left, pd.MultiIndex) and isinstance(right, pd.MultiIndex) and left.names == right.names
        )
    return True


def _require_axis(array: xr.DataArray, dim: str, *, name: str) -> pd.Index:
    if dim not in array.dims or dim not in array.indexes:
        raise ValueError(f"{name} requires a labelled {dim!r} dimension.")
    index = array.indexes[dim]
    if not index.is_unique:
        raise ValueError(f"{name} {dim!r} labels must be unique.")
    return index


def _require_same_axis(
    array: xr.DataArray,
    dim: str,
    expected: pd.Index,
    *,
    name: str,
) -> None:
    if not _same_index(_require_axis(array, dim, name=name), expected):
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
    """Remove aggregation-owned dimensions without dropping other consumers."""
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
        index_coords = {dim}
        index = inputs.indexes.get(dim)
        if isinstance(index, pd.MultiIndex):
            index_coords.update(name for name in index.names if name is not None)
        extra_coords = [
            name
            for name, value in inputs.coords.items()
            if dim in value.dims and name not in index_coords
        ]
        if consumers or extra_coords:
            raise ValueError(
                f"Aggregation representation dimension {dim!r} is also used by "
                f"non-aggregation variable(s): {consumers + extra_coords!r}."
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
        scale = float(actual_quantity.to(expected_quantity.units).magnitude / expected_quantity.magnitude)
    except Exception as exc:
        raise ValueError(f"{name} units {actual!r} are incompatible with {expected!r}.") from exc
    if not np.isclose(scale, 1.0, rtol=1e-12, atol=0.0):
        raise ValueError(f"{name} units {actual!r} do not have the same numeric scale as {expected!r}.")


def _validate_co2_dataset(inputs: xr.Dataset) -> None:
    """Validate the scientific layout owned by the CO2 artifact."""
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

    Attributes:
        rhime_inputs: Canonical RHIME inputs composed into this recipe-specific
            boundary and passed directly to shared RHIME consumers.
        aggregation_error_mode: Concrete dense or low-rank representation
            stored in :attr:`inv_inputs` and consumed by both CO2 runners.
        provenance: Immutable JSON-safe preparation record. Low-rank artifacts
            include the source-covariance identity and approximation
            diagnostics.
    """

    rhime_inputs: RhimePreparedInputs
    aggregation_error_mode: Co2AggregationErrorMode
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aggregation_error_mode not in ("dense", "low_rank"):
            raise ValueError("CO2 aggregation_error_mode must be 'dense' or 'low_rank'.")
        inputs = self.rhime_inputs.inv_inputs
        _validate_co2_dataset(inputs)
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

    def validated(self) -> Self:
        """Return this structurally validated immutable wrapper."""
        return self

    def to_datatree(self) -> xr.DataTree:
        """Validate and materialize aggregation error into the versioned schema."""
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
        serializable = RhimePreparedInputs(
            inv_inputs=serializable_inputs,
            basis_functions=self.basis_functions,
            site_metadata=self.site_metadata,
        )
        tree = xr.DataTree.from_dict({"rhime_inputs": serializable.to_datatree()})
        tree.attrs = {
            "schema": CO2_PREPARED_INPUTS_SCHEMA,
            "schema_version": CO2_PREPARED_INPUTS_SCHEMA_VERSION,
            "aggregation_error_mode": self.aggregation_error_mode,
            "provenance_json": json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False),
        }
        return tree

    @classmethod
    def from_datatree(cls, tree: xr.DataTree) -> Self:
        """Restore CO2 inputs from the dedicated version-1 schema."""
        if tree.attrs.get("schema") != CO2_PREPARED_INPUTS_SCHEMA:
            raise ValueError(
                f"Expected Co2PreparedInputs schema {CO2_PREPARED_INPUTS_SCHEMA!r}, "
                f"got {tree.attrs.get('schema')!r}."
            )
        version = tree.attrs.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, Integral) or version != 1:
            raise ValueError(
                f"Expected Co2PreparedInputs schema_version "
                f"{CO2_PREPARED_INPUTS_SCHEMA_VERSION}, got {version!r}."
            )
        if "rhime_inputs" not in tree.children:
            raise KeyError("Missing required Co2PreparedInputs node 'rhime_inputs'.")
        mode = tree.attrs.get("aggregation_error_mode")
        if mode not in ("dense", "low_rank"):
            raise ValueError(f"Invalid serialized CO2 aggregation_error_mode {mode!r}.")
        try:
            provenance = json.loads(str(tree.attrs.get("provenance_json", "{}")))
        except json.JSONDecodeError as exc:
            raise ValueError("Serialized CO2 provenance is not valid JSON.") from exc
        prepared = cls(
            RhimePreparedInputs.from_datatree(cast(xr.DataTree, tree["rhime_inputs"])),
            aggregation_error_mode=cast(Co2AggregationErrorMode, mode),
            provenance=provenance,
        )
        resolve_aggregation_error(prepared.inv_inputs, prepared.aggregation_error_mode)
        return prepared

    def save(
        self,
        output_file: str | Path,
        output_format: Literal["netcdf", "zarr"] | None = None,
    ) -> None:
        """Save the prepared CO2 artifact to NetCDF or Zarr."""
        save_datatree(self.to_datatree(), output_file, output_format)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """Load a fully materialized prepared CO2 artifact."""
        return cls.from_datatree(open_datatree_loaded(file_path))


def prepare_co2_inputs(
    canonical_inputs: RhimePreparedInputs,
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
    larger than the observation count use full rank.

    Inputs are borrowed and are not mutated. The returned artifact retains the
    canonical observations, observation error, optional boundary data, state
    activity, basis functions, and site metadata.

    Args:
        canonical_inputs: Canonical RHIME observations and metadata whose
            observation and retained-state labels match ``reduction``.
        reduction: One coherent Gaussian reduction containing all linked
            retained-prior and observation products.
        aggregation_error_rank: Positive LRPD rank, defaulting to 512. Values
            larger than the observation count are capped at that count. Pass
            ``None`` to store the exact dense covariance.
        provenance: Optional JSON-serializable project or preparation
            provenance. The reduction strategy and LRPD diagnostics are added
            by this boundary.

    Returns:
        A validated, serializable artifact accepted by both CO2 runners.

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
        if array.attrs.get("units") != observation_units:
            raise ValueError(f"Coherent {name} units must match the canonical observations.")
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
    if approximation is not None:
        metadata["aggregation_error"] = dict(approximation.diagnostics)
        metadata["aggregation_error"]["source_covariance_sha256"] = (
            approximation.source_covariance_sha256
        )
    return Co2PreparedInputs(
        RhimePreparedInputs(
            inv_inputs=mapped,
            basis_functions=canonical.basis_functions,
            site_metadata=canonical.site_metadata,
        ),
        aggregation_error_mode=mode,
        provenance=metadata,
    )


__all__ = [
    "CO2_PREPARED_INPUTS_SCHEMA",
    "CO2_PREPARED_INPUTS_SCHEMA_VERSION",
    "Co2PreparedInputs",
    "prepare_co2_inputs",
]
