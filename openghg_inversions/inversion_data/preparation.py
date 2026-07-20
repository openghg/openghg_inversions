"""Prepare inversion data for modern RHIME and legacy fixedbasis runners.

``prepare_rhime_inputs`` returns backend-neutral observations, sensitivities,
basis metadata, and site metadata; component-specific model arrays are
intentionally absent. ``prepare_fixedbasis_inversion_data`` is a compatibility
adapter that retains the input variables required by ``fixedbasisMCMC``,
including its sigma-period index.

Preparation can read OpenGHG object stores or local merged-data artifacts,
write merged-data and basis artifacts, emit warnings and progress messages,
and record timing information. Neither public entry point constructs a PyMC
model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import basis_functions_wrapper, make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.filters import filtering
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data.get_data import convert_to_list, data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.serialization import (
    open_datatree_loaded,
    reset_serialisation_multiindexes,
    restore_serialisation_multiindexes,
    save_datatree,
)

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float
RHIME_PREPARED_INPUTS_SCHEMA = "openghg_inversions.rhime_prepared_inputs"
RHIME_PREPARED_INPUTS_SCHEMA_VERSION = 1
_RHIME_PREPARED_INPUTS_METADATA_ATTR = "metadata"
_RHIME_PREPARED_INPUTS_BASIS_SCHEMA = "openghg_inversions.flux_weighted_basis"
_RHIME_PREPARED_INPUTS_BASIS_SCHEMA_VERSION = 1
_RHIME_PREPARED_INPUTS_METADATA_FIELDS = frozenset(
    {
        "sites",
        "averaging_period",
        "basis_artifact_source",
        "basis_artifact_path",
        "site_lats",
        "site_lons",
    }
)


@dataclass
class FixedBasisPreparedData:
    """Data prepared for the legacy fixedbasis runner.

    Args:
        fp_all: Raw merged-data container returned by data gathering or reload.
        fp_data: Forward-model data after basis functions and filters.
        inv_inputs: Canonical inversion inputs consumed by model builders.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_objects: Basis objects returned by ``basis_functions_wrapper``.
        basis_artifact_source: Source of the flux basis artifact.
        basis_artifact_path: Path to the flux basis artifact, when loaded or saved.
    """

    fp_all: dict
    fp_data: dict | None = None
    inv_inputs: xr.Dataset | None = None
    sites: list[str] = field(default_factory=list)
    averaging_period: list[str | None] = field(default_factory=list)
    basis_objects: dict[str, BasisFunctions] = field(default_factory=dict)
    basis_artifact_source: str = "generated"
    basis_artifact_path: str | None = None


@dataclass(frozen=True)
class RhimePreparedInputs:
    """Modern RHIME preparation and durable serialization contract.

    Serialized ``inv_inputs`` must restore ``nmeasure`` as a pandas MultiIndex
    with levels ``(site, time)``. ``averaging_period`` and any release
    coordinate tuples are site-aligned. Latitude and longitude must either
    both be present or both be absent; ``NaN`` represents a missing coordinate,
    while infinite coordinates are invalid.

    Args:
        inv_inputs: Canonical inversion inputs consumed by RHIME model
            builders.
        basis_functions: Retained flux basis object used to derive
            output-boundary basis and flux arrays.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_artifact_source: Description of whether the basis was generated
            or loaded from an artifact.
        basis_artifact_path: Path to the basis artifact, when loaded or saved.
        site_lats: Release latitudes aligned to ``sites``, when available.
        site_lons: Release longitudes aligned to ``sites``, when available.
    """

    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    basis_artifact_source: str
    basis_artifact_path: str | None = None
    site_lats: tuple[float, ...] | None = None
    site_lons: tuple[float, ...] | None = None

    def to_datatree(self) -> xr.DataTree:
        """Convert prepared RHIME inputs to the versioned DataTree schema.

        The basis object is embedded in the artifact. ``basis_artifact_path``
        is retained only as provenance and is never read while serializing or
        reconstructing the prepared inputs.

        Returns:
            DataTree containing ``inv_inputs`` and ``basis_functions`` child
            nodes plus strictly encoded site-aligned metadata.

        Raises:
            ValueError: If a metadata field has the wrong type, site-aligned
                fields have inconsistent lengths, coordinates are infinite,
                metadata cannot be encoded as strict JSON, or an inversion-input
                MultiIndex contains unnamed levels.
        """
        metadata = _rhime_prepared_inputs_metadata_for_serialisation(self)
        dt = xr.DataTree.from_dict(
            {
                "inv_inputs": xr.DataTree(reset_serialisation_multiindexes(self.inv_inputs)),
                "basis_functions": self.basis_functions.to_datatree(),
            }
        )
        dt.attrs = {
            "schema": RHIME_PREPARED_INPUTS_SCHEMA,
            "schema_version": RHIME_PREPARED_INPUTS_SCHEMA_VERSION,
            _RHIME_PREPARED_INPUTS_METADATA_ATTR: json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        return dt

    @classmethod
    def from_datatree(cls: type[Self], dt: xr.DataTree) -> Self:
        """Construct prepared RHIME inputs from a version-1 DataTree.

        Args:
            dt: DataTree using the ``openghg_inversions.rhime_prepared_inputs``
                schema.

        Returns:
            Reconstructed prepared inputs with the canonical inversion-input
            MultiIndexes and embedded basis object restored, including retained
            multisource basis ordering.

        Raises:
            ValueError: If the prepared or embedded BasisFunctions schema,
                schema version, metadata, or serialized MultiIndex description
                is missing or malformed, or ``nmeasure`` is not canonical
                ``(site, time)``.
            KeyError: If a required child node is missing.
        """
        schema = dt.attrs.get("schema")
        if schema != RHIME_PREPARED_INPUTS_SCHEMA:
            raise ValueError(
                f"Expected RhimePreparedInputs schema {RHIME_PREPARED_INPUTS_SCHEMA!r}, got {schema!r}."
            )

        version = dt.attrs.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, Integral) or version != 1:
            raise ValueError(
                f"Expected RhimePreparedInputs schema_version "
                f"{RHIME_PREPARED_INPUTS_SCHEMA_VERSION}, got {version!r}."
            )

        missing_nodes = [name for name in ("inv_inputs", "basis_functions") if name not in dt.children]
        if missing_nodes:
            raise KeyError(f"Missing required RhimePreparedInputs node(s): {missing_nodes!r}.")

        metadata = _load_rhime_prepared_inputs_metadata(dt.attrs)
        inv_inputs = _restore_canonical_rhime_inv_inputs(cast(xr.DataTree, dt["inv_inputs"]).to_dataset())
        basis_functions_dt = cast(xr.DataTree, dt["basis_functions"])
        _validate_rhime_prepared_inputs_basis_schema(basis_functions_dt)
        basis_functions = BasisFunctions.from_datatree(basis_functions_dt)
        return cls(
            inv_inputs=inv_inputs,
            basis_functions=basis_functions,
            sites=metadata["sites"],
            averaging_period=metadata["averaging_period"],
            basis_artifact_source=metadata["basis_artifact_source"],
            basis_artifact_path=metadata["basis_artifact_path"],
            site_lats=metadata["site_lats"],
            site_lons=metadata["site_lons"],
        )

    def save(
        self,
        output_file: str | Path,
        output_format: Literal["netcdf", "zarr"] | None = None,
    ) -> None:
        """Save prepared RHIME inputs to NetCDF or Zarr.

        Args:
            output_file: Destination artifact path. Saving writes and may
                overwrite this artifact.
            output_format: Storage format. When omitted, infer it from a
                ``.nc`` or ``.zarr`` suffix. An explicit format adds or
                replaces the corresponding suffix.

        Raises:
            ValueError: If metadata is invalid or the output format cannot be
                inferred.
        """
        save_datatree(self.to_datatree(), output_file, output_format)

    @classmethod
    def load(cls: type[Self], file_path: str | Path) -> Self:
        """Load prepared RHIME inputs from a NetCDF or Zarr artifact.

        Args:
            file_path: Prepared-input artifact previously written by ``save``.

        Returns:
            Fully loaded prepared RHIME inputs with no open file handles.

        Raises:
            OSError: If the artifact cannot be opened.
            RuntimeError: If all available storage backends fail at runtime.
            ValueError: If the artifact schema or metadata is invalid.
            KeyError: If a required child node is missing.
        """
        return cls.from_datatree(open_datatree_loaded(file_path))


def _restore_canonical_rhime_inv_inputs(ds: xr.Dataset) -> xr.Dataset:
    """Restore and validate the canonical site/time measurement index.

    Args:
        ds: Serialized canonical inversion inputs with MultiIndex restoration
            metadata.

    Returns:
        Inversion inputs with ``nmeasure=(site, time)`` restored.

    Raises:
        ValueError: If restoration metadata is missing, empty, malformed, or
            inapplicable, or the restored ``nmeasure`` index is not a
            ``(site, time)`` pandas MultiIndex.
    """
    restored = restore_serialisation_multiindexes(ds, strict=True)
    nmeasure_index = restored.indexes.get("nmeasure")
    if not isinstance(nmeasure_index, pd.MultiIndex):
        raise ValueError(
            "RhimePreparedInputs inv_inputs must restore a 'nmeasure' MultiIndex with levels "
            "('site', 'time')."
        )
    if tuple(nmeasure_index.names) != ("site", "time"):
        raise ValueError(
            "RhimePreparedInputs inv_inputs 'nmeasure' MultiIndex must have levels "
            f"('site', 'time'), got {tuple(nmeasure_index.names)!r}."
        )
    return restored


def _validate_rhime_prepared_inputs_basis_schema(dt: xr.DataTree) -> None:
    """Validate the exact embedded BasisFunctions schema used by version 1.

    Args:
        dt: Embedded BasisFunctions DataTree.

    Raises:
        ValueError: If the schema or integer schema version is missing or does
            not exactly match the prepared-input v1 contract.
    """
    schema = dt.attrs.get("schema")
    if schema != _RHIME_PREPARED_INPUTS_BASIS_SCHEMA:
        raise ValueError(
            "Expected embedded BasisFunctions schema "
            f"{_RHIME_PREPARED_INPUTS_BASIS_SCHEMA!r}, got {schema!r}."
        )

    version = dt.attrs.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, Integral)
        or version != _RHIME_PREPARED_INPUTS_BASIS_SCHEMA_VERSION
    ):
        raise ValueError(
            "Expected embedded BasisFunctions schema_version "
            f"{_RHIME_PREPARED_INPUTS_BASIS_SCHEMA_VERSION}, got {version!r}."
        )


def _raise_nonfinite_json_constant(value: str) -> NoReturn:
    """Reject non-standard JSON constants such as NaN and Infinity.

    Args:
        value: Non-standard constant encountered by ``json.loads``.

    Raises:
        ValueError: Always, because prepared metadata uses strict JSON.
    """
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _validate_string_tuple(value: object, *, field_name: str, allow_none: bool) -> tuple[str | None, ...]:
    """Validate and normalize a tuple of string-like metadata values.

    Args:
        value: Candidate tuple.
        field_name: Metadata field name used in validation errors.
        allow_none: Whether individual tuple entries may be ``None``.

    Returns:
        The validated tuple.

    Raises:
        ValueError: If ``value`` is not a tuple or contains an invalid entry.
    """
    if not isinstance(value, tuple):
        raise ValueError(f"RhimePreparedInputs {field_name!r} must be a tuple.")
    if not all(isinstance(item, str) or (allow_none and item is None) for item in value):
        expected = "strings or None" if allow_none else "strings"
        raise ValueError(f"RhimePreparedInputs {field_name!r} must contain only {expected}.")
    return value


def _validate_coordinate_tuple(value: object, *, field_name: str) -> tuple[float, ...] | None:
    """Validate and normalize optional site coordinates, allowing missing NaNs.

    Args:
        value: Optional tuple of numeric site coordinates.
        field_name: Metadata field name used in validation errors.

    Returns:
        A tuple of floats, preserving missing values as NaN, or ``None``.

    Raises:
        ValueError: If the container or entries have invalid types, or a
            coordinate is infinite.
    """
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise ValueError(f"RhimePreparedInputs {field_name!r} must be a tuple or None.")
    coordinates: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"RhimePreparedInputs {field_name!r} must contain only numbers.")
        coordinate = float(item)
        if np.isinf(coordinate):
            raise ValueError(f"RhimePreparedInputs {field_name!r} must not contain infinity.")
        coordinates.append(coordinate)
    return tuple(coordinates)


def _validate_rhime_prepared_inputs_metadata(
    *,
    sites: object,
    averaging_period: object,
    basis_artifact_source: object,
    basis_artifact_path: object,
    site_lats: object,
    site_lons: object,
) -> dict[str, Any]:
    """Validate prepared-input metadata types and site alignment.

    Args:
        sites: Ordered retained site names.
        averaging_period: Site-aligned strings or missing values.
        basis_artifact_source: Basis provenance source label.
        basis_artifact_path: Optional provenance-only artifact path.
        site_lats: Optional site-aligned release latitudes.
        site_lons: Optional site-aligned release longitudes.

    Returns:
        Validated and normalized metadata values.

    Raises:
        ValueError: If field types are invalid, site-aligned lengths differ,
            coordinates are infinite, or only one coordinate tuple is present.
    """
    validated_sites = _validate_string_tuple(sites, field_name="sites", allow_none=False)
    validated_periods = _validate_string_tuple(
        averaging_period,
        field_name="averaging_period",
        allow_none=True,
    )
    if not isinstance(basis_artifact_source, str):
        raise ValueError("RhimePreparedInputs 'basis_artifact_source' must be a string.")
    if basis_artifact_path is not None and not isinstance(basis_artifact_path, str):
        raise ValueError("RhimePreparedInputs 'basis_artifact_path' must be a string or None.")

    validated_lats = _validate_coordinate_tuple(site_lats, field_name="site_lats")
    validated_lons = _validate_coordinate_tuple(site_lons, field_name="site_lons")
    if (validated_lats is None) != (validated_lons is None):
        raise ValueError("RhimePreparedInputs 'site_lats' and 'site_lons' must both be set or both be None.")
    aligned_fields = {
        "averaging_period": validated_periods,
        "site_lats": validated_lats,
        "site_lons": validated_lons,
    }
    for field_name, value in aligned_fields.items():
        if value is not None and len(value) != len(validated_sites):
            raise ValueError(
                f"RhimePreparedInputs {field_name!r} has length {len(value)}, "
                f"but 'sites' has length {len(validated_sites)}."
            )

    return {
        "sites": validated_sites,
        "averaging_period": validated_periods,
        "basis_artifact_source": basis_artifact_source,
        "basis_artifact_path": basis_artifact_path,
        "site_lats": validated_lats,
        "site_lons": validated_lons,
    }


def _rhime_prepared_inputs_metadata_for_serialisation(
    prepared: RhimePreparedInputs,
) -> dict[str, Any]:
    """Return validated JSON-compatible metadata for prepared inputs.

    Missing coordinate NaNs are encoded as JSON-compatible ``None`` values;
    infinities remain invalid.

    Args:
        prepared: Prepared inputs whose metadata should be encoded.

    Returns:
        Exact schema-v1 metadata with tuples converted to lists.

    Raises:
        ValueError: If metadata types, coordinate values, or site alignment are
            invalid.
    """
    metadata = _validate_rhime_prepared_inputs_metadata(
        sites=prepared.sites,
        averaging_period=prepared.averaging_period,
        basis_artifact_source=prepared.basis_artifact_source,
        basis_artifact_path=prepared.basis_artifact_path,
        site_lats=prepared.site_lats,
        site_lons=prepared.site_lons,
    )
    result = {key: list(value) if isinstance(value, tuple) else value for key, value in metadata.items()}
    for field_name in ("site_lats", "site_lons"):
        coordinates = metadata[field_name]
        if coordinates is not None:
            result[field_name] = [None if np.isnan(value) else value for value in coordinates]
    return result


def _load_rhime_prepared_inputs_metadata(attrs: Mapping[Any, Any]) -> dict[str, Any]:
    """Decode and strictly validate prepared-input metadata from root attrs.

    JSON null coordinate entries are restored as float NaNs. The metadata must
    contain exactly the fields defined by the prepared-input v1 schema.

    Args:
        attrs: Root DataTree attributes containing the JSON metadata string.

    Returns:
        Validated metadata with tuple-shaped site-aligned values.

    Raises:
        ValueError: If metadata is missing, is not strict JSON, has missing or
            unexpected fields, contains invalid values, or is not site-aligned.
    """
    raw_metadata = attrs.get(_RHIME_PREPARED_INPUTS_METADATA_ATTR)
    if not isinstance(raw_metadata, str):
        raise ValueError("RhimePreparedInputs metadata must be present as a JSON string.")
    try:
        payload = json.loads(raw_metadata, parse_constant=_raise_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("RhimePreparedInputs metadata is not valid strict JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("RhimePreparedInputs metadata must decode to a JSON object.")

    payload_fields = frozenset(payload)
    if payload_fields != _RHIME_PREPARED_INPUTS_METADATA_FIELDS:
        missing = sorted(_RHIME_PREPARED_INPUTS_METADATA_FIELDS - payload_fields)
        unexpected = sorted(payload_fields - _RHIME_PREPARED_INPUTS_METADATA_FIELDS)
        raise ValueError(
            "RhimePreparedInputs metadata fields do not match schema version 1: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )

    for field_name in ("sites", "averaging_period"):
        if not isinstance(payload[field_name], list):
            raise ValueError(f"RhimePreparedInputs metadata {field_name!r} must be a JSON array.")
    for field_name in ("site_lats", "site_lons"):
        if payload[field_name] is not None and not isinstance(payload[field_name], list):
            raise ValueError(f"RhimePreparedInputs metadata {field_name!r} must be a JSON array or null.")

    decoded_coordinates = {
        field_name: (
            None
            if payload[field_name] is None
            else tuple(np.nan if value is None else value for value in payload[field_name])
        )
        for field_name in ("site_lats", "site_lons")
    }

    return _validate_rhime_prepared_inputs_metadata(
        sites=tuple(payload["sites"]),
        averaging_period=tuple(payload["averaging_period"]),
        basis_artifact_source=payload["basis_artifact_source"],
        basis_artifact_path=payload["basis_artifact_path"],
        site_lats=decoded_coordinates["site_lats"],
        site_lons=decoded_coordinates["site_lons"],
    )


@dataclass
class _MergedInversionData:
    """Merged data and site-aligned metadata shared by preparation paths."""

    fp_all: dict
    sites: list[str]
    averaging_period: list[str | None]


def _filter_site_aligned_value(value: object, keep_indices: list[int]) -> object:
    """Filter values that are aligned to the sites list."""
    if value is None or isinstance(value, str | bytes):
        return value
    if not isinstance(value, Sequence):
        return value
    return [item for index, item in enumerate(value) if index in keep_indices]


def _first_scalar_data_value(ds: xr.Dataset, names: tuple[str, ...]) -> float:
    """Return the first scalar value found in a dataset variable or coordinate."""
    for name in names:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values).reshape(-1)
        if values.size:
            return float(values[0])
    return np.nan


def _site_release_coordinates(
    fp_data: dict[str, xr.Dataset],
    sites: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return release lat/lon values aligned to retained sites."""
    lats: list[float] = []
    lons: list[float] = []
    for site in sites:
        site_data = fp_data.get(site)
        if site_data is None:
            lats.append(np.nan)
            lons.append(np.nan)
            continue
        lats.append(_first_scalar_data_value(site_data, ("release_lat", "sitelats")))
        lons.append(_first_scalar_data_value(site_data, ("release_lon", "sitelons")))
    return tuple(lats), tuple(lons)


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    sites: list[str],
    inlet: Any,
    fp_height: Any,
    instrument: Any,
    max_level: int | None,
    averaging_period: list[str | None],
) -> tuple[list[str], Any, Any, Any, int | None, list[str | None]]:
    """Align site-level options when loaded merged data lacks requested sites."""
    sites_merged = [site for site in fp_all if not site.startswith(".")]
    if all(site in sites_merged for site in sites):
        return sites, inlet, fp_height, instrument, max_level, list(averaging_period)

    keep_indices = [index for index, site in enumerate(sites) if site in sites_merged]
    dropped_sites = [site for site in sites if site not in sites_merged]
    if not keep_indices:
        raise ValueError(
            "Loaded merged data does not include any requested sites. "
            f"Requested sites: {sites}. Available merged-data sites: {sites_merged}."
        )

    sites = [site for index, site in enumerate(sites) if index in keep_indices]
    averaging_period = [period for index, period in enumerate(averaging_period) if index in keep_indices]

    print(f"\nDropping {dropped_sites} sites as they are not included in the merged data object.\n")
    return (
        sites,
        _filter_site_aligned_value(inlet, keep_indices),
        _filter_site_aligned_value(fp_height, keep_indices),
        _filter_site_aligned_value(instrument, keep_indices),
        max_level,
        averaging_period,
    )


def _select_fp_all_sites(fp_all: dict, sites: list[str]) -> dict:
    """Keep only requested site entries and metadata from a merged-data object."""
    site_names = set(sites)
    return {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}


def _normalise_averaging_period(
    averaging_period: list[str | None] | str | None, *, nsites: int
) -> list[str | None]:
    """Normalize and validate site-aligned averaging periods."""
    normalized = convert_to_list(averaging_period, length=nsites, name="averaging_period")
    invalid_periods = [period for period in normalized if period is not None and not isinstance(period, str)]
    if invalid_periods:
        raise ValueError(
            f"`averaging_period` entries must be strings or None. Invalid value(s): {invalid_periods!r}."
        )
    return normalized


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: list[str],
    start_date: str,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    calculate_min_error: Literal["percentile", "residual"] | None,
    min_error_options: dict | None,
) -> xr.Dataset:
    """Create backend-neutral inversion inputs with min-error compatibility.

    Args:
        fp_data: Filtered per-site observations and sensitivity data.
        sites: Retained sites in observation order.
        start_date: Anchor for fixed-duration boundary-condition periods.
        bc_freq: Optional boundary-condition period frequency.
        min_error: Minimum-error value or calculation method.
        calculate_min_error: Deprecated minimum-error calculation argument.
        min_error_options: Options for calculated minimum error.

    Returns:
        Canonical observation-aligned inputs without component-specific model
        data.

    Warns:
        FutureWarning: If ``calculate_min_error`` is supplied.
    """
    if calculate_min_error is not None:
        warnings.warn(
            "`calculate_min_error` is deprecated. Please use `min_error` to pass the calculation method instead.",
            FutureWarning,
            stacklevel=3,
        )
        min_error = calculate_min_error

    if min_error is None:
        min_error = 0.0
    elif isinstance(min_error, int) and not isinstance(min_error, bool):
        min_error = float(min_error)
    elif isinstance(min_error, dict):
        missing_sites = [site for site in sites if site not in min_error]
        if missing_sites:
            raise ValueError(
                "`min_error` dictionaries must include a value for every retained site. "
                f"Missing site(s): {missing_sites!r}."
            )

    min_error_options = min_error_options or {}
    return make_inv_inputs(
        fp_data,
        sites=sites,
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_per_site=min_error_options.get("by_site", False),
        start_date=start_date,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values")
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values")


def _prepare_merged_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> _MergedInversionData:
    """Gather or reload merged data and align site metadata.

    ``flux_sources`` contains modern OpenGHG flux ``source`` values. This
    helper passes them to lower-level data loading through the legacy
    ``emissions_name`` argument.
    """
    if use_tracer:
        raise ValueError("Tracer inversions are not supported by this preparation path.")
    if not sites:
        raise ValueError("At least one site must be specified for inversion data preparation.")

    averaging_period = _normalise_averaging_period(averaging_period, nsites=len(sites))
    rerun_merge = True
    fp_all: dict | None = None
    if reload_merged_data and merged_data_dir is not None:
        try:
            fp_all = load_merged_data(merged_data_dir, species, start_date, output_name, merged_data_name)
        except ValueError as exc:
            print(f"{exc}, re-running data merge.")
        else:
            print("Successfully read in merged data.\n")
            fp_all[".split_by_sectors"] = split_by_sectors
            rerun_merge = False
            sites, inlet, fp_height, instrument, max_level, averaging_period = (
                _drop_sites_missing_from_loaded_data(
                    fp_all=fp_all,
                    sites=sites,
                    inlet=inlet,
                    fp_height=fp_height,
                    instrument=instrument,
                    max_level=max_level,
                    averaging_period=averaging_period,
                )
            )
            fp_all = _select_fp_all_sites(fp_all, sites)
    elif reload_merged_data:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    if rerun_merge:
        (
            fp_all,
            sites,
            inlet,
            fp_height,
            instrument,
            averaging_period,
        ) = data_processing_surface_notracer(
            species=species,
            sites=sites,
            domain=domain,
            averaging_period=averaging_period,
            start_date=start_date,
            end_date=end_date,
            obs_data_level=obs_data_level,
            platform=platform,
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
            emissions_name=flux_sources,
            inlet=inlet,
            instrument=instrument,
            max_level=max_level,
            calibration_scale=calibration_scale,
            use_bc=use_bc,
            bc_input=bc_input,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            split_by_sectors=split_by_sectors,
            averagingerror=averaging_error,
            save_merged_data=save_merged_data,
            merged_data_name=merged_data_name,
            merged_data_dir=merged_data_dir,
            output_name=output_name,
            flux_non_finite_check=flux_non_finite_check,
        )

    if fp_all is None:
        raise RuntimeError("Data preparation did not create or load merged data.")
    if not sites:
        raise ValueError("No sites remain after data gathering.")

    flux_entries = fp_all.get(".flux")
    if isinstance(flux_entries, Mapping):
        for source, flux_data in flux_entries.items():
            data = getattr(flux_data, "data", None)
            if isinstance(data, xr.Dataset) and "flux" in data:
                data["flux"] = sanitize_flux_nonfinite(
                    data["flux"],
                    context="merged inversion data preparation",
                    source=str(source),
                    check=flux_non_finite_check,
                    warn=flux_non_finite_check == "count",
                )

    return _MergedInversionData(
        fp_all=fp_all,
        sites=sites,
        averaging_period=cast(list[str | None], averaging_period),
    )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    sites: list[str],
    averaging_period: list[str | None],
    filters: Any,
) -> tuple[dict, list[str], list[str | None]]:
    """Apply filters and keep site-aligned metadata in sync."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in sites:
        if fp_data[site].time.values.shape[0] == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        keep_indices = [index for index, site in enumerate(sites) if site not in dropped_sites]
        if not keep_indices:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        sites = [site for index, site in enumerate(sites) if index in keep_indices]
        averaging_period = [period for index, period in enumerate(averaging_period) if index in keep_indices]
        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data, sites, averaging_period


def _set_domain_attrs(fp_data: dict, sites: list[str], domain: str) -> None:
    """Attach the legacy domain attribute expected by downstream code."""
    for site in sites:
        fp_data[site].attrs["Domain"] = domain


def _bc_basis_directory_arg(bc_basis_directory: str | Path | None) -> str | None:
    """Normalize BC basis directory arguments for legacy helpers."""
    return str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory


def _validate_multisector_sensitivity_sources(
    sensitivity: xr.DataArray,
    *,
    site: str,
    flux_sources: list[str],
) -> xr.DataArray:
    """Validate and order one site's source-resolved sensitivity."""
    if "source" not in sensitivity.coords:
        raise ValueError(
            f"Site {site!r} sensitivity is missing the 'source' coordinate required for "
            f"flux source(s) {flux_sources!r}."
        )

    source_labels = [str(source) for source in sensitivity.coords["source"].values]
    available_sources = list(dict.fromkeys(source_labels))
    duplicate_sources = (
        [source for source in available_sources if source_labels.count(source) > 1]
        if "source" in sensitivity.dims
        else []
    )
    missing_sources = [source for source in flux_sources if source not in available_sources]
    extra_sources = [source for source in available_sources if source not in flux_sources]
    if duplicate_sources or missing_sources or extra_sources:
        raise ValueError(
            f"Site {site!r} sensitivity source layout does not match requested flux sources; "
            f"missing source(s): {missing_sources!r}; extra source(s): {extra_sources!r}; "
            f"duplicate source(s): {duplicate_sources!r}."
        )
    if "source" in sensitivity.dims:
        return sensitivity.sel(source=flux_sources)
    return sensitivity


def _rhime_site_data_from_basis_functions(
    *,
    merged: _MergedInversionData,
    basis_functions: BasisFunctions,
    domain: str,
    split_by_sectors: bool,
    flux_sources: list[str],
    use_bc: bool,
    bc_basis_case: str,
    bc_basis_directory: str | None,
) -> dict:
    """Apply retained basis functions to one prepared merged-data stage."""
    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_x_flux_name = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"

    for site in merged.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            continue
        fp_x_flux = fp_data[site][fp_x_flux_name]
        timing_start = timer_start()
        sensitivity = basis_functions.sensitivity(fp_x_flux)
        state_dims = [dim for dim in sensitivity.dims if dim not in fp_x_flux.dims]
        if "region" in sensitivity.dims:
            state_dim = "region"
        elif len(state_dims) == 1:
            state_dim = cast(str, state_dims[0])
        else:
            raise ValueError(
                "Could not identify the RHIME sensitivity state dimension from "
                f"sensitivity dims {sensitivity.dims!r} and fp_x_flux dims {fp_x_flux.dims!r}."
            )
        if split_by_sectors:
            sensitivity = _validate_multisector_sensitivity_sources(
                sensitivity,
                site=site,
                flux_sources=flux_sources,
            )
        if "source" in sensitivity.coords and "source" not in sensitivity.dims:
            fp_data[site] = fp_data[site].drop_vars(fp_x_flux_name)
            orphan_dims = [
                dim
                for dim in fp_x_flux.dims
                if dim in fp_data[site].dims
                and all(dim not in variable.dims for variable in fp_data[site].data_vars.values())
            ]
            if orphan_dims:
                fp_data[site] = fp_data[site].drop_dims(orphan_dims)
        fp_data[site]["H"] = sensitivity
        log_timing(
            "rhime.prepare_inputs.footprint_sensitivity",
            timer_seconds(timing_start),
            site=site,
            nmeasure=fp_data[site].sizes.get("time"),
            state_size=sensitivity.sizes.get(state_dim),
            sources=sensitivity.sizes.get("source"),
        )

    if use_bc:
        with timed("rhime.prepare_inputs.bc_sensitivity", sites=len(merged.sites)):
            fp_data = bc_sensitivity(
                fp_data,
                domain=domain,
                basis_case=bc_basis_case,
                bc_basis_directory=bc_basis_directory,
            )

    return fp_data


def _filter_merged_inversion_data(
    *,
    merged: _MergedInversionData,
    filters: Any,
) -> _MergedInversionData:
    """Filter merged RHIME data as a separate pre-basis preparation stage.

    Args:
        merged: Merged site data and site-aligned metadata from data gathering
            or reload.
        filters: Filter configuration accepted by
            :func:`openghg_inversions.filters.filtering`.

    Returns:
        Merged data containing filtered site datasets, with empty sites and
        their aligned averaging periods removed. If no filters are configured
        and all sites contain data, the original merged data are returned.

    Raises:
        ValueError: If every requested site is removed by filtering.
    """
    if filters is None and all(merged.fp_all[site].time.values.shape[0] > 0 for site in merged.sites):
        return merged

    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_data, sites, averaging_period = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        sites=merged.sites,
        averaging_period=merged.averaging_period,
        filters=filters,
    )
    fp_all = {key: value for key, value in merged.fp_all.items() if key.startswith(".")}
    fp_all.update(fp_data)
    return _MergedInversionData(fp_all=fp_all, sites=sites, averaging_period=averaging_period)


def prepare_fixedbasis_inversion_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    calculate_min_error: Literal["percentile", "residual"] | None = None,
    min_error_options: dict | None = None,
    return_basis_objects: bool = False,
    merged_data_only: bool = False,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> FixedBasisPreparedData:
    """Prepare data for legacy ``fixedbasisMCMC`` and its output adapters.

    This adapter preserves the fixed-basis inversion-input contract. Unless
    ``merged_data_only`` is true, the returned ``inv_inputs`` includes
    ``sigma_freq_index(nmeasure)`` derived from ``sigma_freq`` and anchored to
    ``start_date``. Modern RHIME preparation intentionally omits this
    component-specific variable.

    Returns:
        Prepared legacy data, including forward-model inputs and optional basis
        objects. When ``merged_data_only`` is true, only merged data and
        retained site metadata are populated.

    Warns:
        FutureWarning: If deprecated ``calculate_min_error`` is supplied.
    """
    merged = _prepare_merged_data(
        species=species,
        sites=sites,
        domain=domain,
        averaging_period=averaging_period,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        flux_sources=flux_sources,
        split_by_sectors=split_by_sectors,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        met_model=met_model,
        fp_model=fp_model,
        fp_height=fp_height,
        fp_species=fp_species,
        inlet=inlet,
        instrument=instrument,
        max_level=max_level,
        calibration_scale=calibration_scale,
        obs_data_level=obs_data_level,
        platform=platform,
        use_tracer=use_tracer,
        use_bc=use_bc,
        bc_input=bc_input,
        averaging_error=averaging_error,
        reload_merged_data=reload_merged_data,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        flux_non_finite_check=flux_non_finite_check,
    )

    if merged_data_only:
        return FixedBasisPreparedData(
            fp_all=merged.fp_all,
            sites=merged.sites,
            averaging_period=merged.averaging_period,
        )

    bc_basis_directory_arg = _bc_basis_directory_arg(bc_basis_directory)

    basis_result = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=bc_basis_directory_arg,
        country_directory=country_directory,
        fp_all=merged.fp_all,
        use_bc=use_bc,
        species=species,
        domain=domain,
        start_date=start_date,
        fix_outer_regions=fix_basis_outer_regions,
        emissions_name=flux_sources,
        outputname=output_name,
        output_path=basis_output_path,
        return_basis_objects=True,
    )
    fp_data, fixedbasis_basis_objects = cast(tuple[dict, dict[str, BasisFunctions]], basis_result)
    emissions_basis = fixedbasis_basis_objects["emissions"]
    basis_source = emissions_basis.basis_artifact_source or "generated"
    basis_path = getattr(emissions_basis, "basis_artifact_path", None)
    basis_objects = fixedbasis_basis_objects if return_basis_objects else {}

    fp_data, prepared_sites, prepared_averaging_period = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        sites=merged.sites,
        averaging_period=merged.averaging_period,
        filters=filters,
    )
    _set_domain_attrs(fp_data, prepared_sites, domain)

    inv_inputs = _make_inv_inputs(
        fp_data=fp_data,
        sites=prepared_sites,
        start_date=start_date,
        bc_freq=bc_freq,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_options=min_error_options,
    )
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=sigma_freq,
        anchor_time=start_date,
    )
    inv_inputs["sigma_freq_index"] = sigma_alignment.period_index.rename("sigma_freq_index")
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)

    return FixedBasisPreparedData(
        fp_all=merged.fp_all,
        fp_data=fp_data,
        inv_inputs=inv_inputs,
        basis_objects=basis_objects,
        basis_artifact_source=basis_source,
        basis_artifact_path=basis_path,
        sites=prepared_sites,
        averaging_period=prepared_averaging_period,
    )


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: dict | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimePreparedInputs:
    """Prepare modern RHIME inputs without exposing legacy fixedbasis containers.

    Observation filters are applied once to merged data before basis loading or
    generation. The same filtered site datasets and aligned metadata are then
    used for sensitivity construction.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        sites: Requested observation site names.
        domain: Model domain name.
        averaging_period: Observation averaging period, either scalar or
            site-aligned.
        start_date: Inclusive inversion start date.
        end_date: Exclusive inversion end date.
        output_name: Base output name used for data and basis artifacts.
        flux_sources: OpenGHG flux ``source`` values requested for the run.
        split_by_sectors: Whether to keep sector-resolved sensitivity inputs
            with a ``source`` provenance coordinate. Semantic sector names are
            applied later by the model specification.
        use_tracer: Unsupported placeholder for tracer inversions, where an
            additional species constrains the primary species through linked
            forward models.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.
    """
    with timed("rhime.prepare_inputs.merged_data", sites=len(sites), split_by_sectors=split_by_sectors):
        merged = _prepare_merged_data(
            species=species,
            sites=sites,
            domain=domain,
            averaging_period=averaging_period,
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
            flux_sources=flux_sources,
            split_by_sectors=split_by_sectors,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
            inlet=inlet,
            instrument=instrument,
            max_level=max_level,
            calibration_scale=calibration_scale,
            obs_data_level=obs_data_level,
            platform=platform,
            use_tracer=use_tracer,
            use_bc=use_bc,
            bc_input=bc_input,
            averaging_error=averaging_error,
            reload_merged_data=reload_merged_data,
            save_merged_data=save_merged_data,
            merged_data_dir=merged_data_dir,
            merged_data_name=merged_data_name,
            flux_non_finite_check=flux_non_finite_check,
        )

    with timed("rhime.prepare_inputs.obs_filtering", sites=len(merged.sites), filters=filters is not None):
        filtered_merged = _filter_merged_inversion_data(merged=merged, filters=filters)

    with timed(
        "rhime.prepare_inputs.basis_build",
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
    ):
        basis_functions = make_basis_functions(
            basis_algorithm=basis_algorithm,
            nbasis=nbasis,
            fp_basis_case=fp_basis_case,
            basis_directory=basis_directory,
            country_directory=country_directory,
            fp_all=filtered_merged.fp_all,
            species=species,
            domain=domain,
            start_date=start_date,
            fix_outer_regions=fix_basis_outer_regions,
            emissions_name=flux_sources,
            outputname=output_name,
            output_path=basis_output_path,
        )

    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(filtered_merged.sites)):
        fp_data = _rhime_site_data_from_basis_functions(
            merged=filtered_merged,
            basis_functions=basis_functions,
            domain=domain,
            split_by_sectors=split_by_sectors,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=_bc_basis_directory_arg(bc_basis_directory),
        )
    basis_source = basis_functions.basis_artifact_source or "generated"
    basis_path = getattr(basis_functions, "basis_artifact_path", None)
    _set_domain_attrs(fp_data, filtered_merged.sites, domain)

    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(filtered_merged.sites)):
        inv_inputs = _make_inv_inputs(
            fp_data=fp_data,
            sites=filtered_merged.sites,
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            calculate_min_error=None,
            min_error_options=min_error_options,
        )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    site_lats, site_lons = _site_release_coordinates(fp_data, filtered_merged.sites)
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(filtered_merged.sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )

    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        basis_artifact_source=basis_source,
        basis_artifact_path=basis_path,
        sites=tuple(filtered_merged.sites),
        averaging_period=tuple(filtered_merged.averaging_period),
        site_lats=site_lats,
        site_lons=site_lons,
    )
