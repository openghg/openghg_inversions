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

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import basis_functions_wrapper, make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
)
from openghg_inversions.filters import filtering
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data.get_data import convert_to_list, data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    open_datatree_loaded,
    save_datatree,
)
from openghg_inversions.sigma import SigmaAlignment

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float
RHIME_PREPARED_INPUTS_SCHEMA = "openghg_inversions.rhime_prepared_inputs"
RHIME_PREPARED_INPUTS_SCHEMA_VERSION = 1
_SITE_AVERAGING_PERIOD = "averaging_period"
_SITE_LAT = "release_lat"
_SITE_LON = "release_lon"


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


@dataclass(frozen=True, init=False)
class RhimePreparedInputs:
    """Modern RHIME preparation and durable serialization contract.

    Site labels and site-aligned metadata are owned by ``site_metadata``.
    Integer ``inv_inputs.site_indicator`` values are derived as zero-based
    project-schema positions into its ``site`` coordinate; they are distinct
    from CF compression-by-gathering indexes. The invariant is
    ``site_metadata.site[site_indicator] == nmeasure.site``.
    ``site_names`` remains available in memory for existing model code, but is
    regenerated from ``site_metadata`` rather than serialized.

    Args:
        inv_inputs: Canonical inversion inputs consumed by RHIME model
            builders.
        basis_functions: Retained flux basis object used to derive
            output-boundary basis and flux arrays.
        site_metadata: Keyword-only dataset indexed by the authoritative
            ``site`` coordinate.
            It must contain ``averaging_period`` and may contain the paired
            ``release_lat`` and ``release_lon`` variables. Every variable has
            length ``site``. Release coordinates preserve their upstream units;
            preparation currently supplies geographic degrees.
        sites: Deprecated compatibility input used to construct
            ``site_metadata`` when that dataset is omitted.
        averaging_period: Deprecated site-aligned compatibility input.
        basis_artifact_source: Deprecated compatibility provenance value,
            normalized into ``basis_functions.metadata``.
        basis_artifact_path: Deprecated compatibility provenance path,
            normalized into ``basis_functions.metadata``.
        site_lats: Deprecated site-aligned compatibility input.
        site_lons: Deprecated site-aligned compatibility input.

    Raises:
        ValueError: If site metadata, measurement indexing, or multi-source
            labels are inconsistent.
    """

    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    site_metadata: xr.Dataset

    def __init__(
        self,
        inv_inputs: xr.Dataset,
        basis_functions: BasisFunctions,
        sites: Sequence[str] | None = None,
        averaging_period: Sequence[str | None] | None = None,
        basis_artifact_source: str | None = None,
        basis_artifact_path: str | None = None,
        site_lats: Sequence[float] | None = None,
        site_lons: Sequence[float] | None = None,
        *,
        site_metadata: xr.Dataset | None = None,
    ) -> None:
        """Initialize and normalize prepared inputs.

        The legacy site-aligned arguments retain their previous positional
        order while callers migrate to the keyword-only labeled
        ``site_metadata`` dataset. They are immediately normalized into that
        dataset and are not retained as parallel state.

        Args:
            inv_inputs: Canonical inversion inputs with a site/time measurement
                MultiIndex and integer site indicators.
            basis_functions: Retained basis object and its provenance.
            site_metadata: Authoritative site-indexed metadata dataset.
            sites: Deprecated site-label compatibility input.
            averaging_period: Deprecated site-aligned compatibility input.
            basis_artifact_source: Deprecated basis-provenance compatibility
                input.
            basis_artifact_path: Deprecated basis-path compatibility input.
            site_lats: Deprecated release-latitude compatibility input.
            site_lons: Deprecated release-longitude compatibility input.

        Raises:
            ValueError: If both metadata representations are supplied or the
                prepared-input semantic invariants do not hold.
        """
        validate_basis_functions = getattr(basis_functions, "validated", None)
        if callable(validate_basis_functions):
            basis_functions = cast(BasisFunctions, validate_basis_functions())
        if site_metadata is None:
            if sites is None or averaging_period is None:
                raise ValueError(
                    "RhimePreparedInputs requires site_metadata, or both sites and averaging_period."
                )
            site_metadata = _make_site_metadata(
                sites=sites,
                averaging_period=averaging_period,
                site_lats=site_lats,
                site_lons=site_lons,
            )
        elif any(value is not None for value in (sites, averaging_period, site_lats, site_lons)):
            raise ValueError("Pass site_metadata or legacy site-aligned arguments, not both.")

        existing_metadata = getattr(basis_functions, "metadata", {})
        metadata = dict(existing_metadata)
        if basis_artifact_source is not None:
            metadata[BASIS_ARTIFACT_SOURCE_ATTR] = basis_artifact_source
        if basis_artifact_path is not None:
            metadata[BASIS_ARTIFACT_PATH_ATTR] = basis_artifact_path
        if metadata != existing_metadata:
            basis_functions = basis_functions.with_metadata(metadata)

        normalized_site_metadata = _normalize_site_metadata(site_metadata)
        normalized_inv_inputs, normalized_site_metadata = _canonicalize_rhime_inv_inputs(
            inv_inputs,
            site_metadata=normalized_site_metadata,
            basis_functions=basis_functions,
        )
        object.__setattr__(self, "inv_inputs", normalized_inv_inputs)
        object.__setattr__(self, "basis_functions", basis_functions)
        object.__setattr__(self, "site_metadata", normalized_site_metadata)

    @property
    def sites(self) -> tuple[str, ...]:
        """Return retained site labels in indicator-decoding order."""
        return tuple(str(site) for site in self.site_metadata["site"].values)

    @property
    def averaging_period(self) -> tuple[str | None, ...]:
        """Return averaging periods aligned to :attr:`sites`."""
        return tuple(
            None if value is None or pd.isna(value) else str(value)
            for value in self.site_metadata[_SITE_AVERAGING_PERIOD].values
        )

    @property
    def site_lats(self) -> tuple[float, ...] | None:
        """Return site release latitudes when retained."""
        if _SITE_LAT not in self.site_metadata:
            return None
        return tuple(float(value) for value in self.site_metadata[_SITE_LAT].values)

    @property
    def site_lons(self) -> tuple[float, ...] | None:
        """Return site release longitudes when retained."""
        if _SITE_LON not in self.site_metadata:
            return None
        return tuple(float(value) for value in self.site_metadata[_SITE_LON].values)

    @property
    def basis_artifact_source(self) -> str:
        """Return retained basis provenance, defaulting to ``generated``."""
        return self.basis_functions.basis_artifact_source or "generated"

    @property
    def basis_artifact_path(self) -> str | None:
        """Return the provenance-only basis artifact path."""
        return self.basis_functions.basis_artifact_path

    def validated(self) -> Self:
        """Return a freshly canonicalized copy of these prepared inputs.

        This re-establishes the semantic invariants after possible in-place
        mutation of the contained xarray objects.

        Returns:
            Prepared inputs normalized from the current xarray values.
        """
        return type(self)(
            inv_inputs=self.inv_inputs,
            basis_functions=self.basis_functions,
            site_metadata=self.site_metadata,
        )

    def to_datatree(self) -> xr.DataTree:
        """Convert prepared RHIME inputs to the versioned DataTree schema.

        The basis object is embedded in the artifact. ``basis_artifact_path``
        is retained only as provenance and is never read while serializing or
        reconstructing the prepared inputs.

        Returns:
            DataTree containing ``inv_inputs``, ``basis_functions``, and
            ``site_metadata`` child nodes.

        Raises:
            ValueError: If site metadata or inversion inputs no longer satisfy
                the prepared-input invariants.
        """
        prepared = self.validated()
        site_metadata = prepared.site_metadata
        inv_inputs = prepared.inv_inputs.drop_vars("site_names", errors="ignore")
        multiindex_dims = tuple(
            dim
            for dim in inv_inputs.dims
            if isinstance(dim, str) and isinstance(inv_inputs.indexes.get(dim), pd.MultiIndex)
        )
        if "nmeasure" not in multiindex_dims:
            raise ValueError("RhimePreparedInputs inv_inputs requires the nmeasure MultiIndex.")
        dt = xr.DataTree.from_dict(
            {
                "inv_inputs": xr.DataTree(encode_cf_multiindexes(inv_inputs, multiindex_dims)),
                "basis_functions": prepared.basis_functions.to_datatree(),
                "site_metadata": xr.DataTree(_site_metadata_for_serialisation(site_metadata)),
            }
        )
        dt.attrs = {
            "schema": RHIME_PREPARED_INPUTS_SCHEMA,
            "schema_version": RHIME_PREPARED_INPUTS_SCHEMA_VERSION,
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
            ValueError: If the prepared schema, site metadata, serialized
                MultiIndex, site indicators, or source labels are malformed.
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

        missing_nodes = [
            name for name in ("inv_inputs", "basis_functions", "site_metadata") if name not in dt.children
        ]
        if missing_nodes:
            raise KeyError(f"Missing required RhimePreparedInputs node(s): {missing_nodes!r}.")

        encoded_inv_inputs = cast(xr.DataTree, dt["inv_inputs"]).to_dataset()
        compressed_indexes = tuple(
            name
            for name, coordinate in encoded_inv_inputs.coords.items()
            if isinstance(name, str) and coordinate.dims == (name,) and "compress" in coordinate.attrs
        )
        if "nmeasure" not in compressed_indexes:
            raise ValueError(
                "RhimePreparedInputs inv_inputs CF gathered coordinate 'nmeasure' "
                "is missing its 'compress' metadata."
            )
        inv_inputs = decode_cf_multiindexes(encoded_inv_inputs, compressed_indexes)
        basis_functions_dt = cast(xr.DataTree, dt["basis_functions"])
        basis_functions = BasisFunctions.from_datatree(basis_functions_dt)
        return cls(
            inv_inputs=inv_inputs,
            basis_functions=basis_functions,
            site_metadata=_site_metadata_from_serialisation(
                cast(xr.DataTree, dt["site_metadata"]).to_dataset()
            ),
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


def _make_site_metadata(
    *,
    sites: Sequence[str],
    averaging_period: Sequence[str | None],
    site_lats: Sequence[float] | None = None,
    site_lons: Sequence[float] | None = None,
) -> xr.Dataset:
    """Construct labeled site metadata from compatibility inputs.

    Args:
        sites: Authoritative labels in site-indicator decoding order.
        averaging_period: Observation periods aligned to ``sites``.
        site_lats: Optional release latitudes aligned to ``sites``.
        site_lons: Optional release longitudes aligned to ``sites``.

    Returns:
        Dataset containing the labeled site metadata.

    Raises:
        ValueError: During normalization if lengths differ, averaging periods
            are invalid, or only one release-coordinate vector is supplied.
    """
    data_vars: dict[str, tuple[str, object]] = {
        _SITE_AVERAGING_PERIOD: (
            "site",
            np.asarray(list(averaging_period), dtype=object),
        )
    }
    if site_lats is not None:
        data_vars[_SITE_LAT] = ("site", np.asarray(list(site_lats), dtype=float))
    if site_lons is not None:
        data_vars[_SITE_LON] = ("site", np.asarray(list(site_lons), dtype=float))
    result = xr.Dataset(data_vars, coords={"site": [str(site) for site in sites]})
    if _SITE_LAT in result:
        result[_SITE_LAT].attrs["units"] = "degrees_north"
        result[_SITE_LON].attrs["units"] = "degrees_east"
    return result


def _normalize_site_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Validate and normalize the labeled site metadata dataset.

    Args:
        ds: Candidate site metadata.

    Returns:
        A copy with normalized string site labels and averaging periods.

    Raises:
        ValueError: If site labels are missing or duplicated, required
            site-aligned variables are absent or misdimensioned, or coordinate
            variables are incomplete or infinite.
    """
    if "site" not in ds.coords or ds["site"].dims != ("site",):
        raise ValueError("RhimePreparedInputs site_metadata requires a one-dimensional 'site' coordinate.")
    sites = tuple(str(site) for site in ds["site"].values)
    if len(set(sites)) != len(sites):
        raise ValueError("RhimePreparedInputs site_metadata 'site' labels must be unique.")
    if _SITE_AVERAGING_PERIOD not in ds:
        raise ValueError("RhimePreparedInputs site_metadata is missing 'averaging_period'.")
    invalid_variables = [name for name, variable in ds.data_vars.items() if variable.dims != ("site",)]
    if invalid_variables:
        raise ValueError(
            "RhimePreparedInputs site_metadata variables must each have only dimension 'site'; "
            f"invalid variables: {invalid_variables!r}."
        )
    invalid_coordinates = [
        name
        for name, coordinate in ds.coords.items()
        if name != "site" and any(dim != "site" for dim in coordinate.dims)
    ]
    if invalid_coordinates:
        raise ValueError(
            "RhimePreparedInputs site_metadata auxiliary coordinates must be scalar or "
            "site-aligned; "
            f"invalid coordinates: {invalid_coordinates!r}."
        )

    periods: list[str | None] = []
    for value in ds[_SITE_AVERAGING_PERIOD].values:
        if value is None or pd.isna(value):
            periods.append(None)
        elif isinstance(value, str | np.str_):
            period = str(value)
            periods.append(period or None)
        else:
            raise ValueError(
                "RhimePreparedInputs site_metadata 'averaging_period' values must be strings or missing."
            )

    has_lat = _SITE_LAT in ds
    has_lon = _SITE_LON in ds
    if has_lat != has_lon:
        raise ValueError(
            "RhimePreparedInputs site_metadata must contain both release_lat and release_lon, or neither."
        )

    result = ds.copy()
    result = result.assign_coords(site=ds["site"].copy(data=np.asarray(sites, dtype=str)))
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(data=np.asarray(periods, dtype=object))
    for name in (_SITE_LAT, _SITE_LON):
        if name not in result:
            continue
        if result[name].dims != ("site",):
            raise ValueError(f"RhimePreparedInputs site_metadata {name!r} must have dimension 'site'.")
        values = np.asarray(result[name].data, dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"RhimePreparedInputs site_metadata {name!r} must not contain infinity.")
        result[name] = ds[name].copy(data=values)
    return result


def _site_metadata_for_serialisation(ds: xr.Dataset) -> xr.Dataset:
    """Encode missing periods with the reserved empty-string storage sentinel."""
    result = ds.copy()
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(
        data=np.asarray(["" if value is None else str(value) for value in ds[_SITE_AVERAGING_PERIOD].values])
    )
    return result


def _site_metadata_from_serialisation(ds: xr.Dataset) -> xr.Dataset:
    """Restore missing averaging periods from their storage representation."""
    result = ds.copy()
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(
        data=np.asarray(
            [None if str(value) == "" else str(value) for value in ds[_SITE_AVERAGING_PERIOD].values],
            dtype=object,
        )
    )
    return result


def _multi_source_basis_labels(basis_functions: BasisFunctions) -> tuple[str, ...] | None:
    """Return the retained basis object's semantic source order, when present."""
    source_labels = getattr(basis_functions, "source_labels", None)
    return tuple(source_labels) if source_labels is not None else None


def _canonicalize_rhime_inv_inputs(
    ds: xr.Dataset,
    *,
    site_metadata: xr.Dataset,
    basis_functions: BasisFunctions,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Derive site lookup state and canonicalize source ordering.

    Args:
        ds: Canonical inversion inputs to validate.
        site_metadata: Site lookup indexed in indicator-decoding order.
        basis_functions: Basis whose operator defines multi-source order.

    Returns:
        Inversion inputs with regenerated ``site_indicator`` and ``site_names``
        plus site metadata selected into the observed-site order. For
        multi-source bases, variables are also reordered to the basis source
        order, whether sources form a dimension or a level of the gathered
        state index.

    Raises:
        ValueError: If measurement indexes, site labels, or source labels do
            not satisfy the semantic prepared-input contract.
    """
    nmeasure_index = ds.indexes.get("nmeasure")
    if not isinstance(nmeasure_index, pd.MultiIndex):
        raise ValueError(  # noqa: TRY004 - malformed artifact schema, not an argument type error
            "RhimePreparedInputs inv_inputs must have a 'nmeasure' MultiIndex with levels ('site', 'time')."
        )
    if tuple(nmeasure_index.names) != ("site", "time"):
        raise ValueError(
            "RhimePreparedInputs inv_inputs 'nmeasure' MultiIndex must have levels "
            f"('site', 'time'), got {tuple(nmeasure_index.names)!r}."
        )
    measurement_sites_raw = tuple(nmeasure_index.get_level_values("site"))
    if not all(isinstance(site, str) for site in measurement_sites_raw):
        raise ValueError("RhimePreparedInputs nmeasure site labels must all be strings.")
    observed_sites = tuple(dict.fromkeys(measurement_sites_raw))
    metadata_sites = tuple(site_metadata["site"].values.tolist())
    missing_sites = [site for site in observed_sites if site not in metadata_sites]
    if missing_sites:
        raise ValueError(
            f"RhimePreparedInputs site_metadata is missing observed site labels: {missing_sites!r}."
        )
    site_metadata = site_metadata.sel(site=list(observed_sites))
    site_lookup = {site: index for index, site in enumerate(observed_sites)}
    indicator_values = np.fromiter(
        (site_lookup[site] for site in measurement_sites_raw),
        dtype=np.int64,
        count=len(measurement_sites_raw),
    )

    result = ds.drop_vars(("site_indicator", "site_names"), errors="ignore")
    result["site_indicator"] = xr.DataArray(
        indicator_values,
        dims=("nmeasure",),
        name="site_indicator",
    )
    result["site_names"] = xr.DataArray(
        np.asarray(observed_sites, dtype=str),
        dims=("nsite",),
        name="site_names",
    )

    expected_sources = _multi_source_basis_labels(basis_functions)
    if expected_sources is None:
        return result, site_metadata
    if len(set(expected_sources)) != len(expected_sources):
        raise ValueError("RhimePreparedInputs multi-source basis labels must be unique.")
    if "H" not in result:
        raise ValueError("RhimePreparedInputs inv_inputs must contain H for a multi-source basis.")
    sensitivity = result["H"]
    state_dim: str | None = None
    if "source" in sensitivity.dims:
        actual_sources = tuple(str(source) for source in sensitivity["source"].values)
        if len(set(actual_sources)) != len(actual_sources):
            raise ValueError("RhimePreparedInputs inv_inputs H source labels must be unique.")
    else:
        gathered_dims = [
            str(dim)
            for dim in sensitivity.dims
            if isinstance(sensitivity.indexes.get(dim), pd.MultiIndex)
            and "source" in sensitivity.indexes[dim].names
        ]
        if len(gathered_dims) != 1:
            raise ValueError(
                "RhimePreparedInputs inv_inputs H must have either a source dimension or "
                "one gathered state MultiIndex containing a 'source' level for a multi-source basis."
            )
        state_dim = gathered_dims[0]
        state_index = sensitivity.indexes[state_dim]
        source_values = state_index.get_level_values("source").to_numpy()
        if not all(isinstance(source, str | np.str_) for source in source_values):
            raise ValueError("RhimePreparedInputs gathered H source labels must all be strings.")
        source_labels = tuple(str(source) for source in source_values)
        _, first_indices = np.unique(source_values, return_index=True)
        actual_sources = tuple(str(source_values[index]) for index in np.sort(first_indices))
    if set(actual_sources) != set(expected_sources):
        missing = sorted(set(expected_sources) - set(actual_sources))
        unexpected = sorted(set(actual_sources) - set(expected_sources))
        raise ValueError(
            "RhimePreparedInputs inv_inputs H source labels do not match the basis operator: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    if state_dim is not None:
        state_order = [
            index
            for source in expected_sources
            for index, actual_source in enumerate(source_labels)
            if actual_source == source
        ]
        return result.isel({state_dim: state_order}), site_metadata
    result = result.assign_coords(source=("source", list(actual_sources)))
    return result.sel(source=list(expected_sources)), site_metadata


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
        site_metadata=_make_site_metadata(
            sites=filtered_merged.sites,
            averaging_period=filtered_merged.averaging_period,
            site_lats=site_lats,
            site_lons=site_lons,
        ),
    )
