"""Shared mechanics and durable data contracts for inversion preparation.

``prepare_rhime_inputs`` returns backend-neutral observations, sensitivities,
basis metadata, and site metadata; component-specific model arrays are
intentionally absent. The temporary legacy fixed-basis orchestration is owned
by :mod:`openghg_inversions.hbmcmc.preparation` and composes the lower-level
retrieval, filtering, basis, and array helpers retained here.

``RhimePreparedInputs`` validates the relationships between these labeled
arrays when it is constructed. When the retained basis-functions object
provides ``validated()``, preparation uses the returned copy after that method
has rechecked its mutable flux, operator data, and source labels. Compatible
objects without that hook are retained unchanged.

Preparation can read OpenGHG object stores or local merged-data artifacts,
write merged-data and basis artifacts, emit warnings and progress messages,
and record timing information. These backend-neutral preparation functions do
not construct a PyMC model.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
)
from openghg_inversions.filters import filtering
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data._site_options import (
    expand_site_option,
    is_column_observation,
)
from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import normalise_min_error_options
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    open_datatree_loaded,
    save_datatree,
)

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float
RHIME_PREPARED_INPUTS_SCHEMA = "openghg_inversions.rhime_prepared_inputs"
RHIME_PREPARED_INPUTS_SCHEMA_VERSION = 1
_SITE_AVERAGING_PERIOD = "averaging_period"
SiteStringOption = Sequence[str | None] | str | None
SiteInletOption = Sequence[str | slice | None] | str | None
SiteIntegerOption = Sequence[int | None] | int | None

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
        site_metadata: Dataset indexed by the authoritative ``site``
            coordinate. Every variable contains exactly one value per site,
            and ``averaging_period`` is required. Observation metadata that
            is genuinely constant per site may also be stored here. Values
            that vary within a site, such as satellite or aircraft release
            locations, must instead remain observation-aligned arrays. Such
            arrays may be carried alongside the inversion arrays without
            implying that model builders consume them.
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
        site_metadata: xr.Dataset,
    ) -> None:
        """Initialize and normalize prepared inputs.

        Args:
            inv_inputs: Canonical inversion inputs with a site/time measurement
                MultiIndex and integer site indicators.
            basis_functions: Retained basis object and its provenance.
            site_metadata: Authoritative site-indexed metadata dataset.
        Raises:
            ValueError: If the prepared-input semantic invariants do not hold.
        """
        validate_basis_functions = getattr(basis_functions, "validated", None)
        if callable(validate_basis_functions):
            basis_functions = cast(BasisFunctions, validate_basis_functions())

        normalized_site_metadata = _normalize_site_metadata(site_metadata)
        normalized_inv_inputs, normalized_site_metadata = _canonicalize_rhime_inv_inputs(
            inv_inputs,
            site_metadata=normalized_site_metadata,
            basis_functions=basis_functions,
        )
        object.__setattr__(self, "inv_inputs", normalized_inv_inputs)
        object.__setattr__(self, "basis_functions", basis_functions)
        object.__setattr__(self, "site_metadata", normalized_site_metadata)

    @classmethod
    def from_legacy_inputs(
        cls: type[Self],
        inv_inputs: xr.Dataset,
        basis_functions: BasisFunctions,
        sites: Sequence[str],
        averaging_period: Sequence[str | None],
        basis_artifact_source: str | None = None,
        basis_artifact_path: str | None = None,
        site_lats: Sequence[float] | None = None,
        site_lons: Sequence[float] | None = None,
    ) -> Self:
        """Adapt the former positional fields to the labeled-data contract.

        Args:
            inv_inputs: Canonical inversion inputs.
            basis_functions: Retained basis object.
            sites: Site labels in indicator-decoding order.
            averaging_period: Observation periods aligned to ``sites``.
            basis_artifact_source: Optional basis provenance value.
            basis_artifact_path: Optional basis provenance path.
            site_lats: Optional legacy release latitudes, one per site.
            site_lons: Optional legacy release longitudes, one per site.

        Returns:
            Prepared inputs using labeled site metadata.
        """
        existing_metadata = basis_functions.metadata
        metadata = dict(existing_metadata)
        if basis_artifact_source is not None:
            metadata[BASIS_ARTIFACT_SOURCE_ATTR] = basis_artifact_source
        if basis_artifact_path is not None:
            metadata[BASIS_ARTIFACT_PATH_ATTR] = basis_artifact_path
        if metadata != existing_metadata:
            basis_functions = basis_functions.with_metadata(metadata)
        if (site_lats is None) != (site_lons is None):
            raise ValueError("Legacy site_lats and site_lons must be supplied together.")
        if site_lats is not None and site_lons is not None:
            if len(site_lats) != len(sites) or len(site_lons) != len(sites):
                raise ValueError("Legacy release coordinates must have one value per site.")
            has_release_lat = "release_lat" in inv_inputs
            has_release_lon = "release_lon" in inv_inputs
            if has_release_lat != has_release_lon:
                raise ValueError("Observation-aligned release_lat and release_lon must be supplied together.")
            if not has_release_lat:
                site_lookup = {str(site): index for index, site in enumerate(sites)}
                measurement_sites = tuple(str(site) for site in inv_inputs["site"].values)
                missing_sites = sorted(set(measurement_sites) - set(site_lookup))
                if missing_sites:
                    raise ValueError(f"Legacy release coordinates are missing site(s): {missing_sites!r}.")
                inv_inputs = inv_inputs.assign_coords(
                    release_lat=(
                        "nmeasure",
                        [site_lats[site_lookup[site]] for site in measurement_sites],
                    ),
                    release_lon=(
                        "nmeasure",
                        [site_lons[site_lookup[site]] for site in measurement_sites],
                    ),
                )
                inv_inputs["release_lat"].attrs["units"] = "degrees_north"
                inv_inputs["release_lon"].attrs["units"] = "degrees_east"
        return cls(
            inv_inputs=inv_inputs,
            basis_functions=basis_functions,
            site_metadata=_make_site_metadata(sites=sites, averaging_period=averaging_period),
        )

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
) -> xr.Dataset:
    """Construct labeled site metadata from compatibility inputs.

    Args:
        sites: Authoritative labels in site-indicator decoding order.
        averaging_period: Observation periods aligned to ``sites``.
    Returns:
        Dataset containing the labeled site metadata.

    Raises:
        ValueError: During normalization if lengths differ or averaging periods
            are invalid.
    """
    return xr.Dataset(
        {
            _SITE_AVERAGING_PERIOD: (
                "site",
                np.asarray(list(averaging_period), dtype=object),
            )
        },
        coords={"site": [str(site) for site in sites]},
    )


def _normalize_site_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Validate and normalize the labeled site metadata dataset.

    Args:
        ds: Candidate site metadata.

    Returns:
        A copy with normalized string site labels and averaging periods.

    Raises:
        ValueError: If site labels are missing or duplicated, required
            site-aligned variables are absent or misdimensioned, or coordinate
            variables are incomplete.
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

    result = ds.copy()
    result = result.assign_coords(site=ds["site"].copy(data=np.asarray(sites, dtype=str)))
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(data=np.asarray(periods, dtype=object))
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


def _normalise_site_strings(
    value: Sequence[str | None] | str | None,
    *,
    length: int,
    name: str,
) -> list[str | None]:
    """Normalize and validate one optional-string value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str)]
    if invalid:
        raise ValueError(f"`{name}` entries must be strings or None. Invalid value(s): {invalid!r}.")
    return normalized


def _normalise_site_integers(
    value: Sequence[int | None] | int | None,
    *,
    length: int,
    name: str,
) -> list[int | None]:
    """Normalize and validate one optional integer value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))

    invalid = [
        item
        for item in normalized
        if item is not None and (not isinstance(item, Integral) or isinstance(item, bool))
    ]
    if invalid:
        raise ValueError(f"`{name}` entries must be integers or None. Invalid value(s): {invalid!r}.")
    return [None if item is None else int(item) for item in normalized]


def _normalise_site_inlets(
    value: Sequence[str | slice | None] | str | None,
    *,
    length: int,
) -> list[str | slice | None]:
    """Normalize inlet selectors, including legacy per-site slice selectors."""
    normalized = list(expand_site_option(value, nsites=length, name="inlet"))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str | slice)]
    if invalid:
        raise ValueError(f"`inlet` entries must be strings, slices, or None. Invalid value(s): {invalid!r}.")
    return normalized


@dataclass(frozen=True)
class _SiteOptions:
    """All runner inputs whose positions are aligned to ``sites``.

    Every field has the same length and ordering. Selection always creates a
    new complete record so no option can drift independently from its site.
    """

    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    inlet: tuple[str | slice | None, ...]
    fp_height: tuple[str | None, ...]
    instrument: tuple[str | None, ...]
    platform: tuple[str | None, ...]
    obs_data_level: tuple[str | None, ...]
    met_model: tuple[str | None, ...]
    max_level: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Freeze supplied sequences and enforce the common-length invariant."""
        field_names = (
            "sites",
            "averaging_period",
            "inlet",
            "fp_height",
            "instrument",
            "platform",
            "obs_data_level",
            "met_model",
            "max_level",
        )
        for name in field_names:
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if not self.sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError(f"Site names must be unique: {self.sites!r}.")

        expected_length = len(self.sites)
        misaligned = {
            name: len(getattr(self, name))
            for name in field_names[1:]
            if len(getattr(self, name)) != expected_length
        }
        if misaligned:
            raise ValueError(
                "Every site-aligned option must have the same length as `sites`; "
                f"expected {expected_length}, got {misaligned!r}."
            )

    @classmethod
    def from_inputs(
        cls,
        *,
        sites: Sequence[str],
        averaging_period: Sequence[str | None] | str | None,
        inlet: Sequence[str | slice | None] | str | None,
        fp_height: Sequence[str | None] | str | None,
        instrument: Sequence[str | None] | str | None,
        platform: Sequence[str | None] | str | None,
        obs_data_level: Sequence[str | None] | str | None,
        met_model: Sequence[str | None] | str | None,
        max_level: Sequence[int | None] | int | None,
    ) -> _SiteOptions:
        """Normalize all site options and validate their common length.

        Site names are uppercased. Scalar option values are broadcast, while
        sequences must match the number of sites. Inlets also support legacy
        ``slice`` selectors; maximum levels reject booleans.

        Raises:
            ValueError: If no sites are supplied, site names are duplicated,
                an option has the wrong length, or an entry has an invalid
                type.
        """
        normalized_sites = [site.upper() for site in sites]
        if not normalized_sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(normalized_sites)) != len(normalized_sites):
            raise ValueError(f"Site names must be unique: {normalized_sites!r}.")
        nsites = len(normalized_sites)
        return cls(
            sites=tuple(normalized_sites),
            averaging_period=tuple(
                _normalise_site_strings(averaging_period, length=nsites, name="averaging_period")
            ),
            inlet=tuple(_normalise_site_inlets(inlet, length=nsites)),
            fp_height=tuple(_normalise_site_strings(fp_height, length=nsites, name="fp_height")),
            instrument=tuple(_normalise_site_strings(instrument, length=nsites, name="instrument")),
            platform=tuple(_normalise_site_strings(platform, length=nsites, name="platform")),
            obs_data_level=tuple(
                _normalise_site_strings(obs_data_level, length=nsites, name="obs_data_level")
            ),
            met_model=tuple(_normalise_site_strings(met_model, length=nsites, name="met_model")),
            max_level=tuple(_normalise_site_integers(max_level, length=nsites, name="max_level")),
        )

    def select_indices(self, indices: Sequence[int]) -> _SiteOptions:
        """Return a new complete option record restricted to ``indices``."""

        def select(values: Sequence[Any]) -> tuple[Any, ...]:
            return tuple(values[index] for index in indices)

        return _SiteOptions(
            sites=select(self.sites),
            averaging_period=select(self.averaging_period),
            inlet=select(self.inlet),
            fp_height=select(self.fp_height),
            instrument=select(self.instrument),
            platform=select(self.platform),
            obs_data_level=select(self.obs_data_level),
            met_model=select(self.met_model),
            max_level=select(self.max_level),
        )

    @property
    def is_column(self) -> bool:
        """Whether any retained site uses a supported column-data selector."""
        return any(
            is_column_observation(inlet, platform)
            for inlet, platform in zip(self.inlet, self.platform, strict=True)
        )

    def retain_sites(self, retained_sites: Sequence[str], *, context: str) -> _SiteOptions:
        """Return options for retained sites in their supplied order.

        Raises:
            ValueError: If requested or retained names are duplicated, or a
                retained name was not in the original request.
        """
        normalized_retained = [site.upper() for site in retained_sites]
        index_by_site = {site: index for index, site in enumerate(self.sites)}
        if len(index_by_site) != len(self.sites):
            raise ValueError(f"{context} cannot align duplicate requested site names: {self.sites!r}.")

        missing_sites = [site for site in normalized_retained if site not in index_by_site]
        if missing_sites:
            raise ValueError(f"{context} returned site(s) that were not requested: {missing_sites!r}.")
        if len(set(normalized_retained)) != len(normalized_retained):
            raise ValueError(f"{context} returned duplicate site names: {normalized_retained!r}.")

        return self.select_indices([index_by_site[site] for site in normalized_retained])


@dataclass
class RhimeMergedData:
    """Merged RHIME data and complete site-aligned metadata between stages.

    Args:
        fp_all: Merged per-site datasets plus shared flux, boundary-condition,
            and calibration entries.
        site_options: Complete site-aligned acquisition options retained after
            retrieval or filtering.

    Notes:
        This is a supported orchestration handoff. Its datasets remain
        backend-neutral and may be Dask-backed; later stages must treat them as
        borrowed.
    """

    fp_all: dict
    site_options: _SiteOptions

    @property
    def sites(self) -> tuple[str, ...]:
        """Retained site names."""
        return self.site_options.sites

    @property
    def averaging_period(self) -> tuple[str | None, ...]:
        """Retained averaging periods aligned to :attr:`sites`."""
        return self.site_options.averaging_period

    @property
    def platform(self) -> tuple[str | None, ...]:
        """Retained observation platforms aligned to :attr:`sites`."""
        return self.site_options.platform


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    site_options: _SiteOptions,
) -> _SiteOptions:
    """Align site-level options when loaded merged data lacks requested sites."""
    sites_merged = [site for site in fp_all if not site.startswith(".")]
    if all(site in sites_merged for site in site_options.sites):
        return site_options

    keep_indices = [index for index, site in enumerate(site_options.sites) if site in sites_merged]
    dropped_sites = [site for site in site_options.sites if site not in sites_merged]
    if not keep_indices:
        raise ValueError(
            "Loaded merged data does not include any requested sites. "
            f"Requested sites: {site_options.sites}. Available merged-data sites: {sites_merged}."
        )

    print(f"\nDropping {dropped_sites} sites as they are not included in the merged data object.\n")
    return site_options.select_indices(keep_indices)


def _select_fp_all_sites(fp_all: dict, sites: Sequence[str]) -> dict:
    """Keep requested sites and prune site-keyed calibration scales."""
    site_names = set(sites)
    selected = {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}

    scales = selected.get(".scales")
    if isinstance(scales, Mapping):
        selected[".scales"] = {site: scales[site] for site in sites if site in scales}

    return selected


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: Sequence[str],
    start_date: str,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    calculate_min_error: Literal["percentile", "residual"] | None,
    min_error_per_site: bool,
) -> xr.Dataset:
    """Create backend-neutral inversion inputs with min-error compatibility.

    Args:
        fp_data: Filtered per-site observations and sensitivity data.
        sites: Retained sites in observation order.
        start_date: Anchor for fixed-duration boundary-condition periods.
        bc_freq: Optional boundary-condition period frequency.
        min_error: Minimum-error value or calculation method.
        calculate_min_error: Deprecated minimum-error calculation argument.
        min_error_per_site: Whether calculated minimum error varies by site.

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

    return make_inv_inputs(
        fp_data,
        sites=list(sites),
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_per_site=min_error_per_site,
        start_date=start_date,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values")
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values")


def _platform_by_site(sites: Sequence[str], platform: Any) -> dict[str, str | None]:
    """Return platform values keyed by site name."""
    if isinstance(platform, str) or platform is None:
        return {site: platform for site in sites}
    try:
        values = list(platform)
    except TypeError:
        return {site: None for site in sites}
    if len(values) != len(sites):
        return {site: None for site in sites}
    return {site: None if value is None else str(value) for site, value in zip(sites, values, strict=True)}


def _scale_satellite_bc_sensitivity_to_column_signal(
    inv_inputs: xr.Dataset,
    *,
    sites: Sequence[str],
    platform: Any,
) -> xr.Dataset:
    """Scale satellite BC sensitivity into the same corrected column space as ``mf``.

    OpenGHG column retrieval subtracts OCO prior-factor terms from XCO2 before
    inversion. Boundary-condition sensitivities arrive as a full-column baseline
    contribution, so satellite rows must be reduced before the model sees them.
    """
    required_vars = {"H_bc", "mf", "mf_prior_factor", "mf_prior_upper_level_factor", "site"}
    if not required_vars <= set(inv_inputs.variables):
        return inv_inputs

    platform_lookup = _platform_by_site(sites, platform)
    satellite_sites = {
        site for site, value in platform_lookup.items() if value is not None and "satellite" in value.lower()
    }
    if not satellite_sites:
        return inv_inputs

    site_values = inv_inputs["site"].astype(str)
    satellite_mask = site_values.isin(list(satellite_sites))
    if not bool(satellite_mask.any()):
        return inv_inputs

    with xr.set_options(keep_attrs="default"):
        raw_column = (
            inv_inputs["mf"] + inv_inputs["mf_prior_factor"] + inv_inputs["mf_prior_upper_level_factor"]
        )
        # TODO(#553): This is a deliberate BC-scaling workaround while the
        # retrieval information needed for an exact corrected-column transform
        # is unavailable. It is not a resolution of the underlying
        # mathematical limitation; replace it with retrieval-aware handling.
        scale = xr.where(raw_column > 0, inv_inputs["mf"] / raw_column, 1.0)
        scale = scale.clip(min=0.0, max=1.0).where(satellite_mask, 1.0)

    result = inv_inputs.copy()
    attrs = dict(result["H_bc"].attrs)
    result["H_bc"] = result["H_bc"] * scale
    result["H_bc"].attrs = attrs
    result["H_bc"].attrs["satellite_column_bc_scale"] = (
        "Applied to satellite rows using mf / (mf + mf_prior_factor + mf_prior_upper_level_factor)."
    )
    return result


def _prepare_merged_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimeMergedData:
    """Gather or reload merged data and align site metadata.

    ``flux_sources`` contains modern OpenGHG flux ``source`` values. This
    helper passes them to lower-level data loading through the legacy
    ``emissions_name`` argument. Retrieval may access OpenGHG object stores,
    print progress, and optionally save merged data. Reload reads a local
    artifact. Both paths retain one complete :class:`_SiteOptions` record.

    Returns:
        Merged per-site data and aligned retained-site options.

    Raises:
        ValueError: If site options are invalid, no requested sites are loaded,
            or retrieval returns misaligned metadata.
        RuntimeError: If neither retrieval nor reload produces merged data.
    """
    if use_tracer:
        raise ValueError("Tracer inversions are not supported by this preparation path.")
    site_options = _SiteOptions.from_inputs(
        sites=sites,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
    )
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
            site_options = _drop_sites_missing_from_loaded_data(
                fp_all=fp_all,
                site_options=site_options,
            )
            fp_all = _select_fp_all_sites(fp_all, site_options.sites)
    elif reload_merged_data:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    if rerun_merge:
        (
            fp_all,
            retained_sites,
            retained_inlet,
            retained_fp_height,
            retained_instrument,
            retained_averaging_period,
        ) = data_processing_surface_notracer(
            species=species,
            sites=list(site_options.sites),
            domain=domain,
            averaging_period=list(site_options.averaging_period),
            start_date=start_date,
            end_date=end_date,
            obs_data_level=list(site_options.obs_data_level),
            platform=list(site_options.platform),
            met_model=list(site_options.met_model),
            fp_model=fp_model,
            fp_height=list(site_options.fp_height),
            fp_species=fp_species,
            emissions_name=flux_sources,
            inlet=list(site_options.inlet),
            instrument=list(site_options.instrument),
            max_level=list(site_options.max_level),
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
        site_options = site_options.retain_sites(retained_sites, context="Data gathering")
        retained_count = len(site_options.sites)
        returned_metadata = {
            "averaging_period": retained_averaging_period,
            "inlet": retained_inlet,
            "fp_height": retained_fp_height,
            "instrument": retained_instrument,
        }
        misaligned_lengths = {
            name: len(values) for name, values in returned_metadata.items() if len(values) != retained_count
        }
        if misaligned_lengths:
            raise ValueError(
                "Data gathering returned metadata with lengths that do not match retained sites; "
                f"expected {retained_count}, got {misaligned_lengths!r}."
            )

    if fp_all is None:
        raise RuntimeError("Data preparation did not create or load merged data.")
    if not site_options.sites:
        raise ValueError("No sites remain after data gathering.")
    fp_all = _select_fp_all_sites(fp_all, site_options.sites)

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

    return RhimeMergedData(
        fp_all=fp_all,
        site_options=site_options,
    )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    site_options: _SiteOptions,
    filters: Any,
) -> tuple[dict, _SiteOptions]:
    """Apply filters and keep site-aligned metadata in sync."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in site_options.sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in site_options.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        keep_indices = [index for index, site in enumerate(site_options.sites) if site not in dropped_sites]
        if not keep_indices:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        site_options = site_options.select_indices(keep_indices)
        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data, site_options


def _set_domain_attrs(fp_data: dict, sites: Sequence[str], domain: str) -> None:
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
    merged: RhimeMergedData,
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
    merged: RhimeMergedData,
    filters: Any,
) -> RhimeMergedData:
    """Filter merged RHIME data as a separate pre-basis preparation stage.

    Args:
        merged: Merged site data and site-aligned metadata from data gathering
            or reload.
        filters: Filter configuration accepted by
            :func:`openghg_inversions.filters.filtering`.

    Returns:
        Merged data containing filtered site datasets, with empty sites and
        all of their aligned options removed. If no filters are configured and
        all sites contain data, the original merged data are returned.

    Raises:
        ValueError: If every requested site is removed by filtering.
    """
    if filters is None and all(merged.fp_all[site].sizes.get("time", 0) > 0 for site in merged.sites):
        return merged

    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_data, site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    fp_all = _select_fp_all_sites({**merged.fp_all, **fp_data}, site_options.sites)
    return RhimeMergedData(fp_all=fp_all, site_options=site_options)


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
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
    min_error_options: Mapping[str, Any] | None = None,
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
        inlet: Inlet selector, either scalar or aligned to ``sites``. Entries
            may be strings, legacy ``slice`` selectors, or ``None``.
        fp_height: Footprint inlet height, either scalar or aligned to
            ``sites``.
        instrument: Observation instrument, either scalar or aligned to
            ``sites``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        obs_data_level: Observation data level, either scalar or aligned to
            ``sites``.
        met_model: Footprint meteorological model, either scalar or aligned to
            ``sites``.
        max_level: Maximum column level, either scalar or aligned to ``sites``.
            Entries must be integers or ``None``.
        min_error: Numeric minimum error or ``"residual"``/``"percentile"``
            calculation method.
        min_error_options: Calculated minimum-error options. The only supported
            key is boolean ``by_site``.
        use_tracer: Unsupported placeholder for tracer inversions, where an
            additional species constrains the primary species through linked
            forward models.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, or if minimum-error options are invalid.
    """
    min_error_options = normalise_min_error_options(min_error_options)
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
            min_error_per_site=min_error_options["by_site"],
        )
    inv_inputs = _scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=filtered_merged.sites,
        platform=filtered_merged.site_options.platform,
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
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
        ),
    )


def __getattr__(name: str) -> Any:
    """Provide warning-emitting aliases for the former fixed-basis location."""
    if name not in {"FixedBasisPreparedData", "prepare_fixedbasis_inversion_data"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"{__name__}.{name} has moved to openghg_inversions.hbmcmc.preparation; "
        "the old import path is deprecated.",
        FutureWarning,
        stacklevel=2,
    )
    from openghg_inversions.hbmcmc import preparation as fixedbasis_preparation

    return getattr(fixedbasis_preparation, name)
