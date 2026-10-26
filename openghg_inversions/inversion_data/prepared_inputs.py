"""Durable, backend-neutral prepared inputs for standard and multisector RHIME.

This module owns labelled input validation and the existing versioned artifact
schema. Construction borrows numerical arrays; explicit ``save`` and ``load``
methods own serialization and materialization. Acquisition and scientific
preparation construct these values without making them responsible for data
retrieval or model construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from typing import Self

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    open_datatree_loaded,
    save_datatree,
)

PREPARED_INPUTS_SCHEMA = "openghg_inversions.rhime_prepared_inputs"
PREPARED_INPUTS_SCHEMA_VERSION = 3
_SITE_AVERAGING_PERIOD = "averaging_period"
_SITE_STRING_ENCODING = "_rhime_nullable_string_encoding"

__all__ = [
    "PreparedInputs",
    "PREPARED_INPUTS_SCHEMA",
    "PREPARED_INPUTS_SCHEMA_VERSION",
]


@dataclass(frozen=True, init=False)
class PreparedInputs:
    """Modern RHIME preparation and durable serialization contract.

    Site labels and site-aligned metadata are owned by ``site_metadata``.
    Integer ``inv_inputs.site_indicator`` values are derived as zero-based
    project-schema positions into its ``site`` coordinate; they are distinct
    from CF compression-by-gathering indexes. The invariant is
    ``site_metadata.site[site_indicator] == nmeasure.site``.
    ``site_indicator`` and ``site_names`` remain available in memory but are
    regenerated from measurement labels and ``site_metadata`` rather than serialized.

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
        inv_inputs = prepared.inv_inputs.drop_vars(("site_indicator", "site_names"), errors="ignore")
        multiindex_dims = tuple(
            dim
            for dim in inv_inputs.dims
            if isinstance(dim, str) and isinstance(inv_inputs.indexes.get(dim), pd.MultiIndex)
        )
        if "nmeasure" not in multiindex_dims:
            raise ValueError("PreparedInputs inv_inputs requires the nmeasure MultiIndex.")
        dt = xr.DataTree.from_dict(
            {
                "inv_inputs": xr.DataTree(encode_cf_multiindexes(inv_inputs, multiindex_dims)),
                "basis_functions": prepared.basis_functions.to_datatree(),
                "site_metadata": xr.DataTree(_site_metadata_for_serialisation(site_metadata)),
            }
        )
        dt.attrs = {
            "schema": PREPARED_INPUTS_SCHEMA,
            "schema_version": PREPARED_INPUTS_SCHEMA_VERSION,
            "representation": "centered" if "H_bc_anomaly" in inv_inputs else "ordinary",
        }
        return dt

    @classmethod
    def from_datatree(cls: type[Self], dt: xr.DataTree) -> Self:
        """Restore current inputs or validated ordinary raw version-2 inputs.

        Version 3 declares ``representation`` as ``ordinary`` or ``centered``.
        Version 2 is accepted only for the ordinary raw format from the reusable
        input lifecycle. Older baked and ambiguous centered artifacts require
        regeneration with their original acquisition and preparation choices.

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
        if schema != PREPARED_INPUTS_SCHEMA:
            raise ValueError(
                f"Expected PreparedInputs schema {PREPARED_INPUTS_SCHEMA!r}, got {schema!r}."
            )

        version = dt.attrs.get("schema_version")
        if (isinstance(version, bool) or not isinstance(version, Integral)
                or version not in (2, PREPARED_INPUTS_SCHEMA_VERSION)):
            raise ValueError(
                f"Expected PreparedInputs schema_version "
                f"2 (ordinary raw) or {PREPARED_INPUTS_SCHEMA_VERSION}, got {version!r}. "
                "Older preparations may contain baked model policies; use their matching "
                "software version or regenerate the prepared inputs."
            )

        missing_nodes = [
            name for name in ("inv_inputs", "basis_functions", "site_metadata") if name not in dt.children
        ]
        if missing_nodes:
            raise KeyError(f"Missing required PreparedInputs node(s): {missing_nodes!r}.")

        encoded_inv_inputs = cast(xr.DataTree, dt["inv_inputs"]).to_dataset()
        compressed_indexes = tuple(
            name
            for name, coordinate in encoded_inv_inputs.coords.items()
            if isinstance(name, str) and coordinate.dims == (name,) and "compress" in coordinate.attrs
        )
        if "nmeasure" not in compressed_indexes:
            raise ValueError(
                "PreparedInputs inv_inputs CF gathered coordinate 'nmeasure' "
                "is missing its 'compress' metadata."
            )
        inv_inputs = decode_cf_multiindexes(encoded_inv_inputs, compressed_indexes)
        from openghg_inversions.boundary_sensitivity import validate_boundary_operands

        if version == 2:
            centered_names = {"H_bc_anomaly", "H_bc_mass", "bc_reference"}
            if (centered_names.intersection(inv_inputs.variables)
                    or "boundary_representation" in inv_inputs.attrs
                    or "representation" in dt.attrs):
                raise ValueError(
                    "Ambiguous version-2 prepared artifact; only known ordinary raw inputs are supported. "
                    "Regenerate centered inputs using the current format."
                )
            _validate_ordinary_raw_v2(inv_inputs)
        try:
            centered = validate_boundary_operands(inv_inputs)
        except ValueError as error:
            raise ValueError(f"{error} Regenerate incomplete or incompatible prepared inputs.") from error
        if version == PREPARED_INPUTS_SCHEMA_VERSION:
            representation = dt.attrs.get("representation")
            expected = "centered" if centered else "ordinary"
            if representation != expected:
                raise ValueError(
                    f"Prepared representation must be {expected!r} for its operands, got {representation!r}. "
                    "Regenerate incomplete or mismatched prepared inputs."
                )
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
    """Construct labeled site metadata in authoritative site order.

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
        raise ValueError("PreparedInputs site_metadata requires a one-dimensional 'site' coordinate.")
    sites = tuple(str(site) for site in ds["site"].values)
    if len(set(sites)) != len(sites):
        raise ValueError("PreparedInputs site_metadata 'site' labels must be unique.")
    if _SITE_AVERAGING_PERIOD not in ds:
        raise ValueError("PreparedInputs site_metadata is missing 'averaging_period'.")
    invalid_variables = [name for name, variable in ds.data_vars.items() if variable.dims != ("site",)]
    if invalid_variables:
        raise ValueError(
            "PreparedInputs site_metadata variables must each have only dimension 'site'; "
            f"invalid variables: {invalid_variables!r}."
        )
    invalid_coordinates = [
        name
        for name, coordinate in ds.coords.items()
        if name != "site" and any(dim != "site" for dim in coordinate.dims)
    ]
    if invalid_coordinates:
        raise ValueError(
            "PreparedInputs site_metadata auxiliary coordinates must be scalar or "
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
                "PreparedInputs site_metadata 'averaging_period' values must be strings or missing."
            )

    result = ds.copy()
    result = result.assign_coords(site=ds["site"].copy(data=np.asarray(sites, dtype=str)))
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(data=np.asarray(periods, dtype=object))
    return result


def _site_metadata_for_serialisation(ds: xr.Dataset) -> xr.Dataset:
    """Encode nullable strings explicitly, retaining the legacy period sentinel."""
    result = ds.copy(deep=False)
    for name, variable in ds.variables.items():
        if _SITE_STRING_ENCODING in variable.attrs:
            raise ValueError(f"Site metadata {name!r} uses reserved attribute {_SITE_STRING_ENCODING!r}.")
        if name == _SITE_AVERAGING_PERIOD or variable.dtype.kind != "O":
            continue
        values = variable.to_numpy()
        missing = pd.isna(values)
        if missing.any() and all(isinstance(value, str) for value in values[~missing]):
            encoded = np.asarray(
                [json.dumps(None if absent else str(value)) for value, absent in zip(values.flat, missing.flat)]
            ).reshape(values.shape)
            result[name] = variable.copy(data=encoded)
            result[name].attrs[_SITE_STRING_ENCODING] = "json_v1"
    result[_SITE_AVERAGING_PERIOD] = ds[_SITE_AVERAGING_PERIOD].copy(
        data=np.asarray(["" if value is None else str(value) for value in ds[_SITE_AVERAGING_PERIOD].values])
    )
    return result


def _site_metadata_from_serialisation(ds: xr.Dataset) -> xr.Dataset:
    """Restore nullable strings and legacy missing averaging periods."""
    result = ds.copy(deep=False)
    for name, variable in ds.variables.items():
        if _SITE_STRING_ENCODING not in variable.attrs:
            continue
        if variable.attrs[_SITE_STRING_ENCODING] != "json_v1":
            raise ValueError(f"Unsupported nullable string encoding for site metadata {name!r}.")
        try:
            values = [json.loads(value) for value in variable.to_numpy().flat]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid nullable string encoding for site metadata {name!r}.") from error
        if any(value is not None and not isinstance(value, str) for value in values):
            raise ValueError(f"Nullable site metadata {name!r} must contain strings or null.")
        result[name] = variable.copy(data=np.asarray(values, dtype=object).reshape(variable.shape))
        del result[name].attrs[_SITE_STRING_ENCODING]
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
    from openghg_inversions.boundary_sensitivity import validate_boundary_operands

    validate_boundary_operands(ds)
    nmeasure_index = ds.indexes.get("nmeasure")
    if not isinstance(nmeasure_index, pd.MultiIndex):
        raise ValueError(  # noqa: TRY004 - malformed artifact schema, not an argument type error
            "PreparedInputs inv_inputs must have a 'nmeasure' MultiIndex with levels ('site', 'time')."
        )
    if tuple(nmeasure_index.names) != ("site", "time"):
        raise ValueError(
            "PreparedInputs inv_inputs 'nmeasure' MultiIndex must have levels "
            f"('site', 'time'), got {tuple(nmeasure_index.names)!r}."
        )
    measurement_sites_raw = tuple(nmeasure_index.get_level_values("site"))
    if not all(isinstance(site, str) for site in measurement_sites_raw):
        raise ValueError("PreparedInputs nmeasure site labels must all be strings.")
    observed_sites = tuple(dict.fromkeys(measurement_sites_raw))
    metadata_sites = tuple(site_metadata["site"].values.tolist())
    missing_sites = [site for site in observed_sites if site not in metadata_sites]
    if missing_sites:
        raise ValueError(
            f"PreparedInputs site_metadata is missing observed site labels: {missing_sites!r}."
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
        raise ValueError("PreparedInputs multi-source basis labels must be unique.")
    if "H" not in result:
        raise ValueError("PreparedInputs inv_inputs must contain H for a multi-source basis.")
    sensitivity = result["H"]
    state_dim: str | None = None
    if "source" in sensitivity.dims:
        actual_sources = tuple(str(source) for source in sensitivity["source"].values)
        if len(set(actual_sources)) != len(actual_sources):
            raise ValueError("PreparedInputs inv_inputs H source labels must be unique.")
    else:
        gathered_dims = [
            str(dim)
            for dim in sensitivity.dims
            if isinstance(sensitivity.indexes.get(dim), pd.MultiIndex)
            and "source" in sensitivity.indexes[dim].names
        ]
        if len(gathered_dims) != 1:
            raise ValueError(
                "PreparedInputs inv_inputs H must have either a source dimension or "
                "one gathered state MultiIndex containing a 'source' level for a multi-source basis."
            )
        state_dim = gathered_dims[0]
        state_index = sensitivity.indexes[state_dim]
        source_values = state_index.get_level_values("source").to_numpy()
        if not all(isinstance(source, str | np.str_) for source in source_values):
            raise ValueError("PreparedInputs gathered H source labels must all be strings.")
        source_labels = tuple(str(source) for source in source_values)
        _, first_indices = np.unique(source_values, return_index=True)
        actual_sources = tuple(str(source_values[index]) for index in np.sort(first_indices))
    if set(actual_sources) != set(expected_sources):
        missing = sorted(set(expected_sources) - set(actual_sources))
        unexpected = sorted(set(actual_sources) - set(expected_sources))
        raise ValueError(
            "PreparedInputs inv_inputs H source labels do not match the basis operator: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    if state_dim is not None:
        state_order = [
            index
            for source in expected_sources
            for index, actual_source in enumerate(source_labels)
            if actual_source == source
        ]
        state_indexers = {state_dim: state_order}
        covariance_dim = f"{state_dim}_cov"
        if covariance_dim in result.dims:
            if result.sizes[covariance_dim] != len(state_order):
                raise ValueError(
                    f"PreparedInputs {covariance_dim} must match the gathered state length."
                )
            state_indexers[covariance_dim] = state_order
        return result.isel(state_indexers), site_metadata
    result = result.assign_coords(source=("source", list(actual_sources)))
    return result.sel(source=list(expected_sources)), site_metadata


def _validate_ordinary_raw_v2(inputs: xr.Dataset) -> None:
    """Accept only the known ordinary spatial boundary layout of version 2."""
    if "H_bc" not in inputs:
        return
    boundary = inputs["H_bc"]
    if (set(boundary.dims) != {"bc_region", "nmeasure"}
            or isinstance(boundary.indexes.get("bc_region"), pd.MultiIndex)
            or "bc_period" in boundary.coords
            or boundary.attrs.get("boundary_sensitivity_preparation") != "raw-spatial"
            or "bc_frequency" in boundary.attrs or "bc_anchor_time" in boundary.attrs):
        raise ValueError(
            "Version-2 prepared inputs require ordinary raw spatial H_bc; baked or unknown "
            "boundary layouts are unsupported. Regenerate inputs with the current preparation API."
        )
