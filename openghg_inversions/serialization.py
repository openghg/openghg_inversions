"""Shared serialization helpers for modern OpenGHG inversion artifacts.

This module contains the storage mechanics shared by modern artifact
containers. It saves and eagerly loads xarray ``DataTree`` objects, converts
ArviZ ``InferenceData`` groups to and from trees, and expands pandas
``MultiIndex`` coordinates into representations supported by NetCDF and Zarr.

Object-specific modules remain responsible for schema names, schema versions,
required child nodes, and metadata validation. Prepared artifacts can use the
CF compression-by-gathering convention through explicit encoding and decoding
helpers. The older project-specific MultiIndex restoration remains deliberately
forgiving so existing or partially malformed inversion-output artifacts remain
loadable without reintroducing invalid indexes. Versioned schemas may request
strict restoration, which rejects missing, malformed, empty, or inapplicable
restoration metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import arviz as az
from cf_xarray.coding import decode_compress_to_multi_index, encode_multi_index_as_compress
import pandas as pd
import xarray as xr


MULTIINDEX_DIMS_ATTR = "openghg_inversions:multiindex_dims"
MULTIINDEX_SCHEMA_VERSION = 1


def _validate_multiindex(
    index: pd.MultiIndex,
    *,
    dim: str,
    level_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate the semantic identity carried by a MultiIndex."""
    names = tuple(index.names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"MultiIndex {dim!r} must have non-empty string level names.")
    if len(set(names)) != len(names):
        raise ValueError(f"MultiIndex {dim!r} has duplicate level names.")

    if level_names is not None:
        expected = _normalise_cf_multiindex_names(level_names)
        if names != expected:
            raise ValueError(
                f"MultiIndex {dim!r} has levels {list(names)!r}; expected {list(expected)!r} in that order."
            )

    if not index.is_unique:
        duplicate = index[index.duplicated()][0]
        raise ValueError(f"MultiIndex {dim!r} contains duplicate label {duplicate!r}.")
    return cast(tuple[str, ...], names)


def normalise_declared_multiindex(
    ds: xr.Dataset,
    dim: str,
    level_names: str | Iterable[str],
) -> xr.Dataset:
    """Normalize a MultiIndex or its declared expanded form.

    The owning dimension and ordered semantic level names are explicit.  An
    already-indexed dataset is validated in place; an expanded representation
    is reconstructed only when every level is one-dimensional on ``dim`` and
    the resulting labels are unique.
    """
    levels = _normalise_cf_multiindex_names(level_names)
    if dim not in ds.dims:
        raise ValueError(f"MultiIndex dimension {dim!r} is missing from the Dataset.")

    index = ds.indexes.get(dim)
    if isinstance(index, pd.MultiIndex):
        _validate_multiindex(index, dim=dim, level_names=levels)
        return ds

    missing = [level for level in levels if level not in ds.coords]
    if missing:
        raise ValueError(f"MultiIndex {dim!r} is missing level coordinate(s) {missing!r}.")
    for level in levels:
        coordinate = ds[level]
        if coordinate.dims != (dim,):
            raise ValueError(f"MultiIndex level coordinate {level!r} must be one-dimensional on {dim!r}.")
        if coordinate.sizes[dim] != ds.sizes[dim]:
            raise ValueError(f"MultiIndex level coordinate {level!r} has a length inconsistent with {dim!r}.")

    multiindex = pd.MultiIndex.from_arrays(
        [ds[level].values for level in levels],
        names=levels,
    )
    _validate_multiindex(multiindex, dim=dim, level_names=levels)

    # Install the index explicitly so xarray does not rely on deprecated
    # implicit pandas.MultiIndex promotion.
    result = ds.drop_vars(list(levels))
    return result.assign_coords(xr.Coordinates.from_pandas_multiindex(multiindex, dim))


def _normalise_cf_multiindex_names(index_names: str | Iterable[str]) -> tuple[str, ...]:
    """Return validated, deterministic names for explicit CF codec calls."""
    names = (index_names,) if isinstance(index_names, str) else tuple(index_names)
    if not names:
        raise ValueError("At least one MultiIndex name must be provided.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("MultiIndex names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("MultiIndex names must be unique.")
    return names


def encode_cf_multiindexes(ds: xr.Dataset, index_names: str | Iterable[str]) -> xr.Dataset:
    """Encode named MultiIndexes using CF compression by gathering.

    This codec is intended for versioned prepared-input artifacts. Unlike
    :func:`reset_serialisation_multiindexes`, it emits the interoperable CF
    ``compress`` attribute and requires callers to name every index they intend
    to encode.

    Args:
        ds: Dataset containing the MultiIndex dimensions.
        index_names: Explicit dimension name or names to encode.

    Returns:
        A new Dataset with the requested MultiIndexes encoded as integer
        gathering coordinates.

    Raises:
        ValueError: If no names are supplied, a requested dimension is not a
            named MultiIndex, level names are invalid or ambiguous, or existing
            ``compress`` metadata would be overwritten.
    """
    names = _normalise_cf_multiindex_names(index_names)
    all_level_names: set[str] = set()

    for name in names:
        if name not in ds.dims:
            raise ValueError(f"CF MultiIndex dimension {name!r} is missing from the Dataset.")

        index = ds.indexes.get(name)
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(f"Dimension {name!r} is not backed by a pandas MultiIndex.")

        level_names = _validate_multiindex(index, dim=name)

        repeated_level_names = all_level_names.intersection(level_names)
        if repeated_level_names:
            repeated = ", ".join(repr(level_name) for level_name in sorted(repeated_level_names))
            raise ValueError(f"CF MultiIndexes cannot share level names; repeated names: {repeated}.")
        all_level_names.update(level_names)

        coordinate = ds[name]
        if "compress" in coordinate.attrs or "compress" in coordinate.encoding:
            raise ValueError(f"MultiIndex {name!r} already has CF 'compress' metadata.")

    try:
        return encode_multi_index_as_compress(ds, idxnames=names)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not encode CF MultiIndexes {names!r}: {exc}") from exc


def decode_cf_multiindexes(ds: xr.Dataset, index_names: str | Iterable[str]) -> xr.Dataset:
    """Decode named CF gathering coordinates into pandas MultiIndexes.

    The requested coordinates and their ``compress`` metadata are validated
    before calling ``cf_xarray``. Names are always explicit: this deliberately
    avoids ``cf_xarray``'s auto-detection path, which can drop unrelated
    coordinates when no compressed indexes are present.

    Args:
        ds: Dataset containing CF compression-by-gathering coordinates.
        index_names: Explicit gathered-coordinate name or names to decode.

    Returns:
        A new Dataset with the requested coordinates decoded as MultiIndexes.

    Raises:
        ValueError: If no names are supplied or a requested coordinate has
            missing, malformed, ambiguous, or inapplicable CF ``compress``
            metadata.
    """
    names = _normalise_cf_multiindex_names(index_names)
    normalised = ds
    all_level_names: set[str] = set()

    for name in names:
        if name not in ds.coords or name not in ds.dims:
            raise ValueError(f"CF gathered coordinate {name!r} is missing from the Dataset.")

        coordinate = ds[name]
        if coordinate.dims != (name,):
            raise ValueError(
                f"CF gathered coordinate {name!r} must be one-dimensional on dimension {name!r}."
            )
        if not pd.api.types.is_integer_dtype(coordinate.dtype):
            raise ValueError(f"CF gathered coordinate {name!r} must contain integer indices.")

        compress = coordinate.attrs.get("compress")
        if not isinstance(compress, str) or not compress.strip():
            raise ValueError(
                f"CF gathered coordinate {name!r} requires a non-empty string 'compress' attribute."
            )

        level_names = tuple(compress.split())
        if len(set(level_names)) != len(level_names):
            raise ValueError(f"CF gathered coordinate {name!r} has duplicate names in 'compress'.")

        repeated_level_names = all_level_names.intersection(level_names)
        if repeated_level_names:
            repeated = ", ".join(repr(level_name) for level_name in sorted(repeated_level_names))
            raise ValueError(f"CF gathered coordinates cannot share level names; repeated names: {repeated}.")
        all_level_names.update(level_names)

        for level_name in level_names:
            if level_name not in ds.coords:
                raise ValueError(
                    f"CF gathered coordinate {name!r} references missing level coordinate {level_name!r}."
                )
            if ds[level_name].dims != (level_name,):
                raise ValueError(
                    f"CF level coordinate {level_name!r} must be one-dimensional on its own dimension."
                )
            if ds.sizes[level_name] == 0:
                raise ValueError(f"CF level coordinate {level_name!r} cannot be empty.")

        canonical_compress = " ".join(level_names)
        if compress != canonical_compress:
            if normalised is ds:
                normalised = ds.copy()
            normalised[name].attrs = dict(normalised[name].attrs)
            normalised[name].attrs["compress"] = canonical_compress

    try:
        decoded = decode_compress_to_multi_index(normalised, idxnames=names)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not decode CF MultiIndexes {names!r}: {exc}") from exc

    for name in names:
        index = decoded.indexes.get(name)
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(f"Decoded CF coordinate {name!r} is not a pandas MultiIndex.")
        compress = normalised[name].attrs["compress"]
        _validate_multiindex(index, dim=name, level_names=compress.split())

    missing_coords = {
        name: coordinate
        for name, coordinate in normalised.coords.items()
        if name not in decoded.coords and all(dim not in all_level_names for dim in coordinate.dims)
    }
    if missing_coords:
        decoded = decoded.assign_coords(missing_coords)
    return decoded


def save_datatree(
    dt: xr.DataTree,
    output_file: str | Path,
    output_format: Literal["netcdf", "zarr"] | None = None,
) -> None:
    """Save a DataTree to NetCDF or Zarr.

    This writes the tree, replacing an existing destination artifact.

    Args:
        dt: DataTree to persist.
        output_file: Requested artifact path. When ``output_format`` is
            supplied, the matching ``.nc`` or ``.zarr`` suffix is added or
            replaces an existing suffix as needed.
        output_format: Storage format. When omitted, infer the format from an
            existing ``.nc`` or ``.zarr`` suffix.

    Raises:
        ValueError: If the format cannot be inferred or is unsupported.
    """
    output_path = Path(output_file)

    if output_format is None:
        try:
            output_format = {".nc": "netcdf", ".zarr": "zarr"}[output_path.suffix]  # type: ignore[assignment]
        except KeyError:
            raise ValueError(
                f"Output file {output_path} does not end in '.nc' or '.zarr'; please specify `output_format`."
            ) from None

    if output_format == "netcdf":
        if output_path.suffix != ".nc":
            output_path = output_path.with_suffix(".nc")
        dt.to_netcdf(output_path)
    elif output_format == "zarr":
        if output_path.suffix != ".zarr":
            output_path = output_path.with_suffix(".zarr")
        dt.to_zarr(output_path, mode="w")
    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")


def open_datatree_loaded(file_path: str | Path) -> xr.DataTree:
    """Open and eagerly load a DataTree artifact.

    NetCDF is first attempted with ``h5netcdf`` for compatibility with modern
    inversion outputs, then with xarray's default engine. Loading happens while
    the file context is open, so the returned tree owns its data and does not
    retain references to closed file handles.

    Args:
        file_path: NetCDF or Zarr DataTree artifact to load.

    Returns:
        Fully loaded DataTree.

    Raises:
        OSError: If no backend can open the artifact.
        RuntimeError: If opening fails due to a backend runtime error.
        ValueError: If no backend can interpret the artifact.
    """
    open_errors: list[Exception] = []
    for engine in ("h5netcdf", None):
        try:
            dt = (
                xr.open_datatree(file_path, engine=engine)
                if engine is not None
                else xr.open_datatree(file_path)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            open_errors.append(exc)
        else:
            with dt:
                return dt.load()

    raise open_errors[-1]


def inferencedata_to_datatree(idata: az.InferenceData) -> xr.DataTree:
    """Convert ArviZ InferenceData groups to a serializable DataTree.

    Args:
        idata: InferenceData whose root attributes and groups should become
            the tree root and child nodes.

    Returns:
        DataTree containing the InferenceData root attributes and one child
        dataset per group.
    """
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=dict(idata.attrs)),
            **{group: reset_serialisation_multiindexes(idata[group]) for group in idata.groups()},
        }
    )


def inferencedata_from_datatree(dt: xr.DataTree) -> az.InferenceData:
    """Reconstruct ArviZ InferenceData from a group DataTree.

    Args:
        dt: DataTree containing root attributes and one child dataset per
            InferenceData group.

    Returns:
        Reconstructed InferenceData with root attributes and valid serialized
        MultiIndexes restored.
    """
    return cast(Any, az.InferenceData)(
        attrs=dict(dt.attrs),
        **{group: restore_serialisation_multiindexes(child.to_dataset()) for group, child in dt.items()},
    )


def save_inferencedata(
    idata: az.InferenceData,
    output_file: str | Path,
    output_format: Literal["netcdf", "zarr"] | None = None,
) -> None:
    """Save InferenceData through the declared MultiIndex boundary.

    Args:
        idata: InferenceData whose groups and root attributes should be saved.
        output_file: Destination NetCDF file or Zarr store.
        output_format: Explicit backend, or ``None`` to infer it from the path.
    """
    save_datatree(inferencedata_to_datatree(idata), output_file, output_format)


def load_inferencedata(file_path: str | Path) -> az.InferenceData:
    """Load InferenceData and restore every valid declared MultiIndex.

    Malformed declarations are removed and left expanded rather than guessed,
    matching :func:`restore_declared_multiindexes`' forgiving default.

    Args:
        file_path: NetCDF file or Zarr store written by
            :func:`save_inferencedata`.

    Returns:
        Fully loaded InferenceData with valid semantic indexes reconstructed.
    """
    return inferencedata_from_datatree(open_datatree_loaded(file_path))


def encode_multiindexes_for_storage(ds: xr.Dataset) -> xr.Dataset:
    """Expand all semantic MultiIndexes and attach versioned schema metadata.

    Args:
        ds: Dataset containing zero or more pandas MultiIndex dimensions.

    Returns:
        A serialization copy with ordinary level coordinates and declarations
        of their owner, order, uniqueness, and reconstruction policy.

    Raises:
        ValueError: If an index has missing, repeated, or duplicate semantic
            labels.
    """
    result = ds
    records: list[dict[str, object]] = []
    for dim, index in ds.indexes.items():
        if dim not in result.dims or not isinstance(index, pd.MultiIndex):
            continue
        dim_name = str(dim)
        level_names = _validate_multiindex(index, dim=dim_name)
        result = result.reset_index(dim)
        records.append(
            {
                "dim": dim_name,
                "levels": list(level_names),
                "reconstruct": True,
                "unique": True,
                "order": "preserve",
            }
        )

    if records:
        result = result.copy()
        result.attrs = dict(result.attrs)
        result.attrs[MULTIINDEX_DIMS_ATTR] = json.dumps(
            {"version": MULTIINDEX_SCHEMA_VERSION, "dims": records}
        )
    return result


def reset_serialisation_multiindexes(ds: xr.Dataset) -> xr.Dataset:
    """Expand xarray MultiIndexes before DataTree serialization.

    Args:
        ds: Dataset that may contain pandas MultiIndex dimensions.

    Returns:
        Dataset with MultiIndexes reset and restoration metadata stored in an
        attribute. The input dataset is not modified.

    Raises:
        ValueError: If a MultiIndex level is unnamed and cannot be restored
            unambiguously.
    """
    return encode_multiindexes_for_storage(ds)


def restore_serialisation_multiindexes(ds: xr.Dataset, *, strict: bool = False) -> xr.Dataset:
    """Restore valid MultiIndexes expanded during DataTree serialization.

    In the default non-strict mode, malformed restoration metadata is discarded
    instead of raising. This is a compatibility guarantee for existing
    ``InversionOutput`` artifacts: valid index records are restored, while
    invalid records leave their level coordinates expanded. Strict mode raises
    for missing, empty, malformed, or inapplicable records.

    Args:
        ds: Dataset that may carry ``MULTIINDEX_DIMS_ATTR`` metadata.
        strict: Raise for malformed restoration metadata instead of leaving
            index levels expanded. Strict restoration also requires at least
            one recorded MultiIndex dimension. The default preserves forgiving
            historical inversion-output loading.

    Returns:
        Dataset with valid recorded MultiIndexes restored and the private
        serialization attribute removed. The input dataset is not modified.

    Raises:
        ValueError: If ``strict`` is true and restoration metadata is malformed
            or cannot be applied to the dataset.
    """
    raw_multiindex_dims = ds.attrs.get(MULTIINDEX_DIMS_ATTR)
    if raw_multiindex_dims is None:
        if strict:
            raise ValueError("MultiIndex metadata is missing.")
        return ds

    result = ds.copy()
    result.attrs = dict(result.attrs)
    del result.attrs[MULTIINDEX_DIMS_ATTR]

    if isinstance(raw_multiindex_dims, bytes):
        try:
            raw_multiindex_dims = raw_multiindex_dims.decode()
        except UnicodeDecodeError:
            if strict:
                raise ValueError("MultiIndex metadata is not valid UTF-8.") from None
            return result
    if not isinstance(raw_multiindex_dims, str):
        if strict:
            raise ValueError("MultiIndex metadata must be a JSON string.")
        return result

    try:
        payload = json.loads(raw_multiindex_dims)
    except json.JSONDecodeError:
        if strict:
            raise ValueError("MultiIndex metadata is not valid JSON.") from None
        return result

    if isinstance(payload, dict):
        version = payload.get("version")
        if version is not None and version != MULTIINDEX_SCHEMA_VERSION:
            if strict:
                raise ValueError(f"Unsupported MultiIndex metadata version {version!r}.")
            return result

    records = payload.get("dims") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        if strict:
            raise ValueError("MultiIndex metadata must contain a 'dims' list.")
        return result
    if strict and not records:
        raise ValueError("MultiIndex metadata must contain at least one dimension record.")

    for record in records:
        if not isinstance(record, dict):
            if strict:
                raise ValueError("Each MultiIndex metadata record must be an object.")
            continue
        dim = record.get("dim")
        levels = record.get("levels")
        if not isinstance(dim, str) or not isinstance(levels, list):
            if strict:
                raise ValueError("MultiIndex records require string 'dim' and list 'levels' values.")
            continue
        if not levels or not all(isinstance(level, str) and level for level in levels):
            if strict:
                raise ValueError("MultiIndex level names must be non-empty strings.")
            continue
        if len(set(levels)) != len(levels):
            if strict:
                raise ValueError(f"MultiIndex metadata for dimension {dim!r} has duplicate levels.")
            continue

        reconstruct = record.get("reconstruct", True)
        unique = record.get("unique", True)
        order = record.get("order", "preserve")
        if reconstruct is not True or unique is not True or order != "preserve":
            if strict:
                raise ValueError(
                    f"MultiIndex metadata for dimension {dim!r} has unsupported semantic expectations."
                )
            continue
        try:
            result = normalise_declared_multiindex(result, dim, levels)
        except ValueError:
            if strict:
                raise
    return result


def restore_declared_multiindexes(ds: xr.Dataset, *, strict: bool = False) -> xr.Dataset:
    """Restore MultiIndexes from explicit storage declarations.

    Args:
        ds: Dataset carrying expanded coordinates and
            :data:`MULTIINDEX_DIMS_ATTR` metadata.
        strict: Raise a focused error for invalid metadata or semantic labels.
            By default, invalid declarations are removed and their coordinates
            remain expanded.

    Returns:
        Dataset with every valid declared MultiIndex restored explicitly.
    """
    return restore_serialisation_multiindexes(ds, strict=strict)
