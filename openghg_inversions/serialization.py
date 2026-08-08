"""Shared serialization helpers for modern OpenGHG inversion artifacts.

This module contains the storage mechanics shared by modern artifact
containers. It saves and eagerly loads xarray ``DataTree`` objects, prepares
inference trace groups for storage, and expands pandas
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
from typing import Iterable, Literal

from cf_xarray.coding import decode_compress_to_multi_index, encode_multi_index_as_compress
import pandas as pd
import xarray as xr


MULTIINDEX_DIMS_ATTR = "openghg_inversions:multiindex_dims"


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

        level_names = tuple(index.names)
        if any(not isinstance(level_name, str) or not level_name for level_name in level_names):
            raise ValueError(f"MultiIndex {name!r} must have non-empty string level names.")
        if len(set(level_names)) != len(level_names):
            raise ValueError(f"MultiIndex {name!r} has duplicate level names.")

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
        dt.to_zarr(str(output_path), mode="w")  # pyright: ignore[reportArgumentType]
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
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            open_errors.append(exc)
        else:
            with dt:
                return dt.load()

    raise open_errors[-1]


def inferencedata_to_datatree(trace: xr.DataTree) -> xr.DataTree:
    """Prepare inference trace groups as a serializable DataTree.

    Args:
        trace: DataTree whose direct children are inference groups.

    Returns:
        DataTree containing the trace root attributes and one child dataset per
        group, with MultiIndexes expanded for storage.
    """
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=dict(trace.attrs)),
            **{
                group: reset_serialisation_multiindexes(child.to_dataset())
                for group, child in trace.children.items()
            },
        }
    )


def inferencedata_from_datatree(dt: xr.DataTree) -> xr.DataTree:
    """Restore an inference trace from a serialized group DataTree.

    Args:
        dt: DataTree containing root attributes and one child dataset per
            InferenceData group.

    Returns:
        Reconstructed DataTree with root attributes and valid serialized
        MultiIndexes restored in each direct child.
    """
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=dict(dt.attrs)),
            **{
                group: restore_serialisation_multiindexes(child.to_dataset())
                for group, child in dt.children.items()
            },
        }
    )


def load_trace(file_path: str | Path) -> xr.DataTree:
    """Load a standalone inference trace written by ArviZ or xarray.

    Legacy ``InferenceData.to_netcdf`` files and DataTree-native trace files
    share the same root-group layout. Complete ``InversionOutput`` artifacts
    must instead be opened with ``InversionOutput.load``.

    Args:
        file_path: Standalone trace NetCDF or Zarr path.

    Returns:
        Eagerly loaded DataTree with serialized MultiIndexes restored.

    Raises:
        ValueError: If the path contains a complete inversion artifact rather
            than a standalone trace.
    """
    dt = open_datatree_loaded(file_path)
    if dt.attrs.get("schema") == "openghg_inversions.InversionOutput" or {
        "trace",
        "inv_inputs",
        "basis_functions",
    }.issubset(dt.children):
        raise ValueError(
            "Expected a standalone trace artifact; use InversionOutput.load() "
            "for a complete inversion output."
        )
    return inferencedata_from_datatree(dt)


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
    result = ds
    multiindex_dims: list[dict[str, object]] = []
    for dim, index in ds.indexes.items():
        if dim in result.dims and isinstance(index, pd.MultiIndex):
            level_names = list(index.names)
            if any(name is None for name in level_names):
                raise ValueError(f"Cannot serialise unnamed MultiIndex levels for dimension {dim!r}.")
            result = result.reset_index(dim)
            multiindex_dims.append({"dim": str(dim), "levels": [str(name) for name in level_names]})
    if multiindex_dims:
        result = result.copy()
        result.attrs = dict(result.attrs)
        result.attrs[MULTIINDEX_DIMS_ATTR] = json.dumps({"dims": multiindex_dims})
    return result


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
        if not all(isinstance(level, str) for level in levels):
            if strict:
                raise ValueError("MultiIndex level names must be strings.")
            continue
        if dim in result.dims and all(level in result and result[level].dims == (dim,) for level in levels):
            result = result.set_index({dim: levels})
        elif strict:
            raise ValueError(
                f"MultiIndex metadata for dimension {dim!r} does not align with stored level coordinates."
            )
    return result
