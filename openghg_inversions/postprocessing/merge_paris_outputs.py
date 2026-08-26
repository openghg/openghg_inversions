"""Merge sequential PARIS NetCDF outputs without losing template coordinates."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

ParisOutputType = Literal["flux", "concentration"]
ParisTemplateVersion = Literal["legacy", "latest"]

_PLATFORM_IDENTIFIER = "_platform_identifier"


def _output_type(ds: xr.Dataset) -> ParisOutputType:
    if {"latitude", "longitude"} <= set(ds.dims):
        return "flux"
    if "index" in ds.dims or {"time", "nsite"} <= set(ds.dims):
        return "concentration"
    raise ValueError("Could not identify the input as a PARIS flux or concentration output.")


def _template_version(ds: xr.Dataset, output_type: ParisOutputType) -> ParisTemplateVersion:
    if output_type == "concentration":
        return "latest" if "index" in ds.dims else "legacy"
    latest_names = {"flux_total_prior_country", "flux_total_posterior_country"}
    return "latest" if latest_names & set(ds.data_vars) else "legacy"


def _repair_duplicate_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """Give repeated covariance axes distinct names before xarray operations."""
    result = ds
    for name in list(ds.data_vars):
        variable = ds[name].variable
        if len(variable.dims) == len(set(variable.dims)):
            continue

        counts: dict[str, int] = {}
        dimensions = []
        repeated_coordinates: dict[str, tuple[str, np.ndarray]] = {}
        for dimension in variable.dims:
            counts[dimension] = counts.get(dimension, 0) + 1
            replacement = dimension if counts[dimension] == 1 else f"{dimension}_{counts[dimension]}"
            dimensions.append(replacement)
            if replacement != dimension:
                values = (
                    np.asarray(ds[dimension].values)
                    if dimension in ds.coords
                    else np.arange(ds.sizes[dimension])
                )
                repeated_coordinates[replacement] = (replacement, values)

        attrs = dict(variable.attrs)
        encoding = dict(variable.encoding)
        data = variable.data
        result = result.drop_vars(name)
        result[name] = xr.DataArray(data, dims=dimensions, attrs=attrs)
        result[name].encoding = encoding
        result = result.assign_coords(repeated_coordinates)

        if name == "covariance_flux_sectors_posterior_country":
            result[name] = result[name].transpose("sector_2", "sector", "country", "time")
        elif name.startswith("covariance_flux_") and name.endswith("_country"):
            result[name] = result[name].transpose("country", "country_2", "time")

    return result


def _prepare_legacy_concentration(ds: xr.Dataset) -> xr.Dataset:
    if "sitenames" not in ds.coords or ds["sitenames"].dims != ("nsite",):
        raise ValueError("Legacy PARIS concentration outputs require sitenames(nsite).")
    return ds.rename_vars(sitenames="nsite").set_xindex("nsite")


def _restore_legacy_concentration(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename_vars(nsite="sitenames")


def _prepare_latest_concentration(ds: xr.Dataset) -> xr.Dataset:
    if "platform" not in ds.coords or "number_of_identifier" not in ds:
        if "site" in ds.coords and ds["site"].dims == ("index",):
            identifiers = np.asarray(ds["site"].values, dtype=str)
            return ds.drop_vars("site").assign_coords(
                {_PLATFORM_IDENTIFIER: ("index", identifiers)}
            )
        raise ValueError(
            "Latest PARIS concentration outputs require platform(platform) and "
            "number_of_identifier(index)."
        )

    platforms = np.asarray(ds["platform"].values, dtype=str)
    indices = np.asarray(ds["number_of_identifier"].values)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("number_of_identifier must contain integer platform indices.")
    if np.any(indices < 0) or np.any(indices >= len(platforms)):
        raise ValueError("number_of_identifier contains an out-of-range platform index.")

    identifiers = platforms[indices]
    return ds.drop_vars(["platform", "number_of_identifier", "site"], errors="ignore").assign_coords(
        {_PLATFORM_IDENTIFIER: ("index", identifiers)}
    )


def _restore_latest_concentration(ds: xr.Dataset, source: xr.Dataset) -> xr.Dataset:
    identifiers = np.asarray(ds[_PLATFORM_IDENTIFIER].values, dtype=str)
    platforms = list(dict.fromkeys(identifiers))
    platform_indices = {platform: index for index, platform in enumerate(platforms)}

    result = ds.drop_vars(_PLATFORM_IDENTIFIER)
    result["number_of_identifier"] = (
        "index",
        np.asarray([platform_indices[value] for value in identifiers], dtype="int16"),
    )
    result = result.assign_coords(platform=("platform", np.asarray(platforms, dtype=object)))

    if "number_of_identifier" in source:
        result["number_of_identifier"].attrs = dict(source["number_of_identifier"].attrs)
    if "platform" in source:
        result["platform"].attrs = dict(source["platform"].attrs)
    return result


def merge_paris_outputs(
    input_files: Sequence[str | Path],
    output_file: str | Path,
    *,
    output_type: ParisOutputType | None = None,
) -> Path:
    """Merge PARIS files from one template version along their observation time axis.

    The template version is detected from the dataset schema. When ``output_type``
    is supplied, inputs of the other type are ignored so broad shell globs can be
    reused. Selected inputs must share one template version; legacy and latest
    files cannot be mixed in one output because their variable contracts differ.
    """
    paths = [Path(path) for path in input_files]
    if len(paths) < 2:
        raise ValueError("At least two PARIS input files are required.")

    output_path = Path(output_file)
    with ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="Duplicate dimension names present")
        datasets = [stack.enter_context(xr.open_dataset(path)) for path in paths]
        types = [_output_type(ds) for ds in datasets]
        if output_type is not None:
            selected = [
                ds for ds, kind in zip(datasets, types) if kind == output_type
            ]
            if not selected:
                raise ValueError(f"No {output_type} outputs were found in the input files.")
            if len(selected) < 2:
                raise ValueError(f"At least two {output_type} output files are required.")
            datasets = selected
            detected_type = output_type
        else:
            detected_types = set(types)
            if len(detected_types) != 1:
                flux_files = [path.name for path, kind in zip(paths, types) if kind == "flux"]
                concentration_files = [
                    path.name for path, kind in zip(paths, types) if kind == "concentration"
                ]
                raise ValueError(
                    "PARIS flux and concentration files cannot be merged together. "
                    f"Flux files: {flux_files}; concentration files: {concentration_files}. "
                    "Pass --type flux or --type concentration to select one product from a broad glob."
                )
            detected_type = detected_types.pop()

        versions = {_template_version(ds, detected_type) for ds in datasets}
        if len(versions) != 1:
            raise ValueError("PARIS files from different template versions cannot be mixed in one output.")
        version = versions.pop()

        repaired = [_repair_duplicate_dimensions(ds) for ds in datasets]
        if detected_type == "concentration" and version == "legacy":
            prepared = [_prepare_legacy_concentration(ds) for ds in repaired]
            concat_dim = "time"
        elif detected_type == "concentration":
            prepared = [_prepare_latest_concentration(ds) for ds in repaired]
            concat_dim = "index"
        else:
            prepared = repaired
            concat_dim = "time"

        country_fraction = None
        if detected_type == "flux" and "country_fraction" in prepared[0]:
            country_fraction = prepared[0]["country_fraction"]
            prepared = [ds.drop_vars("country_fraction", errors="ignore") for ds in prepared]

        merged = xr.concat(
            prepared,
            dim=concat_dim,
            data_vars="minimal",
            coords="minimal",
            compat="equals",
            join="outer",
        ).sortby("time")

        if detected_type == "concentration" and version == "legacy":
            merged = _restore_legacy_concentration(merged)
        elif detected_type == "concentration":
            merged = _restore_latest_concentration(merged, datasets[0])
        if country_fraction is not None:
            merged["country_fraction"] = country_fraction

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(output_path)

    return output_path
