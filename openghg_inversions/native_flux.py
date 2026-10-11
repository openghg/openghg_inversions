"""Resolve native prior flux separately from observation-aligned responses."""

from collections.abc import Mapping

import xarray as xr
from openghg.analyse._utils import match_dataset_dims, stack_datasets

from openghg_inversions.array_ops import force_align
from openghg_inversions.flux_sanitization import sanitize_flux_nonfinite


def resolve_runtime_flux(
    flux_data: Mapping[str, xr.Dataset], *, split_by_sectors: bool = False
) -> xr.DataArray:
    """Combine or stack all borrowed runtime source fluxes in mapping order.

    Native prior flux retains its inventory time axis, independently of
    observation-aligned footprint-times-flux responses. Borrowed inputs are
    unchanged and Dask-backed payloads remain lazy. Non-finite values are
    replaced with zero using the retained-basis sanitation policy.

    Args:
        flux_data: Non-empty mapping of source labels to datasets containing
            ``flux`` on the native spatial grid, optionally with ``time``.
            Sources must already use compatible flux units.
        split_by_sectors: If true, retain every source along ``source`` in
            mapping insertion order. Sources must have identical dimensions
            and either all be timeless or share an explicit unique time index.
            Spatial coordinates are matched to the first source within
            :func:`~openghg_inversions.array_ops.force_align` tolerances.
            Otherwise combine sources using the OpenGHG ``ModelScenario``
            convention: sum timeless fields, or forward-fill timed fields onto
            a common time axis and sum them.

    Returns:
        Native prior flux containing all supplied sources, separately when
        ``split_by_sectors=True`` and combined otherwise.

    Raises:
        KeyError: If a source dataset has no ``flux`` variable.
        ValueError: If the mapping is empty or sector-resolved source labels,
            dimensions, time indexes, or spatial coordinates are incompatible.
    """
    if not flux_data:
        raise ValueError("Cannot construct BasisFunctions object: runtime flux is missing or empty.")
    flux_arrays = {key: value["flux"] for key, value in flux_data.items()}
    if split_by_sectors:
        flux = _stack_flux_sources_with_alignment(flux_arrays)
    else:
        flux = _combine_flux_sources_like_modelscenario(flux_arrays)
    return sanitize_flux_nonfinite(
        flux, context="retained basis runtime flux", trust_attrs=len(flux_arrays) == 1,
    )


def _combine_flux_sources_like_modelscenario(flux_arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Combine source fluxes using the ``ModelScenario`` convention.

    Args:
        flux_arrays: Flux arrays keyed by source label.

    Returns:
        A single flux array with non-time grid dimensions aligned and multiple
        non-sectoral sources summed or time-stacked as ``ModelScenario`` did.
    """
    flux_datasets = [
        arr.rename("flux").to_dataset() if arr.name != "flux" else arr.to_dataset()
        for arr in flux_arrays.values()
    ]

    if len(flux_datasets) == 1:
        return flux_datasets[0]["flux"]

    dims = [dim for dim in flux_datasets[0].dims if dim != "time"]
    flux_datasets = match_dataset_dims(flux_datasets, dims=dims)
    if "time" in flux_datasets[0].dims:
        flux_stacked = stack_datasets(flux_datasets, dim="time", method="ffill")
    else:
        flux_stacked = flux_datasets[0]
        for flux_dataset in flux_datasets[1:]:
            flux_stacked = flux_stacked + flux_dataset

    return flux_stacked["flux"]


def _require_matching_source_time_indexes(
    flux_arrays: Mapping[str, xr.DataArray],
) -> None:
    """Reject source fluxes whose native time axes cannot be stacked safely.

    Sources may all be timeless or may all use the same exact time index.
    Mixed timed/timeless inputs and unequal indexes require an explicit
    temporal resampling policy and are therefore rejected here.

    Args:
        flux_arrays: Source-labelled flux arrays.

    Raises:
        ValueError: If the mapping is empty, source labels are invalid, or
            time dimensions/indexes are incompatible.
    """
    if not flux_arrays:
        raise ValueError("Cannot stack an empty source-flux mapping.")

    source_labels = tuple(flux_arrays)
    if not all(isinstance(source, str) for source in source_labels):
        raise ValueError("Source-flux mapping labels must all be strings.")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("Source-flux mapping labels must be unique.")

    timed_sources = {source: "time" in flux.dims for source, flux in flux_arrays.items()}
    if len(set(timed_sources.values())) > 1:
        raise ValueError(
            "Cannot stack timed and timeless source fluxes; explicitly align or "
            "resample every source to a common time index first."
        )
    if not any(timed_sources.values()):
        return

    reference_source = source_labels[0]
    for source in source_labels:
        flux = flux_arrays[source]
        if "time" not in flux.coords or flux["time"].dims != ("time",):
            raise ValueError(
                "Cannot stack source fluxes without an explicit one-dimensional time coordinate; "
                f"source {source!r} has only a positional time dimension."
            )
        if not flux.get_index("time").is_unique:
            raise ValueError(
                "Cannot stack source fluxes with duplicate time coordinates; "
                f"source {source!r} must be made unique first."
            )

    reference_index = flux_arrays[reference_source].get_index("time")
    unequal_sources = [
        source
        for source in source_labels[1:]
        if not flux_arrays[source].get_index("time").equals(reference_index)
    ]
    if unequal_sources:
        raise ValueError(
            "Cannot stack source fluxes with unequal time indexes; explicitly "
            "align or resample every source to a common time index first. "
            f"Reference source {reference_source!r}; unequal sources {unequal_sources!r}."
        )


def _stack_flux_sources_with_alignment(
    flux_arrays: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """Stack source fluxes after exact temporal and dimensional validation.

    Args:
        flux_arrays: Source-labeled arrays in the output source order. Arrays
            must have identical dimension sets. Timed arrays must all carry the
            same explicit, unique ``time`` index. Non-time grid coordinates are
            aligned to the first source without interpolation.

    Returns:
        Flux with a leading ``source`` dimension in mapping insertion order and
        all remaining dimensions in the first source's order.

    Raises:
        ValueError: If the mapping is empty, labels are invalid, dimension sets
            differ, time coordinates are missing or duplicated, or indexes
            cannot be aligned exactly.
    """
    _require_matching_source_time_indexes(flux_arrays)
    first_key = next(iter(flux_arrays))
    reference = flux_arrays[first_key]
    reference_dims = set(reference.dims)
    dims_to_align = [dim for dim in reference.dims if dim != "time"]

    aligned_flux = {}
    for key, arr in flux_arrays.items():
        if set(arr.dims) != reference_dims:
            raise ValueError(
                "Cannot stack source fluxes with different dimensions; "
                f"reference source {first_key!r} has {reference.dims!r}, "
                f"source {key!r} has {arr.dims!r}."
            )
        aligned_flux[key] = force_align(
            arr.transpose(*reference.dims),
            reference=reference,
            dims=dims_to_align,
        )

    return xr.concat(
        [arr.expand_dims({"source": [key]}) for key, arr in aligned_flux.items()],
        dim="source",
        join="exact",
    )
