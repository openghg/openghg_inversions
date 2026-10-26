"""Named scientific preparation stages for RHIME recipes.

The functions in this module form the backend-neutral preparation spine used
by :func:`openghg_inversions.rhime.run_standard` and copied project runners:

``observation errors -> filter -> basis -> sensitivities -> labelled assembly``.

Acquisition is owned by :mod:`openghg_inversions.inversion_data.acquisition`;
stages accept explicit resolved choices.

Acquired merged data stays acquired: these stages take its per-site datasets
(and, when sites can be dropped, its ``SiteOptions``) and return new mappings,
never a ``MergedData`` carrying derived values. Supplied datasets are borrowed
and never mutated. Filtering may
compute when a selected filter cannot operate lazily.  Basis construction may
read or fit a basis; runners own optional exports. Sensitivity construction may execute
the basis and boundary-condition algorithms.  Labelled assembly validates the
durable :class:`~openghg_inversions.inversion_data.PreparedInputs`
artifact and jointly materializes core payloads before dropping unusable
observations. This assembly boundary preserves sparse payloads and leaves
unselected extensions lazy. Backend-specific selection and densification
remain in :mod:`openghg_inversions.rhime.materialization`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import BASIS_ARTIFACT_PATH_ATTR, BasisFunctions
from openghg_inversions.boundary_sensitivity import scale_satellite_boundary_sensitivity_to_column_signal
from openghg_inversions.filters import filtering
from openghg_inversions.inversion_data import MergedData, PreparedInputs, SiteOptions
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.inversion_data.prepared_inputs import _make_site_metadata
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import MinErrorConfig

__all__ = [
    "assemble_inputs",
    "prepare_observation_errors",
    "build_sensitivities",
    "filter_observations",
    "resolve_basis_inputs",
    "resolve_bc_basis",
    "acquire_merged_data",
    "export_basis",
]


def filter_observations(
    site_data: Mapping[str, xr.Dataset],
    site_options: SiteOptions,
    *,
    filters: Any = None,
) -> tuple[dict[str, xr.Dataset], SiteOptions]:
    """Filter borrowed observations and drop empty sites with aligned selectors.

    The stage may compute site data if a filter cannot operate lazily. It never
    constructs basis functions or model inputs.

    Args:
        site_data: Borrowed per-site observation and footprint datasets keyed
            by ``site_options.sites``.
        site_options: Selectors aligned to ``site_data``.
        filters: Filter name, list of names, or mapping from site names to
            filters accepted by :func:`openghg_inversions.filters.filtering`.
            ``None`` applies no filters. The caller's mapping is not modified.

    Returns:
        Retained per-site datasets in selector order, and the selectors aligned
        to them.

    Raises:
        ValueError: If no sites remain after filtering.
        KeyError: If a requested filter is unknown.
    """
    if isinstance(filters, dict):
        # The filter implementation normalizes this working mapping.
        filters = dict(filters)

    with timed(
        "rhime.prepare_inputs.obs_filtering",
        sites=len(site_data), filters=filters is not None,
    ):
        # Selectors, not mapping order, fix the retained site order.
        ordered = {site: site_data[site] for site in site_options.sites}
        if filters is None and all(dataset.sizes.get("time", 0) > 0 for dataset in ordered.values()):
            return ordered, site_options
        retained = _apply_filters_and_drop_empty_sites(
            fp_data={site: dataset.copy() for site, dataset in ordered.items()},
            sites=site_options.sites,
            filters=filters,
        )
        return retained, site_options.retain_sites(tuple(retained), context="Observation filtering")


def build_sensitivities(
    site_data: Mapping[str, xr.Dataset],
    basis_functions: BasisFunctions,
    *,
    flux_sources: Sequence[str],
    use_bc: bool = True,
    bc_basis: xr.Dataset | None = None,
    multisector: bool,
) -> dict[str, xr.Dataset]:
    """Construct labelled flux and optional boundary-condition sensitivities.

    The stage creates per-site dataset copies, computes the basis projection,
    using preloaded boundary-condition basis data. ``site_data`` and
    ``basis_functions`` remain borrowed.

    Args:
        site_data: Filtered per-site observations and footprints.
        basis_functions: Retained basis and fluxes to project onto observations.
        flux_sources: Source names used to align the flux sensitivities.
        use_bc: Whether to construct boundary-condition sensitivities.
        bc_basis: Loaded boundary basis; None uses acquired NESW directional
            contributions without a basis file. Ignored when BCs are disabled.
        multisector: Whether to retain separate source sensitivities instead
            of the standard single-source layout.

    Returns:
        Per-site datasets containing flux and, when requested,
        boundary-condition sensitivities for labelled assembly.

    Raises:
        ValueError: If the basis or source layout is incompatible with the
            selected sensitivity layout.
    """
    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(site_data)):
        fp_data = {site: dataset.copy() for site, dataset in site_data.items()}
        fp_x_flux_name = "fp_x_flux_sectoral" if multisector else "fp_x_flux"

        for site in site_data:
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
            if multisector:
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
            with timed("rhime.prepare_inputs.bc_sensitivity", sites=len(site_data)):
                fp_data = bc_sensitivity(
                    fp_data,
                    bc_basis=bc_basis,
                )

        return fp_data


def assemble_inputs(
    site_data: Mapping[str, xr.Dataset],
    basis_functions: BasisFunctions,
    *,
    site_options: SiteOptions,
    domain: str,
    start_date: str,
    bc_freq: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: Mapping[str, Any] | None = None,
    use_bc: bool = True,
) -> PreparedInputs:
    """Construct and validate durable, backend-neutral RHIME model inputs.

    The stage attaches domain metadata to shallow per-site copies, assembles
    observation-aligned arrays, applies the satellite boundary-condition
    scaling, and retains basis and site metadata. It also preserves the existing
    construction of the minimum-error floor and boundary-condition temporal
    parameterization. Those are inverse-model settings, not properties of the
    acquired data; moving them to their model components is a later semantic
    change. Core payloads are jointly materialized before invalid observations
    are dropped; backend selection and densification happen separately.

    Args:
        site_data: Per-site datasets from sensitivity construction. Domain
            metadata is attached to shallow copies, leaving these datasets
            unchanged.
        basis_functions: Basis and prior fluxes to retain with the inputs.
        site_options: Selectors for the retained sites, as returned by
            :func:`filter_observations`.
        domain: Domain label attached to the assembled site data.
        start_date: Requested start date used to anchor temporal parameters.
        bc_freq: Frequency of boundary-condition scaling parameters; ``None``
            keeps one period across the run.
        min_error: Minimum observation-error floor as a numeric value or a
            mapping covering every retained site, or the named ``"residual"``
            or ``"percentile"`` calculation.
        min_error_options: Resolved mapping with boolean ``by_site``, as
            supplied by ``RhimeConfig``. ``None`` uses a shared calculation.
            Explicit keyword calls must supply already-normalized options.
        use_bc: Whether missing boundary-condition inputs should be checked.

    Returns:
        Validated, backend-neutral inputs with observation-aligned arrays,
        retained basis functions and site metadata. Core assembly payloads are
        eager, possibly sparse; unselected extensions may remain lazy.

    Raises:
        ValueError: If assembled inputs fail their alignment or scientific
            input contracts.
        KeyError: If the resolved ``by_site`` option is absent.
    """
    owned_site_data = {site: dataset.copy(deep=False) for site, dataset in site_data.items()}
    for site in site_options.sites:
        owned_site_data[site].attrs["Domain"] = domain
    # These inverse-model settings are materialized into labelled arrays here
    # to preserve the existing numerical contract while preparation is split.
    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(site_options.sites)):
        inv_inputs = make_inv_inputs(
            fp_data=owned_site_data,
            sites=list(site_options.sites),
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            min_error_per_site=False if min_error_options is None else min_error_options["by_site"],
        )
    inv_inputs = scale_satellite_boundary_sensitivity_to_column_signal(
        inv_inputs,
        sites=site_options.sites,
        platform=site_options.platform,
        observation_max_level=site_options.max_level,
        footprint_max_level=tuple(
            owned_site_data[site].attrs.get("footprint_max_level") for site in site_options.sites
        ),
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    basis_source = basis_functions.basis_artifact_source or "generated"
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(site_options.sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )
    site_metadata = _make_site_metadata(
        sites=site_options.sites,
        averaging_period=site_options.averaging_period,
    )
    def footprint_attr(site: str, name: str) -> str:
        value = owned_site_data[site].attrs.get(f"footprint_{name}")
        return value if isinstance(value, str) and value.strip().lower() not in {"", "none", "not_set"} else ""

    for name in ("transport_model", "transport_model_version", "met_model"):
        site_metadata[name] = (
            "site",
            [footprint_attr(site, name) for site in site_options.sites],
        )
    return PreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=site_metadata,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(
            f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values", stacklevel=3
        )
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(
            f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values",
            stacklevel=3,
        )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    sites: Sequence[str],
    filters: Any,
) -> dict:
    """Apply observation filters and remove empty site datasets."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        if not fp_data:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data


def _validate_multisector_sensitivity_sources(
    sensitivity: xr.DataArray,
    *,
    site: str,
    flux_sources: Sequence[str],
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
        return sensitivity.sel(source=list(flux_sources))
    return sensitivity




def resolve_bc_basis(data_config: Any) -> xr.Dataset | None:
    """Load enabled exact BC resources, or select the built-in directional basis.

    Configuration owns algorithm validation. Exact files take precedence over
    that inactive algorithm choice. With no files, resolved ``"NESW"`` uses
    acquired directional contributions through the scientific ``None`` input.
    Disabled BCs perform no resource I/O. Selected files are loaded before
    acquisition using the existing time combination and spatial alignment.
    The configuration and borrowed scientific inputs remain unchanged.

    Args:
        data_config: Resolved acquisition/preparation configuration.

    Returns:
        Loaded labelled BC basis, or None for disabled BCs or built-in NESW.

    Raises:
        ValueError: If the exact file selection is empty.
        OSError: If a selected file is missing or unreadable.
    """
    from openghg_inversions.basis.support import load_bc_basis

    if not data_config.use_bc or data_config.bc_basis_paths is None:
        return None
    return load_bc_basis(data_config.bc_basis_paths)


def resolve_basis_inputs(data_config: Any) -> dict[str, Any]:
    """Resolve exact scientific inputs before acquisition.

    Configuration resolution itself stays data-free. This application boundary
    reads selected maps and partitions and checks retained paths from canonical
    configuration. Numerical preparation receives only the resulting exact
    artifact path and borrowed labelled arrays. Reporting resources are
    validated separately by the output owner.

    Args:
        data_config: Resolved :class:`~openghg_inversions.rhime.params.DataConfig`.
            Explicit retained artifacts and partition files bypass algorithm
            support defaults; explicitly selected support files are still read.
            Plural partition paths are sorted before loading; the first file
            supplies the reference spatial coordinates.

    Returns:
        Keyword arguments for :func:`openghg_inversions.basis.prepare_basis`:
        an optional exact ``basis_path`` or eager ``basis_partition``, plus
        eager support maps or None. Retained artifacts are checked for existence
        here and loaded later by basis preparation. The request is not mutated.

    Raises:
        ValueError: If mutually exclusive saved input forms are supplied.
        FileNotFoundError: If a selected artifact or resource is missing.
        KeyError: If a map or partition lacks its required variable.

    """
    from openghg_inversions.basis.support import load_basis_partition, resolve_basis_support_maps

    if sum(value is not None for value in (
        data_config.basis_path, data_config.basis_partition_path, data_config.basis_partition_paths,
    )) > 1:
        raise ValueError("Supply either basis_path, basis_partition_path or basis_partition_paths.")
    selected: dict[str, Any] = {}
    landsea_path = data_config.landsea_path
    outer_path = data_config.outer_regions_path
    if data_config.basis_path is not None:
        selected["basis_path"] = data_config.basis_path
    if data_config.basis_partition_path is not None:
        selected["basis_partition"] = load_basis_partition(data_config.basis_partition_path)
    if data_config.basis_partition_paths is not None:
        if not data_config.basis_partition_paths:
            raise ValueError("basis_partition_paths must select at least one file.")
        from openghg_inversions.utils import read_netcdfs

        with read_netcdfs(sorted(data_config.basis_partition_paths)) as dataset:
            selected["basis_partition"] = dataset["basis"].load()
    if selected.get("basis_path") is not None and not Path(selected["basis_path"]).exists():
        raise FileNotFoundError(selected["basis_path"])
    loading = "basis_path" in selected or "basis_partition" in selected
    selected.update(resolve_basis_support_maps(
        domain=data_config.domain, basis_algorithm=None if loading else data_config.basis_algorithm,
        fix_outer_regions=data_config.fix_basis_outer_regions and not loading,
        landsea_path=landsea_path, outer_regions_path=outer_path,
    ))
    return selected


def acquire_merged_data(data_config: Any) -> MergedData:
    """Acquire the configured observations, transport and flux.

    Args:
        data_config: Resolved :class:`~openghg_inversions.rhime.params.DataConfig`.

    Returns:
        Acquired merged data for scientific preparation. To reuse supplied data,
        the runner calls its ``validate_for_preparation`` method instead.
    """
    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(data_config.site_options.sites), split_by_sectors=data_config.split_by_sectors,
    ):
        return MergedData.from_options(
            **data_config.select(
                "site_options", "species", "domain", "start_date", "end_date", "flux_sources",
                "split_by_sectors", "bc_store", "obs_store", "footprint_store", "flux_store",
                "flux_domain", "fp_model", "fp_species", "calibration_scale", "use_bc", "bc_input",
                "flux_non_finite_check",
            ),
        )


def export_basis(basis_functions: BasisFunctions, path: str | Path | None) -> BasisFunctions:
    """Save a basis to the runner-selected exact path and record that path.

    Args:
        basis_functions: Basis returned by scientific preparation.
        path: Exact export destination, or ``None`` to skip the export.

    Returns:
        ``basis_functions`` unchanged without a path; otherwise a copy whose
        metadata names the saved artifact.
    """
    if path is None:
        return basis_functions
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    basis_functions.save(path)
    return basis_functions.with_metadata({BASIS_ARTIFACT_PATH_ATTR: str(path)})
