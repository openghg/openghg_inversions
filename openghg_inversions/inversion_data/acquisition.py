"""Retrieve or reload RHIME observations, transport, flux and boundary data.

Acquisition owns store/cache access and consumes the complete site-aligned
selection record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
scientific filtering, basis construction and sensitivity preparation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from openghg_inversions._timing import timed
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck, sanitize_flux_nonfinite
from openghg_inversions.inversion_data._site_options import (
    SiteBooleanOption as SiteBooleanOption,
    SiteInletOption as SiteInletOption,
    SiteIntegerOption as SiteIntegerOption,
    SiteStringOption as SiteStringOption,
    SiteOptions as SiteOptions,
)
from openghg_inversions.inversion_data.get_data import (
    _retrieve_inversion_data_from_options,
)
from openghg_inversions.inversion_data.serialise import OutputFormat, _save_merged_data, load_merged_data


@dataclass
class RhimeMergedData:
    """Merged RHIME data and complete site-aligned metadata between stages.

    Args:
        fp_all: Merged per-site datasets plus shared flux, boundary-condition,
            and calibration entries.
        site_options: Complete resolved selectors for the sites present in
            ``fp_all``, in their retained order. These values are authoritative
            when the handoff is supplied to :func:`load_rhime_data`.

    Notes:
        This is a supported orchestration handoff. Its datasets remain
        backend-neutral and may be Dask-backed; later stages must treat them as
        borrowed. Construction does not copy or compute data, normalize
        selectors, or perform acquisition. Use :meth:`SiteOptions.from_inputs`
        for external shorthand and :meth:`save` for explicit serialization.
    """

    fp_all: dict
    site_options: SiteOptions

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

    def save(
        self,
        merged_data_dir: str | Path,
        *,
        species: str | None = None,
        start_date: str | None = None,
        output_name: str | None = None,
        merged_data_name: str | None = None,
        output_format: OutputFormat = "zarr.zip",
    ) -> None:
        """Write the merged scientific datasets in the established cache format.

        This is an explicit filesystem and numerical execution boundary:
        serialization computes lazy payloads and may rechunk them for Zarr.
        It writes ``fp_all``; it does not introduce a serialized site-options
        schema. Cache loading aligns selectors from the current request.

        Args:
            merged_data_dir: Destination directory, created if absent.
            species: Gas name used in the default artifact filename.
            start_date: Requested start date used in the default filename.
            output_name: Run name used in the default filename.
            merged_data_name: Explicit artifact name. When omitted, ``species``,
                ``start_date`` and ``output_name`` must all be supplied.
                An ``.nc``, ``.zarr`` or ``.zarr.zip`` suffix selects its format.
            output_format: Format used when the name has no recognized suffix:
                ``"netcdf"``, ``"zarr"`` or ``"zarr.zip"``.

        Raises:
            ValueError: If naming inputs are missing, the format is unsupported,
                or the obsolete pickle suffix is used.
        """
        _save_merged_data(
            self.fp_all, merged_data_dir, species=species, start_date=start_date,
            output_name=output_name, merged_data_name=merged_data_name,
            output_format=output_format,
        )


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    site_options: SiteOptions,
) -> SiteOptions:
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


def _validate_loaded_time_resolved_selector(
    fp_all: Mapping[str, Any],
    site_options: SiteOptions,
) -> None:
    """Reject cached sites whose explicit time-resolution selector differs."""
    mismatched_sites: list[str] = []
    for site, selector in zip(site_options.sites, site_options.time_resolved, strict=True):
        if selector is None or site not in fp_all:
            continue
        site_data = fp_all[site]
        cached_selector = (
            site_data.attrs.get("openghg_inversions_time_resolved")
            if isinstance(site_data, xr.Dataset)
            else None
        )
        if cached_selector != str(selector).lower():
            mismatched_sites.append(site)
    if mismatched_sites:
        raise ValueError(
            "Loaded merged data does not match the requested `time_resolved` selector for "
            f"site(s): {mismatched_sites!r}."
        )


def _validate_loaded_sector_layout(fp_all: Mapping[str, Any], *, split_by_sectors: bool) -> None:
    """Reject supplied or cached merged data with a different sector layout.

    The serialized ``.split_by_sectors`` marker records whether the cache
    contains source-resolved sensitivities.  Missing provenance is treated as
    the legacy combined layout, so it cannot be relabelled as sector-resolved.

    Args:
        fp_all: Supplied or loaded merged data and its layout metadata.
        split_by_sectors: Whether the current run requires source-resolved
            sensitivities.

    Raises:
        ValueError: If the sector layout cannot satisfy this run.
    """
    stored_split_by_sectors = bool(fp_all.get(".split_by_sectors", False))
    if stored_split_by_sectors != split_by_sectors:
        raise ValueError(
            "Merged data has an incompatible `split_by_sectors` layout: "
            f"artifact split_by_sectors={stored_split_by_sectors!r}, "
            f"requested split_by_sectors={split_by_sectors!r}."
        )


def _select_fp_all_sites(fp_all: dict, sites: Sequence[str]) -> dict:
    """Keep requested sites and shared entries."""
    site_names = set(sites)
    return {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}


def load_rhime_data(
    *,
    site_options: SiteOptions,
    merged_data: RhimeMergedData | None = None,
    species: str,
    domain: str,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: Sequence[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    emissions_domain: str | None = None,
    fp_model: str | None = None,
    fp_species: str | None = None,
    calibration_scale: str | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimeMergedData:
    """Retrieve, reload, or accept borrowed merged RHIME data.

    Selectors are complete aligned values; this boundary does not expand
    shorthand. Supplied data remains authoritative and returns unchanged
    after its sector layout is checked, bypassing stores, caches and writes.
    Otherwise retrieval may access OpenGHG stores, reload a merged artifact,
    optionally save data and sanitize flux values. Cache selector/layout checks
    and retained-site alignment belong here.

    Args:
        site_options: Complete resolved requested selectors. Build external
            shorthand with :meth:`SiteOptions.from_inputs` before calling.
        merged_data: Supplied handoff to return unchanged after layout checking.
            Its retained options take precedence over ``site_options`` and it
            bypasses acquisition, reload, normalization and automatic saving.
        species: Primary gas used in store queries and default cache naming.
        domain: Footprint and boundary-condition domain.
        start_date: Inclusive requested start date for retrieval and cache naming.
        end_date: Exclusive requested end date for retrieval.
        output_name: Run name used in the default merged-cache filename.
        flux_sources: OpenGHG flux ``source`` values. Required for fresh
            acquisition; supplied and cached data already contain their fluxes.
        split_by_sectors: Whether fresh retrieval keeps source-resolved
            sensitivities. Supplied/cached layout must match this choice;
            missing layout metadata denotes the combined layout.
        bc_store: Boundary-condition object-store name.
        obs_store: Observation object-store name.
        footprint_store: Footprint object-store name.
        emissions_store: Flux object-store name.
        emissions_domain: Optional flux-domain selector. When it differs from
            ``domain``, fresh retrieval interpolates flux onto the footprint grid.
        fp_model: Optional footprint transport-model selector.
        fp_species: Optional species selector for footprint retrieval.
        calibration_scale: Optional target calibration scale for observations.
        use_bc: Whether fresh acquisition includes boundary-condition data.
        bc_input: Optional boundary-condition source selector.
        averaging_error: Whether observation error includes variability within
            each averaging period as well as repeatability.
        reload_merged_data: Attempt cache loading when ``merged_data_dir`` is
            supplied. A recoverable load ``ValueError`` or missing directory
            falls back to fresh acquisition; incompatible cached layout or
            explicit time-resolution selectors are rejected without fallback.
        save_merged_data: Save newly acquired merged data when a directory is
            supplied. Defaults to false; successful reload and supplied-data
            paths never save. A missing destination prints a message.
        merged_data_dir: Cache directory used for optional loading and saving.
        merged_data_name: Explicit cache name. Otherwise the name is formed from
            ``species``, ``start_date`` and ``output_name``. For saving, an
            ``.nc``, ``.zarr`` or ``.zarr.zip`` suffix selects the format.
        flux_non_finite_check: ``"lazy"`` replaces non-finite flux with zero
            lazily; ``"count"`` computes counts and warns when values are found.

    Returns:
        Borrowed merged datasets and complete retained selectors. Retrieval
        order determines fresh-data ordering; reload selects sites in requested
        order. A supplied compatible handoff is returned by identity.

    Raises:
        SearchError: If fresh retrieval finds no usable requested site data.
        ValueError: If supplied data has an incompatible sector layout, cached
            selectors/layout are incompatible, retained names are invalid,
            fresh retrieval inputs cannot be merged, or saving inputs are invalid.
    """
    if merged_data is not None:
        _validate_loaded_sector_layout(merged_data.fp_all, split_by_sectors=split_by_sectors)
        return merged_data

    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(site_options.sites),
        split_by_sectors=split_by_sectors,
    ):
        fp_all: dict | None = None
        if reload_merged_data and merged_data_dir is not None:
            try:
                fp_all = load_merged_data(merged_data_dir, species, start_date, output_name, merged_data_name)
            except ValueError as exc:
                print(f"{exc}, re-running data merge.")
        elif reload_merged_data:
            print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

        if fp_all is not None:
            _validate_loaded_time_resolved_selector(fp_all, site_options)
            _validate_loaded_sector_layout(fp_all, split_by_sectors=split_by_sectors)
            print("Successfully read in merged data.\n")
            fp_all[".split_by_sectors"] = split_by_sectors
            site_options = _drop_sites_missing_from_loaded_data(fp_all=fp_all, site_options=site_options)
        else:
            # Requested options remain authoritative; legacy metadata is redundant.
            fp_all, retained_sites, *_ = _retrieve_inversion_data_from_options(
                site_options=site_options,
                species=species,
                domain=domain,
                start_date=start_date,
                end_date=end_date,
                fp_model=fp_model,
                fp_species=fp_species,
                emissions_name=flux_sources,
                calibration_scale=calibration_scale,
                use_bc=use_bc,
                bc_input=bc_input,
                bc_store=bc_store,
                obs_store=obs_store,
                footprint_store=footprint_store,
                emissions_store=emissions_store,
                emissions_domain=emissions_domain,
                split_by_sectors=split_by_sectors,
                averagingerror=averaging_error,
                save_merged_data=save_merged_data,
                merged_data_name=merged_data_name,
                merged_data_dir=merged_data_dir,
                output_name=output_name,
                flux_non_finite_check=flux_non_finite_check,
            )
            site_options = site_options.retain_sites(retained_sites, context="Data gathering")

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
