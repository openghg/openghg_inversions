"""Retrieve or reload RHIME observations, transport, flux and boundary data.

Acquisition owns store/cache access and the complete site-aligned selection
record. It returns a borrowed ``RhimeMergedData`` handoff for subsequent
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
from openghg_inversions.inversion_data import _site_options
from openghg_inversions.inversion_data.get_data import retrieve_inversion_data
from openghg_inversions.inversion_data.serialise import OutputFormat, _save_merged_data, load_merged_data


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
    site_options: _site_options.SiteOptions

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

    @classmethod
    def load(
        cls,
        merged_data_dir: str | Path,
        *,
        site_options: _site_options.SiteOptions,
        species: str | None = None,
        start_date: str | None = None,
        output_name: str | None = None,
        merged_data_name: str | None = None,
        output_format: OutputFormat | None = None,
        split_by_sectors: bool = False,
        flux_non_finite_check: FluxNonFiniteCheck = "lazy",
    ) -> RhimeMergedData:
        """Read a current-format merged artifact with caller-supplied selectors.

        The codec does not store complete site options. ``site_options`` must
        therefore describe the requested sites; loading retains their requested
        order and drops absent sites. Explicit time-resolution selectors and
        sector layout must match the artifact. Naming and format follow
        :func:`load_merged_data`. Flux sanitation follows fresh acquisition.

        This filesystem boundary may materialize arrays through the current
        codec. Missing paths, invalid artifacts, incompatible selectors and
        empty retained selections raise; loading never retrieves fresh data.
        """
        if merged_data_dir is None:
            raise ValueError("Explicit merged-data reload requires `merged_data_dir`.")
        fp_all = load_merged_data(
            merged_data_dir,
            species,
            start_date,
            output_name,
            merged_data_name,
            output_format=output_format,
        )
        _validate_loaded_time_resolved_selector(fp_all, site_options)
        _validate_loaded_sector_layout(fp_all, split_by_sectors=split_by_sectors)
        site_options = _drop_sites_missing_from_loaded_data(fp_all=fp_all, site_options=site_options)
        fp_all = _select_fp_all_sites(fp_all, site_options.sites)
        fp_all[".split_by_sectors"] = split_by_sectors
        _sanitize_merged_flux(fp_all, flux_non_finite_check)
        return cls(fp_all=fp_all, site_options=site_options)

    @classmethod
    def from_options(
        cls,
        *,
        species: str,
        site_options: _site_options.SiteOptions,
        domain: str,
        start_date: str,
        end_date: str,
        output_name: str,
        flux_sources: list[str] | None,
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
        save_merged_data: bool = False,
        merged_data_dir: str | None = None,
        merged_data_name: str | None = None,
        flux_non_finite_check: FluxNonFiniteCheck = "lazy",
    ) -> RhimeMergedData:
        """Acquire fresh merged data using complete aligned selectors.

        Reads OpenGHG stores and retains all selector fields together when sites
        are unavailable. Returned datasets may be lazy and remain borrowed by
        later preparation. ``flux_sources`` names OpenGHG sources; ``averaging_error``
        controls inclusion of observation variability. Optional saving uses the
        current merged-data codec and is disabled by default. Retrieval never
        attempts cache loading; use :meth:`load` for an explicit artifact request.
        Store, selector, merge and optional serialization errors propagate.
        """
        # Requested options remain authoritative; legacy metadata is redundant.
        fp_all, retained_sites, *_ = retrieve_inversion_data(
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
            time_resolved=list(site_options.time_resolved),
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
        _sanitize_merged_flux(fp_all, flux_non_finite_check)
        return cls(fp_all=fp_all, site_options=site_options)

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
            self.fp_all,
            merged_data_dir,
            species=species,
            start_date=start_date,
            output_name=output_name,
            merged_data_name=merged_data_name,
            output_format=output_format,
        )


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    site_options: _site_options.SiteOptions,
) -> _site_options.SiteOptions:
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
    site_options: _site_options.SiteOptions,
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
    """Reject a cached merged-data artifact with a different sector layout.

    The serialized ``.split_by_sectors`` marker records whether the cache
    contains source-resolved sensitivities.  Missing provenance is treated as
    the legacy combined layout, so it cannot be relabelled as sector-resolved.

    Args:
        fp_all: Loaded merged-data artifact and its serialized metadata.
        split_by_sectors: Whether the current run requires source-resolved
            sensitivities.

    Raises:
        ValueError: If the cached sector layout cannot satisfy this run.
    """
    stored_split_by_sectors = bool(fp_all.get(".split_by_sectors", False))
    if stored_split_by_sectors != split_by_sectors:
        raise ValueError(
            "Loaded merged data has an incompatible `split_by_sectors` layout: "
            f"artifact split_by_sectors={stored_split_by_sectors!r}, "
            f"requested split_by_sectors={split_by_sectors!r}."
        )


def _select_fp_all_sites(fp_all: dict, sites: Sequence[str]) -> dict:
    """Keep requested sites and shared entries."""
    site_names = set(sites)
    return {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}


def _retrieve_or_reload_merged_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: _site_options.SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    emissions_domain: str | None = None,
    met_model: _site_options.SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: _site_options.SiteStringOption = None,
    fp_species: str | None = None,
    time_resolved: _site_options.SiteBooleanOption = None,
    inlet: _site_options.SiteInletOption = None,
    instrument: _site_options.SiteStringOption = None,
    max_level: _site_options.SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: _site_options.SiteStringOption = None,
    platform: _site_options.SiteStringOption = None,
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
    artifact. Both paths retain one complete :class:`_site_options.SiteOptions` record. Explicit
    reload errors propagate without attempting fresh acquisition.

    Returns:
        Merged per-site data and aligned retained-site options.

    Raises:
        ValueError: If site options are invalid, no requested sites are loaded,
            or retrieval returns invalid retained-site names.
    """
    site_options = _site_options.SiteOptions.from_inputs(
        sites=sites,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
        time_resolved=time_resolved,
    )
    if reload_merged_data:
        if merged_data_dir is None:
            raise ValueError("Explicit merged-data reload requires `merged_data_dir`.")
        return RhimeMergedData.load(
            merged_data_dir,
            site_options=site_options,
            species=species,
            start_date=start_date,
            output_name=output_name,
            merged_data_name=merged_data_name,
            split_by_sectors=split_by_sectors,
            flux_non_finite_check=flux_non_finite_check,
        )
    return RhimeMergedData.from_options(
        species=species,
        site_options=site_options,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        flux_sources=flux_sources,
        split_by_sectors=split_by_sectors,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        emissions_domain=emissions_domain,
        fp_model=fp_model,
        fp_species=fp_species,
        calibration_scale=calibration_scale,
        use_bc=use_bc,
        bc_input=bc_input,
        averaging_error=averaging_error,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        flux_non_finite_check=flux_non_finite_check,
    )


def retrieve_or_reload_rhime_data(
    data_args: Mapping[str, Any],
    *,
    multisector: bool,
    merged_data: RhimeMergedData | None = None,
) -> RhimeMergedData:
    """Retrieve, reload, or accept externally supplied merged RHIME data.

    Passing ``merged_data`` is the explicit no-I/O path.  The object remains
    borrowed and is returned unchanged after a sector-layout compatibility check.
    Otherwise this stage may read OpenGHG stores or a local merged artifact,
    optionally write merged data, sanitize flux arrays, print progress, and
    emit warnings.  ``data_args`` is never mutated.

    Raises:
        ValueError: If ``data_args`` requests unsupported ``use_tracer=True``
            or the supplied merged data has an incompatible sector layout.
    """
    if data_args.get("use_tracer", False):
        raise ValueError("`use_tracer=True` is not supported; tracer inversions are not implemented.")
    if merged_data is not None:
        stored_multisector = bool(merged_data.fp_all.get(".split_by_sectors", False))
        if stored_multisector != multisector:
            raise ValueError(
                "External RHIME merged data has an incompatible sector layout: "
                f"artifact split_by_sectors={stored_multisector!r}, "
                f"runner multisector={multisector!r}."
            )
        return merged_data

    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(data_args["sites"]),
        split_by_sectors=multisector,
    ):
        return _retrieve_or_reload_merged_data(
            species=data_args["species"],
            sites=data_args["sites"],
            domain=data_args["domain"],
            averaging_period=data_args["averaging_period"],
            start_date=data_args["start_date"],
            end_date=data_args["end_date"],
            output_name=data_args["output_name"],
            flux_sources=data_args["flux_sources"],
            split_by_sectors=multisector,
            bc_store=data_args["bc_store"],
            obs_store=data_args["obs_store"],
            footprint_store=data_args["footprint_store"],
            emissions_store=data_args["emissions_store"],
            emissions_domain=data_args["emissions_domain"],
            met_model=data_args["met_model"],
            fp_model=data_args["fp_model"],
            fp_height=data_args["fp_height"],
            fp_species=data_args["fp_species"],
            time_resolved=data_args["time_resolved"],
            inlet=data_args["inlet"],
            instrument=data_args["instrument"],
            max_level=data_args["max_level"],
            calibration_scale=data_args["calibration_scale"],
            obs_data_level=data_args["obs_data_level"],
            platform=data_args["platform"],
            use_bc=data_args["use_bc"],
            bc_input=data_args["bc_input"],
            averaging_error=data_args["averaging_error"],
            reload_merged_data=data_args["reload_merged_data"],
            save_merged_data=data_args["save_merged_data"],
            merged_data_dir=data_args["merged_data_dir"],
            merged_data_name=data_args["merged_data_name"],
            flux_non_finite_check=data_args["flux_non_finite_check"],
        )


def _sanitize_merged_flux(fp_all: dict, flux_non_finite_check: FluxNonFiniteCheck) -> None:
    """Apply the established flux policy at acquisition and cache boundaries."""
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
